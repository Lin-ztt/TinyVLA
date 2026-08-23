#!/usr/bin/env python3
"""Record one front-view LIBERO rollout for base, SFT, DSRL, or EXPO."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.scripts.lerobot_eval import rollout as lerobot_rollout
from lerobot.utils.random_utils import set_seed

from tinyvla.dsrl import SAC
from tinyvla.dsrl.rollout import RolloutConfig as DSRLRolloutConfig
from tinyvla.dsrl.rollout import rollout_episode as dsrl_rollout
from tinyvla.expo import EXPOConfig, EXPOLearner
from tinyvla.expo.rollout import EXPORolloutConfig
from tinyvla.expo.rollout import rollout_episode as expo_rollout
from tinyvla.libero import make_libero_config, make_smolvla_stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("base", "sft", "dsrl", "expo"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--learner-checkpoint", type=Path)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=280)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def capture_frontview(env: Any, width: int, height: int) -> np.ndarray:
    low_level = env.envs[0]
    if low_level._env is None:
        raise RuntimeError("LIBERO renderer is not initialized")
    frame = low_level._env.env.sim.render(camera_name="frontview", width=width, height=height)
    return np.asarray(frame)[::-1, ::-1].copy()


def load_dsrl_actor(path: Path, device: torch.device) -> SAC:
    state = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    sac_state = state.get("sac", state)
    learner, _ = SAC.from_checkpoint_state(sac_state, device=device)
    return learner.eval()


def make_expo_config(raw: dict[str, Any]) -> EXPOConfig:
    values = raw["expo"]
    return EXPOConfig(
        actor_lr=float(values["actor_lr"]),
        critic_lr=float(values["critic_lr"]),
        temperature_lr=float(values["temperature_lr"]),
        tau=float(values["tau"]),
        edit_scale=float(values["edit_scale"]),
        initial_temperature=float(values["initial_temperature"]),
        target_entropy=float(values["target_entropy"]),
        num_qs=int(values["num_qs"]),
        num_min_qs=int(values["num_min_qs"]),
        crop_padding=int(values["crop_padding"]),
        rotation_degrees=float(values["rotation_degrees"]),
        color_jitter=float(values["color_jitter"]),
    )


def load_expo_learner(path: Path, config: dict[str, Any], device: torch.device) -> EXPOLearner:
    state = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    learner, _ = EXPOLearner.from_checkpoint_state(state["expo"], device=device)
    return learner.eval()


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.fps, args.max_episode_steps) <= 0:
        raise ValueError("width, height, fps, and max_episode_steps must be positive")
    if args.method in ("dsrl", "expo") and args.config is None:
        raise ValueError("--config is required for DSRL and EXPO")

    raw = yaml.safe_load(args.config.read_text()) if args.config else {}
    environment = raw.get("environment", {})
    suite = str(environment.get("suite", args.suite))
    task_id = int(environment.get("task_id", args.task_id))
    max_episode_steps = int(environment.get("max_episode_steps", args.max_episode_steps))
    device = torch.device(args.device)
    set_seed(args.seed)
    env_config = make_libero_config(suite, [task_id], max_episode_steps)
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs[suite][task_id]
    env.set_attr("init_state_id", args.init_state_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(".json")
    writer = imageio.get_writer(
        args.output, fps=args.fps, codec="libx264", quality=8, macro_block_size=1
    )
    frame_count = 0

    def render_callback(current_env: Any) -> None:
        nonlocal frame_count
        frame = capture_frontview(current_env, args.width, args.height)
        writer.append_data(frame)
        frame_count += 1
        if frame_count == 1:
            Image.fromarray(frame).save(args.output.with_suffix(".png"))

    try:
        stack = make_smolvla_stack(args.checkpoint, env_config, device)
        if args.method in ("base", "sft"):
            result = lerobot_rollout(
                env=env,
                policy=stack.policy,
                env_preprocessor=stack.env_preprocessor,
                env_postprocessor=stack.env_postprocessor,
                preprocessor=stack.preprocessor,
                postprocessor=stack.postprocessor,
                seeds=[args.seed],
                render_callback=render_callback,
            )
            success = bool(result["success"].any().item())
        elif args.method == "dsrl":
            actor_path = args.actor_checkpoint or raw.get("actor_checkpoint")
            if actor_path is None:
                raise ValueError("--actor-checkpoint is required for DSRL")
            learner = load_dsrl_actor(Path(actor_path), device)
            image_keys = tuple(environment.get("sac_image_keys", ("observation.images.image",)))
            result = dsrl_rollout(
                env,
                stack.policy,
                stack.env_preprocessor,
                stack.preprocessor,
                stack.postprocessor,
                stack.env_postprocessor,
                None,
                DSRLRolloutConfig(
                    max_episode_steps=max_episode_steps,
                    seed=args.seed,
                    execute_horizon=int(environment.get("execute_horizon", 20)),
                    discount=float(environment.get("discount", 0.999)),
                    sac_image_keys=image_keys,
                ),
                noise_fn=lambda observation: learner.act(
                    torch.from_numpy(observation["pixels"]).unsqueeze(0),
                    torch.from_numpy(observation["state"]).unsqueeze(0),
                    deterministic=False,
                )[0].cpu().numpy().astype(np.float32),
                noise_source="actor_stochastic",
                render_callback=render_callback,
            )
            success = bool(result["success"])
        else:
            learner_path = args.learner_checkpoint or raw.get("learner_checkpoint")
            if learner_path is None:
                raise ValueError("--learner-checkpoint is required for EXPO")
            learner = load_expo_learner(Path(learner_path), raw, device)
            result = expo_rollout(
                env,
                stack.policy,
                learner,
                stack.env_preprocessor,
                stack.preprocessor,
                stack.postprocessor,
                stack.env_postprocessor,
                None,
                EXPORolloutConfig(
                    max_episode_steps=max_episode_steps,
                    discount=float(environment.get("discount", 0.99)),
                    seed=args.seed,
                    deterministic_actor=False,
                ),
                render_callback=render_callback,
            )
            success = bool(result["success"])
    finally:
        writer.close()
        close_envs(envs)

    report = {
        "status": "passed",
        "method": args.method,
        "suite": suite,
        "task_id": task_id,
        "init_state_id": args.init_state_id,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "success": success,
        "frames": frame_count,
        "fps": args.fps,
        "resolution": {"width": args.width, "height": args.height},
        "video": str(args.output.resolve()),
    }
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
