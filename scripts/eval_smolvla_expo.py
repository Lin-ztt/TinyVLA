#!/usr/bin/env python3
"""Evaluate an untrained or trained EXPO editor on one LIBERO task."""

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

import numpy as np
import torch
import yaml
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.utils.random_utils import set_seed

from tinyvla.expo import EXPOConfig, EXPOLearner
from tinyvla.expo.rollout import EXPORolloutConfig, rollout_episode
from tinyvla.libero import make_libero_config, make_smolvla_stack


CHECKPOINT_FORMAT_VERSION = 3
ENTROPY_COORDINATE = "residual_unscaled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--learner-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def make_expo_config(raw: dict[str, Any]) -> EXPOConfig:
    expo = raw["expo"]
    return EXPOConfig(
        actor_lr=float(expo["actor_lr"]),
        critic_lr=float(expo["critic_lr"]),
        temperature_lr=float(expo["temperature_lr"]),
        tau=float(expo["tau"]),
        edit_scale=float(expo["edit_scale"]),
        initial_temperature=float(expo["initial_temperature"]),
        target_entropy=float(expo["target_entropy"]),
        num_qs=int(expo["num_qs"]),
        num_min_qs=int(expo["num_min_qs"]),
        crop_padding=int(expo["crop_padding"]),
        rotation_degrees=float(expo["rotation_degrees"]),
        color_jitter=float(expo["color_jitter"]),
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = [item for episode in episodes for item in episode["transitions"]]
    successes = [episode for episode in episodes if episode["success"]]
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes) if episodes else None,
        "mean_completion_steps_success_only": (
            float(np.mean([episode["environment_steps"] for episode in successes]))
            if successes
            else None
        ),
        "mean_environment_steps": (
            float(np.mean([episode["environment_steps"] for episode in episodes]))
            if episodes
            else None
        ),
        "chunk_transitions": len(transitions),
        "edited_selection_rate": (
            float(np.mean([item["selected_edited"] for item in transitions]))
            if transitions
            else None
        ),
        "mean_residual_norm": (
            float(np.mean([item["residual_norm"] for item in transitions]))
            if transitions
            else None
        ),
        "mean_candidate_inference_seconds": (
            float(np.mean([item["inference_seconds"] for item in transitions]))
            if transitions
            else None
        ),
        "normalized_action_out_of_range_rate": (
            float(np.mean([item["normalized_action_out_of_range_rate"] for item in transitions]))
            if transitions
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    with args.config.open() as file:
        config = yaml.safe_load(file)
    environment = config["environment"]
    evaluation = config["evaluation"]
    checkpoint_path = Path(config["checkpoint"]).resolve()
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "evaluation.json"
    device = torch.device(config["device"])
    learner_checkpoint = args.learner_checkpoint.resolve() if args.learner_checkpoint else None

    suite = str(environment["suite"])
    task_id = int(environment["task_id"])
    max_episode_steps = int(environment["max_episode_steps"])
    env_config = make_libero_config(suite, [task_id], max_episode_steps)
    set_seed(int(config["learner_seed"]))
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs[suite][task_id]

    try:
        stack = make_smolvla_stack(checkpoint_path, env_config, device)
        policy = stack.policy
        preprocessor = stack.preprocessor
        postprocessor = stack.postprocessor
        env_preprocessor = stack.env_preprocessor
        env_postprocessor = stack.env_postprocessor
        if learner_checkpoint is None:
            learner = EXPOLearner(make_expo_config(config), device=device)
            mode = "untrained"
        else:
            state = torch.load(learner_checkpoint, map_location="cpu", weights_only=False)
            if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
                raise ValueError("Unsupported EXPO training checkpoint format")
            if state.get("entropy_coordinate") != ENTROPY_COORDINATE:
                raise ValueError("EXPO checkpoint uses an incompatible entropy coordinate")
            trained_environment = state["config"]["environment"]
            if trained_environment["suite"] != suite or int(trained_environment["task_id"]) != task_id:
                raise ValueError("EXPO checkpoint task does not match evaluation task")
            learner, _ = EXPOLearner.from_checkpoint_state(state["expo"], device=device)
            mode = "trained"
        learner.eval()

        episodes: list[dict[str, Any]] = []
        init_state_start = int(evaluation["init_state_start"])
        episode_seed_start = int(evaluation["seed"])
        for episode_index in range(int(evaluation["episodes"])):
            init_state_index = init_state_start + episode_index
            episode_seed = episode_seed_start + episode_index
            env.set_attr("init_state_id", init_state_index)
            set_seed(episode_seed)
            result = rollout_episode(
                env,
                policy,
                learner,
                env_preprocessor,
                preprocessor,
                postprocessor,
                env_postprocessor,
                None,
                EXPORolloutConfig(
                    max_episode_steps=max_episode_steps,
                    discount=float(environment["discount"]),
                    bootstrap_on_truncation=bool(environment["bootstrap_on_truncation"]),
                    seed=episode_seed,
                    deterministic_actor=False,
                ),
            )
            result.update(
                {
                    "episode_index": episode_index,
                    "init_state_index": init_state_index,
                    "seed": episode_seed,
                    "mode": mode,
                }
            )
            episodes.append(result)
            report = {
                "status": "running",
                "mode": mode,
                "learner_checkpoint": str(learner_checkpoint) if learner_checkpoint else None,
                "config": config,
                "episodes": episodes,
                "summary": summarize(episodes),
            }
            write_json(result_path, report)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "episode": episode_index + 1,
                        "success": result["success"],
                        "steps": result["environment_steps"],
                    }
                ),
                flush=True,
            )

        report["status"] = "passed"
        write_json(result_path, report)
        print(json.dumps(report["summary"], indent=2), flush=True)
    finally:
        close_envs(envs)


if __name__ == "__main__":
    main()
