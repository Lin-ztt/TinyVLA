#!/usr/bin/env python3
"""Create high-resolution LIBERO front-view showcase images and videos."""

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
from PIL import Image

from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.random_utils import set_seed

from tinyvla.libero import make_libero_config, make_smolvla_stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/runs/sft/train/smolvla_libero_40tasks_all/checkpoints/050000/pretrained_model"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/showcase/libero_spatial_frontview"))
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", default="0,1,2,3,4")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--init-state-start", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--max-episode-steps", type=int, default=280)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def capture_frontview(env: Any, width: int, height: int) -> np.ndarray:
    """Render the complete scene from LIBERO's fixed frontview camera."""
    low_level = env.envs[0]
    if low_level._env is None:
        raise RuntimeError("LIBERO renderer is not initialized")
    frame = low_level._env.env.sim.render(camera_name="frontview", width=width, height=height)
    # MuJoCo returns the offscreen image upside down for this wrapper.
    return np.asarray(frame)[::-1, ::-1].copy()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing SmolVLA checkpoint: {checkpoint}")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("width, height and fps must be positive")
    task_ids = [int(item.strip()) for item in args.task_ids.split(",") if item.strip()]
    if not task_ids:
        raise ValueError("--task-ids must contain at least one task id")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed_start)
    env_config = make_libero_config(args.suite, task_ids, args.max_episode_steps)
    envs = make_env(env_config, n_envs=1, use_async_envs=False)

    try:
        stack = make_smolvla_stack(checkpoint, env_config, "cuda")
        policy = stack.policy
        preprocessor = stack.preprocessor
        postprocessor = stack.postprocessor
        env_preprocessor = stack.env_preprocessor
        env_postprocessor = stack.env_postprocessor

        for task_id in task_ids:
            env = envs[args.suite][task_id]
            task_dir = args.output_dir / f"task_{task_id:02d}"
            task_dir.mkdir(parents=True, exist_ok=True)
            description = str(env.call("task_description")[0])
            found: dict[str, dict[str, Any]] = {}

            for attempt in range(args.max_attempts):
                seed = args.seed_start + attempt
                init_state_id = args.init_state_start + attempt
                env.set_attr("init_state_id", init_state_id)
                temporary_video = task_dir / f"attempt_{attempt:02d}.mp4"
                writer = imageio.get_writer(
                    temporary_video,
                    fps=args.fps,
                    codec="libx264",
                    quality=8,
                    macro_block_size=1,
                )
                frame_count = 0
                try:
                    def render_callback(current_env: Any) -> None:
                        nonlocal frame_count
                        frame = capture_frontview(current_env, args.width, args.height)
                        writer.append_data(frame)
                        frame_count += 1
                        if not (task_dir / "environment.png").exists():
                            Image.fromarray(frame).save(task_dir / "environment.png")

                    result = rollout(
                        env=env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        seeds=[seed],
                        render_callback=render_callback,
                    )
                    success = bool(result["success"].any().item())
                finally:
                    writer.close()

                status = "success" if success else "failure"
                if status not in found:
                    destination = task_dir / f"{status}.mp4"
                    temporary_video.replace(destination)
                    found[status] = {
                        "attempt": attempt,
                        "seed": seed,
                        "init_state_id": init_state_id,
                        "video": str(destination.resolve()),
                        "frames": frame_count,
                    }
                    print(
                        json.dumps(
                            {"suite": args.suite, "task_id": task_id, "status": status, **found[status]},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                else:
                    temporary_video.unlink(missing_ok=True)

                if len(found) == 2:
                    break

            if len(found) != 2:
                raise RuntimeError(
                    f"Could not find both success and failure for {args.suite} task {task_id}; found={list(found)}"
                )
            metadata = {
                "suite": args.suite,
                "task_id": task_id,
                "task_description": description,
                "checkpoint": str(checkpoint),
                "frontview_camera": "frontview",
                "resolution": {"width": args.width, "height": args.height},
                "fps": args.fps,
                "environment_image": str((task_dir / "environment.png").resolve()),
                "videos": found,
            }
            (task_dir / "metadata.json").write_text(
                json.dumps(jsonable(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
            )

        print(json.dumps({"status": "passed", "output_dir": str(args.output_dir.resolve())}, indent=2))
    finally:
        close_envs(envs)


if __name__ == "__main__":
    main()
