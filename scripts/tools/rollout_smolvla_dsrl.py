#!/usr/bin/env python3
"""Collect one LIBERO episode with random structured SmolVLA noise."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.utils.random_utils import set_seed

from tinyvla.dsrl import ReplayBuffer
from tinyvla.dsrl.rollout import RolloutConfig, rollout_episode
from tinyvla.libero import make_libero_config, make_smolvla_stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sft/libero_40tasks"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/dsrl/step4"))
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--execute-horizon", type=int, default=20)
    parser.add_argument("--discount", type=float, default=0.999)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing SmolVLA checkpoint: {checkpoint}")

    set_seed(args.seed)
    env_config = make_libero_config(args.suite, [args.task_id], args.max_episode_steps)
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs[args.suite][args.task_id]

    try:
        stack = make_smolvla_stack(checkpoint, env_config, args.device)
        policy = stack.policy
        preprocessor = stack.preprocessor
        postprocessor = stack.postprocessor
        env_preprocessor = stack.env_preprocessor
        env_postprocessor = stack.env_postprocessor
        rollout_config = RolloutConfig(
            execute_horizon=args.execute_horizon,
            max_episode_steps=args.max_episode_steps,
            discount=args.discount,
            seed=args.seed,
        )
        capacity = (args.max_episode_steps + args.execute_horizon - 1) // args.execute_horizon
        buffer = ReplayBuffer(capacity=capacity, seed=args.seed)
        result = rollout_episode(
            env,
            policy,
            env_preprocessor,
            preprocessor,
            postprocessor,
            env_postprocessor,
            buffer,
            rollout_config,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        buffer_path = args.output_dir / "replay_buffer.npz"
        result_path = args.output_dir / "rollout.json"
        buffer.save(buffer_path)
        report = {
            "status": "passed",
            "suite": args.suite,
            "task_id": args.task_id,
            "seed": args.seed,
            "checkpoint": str(checkpoint),
            "policy_frozen": not any(parameter.requires_grad for parameter in policy.parameters()),
            "rollout_config": {
                "execute_horizon": args.execute_horizon,
                "chunk_size": 50,
                "action_dim": 32,
                "max_episode_steps": args.max_episode_steps,
                "discount": args.discount,
                "bootstrap_on_truncation": True,
            },
            "episode": result,
            "buffer": buffer.diagnostics(),
            "buffer_path": str(buffer_path.resolve()),
        }
        with result_path.open("w") as file:
            json.dump(report, file, indent=2)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        close_envs(envs)


if __name__ == "__main__":
    main()
