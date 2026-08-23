#!/usr/bin/env python3
"""Evaluate native, structured-random, and learned SmolVLA noise."""

from __future__ import annotations

import argparse
import json
import math
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

from tinyvla.dsrl import SAC
from tinyvla.dsrl.rollout import RolloutConfig, rollout_episode
from tinyvla.libero import make_libero_config, make_smolvla_stack


MAX_EPISODE_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
VALID_MODES = (
    "native",
    "structured_random",
    "structured_gaussian",
    "learned",
    "learned_deterministic",
    "learned_stochastic",
)
LEARNED_MODES = {"learned", "learned_deterministic", "learned_stochastic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--suite")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--modes", choices=VALID_MODES, nargs="+")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--init-state-start", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if args.config is not None:
        with args.config.open() as file:
            raw = yaml.safe_load(file) or {}
    environment = raw.get("environment", {})
    evaluation = raw.get("evaluation", {})

    suite = args.suite or environment.get("suite")
    task_ids = args.task_ids or environment.get("task_ids")
    if suite is None or task_ids is None:
        raise ValueError("suite and task_ids are required")
    if suite not in MAX_EPISODE_STEPS:
        raise ValueError(f"Unsupported LIBERO suite: {suite}")

    modes = list(args.modes or evaluation.get("modes", []))
    if not modes or len(modes) != len(set(modes)) or any(mode not in VALID_MODES for mode in modes):
        raise ValueError(f"modes must be unique values from {VALID_MODES}")
    episodes = int(args.episodes or evaluation.get("episodes", 0))
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    actor_checkpoint = args.actor_checkpoint or raw.get("actor_checkpoint")
    if LEARNED_MODES.intersection(modes) and actor_checkpoint is None:
        raise ValueError("actor_checkpoint is required for learned evaluation")
    execute_horizon = int(environment.get("execute_horizon", 20))
    if execute_horizon != 20:
        raise ValueError("Step 7 requires execute_horizon=20")
    init_state_start = int(
        args.init_state_start
        if args.init_state_start is not None
        else environment.get("init_state_start", 0)
    )
    if init_state_start < 0:
        raise ValueError("init_state_start must be non-negative")

    return {
        "seed": int(args.seed if args.seed is not None else raw.get("seed", 2000)),
        "checkpoint": str(args.checkpoint or raw.get("checkpoint", "models/sft/libero_40tasks")),
        "actor_checkpoint": str(actor_checkpoint) if actor_checkpoint is not None else None,
        "output_dir": str(args.output_dir or raw.get("output_dir", "outputs/runs/dsrl/step7")),
        "device": str(args.device or raw.get("device", "cuda")),
        "environment": {
            "suite": str(suite),
            "task_ids": [int(task_id) for task_id in task_ids],
            "max_episode_steps": int(
                args.max_episode_steps
                if args.max_episode_steps is not None
                else environment.get("max_episode_steps", MAX_EPISODE_STEPS[suite])
            ),
            "execute_horizon": execute_horizon,
            "init_state_start": init_state_start,
            "discount": float(environment.get("discount", 0.999)),
            "sac_image_keys": list(
                environment.get("sac_image_keys", ["observation.images.image"])
            ),
        },
        "evaluation": {
            "episodes": episodes,
            "modes": modes,
            "bootstrap_samples": int(evaluation.get("bootstrap_samples", 10_000)),
            "confidence_level": float(evaluation.get("confidence_level", 0.95)),
        },
    }


def bootstrap_interval(
    values: list[float], samples: int, confidence_level: float, seed: int
) -> list[float] | None:
    if not values:
        return None
    if samples <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("Invalid bootstrap configuration")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return [float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))]


def summarize_values(
    values: list[float], samples: int, confidence_level: float, seed: int
) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "standard_error": None, "bootstrap_ci": None}
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "standard_error": standard_error,
        "bootstrap_ci": bootstrap_interval(values, samples, confidence_level, seed),
    }


def summarize_results(
    results: list[dict[str, Any]], samples: int, confidence_level: float, seed: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    groups = sorted({(int(item["task_id"]), str(item["mode"])) for item in results})
    for group_index, (task_id, mode) in enumerate(groups):
        episodes = [
            item for item in results if int(item["task_id"]) == task_id and item["mode"] == mode
        ]
        successes = [float(item["success"]) for item in episodes]
        completion_steps = [
            float(item["completion_steps"])
            for item in episodes
            if item["completion_steps"] is not None
        ]
        episode_steps = [float(item["environment_steps"]) for item in episodes]
        group_seed = seed + group_index * 3
        summary.setdefault(str(task_id), {})[mode] = {
            "episodes": len(episodes),
            "successes": int(sum(successes)),
            "success_rate": summarize_values(
                successes, samples, confidence_level, group_seed
            ),
            "completion_steps_success_only": summarize_values(
                completion_steps, samples, confidence_level, group_seed + 1
            ),
            "environment_steps": summarize_values(
                episode_steps, samples, confidence_level, group_seed + 2
            ),
        }
    return summary


def validate_fair_protocol(
    results: list[dict[str, Any]], task_ids: list[int], modes: list[str], episodes: int
) -> None:
    for task_id in task_ids:
        expected = None
        for mode in modes:
            protocol = [
                (item["episode_index"], item["init_state_index"], item["seed"])
                for item in results
                if item["task_id"] == task_id and item["mode"] == mode
            ]
            if len(protocol) != episodes:
                raise RuntimeError(f"Incomplete evaluation for task {task_id}, mode {mode}")
            if expected is None:
                expected = protocol
            elif protocol != expected:
                raise RuntimeError(f"Evaluation protocol differs for task {task_id}, mode {mode}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)


def load_actor(path: Path, device: torch.device, suite: str, task_ids: list[int]) -> SAC:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") == 1 and "sac" in state:
        sac_state = state["sac"]
        trained_environment = state["config"]["environment"]
    elif "model" in state and "config" in state:
        sac_state = state
        trained_environment = state.get("metadata", {}).get("environment")
    else:
        raise ValueError("Expected a unified or SAC-only DSRL checkpoint")
    if trained_environment is None:
        raise ValueError("SAC checkpoint does not identify its training task")
    if trained_environment["suite"] != suite or task_ids != [int(trained_environment["task_id"])]:
        raise ValueError("Learned Actor must be evaluated on its training task")
    learner, _ = SAC.from_checkpoint_state(sac_state, device=device)
    return learner.eval()


def main() -> None:
    args = parse_args()
    settings = load_settings(args)
    environment = settings["environment"]
    evaluation = settings["evaluation"]
    suite = environment["suite"]
    task_ids = environment["task_ids"]
    modes = evaluation["modes"]
    seed = settings["seed"]
    device = torch.device(settings["device"])
    checkpoint = Path(settings["checkpoint"]).resolve()
    output_dir = Path(settings["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "evaluation.json"
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing SmolVLA checkpoint: {checkpoint}")

    set_seed(seed)
    env_config = make_libero_config(
        suite, task_ids, int(environment["max_episode_steps"])
    )
    stack = make_smolvla_stack(checkpoint, env_config, device)
    policy = stack.policy
    preprocessor = stack.preprocessor
    postprocessor = stack.postprocessor
    env_preprocessor = stack.env_preprocessor
    env_postprocessor = stack.env_postprocessor
    learner = (
        load_actor(Path(settings["actor_checkpoint"]).resolve(), device, suite, task_ids)
        if LEARNED_MODES.intersection(modes)
        else None
    )
    if learner is not None:
        expected_channels = 3 * len(environment["sac_image_keys"])
        if learner.config.image_shape != (expected_channels, 64, 64):
            raise ValueError("Evaluation camera configuration does not match the Actor checkpoint")

    results: list[dict[str, Any]] = []
    for mode in modes:
        envs = make_env(env_config, n_envs=1, use_async_envs=False)
        try:
            for task_id in task_ids:
                env = envs[suite][task_id]
                env.set_attr("init_state_id", environment["init_state_start"])
                for episode_index in range(evaluation["episodes"]):
                    episode_seed = seed + episode_index
                    noise_rng = np.random.default_rng(episode_seed)

                    if mode == "native":
                        noise_fn = lambda _, rng=noise_rng: rng.standard_normal(
                            (50, 32)
                        ).astype(np.float32)
                    elif mode == "structured_random":
                        noise_fn = lambda _, rng=noise_rng: rng.uniform(
                            -1.0, 1.0, size=32
                        ).astype(np.float32)
                    elif mode == "structured_gaussian":
                        noise_fn = lambda _, rng=noise_rng: rng.standard_normal(32).astype(
                            np.float32
                        )
                    else:
                        assert learner is not None

                        def noise_fn(observation: dict[str, np.ndarray]) -> np.ndarray:
                            pixels = torch.from_numpy(observation["pixels"]).unsqueeze(0)
                            state = torch.from_numpy(observation["state"]).unsqueeze(0)
                            return (
                                learner.act(
                                    pixels,
                                    state,
                                    deterministic=mode != "learned_stochastic",
                                )[0]
                                .cpu()
                                .numpy()
                                .astype(np.float32, copy=False)
                            )

                    set_seed(episode_seed)
                    result = rollout_episode(
                        env,
                        policy,
                        env_preprocessor,
                        preprocessor,
                        postprocessor,
                        env_postprocessor,
                        None,
                        RolloutConfig(
                            execute_horizon=environment["execute_horizon"],
                            max_episode_steps=environment["max_episode_steps"],
                            discount=environment["discount"],
                            seed=episode_seed,
                            sac_image_keys=tuple(environment["sac_image_keys"]),
                        ),
                        noise_fn=noise_fn,
                        noise_source=mode,
                    )
                    result.update(
                        {
                            "mode": mode,
                            "task_id": task_id,
                            "episode_index": episode_index,
                            "init_state_index": environment["init_state_start"] + episode_index,
                            "seed": episode_seed,
                            "completion_steps": result["environment_steps"]
                            if result["success"]
                            else None,
                        }
                    )
                    results.append(result)
                    report = {
                        "status": "running",
                        "config": settings,
                        "episodes": results,
                        "summary": summarize_results(
                            results,
                            evaluation["bootstrap_samples"],
                            evaluation["confidence_level"],
                            seed,
                        ),
                    }
                    write_json(result_path, report)
                    print(
                        json.dumps(
                            {
                                "mode": mode,
                                "task_id": task_id,
                                "episode": episode_index + 1,
                                "seed": episode_seed,
                                "success": result["success"],
                                "steps": result["environment_steps"],
                            }
                        ),
                        flush=True,
                    )
        finally:
            close_envs(envs)

    validate_fair_protocol(results, task_ids, modes, evaluation["episodes"])
    final_report = {
        "status": "passed",
        "config": settings,
        "fair_protocol_validated": True,
        "episodes": results,
        "summary": summarize_results(
            results,
            evaluation["bootstrap_samples"],
            evaluation["confidence_level"],
            seed,
        ),
    }
    write_json(result_path, final_report)
    print(json.dumps(final_report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
