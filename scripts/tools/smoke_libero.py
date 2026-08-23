#!/usr/bin/env python3
"""Reset and step one headless LIBERO environment and save its camera frames."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/libero_smoke"))
    parser.add_argument("--assets-dir", type=Path, default=Path("assets/libero"))
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def configure_libero(output_dir: Path, assets_dir: Path) -> Path:
    """Create LIBERO's required path config without an interactive prompt."""
    spec = importlib.util.find_spec("libero")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("Could not locate the installed LIBERO package")

    package_dir = Path(next(iter(spec.submodule_search_locations)))
    benchmark_root = package_dir / "libero"
    config_dir = output_dir / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())

    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(output_dir.resolve() / "datasets"),
        "assets": str(assets_dir.resolve()),
    }
    config_file = config_dir / "config.yaml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return config_file


def prepare_libero_assets(output_dir: Path, assets_dir: Path) -> Path:
    """Download LIBERO assets to the project disk and register that location."""
    os.environ.setdefault("HF_HOME", str((output_dir / "hf_cache").resolve()))

    import libero.libero as libero_api
    from libero.libero.utils.download_utils import download_assets_from_huggingface

    assets_dir = assets_dir.resolve()
    downloaded = Path(download_assets_from_huggingface(download_dir=str(assets_dir)))
    libero_api._assets_path_cache = str(downloaded)
    return downloaded


def shape_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: shape_tree(item) for key, item in value.items()}
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def save_observation_images(observation: dict[str, Any], output_dir: Path, prefix: str) -> None:
    cameras = observation["pixels"]
    for name, batch in cameras.items():
        image = np.asarray(batch[0])
        visualization = image[::-1, ::-1]
        Image.fromarray(visualization).save(output_dir / f"{prefix}_{name}.png")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_file = configure_libero(args.output_dir, args.assets_dir)
    assets_dir = prepare_libero_assets(args.output_dir, args.assets_dir)

    from lerobot.envs.configs import LiberoEnv

    config = LiberoEnv(
        task=args.suite,
        task_ids=[args.task_id],
        fps=20,
        obs_type="pixels_agent_pos",
        observation_height=args.image_size,
        observation_width=args.image_size,
        render_mode="rgb_array",
        init_states=True,
        hard_reset=True,
        control_mode="relative",
    )
    env = config.create_envs(n_envs=1, use_async_envs=False)[args.suite][args.task_id]

    try:
        observation, reset_info = env.reset(seed=args.seed)
        save_observation_images(observation, args.output_dir, "reset")

        action = np.zeros((1, 7), dtype=np.float32)
        action[:, -1] = -1.0
        rewards = []
        terminated = None
        truncated = None
        step_info = None
        for _ in range(args.steps):
            observation, reward, terminated, truncated, step_info = env.step(action)
            rewards.append(np.asarray(reward).tolist())

        save_observation_images(observation, args.output_dir, "final")
        render_frame = np.asarray(env.call("render")[0])
        Image.fromarray(render_frame).save(args.output_dir / "render.png")

        result = {
            "suite": args.suite,
            "task_id": args.task_id,
            "seed": args.seed,
            "steps": args.steps,
            "mujoco_gl": os.environ["MUJOCO_GL"],
            "libero_config": str(config_file.resolve()),
            "libero_assets": str(assets_dir),
            "action_shape": list(action.shape),
            "action_min": float(action.min()),
            "action_max": float(action.max()),
            "observation": shape_tree(observation),
            "reset_info": to_jsonable(reset_info),
            "rewards": rewards,
            "terminated": to_jsonable(terminated),
            "truncated": to_jsonable(truncated),
            "step_info": to_jsonable(step_info),
            "render_shape": list(render_frame.shape),
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
