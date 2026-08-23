#!/usr/bin/env python3
"""Inspect the local LeRobot LIBERO dataset used by SmolVLA."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader


REPO_ID = "lerobot/libero"
REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/libero"))
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--random-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def tensor_info(value: object) -> str:
    if isinstance(value, torch.Tensor):
        return f"shape={tuple(value.shape)}, dtype={value.dtype}"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def rounded(value: torch.Tensor) -> list[float]:
    return [round(float(item), 5) for item in value.flatten()]


def check_download(root: Path) -> None:
    manifest_path = root / ".cache" / "huggingface" / "trees" / f"{REVISION}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing download manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    missing = [name for name in manifest if not (root / name).is_file()]
    wrong_size = [
        name
        for name, metadata in manifest.items()
        if (root / name).is_file() and (root / name).stat().st_size != metadata["size"]
    ]
    if missing or wrong_size:
        raise RuntimeError(f"Incomplete download: missing={missing}, wrong_size={wrong_size}")

    total_bytes = sum(metadata["size"] for metadata in manifest.values())
    print("[download]")
    print(f"revision: {REVISION}")
    print(f"files: {len(manifest)}")
    print(f"size: {total_bytes / 1024**3:.3f} GiB")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    check_download(root)

    dataset = LeRobotDataset(
        REPO_ID,
        root=root,
        revision=REVISION,
        delta_timestamps={"action": [i / 10 for i in range(args.chunk_size)]},
        download_videos=False,
        video_backend="pyav",
    )

    expected_features = {
        "observation.images.image": (256, 256, 3),
        "observation.images.image2": (256, 256, 3),
        "observation.state": (8,),
        "action": (7,),
    }
    for key, shape in expected_features.items():
        actual = tuple(dataset.features[key]["shape"])
        if actual != shape:
            raise RuntimeError(f"Unexpected {key} shape: {actual}, expected {shape}")

    print("\n[dataset]")
    print(f"root: {root}")
    print(f"robot: {dataset.meta.robot_type}")
    print(f"fps: {dataset.fps}")
    print(f"tasks: {dataset.meta.total_tasks}")
    print(f"episodes: {dataset.num_episodes}")
    print(f"frames: {len(dataset)}")
    print("features:")
    for key, feature in dataset.features.items():
        print(f"  {key}: shape={tuple(feature['shape'])}, dtype={feature['dtype']}")

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    print("\n[first batch]")
    for key, value in batch.items():
        print(f"  {key}: {tensor_info(value)}")
    print(f"  task[0]: {batch['task'][0]}")
    print(f"  state[0]: {rounded(batch['observation.state'][0])}")
    print(f"  action[0, 0]: {rounded(batch['action'][0, 0])}")

    print("\n[normalization statistics]")
    for key in ("observation.state", "action"):
        stats = dataset.meta.stats[key]
        print(f"  {key}")
        for name in ("min", "max", "mean", "std"):
            values = [round(float(item), 5) for item in stats[name]]
            print(f"    {name}: {values}")

    task_names = {
        int(row["task_index"]): str(task)
        for task, row in dataset.meta.tasks.iterrows()
    }
    rng = random.Random(args.seed)
    episode_ids = rng.sample(
        range(dataset.num_episodes), min(args.random_episodes, dataset.num_episodes)
    )

    print("\n[random episode checks]")
    for episode_id in episode_ids:
        episode = dataset.meta.episodes[episode_id]
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        indices = (start, start + (stop - start) // 2, stop - 1)
        print(f"  episode={episode_id}, length={stop - start}")

        for position, index in zip(("first", "middle", "last"), indices, strict=True):
            sample = dataset[index]
            if int(sample["episode_index"]) != episode_id:
                raise RuntimeError(f"Frame {index} crossed an episode boundary")
            for key in expected_features:
                if not torch.isfinite(sample[key]).all():
                    raise RuntimeError(f"Non-finite value in {key} at frame {index}")

            pad_count = int(sample["action_is_pad"].sum())
            print(
                f"    {position}: global={index}, frame={int(sample['frame_index'])}, "
                f"time={float(sample['timestamp']):.1f}s, task={int(sample['task_index'])}, "
                f"valid_actions={args.chunk_size - pad_count}"
            )
            print(f"      language: {task_names[int(sample['task_index'])]}")
            print(f"      state: {rounded(sample['observation.state'])}")
            print(f"      action[0]: {rounded(sample['action'][0])}")
            for camera in ("observation.images.image", "observation.images.image2"):
                image = sample[camera]
                print(
                    f"      {camera}: shape={tuple(image.shape)}, "
                    f"range=[{float(image.min()):.3f}, {float(image.max()):.3f}]"
                )

        last = dataset[stop - 1]
        if int(last["action_is_pad"].sum()) != args.chunk_size - 1:
            raise RuntimeError(f"Incorrect tail padding in episode {episode_id}")
        if not torch.allclose(last["action"], last["action"][0].expand_as(last["action"])):
            raise RuntimeError(f"Padded actions do not repeat the last action in episode {episode_id}")

    print("\nPASS: LIBERO data matches the SmolVLA training interface.")


if __name__ == "__main__":
    main()
