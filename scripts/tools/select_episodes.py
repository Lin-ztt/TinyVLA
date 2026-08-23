#!/usr/bin/env python3
"""Select reproducible LIBERO-Goal subsets for SmolVLA post-training."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import yaml
from lerobot.datasets.compute_stats import RunningQuantileStats
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


REPO_ID = "lerobot/libero"
REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
TARGETS = (
    ("open the middle drawer of the cabinet", 0),
    ("put the bowl on the stove", 1),
    ("turn on the stove", 7),
    ("put the bowl on the plate", 8),
)
GOAL_TARGETS = (
    ("open the middle drawer of the cabinet", 0),
    ("put the bowl on the stove", 1),
    ("put the wine bottle on top of the cabinet", 2),
    ("open the top drawer and put the bowl inside", 3),
    ("put the bowl on top of the cabinet", 4),
    ("push the plate to the front of the stove", 5),
    ("put the cream cheese in the bowl", 6),
    ("turn on the stove", 7),
    ("put the bowl on the plate", 8),
    ("put the wine bottle on the rack", 9),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/libero"))
    parser.add_argument("--output-dir", type=Path, default=Path("configs/tasks"))
    parser.add_argument("--views-dir", type=Path, default=Path("data/views"))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--debug-episodes", type=int, default=2)
    return parser.parse_args()


def task_indices(meta: LeRobotDatasetMetadata) -> dict[str, int]:
    return {str(language): int(row["task_index"]) for language, row in meta.tasks.iterrows()}


def episode_metadata(meta: LeRobotDatasetMetadata) -> dict[int, dict]:
    return {int(row["episode_index"]): row for row in meta.episodes}


def scan_frames(root: Path) -> tuple[pa.Table, dict[int, int], dict[int, int]]:
    table = pads.dataset(root / "data", format="parquet").to_table(
        columns=["episode_index", "task_index", "observation.state", "action"]
    )
    episode_to_task: dict[int, int] = {}
    episode_lengths: dict[int, int] = {}
    for episode, task in zip(
        table["episode_index"].to_pylist(), table["task_index"].to_pylist(), strict=True
    ):
        episode = int(episode)
        task = int(task)
        previous = episode_to_task.setdefault(episode, task)
        if previous != task:
            raise RuntimeError(f"Episode {episode} contains tasks {previous} and {task}")
        episode_lengths[episode] = episode_lengths.get(episode, 0) + 1
    return table, episode_to_task, episode_lengths


def validate_episodes(
    meta: LeRobotDatasetMetadata,
    episode_to_task: dict[int, int],
    scanned_lengths: dict[int, int],
) -> dict[int, dict]:
    rows = episode_metadata(meta)
    if set(rows) != set(episode_to_task):
        raise RuntimeError("Episode metadata and frame data contain different episode indices")
    for episode, row in rows.items():
        expected = int(row["length"])
        if scanned_lengths[episode] != expected:
            raise RuntimeError(
                f"Episode {episode} has {scanned_lengths[episode]} frames, expected {expected}"
            )
    return rows


def choose_episodes(
    episode_to_task: dict[int, int], indices: dict[str, int], count: int, seed: int
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    selected = {}
    for language, _ in TARGETS:
        task_index = indices[language]
        candidates = sorted(
            episode for episode, task in episode_to_task.items() if task == task_index
        )
        if len(candidates) < count:
            raise RuntimeError(f"Task {language!r} only has {len(candidates)} episodes")
        selected[language] = sorted(rng.sample(candidates, count))
    return selected


def subset_stats(table: pa.Table, episodes: list[int], source_stats: dict) -> dict:
    mask = pc.is_in(table["episode_index"], value_set=pa.array(episodes))
    selected = table.filter(mask)
    stats = copy.deepcopy(source_stats)
    for key in ("observation.state", "action"):
        values = np.asarray(selected[key].to_pylist(), dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid values in selected {key}")
        running = RunningQuantileStats()
        running.update(values)
        stats[key] = running.get_statistics()
    return stats


def ensure_symlink(link: Path, target: Path) -> None:
    expected = target.resolve()
    if link.is_symlink():
        if link.resolve() != expected:
            raise RuntimeError(f"Existing symlink {link} points to {link.resolve()}")
        return
    if link.exists():
        raise RuntimeError(f"Cannot create dataset view over existing path: {link}")
    link.symlink_to(Path(os.path.relpath(expected, link.parent.resolve())), target_is_directory=True)


def prepare_view(
    source_root: Path,
    view_root: Path,
    stats: dict,
    episodes: list[int],
    seed: int,
) -> None:
    view_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root / "meta", view_root / "meta", dirs_exist_ok=True)
    ensure_symlink(view_root / "data", source_root / "data")
    if (source_root / "videos").is_dir():
        ensure_symlink(view_root / "videos", source_root / "videos")
    write_stats(stats, view_root)
    marker = {
        "source_root": os.path.relpath(source_root, view_root),
        "revision": REVISION,
        "seed": seed,
        "episodes": episodes,
    }
    (view_root / ".tinyvla_view.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )


def build_config(
    name: str,
    view_root: Path,
    selected: dict[str, list[int]],
    indices: dict[str, int],
    episode_rows: dict[int, dict],
    available: dict[int, int],
    seed: int,
) -> dict:
    tasks = []
    all_episodes = sorted(episode for episodes in selected.values() for episode in episodes)
    target_map = dict(GOAL_TARGETS)
    for language in selected:
        env_task_id = target_map[language]
        episodes = selected[language]
        task_index = indices[language]
        tasks.append(
            {
                "language": language,
                "env_task_id": env_task_id,
                "task_index": task_index,
                "available_episodes": available[task_index],
                "episodes": episodes,
                "frames": sum(int(episode_rows[episode]["length"]) for episode in episodes),
            }
        )
    return {
        "name": name,
        "source": {"repo_id": REPO_ID, "root": "data/libero", "revision": REVISION},
        "dataset": {
            "repo_id": REPO_ID,
            "root": view_root.as_posix(),
            "revision": REVISION,
            "episodes": all_episodes,
        },
        "selection": {"seed": seed, "tasks": len(tasks), "episodes": len(all_episodes)},
        "tasks": tasks,
        "summary": {
            "frames": sum(int(episode_rows[episode]["length"]) for episode in all_episodes)
        },
    }


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.episodes_per_task <= 0 or args.debug_episodes <= 0:
        raise ValueError("Episode counts must be positive")
    if args.debug_episodes > args.episodes_per_task:
        raise ValueError("--debug-episodes cannot exceed --episodes-per-task")

    meta = LeRobotDatasetMetadata(REPO_ID, root=root, revision=REVISION)
    indices = task_indices(meta)
    missing = [language for language, _ in GOAL_TARGETS if language not in indices]
    if missing:
        raise RuntimeError(f"Missing target tasks: {missing}")

    table, episode_to_task, scanned_lengths = scan_frames(root)
    rows = validate_episodes(meta, episode_to_task, scanned_lengths)
    formal = choose_episodes(episode_to_task, indices, args.episodes_per_task, args.seed)
    goal_all = {
        language: sorted(
            episode
            for episode, task in episode_to_task.items()
            if task == indices[language]
        )
        for language, _ in GOAL_TARGETS
    }
    debug_language = TARGETS[0][0]
    debug = {debug_language: formal[debug_language][: args.debug_episodes]}
    available = {
        task_index: sum(task == task_index for task in episode_to_task.values())
        for task_index in indices.values()
    }

    outputs = (
        ("libero_goal_debug", debug),
        ("libero_goal_4tasks", formal),
        ("libero_goal_10tasks_all", goal_all),
    )
    for name, selected in outputs:
        episodes = sorted(episode for values in selected.values() for episode in values)
        stats = subset_stats(table, episodes, meta.stats)
        view_root = args.views_dir / name
        prepare_view(root, view_root.resolve(), stats, episodes, args.seed)
        config = build_config(name, view_root, selected, indices, rows, available, args.seed)
        write_config(args.output_dir / f"{name}.yaml", config)
        print(
            f"{name}: tasks={len(selected)}, episodes={len(episodes)}, "
            f"frames={config['summary']['frames']}, view={view_root}"
        )

    print("PASS: reproducible LIBERO-Goal subsets and dataset views are ready.")


if __name__ == "__main__":
    main()
