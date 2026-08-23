#!/usr/bin/env python3
"""Verify parameter updates and feature statistics in the stage-6 checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("models/upstream/smolvla_base"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/runs/sft/train/smolvla_libero_debug/checkpoints/000100/pretrained_model"
        ),
    )
    parser.add_argument("--stats-count", type=int, default=284)
    return parser.parse_args()


def parameter_group(key: str) -> str:
    if key.startswith("model.vlm_with_expert.vlm."):
        return "vlm"
    if key.startswith("model.vlm_with_expert.lm_expert."):
        return "expert"
    return "projection"


def main() -> None:
    args = parse_args()
    base_path = args.base / "model.safetensors"
    trained_path = args.checkpoint / "model.safetensors"
    counts: Counter[tuple[str, str]] = Counter()
    dtype_changes = Counter()

    with safe_open(base_path, framework="pt", device="cpu") as base, safe_open(
        trained_path, framework="pt", device="cpu"
    ) as trained:
        if set(base.keys()) != set(trained.keys()):
            raise RuntimeError("Base and trained checkpoints contain different tensor keys")
        for key in base.keys():
            before = base.get_tensor(key)
            after = trained.get_tensor(key)
            if before.shape != after.shape:
                raise RuntimeError(f"Shape changed for {key}: {before.shape} -> {after.shape}")
            group = parameter_group(key)
            same_value = torch.equal(before.float(), after.float())
            counts[group, "same" if same_value else "changed"] += 1
            if before.dtype != after.dtype:
                dtype_changes[group] += 1

    config = json.loads((args.checkpoint / "config.json").read_text())
    action_dim = int(config["output_features"]["action"]["shape"][0])
    stats_path = (
        args.checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    )
    with safe_open(stats_path, framework="pt", device="cpu") as stats:
        state_dim = int(stats.get_tensor("observation.state.mean").numel())
        stats_action_dim = int(stats.get_tensor("action.mean").numel())
        state_count = int(stats.get_tensor("observation.state.count").item())
        action_count = int(stats.get_tensor("action.count").item())

    if counts["vlm", "changed"]:
        raise RuntimeError("Frozen VLM tensor values changed")
    if not counts["expert", "changed"] or not counts["projection", "changed"]:
        raise RuntimeError("Expected trainable tensors did not change")
    if (action_dim, stats_action_dim, state_dim) != (7, 7, 8):
        raise RuntimeError("Checkpoint feature dimensions are invalid")
    if (state_count, action_count) != (args.stats_count, args.stats_count):
        raise RuntimeError(
            f"Checkpoint statistics count is {state_count}/{action_count}, "
            f"expected {args.stats_count}"
        )

    result = {
        "tensor_counts": {f"{group}_{status}": count for (group, status), count in counts.items()},
        "dtype_changes": dict(dtype_changes),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "stats_count": state_count,
    }
    print(json.dumps(result, indent=2))
    print("PASS: frozen VLM and trained expert/projection parameters are correct.")


if __name__ == "__main__":
    main()
