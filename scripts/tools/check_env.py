#!/usr/bin/env python3
"""Print the runtime information required by this project."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
from pathlib import Path

import torch


PACKAGES = (
    "lerobot",
    "hf-libero",
    "mujoco",
    "robosuite",
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "accelerate",
    "safetensors",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    usage = shutil.disk_usage(Path.cwd())
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in PACKAGES},
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "disk_free_gib": round(usage.free / 1024**3, 2),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    missing = [name for name, version in report["packages"].items() if version is None]
    if missing:
        raise SystemExit(f"Missing packages: {', '.join(missing)}")
    if not report["cuda_available"]:
        raise SystemExit("CUDA is not available")


if __name__ == "__main__":
    main()
