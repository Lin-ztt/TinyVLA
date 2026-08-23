"""Shared LIBERO and frozen SmolVLA initialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


IMAGE_RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


@dataclass(frozen=True)
class SmolVLAStack:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any
    env_postprocessor: Any


def make_libero_config(
    suite: str,
    task_ids: list[int],
    episode_length: int,
    observation_height: int = 256,
    observation_width: int = 256,
) -> LiberoEnv:
    """Build the common LIBERO environment configuration."""
    return LiberoEnv(
        task=suite,
        task_ids=task_ids,
        episode_length=episode_length,
        observation_height=observation_height,
        observation_width=observation_width,
        control_mode="relative",
    )


def make_smolvla_stack(
    checkpoint: Path,
    env_config: LiberoEnv,
    device: torch.device | str,
) -> SmolVLAStack:
    """Load a frozen SmolVLA and all processors used by LIBERO rollouts."""
    checkpoint = Path(checkpoint).resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing SmolVLA checkpoint: {checkpoint}")

    policy_config = SmolVLAConfig.from_pretrained(checkpoint)
    policy_config.pretrained_path = checkpoint
    policy_config.device = str(device)
    policy_config.use_amp = False
    policy_config.load_vlm_weights = False
    policy = make_policy(policy_config, env_cfg=env_config, rename_map=IMAGE_RENAME_MAP)
    policy.eval().requires_grad_(False)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": IMAGE_RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config,
        policy_cfg=policy_config,
    )
    return SmolVLAStack(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
    )
