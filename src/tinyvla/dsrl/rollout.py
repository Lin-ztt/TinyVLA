"""Chunk-level SmolVLA rollout utilities for DSRL."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class RolloutConfig:
    execute_horizon: int = 20
    chunk_size: int = 50
    action_dim: int = 32
    max_episode_steps: int = 300
    discount: float = 0.999
    bootstrap_on_truncation: bool = True
    seed: int = 1000
    sac_image_keys: tuple[str, ...] = ("observation.images.image",)

    def __post_init__(self) -> None:
        if not 0 < self.execute_horizon <= self.chunk_size:
            raise ValueError("execute_horizon must be in [1, chunk_size]")
        if self.chunk_size != 50 or self.action_dim != 32:
            raise ValueError("The first DSRL version requires chunk_size=50 and action_dim=32")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if len(self.sac_image_keys) not in (1, 2) or len(set(self.sac_image_keys)) != len(
            self.sac_image_keys
        ):
            raise ValueError("sac_image_keys must contain one or two unique image keys")


def prepare_observation(
    observation: dict[str, Any], task_description: str, env_preprocessor: Any
) -> dict[str, Any]:
    from lerobot.envs.utils import preprocess_observation

    processed = preprocess_observation(observation)
    batch_size = next(value.shape[0] for value in processed.values() if isinstance(value, Tensor))
    processed["task"] = [task_description] * batch_size
    return env_preprocessor(processed)


def extract_sac_observation(
    observation: dict[str, Any],
    image_keys: tuple[str, ...] = ("observation.images.image",),
) -> dict[str, np.ndarray]:
    state = observation["observation.state"]
    if state.shape != (1, 8):
        raise ValueError(f"Expected state shape [1, 8], got {tuple(state.shape)}")

    resized_images = []
    for image_key in image_keys:
        pixels = observation[image_key]
        if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
            raise ValueError(
                f"Expected image shape [1, 3, H, W] for {image_key}, "
                f"got {tuple(pixels.shape)}"
            )
        resized = F.interpolate(
            pixels.float(), size=(64, 64), mode="bilinear", antialias=True
        )
        resized_min = float(resized.min().item())
        resized_max = float(resized.max().item())
        if resized_min < -1e-6 or resized_max > 1.0 + 1e-6:
            raise ValueError(
                "Expected processor image values in [0, 1], "
                f"got [{resized_min}, {resized_max}]"
            )
        resized_images.append(resized.clamp_(0.0, 1.0))
    pixels_uint8 = (
        torch.cat(resized_images, dim=1)
        .mul(255.0)
        .round()
        .clamp_(0, 255)
        .to(torch.uint8)
    )
    return {
        "pixels": pixels_uint8[0].cpu().numpy(),
        "state": state[0].to(torch.float32).cpu().numpy(),
    }


def expand_structured_noise(noise: Tensor, chunk_size: int = 50) -> Tensor:
    if noise.ndim != 2 or noise.shape[1] != 32:
        raise ValueError(f"Expected noise shape [B, 32], got {tuple(noise.shape)}")
    return noise.unsqueeze(1).repeat(1, chunk_size, 1)


def infer_action_chunk(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_postprocessor: Any,
    observation: dict[str, Any],
    noise: np.ndarray,
    chunk_size: int = 50,
) -> tuple[np.ndarray, float]:
    policy_observation = preprocessor(copy.deepcopy(observation))
    device = policy_observation["observation.state"].device
    noise_tensor = torch.from_numpy(noise).to(device)
    if noise_tensor.shape == (32,):
        full_noise = expand_structured_noise(noise_tensor.reshape(1, 32), chunk_size)
    elif noise_tensor.shape == (chunk_size, 32):
        full_noise = noise_tensor.unsqueeze(0)
    else:
        raise ValueError(
            f"Expected noise shape (32,) or ({chunk_size}, 32), got {tuple(noise_tensor.shape)}"
        )

    start = time.perf_counter()
    with torch.inference_mode():
        normalized_actions = policy.predict_action_chunk(policy_observation, noise=full_noise)
        actions = postprocessor(normalized_actions)
        actions = env_postprocessor({"action": actions})["action"]
    inference_seconds = time.perf_counter() - start

    action_numpy = actions.detach().to("cpu", dtype=torch.float32).numpy()
    if action_numpy.shape != (1, chunk_size, 7):
        raise ValueError(
            f"Expected action chunk shape [1, {chunk_size}, 7], got {action_numpy.shape}"
        )
    if not np.isfinite(action_numpy).all():
        raise ValueError("SmolVLA action chunk contains NaN or Inf")
    return action_numpy, inference_seconds


def _first_flag(value: Any) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _is_success(info: dict[str, Any]) -> bool:
    if "is_success" in info:
        return _first_flag(info["is_success"])
    final_info = info.get("final_info")
    if isinstance(final_info, dict) and "is_success" in final_info:
        return _first_flag(final_info["is_success"])
    if final_info is not None:
        first = np.asarray(final_info, dtype=object).reshape(-1)[0]
        return bool(first.get("is_success", False)) if isinstance(first, dict) else False
    return False


def rollout_episode(
    env: Any,
    policy: Any,
    env_preprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_postprocessor: Any,
    buffer: ReplayBuffer | None,
    config: RolloutConfig | None = None,
    noise_fn: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
    noise_source: str = "structured_random",
    render_callback: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    config = config or RolloutConfig()
    rng = np.random.default_rng(config.seed)
    observation, _ = env.reset(seed=config.seed)
    task_description = str(env.call("task_description")[0])
    policy.reset()
    if render_callback is not None:
        render_callback(env)

    episode_steps = 0
    episode_return = 0.0
    episode_success = False
    transitions: list[dict[str, Any]] = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        processed = prepare_observation(observation, task_description, env_preprocessor)
        sac_observation = extract_sac_observation(processed, config.sac_image_keys)
        if noise_fn is None:
            noise = rng.uniform(-1.0, 1.0, size=config.action_dim).astype(np.float32)
        else:
            noise = np.asarray(noise_fn(sac_observation), dtype=np.float32)
        structured_noise = noise.shape == (config.action_dim,)
        full_noise = noise.shape == (config.chunk_size, config.action_dim)
        if not structured_noise and not full_noise:
            raise ValueError(
                "Expected selected noise shape "
                f"({config.action_dim},) or ({config.chunk_size}, {config.action_dim}), "
                f"got {noise.shape}"
            )
        if not np.isfinite(noise).all():
            raise ValueError("Selected noise must be finite")
        if full_noise and buffer is not None:
            raise ValueError("Full SmolVLA noise cannot be stored as a 32-dimensional SAC action")
        action_chunk, inference_seconds = infer_action_chunk(
            policy,
            preprocessor,
            postprocessor,
            env_postprocessor,
            processed,
            noise,
            config.chunk_size,
        )

        executed_steps = 0
        chunk_success = False
        env_truncated = False
        remaining_steps = config.max_episode_steps - episode_steps
        for action_index in range(min(config.execute_horizon, remaining_steps)):
            observation, reward, env_terminated, step_truncated, info = env.step(
                action_chunk[:, action_index]
            )
            executed_steps += 1
            episode_steps += 1
            episode_return += float(np.asarray(reward).reshape(-1)[0])
            chunk_success = chunk_success or _is_success(info)
            terminated = _first_flag(env_terminated) or chunk_success
            env_truncated = _first_flag(step_truncated)
            if render_callback is not None:
                render_callback(env)
            if terminated or env_truncated:
                break

        if executed_steps == 0:
            raise RuntimeError("Rollout produced an empty chunk transition")
        truncated = env_truncated or (episode_steps >= config.max_episode_steps and not terminated)
        episode_success = episode_success or chunk_success

        mask = 0.0 if terminated or (truncated and not config.bootstrap_on_truncation) else 1.0
        if buffer is not None:
            next_processed = prepare_observation(observation, task_description, env_preprocessor)
            next_sac_observation = extract_sac_observation(
                next_processed, config.sac_image_keys
            )
            buffer.add(
                observation=sac_observation,
                action=noise,
                reward=0.0 if chunk_success else -1.0,
                discount=config.discount**executed_steps,
                mask=mask,
                next_observation=next_sac_observation,
                terminated=terminated,
                truncated=truncated,
                executed_steps=executed_steps,
            )
        transitions.append(
            {
                "chunk_index": len(transitions),
                "executed_steps": executed_steps,
                "environment_steps": episode_steps,
                "reward": 0.0 if chunk_success else -1.0,
                "discount": config.discount**executed_steps,
                "mask": mask,
                "terminated": terminated,
                "truncated": truncated,
                "success": chunk_success,
                "noise_source": noise_source,
                "noise_min": float(noise.min()),
                "noise_max": float(noise.max()),
                "action_min": float(action_chunk.min()),
                "action_max": float(action_chunk.max()),
                "inference_seconds": inference_seconds,
            }
        )

    return {
        "task_description": task_description,
        "success": episode_success,
        "terminated": terminated,
        "truncated": truncated,
        "environment_steps": episode_steps,
        "chunk_transitions": len(transitions),
        "environment_return": episode_return,
        "transitions": transitions,
    }
