"""CPU replay buffer for DSRL chunk-level transitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-schema, preallocated ring buffer."""

    def __init__(self, capacity: int, seed: int = 0, image_channels: int = 3) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if image_channels not in (3, 6):
            raise ValueError("image_channels must be 3 or 6")

        self.capacity = capacity
        self.image_channels = image_channels
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)
        self.observations = {
            "pixels": np.empty((capacity, image_channels, 64, 64), dtype=np.uint8),
            "state": np.empty((capacity, 8), dtype=np.float32),
        }
        self.next_observations = {
            "pixels": np.empty((capacity, image_channels, 64, 64), dtype=np.uint8),
            "state": np.empty((capacity, 8), dtype=np.float32),
        }
        self.actions = np.empty((capacity, 32), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.masks = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self.executed_steps = np.empty(capacity, dtype=np.int32)

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _array(value: Any, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != shape:
            raise ValueError(f"Expected shape {shape}, got {array.shape}")
        return array.astype(dtype, copy=False)

    def add(
        self,
        observation: dict[str, Any],
        action: Any,
        reward: float,
        discount: float,
        mask: float,
        next_observation: dict[str, Any],
        terminated: bool,
        truncated: bool,
        executed_steps: int,
    ) -> None:
        image_shape = (self.image_channels, 64, 64)
        pixels = self._array(observation["pixels"], image_shape, np.dtype(np.uint8))
        state = self._array(observation["state"], (8,), np.dtype(np.float32))
        next_pixels = self._array(
            next_observation["pixels"], image_shape, np.dtype(np.uint8)
        )
        next_state = self._array(next_observation["state"], (8,), np.dtype(np.float32))
        action_array = self._array(action, (32,), np.dtype(np.float32))

        index = self.position
        self.observations["pixels"][index] = pixels
        self.observations["state"][index] = state
        self.next_observations["pixels"][index] = next_pixels
        self.next_observations["state"][index] = next_state
        self.actions[index] = action_array
        self.rewards[index] = reward
        self.discounts[index] = discount
        self.masks[index] = mask
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.executed_steps[index] = executed_steps

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, Any]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self.rng.integers(0, self.size, size=batch_size)

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array[indices])

        return {
            "observations": {
                "pixels": tensor(self.observations["pixels"]),
                "state": tensor(self.observations["state"]),
            },
            "next_observations": {
                "pixels": tensor(self.next_observations["pixels"]),
                "state": tensor(self.next_observations["state"]),
            },
            "actions": tensor(self.actions),
            "rewards": tensor(self.rewards),
            "discounts": tensor(self.discounts),
            "masks": tensor(self.masks),
            "terminated": tensor(self.terminated),
            "truncated": tensor(self.truncated),
            "executed_steps": tensor(self.executed_steps),
        }

    def diagnostics(self) -> dict[str, Any]:
        fields = {
            "observations.pixels": self.observations["pixels"],
            "observations.state": self.observations["state"],
            "next_observations.pixels": self.next_observations["pixels"],
            "next_observations.state": self.next_observations["state"],
            "actions": self.actions,
            "rewards": self.rewards,
            "discounts": self.discounts,
            "masks": self.masks,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "executed_steps": self.executed_steps,
        }
        return {
            "capacity": self.capacity,
            "image_channels": self.image_channels,
            "size": self.size,
            "position": self.position,
            "fields": {
                name: {"shape": list(array[: self.size].shape), "dtype": str(array.dtype)}
                for name, array in fields.items()
            },
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "image_channels": self.image_channels,
            "size": self.size,
            "position": self.position,
            "rng_state": json.loads(json.dumps(self.rng.bit_generator.state)),
            "observation_pixels": self.observations["pixels"][: self.size].copy(),
            "observation_state": self.observations["state"][: self.size].copy(),
            "next_observation_pixels": self.next_observations["pixels"][: self.size].copy(),
            "next_observation_state": self.next_observations["state"][: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "discounts": self.discounts[: self.size].copy(),
            "masks": self.masks[: self.size].copy(),
            "terminated": self.terminated[: self.size].copy(),
            "truncated": self.truncated[: self.size].copy(),
            "executed_steps": self.executed_steps[: self.size].copy(),
        }

    @classmethod
    def from_checkpoint_state(cls, state: dict[str, Any]) -> ReplayBuffer:
        image_channels = int(
            state.get("image_channels", np.asarray(state["observation_pixels"]).shape[1])
        )
        buffer = cls(int(state["capacity"]), image_channels=image_channels)
        buffer.size = int(state["size"])
        buffer.position = int(state["position"])
        if not 0 <= buffer.size <= buffer.capacity:
            raise ValueError("Invalid replay buffer size in checkpoint")
        if not 0 <= buffer.position < buffer.capacity:
            raise ValueError("Invalid replay buffer position in checkpoint")
        buffer.observations["pixels"][: buffer.size] = state["observation_pixels"]
        buffer.observations["state"][: buffer.size] = state["observation_state"]
        buffer.next_observations["pixels"][: buffer.size] = state[
            "next_observation_pixels"
        ]
        buffer.next_observations["state"][: buffer.size] = state["next_observation_state"]
        for name in (
            "actions",
            "rewards",
            "discounts",
            "masks",
            "terminated",
            "truncated",
            "executed_steps",
        ):
            getattr(buffer, name)[: buffer.size] = state[name]
        buffer.rng.bit_generator.state = state["rng_state"]
        return buffer

    def save(self, path: str | Path) -> None:
        arrays = {
            "observation_pixels": self.observations["pixels"][: self.size],
            "observation_state": self.observations["state"][: self.size],
            "next_observation_pixels": self.next_observations["pixels"][: self.size],
            "next_observation_state": self.next_observations["state"][: self.size],
            "actions": self.actions[: self.size],
            "rewards": self.rewards[: self.size],
            "discounts": self.discounts[: self.size],
            "masks": self.masks[: self.size],
            "terminated": self.terminated[: self.size],
            "truncated": self.truncated[: self.size],
            "executed_steps": self.executed_steps[: self.size],
            "capacity": np.asarray(self.capacity, dtype=np.int64),
            "image_channels": np.asarray(self.image_channels, dtype=np.int64),
            "size": np.asarray(self.size, dtype=np.int64),
            "position": np.asarray(self.position, dtype=np.int64),
            "rng_state": np.asarray(json.dumps(self.rng.bit_generator.state)),
        }
        with Path(path).open("wb") as file:
            np.savez_compressed(file, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> ReplayBuffer:
        with np.load(Path(path), allow_pickle=False) as saved:
            image_channels = int(
                saved["image_channels"]
                if "image_channels" in saved
                else saved["observation_pixels"].shape[1]
            )
            buffer = cls(int(saved["capacity"]), image_channels=image_channels)
            buffer.size = int(saved["size"])
            buffer.position = int(saved["position"])
            buffer.observations["pixels"][: buffer.size] = saved["observation_pixels"]
            buffer.observations["state"][: buffer.size] = saved["observation_state"]
            buffer.next_observations["pixels"][: buffer.size] = saved[
                "next_observation_pixels"
            ]
            buffer.next_observations["state"][: buffer.size] = saved["next_observation_state"]
            for name in (
                "actions",
                "rewards",
                "discounts",
                "masks",
                "terminated",
                "truncated",
                "executed_steps",
            ):
                getattr(buffer, name)[: buffer.size] = saved[name]
            buffer.rng.bit_generator.state = json.loads(str(saved["rng_state"]))
        return buffer
