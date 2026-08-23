"""Preallocated CPU replay buffer for EXPO action-chunk transitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class EXPOReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)
        self.observations = {
            "pixels": np.empty((capacity, 6, 64, 64), dtype=np.uint8),
            "state": np.empty((capacity, 8), dtype=np.float32),
        }
        self.next_observations = {
            "pixels": np.empty((capacity, 6, 64, 64), dtype=np.uint8),
            "state": np.empty((capacity, 8), dtype=np.float32),
        }
        self.executed_actions = np.empty((capacity, 8, 7), dtype=np.float32)
        self.base_actions = np.empty((capacity, 8, 7), dtype=np.float32)
        self.next_base_candidates = np.empty((capacity, 8, 8, 7), dtype=np.float32)
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

    @staticmethod
    def _pad_action(action: Any, executed_steps: int) -> np.ndarray:
        array = np.asarray(action, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 7 or not executed_steps <= array.shape[0] <= 8:
            raise ValueError(
                f"Expected action [steps, 7] with executed_steps <= steps <= 8, got {array.shape}"
            )
        padded = np.empty((8, 7), dtype=np.float32)
        padded[:executed_steps] = array[:executed_steps]
        padded[executed_steps:] = array[executed_steps - 1]
        return padded

    def add(
        self,
        observation: dict[str, Any],
        executed_action: Any,
        base_action: Any,
        reward: float,
        discount: float,
        mask: float,
        next_observation: dict[str, Any],
        next_base_candidates: Any | None,
        terminated: bool,
        truncated: bool,
        executed_steps: int,
    ) -> None:
        if not 1 <= executed_steps <= 8:
            raise ValueError("executed_steps must be in [1, 8]")
        pixels = self._array(observation["pixels"], (6, 64, 64), np.dtype(np.uint8))
        state = self._array(observation["state"], (8,), np.dtype(np.float32))
        next_pixels = self._array(
            next_observation["pixels"], (6, 64, 64), np.dtype(np.uint8)
        )
        next_state = self._array(next_observation["state"], (8,), np.dtype(np.float32))
        executed_action_array = self._pad_action(executed_action, executed_steps)
        base_action_array = self._pad_action(base_action, executed_steps)
        if next_base_candidates is None:
            if mask != 0:
                raise ValueError("Non-terminal transitions require next_base_candidates")
            candidate_array = np.zeros((8, 8, 7), dtype=np.float32)
        else:
            candidate_array = self._array(
                next_base_candidates, (8, 8, 7), np.dtype(np.float32)
            )

        index = self.position
        self.observations["pixels"][index] = pixels
        self.observations["state"][index] = state
        self.next_observations["pixels"][index] = next_pixels
        self.next_observations["state"][index] = next_state
        self.executed_actions[index] = executed_action_array
        self.base_actions[index] = base_action_array
        self.next_base_candidates[index] = candidate_array
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
            "executed_actions": tensor(self.executed_actions),
            "base_actions": tensor(self.base_actions),
            "next_base_candidates": tensor(self.next_base_candidates),
            "rewards": tensor(self.rewards),
            "discounts": tensor(self.discounts),
            "masks": tensor(self.masks),
            "terminated": tensor(self.terminated),
            "truncated": tensor(self.truncated),
            "executed_steps": tensor(self.executed_steps),
        }

    def _fields(self) -> dict[str, np.ndarray]:
        return {
            "observations.pixels": self.observations["pixels"],
            "observations.state": self.observations["state"],
            "next_observations.pixels": self.next_observations["pixels"],
            "next_observations.state": self.next_observations["state"],
            "executed_actions": self.executed_actions,
            "base_actions": self.base_actions,
            "next_base_candidates": self.next_base_candidates,
            "rewards": self.rewards,
            "discounts": self.discounts,
            "masks": self.masks,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "executed_steps": self.executed_steps,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "fields": {
                name: {"shape": list(array[: self.size].shape), "dtype": str(array.dtype)}
                for name, array in self._fields().items()
            },
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "rng_state": json.loads(json.dumps(self.rng.bit_generator.state)),
            "observation_pixels": self.observations["pixels"][: self.size].copy(),
            "observation_state": self.observations["state"][: self.size].copy(),
            "next_observation_pixels": self.next_observations["pixels"][: self.size].copy(),
            "next_observation_state": self.next_observations["state"][: self.size].copy(),
            "executed_actions": self.executed_actions[: self.size].copy(),
            "base_actions": self.base_actions[: self.size].copy(),
            "next_base_candidates": self.next_base_candidates[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "discounts": self.discounts[: self.size].copy(),
            "masks": self.masks[: self.size].copy(),
            "terminated": self.terminated[: self.size].copy(),
            "truncated": self.truncated[: self.size].copy(),
            "executed_steps": self.executed_steps[: self.size].copy(),
        }

    @classmethod
    def from_checkpoint_state(cls, state: dict[str, Any]) -> EXPOReplayBuffer:
        buffer = cls(int(state["capacity"]))
        buffer.size = int(state["size"])
        buffer.position = int(state["position"])
        if not 0 <= buffer.size <= buffer.capacity:
            raise ValueError("Invalid replay buffer size in checkpoint")
        if not 0 <= buffer.position < buffer.capacity:
            raise ValueError("Invalid replay buffer position in checkpoint")
        buffer.observations["pixels"][: buffer.size] = state["observation_pixels"]
        buffer.observations["state"][: buffer.size] = state["observation_state"]
        buffer.next_observations["pixels"][: buffer.size] = state["next_observation_pixels"]
        buffer.next_observations["state"][: buffer.size] = state["next_observation_state"]
        for name in (
            "executed_actions",
            "base_actions",
            "next_base_candidates",
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
        state = self.checkpoint_state()
        state["rng_state"] = np.asarray(json.dumps(state["rng_state"]))
        with Path(path).open("wb") as file:
            np.savez_compressed(file, **state)

    @classmethod
    def load(cls, path: str | Path) -> EXPOReplayBuffer:
        with np.load(Path(path), allow_pickle=False) as saved:
            state = {name: saved[name] for name in saved.files}
        state["capacity"] = int(state["capacity"])
        state["size"] = int(state["size"])
        state["position"] = int(state["position"])
        state["rng_state"] = json.loads(str(state["rng_state"]))
        return cls.from_checkpoint_state(state)
