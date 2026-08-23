from __future__ import annotations

import numpy as np
import pytest
import torch

from tinyvla.dsrl.replay_buffer import ReplayBuffer


def add_transition(buffer: ReplayBuffer, value: int) -> None:
    image_shape = (buffer.image_channels, 64, 64)
    observation = {
        "pixels": np.full(image_shape, value, dtype=np.uint8),
        "state": np.full(8, value, dtype=np.float32),
    }
    next_observation = {
        "pixels": np.full(image_shape, value + 1, dtype=np.uint8),
        "state": np.full(8, value + 1, dtype=np.float32),
    }
    buffer.add(
        observation=observation,
        action=np.full(32, value, dtype=np.float32),
        reward=float(-value),
        discount=0.999 ** (value + 1),
        mask=float(value % 2),
        next_observation=next_observation,
        terminated=value % 2 == 0,
        truncated=value % 3 == 0,
        executed_steps=value + 1,
    )


def assert_batches_equal(left: dict, right: dict) -> None:
    for key in ("observations", "next_observations"):
        for child in ("pixels", "state"):
            assert torch.equal(left[key][child], right[key][child])
    for key in (
        "actions",
        "rewards",
        "discounts",
        "masks",
        "terminated",
        "truncated",
        "executed_steps",
    ):
        assert torch.equal(left[key], right[key])


def test_shapes_dtypes_and_sac_batch_contract() -> None:
    buffer = ReplayBuffer(capacity=4, seed=5)
    add_transition(buffer, 0)

    assert buffer.observations["pixels"].shape == (4, 3, 64, 64)
    assert buffer.observations["pixels"].dtype == np.uint8
    assert buffer.observations["state"].shape == (4, 8)
    assert buffer.observations["state"].dtype == np.float32
    assert buffer.actions.shape == (4, 32)
    assert buffer.actions.dtype == np.float32
    assert buffer.terminated.dtype == np.bool_
    assert buffer.truncated.dtype == np.bool_
    assert buffer.executed_steps.dtype == np.int32

    batch = buffer.sample(3)
    assert batch["observations"]["pixels"].shape == (3, 3, 64, 64)
    assert batch["observations"]["pixels"].dtype == torch.uint8
    assert batch["observations"]["state"].shape == (3, 8)
    assert batch["actions"].shape == (3, 32)
    assert batch["rewards"].shape == (3,)
    assert batch["discounts"].shape == (3,)
    assert batch["masks"].shape == (3,)

    diagnostics = buffer.diagnostics()
    assert diagnostics["capacity"] == 4
    assert diagnostics["size"] == 1
    assert diagnostics["position"] == 1
    assert diagnostics["fields"]["observations.pixels"] == {
        "shape": [1, 3, 64, 64],
        "dtype": "uint8",
    }
    assert diagnostics["fields"]["actions"] == {
        "shape": [1, 32],
        "dtype": "float32",
    }


def test_ring_overwrites_oldest_entries() -> None:
    buffer = ReplayBuffer(capacity=3)
    for value in range(5):
        add_transition(buffer, value)

    assert len(buffer) == 3
    assert buffer.position == 2
    assert buffer.actions[:, 0].tolist() == [3.0, 4.0, 2.0]
    assert buffer.executed_steps.tolist() == [4, 5, 3]


def test_fixed_seed_sampling_is_reproducible() -> None:
    left = ReplayBuffer(capacity=8, seed=17)
    right = ReplayBuffer(capacity=8, seed=17)
    for value in range(8):
        add_transition(left, value)
        add_transition(right, value)

    assert_batches_equal(left.sample(32), right.sample(32))


def test_save_load_preserves_data_pointer_and_rng(tmp_path) -> None:
    original = ReplayBuffer(capacity=8, seed=23)
    for value in range(6):
        add_transition(original, value)
    original.sample(5)

    path = tmp_path / "replay_buffer.npz"
    original.save(path)
    restored = ReplayBuffer.load(path)

    assert restored.capacity == original.capacity
    assert len(restored) == len(original)
    assert restored.position == original.position
    for name in (
        "actions",
        "rewards",
        "discounts",
        "masks",
        "terminated",
        "truncated",
        "executed_steps",
    ):
        assert np.array_equal(
            getattr(restored, name)[: len(restored)], getattr(original, name)[: len(original)]
        )
    for key in ("pixels", "state"):
        assert np.array_equal(
            restored.observations[key][: len(restored)],
            original.observations[key][: len(original)],
        )
        assert np.array_equal(
            restored.next_observations[key][: len(restored)],
            original.next_observations[key][: len(original)],
        )
    assert_batches_equal(original.sample(16), restored.sample(16))


def test_checkpoint_state_preserves_next_sample() -> None:
    original = ReplayBuffer(capacity=8, seed=31)
    for value in range(6):
        add_transition(original, value)
    original.sample(7)

    restored = ReplayBuffer.from_checkpoint_state(original.checkpoint_state())

    assert restored.capacity == original.capacity
    assert restored.size == original.size
    assert restored.position == original.position
    assert_batches_equal(original.sample(32), restored.sample(32))

    add_transition(original, 9)
    add_transition(restored, 9)
    assert original.position == restored.position
    assert_batches_equal(original.sample(16), restored.sample(16))


def test_dual_camera_save_and_checkpoint_round_trip(tmp_path) -> None:
    original = ReplayBuffer(capacity=4, seed=41, image_channels=6)
    for value in range(3):
        add_transition(original, value)

    path = tmp_path / "dual_camera_buffer.npz"
    original.save(path)
    restored_file = ReplayBuffer.load(path)
    restored_state = ReplayBuffer.from_checkpoint_state(original.checkpoint_state())

    for restored in (restored_file, restored_state):
        assert restored.image_channels == 6
        assert restored.observations["pixels"].shape == (4, 6, 64, 64)
        assert np.array_equal(
            restored.observations["pixels"][:3], original.observations["pixels"][:3]
        )


def test_rejects_invalid_shape_and_empty_sampling() -> None:
    buffer = ReplayBuffer(capacity=2)
    with pytest.raises(ValueError, match="empty"):
        buffer.sample(1)
    with pytest.raises(ValueError, match="Expected shape"):
        buffer.add(
            observation={"pixels": np.zeros((64, 64, 3)), "state": np.zeros(8)},
            action=np.zeros(32),
            reward=-1.0,
            discount=0.999,
            mask=1.0,
            next_observation={"pixels": np.zeros((3, 64, 64)), "state": np.zeros(8)},
            terminated=False,
            truncated=False,
            executed_steps=1,
        )
