from __future__ import annotations

import numpy as np
import pytest
import torch

from tinyvla.expo.replay_buffer import EXPOReplayBuffer


def add_transition(buffer: EXPOReplayBuffer, value: int, executed_steps: int = 8) -> None:
    observation = {
        "pixels": np.full((6, 64, 64), value, dtype=np.uint8),
        "state": np.full(8, value, dtype=np.float32),
    }
    next_observation = {
        "pixels": np.full((6, 64, 64), value + 1, dtype=np.uint8),
        "state": np.full(8, value + 1, dtype=np.float32),
    }
    buffer.add(
        observation=observation,
        executed_action=np.repeat(np.arange(8, dtype=np.float32)[:, None], 7, axis=1)
        + value,
        base_action=np.repeat(np.arange(8, dtype=np.float32)[:, None], 7, axis=1)
        + value
        + 100,
        reward=float(value % 2),
        discount=0.99**executed_steps,
        mask=1.0,
        next_observation=next_observation,
        next_base_candidates=np.full((8, 8, 7), value + 2, dtype=np.float32),
        terminated=False,
        truncated=False,
        executed_steps=executed_steps,
    )


def assert_batches_equal(left: dict, right: dict) -> None:
    for parent in ("observations", "next_observations"):
        for child in ("pixels", "state"):
            assert torch.equal(left[parent][child], right[parent][child])
    for key in (
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
        assert torch.equal(left[key], right[key])


def test_shapes_dtypes_and_short_action_padding() -> None:
    buffer = EXPOReplayBuffer(capacity=4, seed=5)
    add_transition(buffer, 0, executed_steps=5)
    assert buffer.observations["pixels"].shape == (4, 6, 64, 64)
    assert buffer.observations["pixels"].dtype == np.uint8
    assert buffer.executed_actions.shape == buffer.base_actions.shape == (4, 8, 7)
    assert buffer.next_base_candidates.shape == (4, 8, 8, 7)
    assert buffer.executed_actions.dtype == buffer.base_actions.dtype == np.float32
    assert np.array_equal(
        buffer.executed_actions[0, 5:], buffer.executed_actions[0, 4:5].repeat(3, axis=0)
    )
    assert np.array_equal(
        buffer.base_actions[0, 5:], buffer.base_actions[0, 4:5].repeat(3, axis=0)
    )
    assert buffer.executed_steps[0] == 5
    assert buffer.next_observations["state"][0, 0] == 1
    assert buffer.next_base_candidates[0, 0, 0, 0] == 2

    batch = buffer.sample(3)
    assert batch["executed_actions"].shape == batch["base_actions"].shape == (3, 8, 7)
    assert batch["next_base_candidates"].shape == (3, 8, 8, 7)
    diagnostics = buffer.diagnostics()
    assert diagnostics["fields"]["next_base_candidates"] == {
        "shape": [1, 8, 8, 7],
        "dtype": "float32",
    }


def test_ring_seeded_sampling_and_save_restore(tmp_path) -> None:
    original = EXPOReplayBuffer(capacity=3, seed=17)
    for value in range(5):
        add_transition(original, value)
    assert len(original) == 3 and original.position == 2
    assert original.observations["state"][:, 0].tolist() == [3.0, 4.0, 2.0]
    original.sample(5)

    checkpoint = original.checkpoint_state()
    state_restored = EXPOReplayBuffer.from_checkpoint_state(checkpoint)
    path = tmp_path / "expo_buffer.npz"
    original.save(path)
    file_restored = EXPOReplayBuffer.load(path)
    assert_batches_equal(state_restored.sample(16), file_restored.sample(16))

    state_restored = EXPOReplayBuffer.from_checkpoint_state(checkpoint)
    assert_batches_equal(original.sample(16), state_restored.sample(16))


def test_terminal_transition_allows_missing_next_candidates() -> None:
    buffer = EXPOReplayBuffer(capacity=1)
    observation = {
        "pixels": np.zeros((6, 64, 64), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
    }
    buffer.add(
        observation=observation,
        executed_action=np.zeros((1, 7), dtype=np.float32),
        base_action=np.ones((1, 7), dtype=np.float32),
        reward=1.0,
        discount=0.99,
        mask=0.0,
        next_observation=observation,
        next_base_candidates=None,
        terminated=True,
        truncated=False,
        executed_steps=1,
    )
    assert np.count_nonzero(buffer.next_base_candidates[0]) == 0
    with pytest.raises(ValueError, match="next_base_candidates"):
        buffer.add(
            observation=observation,
            executed_action=np.zeros((1, 7), dtype=np.float32),
            base_action=np.ones((1, 7), dtype=np.float32),
            reward=0.0,
            discount=0.99,
            mask=1.0,
            next_observation=observation,
            next_base_candidates=None,
            terminated=False,
            truncated=False,
            executed_steps=1,
        )
