from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.train_smolvla_expo import (
    _assert_batches_equal,
    learner_fingerprints,
    restore_training_state,
    save_training_state,
    training_init_state_id,
    validate_checkpoint_format,
)
from tinyvla.expo import EXPOConfig, EXPOLearner, EXPOReplayBuffer


def test_training_init_states_shuffle_each_complete_cycle() -> None:
    environment = {
        "train_init_state_start": 0,
        "train_init_state_count": 50,
        "shuffle_init_states": True,
    }
    first = [training_init_state_id(index, 3000, environment) for index in range(50)]
    second = [training_init_state_id(index, 3000, environment) for index in range(50, 100)]

    assert sorted(first) == sorted(second) == list(range(50))
    assert first != second
    assert first == [training_init_state_id(index, 3000, environment) for index in range(50)]


def test_training_checkpoint_restores_learner_buffer_and_validation(tmp_path) -> None:
    learner = EXPOLearner(
        EXPOConfig(crop_padding=0, rotation_degrees=0, color_jitter=0), device="cpu"
    )
    buffer = EXPOReplayBuffer(capacity=4, seed=9)
    observation = {
        "pixels": np.zeros((6, 64, 64), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
    }
    buffer.add(
        observation=observation,
        executed_action=np.zeros((8, 7), dtype=np.float32),
        base_action=np.ones((8, 7), dtype=np.float32),
        reward=0.0,
        discount=0.99**8,
        mask=1.0,
        next_observation=observation,
        next_base_candidates=np.zeros((8, 8, 7), dtype=np.float32),
        terminated=False,
        truncated=False,
        executed_steps=8,
    )
    counters = {"environment_steps": 8, "chunk_transitions": 1, "gradient_steps": 0}
    config = {"checkpoint": "models/sft/libero_40tasks"}
    fingerprints = learner_fingerprints(learner)

    save_training_state(
        tmp_path,
        learner,
        buffer,
        counters,
        1,
        [{"success": False}],
        [],
        config,
        "policy-hash",
        fingerprints,
    )
    checkpoint = torch.load(tmp_path / "checkpoint_latest.pt", weights_only=False)
    assert checkpoint["format_version"] == 3
    assert checkpoint["entropy_coordinate"] == "residual_unscaled"
    restored = restore_training_state(checkpoint, torch.device("cpu"))
    restored_learner, restored_buffer = restored[:2]

    assert restored[2] == counters and restored[3] == 1
    assert restored[4] == [{"success": False}] and restored[5] == []
    assert restored[6] == fingerprints
    assert len(restored_buffer) == 1
    assert learner_fingerprints(restored_learner) == learner_fingerprints(learner)
    expected_buffer = EXPOReplayBuffer.from_checkpoint_state(checkpoint["replay_buffer"])
    _assert_batches_equal(expected_buffer.sample(4), restored_buffer.sample(4))

    with pytest.raises(ValueError, match="checkpoint format"):
        validate_checkpoint_format({"format_version": 1})
    with pytest.raises(ValueError, match="entropy coordinate"):
        validate_checkpoint_format({"format_version": 3})
