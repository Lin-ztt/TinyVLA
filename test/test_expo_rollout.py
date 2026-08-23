from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tinyvla.expo.candidates import CandidateSelection
from tinyvla.expo.replay_buffer import EXPOReplayBuffer
from tinyvla.expo.rollout import (
    EXPORolloutConfig,
    generate_base_proposals,
    rollout_episode,
    write_selected_action_chunk,
)


class IdentityProcessor:
    def __call__(self, value):
        return value


class FakeEnvProcessor:
    def __call__(self, observation):
        step = observation["observation.robot_state"]["eef"]["pos"][:, :1].float()
        return {
            "observation.images.image": observation["observation.images.image"],
            "observation.images.image2": observation["observation.images.image2"],
            "observation.state": step.repeat(1, 8),
            "task": observation["task"],
        }


class FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls: list[float] = []

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, observation, noise):
        step = float(observation["observation.state"][0, 0])
        self.calls.append(step)
        candidate = torch.arange(8, device=noise.device, dtype=torch.float32).reshape(8, 1, 1)
        return (step + candidate).expand(8, 50, 7).clone()


class FakeLearner:
    def select_action(self, pixels, state, base_candidates, deterministic_actor=False):
        assert pixels.shape == (1, 6, 64, 64)
        assert state.shape == (1, 8)
        residuals = torch.ones_like(base_candidates)
        selected_action = base_candidates[:, 3] + 0.1 * residuals[:, 3]
        return (
            CandidateSelection(
                action=selected_action,
                index=torch.tensor([11]),
                is_edited=torch.tensor([True]),
                score=torch.tensor([2.5]),
            ),
            residuals,
        )


class FakeEnv:
    def __init__(self, success_after: int | None) -> None:
        self.steps = 0
        self.success_after = success_after
        self.actions: list[np.ndarray] = []

    def _observation(self):
        image = np.full((1, 8, 8, 3), self.steps, dtype=np.uint8)
        scalar = np.full((1, 1), self.steps, dtype=np.float64)
        return {
            "pixels": {"image": image, "image2": image.copy()},
            "robot_state": {
                "eef": {
                    "pos": np.repeat(scalar, 3, axis=1),
                    "quat": np.repeat(scalar, 4, axis=1),
                    "mat": np.repeat(scalar[:, :, None], 9, axis=1).reshape(1, 3, 3),
                },
                "gripper": {
                    "qpos": np.repeat(scalar, 2, axis=1),
                    "qvel": np.repeat(scalar, 2, axis=1),
                },
                "joints": {
                    "pos": np.repeat(scalar, 7, axis=1),
                    "vel": np.repeat(scalar, 7, axis=1),
                },
            },
        }

    def reset(self, seed):
        self.steps = 0
        self.actions.clear()
        return self._observation(), {}

    def call(self, name):
        assert name == "task_description"
        return ("test task",)

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        success = self.success_after is not None and self.steps >= self.success_after
        return (
            self._observation(),
            np.asarray([float(success)], dtype=np.float32),
            np.asarray([success]),
            np.asarray([False]),
            {"is_success": np.asarray([success])},
        )


def test_generate_base_proposals_freezes_policy_and_preserves_full_chunks() -> None:
    policy = FakePolicy()
    observation = {"observation.state": torch.zeros(1, 8)}
    noise = torch.randn(8, 50, 32)

    proposals = generate_base_proposals(policy, IdentityProcessor(), observation, noise=noise)

    assert proposals.full_chunks.shape == (8, 50, 7)
    assert proposals.candidates.shape == (1, 8, 8, 7)
    assert torch.equal(proposals.candidates[0], proposals.full_chunks[:, :8])
    assert not policy.training
    assert not policy.weight.requires_grad
    assert len(torch.unique(proposals.candidates[:, :, 0, 0])) == 8


def test_write_selected_action_chunk_changes_only_corresponding_first_eight_steps() -> None:
    full_chunks = torch.arange(8 * 50 * 7, dtype=torch.float32).reshape(8, 50, 7)
    original = full_chunks.clone()
    selected_action = torch.full((1, 8, 7), -5.0)
    selection = CandidateSelection(
        action=selected_action,
        index=torch.tensor([11]),
        is_edited=torch.tensor([True]),
        score=torch.tensor([1.0]),
    )

    output = write_selected_action_chunk(full_chunks, selection)

    assert output.shape == (1, 50, 7)
    assert torch.equal(output[:, :8], selected_action)
    assert torch.equal(output[:, 8:], original[3:4, 8:])
    assert torch.equal(full_chunks, original)


def test_rollout_reuses_next_proposals_to_complete_pending_transition() -> None:
    env = FakeEnv(success_after=3)
    policy = FakePolicy()
    buffer = EXPOReplayBuffer(capacity=4)
    identity = IdentityProcessor()

    result = rollout_episode(
        env,
        policy,
        FakeLearner(),
        FakeEnvProcessor(),
        identity,
        identity,
        identity,
        buffer,
        EXPORolloutConfig(max_episode_steps=20, discount=0.9, seed=7),
    )

    assert result["success"] and result["environment_steps"] == 3
    assert len(buffer) == 1
    assert buffer.executed_steps[0] == 3
    assert buffer.rewards[0] == 1.0 and buffer.masks[0] == 0.0
    assert np.all(buffer.executed_actions[0] == 3.1)
    assert np.all(buffer.base_actions[0] == 3.0)
    assert policy.calls == [0.0]
    assert len(env.actions) == 3 and np.allclose(env.actions, 3.1)
    assert result["transitions"][0]["selected_candidate"] == 11
    assert result["transitions"][0]["selected_edited"]


def test_warmup_executes_only_random_base_candidates() -> None:
    env = FakeEnv(success_after=3)
    policy = FakePolicy()
    buffer = EXPOReplayBuffer(capacity=4)
    identity = IdentityProcessor()

    result = rollout_episode(
        env,
        policy,
        FakeLearner(),
        FakeEnvProcessor(),
        identity,
        identity,
        identity,
        buffer,
        EXPORolloutConfig(
            max_episode_steps=20,
            seed=7,
            warmup_base_only_until=4,
        ),
    )

    transition = result["transitions"][0]
    assert transition["selection_mode"] == "warmup_base"
    assert not transition["selected_edited"]
    assert 0 <= transition["selected_candidate"] < 8
    assert np.array_equal(buffer.executed_actions[0], buffer.base_actions[0])


def test_training_base_exploration_executes_a_base_candidate() -> None:
    env = FakeEnv(success_after=3)
    policy = FakePolicy()
    buffer = EXPOReplayBuffer(capacity=4)
    identity = IdentityProcessor()

    result = rollout_episode(
        env,
        policy,
        FakeLearner(),
        FakeEnvProcessor(),
        identity,
        identity,
        identity,
        buffer,
        EXPORolloutConfig(
            max_episode_steps=20,
            seed=7,
            base_exploration_prob=1.0,
        ),
    )

    transition = result["transitions"][0]
    assert transition["selection_mode"] == "base_exploration"
    assert not transition["selected_edited"]
    assert 0 <= transition["selected_candidate"] < 8
    assert np.array_equal(buffer.executed_actions[0], buffer.base_actions[0])


def test_nonterminal_pending_transition_uses_candidates_from_exact_next_state() -> None:
    env = FakeEnv(success_after=None)
    policy = FakePolicy()
    buffer = EXPOReplayBuffer(capacity=4)
    identity = IdentityProcessor()

    result = rollout_episode(
        env,
        policy,
        FakeLearner(),
        FakeEnvProcessor(),
        identity,
        identity,
        identity,
        buffer,
        EXPORolloutConfig(max_episode_steps=10, discount=0.9, seed=7),
    )

    assert result["truncated"] and len(buffer) == 2
    assert policy.calls == [0.0, 8.0, 10.0]
    assert np.all(buffer.next_observations["state"][0] == 8.0)
    assert np.all(buffer.next_base_candidates[0, :, 0, 0] == np.arange(8) + 8.0)
    assert buffer.executed_steps[:2].tolist() == [8, 2]
    assert np.allclose(buffer.discounts[:2], [0.9**8, 0.9**2])
    assert buffer.masks[:2].tolist() == [1.0, 1.0]
