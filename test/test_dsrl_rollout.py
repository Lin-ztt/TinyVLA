from __future__ import annotations

import numpy as np
import torch

from tinyvla.dsrl.replay_buffer import ReplayBuffer
from tinyvla.dsrl.rollout import (
    RolloutConfig,
    expand_structured_noise,
    extract_sac_observation,
    rollout_episode,
)


class IdentityProcessor:
    def __call__(self, value):
        return value


class FakeEnvProcessor:
    def __call__(self, observation):
        robot_state = observation["observation.robot_state"]
        step = robot_state["eef"]["pos"][:, :1].float()
        return {
            "observation.images.image": observation["observation.images.image"],
            "observation.images.image2": observation["observation.images.image2"],
            "observation.state": step.repeat(1, 8),
            "task": observation["task"],
        }


class FakePolicy:
    def __init__(self) -> None:
        self.noises = []

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, observation, noise):
        self.noises.append(noise.clone())
        return torch.zeros(1, 50, 7, device=noise.device)


class FakeEnv:
    def __init__(self, success_after: int | None) -> None:
        self.steps = 0
        self.success_after = success_after

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
        return self._observation(), {}

    def call(self, name):
        assert name == "task_description"
        return ("test task",)

    def step(self, action):
        assert action.shape == (1, 7)
        self.steps += 1
        success = self.success_after is not None and self.steps >= self.success_after
        return (
            self._observation(),
            np.asarray([float(success)], dtype=np.float32),
            np.asarray([success]),
            np.asarray([False]),
            {"is_success": np.asarray([success])},
        )


def make_components(success_after: int | None):
    return (
        FakeEnv(success_after),
        FakePolicy(),
        FakeEnvProcessor(),
        IdentityProcessor(),
        IdentityProcessor(),
        IdentityProcessor(),
    )


def test_sac_observation_and_structured_noise() -> None:
    observation = {
        "observation.images.image": torch.rand(1, 3, 80, 96),
        "observation.state": torch.arange(8).reshape(1, 8),
    }
    sac_observation = extract_sac_observation(observation)
    assert sac_observation["pixels"].shape == (3, 64, 64)
    assert sac_observation["pixels"].dtype == np.uint8
    assert sac_observation["state"].shape == (8,)
    assert sac_observation["state"].dtype == np.float32

    noise = torch.linspace(-1.0, 1.0, 32).reshape(1, 32)
    expanded = expand_structured_noise(noise)
    assert expanded.shape == (1, 50, 32)
    assert torch.equal(expanded[:, 0], noise)
    assert torch.equal(expanded[:, -1], noise)


def test_sac_observation_tolerates_interpolation_roundoff() -> None:
    pixels = torch.zeros(1, 3, 256, 256)
    pixels[:, :, :128] = 1.0
    observation = {
        "observation.images.image": pixels,
        "observation.state": torch.zeros(1, 8),
    }

    sac_observation = extract_sac_observation(observation)

    assert sac_observation["pixels"].min() == 0
    assert sac_observation["pixels"].max() == 255


def test_dual_camera_observation_preserves_configured_channel_order() -> None:
    observation = {
        "observation.images.image": torch.full((1, 3, 72, 80), 0.25),
        "observation.images.image2": torch.full((1, 3, 90, 70), 0.75),
        "observation.state": torch.arange(8).reshape(1, 8),
    }

    sac_observation = extract_sac_observation(
        observation,
        ("observation.images.image", "observation.images.image2"),
    )

    assert sac_observation["pixels"].shape == (6, 64, 64)
    assert np.all(sac_observation["pixels"][:3] == 64)
    assert np.all(sac_observation["pixels"][3:] == 191)


def test_successful_rollout_writes_chunk_transitions() -> None:
    env, policy, env_processor, preprocessor, postprocessor, env_postprocessor = (
        make_components(success_after=3)
    )
    buffer = ReplayBuffer(capacity=4)
    result = rollout_episode(
        env,
        policy,
        env_processor,
        preprocessor,
        postprocessor,
        env_postprocessor,
        buffer,
        RolloutConfig(execute_horizon=2, max_episode_steps=5, discount=0.9, seed=7),
    )

    assert result["success"] and result["terminated"] and not result["truncated"]
    assert result["environment_steps"] == 3
    assert len(buffer) == 2
    assert buffer.executed_steps[:2].tolist() == [2, 1]
    assert np.allclose(buffer.discounts[:2], [0.9**2, 0.9])
    assert buffer.rewards[:2].tolist() == [-1.0, 0.0]
    assert buffer.masks[:2].tolist() == [1.0, 0.0]
    assert buffer.terminated[:2].tolist() == [False, True]
    assert buffer.observations["state"][:2, 0].tolist() == [0.0, 2.0]
    assert buffer.next_observations["state"][:2, 0].tolist() == [2.0, 3.0]
    assert all(noise.shape == (1, 50, 32) for noise in policy.noises)
    assert np.all(buffer.actions[:2] >= -1.0) and np.all(buffer.actions[:2] <= 1.0)


def test_time_limit_truncation_mask() -> None:
    components = make_components(success_after=None)
    buffer = ReplayBuffer(capacity=4)
    result = rollout_episode(
        *components,
        buffer,
        RolloutConfig(
            execute_horizon=2,
            max_episode_steps=3,
            discount=0.9,
            bootstrap_on_truncation=False,
        ),
    )

    assert not result["terminated"] and result["truncated"]
    assert buffer.executed_steps[:2].tolist() == [2, 1]
    assert buffer.truncated[:2].tolist() == [False, True]
    assert buffer.masks[:2].tolist() == [1.0, 0.0]


def test_rollout_uses_external_noise_selector() -> None:
    components = make_components(success_after=3)
    buffer = ReplayBuffer(capacity=4)
    selected = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    observed_states = []

    def select_noise(observation):
        observed_states.append(float(observation["state"][0]))
        return selected

    result = rollout_episode(
        *components,
        buffer,
        RolloutConfig(execute_horizon=2, max_episode_steps=5),
        noise_fn=select_noise,
        noise_source="actor_stochastic",
    )

    assert observed_states == [0.0, 2.0]
    assert np.array_equal(buffer.actions[:2], np.stack((selected, selected)))
    assert [transition["noise_source"] for transition in result["transitions"]] == [
        "actor_stochastic",
        "actor_stochastic",
    ]


def test_rollout_accepts_structured_gaussian_noise() -> None:
    components = make_components(success_after=1)
    buffer = ReplayBuffer(capacity=1)
    selected = np.linspace(-2.0, 2.0, 32, dtype=np.float32)
    result = rollout_episode(
        *components,
        buffer,
        RolloutConfig(execute_horizon=2, max_episode_steps=5),
        noise_fn=lambda _: selected,
        noise_source="structured_gaussian",
    )

    assert np.array_equal(buffer.actions[0], selected)
    assert result["transitions"][0]["noise_min"] == -2.0
    assert result["transitions"][0]["noise_max"] == 2.0


def test_evaluation_rollout_accepts_full_gaussian_noise_without_buffer() -> None:
    components = make_components(success_after=1)
    selected = np.linspace(-2.0, 2.0, 50 * 32, dtype=np.float32).reshape(50, 32)
    result = rollout_episode(
        *components,
        None,
        RolloutConfig(execute_horizon=2, max_episode_steps=5),
        noise_fn=lambda _: selected,
        noise_source="native",
    )

    policy = components[1]
    assert result["success"]
    assert len(policy.noises) == 1
    assert torch.equal(policy.noises[0], torch.from_numpy(selected).unsqueeze(0))
    assert result["transitions"][0]["noise_min"] == -2.0
    assert result["transitions"][0]["noise_max"] == 2.0


def test_full_noise_cannot_be_written_to_sac_buffer() -> None:
    components = make_components(success_after=1)
    buffer = ReplayBuffer(capacity=1)
    full_noise = np.zeros((50, 32), dtype=np.float32)

    try:
        rollout_episode(*components, buffer, noise_fn=lambda _: full_noise)
    except ValueError as error:
        assert "cannot be stored" in str(error)
    else:
        raise AssertionError("Expected full-noise replay validation to fail")
