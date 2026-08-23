from __future__ import annotations

import math

import pytest
import torch

from tinyvla.dsrl.sac import Actor, Critic, PixelEncoder, SAC, SACConfig, _color_jitter


def make_batch(batch_size: int, device: torch.device, image_channels: int = 3) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(7)
    pixels = torch.randint(
        0, 256, (batch_size, image_channels, 64, 64), dtype=torch.uint8, generator=generator
    )
    next_pixels = torch.randint(
        0,
        256,
        (batch_size, image_channels, 64, 64),
        dtype=torch.uint8,
        generator=generator,
    )
    state = torch.randn(batch_size, 8, generator=generator)
    next_state = torch.randn(batch_size, 8, generator=generator)
    return {
        "observations": {"pixels": pixels.to(device), "state": state.to(device)},
        "next_observations": {
            "pixels": next_pixels.to(device),
            "state": next_state.to(device),
        },
        "actions": torch.empty(batch_size, 32).uniform_(-1, 1, generator=generator).to(device),
        "rewards": (
            torch.randint(0, 2, (batch_size,), generator=generator).float().sub_(1).to(device)
        ),
        "discounts": torch.full((batch_size,), 0.999**20, device=device),
        "masks": torch.randint(0, 2, (batch_size,), generator=generator).float().to(device),
    }


def parameters_changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    pairs = zip(before, module.parameters(), strict=True)
    return any(not torch.equal(old, new.detach()) for old, new in pairs)


def all_finite(module: torch.nn.Module) -> bool:
    return all(torch.isfinite(parameter).all() for parameter in module.parameters())


def test_network_shapes_ranges_and_independence() -> None:
    torch.manual_seed(0)
    config = SACConfig()
    pixels = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8)
    state = torch.randn(4, 8)

    encoder = PixelEncoder()
    actor = Actor(config)
    critic = Critic(config)
    assert encoder(pixels).shape == (4, 50)

    action, log_probability = actor.sample(pixels, state)
    assert action.shape == (4, 32)
    assert log_probability.shape == (4,)
    assert torch.isfinite(action).all() and torch.isfinite(log_probability).all()
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    assert actor.deterministic(pixels, state).shape == (4, 32)

    qs = critic(pixels, state, action)
    assert qs.shape == (4, 10)
    assert torch.isfinite(qs).all()
    assert actor.encoder is not critic.encoder
    assert critic.q_heads[0] is not critic.q_heads[1]
    assert critic.q_heads[0][0].weight.data_ptr() != critic.q_heads[1][0].weight.data_ptr()


def test_target_has_no_entropy_backup() -> None:
    torch.manual_seed(1)
    learner = SAC()
    batch = make_batch(3, learner.device)
    target, next_q, next_log_probability = learner._critic_target(
        batch["next_observations"]["pixels"],
        batch["next_observations"]["state"],
        batch["rewards"],
        batch["discounts"],
        batch["masks"],
    )
    expected = batch["rewards"] + batch["discounts"] * batch["masks"] * next_q
    assert torch.equal(target, expected)
    assert torch.isfinite(next_log_probability).all()


def test_target_critic_uses_soft_update() -> None:
    torch.manual_seed(3)
    learner = SAC()
    batch = make_batch(2, learner.device)
    critic_before = [parameter.detach().clone() for parameter in learner.critic.parameters()]
    target_before = [parameter.detach().clone() for parameter in learner.target_critic.parameters()]
    assert all(
        torch.equal(critic, target)
        for critic, target in zip(critic_before, target_before, strict=True)
    )

    learner.update(batch)
    for old_target, critic, target in zip(
        target_before, learner.critic.parameters(), learner.target_critic.parameters(), strict=True
    ):
        expected = old_target.lerp(critic.detach(), learner.config.tau)
        assert torch.allclose(target, expected, rtol=1e-6, atol=1e-7)


def test_one_hundred_updates_and_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner = SAC(device=device)
    batch = make_batch(32, device)
    actor_before = [parameter.detach().clone() for parameter in learner.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in learner.critic.parameters()]
    temperature_before = learner.log_temperature.detach().clone()

    last_metrics = None
    for _ in range(100):
        last_metrics = learner.update(batch)
        assert all(math.isfinite(value) for value in last_metrics.values())

    assert last_metrics is not None
    assert learner.update_steps.item() == 100
    assert parameters_changed(actor_before, learner.actor)
    assert parameters_changed(critic_before, learner.critic)
    assert not torch.equal(temperature_before, learner.log_temperature.detach())
    assert all_finite(learner.actor) and all_finite(learner.critic)
    assert torch.isfinite(learner.log_temperature)
    for parameter in learner.actor.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()
    for parameter in learner.critic.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()

    fixed_pixels = batch["observations"]["pixels"][:2]
    fixed_state = batch["observations"]["state"][:2]
    expected_action = learner.act(fixed_pixels, fixed_state, deterministic=True).cpu()
    checkpoint_path = tmp_path / "sac.pt"
    learner.save_checkpoint(checkpoint_path, metadata={"environment_steps": 2000})
    restored, metadata = SAC.from_checkpoint(checkpoint_path, device=device)
    restored_action = restored.act(fixed_pixels, fixed_state, deterministic=True).cpu()
    state_restored, state_metadata = SAC.from_checkpoint_state(
        learner.checkpoint_state({"environment_steps": 2000}), device=device
    )
    state_restored_action = state_restored.act(
        fixed_pixels, fixed_state, deterministic=True
    ).cpu()

    assert metadata == {"environment_steps": 2000}
    assert state_metadata == metadata
    assert restored.update_steps.item() == 100
    assert torch.equal(expected_action, restored_action)
    assert torch.equal(expected_action, state_restored_action)
    assert restored.config == learner.config


def test_dual_camera_update_and_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(5)
    config = SACConfig(image_shape=(6, 64, 64), random_crop_padding=4)
    learner = SAC(config)
    batch = make_batch(2, learner.device, image_channels=6)

    metrics = learner.update(batch)
    expected = learner.act(
        batch["observations"]["pixels"], batch["observations"]["state"]
    )
    path = tmp_path / "dual_camera_sac.pt"
    learner.save_checkpoint(path)
    restored, _ = SAC.from_checkpoint(path)

    assert all(math.isfinite(value) for value in metrics.values())
    assert restored.config.image_shape == (6, 64, 64)
    assert torch.equal(
        expected,
        restored.act(batch["observations"]["pixels"], batch["observations"]["state"]),
    )


def test_color_jitter_supports_single_and_dual_camera_updates() -> None:
    torch.manual_seed(0)
    pixels = torch.randint(0, 256, (4, 6, 64, 64), dtype=torch.uint8)
    augmented = _color_jitter(pixels)
    assert augmented.shape == pixels.shape and augmented.dtype == torch.uint8

    for channels in (3, 6):
        learner = SAC(SACConfig(image_shape=(channels, 64, 64), color_jitter=True))
        metrics = learner.update(make_batch(2, learner.device, channels))
        assert all(math.isfinite(value) for value in metrics.values())


def test_first_version_rejects_changed_dsrl_semantics() -> None:
    with pytest.raises(ValueError, match="critic_reduction"):
        SACConfig(critic_reduction="min")
    with pytest.raises(ValueError, match="backup_entropy"):
        SACConfig(backup_entropy=True)
