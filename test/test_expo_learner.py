from __future__ import annotations

import math

import torch

from tinyvla.expo.learner import EXPOConfig, EXPOLearner


def make_batch(batch_size: int, device: torch.device) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(11)
    return {
        "observations": {
            "pixels": torch.randint(
                0, 256, (batch_size, 6, 64, 64), dtype=torch.uint8, generator=generator
            ).to(device),
            "state": torch.randn(batch_size, 8, generator=generator).to(device),
        },
        "next_observations": {
            "pixels": torch.randint(
                0, 256, (batch_size, 6, 64, 64), dtype=torch.uint8, generator=generator
            ).to(device),
            "state": torch.randn(batch_size, 8, generator=generator).to(device),
        },
        "executed_actions": torch.empty(batch_size, 8, 7)
        .uniform_(-1, 1, generator=generator)
        .to(device),
        "base_actions": torch.empty(batch_size, 8, 7)
        .uniform_(-1, 1, generator=generator)
        .to(device),
        "next_base_candidates": torch.empty(batch_size, 8, 8, 7)
        .uniform_(-1, 1, generator=generator)
        .to(device),
        "rewards": torch.randint(0, 2, (batch_size,), generator=generator).float().to(device),
        "discounts": torch.full((batch_size,), 0.99**8, device=device),
        "masks": torch.randint(0, 2, (batch_size,), generator=generator).float().to(device),
    }


def parameters_changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, module.parameters(), strict=True)
    )


def test_actor_critic_shapes_and_two_q_sampling() -> None:
    torch.manual_seed(0)
    learner = EXPOLearner(EXPOConfig(crop_padding=0, rotation_degrees=0, color_jitter=0))
    pixels = torch.randint(0, 256, (2, 6, 64, 64), dtype=torch.uint8)
    state = torch.randn(2, 8)
    base = torch.randn(2, 56)
    embedding = learner.image_encoder(pixels)
    mean, log_std = learner.actor(embedding, state, base)
    residual, log_probability = learner.actor.sample(embedding, state, base)
    qs = learner.critic(embedding, state, base)

    assert embedding.shape == (2, 512)
    assert mean.shape == log_std.shape == residual.shape == (2, 56)
    assert log_probability.shape == (2,)
    assert qs.shape == (2, 10)
    assert torch.isfinite(residual).all() and torch.isfinite(log_probability).all()
    assert torch.all(residual >= -1) and torch.all(residual <= 1)
    sampled = learner._sample_q_indices()
    assert sampled.shape == (2,) and sampled.unique().numel() == 2


def test_dual_camera_training_augmentation() -> None:
    torch.manual_seed(1)
    learner = EXPOLearner()
    pixels = torch.randint(0, 256, (2, 6, 64, 64), dtype=torch.uint8)
    augmented = learner._augment(pixels)
    assert augmented.shape == pixels.shape
    assert augmented.dtype == torch.float32
    assert torch.isfinite(augmented).all()
    assert augmented.min() >= 0 and augmented.max() <= 1


def test_temperature_uses_unscaled_residual_entropy() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = EXPOConfig(crop_padding=0, rotation_degrees=0, color_jitter=0)
    learner = EXPOLearner(config, device=device)

    metrics = learner.update(make_batch(2, device))

    assert metrics["entropy"] > config.target_entropy
    assert metrics["temperature"] < config.initial_temperature


def test_critic_warmup_freezes_actor_and_temperature() -> None:
    torch.manual_seed(12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner = EXPOLearner(
        EXPOConfig(crop_padding=0, rotation_degrees=0, color_jitter=0), device=device
    )
    actor_before = [parameter.detach().clone() for parameter in learner.actor.parameters()]
    temperature_before = learner.temperature.detach().clone()

    metrics = learner.update(make_batch(2, device), update_actor=False)

    assert metrics["actor_updated"] == 0.0
    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(actor_before, learner.actor.parameters(), strict=True)
    )
    assert torch.equal(temperature_before, learner.temperature.detach())


def test_actor_edits_executed_action_that_critic_fits() -> None:
    torch.manual_seed(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner = EXPOLearner(
        EXPOConfig(crop_padding=0, rotation_degrees=0, color_jitter=0), device=device
    )
    batch = make_batch(2, device)
    actor_inputs: list[torch.Tensor] = []
    handle = learner.actor.register_forward_pre_hook(
        lambda _module, args: actor_inputs.append(args[2].detach().cpu())
    )

    learner.eval()
    learner.update(batch)
    handle.remove()

    assert learner.training and learner.image_encoder.training
    assert torch.equal(actor_inputs[-1], batch["executed_actions"].reshape(-1, 56).cpu())
    assert not torch.equal(batch["base_actions"], batch["executed_actions"])


def test_select_action_compares_eight_base_and_eight_edited_candidates() -> None:
    torch.manual_seed(2)
    learner = EXPOLearner(EXPOConfig(crop_padding=0, rotation_degrees=0, color_jitter=0))
    pixels = torch.randint(0, 256, (1, 6, 64, 64), dtype=torch.uint8)
    state = torch.randn(1, 8)
    base_candidates = torch.randn(1, 8, 8, 7)

    selection, residuals = learner.select_action(
        pixels, state, base_candidates, deterministic_actor=True
    )

    assert selection.action.shape == (1, 8, 7)
    assert selection.index.shape == selection.is_edited.shape == selection.score.shape == (1,)
    assert 0 <= selection.index.item() < 16
    assert residuals.shape == (1, 8, 8, 7)
    base_index = int(selection.index.item()) % 8
    expected = base_candidates[:, base_index]
    if selection.is_edited.item():
        expected = expected + learner.config.edit_scale * residuals[:, base_index]
    assert torch.allclose(selection.action, expected)


def test_select_action_uses_eval_mode_for_batchnorm_and_restores_mode() -> None:
    torch.manual_seed(9)
    learner = EXPOLearner(EXPOConfig(crop_padding=0, rotation_degrees=0, color_jitter=0))
    pixels = torch.randint(0, 256, (1, 6, 64, 64), dtype=torch.uint8)
    state = torch.randn(1, 8)
    base_candidates = torch.randn(1, 8, 8, 7)

    learner.train()
    torch.manual_seed(10)
    train_selection, train_residuals = learner.select_action(
        pixels, state, base_candidates, deterministic_actor=True
    )
    assert learner.training and learner.image_encoder.training

    learner.eval()
    torch.manual_seed(10)
    eval_selection, eval_residuals = learner.select_action(
        pixels, state, base_candidates, deterministic_actor=True
    )
    assert not learner.training and not learner.image_encoder.training

    assert torch.equal(train_selection.index, eval_selection.index)
    assert torch.equal(train_selection.score, eval_selection.score)
    assert torch.equal(train_selection.action, eval_selection.action)
    assert torch.equal(train_residuals, eval_residuals)


def test_updates_target_softly_without_entropy_backup_and_restores(tmp_path) -> None:
    torch.manual_seed(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = EXPOConfig(crop_padding=0, rotation_degrees=0, color_jitter=0)
    learner = EXPOLearner(config, device=device)
    batch = make_batch(2, device)
    actor_before = [parameter.detach().clone() for parameter in learner.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in learner.critic.parameters()]
    encoder_before = [parameter.detach().clone() for parameter in learner.image_encoder.parameters()]
    target_before = [parameter.detach().clone() for parameter in learner.target_critic.parameters()]
    temperature_before = learner.log_temperature.detach().clone()

    last_metrics = learner.update(batch)
    for old_target, critic, target in zip(
        target_before, learner.critic.parameters(), learner.target_critic.parameters(), strict=True
    ):
        expected_target = old_target.lerp(critic.detach(), learner.config.tau)
        assert torch.allclose(target, expected_target, rtol=1e-6, atol=1e-7)

    for _ in range(99):
        last_metrics = learner.update(batch)
        assert all(math.isfinite(value) for value in last_metrics.values())

    assert last_metrics is not None and learner.update_steps.item() == 100
    assert parameters_changed(actor_before, learner.actor)
    assert parameters_changed(critic_before, learner.critic)
    assert parameters_changed(encoder_before, learner.image_encoder)
    assert not torch.equal(temperature_before, learner.log_temperature.detach())
    for critic, target in zip(
        learner.critic.parameters(), learner.target_critic.parameters(), strict=True
    ):
        assert torch.isfinite(target).all() and torch.isfinite(critic).all()

    with torch.no_grad():
        next_pixels = batch["next_observations"]["pixels"].float().div(255)
        target_q, next_q, _, _ = learner._critic_target(
            next_pixels,
            batch["next_observations"]["state"],
            batch["next_base_candidates"],
            batch["rewards"],
            batch["discounts"],
            batch["masks"],
        )
        expected = batch["rewards"] + batch["discounts"] * batch["masks"] * next_q
        assert torch.equal(target_q, expected)

    learner.eval()
    fixed_pixels = batch["observations"]["pixels"]
    fixed_state = batch["observations"]["state"]
    fixed_base = batch["base_actions"]
    expected_residual = learner.act_residual(
        fixed_pixels, fixed_state, fixed_base, deterministic=True
    ).cpu()
    path = tmp_path / "expo.pt"
    learner.save_checkpoint(path, metadata={"updates": 100})
    restored, metadata = EXPOLearner.from_checkpoint(path, device=device)
    restored.eval()
    actual_residual = restored.act_residual(
        fixed_pixels, fixed_state, fixed_base, deterministic=True
    ).cpu()
    assert metadata == {"updates": 100}
    assert restored.config == learner.config
    assert restored.update_steps.item() == 100
    assert torch.equal(expected_residual, actual_residual)
