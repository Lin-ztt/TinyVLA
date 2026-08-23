"""PyTorch EXPO-style residual editor and REDQ learner."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import resnet50

from .candidates import CandidateSelection, build_action_candidates, reduce_q_values, select_top_q


@dataclass(frozen=True)
class EXPOConfig:
    image_shape: tuple[int, int, int] = (6, 64, 64)
    state_dim: int = 8
    chunk_length: int = 8
    action_dim: int = 7
    num_base_candidates: int = 8
    image_embedding_dim: int = 512
    state_embedding_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    num_qs: int = 10
    num_min_qs: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    edit_scale: float = 0.1
    initial_temperature: float = 1.0
    target_entropy: float = -28.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    crop_padding: int = 4
    rotation_degrees: float = 5.0
    color_jitter: float = 0.1

    def __post_init__(self) -> None:
        if self.image_shape != (6, 64, 64):
            raise ValueError(f"EXPO requires dual-camera images shaped (6, 64, 64), got {self.image_shape}")
        if (self.state_dim, self.chunk_length, self.action_dim) != (8, 8, 7):
            raise ValueError("EXPO requires state_dim=8, chunk_length=8, and action_dim=7")
        if self.num_base_candidates != 8:
            raise ValueError("EXPO requires eight base candidates")
        if not 0 < self.num_min_qs <= self.num_qs:
            raise ValueError("num_min_qs must be in [1, num_qs]")
        if self.edit_scale <= 0:
            raise ValueError("edit_scale must be positive")
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1]")

    @property
    def flat_action_dim(self) -> int:
        return self.chunk_length * self.action_dim


def _make_mlp(
    input_dim: int, hidden_dims: tuple[int, ...], output_dim: int | None = None
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
        current_dim = hidden_dim
    if output_dim is not None:
        layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class ImageEncoder(nn.Module):
    """Dual-camera ResNet-50 encoder with a 512-dimensional projection."""

    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        backbone = resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(feature_dim, embedding_dim)
        self.output_dim = embedding_dim

    def forward(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (6, 64, 64):
            raise ValueError(f"Expected pixels [B, 6, 64, 64], got {tuple(pixels.shape)}")
        pixels = pixels.float().div(255.0) if pixels.dtype == torch.uint8 else pixels.float()
        return self.projection(self.backbone(pixels))


class EditActor(nn.Module):
    """Tanh-Normal policy over a flattened 8x7 residual action chunk."""

    def __init__(self, config: EXPOConfig) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.state_embedding_dim), nn.ReLU()
        )
        input_dim = config.image_embedding_dim + config.state_embedding_dim + config.flat_action_dim
        self.trunk = _make_mlp(input_dim, config.hidden_dims)
        self.mean = nn.Linear(config.hidden_dims[-1], config.flat_action_dim)
        self.log_std = nn.Linear(config.hidden_dims[-1], config.flat_action_dim)
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(
        self, image_embedding: Tensor, state: Tensor, base_action: Tensor
    ) -> tuple[Tensor, Tensor]:
        if state.ndim != 2 or state.shape[1] != 8:
            raise ValueError(f"Expected state [B, 8], got {tuple(state.shape)}")
        if base_action.ndim != 2 or base_action.shape[1] != 56:
            raise ValueError(f"Expected base action [B, 56], got {tuple(base_action.shape)}")
        features = torch.cat(
            (image_embedding, self.state_encoder(state.float()), base_action.float()), dim=-1
        )
        hidden = self.trunk(features)
        return self.mean(hidden), self.log_std(hidden).clamp(self.log_std_min, self.log_std_max)

    def sample(
        self, image_embedding: Tensor, state: Tensor, base_action: Tensor
    ) -> tuple[Tensor, Tensor]:
        mean, log_std = self(image_embedding, state, base_action)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        residual = torch.tanh(pre_tanh)
        correction = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        log_probability = (distribution.log_prob(pre_tanh) - correction).sum(dim=-1)
        return residual, log_probability

    def deterministic(self, image_embedding: Tensor, state: Tensor, base_action: Tensor) -> Tensor:
        mean, _ = self(image_embedding, state, base_action)
        return torch.tanh(mean)


class Critic(nn.Module):
    """Ten-head state-action value ensemble over complete action chunks."""

    def __init__(self, config: EXPOConfig) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.state_embedding_dim), nn.ReLU()
        )
        input_dim = config.image_embedding_dim + config.state_embedding_dim + config.flat_action_dim
        self.q_heads = nn.ModuleList(
            _make_mlp(input_dim, config.hidden_dims, 1) for _ in range(config.num_qs)
        )

    def forward(
        self,
        image_embedding: Tensor,
        state: Tensor,
        action: Tensor,
        q_indices: Tensor | None = None,
    ) -> Tensor:
        if action.ndim != 2 or action.shape[1] != 56:
            raise ValueError(f"Expected action [B, 56], got {tuple(action.shape)}")
        features = torch.cat(
            (image_embedding, self.state_encoder(state.float()), action.float()), dim=-1
        )
        heads = self.q_heads if q_indices is None else [self.q_heads[int(i)] for i in q_indices]
        return torch.cat([head(features) for head in heads], dim=-1)


def _random_crop(pixels: Tensor, padding: int) -> Tensor:
    if padding == 0:
        return pixels
    padded = F.pad(pixels, (padding, padding, padding, padding), mode="replicate")
    height, width = pixels.shape[-2:]
    offsets = torch.randint(0, 2 * padding + 1, (pixels.shape[0], 2), device=pixels.device)
    return torch.stack(
        [
            padded[i, :, top : top + height, left : left + width]
            for i, (top, left) in enumerate(offsets.tolist())
        ]
    )


def _random_rotation(pixels: Tensor, degrees: float) -> Tensor:
    if degrees == 0:
        return pixels
    angles = torch.empty(pixels.shape[0], device=pixels.device).uniform_(-degrees, degrees)
    radians = angles * math.pi / 180.0
    transforms = torch.zeros(pixels.shape[0], 2, 3, device=pixels.device, dtype=pixels.dtype)
    transforms[:, 0, 0] = radians.cos()
    transforms[:, 0, 1] = -radians.sin()
    transforms[:, 1, 0] = radians.sin()
    transforms[:, 1, 1] = radians.cos()
    grid = F.affine_grid(transforms, pixels.shape, align_corners=False)
    return F.grid_sample(pixels, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _color_jitter(pixels: Tensor, strength: float) -> Tensor:
    if strength == 0:
        return pixels
    batch, channels, height, width = pixels.shape
    views = pixels.reshape(batch, channels // 3, 3, height, width)
    parameter_shape = (batch, 1, 1, 1, 1)
    brightness = torch.empty(parameter_shape, device=pixels.device).uniform_(1 - strength, 1 + strength)
    contrast = torch.empty(parameter_shape, device=pixels.device).uniform_(1 - strength, 1 + strength)
    saturation = torch.empty(parameter_shape, device=pixels.device).uniform_(1 - strength, 1 + strength)
    views = views * brightness
    views = (views - views.mean(dim=(-2, -1), keepdim=True)) * contrast + views.mean(
        dim=(-2, -1), keepdim=True
    )
    gray = views.mean(dim=2, keepdim=True)
    return ((views - gray) * saturation + gray).clamp(0.0, 1.0).reshape_as(pixels)


class EXPOLearner(nn.Module):
    def __init__(self, config: EXPOConfig | None = None, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.config = config or EXPOConfig()
        self.image_encoder = ImageEncoder(self.config.image_embedding_dim)
        self.actor = EditActor(self.config)
        self.critic = Critic(self.config)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(self.config.initial_temperature), dtype=torch.float32)
        )
        self.register_buffer("update_steps", torch.zeros((), dtype=torch.int64))
        self.to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.image_encoder.parameters()) + list(self.critic.parameters()),
            lr=self.config.critic_lr,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=self.config.temperature_lr
        )

    @property
    def device(self) -> torch.device:
        return self.log_temperature.device

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def _augment(self, pixels: Tensor) -> Tensor:
        pixels = _random_crop(pixels, self.config.crop_padding)
        pixels = pixels.float().div(255.0) if pixels.dtype == torch.uint8 else pixels.float()
        pixels = _random_rotation(pixels, self.config.rotation_degrees)
        return _color_jitter(pixels, self.config.color_jitter)

    def _sample_q_indices(self) -> Tensor:
        return torch.randperm(self.config.num_qs, device=self.device)[: self.config.num_min_qs]

    def inference_embedding(self, pixels: Tensor) -> Tensor:
        """Encode rollout observations without batch-size-one BatchNorm updates."""
        was_training = self.image_encoder.training
        self.image_encoder.eval()
        try:
            return self.image_encoder(pixels)
        finally:
            self.image_encoder.train(was_training)

    def _candidate_q_values(
        self,
        critic: Critic,
        image_embedding: Tensor,
        state: Tensor,
        candidates: Tensor,
        q_indices: Tensor,
    ) -> Tensor:
        batch_size, candidate_count = candidates.shape[:2]
        flat_actions = candidates.reshape(batch_size * candidate_count, -1)
        flat_images = image_embedding.repeat_interleave(candidate_count, dim=0)
        flat_states = state.repeat_interleave(candidate_count, dim=0)
        return critic(flat_images, flat_states, flat_actions, q_indices).reshape(
            batch_size, candidate_count, -1
        )

    def _next_candidates(
        self, image_embedding: Tensor, state: Tensor, base_candidates: Tensor
    ) -> Tensor:
        batch_size, candidate_count = base_candidates.shape[:2]
        flat_base = base_candidates.reshape(batch_size * candidate_count, -1)
        flat_images = image_embedding.repeat_interleave(candidate_count, dim=0)
        flat_states = state.repeat_interleave(candidate_count, dim=0)
        residual, _ = self.actor.sample(flat_images, flat_states, flat_base)
        residual = residual.reshape_as(base_candidates)
        return build_action_candidates(base_candidates, residual, self.config.edit_scale)

    @torch.no_grad()
    def _critic_target(
        self,
        next_pixels: Tensor,
        next_state: Tensor,
        next_base_candidates: Tensor,
        rewards: Tensor,
        discounts: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        next_embedding = self.image_encoder(next_pixels)
        candidates = self._next_candidates(next_embedding, next_state, next_base_candidates)
        selection_qs = self._candidate_q_values(
            self.target_critic,
            next_embedding,
            next_state,
            candidates,
            self._sample_q_indices(),
        )
        selected = select_top_q(candidates, selection_qs)
        target_qs = self.target_critic(
            next_embedding,
            next_state,
            selected.action.reshape(selected.action.shape[0], -1),
            self._sample_q_indices(),
        )
        next_q = target_qs.min(dim=-1).values
        target_q = rewards + discounts * masks * next_q
        return target_q, next_q, selected.index, selected.is_edited

    @torch.no_grad()
    def act_residual(
        self, pixels: Tensor, state: Tensor, base_action: Tensor, deterministic: bool = False
    ) -> Tensor:
        embedding = self.inference_embedding(pixels.to(self.device))
        state = state.to(self.device, dtype=torch.float32)
        base_action = base_action.to(self.device, dtype=torch.float32).reshape(-1, 56)
        if deterministic:
            residual = self.actor.deterministic(embedding, state, base_action)
        else:
            residual, _ = self.actor.sample(embedding, state, base_action)
        return residual.reshape(-1, 8, 7)

    @torch.no_grad()
    def select_action(
        self,
        pixels: Tensor,
        state: Tensor,
        base_candidates: Tensor,
        deterministic_actor: bool = False,
    ) -> tuple[CandidateSelection, Tensor]:
        """Edit eight base candidates and select one of the 16 candidates with target Q."""
        pixels = pixels.to(self.device)
        state = state.to(self.device, dtype=torch.float32)
        base_candidates = base_candidates.to(self.device, dtype=torch.float32)
        if pixels.shape != (1, 6, 64, 64) or state.shape != (1, 8):
            raise ValueError("Expected one dual-camera observation")
        if base_candidates.shape != (1, 8, 8, 7):
            raise ValueError(
                f"Expected base candidates [1, 8, 8, 7], got {tuple(base_candidates.shape)}"
            )

        embedding = self.inference_embedding(pixels)
        flat_base = base_candidates.reshape(8, 56)
        candidate_embedding = embedding.repeat_interleave(8, dim=0)
        candidate_state = state.repeat_interleave(8, dim=0)
        if deterministic_actor:
            residuals = self.actor.deterministic(
                candidate_embedding, candidate_state, flat_base
            )
        else:
            residuals, _ = self.actor.sample(candidate_embedding, candidate_state, flat_base)
        residuals = residuals.reshape(1, 8, 8, 7)
        candidates = build_action_candidates(
            base_candidates, residuals, self.config.edit_scale
        )
        q_values = self._candidate_q_values(
            self.target_critic,
            embedding,
            state,
            candidates,
            self._sample_q_indices(),
        )
        scores = reduce_q_values(q_values)
        selection = select_top_q(candidates, scores)
        selection = CandidateSelection(
            action=selection.action,
            index=selection.index,
            is_edited=selection.is_edited,
            score=selection.score,
            mean_edited_q_delta=(
                scores[:, self.config.num_base_candidates :]
                - scores[:, : self.config.num_base_candidates]
            ).mean(dim=1),
        )
        return selection, residuals

    def update(self, batch: dict[str, Any], update_actor: bool = True) -> dict[str, float]:
        self.train()
        observations = batch["observations"]
        next_observations = batch["next_observations"]
        pixels = self._augment(observations["pixels"].to(self.device))
        next_pixels = self._augment(next_observations["pixels"].to(self.device))
        state = observations["state"].to(self.device, dtype=torch.float32)
        next_state = next_observations["state"].to(self.device, dtype=torch.float32)
        executed_actions = batch["executed_actions"].to(
            self.device, dtype=torch.float32
        ).reshape(-1, 56)
        next_base_candidates = batch["next_base_candidates"].to(
            self.device, dtype=torch.float32
        )
        rewards = batch["rewards"].to(self.device, dtype=torch.float32).reshape(-1)
        discounts = batch["discounts"].to(self.device, dtype=torch.float32).reshape(-1)
        masks = batch["masks"].to(self.device, dtype=torch.float32).reshape(-1)

        target_q, next_q, selected_indices, selected_edited = self._critic_target(
            next_pixels, next_state, next_base_candidates, rewards, discounts, masks
        )
        image_embedding = self.image_encoder(pixels)
        actor_image_embedding = image_embedding.detach()
        predicted_qs = self.critic(image_embedding, state, executed_actions)
        critic_loss = F.mse_loss(predicted_qs, target_q.unsqueeze(-1).expand_as(predicted_qs))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, self.config.tau)

        if update_actor:
            self.critic.requires_grad_(False)
            residual, log_probability = self.actor.sample(
                actor_image_embedding, state, executed_actions
            )
            edited_actions = executed_actions + self.config.edit_scale * residual
            policy_q = self.critic(actor_image_embedding, state, edited_actions).mean(dim=-1)
            actor_loss = (self.temperature.detach() * log_probability - policy_q).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critic.requires_grad_(True)

            entropy = -log_probability.detach()
            temperature_loss = (self.temperature * (entropy - self.config.target_entropy)).mean()
            self.temperature_optimizer.zero_grad(set_to_none=True)
            temperature_loss.backward()
            self.temperature_optimizer.step()
            residual_norm = residual.detach().norm(dim=-1).mean()
        else:
            actor_loss = torch.zeros((), device=self.device)
            temperature_loss = torch.zeros((), device=self.device)
            entropy = torch.zeros((), device=self.device)
            residual_norm = torch.zeros((), device=self.device)
        self.update_steps.add_(1)

        metrics = {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "temperature_loss": temperature_loss.detach(),
            "temperature": self.temperature.detach(),
            "entropy": entropy.mean(),
            "actor_updated": torch.tensor(float(update_actor), device=self.device),
            "q": predicted_qs.detach().mean(),
            "target_q": target_q.detach().mean(),
            "next_q": next_q.detach().mean(),
            "edited_selection_rate": selected_edited.float().mean(),
            "selected_candidate": selected_indices.float().mean(),
            "residual_norm": residual_norm,
        }
        return {name: float(value.cpu()) for name, value in metrics.items()}

    def checkpoint_state(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "model": self.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "metadata": metadata or {},
        }

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        torch.save(self.checkpoint_state(metadata), Path(path))

    @classmethod
    def from_checkpoint_state(
        cls, checkpoint: dict[str, Any], device: str | torch.device = "cpu"
    ) -> tuple[EXPOLearner, dict[str, Any]]:
        config_data = checkpoint["config"].copy()
        config_data["image_shape"] = tuple(config_data["image_shape"])
        config_data["hidden_dims"] = tuple(config_data["hidden_dims"])
        learner = cls(EXPOConfig(**config_data), device=device)
        learner.load_state_dict(checkpoint["model"])
        learner.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        learner.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        learner.temperature_optimizer.load_state_dict(checkpoint["temperature_optimizer"])
        return learner, checkpoint["metadata"]

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> tuple[EXPOLearner, dict[str, Any]]:
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
        return cls.from_checkpoint_state(checkpoint, device=device)
