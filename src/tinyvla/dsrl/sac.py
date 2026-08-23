"""PyTorch SAC for selecting structured SmolVLA initial noise."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SACConfig:
    image_shape: tuple[int, int, int] = (3, 64, 64)
    state_dim: int = 8
    action_dim: int = 32
    hidden_dims: tuple[int, ...] = (128, 128, 128)
    latent_dim: int = 50
    num_qs: int = 10
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    discount: float = 0.999
    tau: float = 0.005
    initial_temperature: float = 1.0
    target_entropy: float = -16.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    critic_reduction: str = "mean"
    backup_entropy: bool = False
    random_crop_padding: int = 0
    color_jitter: bool = False

    def __post_init__(self) -> None:
        if self.image_shape not in ((3, 64, 64), (6, 64, 64)):
            raise ValueError(f"Expected a 3- or 6-channel 64x64 image, got {self.image_shape}")
        if self.state_dim != 8 or self.action_dim != 32:
            raise ValueError("The first DSRL version requires state_dim=8 and action_dim=32")
        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must not be empty")
        if self.num_qs <= 0:
            raise ValueError("num_qs must be positive")
        if self.critic_reduction != "mean":
            raise ValueError("The first DSRL version requires critic_reduction='mean'")
        if self.backup_entropy:
            raise ValueError("The first DSRL version requires backup_entropy=False")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.random_crop_padding < 0:
            raise ValueError("random_crop_padding must be non-negative")


def _orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class PixelEncoder(nn.Module):
    """Four-layer DSRL pixel encoder with a 50-dimensional bottleneck."""

    def __init__(
        self, image_shape: tuple[int, int, int] = (3, 64, 64), latent_dim: int = 50
    ) -> None:
        super().__init__()
        channels, height, width = image_shape
        strides = (2, 1, 1, 1)
        layers: list[nn.Module] = []
        in_channels = channels
        for stride in strides:
            layers.extend((nn.Conv2d(in_channels, 32, kernel_size=3, stride=stride), nn.ReLU()))
            in_channels = 32
            height = (height - 3) // stride + 1
            width = (width - 3) // stride + 1
        self.image_shape = image_shape
        self.convs = nn.Sequential(*layers)
        self.projection = nn.Linear(32 * height * width, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

        self.convs.apply(_orthogonal_init)
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != self.image_shape:
            raise ValueError(
                f"Expected pixels with shape [B, {', '.join(map(str, self.image_shape))}], "
                f"got {tuple(pixels.shape)}"
            )
        encoded = pixels.float().div(255.0) if pixels.dtype == torch.uint8 else pixels.float()
        encoded = self.convs(encoded).flatten(start_dim=1)
        return torch.tanh(self.layer_norm(self.projection(encoded)))


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
    network = nn.Sequential(*layers)
    network.apply(_orthogonal_init)
    return network


class Actor(nn.Module):
    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.encoder = PixelEncoder(config.image_shape, config.latent_dim)
        self.trunk = _make_mlp(config.latent_dim + config.state_dim, config.hidden_dims)
        self.mean = nn.Linear(config.hidden_dims[-1], config.action_dim)
        self.log_std = nn.Linear(config.hidden_dims[-1], config.action_dim)
        _orthogonal_init(self.mean, gain=1e-2)
        _orthogonal_init(self.log_std, gain=1e-2)
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, pixels: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        if state.ndim != 2 or state.shape[1] != 8:
            raise ValueError(f"Expected state with shape [B, 8], got {tuple(state.shape)}")
        features = torch.cat((self.encoder(pixels), state.float()), dim=-1)
        hidden = self.trunk(features)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, pixels: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self(pixels, state)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_probability = distribution.log_prob(pre_tanh)
        correction = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        log_probability = (log_probability - correction).sum(dim=-1)
        return action, log_probability

    def deterministic(self, pixels: Tensor, state: Tensor) -> Tensor:
        mean, _ = self(pixels, state)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.encoder = PixelEncoder(config.image_shape, config.latent_dim)
        input_dim = config.latent_dim + config.state_dim + config.action_dim
        self.q_heads = nn.ModuleList(
            _make_mlp(input_dim, config.hidden_dims, 1) for _ in range(config.num_qs)
        )

    def forward(self, pixels: Tensor, state: Tensor, action: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[1] != 8:
            raise ValueError(f"Expected state with shape [B, 8], got {tuple(state.shape)}")
        if action.ndim != 2 or action.shape[1] != 32:
            raise ValueError(f"Expected action with shape [B, 32], got {tuple(action.shape)}")
        features = torch.cat((self.encoder(pixels), state.float(), action.float()), dim=-1)
        return torch.cat([head(features) for head in self.q_heads], dim=-1)


def _random_crop(pixels: Tensor, padding: int) -> Tensor:
    if padding == 0:
        return pixels
    padded = F.pad(pixels, (padding, padding, padding, padding), mode="replicate")
    size = pixels.shape[-1]
    offsets = torch.randint(0, 2 * padding + 1, (pixels.shape[0], 2)).tolist()
    crops = [
        padded[index, :, top : top + size, left : left + size]
        for index, (top, left) in enumerate(offsets)
    ]
    return torch.stack(crops, dim=0)


def _rgb_to_hsv(images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    red, green, blue = images.unbind(dim=2)
    value = torch.maximum(torch.maximum(red, green), blue)
    value_range = value - torch.minimum(torch.minimum(red, green), blue)
    saturation = torch.where(value > 0, value_range / value, 0.0)
    normalizer = torch.where(value_range != 0, 1.0 / (6.0 * value_range), 1e9)
    red_hue = normalizer * (green - blue)
    green_hue = normalizer * (blue - red) + 2.0 / 6.0
    blue_hue = normalizer * (red - green) + 4.0 / 6.0
    hue = torch.where(red == value, red_hue, torch.where(green == value, green_hue, blue_hue))
    hue = hue * (value_range > 0) + (hue < 0)
    return hue, saturation, value


def _hsv_to_rgb(hue: Tensor, saturation: Tensor, value: Tensor) -> Tensor:
    chroma = saturation * value
    minimum = value - chroma
    hue_section = hue.remainder(1.0) * 6.0
    intermediate = chroma * (1.0 - (hue_section.remainder(2.0) - 1.0).abs())
    category = hue_section.floor().to(torch.int64)
    zero = torch.zeros_like(chroma)
    red = torch.where(
        (category == 0) | (category == 5),
        chroma,
        torch.where((category == 1) | (category == 4), intermediate, zero),
    )
    green = torch.where(
        (category == 1) | (category == 2),
        chroma,
        torch.where((category == 0) | (category == 3), intermediate, zero),
    )
    blue = torch.where(
        (category == 3) | (category == 4),
        chroma,
        torch.where((category == 2) | (category == 5), intermediate, zero),
    )
    return torch.stack((red + minimum, green + minimum, blue + minimum), dim=2)


def _color_jitter(pixels: Tensor) -> Tensor:
    original = pixels.float().div(255.0) if pixels.dtype == torch.uint8 else pixels.float()
    batch_size, channels, height, width = original.shape
    images = original.reshape(batch_size, channels // 3, 3, height, width)
    parameter_shape = (batch_size, 1, 1, 1, 1)
    brightness = torch.empty(parameter_shape, device=images.device).uniform_(-0.2, 0.2)
    contrast = torch.empty(parameter_shape, device=images.device).uniform_(0.9, 1.1)
    saturation = torch.empty(parameter_shape, device=images.device).uniform_(0.9, 1.1)
    hue = torch.empty(parameter_shape, device=images.device).uniform_(-0.03, 0.03)
    order = torch.rand(batch_size, 4, device=images.device).argsort(dim=1)

    for order_index in range(4):
        mean = images.mean(dim=(-2, -1), keepdim=True)
        hsv = _rgb_to_hsv(images)
        candidates = (
            (images + brightness).clamp(0.0, 1.0),
            (contrast * (images - mean) + mean).clamp(0.0, 1.0),
            _hsv_to_rgb(hsv[0], (hsv[1] * saturation[:, :, 0]).clamp(0.0, 1.0), hsv[2]),
            _hsv_to_rgb((hsv[0] + hue[:, :, 0]).remainder(1.0), hsv[1], hsv[2]),
        )
        selected = order[:, order_index].reshape(batch_size, 1, 1, 1, 1)
        for transform_index, candidate in enumerate(candidates):
            images = torch.where(selected == transform_index, candidate, images)

    apply = torch.rand(parameter_shape, device=images.device) <= 0.8
    augmented = torch.where(apply, images, original.reshape_as(images))
    return augmented.reshape_as(original).mul(255.0).to(torch.uint8)


class SAC(nn.Module):
    """SAC learner whose action is a 32-dimensional structured noise vector."""

    def __init__(self, config: SACConfig | None = None, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.config = config or SACConfig()
        self.actor = Actor(self.config)
        self.critic = Critic(self.config)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(self.config.initial_temperature), dtype=torch.float32)
        )
        self.register_buffer("update_steps", torch.zeros((), dtype=torch.int64))
        self.to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=self.config.temperature_lr
        )

    @property
    def device(self) -> torch.device:
        return self.log_temperature.device

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    @torch.no_grad()
    def act(self, pixels: Tensor, state: Tensor, deterministic: bool = True) -> Tensor:
        pixels = pixels.to(self.device)
        state = state.to(self.device)
        if deterministic:
            return self.actor.deterministic(pixels, state)
        return self.actor.sample(pixels, state)[0]

    @torch.no_grad()
    def _critic_target(
        self,
        next_pixels: Tensor,
        next_state: Tensor,
        rewards: Tensor,
        discounts: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        next_action, next_log_probability = self.actor.sample(next_pixels, next_state)
        next_q = self.target_critic(next_pixels, next_state, next_action).mean(dim=-1)
        target_q = rewards + discounts * masks * next_q
        return target_q, next_q, next_log_probability

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        observations = batch["observations"]
        next_observations = batch["next_observations"]
        pixels = _random_crop(
            observations["pixels"].to(self.device), self.config.random_crop_padding
        )
        next_pixels = _random_crop(
            next_observations["pixels"].to(self.device), self.config.random_crop_padding
        )
        if self.config.color_jitter:
            pixels = _color_jitter(pixels)
            next_pixels = _color_jitter(next_pixels)
        state = observations["state"].to(self.device, dtype=torch.float32)
        next_state = next_observations["state"].to(self.device, dtype=torch.float32)
        actions = batch["actions"].to(self.device, dtype=torch.float32)
        rewards = batch["rewards"].to(self.device, dtype=torch.float32).reshape(-1)
        discounts = batch["discounts"].to(self.device, dtype=torch.float32).reshape(-1)
        masks = batch["masks"].to(self.device, dtype=torch.float32).reshape(-1)

        target_q, next_q, next_log_probability = self._critic_target(
            next_pixels, next_state, rewards, discounts, masks
        )
        predicted_qs = self.critic(pixels, state, actions)
        critic_loss = F.mse_loss(predicted_qs, target_q.unsqueeze(-1).expand_as(predicted_qs))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, self.config.tau)

        self.critic.requires_grad_(False)
        sampled_action, log_probability = self.actor.sample(pixels, state)
        q_for_policy = self.critic(pixels, state, sampled_action).mean(dim=-1)
        actor_loss = (self.temperature.detach() * log_probability - q_for_policy).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        entropy = -log_probability.detach()
        temperature_loss = (self.temperature * (entropy - self.config.target_entropy)).mean()
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        self.update_steps.add_(1)

        metrics = {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "temperature_loss": temperature_loss.detach(),
            "temperature": self.temperature.detach(),
            "entropy": entropy.mean(),
            "q": predicted_qs.detach().mean(),
            "target_q": target_q.detach().mean(),
            "next_q": next_q.detach().mean(),
            "next_log_probability": next_log_probability.detach().mean(),
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
    ) -> tuple[SAC, dict[str, Any]]:
        config_data = checkpoint["config"].copy()
        config_data["image_shape"] = tuple(config_data["image_shape"])
        config_data["hidden_dims"] = tuple(config_data["hidden_dims"])
        learner = cls(SACConfig(**config_data), device=device)
        learner.load_state_dict(checkpoint["model"])
        learner.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        learner.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        learner.temperature_optimizer.load_state_dict(checkpoint["temperature_optimizer"])
        return learner, checkpoint["metadata"]

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> tuple[SAC, dict[str, Any]]:
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
        return cls.from_checkpoint_state(checkpoint, device=device)
