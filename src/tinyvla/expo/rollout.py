"""Frozen SmolVLA proposal generation and EXPO environment rollout."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from tinyvla.dsrl.rollout import extract_sac_observation, prepare_observation

from .candidates import CandidateSelection
from .replay_buffer import EXPOReplayBuffer


@dataclass(frozen=True)
class BaseProposals:
    full_chunks: Tensor
    candidates: Tensor
    inference_seconds: float


@dataclass(frozen=True)
class EXPORolloutConfig:
    execute_horizon: int = 8
    chunk_size: int = 50
    noise_dim: int = 32
    num_base_candidates: int = 8
    max_episode_steps: int = 300
    discount: float = 0.99
    bootstrap_on_truncation: bool = True
    seed: int = 1000
    image_keys: tuple[str, str] = (
        "observation.images.image",
        "observation.images.image2",
    )
    deterministic_actor: bool = False
    warmup_base_only_until: int = 0
    base_exploration_prob: float = 0.0

    def __post_init__(self) -> None:
        if (self.execute_horizon, self.chunk_size, self.noise_dim) != (8, 50, 32):
            raise ValueError("EXPO requires execute_horizon=8, chunk_size=50, noise_dim=32")
        if self.num_base_candidates != 8:
            raise ValueError("EXPO requires eight base candidates")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if self.warmup_base_only_until < 0:
            raise ValueError("warmup_base_only_until must be non-negative")
        if not 0.0 <= self.base_exploration_prob <= 1.0:
            raise ValueError("base_exploration_prob must be in [0, 1]")


def _repeat_policy_observation(
    observation: dict[str, Any], batch_size: int
) -> dict[str, Any]:
    repeated: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, Tensor):
            if value.ndim == 0 or value.shape[0] != 1:
                raise ValueError(f"Expected batch-one tensor for {key}, got {tuple(value.shape)}")
            repeated[key] = value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        elif isinstance(value, list) and len(value) == 1:
            repeated[key] = value * batch_size
        else:
            repeated[key] = value
    return repeated


def generate_base_proposals(
    policy: Any,
    preprocessor: Any,
    observation: dict[str, Any],
    generator: torch.Generator | None = None,
    noise: Tensor | None = None,
) -> BaseProposals:
    """Generate eight normalized SmolVLA action chunks from independent Gaussian noise."""
    if hasattr(policy, "eval"):
        policy.eval()
    if hasattr(policy, "requires_grad_"):
        policy.requires_grad_(False)

    policy_observation = preprocessor(copy.deepcopy(observation))
    state = policy_observation["observation.state"]
    device = state.device
    if noise is None:
        noise = torch.randn(
            (8, 50, 32),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device)
    else:
        noise = noise.to(device=device, dtype=torch.float32)
    if noise.shape != (8, 50, 32):
        raise ValueError(f"Expected noise [8, 50, 32], got {tuple(noise.shape)}")
    policy_observation = _repeat_policy_observation(policy_observation, batch_size=8)

    start = time.perf_counter()
    with torch.inference_mode():
        full_chunks = policy.predict_action_chunk(policy_observation, noise=noise)
    inference_seconds = time.perf_counter() - start
    if full_chunks.shape != (8, 50, 7):
        raise ValueError(
            f"Expected normalized action chunks [8, 50, 7], got {tuple(full_chunks.shape)}"
        )
    if not torch.isfinite(full_chunks).all():
        raise ValueError("SmolVLA proposals contain NaN or Inf")
    return BaseProposals(
        full_chunks=full_chunks,
        candidates=full_chunks[:, :8].unsqueeze(0),
        inference_seconds=inference_seconds,
    )


def write_selected_action_chunk(
    full_chunks: Tensor,
    selection: CandidateSelection,
) -> Tensor:
    """Put the selected 8-step action into its corresponding normalized 50-step chunk."""
    if full_chunks.shape != (8, 50, 7):
        raise ValueError(f"Expected full chunks [8, 50, 7], got {tuple(full_chunks.shape)}")
    if selection.action.shape != (1, 8, 7) or selection.index.shape != (1,):
        raise ValueError("Expected a single selected action shaped [1, 8, 7]")
    selected_index = int(selection.index.item())
    if not 0 <= selected_index < 16:
        raise ValueError("Selected candidate index must be in [0, 15]")
    base_index = selected_index % 8
    output = full_chunks[base_index : base_index + 1].clone()
    output[:, :8] = selection.action.to(output.device, output.dtype)
    return output


def _first_flag(value: Any) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _is_success(info: dict[str, Any]) -> bool:
    if "is_success" in info:
        return _first_flag(info["is_success"])
    final_info = info.get("final_info")
    if isinstance(final_info, dict) and "is_success" in final_info:
        return _first_flag(final_info["is_success"])
    if final_info is not None:
        first = np.asarray(final_info, dtype=object).reshape(-1)[0]
        return bool(first.get("is_success", False)) if isinstance(first, dict) else False
    return False


def _postprocess_chunk(
    normalized_chunk: Tensor, postprocessor: Any, env_postprocessor: Any
) -> np.ndarray:
    with torch.inference_mode():
        action_chunk = postprocessor(normalized_chunk)
        action_chunk = env_postprocessor({"action": action_chunk})["action"]
    action_chunk = action_chunk.detach().to("cpu", dtype=torch.float32).numpy()
    if action_chunk.shape != (1, 50, 7) or not np.isfinite(action_chunk).all():
        raise ValueError(f"Expected finite postprocessed action chunk [1, 50, 7], got {action_chunk.shape}")
    return action_chunk


def rollout_episode(
    env: Any,
    policy: Any,
    learner: Any,
    env_preprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_postprocessor: Any,
    buffer: EXPOReplayBuffer | None,
    config: EXPORolloutConfig | None = None,
    render_callback: Any | None = None,
) -> dict[str, Any]:
    """Run one EXPO episode and cache next-state SmolVLA candidates in replay."""
    config = config or EXPORolloutConfig()
    observation, _ = env.reset(seed=config.seed)
    task_description = str(env.call("task_description")[0])
    policy.reset()
    if render_callback is not None:
        render_callback(env)

    processed = prepare_observation(observation, task_description, env_preprocessor)
    expo_observation = extract_sac_observation(processed, config.image_keys)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    selection_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1_000_000)
    proposals = generate_base_proposals(policy, preprocessor, processed, generator)

    episode_steps = 0
    episode_return = 0.0
    episode_success = False
    terminated = False
    truncated = False
    transitions: list[dict[str, Any]] = []

    while not (terminated or truncated):
        pixels = torch.from_numpy(expo_observation["pixels"]).unsqueeze(0)
        state = torch.from_numpy(expo_observation["state"]).unsqueeze(0)
        warmup_base_only = (
            buffer is not None and len(buffer) < config.warmup_base_only_until
        )
        base_exploration = (
            buffer is not None
            and not warmup_base_only
            and torch.rand((), generator=selection_generator).item()
            < config.base_exploration_prob
        )
        if warmup_base_only or base_exploration:
            selected_index = torch.randint(
                config.num_base_candidates,
                (1,),
                generator=selection_generator,
            )
            selection = CandidateSelection(
                action=proposals.candidates[:, int(selected_index.item())],
                index=selected_index,
                is_edited=torch.zeros(1, dtype=torch.bool),
                score=torch.zeros(1),
            )
            residuals = torch.zeros_like(proposals.candidates)
        else:
            selection, residuals = learner.select_action(
                pixels,
                state,
                proposals.candidates,
                deterministic_actor=config.deterministic_actor,
            )
        normalized_chunk = write_selected_action_chunk(proposals.full_chunks, selection)
        action_chunk = _postprocess_chunk(normalized_chunk, postprocessor, env_postprocessor)

        executed_steps = 0
        chunk_success = False
        env_truncated = False
        remaining_steps = config.max_episode_steps - episode_steps
        for action_index in range(min(config.execute_horizon, remaining_steps)):
            observation, reward, env_terminated, step_truncated, info = env.step(
                action_chunk[:, action_index]
            )
            executed_steps += 1
            episode_steps += 1
            episode_return += float(np.asarray(reward).reshape(-1)[0])
            chunk_success = chunk_success or _is_success(info)
            terminated = _first_flag(env_terminated) or chunk_success
            env_truncated = _first_flag(step_truncated)
            if render_callback is not None:
                render_callback(env)
            if terminated or env_truncated:
                break
        if executed_steps == 0:
            raise RuntimeError("Rollout produced an empty chunk transition")

        truncated = env_truncated or (episode_steps >= config.max_episode_steps and not terminated)
        episode_success = episode_success or chunk_success
        mask = 0.0 if terminated or (truncated and not config.bootstrap_on_truncation) else 1.0
        next_processed = prepare_observation(observation, task_description, env_preprocessor)
        next_expo_observation = extract_sac_observation(next_processed, config.image_keys)

        next_proposals = None
        if not terminated and (not truncated or mask == 1.0):
            next_proposals = generate_base_proposals(
                policy, preprocessor, next_processed, generator
            )

        pending_transition = {
            "observation": expo_observation,
            "executed_action": selection.action[0].detach().cpu().numpy(),
            "base_action": proposals.candidates[
                0, int(selection.index.item()) % config.num_base_candidates
            ]
            .detach()
            .cpu()
            .numpy(),
            "reward": 1.0 if chunk_success else 0.0,
            "discount": config.discount**executed_steps,
            "mask": mask,
            "next_observation": next_expo_observation,
            "next_base_candidates": (
                None
                if next_proposals is None
                else next_proposals.candidates[0].detach().cpu().numpy()
            ),
            "terminated": terminated,
            "truncated": truncated,
            "executed_steps": executed_steps,
        }
        if buffer is not None:
            buffer.add(**pending_transition)

        residual_norm = float(residuals.norm(dim=(-2, -1)).mean().cpu())
        transitions.append(
            {
                "chunk_index": len(transitions),
                "executed_steps": executed_steps,
                "environment_steps": episode_steps,
                "reward": pending_transition["reward"],
                "discount": pending_transition["discount"],
                "mask": mask,
                "terminated": terminated,
                "truncated": truncated,
                "success": chunk_success,
                "selected_candidate": int(selection.index.item()),
                "selected_edited": bool(selection.is_edited.item()),
                "selected_q": float(selection.score.item()),
                "selection_mode": (
                    "warmup_base"
                    if warmup_base_only
                    else "base_exploration" if base_exploration else "top_q"
                ),
                "mean_edited_q_delta": (
                    None
                    if selection.mean_edited_q_delta is None
                    else float(selection.mean_edited_q_delta.item())
                ),
                "residual_norm": residual_norm,
                "normalized_action_min": float(selection.action.min().item()),
                "normalized_action_max": float(selection.action.max().item()),
                "normalized_action_out_of_range_rate": float(
                    selection.action.abs().gt(1.0).float().mean().item()
                ),
                "action_min": float(action_chunk[:, :executed_steps].min()),
                "action_max": float(action_chunk[:, :executed_steps].max()),
                "inference_seconds": proposals.inference_seconds,
            }
        )

        if not (terminated or truncated):
            processed = next_processed
            expo_observation = next_expo_observation
            if next_proposals is None:
                raise RuntimeError("Missing proposals for the next decision")
            proposals = next_proposals

    return {
        "task_description": task_description,
        "success": episode_success,
        "terminated": terminated,
        "truncated": truncated,
        "environment_steps": episode_steps,
        "chunk_transitions": len(transitions),
        "environment_return": episode_return,
        "transitions": transitions,
    }
