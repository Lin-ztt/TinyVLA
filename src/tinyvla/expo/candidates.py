"""Pure tensor operations for EXPO action candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CandidateSelection:
    action: Tensor
    index: Tensor
    is_edited: Tensor
    score: Tensor
    mean_edited_q_delta: Tensor | None = None


def build_action_candidates(
    base_actions: Tensor,
    residuals: Tensor,
    edit_scale: float,
) -> Tensor:
    """Return base candidates followed by their element-wise edited versions."""
    if base_actions.ndim != 4:
        raise ValueError(f"Expected base actions [B, N, C, A], got {tuple(base_actions.shape)}")
    if residuals.shape != base_actions.shape:
        raise ValueError(
            f"Residual shape {tuple(residuals.shape)} does not match "
            f"base shape {tuple(base_actions.shape)}"
        )
    edited_actions = base_actions + edit_scale * residuals
    return torch.cat((base_actions, edited_actions), dim=1)


def reduce_q_values(q_values: Tensor) -> Tensor:
    """Reduce optional Q-ensemble outputs to one conservative score per candidate."""
    if q_values.ndim == 2:
        return q_values
    if q_values.ndim == 3:
        return q_values.min(dim=-1).values
    raise ValueError(
        f"Expected Q values [B, candidates] or [B, candidates, Q], got {tuple(q_values.shape)}"
    )


def select_top_q(candidates: Tensor, q_values: Tensor) -> CandidateSelection:
    """Select the highest-scoring complete action chunk for every batch item."""
    if candidates.ndim != 4:
        raise ValueError(f"Expected candidates [B, N, C, A], got {tuple(candidates.shape)}")
    scores = reduce_q_values(q_values)
    expected_shape = candidates.shape[:2]
    if scores.shape != expected_shape:
        raise ValueError(f"Expected Q scores {expected_shape}, got {tuple(scores.shape)}")
    if candidates.shape[1] % 2 != 0:
        raise ValueError("Expected equal numbers of base and edited candidates")

    indices = scores.argmax(dim=1)
    batch_indices = torch.arange(candidates.shape[0], device=candidates.device)
    return CandidateSelection(
        action=candidates[batch_indices, indices],
        index=indices,
        is_edited=indices >= candidates.shape[1] // 2,
        score=scores[batch_indices, indices],
    )
