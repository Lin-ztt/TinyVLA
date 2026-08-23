from __future__ import annotations

import torch

from tinyvla.expo.candidates import build_action_candidates, select_top_q


def test_elementwise_edit_and_zero_scale() -> None:
    base = torch.arange(2 * 8 * 8 * 7, dtype=torch.float32).reshape(2, 8, 8, 7)
    residual = torch.zeros_like(base)
    residual[0, 3, 4, 5] = 2.0

    candidates = build_action_candidates(base, residual, edit_scale=0.1)
    assert candidates.shape == (2, 16, 8, 7)
    assert torch.equal(candidates[:, :8], base)
    difference = candidates[:, 8:] - base
    assert torch.count_nonzero(difference) == 1
    assert torch.isclose(difference[0, 3, 4, 5], torch.tensor(0.2), atol=1e-4)
    assert torch.equal(build_action_candidates(base, residual, 0.0)[:, 8:], base)


def test_top_q_selection_matches_action_and_candidate_type() -> None:
    base = torch.zeros(2, 8, 8, 7)
    residual = torch.ones_like(base)
    candidates = build_action_candidates(base, residual, edit_scale=0.1)
    candidates[:, :, 0, 0] = torch.arange(16)
    q_values = torch.zeros(2, 16, 2)
    q_values[0, 5] = torch.tensor((4.0, 3.0))
    q_values[1, 13] = torch.tensor((8.0, 7.0))

    selected = select_top_q(candidates, q_values)
    assert selected.index.tolist() == [5, 13]
    assert selected.is_edited.tolist() == [False, True]
    assert torch.equal(selected.action[0], candidates[0, 5])
    assert torch.equal(selected.action[1], candidates[1, 13])
    assert selected.score.tolist() == [3.0, 7.0]
