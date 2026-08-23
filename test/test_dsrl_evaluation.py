from __future__ import annotations

from scripts.eval_smolvla_dsrl import (
    bootstrap_interval,
    load_settings,
    summarize_results,
    validate_fair_protocol,
)


def test_bootstrap_and_summary_are_deterministic() -> None:
    values = [0.0, 1.0, 1.0, 0.0]
    assert bootstrap_interval(values, 1000, 0.95, 7) == bootstrap_interval(
        values, 1000, 0.95, 7
    )

    results = [
        {
            "task_id": 2,
            "mode": "native",
            "success": success,
            "completion_steps": steps if success else None,
            "environment_steps": steps,
        }
        for success, steps in ((True, 10), (False, 20), (True, 14), (False, 20))
    ]
    summary = summarize_results(results, 1000, 0.95, 7)["2"]["native"]
    assert summary["successes"] == 2
    assert summary["success_rate"]["mean"] == 0.5
    assert summary["completion_steps_success_only"]["mean"] == 12.0
    assert summary["environment_steps"]["mean"] == 16.0


def test_fair_protocol_requires_identical_episode_keys() -> None:
    results = []
    for mode in ("native", "learned"):
        for episode_index in range(2):
            results.append(
                {
                    "task_id": 1,
                    "mode": mode,
                    "episode_index": episode_index,
                    "init_state_index": episode_index,
                    "seed": 100 + episode_index,
                }
            )
    validate_fair_protocol(results, [1], ["native", "learned"], 2)

    results[-1]["seed"] = 999
    try:
        validate_fair_protocol(results, [1], ["native", "learned"], 2)
    except RuntimeError as error:
        assert "protocol differs" in str(error)
    else:
        raise AssertionError("Expected mismatched protocols to fail")


def test_stochastic_learned_mode_requires_actor_checkpoint() -> None:
    class Args:
        config = None
        suite = "libero_10"
        task_ids = [1]
        modes = ["learned_stochastic"]
        episodes = 2
        seed = 7
        max_episode_steps = 400
        init_state_start = 0
        checkpoint = None
        actor_checkpoint = None
        output_dir = None
        device = None

    try:
        load_settings(Args())
    except ValueError as error:
        assert "actor_checkpoint" in str(error)
    else:
        raise AssertionError("Expected learned_stochastic without a checkpoint to fail")
