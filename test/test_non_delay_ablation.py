from __future__ import annotations

import pytest

import non_delay_ablation as experiment


def _row(*, episode: int, forced: int, longest: int, terminated: bool = True):
    row = {
        "episode": str(episode),
        "instance_seed": str(1000 + episode),
        "steps": "10",
        "forced_actions": str(forced),
        "forced_action_state_count": str(forced),
        "forced_action_chain_count": "1" if forced else "0",
        "forced_worker_pair_non_delay_count": str(forced // 2),
        "longest_forced_action_chain": str(longest),
        "terminated": str(terminated),
        "flow_time_objective": str(100.0 + episode),
        "reconfiguration_cost": str(20.0 + episode),
        "worker_load_variance": str(2.0 + episode),
        "forced_action_ratio": str(forced / 10.0),
    }
    for field in experiment.FORCED_ACTION_COUNT_FIELDS:
        row.setdefault(field, "0")
    row["forced_action_state_count"] = str(forced)
    row["forced_action_chain_count"] = "1" if forced else "0"
    row["forced_worker_pair_non_delay_count"] = str(forced // 2)
    return row


def test_non_delay_ablation_design_has_only_the_intended_difference():
    assert experiment.validate_design() == {
        "experiment_name",
        "environment.worker_resource_control.non_delay_worker_dispatch",
    }
    enabled = experiment.effective_config(True)
    disabled = experiment.effective_config(False)
    assert enabled["training"]["episodes"] == disabled["training"][
        "episodes"
    ] == 600
    assert enabled["seed"] == disabled["seed"] == 11
    assert enabled["training"]["forced_action_compression"]
    assert disabled["training"]["forced_action_compression"]


def test_training_summary_audits_forced_counts_and_long_tail():
    summary = experiment._training_summary(
        [_row(episode=0, forced=4, longest=3), _row(episode=1, forced=6, longest=5)]
    )
    assert summary["completion_rate"] == 1.0
    assert summary["forced_action_state_count"] == 10
    assert summary["forced_action_ratio"] == pytest.approx(0.5)
    assert summary["maximum_forced_action_chain"] == 5
    assert summary["forced_worker_pair_non_delay_count"] == 5

    invalid = _row(episode=0, forced=4, longest=3)
    invalid["forced_action_state_count"] = "3"
    with pytest.raises(RuntimeError, match="diagnostics disagree"):
        experiment._training_summary([invalid])


def test_paired_statistic_uses_off_minus_on_direction():
    result = experiment._paired_statistic([3.0, 4.0, 5.0], [2.0, 4.0, 4.0])
    assert result["off_minus_on_mean"] == pytest.approx(-2.0 / 3.0)
    assert result["count"] == 3
