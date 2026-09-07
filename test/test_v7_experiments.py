from __future__ import annotations

import json
import pytest

from configs import load_config
from train import TrainingPhaseController


def test_default_and_e2_7_training_configs_use_cuda():
    assert load_config("configs/default.json")["device"] == "cuda"
    assert load_config(
        "configs/v7/e2_7_e1_warmstart_safe_gate_v2_1.json"
    )["device"] == "cuda"


def test_recursive_config_extends_replaces_lists_and_detects_cycles(tmp_path):
    base = tmp_path / "base.json"
    child = tmp_path / "child.json"
    base.write_text(
        json.dumps({"nested": {"keep": 1, "replace": [1, 2]}}),
        encoding="utf-8",
    )
    child.write_text(
        json.dumps(
            {
                "extends": "base.json",
                "nested": {"replace": [3], "added": 2},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_config(child)
    assert loaded["nested"] == {"keep": 1, "replace": [3], "added": 2}

    base.write_text(json.dumps({"extends": "child.json"}), encoding="utf-8")
    with pytest.raises(ValueError, match="extends cycle"):
        load_config(child)


def test_balanced_guard_delays_sampled_evaluation_until_greedy_eligible():
    config = load_config("configs/v7/c0_v6_control.json")
    controller = TrainingPhaseController.from_config(config)
    anchor_score = (-1.0, 0.5, 100.0, 10.0)
    assert controller.observe_validation(
        1.0, completed_episodes=1, score=anchor_score,
        normalized_quality_score=0.5,
    ) == "feasibility"
    assert controller.observe_validation(
        1.0, completed_episodes=2, score=anchor_score,
        normalized_quality_score=0.5,
    ) == "feasibility"
    assert controller.observe_validation(
        1.0, completed_episodes=3, score=anchor_score,
        normalized_quality_score=0.5,
    ) == "transition"
    assert controller.observe_sampled_guard(
        {
            "completion_rate": 1.0,
            "fatigue_cvar90": 0.2,
            "fatigue_safe_line_pass": True,
        },
        completed_episodes=3,
        transition_anchor=True,
    ) == "transition"

    candidate_score = (-1.0, 0.4999, 100.0, 10.5)
    assert controller.observe_validation(
        1.0, completed_episodes=4, score=candidate_score,
        normalized_quality_score=0.4999,
    ) == "sampled_guard_pending"
    assert controller.accepted_normalized_quality_score == 0.5
    assert controller.last_promotion_diagnostics[
        "promotion_sampled_guard_executed"
    ] is False
    assert controller.observe_sampled_guard(
        {
            "completion_rate": 0.98,
            "fatigue_cvar90": 0.21,
            "fatigue_safe_line_pass": True,
        },
        completed_episodes=4,
    ) == "accepted"
    assert controller.accepted_normalized_quality_score == pytest.approx(0.4999)

    assert controller.observe_validation(
        1.0, completed_episodes=5,
        score=(-1.0, 0.4997, 100.0, 10.5),
        normalized_quality_score=0.4997,
    ) == "sampled_guard_pending"
    assert controller.observe_sampled_guard(
        {
            "completion_rate": 1.0,
            "fatigue_cvar90": 0.20,
            "fatigue_safe_line_pass": False,
        },
        completed_episodes=5,
    ) == "rejected"
    assert controller.last_promotion_diagnostics[
        "promotion_decision_reason"
    ] == "sampled_fatigue_safety_violation"


def test_e1_configs_keep_p5_e6_disabled():
    for path in (
        "configs/v7/c0_v6_control.json",
        "configs/v7/e1_context_exception.json",
    ):
        config = load_config(path)
        assert config["experiment_suite_version"] == "v7_e1_protocol_v2"
        assert config["evaluation"]["quality_metric"]["version"] == (
            "canonical_bounded_quality_v1"
        )
        assert config["training"]["forced_action_compression"] is False
        assert config["environment"]["worker_resource_control"][
            "non_delay_worker_dispatch"
        ] is True
