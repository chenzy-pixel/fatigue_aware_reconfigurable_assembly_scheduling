from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import v7_experiments as experiments

from configs import load_config
from train import TrainingPhaseController
from v7_experiments import (
    ARMS,
    FORMAL_SEEDS,
    SCREEN_SEEDS,
    config_differences,
    exact_wilcoxon_two_sided,
    validate_arm_configs,
)


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


def test_v7_matrix_counts_and_arm_difference_whitelist():
    differences = validate_arm_configs()
    assert len(ARMS) * len(SCREEN_SEEDS) == 21
    assert len(ARMS) * len(FORMAL_SEEDS) == 35
    assert differences["e5"] == {"reward.variance_scale"}
    assert not any(
        path.startswith("training.forced_action_compression")
        for changed in differences.values()
        for path in changed
    )


def test_e5_has_exactly_one_non_metadata_treatment_difference():
    c0 = load_config(ARMS["c0"])
    e5 = load_config(ARMS["e5"])
    changed = config_differences(c0, e5) - {
        "experiment_name",
        "method_version",
    }
    assert changed == {"reward.variance_scale"}
    assert c0["reward"]["variance_scale"] == 50.0
    assert e5["reward"]["variance_scale"] == 20.0


def test_exact_wilcoxon_uses_algorithm_seed_as_independent_unit():
    result = exact_wilcoxon_two_sided([1, 2, 3, 4, 5])
    assert result == {"n": 5, "statistic": 15.0, "p_value": 0.0625}
    assert exact_wilcoxon_two_sided([0, 0])["p_value"] == 1.0


def test_balanced_guard_delays_sampled_evaluation_until_greedy_eligible():
    config = load_config(ARMS["c0"])
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


def test_all_v7_configs_keep_p5_e6_disabled():
    for path in ARMS.values():
        config = load_config(path)
        assert config["training"]["forced_action_compression"] is False
        assert config["environment"]["worker_resource_control"][
            "non_delay_worker_dispatch"
        ] is True


def test_orchestrator_dry_run_does_not_create_state(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "ROOT", tmp_path)
    monkeypatch.setattr(experiments, "suite_input_hash", lambda: "a" * 64)
    orchestrator = experiments.Orchestrator(
        parallel_envs=3, dry_run=True, max_retries=0
    )
    artifact = orchestrator._execute(
        "dry_task",
        lambda run: ["python", "noop.py", run],
        artifact_builder=lambda run: tmp_path / "runs" / run,
        required="summary.json",
    )
    assert artifact.name == "dry_task"
    assert not orchestrator.state_path.exists()


def test_orchestrator_retries_then_resumes_by_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "ROOT", tmp_path)
    monkeypatch.setattr(experiments, "suite_input_hash", lambda: "b" * 64)
    attempts = []
    artifact_holder = {}

    def artifact_builder(run):
        path = tmp_path / "runs" / run
        artifact_holder["path"] = path
        return path

    def fake_run(*args, **kwargs):
        attempts.append(args)
        if len(attempts) == 1:
            return subprocess.CompletedProcess(args[0], 1)
        path = artifact_holder["path"]
        path.mkdir(parents=True, exist_ok=True)
        (path / "summary.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(experiments.subprocess, "run", fake_run)
    orchestrator = experiments.Orchestrator(
        parallel_envs=2, dry_run=False, max_retries=1
    )
    first = orchestrator._execute(
        "retry_task",
        lambda run: ["python", "noop.py", run],
        artifact_builder=artifact_builder,
        required="summary.json",
    )
    assert first.name == "retry_task_r2"
    assert len(attempts) == 2
    resumed = orchestrator._execute(
        "retry_task",
        lambda run: ["python", "noop.py", run],
        artifact_builder=artifact_builder,
        required="summary.json",
    )
    assert resumed == first
    assert len(attempts) == 2
    state = json.loads(orchestrator.state_path.read_text(encoding="utf-8"))
    assert state["tasks"]["retry_task"]["status"] == "complete"
