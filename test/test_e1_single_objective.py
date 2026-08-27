from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from agent.baselines import HeuristicPolicy
from configs import load_config
from configs.config import public_config
from environment import AssemblySchedulingEnv, proxy_return_from_metrics
from result.io import write_csv, write_json
from single_objective_analysis import OBJECTIVE_FIELDS, main as analysis_main
from train import (
    SINGLE_OBJECTIVE_PROMOTION_MODE,
    TrainingPhaseController,
    ValidationStabilityController,
    _checkpoint_eligible_validation_event,
    _single_objective_failure_rows,
    _single_objective_guard_score,
    _validate_single_objective_validation_protocol,
)


CONFIGS = {
    "flow": "configs/v7/e1_single_flow.json",
    "cost": "configs/v7/e1_single_cost.json",
    "variance": "configs/v7/e1_single_variance.json",
}


def _transitioned_controller(objective: str) -> TrainingPhaseController:
    config = load_config(CONFIGS[objective])
    config["training"]["two_stage"]["consecutive_validations"] = 1
    controller = TrainingPhaseController.from_config(config)
    score = (-1.0, 100.0, 0.0, 0.0)
    assert controller.observe_validation(
        1.0,
        completed_episodes=10,
        score=score,
        truncated_count=0,
        schedule_violation_count=0,
    ) == "transition"
    return controller


def _validation(flow: float, cost: float, variance: float) -> dict[str, object]:
    return {
        "completion_rate": 1.0,
        "truncated_count": 0,
        "schedule_violation_count": 0,
        "mean_flow_time_objective": flow,
        "mean_reconfiguration_cost": cost,
        "mean_worker_load_variance": variance,
        "all_instance_metrics": {
            "flow_time_objective": {"mean": flow},
            "reconfiguration_cost": {"mean": cost},
            "worker_load_variance": {"mean": variance},
        },
    }


def _raw_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_single_objective_base_changes_only_requested_e1_protocol_fields():
    e1 = load_config("configs/v7/e1_context_exception.json")
    base = load_config("configs/v7/e1_single_objective_base.json")

    expected = public_config(e1)
    expected["experiment_name"] = "v7_e1_single_objective"
    expected["experiment_suite_version"] = (
        "v7_e1_single_objective_protocol_v1"
    )
    expected["environment"]["worker_resource_control"]["mode"] = (
        "matching_admission_recovery_v2"
    )
    expected["environment"]["production_defer"]["shield"] = {
        "enabled": True,
        "version": "deadline_progress_viability_shield_v2",
        "deadline_reserve_ticks": 1,
        "soft_risk_threshold": 0.8,
        "soft_risk_coefficient": 0.0,
    }
    expected["training"]["two_stage"][
        "quality_checkpoint_promotion"
    ] = SINGLE_OBJECTIVE_PROMOTION_MODE
    expected["training"]["validation_instance_limit"] = 500
    expected["training"]["two_stage"]["quality_completion_floor"] = 0.95
    expected["training"]["two_stage"]["quality_promotion_constraints"] = None
    expected["training"]["two_stage"]["single_objective_promotion"] = {
        "window_size": 5,
        "window_statistic": "median",
        "rollback_below_floor_consecutive": 2,
        "formal_completion_target": 1.0,
        "formal_truncation_target": 0,
        "formal_violation_target": 0,
        "physical_safety_required": True,
    }

    assert public_config(base) == expected
    assert base["network"] == e1["network"]
    assert base["ppo"] == e1["ppo"]
    assert base["reward"] == e1["reward"]
    assert "preference_adapter" not in base["network"]
    assert "preference_stage_schedule" not in base["training"]
    assert "warm_start" not in base["training"]
    assert "pareto_promotion" not in base["training"]["two_stage"]


@pytest.mark.parametrize("objective", tuple(CONFIGS))
def test_child_config_only_changes_strict_one_hot_weights(objective: str):
    base = load_config("configs/v7/e1_single_objective_base.json")
    child = load_config(CONFIGS[objective])
    expected_weights = {
        name: 1.0 if name == objective else 0.0 for name in CONFIGS
    }
    expected = public_config(base)
    expected["reward"]["quality_weights"] = expected_weights
    assert public_config(child) == expected

    raw = _raw_json(CONFIGS[objective])
    assert set(raw) == {"extends", "reward"}
    assert raw["reward"] == {"quality_weights": expected_weights}
    shield = child["environment"]["production_defer"]["shield"]
    assert shield["enabled"] is True
    assert shield["soft_risk_coefficient"] == 0.0


@pytest.mark.parametrize("objective", tuple(CONFIGS))
def test_each_promotion_mode_uses_only_its_raw_objective(objective: str):
    controller = _transitioned_controller(objective)
    assert controller.accepted_single_objective_value is None
    assert controller.accepted_quality_episode is None
    assert not _checkpoint_eligible_validation_event(
        "transition", SINGLE_OBJECTIVE_PROMOTION_MODE
    )

    anchor_validation = _validation(100.0, 20.0, 3.0)
    anchor_score = _single_objective_guard_score(anchor_validation, objective)
    events = [
        controller.observe_validation(
            1.0, completed_episodes=20 + index, score=anchor_score
        )
        for index in range(5)
    ]
    assert events[:4] == ["window_warmup"] * 4
    assert events[-1] == "formal_promoted"
    anchor_value = anchor_validation[OBJECTIVE_FIELDS[objective]]
    assert controller.accepted_single_objective_value == anchor_value
    assert controller.formal_single_objective_window_value == anchor_value

    other_names = [name for name in CONFIGS if name != objective]
    non_target_improvement = dict(anchor_validation)
    for name in other_names:
        non_target_improvement[OBJECTIVE_FIELDS[name]] *= 0.1
    non_target_improvement["all_instance_metrics"] = {
        "flow_time_objective": {
            "mean": non_target_improvement["mean_flow_time_objective"]
        },
        "reconfiguration_cost": {
            "mean": non_target_improvement["mean_reconfiguration_cost"]
        },
        "worker_load_variance": {
            "mean": non_target_improvement["mean_worker_load_variance"]
        },
    }
    unchanged_target_score = _single_objective_guard_score(
        non_target_improvement, objective
    )
    unchanged_events = [
        controller.observe_validation(
            1.0, completed_episodes=30 + index, score=unchanged_target_score
        )
        for index in range(5)
    ]
    assert "formal_promoted" not in unchanged_events
    assert controller.accepted_single_objective_value == anchor_value

    target_improvement = dict(anchor_validation)
    target_improvement[OBJECTIVE_FIELDS[objective]] = anchor_value - 0.25
    target_improvement["all_instance_metrics"] = {
        "flow_time_objective": {
            "mean": target_improvement["mean_flow_time_objective"]
        },
        "reconfiguration_cost": {
            "mean": target_improvement["mean_reconfiguration_cost"]
        },
        "worker_load_variance": {
            "mean": target_improvement["mean_worker_load_variance"]
        },
    }
    improved_score = _single_objective_guard_score(
        target_improvement, objective
    )
    improved_events = [
        controller.observe_validation(
            1.0, completed_episodes=40 + index, score=improved_score
        )
        for index in range(5)
    ]
    assert "formal_promoted" in improved_events
    assert controller.accepted_single_objective_value == anchor_value - 0.25
    assert controller.last_promotion_diagnostics["formal_anchor_value"] == (
        anchor_value - 0.25
    )


def test_95_percent_candidates_are_exploratory_only_and_window_warms_up():
    controller = _transitioned_controller("flow")
    events = [
        controller.observe_validation(
            0.95,
            completed_episodes=20 + index,
            score=(-0.95, 99.0, 0.0, 0.0),
            truncated_count=1,
            schedule_violation_count=0,
        )
        for index in range(5)
    ]
    assert events[:4] == ["window_warmup"] * 4
    assert events[-1] == "exploratory_promoted"
    assert controller.accepted_single_objective_value is None
    assert controller.exploratory_single_objective_value == 99.0
    assert controller.last_promotion_diagnostics["window_count"] == 5
    assert not _checkpoint_eligible_validation_event(
        "exploratory_promoted", SINGLE_OBJECTIVE_PROMOTION_MODE
    )


def test_single_objective_rejects_only_exploration_gate_failures():
    controller = _transitioned_controller("flow")
    for completion, violations, physical, reason in (
        (0.949, 0, True, "completion_below_floor"),
        (0.95, 1, True, "schedule_violation_nonzero"),
        (0.95, 0, False, "physical_safety_failed"),
    ):
        event = controller.observe_validation(
            completion,
            completed_episodes=20,
            score=(-completion, 99.0, 0.0, 0.0),
            schedule_violation_count=violations,
            physical_safety_pass=physical,
        )
        assert event == "rejected"
        assert controller.last_promotion_diagnostics[
            "promotion_decision_reason"
        ] == reason
        assert controller.last_promotion_diagnostics["window_count"] == 0


def test_single_objective_rejects_non_one_hot_weights_immediately():
    config = load_config(CONFIGS["flow"])
    config["reward"]["quality_weights"] = {
        "flow": 0.5,
        "cost": 0.5,
        "variance": 0.0,
    }
    with pytest.raises(ValueError, match="strictly one-hot"):
        TrainingPhaseController.from_config(config)


def test_phase_one_keeps_the_original_three_consecutive_100_percent_gate():
    controller = TrainingPhaseController.from_config(
        load_config(CONFIGS["flow"])
    )
    score = (-1.0, 100.0, 0.0, 0.0)
    assert controller.observe_validation(1.0, completed_episodes=10, score=score) == "feasibility"
    assert controller.observe_validation(0.99, completed_episodes=20, score=score) == "feasibility"
    assert controller.consecutive_successes == 0
    assert controller.observe_validation(1.0, completed_episodes=30, score=score) == "feasibility"
    assert controller.observe_validation(1.0, completed_episodes=40, score=score) == "feasibility"
    assert controller.observe_validation(1.0, completed_episodes=50, score=score) == "transition"


def test_serial_and_parallel_promotion_paths_share_the_same_decisions():
    serial = _transitioned_controller("variance")
    parallel = _transitioned_controller("variance")
    validations = [
        _validation(100.0, 20.0, 3.0),
        _validation(100.0, 20.0, 3.0),
        _validation(100.0, 20.0, 3.0),
        _validation(100.0, 20.0, 3.0),
        _validation(1000.0, 200.0, 2.5),
    ]
    serial_events = []
    parallel_events = []
    for index, validation in enumerate(validations, start=2):
        score = _single_objective_guard_score(validation, "variance")
        arguments = {
            "completed_episodes": index * 10,
            "score": score,
            "truncated_count": 0,
            "schedule_violation_count": 0,
        }
        serial_events.append(serial.observe_validation(1.0, **arguments))
        parallel_events.append(parallel.observe_validation(1.0, **arguments))
    assert serial_events == parallel_events
    assert serial_events[:4] == ["window_warmup"] * 4
    assert serial_events[-1] == "formal_promoted"
    assert serial.as_dict() == parallel.as_dict()


def test_individual_improvement_does_not_promote_until_window_median_improves():
    controller = _transitioned_controller("flow")
    baseline = (-1.0, 100.0, 0.0, 0.0)
    for index in range(5):
        controller.observe_validation(
            0.95, completed_episodes=20 + index, score=baseline
        )
    assert controller.exploratory_single_objective_window_value == 100.0
    assert controller.observe_validation(
        0.95, completed_episodes=30, score=(-0.95, 50.0, 0.0, 0.0)
    ) == "not_promoted"
    assert controller.exploratory_single_objective_window_value == 100.0
    assert controller.observe_validation(
        0.95, completed_episodes=31, score=(-0.95, 50.0, 0.0, 0.0)
    ) == "not_promoted"
    assert controller.observe_validation(
        0.95, completed_episodes=32, score=(-0.95, 50.0, 0.0, 0.0)
    ) == "exploratory_promoted"
    assert controller.exploratory_single_objective_window_value == 50.0


def test_formal_promotion_requires_current_100_percent_candidate():
    controller = _transitioned_controller("flow")
    for index in range(5):
        controller.observe_validation(
            0.95,
            completed_episodes=20 + index,
            score=(-0.95, 100.0, 0.0, 0.0),
        )
    assert controller.accepted_single_objective_value is None
    for index in range(3):
        event = controller.observe_validation(
            1.0,
            completed_episodes=30 + index,
            score=(-1.0, 80.0, 0.0, 0.0),
        )
    assert event == "formal_promoted"
    assert controller.accepted_single_objective_value == 80.0
    assert _checkpoint_eligible_validation_event(
        "formal_promoted", SINGLE_OBJECTIVE_PROMOTION_MODE
    )


@pytest.mark.parametrize(
    ("truncated_count", "violations", "physical_safety_pass"),
    [(1, 0, True), (0, 1, True), (0, 0, False)],
)
def test_formal_track_rejects_failed_hard_gates(
    truncated_count: int, violations: int, physical_safety_pass: bool
):
    controller = _transitioned_controller("flow")
    for index in range(4):
        assert controller.observe_validation(
            0.95,
            completed_episodes=20 + index,
            score=(-0.95, 100.0, 0.0, 0.0),
        ) == "window_warmup"
    event = controller.observe_validation(
        1.0,
        completed_episodes=25,
        score=(-1.0, 90.0, 0.0, 0.0),
        truncated_count=truncated_count,
        schedule_violation_count=violations,
        physical_safety_pass=physical_safety_pass,
    )
    assert event != "formal_promoted"
    assert controller.accepted_single_objective_value is None


def test_single_objective_rollback_is_strict_below_95_and_requires_two():
    config = load_config(CONFIGS["flow"])
    controller = ValidationStabilityController.from_config(config)
    controller.observe_greedy(
        (-1.0, 100.0, 0.0, 0.0), 1.0,
        completed_episodes=10, feasibility_phase=False,
    )
    exact_floor = controller.observe_greedy(
        (-0.95, 101.0, 0.0, 0.0), 0.95,
        completed_episodes=20, feasibility_phase=False,
    )
    assert not exact_floor["degraded"]
    first_below = controller.observe_greedy(
        (-0.949, 102.0, 0.0, 0.0), 0.949,
        completed_episodes=30, feasibility_phase=False,
    )
    assert first_below["degraded"] and not first_below["rollback"]
    second_below = controller.observe_greedy(
        (-0.948, 103.0, 0.0, 0.0), 0.948,
        completed_episodes=40, feasibility_phase=False,
    )
    assert second_below["rollback"]
    assert controller.rollback_consecutive_required == 2


def test_failure_detail_rows_record_the_required_tail_diagnostics():
    failures = _single_objective_failure_rows(
        [
            {
                "instance_id": "ok",
                "terminated": True,
                "truncated": False,
                "schedule_violation_count": 0,
                "unfinished_orders": 0,
                "maximum_worker_fatigue": 1.0,
                "safe_fatigue_limit": 1.0,
            },
            {
                "instance_id": "tail",
                "terminated": False,
                "truncated": True,
                "schedule_violation_count": 1,
                "unfinished_orders": 2,
                "maximum_worker_fatigue": 1.2,
                "safe_fatigue_limit": 1.0,
            },
        ],
        episode=42,
    )
    assert failures == [
        {
            "episode": 42,
            "instance_id": "tail",
            "truncated": True,
            "schedule_violation_count": 1,
            "unfinished_orders": 2,
            "maximum_worker_fatigue": 1.2,
            "safe_fatigue_limit": 1.0,
            "failure_reason": "incomplete;truncated;schedule_violation;physical_safety",
        }
    ]


def test_formal_run_requires_a_500_instance_validation_manifest(tmp_path: Path):
    config = load_config(CONFIGS["flow"])
    config["paths"]["manifests_root"] = str(tmp_path / "manifests")
    with pytest.raises(FileNotFoundError, match="validation manifest"):
        _validate_single_objective_validation_protocol(
            config, smoke=False, validation_limit=500
        )
    manifest_path = tmp_path / "manifests" / "validation" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    write_json(manifest_path, {"instance_count": 20, "files": []})
    with pytest.raises(ValueError, match="500-instance manifest"):
        _validate_single_objective_validation_protocol(
            config, smoke=False, validation_limit=500
        )
    write_json(manifest_path, {"instance_count": 500, "files": [None] * 500})
    _validate_single_objective_validation_protocol(
        config, smoke=False, validation_limit=500
    )


@pytest.mark.parametrize("objective", tuple(CONFIGS))
def test_one_hot_quality_reward_identity(
    objective: str,
    fixed_instance,
):
    config = load_config(CONFIGS[objective])
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    assert environment.matching_admission_enabled
    assert environment.matching_recovery_enabled
    assert environment.completion_viability_shield_enabled
    policy = HeuristicPolicy()
    base_reward_sum = 0.0
    while not (environment.terminated or environment.truncated):
        action = policy.select_action(environment)
        _, reward, _, _, _ = environment.step(action)
        base_reward_sum += reward.base_scalarize(config["reward"], "quality")
    metrics = environment.metrics()
    assert metrics["production_defer_shield_candidate_count"] > 0
    assert metrics["future_installation_admission_candidate_count"] > 0
    assert base_reward_sum == pytest.approx(
        proxy_return_from_metrics(metrics, config["reward"], "quality"),
        abs=1e-8,
    )


def test_convergence_entry_writes_five_panel_artifacts(tmp_path: Path):
    run_directory = tmp_path / "flow_run"
    output_directory = tmp_path / "analysis"
    run_directory.mkdir()
    config = load_config(CONFIGS["flow"])
    write_json(run_directory / "config.json", public_config(config))
    write_json(
        run_directory / "summary.json",
        {
            "training_phase": {
                "phase_transition_episode": 20,
                "accepted_quality_episode": 40,
            }
        },
    )
    rows = []
    for episode, flow in ((10, 120.0), (20, 110.0), (30, 100.0), (40, 99.0)):
        rows.append(
            {
                "episode": episode,
                "completion_rate": 1.0,
                "truncated_count": 0,
                "schedule_violation_count": 0,
                "mean_flow_time_objective": flow,
                "mean_reconfiguration_cost": 20.0 + episode,
                "mean_worker_load_variance": 3.0,
                "candidate_phase": (
                    "feasibility" if episode <= 20 else "quality"
                ),
                "phase_after_validation": (
                    "quality" if episode >= 20 else "feasibility"
                ),
                "validation_event": (
                    "transition"
                    if episode == 20
                    else "promoted"
                    if episode == 40
                    else "feasibility"
                ),
            }
        )
    write_csv(run_directory / "validation_log.csv", rows)

    assert analysis_main(
        [
            "--flow-run",
            str(run_directory),
            "--output-dir",
            str(output_directory),
        ]
    ) == 0
    for name in (
        "flow_convergence_data.csv",
        "flow_convergence.pdf",
        "flow_convergence.png",
        "convergence_diagnostics.json",
        "convergence_report.md",
    ):
        path = output_directory / name
        assert path.is_file()
        assert path.stat().st_size > 0
    with (output_directory / "flow_convergence_data.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        plotted = list(csv.DictReader(handle))
    assert [int(row["completed_episodes"]) for row in plotted] == [
        10,
        20,
        30,
        40,
    ]
