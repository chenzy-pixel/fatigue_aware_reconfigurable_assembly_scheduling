from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

import train as train_module
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
    _promote_accepted_checkpoint,
    _reevaluate_checkpoint_from_disk,
    _single_objective_failure_rows,
    _single_objective_guard_score,
    _validate_single_objective_validation_protocol,
)


CONFIGS = {
    "flow": "configs/e1/single_flow.json",
    "cost": "configs/e1/single_cost.json",
    "variance": "configs/e1/single_variance.json",
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
    assert controller.is_formally_accepted is False
    assert (
        controller.formal_training_status
        == "single_objective_98_candidate_not_reached"
    )
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


def _audit(*, objective_value: float, failed: int = 0, violation: int = 0,
           physical: bool = True) -> dict[str, object]:
    completed = 200 - failed
    return {
        "instance_count": 200,
        "completed_count": completed,
        "completion_rate": completed / 200.0,
        "truncated_count": failed,
        "schedule_violation_count": violation,
        "physical_safety_pass": physical,
        "single_objective_value": objective_value,
    }


def _raw_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_default_is_the_complete_latest_single_objective_protocol():
    base = load_config("configs/default.json")
    assert base["experiment_suite_version"] == "v7_e1_single_objective_protocol_v5"
    assert base["training"]["two_stage"]["quality_checkpoint_promotion"] == SINGLE_OBJECTIVE_PROMOTION_MODE
    assert base["runtime_manifest"]["candidate_ranker"] == "bounded_ranker_scale_v7"
    assert base["runtime_manifest"]["worker_feasibility"] == "instant_physical_pair_mask_v1"
    assert base["runtime_manifest"]["wait_mask"] == "progress_certified_wait_v2"
    assert base["runtime_manifest"]["observation_schema"] == 4
    assert set(base["environment"]) == {
        "max_decisions",
        "max_zero_time_actions",
    }


@pytest.mark.parametrize("objective", tuple(CONFIGS))
def test_child_config_only_changes_strict_one_hot_weights(objective: str):
    base = load_config("configs/default.json")
    child = load_config(CONFIGS[objective])
    expected_weights = {
        name: 1.0 if name == objective else 0.0 for name in CONFIGS
    }
    expected = public_config(base)
    expected["reward"]["quality_weights"] = expected_weights
    expected["experiment_name"] = f"e1_single_{objective}"
    assert public_config(child) == expected

    raw = _raw_json(CONFIGS[objective])
    assert set(raw) == {"extends", "experiment_name", "reward"}
    assert raw["reward"] == {"quality_weights": expected_weights}


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
    assert events[-1] == "audit_required"
    anchor_value = anchor_validation[OBJECTIVE_FIELDS[objective]]
    assert controller.accepted_single_objective_value is None
    assert controller.single_objective_candidate_anchor_value == anchor_value
    assert controller.observe_single_objective_audit(
        _audit(objective_value=anchor_value),
        completed_episodes=24,
        window_median=anchor_value,
    ) == "accepted"
    assert controller.accepted_single_objective_value == anchor_value
    assert controller.is_formally_accepted is True
    assert controller.formal_training_status == "accepted_98_experiment_candidate"

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
    assert "audit_required" not in unchanged_events
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
    assert "audit_required" in improved_events
    assert controller.observe_single_objective_audit(
        _audit(objective_value=anchor_value - 0.25),
        completed_episodes=44,
        window_median=anchor_value - 0.25,
    ) == "accepted"
    assert controller.accepted_single_objective_value == anchor_value - 0.25


def test_only_audited_acceptance_persists_accepted_and_best_checkpoints(
    tmp_path: Path,
):
    config = load_config(CONFIGS["flow"])
    controller = _transitioned_controller("flow")
    accepted_checkpoint = tmp_path / "accepted_checkpoint.pt"
    best_checkpoint = tmp_path / "best_checkpoint.pt"

    class RecordingAgent:
        def __init__(self) -> None:
            self.saved_metadata: dict | None = None

        def save(self, path: Path, metadata: dict | None = None) -> None:
            self.saved_metadata = dict(metadata or {})
            Path(path).write_bytes(b"accepted-candidate-a")

    agent = RecordingAgent()
    for event in (
        "transition",
        "audit_required",
        "audit_rejected",
        "audit_passed_not_accepted",
    ):
        assert not _promote_accepted_checkpoint(
            event=event,
            config=config,
            phase_controller=controller,
            agent=agent,
            accepted_checkpoint=accepted_checkpoint,
            best_checkpoint=best_checkpoint,
            completed_episodes=10,
            parallel_envs=1,
            validation_row={"validation_event": event},
        )
    assert not accepted_checkpoint.exists()
    assert not best_checkpoint.exists()
    assert agent.saved_metadata is None

    score = (-1.0, 80.0, 0.0, 0.0)
    for episode in range(20, 25):
        event = controller.observe_validation(
            1.0,
            completed_episodes=episode,
            score=score,
        )
    assert event == "audit_required"
    assert controller.observe_single_objective_audit(
        _audit(objective_value=80.0),
        completed_episodes=24,
        window_median=80.0,
    ) == "accepted"

    assert _promote_accepted_checkpoint(
        event="accepted",
        config=config,
        phase_controller=controller,
        agent=agent,
        accepted_checkpoint=accepted_checkpoint,
        best_checkpoint=best_checkpoint,
        completed_episodes=24,
        parallel_envs=1,
        validation_row={"validation_event": "accepted"},
    )
    assert accepted_checkpoint.read_bytes() == best_checkpoint.read_bytes()
    assert agent.saved_metadata is not None
    assert agent.saved_metadata["checkpoint_role"] == "accepted"
    assert agent.saved_metadata["single_objective_name"] == "flow"
    assert agent.saved_metadata["single_objective_window_statistic"] == "median"
    assert agent.saved_metadata["accepted_single_objective_value"] == 80.0
    assert agent.saved_metadata["accepted_single_objective_audit_value"] == 80.0
    assert agent.saved_metadata["formal_eligible"] is True
    assert "quality_score" not in agent.saved_metadata


def test_final_evaluation_loads_accepted_checkpoint_instead_of_online_agent(
    config,
    monkeypatch,
    tmp_path: Path,
):
    accepted_checkpoint = tmp_path / "accepted_checkpoint.pt"
    accepted_checkpoint.write_text("candidate-a", encoding="utf-8")
    online_agent = type("OnlineAgent", (), {"identity": "candidate-b"})()
    loaded_paths: list[Path] = []

    class IsolatedEvaluationAgent:
        def __init__(self, *_args, **_kwargs) -> None:
            self.identity = "unloaded"

        def load(self, path: Path, *, load_optimizer: bool) -> dict:
            assert load_optimizer is False
            loaded_paths.append(Path(path))
            self.identity = Path(path).read_text(encoding="utf-8")
            return {"loaded_identity": self.identity}

    def evaluate_loaded_agent(*_args, ppo_agent, **_kwargs):
        rows = [
            {
                "schedule_violation_count": 0,
                "maximum_worker_fatigue": 0.0,
                "safe_fatigue_limit": 1.0,
            }
        ]
        return rows, None, None, {
            "parallel_envs": 1,
            "evaluated_identity": ppo_agent.identity,
        }

    monkeypatch.setattr(train_module, "PPOAgent", IsolatedEvaluationAgent)
    monkeypatch.setattr(train_module, "build_actor_critic", lambda *_args: object())
    monkeypatch.setattr(train_module, "evaluate_dataset", evaluate_loaded_agent)
    monkeypatch.setattr(train_module, "build_provenance", lambda *_args, **_kwargs: {})

    evaluation = _reevaluate_checkpoint_from_disk(
        config,
        checkpoint=accepted_checkpoint,
        bootstrap_observation=object(),
        dataset_name="validation",
        instance_limit=200,
        sampling_seeds=[],
        greedy_only=True,
    )

    assert online_agent.identity == "candidate-b"
    assert loaded_paths == [accepted_checkpoint]
    assert evaluation["checkpoint"] == str(accepted_checkpoint)
    assert evaluation["checkpoint_metadata"] == {
        "loaded_identity": "candidate-a"
    }
    assert evaluation["greedy"]["evaluated_identity"] == "candidate-a"


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
    assert events[-1] == "audit_required"
    assert controller.accepted_single_objective_value is None
    assert controller.single_objective_candidate_anchor_value == 99.0
    assert controller.last_promotion_diagnostics["window_count"] == 5
    assert not _checkpoint_eligible_validation_event(
        "audit_required", SINGLE_OBJECTIVE_PROMOTION_MODE
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


def test_rejected_formal_audit_does_not_set_formal_acceptance():
    controller = _transitioned_controller("flow")
    for episode in range(20, 25):
        event = controller.observe_validation(
            1.0,
            completed_episodes=episode,
            score=(-1.0, 90.0, 0.0, 0.0),
        )
    assert event == "audit_required"
    assert controller.observe_single_objective_audit(
        _audit(objective_value=90.0, physical=False),
        completed_episodes=24,
        window_median=90.0,
    ) == "audit_rejected"
    assert controller.is_formally_accepted is False
    assert (
        controller.formal_training_status
        == "single_objective_98_candidate_not_reached"
    )


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
    assert serial_events[-1] == "audit_required"
    assert serial.observe_single_objective_audit(
        _audit(objective_value=2.5),
        completed_episodes=50,
        window_median=2.5,
    ) == "accepted"
    assert parallel.observe_single_objective_audit(
        _audit(objective_value=2.5),
        completed_episodes=50,
        window_median=2.5,
    ) == "accepted"
    assert serial.as_dict() == parallel.as_dict()


def test_individual_improvement_does_not_promote_until_window_median_improves():
    controller = _transitioned_controller("flow")
    baseline = (-1.0, 100.0, 0.0, 0.0)
    for index in range(5):
        controller.observe_validation(
            0.95, completed_episodes=20 + index, score=baseline
        )
    assert controller.single_objective_candidate_anchor_value == 100.0
    assert controller.observe_validation(
        0.95, completed_episodes=30, score=(-0.95, 50.0, 0.0, 0.0)
    ) == "not_promoted"
    assert controller.single_objective_candidate_anchor_value == 100.0
    assert controller.observe_validation(
        0.95, completed_episodes=31, score=(-0.95, 50.0, 0.0, 0.0)
    ) == "not_promoted"
    assert controller.observe_validation(
        0.95, completed_episodes=32, score=(-0.95, 50.0, 0.0, 0.0)
    ) == "audit_required"
    assert controller.single_objective_candidate_anchor_value == 50.0


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
    assert event == "audit_required"
    assert controller.observe_single_objective_audit(
        _audit(objective_value=80.0),
        completed_episodes=32,
        window_median=80.0,
    ) == "accepted"
    assert controller.accepted_single_objective_value == 80.0
    assert _checkpoint_eligible_validation_event(
        "accepted", SINGLE_OBJECTIVE_PROMOTION_MODE
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
    if violations or not physical_safety_pass:
        assert event == "rejected"
    else:
        assert event == "audit_required"
        audit_failed = controller.observe_single_objective_audit(
            _audit(objective_value=90.0, failed=1 if truncated_count else 0),
            completed_episodes=25,
            window_median=90.0,
        )
        assert audit_failed == "accepted"
    if violations or not physical_safety_pass:
        assert controller.accepted_single_objective_value is None
    else:
        assert controller.accepted_single_objective_value == 90.0


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


def test_formal_run_requires_at_least_the_200_audit_instances(tmp_path: Path):
    config = load_config(CONFIGS["flow"])
    config["paths"]["manifests_root"] = str(tmp_path / "manifests")
    with pytest.raises(FileNotFoundError, match="validation manifest"):
        _validate_single_objective_validation_protocol(
            config, smoke=False, validation_limit=50
        )
    manifest_path = tmp_path / "manifests" / "validation" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    write_json(
        manifest_path,
        {
            "generator_version": config["generator"]["version"],
            "instance_count": 20,
            "files": [],
        },
    )
    with pytest.raises(ValueError, match="at least 200 instances"):
        _validate_single_objective_validation_protocol(
            config, smoke=False, validation_limit=50
        )
    write_json(
        manifest_path,
        {
            "generator_version": "0.0.0",
            "instance_count": 200,
            "files": [None] * 200,
        },
    )
    with pytest.raises(ValueError, match="stale generator fingerprint"):
        _validate_single_objective_validation_protocol(
            config, smoke=False, validation_limit=50
        )
    write_json(
        manifest_path,
        {
            "generator_version": config["generator"]["version"],
            "instance_count": 200,
            "files": [None] * 200,
        },
    )
    _validate_single_objective_validation_protocol(
        config, smoke=False, validation_limit=50
    )
    write_json(
        manifest_path,
        {
            "generator_version": config["generator"]["version"],
            "instance_count": 500,
            "files": [None] * 500,
        },
    )
    _validate_single_objective_validation_protocol(
        config, smoke=False, validation_limit=50
    )


@pytest.mark.parametrize("objective", tuple(CONFIGS))
def test_one_hot_quality_reward_identity(
    objective: str,
    fixed_instance,
):
    config = load_config(CONFIGS[objective])
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    policy = HeuristicPolicy()
    base_reward_sum = 0.0
    while not (environment.terminated or environment.truncated):
        action = policy.select_action(environment)
        _, reward, _, _, _ = environment.step(action)
        base_reward_sum += reward.base_scalarize(config["reward"], "quality")
    metrics = environment.metrics()
    assert metrics["wait_action_count"] > 0
    assert metrics["wait_total_ticks"] >= 0
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
                    else "accepted"
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
