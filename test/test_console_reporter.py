from __future__ import annotations

import pytest

from result.console_reporter import TrainingConsoleReporter, report_training_failure


def _config() -> dict:
    return {
        "experiment_name": "v8_universal",
        "seed": 11,
        "device": "cuda",
        "ppo": {"learning_rate": 1.0e-4},
        "training": {
            "two_stage": {
                "single_objective_promotion": {
                    "audit_instance_limit": 200,
                },
                "pareto_promotion": {
                    "validation_instance_limit": 50,
                    "audit_instance_limit": 200,
                    "preference_count": 66,
                },
            }
        },
    }


def _reporter(objective: str | None = "flow"):
    output: list[str] = []
    reporter = TrainingConsoleReporter(
        config=_config(),
        total_episodes=2000,
        parallel_envs=20,
        validation_instance_limit=50,
        objective_name=objective,
        started_at=0.0,
        writer=output.append,
    )
    return reporter, output


def _phase_state(objective: str = "flow") -> dict:
    return {
        "consecutive_validations_required": 3,
        "single_objective_name": objective,
        "single_objective_window_size": 5,
        "single_objective_window_values": [1050.0, 1047.4],
        "single_objective_candidate_anchor_value": 1048.7,
        "single_objective_candidate_episode": 500,
        "single_objective_audit_completion_target": 0.98,
        "single_objective_audit_max_failed_instances": 4,
        "accepted_single_objective_window_value": 1048.7,
        "accepted_single_objective_audit_value": 1041.8,
        "accepted_quality_episode": 500,
        "formal_training_status": "accepted_98_experiment_candidate",
        "is_formally_accepted": True,
    }


def _update() -> dict:
    return {
        "episode_end": 99,
        "update_id": 5,
        "policy_loss": -0.0182,
        "value_loss": 0.2914,
        "entropy": 1.421,
        "approx_kl": 0.0061,
        "learning_rate": 1.0e-4,
    }


def _episode_rows() -> list[dict]:
    return [
        {
            "terminated": True,
            "truncated": False,
            "reward": -0.8,
            "flow_time_objective": 1200.0,
            "reconfiguration_cost": 450.0,
            "worker_load_variance": 13.0,
        },
        {
            "terminated": False,
            "truncated": True,
            "reward": -1.0,
            "flow_time_objective": 1300.0,
            "reconfiguration_cost": 470.0,
            "worker_load_variance": 15.0,
        },
    ]


def _validation_row(phase: str = "quality") -> dict:
    return {
        "episode": 500,
        "instance_count": 50,
        "completion_rate": 1.0,
        "candidate_phase": phase,
        "consecutive_completion_successes": 3,
        "mean_flow_time_objective": 1036.2,
        "mean_reconfiguration_cost": 438.5,
        "mean_worker_load_variance": 12.61,
        "window_objective_statistic": 1048.7,
    }


def test_single_objective_run_and_training_update_are_compact():
    reporter, output = _reporter("flow")

    reporter.start_run()
    reporter.training_update(_update(), _episode_rows())

    text = "\n".join(output)
    assert "[RUN] single_flow | seed=11 | envs=20 | cuda" in text
    assert "episodes=2000 | val=50 | audit=200 | lr=1.0e-4" in text
    assert text.count("[完成率阶段] FEASIBILITY") == 1
    assert "[train] ep 100/2000 | upd 5" in text
    assert "complete 50.0% | reward -0.900" in text
    assert "flow 1250.000" in text
    assert "p_loss -0.0182 | v_loss 0.291 | ent 1.42 | kl 0.0061 | lr 1.0e-4" in text
    assert "reward_phase" not in text


@pytest.mark.parametrize(
    ("objective", "expected_metrics"),
    [
        ("flow", "flow 1036.200 | cost 438.500 | variance 12.610"),
        ("cost", "cost 438.500 | flow 1036.200 | variance 12.610"),
        ("variance", "variance 12.610 | flow 1036.200 | cost 438.500"),
    ],
)
def test_single_objective_validation_promotes_primary_metric_first(
    objective,
    expected_metrics,
):
    reporter, output = _reporter(objective)
    state = _phase_state(objective)

    reporter.validation(
        _validation_row(),
        status="candidate",
        phase_state=state,
    )

    text = "\n".join(output)
    assert expected_metrics in text
    assert "window 2/5" in text
    assert "window_med 1048.700 | best_med 1048.700 @ ep500" in text
    assert "status=candidate" in text


def test_single_objective_lifecycle_orders_phase_audit_acceptance_and_done():
    reporter, output = _reporter("flow")
    state = _phase_state()
    audit = {
        "episode": 500,
        "single_objective_target": "flow",
        "audit_instance_count": 200,
        "audit_failed_instance_count": 2,
        "audit_completion_rate": 0.99,
        "audit_schedule_violation_count": 0,
        "audit_physical_safety_pass": True,
        "audit_single_objective_value": 1041.8,
    }

    reporter.start_run()
    reporter.transition("phase1_checkpoint.pt")
    reporter.single_objective_audit(
        audit,
        status="ACCEPTED",
        phase_state=state,
    )
    reporter.accepted(
        episode=500,
        checkpoint="accepted_checkpoint.pt",
        phase_state=state,
    )
    reporter.done(
        phase_state=state,
        run_directory="result/runs/single_flow_seed11",
        accepted_checkpoint="accepted_checkpoint.pt",
        last_checkpoint="last_checkpoint.pt",
        elapsed_seconds=7980,
    )

    text = "\n".join(output)
    markers = [
        "[RUN]",
        "[完成率阶段]",
        "[PHASE]",
        "[质量阶段]",
        "[AUDIT]",
        "[ACCEPTED]",
        "[DONE]",
    ]
    offsets = [text.index(marker) for marker in markers]
    assert offsets == sorted(offsets)
    assert text.count("[质量阶段] QUALITY") == 1
    assert "complete 99.0% | failed 2/200" in text
    assert "target complete>=98.0% | failed<=4 | violations=0 | safety=PASS" in text
    assert "[ACCEPTED] flow=1041.800 | ep=500" in text
    assert "time=2h 13m" in text
    assert "best=1041.800 @ ep500" in text
    assert "verified=true" in text


def test_done_without_acceptance_names_last_checkpoint():
    reporter, output = _reporter("cost")
    state = {
        **_phase_state("cost"),
        "formal_training_status": "single_objective_98_candidate_not_reached",
        "is_formally_accepted": False,
    }

    reporter.done(
        phase_state=state,
        run_directory="result/runs/no_candidate",
        accepted_checkpoint=None,
        last_checkpoint="last_checkpoint.pt",
        elapsed_seconds=9,
    )

    text = "\n".join(output)
    assert "best=none" in text
    assert "checkpoint=none" in text
    assert "last=last_checkpoint.pt" in text


def test_pareto_validation_and_audit_show_grid_gates_and_intervals():
    reporter, output = _reporter(None)
    state = {
        "formal_training_status": "accepted_v8_pareto_candidate",
        "is_formally_accepted": True,
        "accepted_quality_episode": 700,
    }
    result = {
        "instance_count": 50,
        "preference_count": 66,
        "safety_gate_pass": True,
        "endpoint_gate_pass": True,
        "safety": {
            "minimum_completion_rate": 0.98,
            "failed_instance_count": 1,
            "schedule_violation_count": 0,
        },
        "endpoints": {
            "means": {"flow": 1010.0, "cost": 420.0, "variance": 10.5}
        },
        "primary_delta": {
            "estimate": -0.02,
            "lower": -0.03,
            "upper": -0.01,
        },
        "hypervolume_delta": {
            "estimate": 0.04,
            "lower": 0.01,
            "upper": 0.07,
        },
    }

    reporter.start_run()
    reporter.training_update(_update(), _episode_rows(), primary_value=0.314)
    reporter.validation(
        _validation_row(),
        status="candidate",
        phase_state=state,
        pareto_result=result,
    )
    audit_result = {**result, "instance_count": 200}
    reporter.pareto_audit(audit_result, episode=700, status="ACCEPTED")
    reporter.accepted(
        episode=700,
        checkpoint="accepted_checkpoint.pt",
        phase_state=state,
    )

    text = "\n".join(output)
    assert "[RUN] v8_universal" in text
    assert "val=50×66 | audit=200×66" in text
    assert "quality 0.314" in text
    assert "[val] ep 500 | n=50 × prefs=66" in text
    assert "flow 1010.000 | cost 420.000 | variance 10.500" in text
    assert "primary_delta -0.02 | 95% CI [-0.03, -0.01]" in text
    assert "hv_delta 0.04 | 95% CI [0.01, 0.07]" in text
    assert "[AUDIT] ep 700 | n=200 × prefs=66" in text
    assert "[ACCEPTED] pareto | ep=700" in text


def test_pareto_missing_optional_statistics_degrades_to_readable_output():
    reporter, output = _reporter(None)
    result = {
        "instance_count": 50,
        "preference_count": 66,
        "safety_gate_pass": False,
        "endpoint_gate_pass": False,
        "safety": {},
        "endpoints": {},
    }

    reporter.validation(
        _validation_row(),
        status="reject_candidate_gate_failure",
        phase_state={},
        pareto_result=result,
    )

    text = "\n".join(output)
    assert "min_complete n/a" in text
    assert "flow n/a | cost n/a | variance n/a" in text
    assert "gates safety=FAIL | endpoints=FAIL" in text
    assert "primary_delta" not in text


def test_failure_block_is_compact_and_includes_exit_code():
    output: list[str] = []

    report_training_failure(
        RuntimeError("diagnostic failure"),
        run_directory="result/runs/failed",
        exit_code=1,
        writer=output.append,
    )

    text = "\n".join(output)
    assert "[FAILED] RuntimeError" in text
    assert "message=diagnostic failure" in text
    assert "exit_code=1" in text
    assert "artifacts=result/runs/failed" in text
