from __future__ import annotations

from configs import load_config
from result.console_reporter import TrainingConsoleReporter, report_training_failure


def _reporter(objective: str | None = "flow"):
    output: list[str] = []
    reporter = TrainingConsoleReporter(
        config=load_config("configs/e1/single_flow.json" if objective else "configs/default.json"),
        total_episodes=2000,
        parallel_envs=20,
        validation_instance_limit=50,
        objective_name=objective,
        started_at=0.0,
        writer=output.append,
    )
    return reporter, output


def test_single_stage_run_and_training_update_are_compact():
    reporter, output = _reporter("flow")
    reporter.start_run()
    reporter.training_update(
        {
            "episode_end": 99,
            "update_id": 5,
            "policy_loss": -0.0182,
            "value_loss": 0.2914,
            "entropy": 1.421,
            "approx_kl": 0.0061,
            "learning_rate": 1e-4,
        },
        [
            {
                "task_succeeded": True,
                "reward": 0.6,
                "operation_progress": 1.0,
                "preference_quality_score": 0.4,
            },
            {
                "task_succeeded": False,
                "reward": -0.2,
                "operation_progress": 0.8,
                "preference_quality_score": 1.0,
            },
        ],
    )
    text = "\n".join(output)
    assert "[RUN] single_flow | seed=11 | envs=20" in text
    assert "val=50 × repeats=3 × prefs=1 | sampled T=1.0" in text
    assert "reward=ΔP-ΔQ | gamma=1.0" in text
    assert "complete 50.0% | reward 0.200 | progress 0.900" in text
    assert "phase" not in text.lower()
    assert "audit" not in text.lower()


def test_validation_reports_sampled_rank():
    reporter, output = _reporter(None)
    row = {
        "episode": 100,
        "completion_rate": 0.95,
        "preference_balanced_quality_score": 0.31,
        "mean_operation_progress": 0.98,
        "physical_safety_pass": True,
        "checkpoint_event": "best_improved",
    }
    reporter.validation(row, selector_state={"best_episode": 100})
    text = "\n".join(output)
    assert "sampled complete 95.0% | quality 0.310000" in text
    assert "safety=PASS" in text
    assert "greedy" not in text.lower()
    assert "event=best_improved | best_ep=100" in text


def test_done_never_substitutes_last_when_no_safe_best():
    reporter, output = _reporter("flow")
    reporter.done(
        selector_state={"has_best": False},
        run_directory="result/runs/no_best",
        best_checkpoint=None,
        last_checkpoint="last_checkpoint.pt",
        elapsed_seconds=9,
    )
    text = "\n".join(output)
    assert "status=no_safe_best" in text
    assert "best=none" in text
    assert "checkpoint=none" in text
    assert "last=last_checkpoint.pt" in text


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
