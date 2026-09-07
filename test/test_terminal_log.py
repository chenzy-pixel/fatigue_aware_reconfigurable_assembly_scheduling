from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import train as training_module
from result.terminal_log import capture_terminal_output


def test_terminal_capture_tees_stdout_stderr_and_restores_streams(
    tmp_path,
    monkeypatch,
):
    visible_stdout = io.StringIO()
    visible_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", visible_stdout)
    monkeypatch.setattr(sys, "stderr", visible_stderr)
    log_path = tmp_path / "terminal.log"

    with capture_terminal_output(log_path):
        print("stdout-line")
        print("stderr-line", file=sys.stderr)

    assert sys.stdout is visible_stdout
    assert sys.stderr is visible_stderr
    assert visible_stdout.getvalue() == "stdout-line\n"
    assert visible_stderr.getvalue() == "stderr-line\n"
    assert log_path.read_text(encoding="utf-8") == (
        "stdout-line\nstderr-line\n"
    )


def _main_config(result_root: Path) -> dict:
    return {
        "paths": {"result_root": str(result_root)},
        "training": {"episodes": 2000},
    }


def test_original_train_command_automatically_saves_terminal_log(
    tmp_path,
    monkeypatch,
):
    result_root = tmp_path / "runs"
    run_name = "automatic_terminal_log"

    def fake_train(config, **kwargs):
        run_directory = result_root / kwargs["run_name"]
        run_directory.mkdir(parents=True)
        print("training-progress")
        print("training-warning", file=sys.stderr)
        return run_directory

    monkeypatch.setattr(
        training_module,
        "load_config",
        lambda path: _main_config(result_root),
    )
    monkeypatch.setattr(training_module, "train", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/v7/e1_single_flow.json",
            "--algorithm-seed",
            "11",
            "--parallel-envs",
            "20",
            "--run-name",
            run_name,
        ],
    )

    assert training_module.main() == 0

    log_path = result_root / run_name / "terminal.log"
    log = log_path.read_text(encoding="utf-8")
    assert "command=" in log
    assert "training-progress" in log
    assert "training-warning" in log
    assert f"training artifacts: {result_root / run_name}" in log
    assert "exit_code=0" in log
    assert not list(result_root.glob("*.tmp"))


def test_training_exception_traceback_is_saved_inside_created_run(
    tmp_path,
    monkeypatch,
):
    result_root = tmp_path / "runs"
    run_name = "failed_terminal_log"

    def failing_train(config, **kwargs):
        run_directory = result_root / kwargs["run_name"]
        run_directory.mkdir(parents=True)
        print("before-failure")
        raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(
        training_module,
        "load_config",
        lambda path: _main_config(result_root),
    )
    monkeypatch.setattr(training_module, "train", failing_train)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--run-name", run_name],
    )

    assert training_module.main() == 1

    log = (result_root / run_name / "terminal.log").read_text(
        encoding="utf-8"
    )
    assert "before-failure" in log
    assert "Traceback (most recent call last)" in log
    assert "RuntimeError: diagnostic failure" in log
    assert "exit_code=1" in log
    failure = json.loads(
        (result_root / run_name / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["exception_type"] == "RuntimeError"
    assert failure["message"] == "diagnostic failure"
    assert (result_root / run_name / "failure_partial.csv").exists()
