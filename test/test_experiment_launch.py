"""Batch launching must preserve budgets and stop before an invalid next run."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from configs import load_config
import scripts.run_experiments as launch


def configure_launcher(monkeypatch, tmp_path, arguments):
    configs = {name: load_config(path) for name, path in {
        "default.json": "configs/default.json", "single_flow.json": "configs/e1/single_flow.json",
        "single_cost.json": "configs/e1/single_cost.json",
        "single_workload.json": "configs/e1/single_workload.json",
    }.items()}
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    monkeypatch.setattr(launch, "load_config", lambda path: deepcopy(configs[Path(path).name]))
    monkeypatch.setattr(launch, "startup_preflight", lambda configs, **kwargs: {
        label: {"probe_network": kwargs["probe_network"]} for label in configs})
    monkeypatch.setattr("sys.argv", ["run_experiments.py", *arguments])


def test_batch_is_sequential_and_preserves_update_budget(monkeypatch, tmp_path):
    configure_launcher(monkeypatch, tmp_path, ["--config", "configs/e1/single_flow.json", "configs/default.json",
                                             "--parallel-envs", "4", "--seed", "23"])
    calls = []

    def child(command, *, cwd):
        assert cwd == tmp_path
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        calls.append(config)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, "run", child)
    launch.main()
    assert [c["training"]["episodes"] for c in calls] == [1000, 2000]
    assert all(c["seed"] == 23 and c["training"]["episodes_per_update"] == 20
               and c["training"]["parallel_envs"] == 4 for c in calls)
    status = json.loads(next(tmp_path.rglob("launch.json")).read_text())
    assert [job["status"] for job in status["jobs"]] == ["completed", "completed"]


def test_preflight_only_never_starts_training(monkeypatch, tmp_path):
    configure_launcher(monkeypatch, tmp_path, ["--preflight-only"])
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: pytest.fail("preflight launched training"))
    launch.main()
    report = json.loads(next(tmp_path.rglob("preflight.json")).read_text())
    assert all(value["probe_network"] for value in report.values())
    assert not (tmp_path / "result/runs").exists()


def test_failed_child_stops_the_remaining_experiment(monkeypatch, tmp_path):
    configure_launcher(monkeypatch, tmp_path, ["--config", "configs/e1/single_flow.json", "configs/default.json"])
    calls = []
    monkeypatch.setattr(launch.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or SimpleNamespace(returncode=7))
    with pytest.raises(SystemExit) as failure:
        launch.main()
    assert failure.value.code == 7 and len(calls) == 1
    status = json.loads(next(tmp_path.rglob("launch.json")).read_text())
    assert [job["status"] for job in status["jobs"]] == ["failed", "pending"]


def test_fatigue_suite_dispatches_all_matched_arms(monkeypatch, tmp_path):
    from scripts.ablation_protocol import FATIGUE, ROLE_CONFIGS
    configure_launcher(monkeypatch, tmp_path, ["--suite", "fatigue"])
    calls = []
    monkeypatch.setattr(launch, "launch_configurations", lambda configs, args: calls.append(configs))
    launch.main()
    assert calls == [{role: ROLE_CONFIGS[role] for role in FATIGUE}]
    assert len(FATIGUE) == 6


def test_config_and_suite_are_mutually_exclusive(monkeypatch, tmp_path):
    configure_launcher(monkeypatch, tmp_path, ["--config", "configs/default.json", "--suite", "structural"])
    with pytest.raises(SystemExit) as error:
        launch.main()
    assert error.value.code == 2
