"""The remote launchers select the new experiments and honor preflight-only."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from configs import load_config
import scripts.run_flow_excess_training as launch


def configure_launcher(monkeypatch, tmp_path, arguments):
    configs = {key: load_config(path) for key, path in launch.CONFIGS.items()}
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    monkeypatch.setattr(launch, "load_config", lambda path: deepcopy(
        configs["flow" if path.name == "single_flow.json" else "universal"]))
    monkeypatch.setattr(launch, "startup_preflight", lambda configs, **kwargs: {
        label: {"probe_network": kwargs["probe_network"]} for label in configs})
    monkeypatch.setattr("sys.argv", ["run_flow_excess_training.py", *arguments])


def test_modified_only_launch_is_sequential_and_preserves_budgets(monkeypatch, tmp_path):
    configure_launcher(monkeypatch, tmp_path, [])
    calls = []

    def child(command, *, cwd):
        assert cwd == tmp_path
        config = json.loads(open(command[command.index("--config") + 1], encoding="utf-8").read())
        assert config["objective_scalarizer"]["flow_mode"] == "excess_proportional_lb_v1"
        calls.append(config)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, "run", child)
    launch.main()
    assert [config["training"]["episodes"] for config in calls] == [1000, 2000]
    assert [config["training"]["episodes_per_update"] for config in calls] == [20, 20]
    assert [config["preference"]["quality"]["mode"] for config in calls] == ["fixed", "universal_sobol_v1"]
    status = json.loads(next(tmp_path.rglob("launch.json")).read_text())
    assert [job["status"] for job in status["jobs"]] == ["completed", "completed"]


def test_preflight_only_never_starts_training(monkeypatch, tmp_path):
    configure_launcher(monkeypatch, tmp_path, ["--preflight-only"])
    monkeypatch.setattr(launch.subprocess, "run", lambda *args, **kwargs: pytest.fail("preflight launched training"))
    launch.main()
    report = json.loads(next(tmp_path.rglob("preflight.json")).read_text())
    assert all(value["probe_network"] for value in report.values())
    assert not (tmp_path / "result/runs").exists()


def test_single_flow_wrapper_default_selects_one_experiment(monkeypatch, tmp_path):
    configure_launcher(monkeypatch, tmp_path, [])
    calls = []
    monkeypatch.setattr(launch.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or SimpleNamespace(returncode=0))
    launch.main(default_experiment="flow")
    assert len(calls) == 1
    status = json.loads(next(tmp_path.rglob("launch.json")).read_text())
    assert status["jobs"][0]["experiment"] == "flow"


def test_failed_child_stops_the_remaining_experiment(monkeypatch, tmp_path):
    configure_launcher(monkeypatch, tmp_path, [])
    calls = []
    monkeypatch.setattr(launch.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or SimpleNamespace(returncode=7))
    with pytest.raises(SystemExit) as failure:
        launch.main()
    assert failure.value.code == 7 and len(calls) == 1
    status = json.loads(next(tmp_path.rglob("launch.json")).read_text())
    assert [job["status"] for job in status["jobs"]] == ["failed", "pending"]


def test_worker_override_preserves_universal_update_budget(monkeypatch, tmp_path):
    configure_launcher(monkeypatch, tmp_path, ["--experiment", "universal", "--parallel-envs", "4"])
    configs = []

    def child(command, **kwargs):
        with open(command[command.index("--config") + 1], encoding="utf-8") as stream:
            configs.append(json.load(stream))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, "run", child)
    launch.main()
    assert configs[0]["training"]["parallel_envs"] == 4
    assert configs[0]["training"]["episodes_per_update"] == 20
