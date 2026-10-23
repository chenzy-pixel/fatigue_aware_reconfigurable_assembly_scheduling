import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import heuristic_search as launch


def test_worker_grid_respects_cpu_and_memory_without_an_eight_worker_cap():
    hardware = {"physical_cpus": 24, "logical_cpus": 32, "available_memory_bytes": 64*2**30}
    assert launch.worker_grid([4, 8, 12, 32], hardware) == [4, 8, 12]
    hardware["available_memory_bytes"] = 6*2**30
    assert launch.worker_grid([4, 8, 12], hardware) == [4]
    hardware.update(physical_cpus=None, logical_cpus=2, available_memory_bytes=None)
    assert launch.worker_grid([4, 8], hardware) == [1]


@pytest.fixture
def frozen_source(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    files = []
    metadata = {}
    for pressure in range(7):
        for replicate in range(3):
            name = f"instance_{pressure}_{replicate}.json"
            path = tmp_path / "data/instances/v2/validation" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            files.append({"path": name, "sha256": launch.sha(path)})
            metadata[name] = SimpleNamespace(instance=SimpleNamespace(instance_id=f"case_{pressure}_{replicate}"),
                                             metadata={"pressure_type": str(pressure)})
    manifest = tmp_path / "data/manifests/v2/validation/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    config = {"saved_config_marker": "historical_run"}
    launch.atomic_json(source / "identity.json", {"sha256": "source-id", "config": config})
    launch.atomic_json(source / "subsets.json", {"tuning": {"files": files, "dataset_manifest_sha256": launch.sha(manifest)}})
    launch.atomic_json(source / "search_selection.json", {"population_sizes": launch.SIZES})
    for name in ("scripts/heuristic_search.py", "scripts/comparison_suite.py", "analysis/comparator_study.py", "utils.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen source", encoding="utf-8")
    monkeypatch.setattr(launch, "load_generated_record", lambda path: metadata[Path(path).name])
    monkeypatch.setattr(launch, "host", lambda: {"physical_cpus": 8, "logical_cpus": 16, "available_memory_bytes": 16*2**30})
    return source, config


def test_frozen_source_keeps_exact_seven_and_saved_config(frozen_source):
    source, original = frozen_source
    config, records, plan, hardware = launch.inputs(source, "seven", [11, 23, 37], 1800.)
    assert config == original
    assert [r.instance.instance_id for r in records] == [f"case_{i}_0" for i in range(7)]
    assert plan["seconds_per_request"] == 1800 and plan["population_sizes"] == {"nsga2_ls": 16, "moead": 32}
    assert "available_memory_bytes" not in plan["hardware"]
    assert hardware["available_memory_bytes"] is not None


def test_frozen_source_rejects_changed_instance(frozen_source):
    source, _ = frozen_source
    path = source.parent / "data/instances/v2/validation/instance_0_0.json"
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="instance hash"):
        launch.inputs(source, "seven", [11], 1800.)


def test_preflight_matrix_and_concurrency_identity(frozen_source, tmp_path):
    source, _ = frozen_source
    _, _, plan, hardware = launch.inputs(source, "seven", [11, 23, 37], 1800.)
    output = tmp_path / "output"
    output.mkdir()
    identity = launch.preflight(plan, hardware, output, 4)
    before = (output / "identity.json").read_bytes()
    assert json.loads((output / "preflight.json").read_text())["requests"] == 168
    assert launch.preflight(plan, hardware, output, 4) == identity
    with pytest.raises(ValueError, match="concurrency changed"):
        launch.preflight(plan, hardware, output, 8)
    assert (output / "identity.json").read_bytes() == before


def test_heavy_launch_rejects_missing_remote_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["heuristic_search", "--stage", "run"])
    with pytest.raises(SystemExit) as error:
        launch.main()
    assert error.value.code == 2


def test_benchmark_does_not_silently_ignore_a_changed_worker_grid(tmp_path):
    plan = {"version": "frozen"}
    launch.atomic_json(tmp_path / "benchmark.json", {"identity": launch.digest(plan), "requested_grid": [1], "recommended_workers": 1})
    with pytest.raises(ValueError, match="worker grid changed"):
        launch.benchmark({}, [], plan, {}, tmp_path, [1, 2])
