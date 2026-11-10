"""Same-root DCAN refresh, immutable existing results and execution provenance."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from agent.comparators.common import atomic_json, digest
from agent.comparators.learning import check_learning_checkpoint
from agent.comparators.training_budget import training_budget
from configs.experiment_suite import arm_config, identity_payload, load_suite
from configs import experiment_reuse as reuse
from scripts import experiment_suite as executor


def snapshot(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("experiment_identity.json", json.dumps(payload))


@pytest.fixture(scope="module")
def current_payload():
    return identity_payload(load_suite(), "cpu")


@pytest.fixture
def experiment(tmp_path, current_payload):
    current = deepcopy(current_payload)
    previous = deepcopy(current)
    manifest = reuse.compatibility_manifest()
    for path, rule in manifest["sources"].items():
        if rule["before"] == [None]:
            previous["sources"].pop(path, None)
        else:
            previous["sources"][path] = rule["before"][0]
    atomic_json(tmp_path/"preflight.json", {"status":"passed", "identity":digest(previous),
                                           "source_hashes":previous["sources"]})
    snapshot(tmp_path/"source.zip", previous)
    sentinel = tmp_path/"training"/"full"/"11"/"summary.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b'{"completed_original_run":true}\n')
    return tmp_path, previous, current


def register(experiment):
    root, previous, current = experiment
    plan = reuse.prepare_refresh(root, current)
    destination = root/"executions"/digest(current)
    atomic_json(destination/"preflight.json", {"status":"passed", "identity":digest(current)})
    snapshot(destination/"source.zip", current)
    return reuse.publish_refresh(root, plan)


def test_register_keeps_original_results_and_identity(experiment):
    root, previous, current = experiment
    paths = [root/"preflight.json", root/"source.zip", root/"training/full/11/summary.json"]
    before = [p.read_bytes() for p in paths]
    value = register(experiment)
    assert executor.check_preflight(root, digest(current)) == digest(previous)
    assert value["comparison_identity"] == digest(previous)
    assert reuse.arm_execution_identity(root, "dcan") == digest(current)
    assert reuse.arm_execution_identity(root, "full") is None
    assert [p.read_bytes() for p in paths] == before
    assert not (root/"training/dcan").exists()


@pytest.mark.parametrize("path", ["training/dcan/11/status.json", "evaluation/frontier/dcan/11/unit.json",
                                   "evaluation/independent_request/dcan/11/unit.json"])
def test_existing_dcan_files_block_refresh_without_mutation(experiment, path):
    root, _, current = experiment
    file = root/path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"old_dcan")
    before = (root/"preflight.json").read_bytes()
    with pytest.raises(ValueError, match="clear the DCAN results"):
        reuse.prepare_refresh(root, current)
    assert file.read_bytes() == b"old_dcan"
    assert (root/"preflight.json").read_bytes() == before
    assert not (root/reuse.REGISTRATION).exists()


@pytest.mark.parametrize("change", ["config", "data", "unreviewed_source", "unknown_before", "unknown_after"])
def test_refresh_rejects_unapproved_changes(experiment, change):
    _, old, new = experiment
    if change == "config": new["configs"]["full"]["ppo"]["learning_rate"] *= 2
    if change == "data": new["data"] = {"modified":True}
    if change == "unreviewed_source": new["sources"]["environment/env.py"] = "changed"
    if change == "unknown_before": old["sources"]["agent/comparators/dcan.py"] = "unknown"
    if change == "unknown_after": new["sources"]["agent/comparators/dcan.py"] = "unreviewed"
    with pytest.raises(ValueError):
        reuse.compare_payloads(old, new, reuse.compatibility_manifest())


def test_frozen_source_manifest_matches_current_sources(current_payload):
    manifest = reuse.compatibility_manifest()
    for path, rule in manifest["sources"].items():
        assert current_payload["sources"][path] in rule["after"], path


@pytest.mark.parametrize("name", ["preflight.json", "source.zip", "execution_preflight", "execution_source"])
def test_registration_rejects_changed_evidence(experiment, name):
    root, _, current = experiment
    register(experiment)
    path = root/name
    if name.startswith("execution_"):
        path = root/"executions"/digest(current)/("preflight.json" if name.endswith("preflight") else "source.zip")
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="evidence differs"):
        executor.check_preflight(root, digest(current))


def test_original_snapshot_identity_and_hashes_are_required(experiment):
    root, previous, current = experiment
    old = executor.read_json(root/"preflight.json")
    old["identity"] = "wrong"
    atomic_json(root/"preflight.json", old)
    with pytest.raises(ValueError, match="does not match"):
        reuse.prepare_refresh(root, current)
    old["identity"] = digest(previous)
    old["source_hashes"] = {}
    atomic_json(root/"preflight.json", old)
    with pytest.raises(ValueError, match="source hashes differ"):
        reuse.prepare_refresh(root, current)


def test_training_dispatch_and_checkpoint_execution_guard(experiment, monkeypatch):
    root, previous, current = experiment
    register(experiment)
    suite = load_suite()
    monkeypatch.setattr(executor, "load_dataset_split", lambda *a: list(range(27)))
    calls = []
    monkeypatch.setattr(executor, "train_request", lambda *args, **kwargs: calls.append((args, kwargs)))
    executor.train(suite, root, digest(previous), "cpu", ["full", "dcan"], [11])
    for args, kwargs in calls:
        arm = args[0]["comparison"]["arm"]
        assert args[7] == root/"training"/arm/"11"
        assert args[8] == digest(previous)
        assert kwargs["episodes"] == 2000
        assert args[0]["comparison"].get("execution_identity") == (digest(current) if arm == "dcan" else None)
    config = calls[1][0][0]
    meta = {"algorithm":"dcan", "seed":11, "comparison_identity":digest(previous), "training_budget":training_budget(config)}
    path = root/"checkpoint.pt"
    torch.save({"metadata":meta}, path)
    with pytest.raises(ValueError, match="execution identity mismatch"):
        check_learning_checkpoint(path, config, "dcan", 11, digest(previous))
    meta["execution_identity"] = digest(current)
    torch.save({"metadata":meta}, path)
    check_learning_checkpoint(path, config, "dcan", 11, digest(previous))
    assert reuse.execution_job(["same", 11], None) == digest(["same", 11])
    assert reuse.execution_job(["same", 11], digest(current)) != digest(["same", 11])


def test_evaluation_cache_binds_the_new_execution(experiment, monkeypatch):
    root, previous, current = experiment
    register(experiment)
    config = arm_config(load_suite(), "dcan", 11)
    record = SimpleNamespace(instance=SimpleNamespace(instance_id="case"), metadata={"cell_id":"C01", "scale":"small"})
    monkeypatch.setattr(executor, "load_dataset_split", lambda *a: [record])
    monkeypatch.setattr(executor, "preferences", lambda *a: [(1., 0., 0.)])
    monkeypatch.setattr(executor, "load_agent", lambda *a, **kw: (config, object(), {}))
    calls = []
    def solve(*args, **kwargs):
        calls.append(1)
        return {"seconds":1., "requests":[], "deliveries":[]}
    monkeypatch.setattr(executor, "learning_unit", solve)
    path = root/"evaluation/independent_request/dcan/11/case.json"
    atomic_json(path, {"job_id":digest([digest(previous), "independent_request", "dcan", 11, "case", None]), "status":"complete"})
    with pytest.raises(ValueError, match="artifact identity differs"):
        executor.evaluate(load_suite(), root, digest(previous), "cpu", ["dcan"], [11], ["independent_request"])
    assert not calls
    path.unlink()
    for _ in range(2):
        executor.evaluate(load_suite(), root, digest(previous), "cpu", ["dcan"], [11], ["independent_request"])
    assert len(calls) == 1
    saved = executor.read_json(path)
    assert saved["identity"] == digest(previous)
    assert saved["execution_identity"] == digest(current)
    assert saved["content_sha256"] == digest({k:v for k,v in saved.items() if k != "content_sha256"})


def test_refresh_cli_publishes_once_and_routes_to_original_root(experiment, monkeypatch):
    import sys
    import configs.experiment_suite as specification
    root, previous, current = experiment
    monkeypatch.setattr(executor, "suite_identity", lambda *a: digest(current))
    monkeypatch.setattr(specification, "identity_payload", lambda *a: current)
    probes = []
    def probe(*args):
        probes.append(1)
        return {"status":"passed", "identity":digest(current)}
    monkeypatch.setattr(executor, "preflight", probe)
    def archive(destination, *args):
        assert destination == root/"executions"/digest(current)
        snapshot(destination/"source.zip", current)
        return current["sources"]
    monkeypatch.setattr(executor, "snapshot_sources", archive)
    command = ["suite", "--output-dir", str(root), "--device", "cpu"]
    for flags in (["--stage", "preflight", "--refresh-dcan"], ["--stage", "preflight", "--refresh-dcan"], ["--stage", "preflight"]):
        monkeypatch.setattr(sys, "argv", command+flags)
        executor.main()
    assert len(probes) == 1
    calls = []
    monkeypatch.setattr(executor, "train", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", command+["--stage", "train", "--arms", "dcan", "--seeds", "11", "--remote-run"])
    executor.main()
    assert calls[0][1:3] == (root, digest(previous))
    assert calls[0][4:6] == (["dcan"], [11])


def test_refresh_flag_is_scoped_to_preflight(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["suite", "--stage", "train", "--refresh-dcan"])
    with pytest.raises(SystemExit):
        executor.main()
