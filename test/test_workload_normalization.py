"""Frozen workload provenance must agree with the active objective identity."""
from copy import deepcopy
import json

import pytest

from configs import load_config
from configs.config import PROJECT_ROOT
from configs.normalization import apply_normalization_manifest, canonical_json_sha256, file_sha256, load_normalization_manifest


@pytest.mark.parametrize("q,folder",[(.25,"q025"),(.5,"q05"),(.75,"q075")])
@pytest.mark.parametrize("raw", [False, True])
def test_frozen_workload_scale(q, folder, raw):
    path = "configs/e1/single_workload.json" if not raw and folder == "q05" else f"configs/controls/{'raw_' if raw else ''}single_owa_{folder}.json"
    config = load_config(path)
    scalarizer = config["objective_scalarizer"]
    manifest = load_normalization_manifest(PROJECT_ROOT / scalarizer["normalization_manifest"], expected_sha256=scalarizer["normalization_manifest_sha256"])
    assert config["workload_objective"]["q"] == q
    assert config["network"]["workload_q"] == q
    assert config["network"]["workload_scale"] == scalarizer["scales"]["workload"]
    assert manifest["workload_source"]["trajectory_count"] == 50
    assert manifest["workload_source"]["successful_count"] == 49
    assert scalarizer["scales"]["cost"] == 353.27
    assert scalarizer["scales"]["flow"] == (1089.15 if raw else 368.3143)


def test_mode_or_q_mismatch_rejected():
    original = load_config("configs/default.json")
    for mode,q in (("variance",None),("geometric_owa",.25)):
        c = deepcopy(original)
        c["workload_objective"] = {"mode":mode,"q":q}
        for key in list(c["network"]):
            if key.startswith("workload_"):
                del c["network"][key]
        with pytest.raises(ValueError, match="workload identity"):
            apply_normalization_manifest(c, project_root=PROJECT_ROOT)


def test_tampered_scale_cannot_be_rehashed_into_valid_manifest(tmp_path):
    c=load_config("configs/default.json")
    p=PROJECT_ROOT/c["objective_scalarizer"]["normalization_manifest"]
    data=json.loads(p.read_text())
    data["scales"]["workload"] += 1
    data.pop("content_sha256")
    data["content_sha256"]=canonical_json_sha256(data)
    target=tmp_path/"tampered.json"
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="successful reference mean"):
        load_normalization_manifest(target,expected_sha256=file_sha256(target))


def test_existing_calibration_tag_rejected_before_reference_reads_or_writes(monkeypatch, tmp_path):
    import scripts.build_workload_normalization as builder
    monkeypatch.setattr("sys.argv", ["build_workload_normalization.py", "--output", str(tmp_path),
                                    "--reference-config", "missing-reference.json"])
    monkeypatch.setattr(builder, "load_config", lambda *_: pytest.fail("must check frozen destinations first"))
    with pytest.raises(FileExistsError, match="tag already exists"):
        builder.main()
    assert not list(tmp_path.iterdir())
