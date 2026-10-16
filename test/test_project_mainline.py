"""The public defaults and saved experimental identities must survive cleanup."""
import json

import pytest

from configs import load_config, project_path
from configs.config import public_config
from configs.runtime import runtime_manifest
from result.provenance import effective_config_snapshot


def test_mainline_contract_and_three_single_objective_overrides():
    base = load_config("configs/default.json")
    assert base["network"]["actor_head_variant"] == "shared_preference"
    assert base["network"]["observation_schema_version"] == 12
    assert base["network"]["global_objective_encoding"] == "log2_v1"
    assert base["objective_scalarizer"]["flow_mode"] == "excess_proportional_lb_v1"
    assert base["objective_scalarizer"]["scales"] == {
        "flow": 368.3143, "cost": 353.27, "workload": 21.03145448655653,
    }
    assert base["ppo"]["gamma"] == 1 and base["ppo"]["gae_lambda"] == .95
    assert base["training"]["episodes"] == 2000
    assert base["training"]["episodes_per_update"] == 20
    assert base["training"]["formal_evaluation"]["sample_aggregation"] == "best_of_3_v1"
    for index, name in enumerate(("flow", "cost", "workload")):
        child = load_config(f"configs/e1/single_{name}.json")
        assert child["preference"]["quality"]["fixed"] == [float(i == index) for i in range(3)]
        assert child["training"]["episodes"] == 1000
        assert child["training"]["episodes_per_update"] == 20
        for field in ("network", "ppo", "reward", "objective_scalarizer"):
            assert child[field] == base[field]


def test_complete_legacy_snapshot_keeps_implicit_defaults_and_hash(tmp_path):
    snapshot = public_config(load_config("configs/controls/raw.json"))
    snapshot["objective_scalarizer"].pop("flow_mode")
    snapshot["training"]["formal_evaluation"].pop("sample_aggregation")
    snapshot["runtime_manifest"] = runtime_manifest(snapshot)
    expected = effective_config_snapshot(snapshot)["sha256"]
    snapshot["extends"] = str(tmp_path / "retired_preset.json")
    saved = tmp_path / "config.json"
    saved.write_text(json.dumps(snapshot), encoding="utf-8")
    restored = load_config(saved)
    assert restored["runtime_manifest"]["actor_head_variant"] == "shared_preference"
    assert restored["runtime_manifest"]["flow_mode"] == "raw_v1"
    assert "sample_aggregation" not in restored["training"]["formal_evaluation"]
    assert effective_config_snapshot(restored)["sha256"] == expected


@pytest.mark.parametrize("path", sorted(
    p.relative_to(project_path(".")).as_posix()
    for p in project_path("configs").rglob("*.json")
    if not {"archive", "manifests"}.intersection(p.parts) and p.name != "mo_alns_manifest.json"
))
def test_retained_configuration_loads(path):
    config = load_config(path)
    assert config["runtime_manifest"]["observation_schema"] == 12
