"""Verify saved Flow experiment evidence and write its reproducibility receipt."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs import load_config
from environment.types import FLOW_EXCESS, FLOW_RAW, flow_mode
from result.io import write_json


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_report(path):
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    failures = [{"class": c.get("classname"), "name": c.get("name"),
                 "message": c.find("failure").get("message")} for c in cases if c.find("failure") is not None]
    errors = [{"class": c.get("classname"), "name": c.get("name"),
               "message": c.find("error").get("message")} for c in cases if c.find("error") is not None]
    return {"tests": len(cases), "passed": sum(c.find("failure") is None and
            c.find("error") is None and c.find("skipped") is None for c in cases),
            "skipped": sum(c.find("skipped") is not None for c in cases), "failures": failures, "errors": errors}


def main():
    output = ROOT / "result/analysis/flow_excess_integration_20261008"
    replay = read_rows(output / "native_historical_replay.csv")
    assert len(replay) == 180 and max(float(r["max_error"]) for r in replay) < 1e-8
    heuristic = read_rows(output / "fixed_validation_metrics.csv")
    assert len(heuristic) == 50 and all(int(r["schedule_violations"]) == 0 for r in heuristic)
    wait = json.loads((output / "wait_audit_summary.json").read_text())
    assert wait["legal_wait_states"] == len(read_rows(output / "legal_wait_projections.csv"))
    smoke_runs = json.loads((output / "smoke_runs.json").read_text())
    modes = {label: flow_mode(load_config(Path(path) / "config.json")) for label, path in smoke_runs.items()}
    assert modes == {"raw": FLOW_RAW, "excess": FLOW_EXCESS}
    primary = json.loads((output / "matched_hv/raw_primary/summary.json").read_text())
    supplementary = json.loads((output / "matched_hv/excess_supplementary/summary.json").read_text())
    assert primary["objective_scales"]["flow"] == 1089.15
    assert supplementary["objective_scales"]["flow"] == 368.3143
    assert primary["candidate_count"] == supplementary["candidate_count"] == 132
    full, related = test_report(output / "pytest.xml"), test_report(output / "related_tests.xml")
    assert not related["failures"] and not related["errors"] and not full["errors"]
    assert {r["name"] for r in full["failures"]} <= {"test_legacy_data_is_preserved_and_loadable"}
    artifacts = [p for p in output.rglob("*") if p.is_file() and p.name != "acceptance.json"]
    report = {
        "schema": "flow_excess_acceptance_v1", "smoke_runs": smoke_runs,
        "historical_replay": {"trajectories": len(replay), "states": sum(int(r["states"]) for r in replay),
                              "max_error": max(float(r["max_error"]) for r in replay)},
        "heuristic": {"instances": len(heuristic), "successful": sum(r["success"] == "True" for r in heuristic)},
        "full_tests": full, "related_tests": related, "wait_audit": wait,
        "timing": json.loads((output / "environment_timing_summary.json").read_text()),
        "ppo_updates": read_rows(output / "ppo_decision_updates.csv"),
        "ppo_wait_audit": json.loads((output / "ppo_wait_audit_summary.json").read_text()),
        "artifact_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
        "verification_source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "environment/env.py", "environment/time_context.py", "environment/types.py", "agent/ppo/network.py",
            "scripts/run_flow_excess_smoke.py", "analysis/flow_evaluation.py", "analysis/v8_pareto_analysis.py")},
        "scope": "implementation and short-run validation; full multi-seed training is a separate experiment",
    }
    write_json(output / "acceptance.json", report)
    print({"full_tests": full, "related_tests": related, "replay": report["historical_replay"]})


if __name__ == "__main__":
    main()
