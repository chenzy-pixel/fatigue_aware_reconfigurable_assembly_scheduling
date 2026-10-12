"""Freeze OWA scales from one deterministic, full validation reference rollout."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.baselines import HeuristicPolicy
from configs import load_config
from configs.config import public_config
from configs.normalization import canonical_json_sha256, file_sha256, write_immutable_manifest, load_normalization_manifest, WORKLOAD_MANIFEST_SCHEMA
from data import load_dataset_split
from environment import AssemblySchedulingEnv
from environment.workload import workload_config, workload_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-config", default="configs/e1/single_variance.json")
    parser.add_argument("--output", default="result/audits/workload_owa/calibration")
    parser.add_argument("--reuse-reference", action="store_true", help="rebuild manifests from this run's recorded reference rows")
    parser.add_argument("--reference-directory", help="directory containing immutable reference_config.json and reference_rows.json")
    parser.add_argument("--manifest-tag", default="20261011", help="unique tag for immutable output files")
    args = parser.parse_args()
    if not args.manifest_tag.isalnum():
        raise ValueError("manifest tag must be alphanumeric")
    destinations = [ROOT / f"configs/manifests/workload_{label}_q{int(q*100):02d}_{args.manifest_tag}.json"
                    for q in (.25, .5, .75) for label in ("raw", "excess")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("immutable workload manifest tag already exists; choose a new --manifest-tag")
    out = ROOT / args.output / args.manifest_tag
    if (out / "reference_config.json").exists() or (out / "reference_rows.json").exists():
        raise FileExistsError("calibration evidence already exists; use a new output/tag")
    config = load_config(args.reference_config)
    if workload_config(config)["mode"] != "variance":
        raise ValueError("calibration reference must use the variance workload objective")
    dataset = load_dataset_split(config, "validation")
    selection_path = ROOT / "data/manifests/v2/validation/subsets.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["target"]
    if selection["dataset_manifest_sha256"] != file_sha256(dataset.manifest_path):
        # Dataset files use canonical bytes, and fixed manifests may contain a final newline.
        from result.provenance import dataset_manifest_snapshot
        if selection["dataset_manifest_sha256"] != dataset_manifest_snapshot(dataset.manifest_path)["sha256"]:
            raise ValueError("fixed calibration subset does not match validation manifest")
    indices = selection["instance_indices"]
    if len(indices) != 50 or len(set(indices)) != 50:
        raise ValueError("calibration requires the fixed 50 validation indices")
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    if args.reuse_reference:
        reference = ROOT / args.reference_directory if args.reference_directory else out
        stored_config = json.loads((reference / "reference_config.json").read_text(encoding="utf-8"))
        if stored_config != public_config(config):
            raise ValueError("recorded reference configuration changed")
        rows = json.loads((reference / "reference_rows.json").read_text(encoding="utf-8"))
        if [row["instance_id"] for row in rows] != [dataset[index].instance.instance_id for index in indices]:
            raise ValueError("recorded reference instances do not match the fixed selection")
    for number, index in enumerate(() if args.reuse_reference else indices, 1):
        record = dataset[index]
        env = AssemblySchedulingEnv(config)
        env.reset(record.instance, build_observation=False)
        policy = HeuristicPolicy()
        while not env.task_done:
            env.step(policy.select_action(env), build_observation=False)
        violations = env.validate_schedule()
        physical = not violations and env.metrics()["maximum_worker_fatigue"] <= record.instance.fatigue.maximum_safe_fatigue + 1e-8
        rows.append({"instance_id": record.instance.instance_id, "success": env.task_succeeded,
                     "physical_valid": physical, "failure_reason": env.terminal_reason,
                     "worker_loads": {w.spec.id: float(w.load) for w in env.workers},
                     "total_labor": sum(float(w.load) for w in env.workers),
                     "reconfiguration_count": len(env.reconfigurations)})
        if number % 10 == 0:
            print(f"Calibration {number}/50", flush=True)
    source = {"policy": "HeuristicPolicy", "config_sha256": canonical_json_sha256(public_config(config)),
              "validation_dataset_manifest_sha256": selection["dataset_manifest_sha256"],
              "validation_subset_sha256": selection["subset_sha256"],
              "reference_config": args.reference_config,
              "selection_file_sha256": file_sha256(selection_path),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": file_sha256(__file__), "trajectory_count": 50,
              "successful_count": sum(row["success"] and row["physical_valid"] for row in rows)}
    with (out / "reference_config.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(public_config(config), indent=2))
    with (out / "reference_rows.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(rows, indent=2))
    manifests = {}
    for q in (.25, .5, .75):
        identity = workload_config({"workload_objective": {"mode": "geometric_owa", "q": q}})
        qrows = [{**row, "workload_value": workload_value(list(row["worker_loads"].values()), {"workload_objective": identity})} for row in rows]
        good = [row["workload_value"] for row in qrows if row["success"] and row["physical_valid"]]
        if not good:
            raise ValueError("no successful physically valid calibration references")
        for label, path in (("raw", "configs/manifests/v2_selected_scales_20261002.json"),
                            ("excess", "configs/manifests/flow_excess_scales_20261008.json")):
            inherited = load_normalization_manifest(ROOT / path, expected_sha256=file_sha256(ROOT / path))
            payload = {"schema_version": WORKLOAD_MANIFEST_SCHEMA, "selection": "successful_heuristic_validation_mean",
                       "bounded_objective": "q_i=J_i/(s_i+J_i)", "flow_mode": inherited.get("flow_mode", "raw_v1"),
                       "workload_objective": identity, "scales": {"flow": inherited["scales"]["flow"],
                       "cost": inherited["scales"]["cost"], "workload": statistics.fmean(good)},
                       "inherited_manifest": {"path": path, "sha256": file_sha256(ROOT / path)},
                       "workload_source": {**source, "rows": deepcopy(qrows), "reference_rows_sha256": canonical_json_sha256({"rows": qrows})}}
            payload["content_sha256"] = canonical_json_sha256(payload)
            destination = ROOT / f"configs/manifests/workload_{label}_q{int(q*100):02d}_{args.manifest_tag}.json"
            digest = write_immutable_manifest(destination, payload)
            load_normalization_manifest(destination, expected_sha256=digest)
            manifests[f"{label}_q{int(q*100):02d}"] = {"path": str(destination.relative_to(ROOT)).replace("\\", "/"), "sha256": digest, "workload_scale": statistics.fmean(good)}
    (out / "manifests.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(json.dumps({"successful": source["successful_count"], "manifests": manifests}, indent=2), flush=True)


if __name__ == "__main__":
    main()
