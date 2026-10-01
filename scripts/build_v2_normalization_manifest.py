"""Freeze the user-selected rounded V2 validation references with provenance."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from configs.normalization import (
    SELECTED_VALIDATION_MANIFEST_SCHEMA, canonical_json_sha256, file_sha256,
    load_normalization_manifest, write_immutable_manifest,
)

REFERENCES = {
    "flow": (500, 1089.15, 2, "flow_time_objective"),
    "cost": (500, 353.27, 2, "reconfiguration_cost"),
    "variance": (360, 2.2629, 4, "worker_load_variance"),
}


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_manifest(run_root: Path, project_root: Path) -> dict:
    sources = {}
    manifest_hashes, subset_hashes = set(), set()
    for objective, (episode, scale, digits, field) in REFERENCES.items():
        run = run_root / f"{objective}_v2_seed11_best_plus500"
        log_path = run / "validation_log.csv"
        rows_path = run / "sampled_validation_instance_metrics.csv"
        config_path = run / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        logged = next(row for row in read_csv(log_path) if int(row["episode"]) == episode)
        rows = [row for row in read_csv(rows_path) if int(row["validation_episode"]) == episode]
        units = {(row["instance_id"], int(row["sampling_repeat"])) for row in rows}
        if len(rows) != 150 or len(units) != 150 or len({row["instance_id"] for row in rows}) != 50:
            raise ValueError(f"{objective}: expected 50 instances x 3 repeats")
        good = [row for row in rows if row["task_succeeded"] == "True"]
        mean = float(logged[f"mean_{field}"])
        if not math.isclose(statistics.fmean(float(row[field]) for row in good), mean, abs_tol=1e-9):
            raise ValueError(f"{objective}: logged/raw means disagree")
        if not math.isclose(scale, round(mean, digits), abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"{objective}: user value disagrees with rounded validation reference")
        if not math.isclose(len(good)/len(rows), float(logged["completion_rate"]), abs_tol=1e-12):
            raise ValueError(f"{objective}: completion count")
        manifest_hashes.add(logged["dataset_manifest_sha256"])
        subset_hashes.add(logged["subset_sha256"])
        sources[objective] = {
            "run": run.relative_to(project_root).as_posix() if run.is_relative_to(project_root) else str(run),
            "algorithm_seed": int(config["seed"]), "validation_episode": episode,
            "successful_trajectory_mean": mean, "rounding_digits": digits,
            "successful_count": len(good), "trajectory_count": len(rows),
            "config_sha256": file_sha256(config_path),
            "validation_log_sha256": file_sha256(log_path),
            "sampled_validation_rows_sha256": file_sha256(rows_path),
        }
    if len(manifest_hashes) != 1 or len(subset_hashes) != 1:
        raise ValueError("source validation protocols differ")
    payload = {
        "schema_version": SELECTED_VALIDATION_MANIFEST_SCHEMA,
        "selection": "user_selected_rounded_validation_means",
        "selected_date": "2026-10-02", "bounded_objective": "q_i=J_i/(s_i+J_i)",
        "scales": {objective: row[1] for objective, row in REFERENCES.items()},
        "validation_dataset_manifest_sha256": manifest_hashes.pop(),
        "validation_subset_sha256": subset_hashes.pop(), "sources": sources,
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=root/"result/continuation/v2_seed11_best_plus500/runs")
    parser.add_argument("--output", type=Path, default=root/"configs/manifests/v2_selected_scales_20261002.json")
    args = parser.parse_args()
    manifest = build_manifest(args.run_root.resolve(), root)
    digest = write_immutable_manifest(args.output, manifest)
    load_normalization_manifest(args.output, expected_sha256=digest)
    print(json.dumps({"path": str(args.output), "sha256": digest, "scales": manifest["scales"]}))


if __name__ == "__main__":
    main()
