"""Verify completed v2 development comparison artifacts against their contracts."""

from collections import Counter
import csv
import json
import math
from pathlib import Path

import torch

from configs import project_path
from configs.config import PROJECT_ROOT
from configs.normalization import apply_normalization_manifest
from configs.runtime import attach_runtime_manifest
from configs.formal_preferences import formal_preferences
from data.dataset import load_dataset_split
from data.distribution import protocol_hashes, training_sampling_plan
from data.selection import subset_snapshot
from result.io import write_json
from scripts.prepare_data_v2 import verify_legacy_files


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_run_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    saved_runtime_manifest = config.pop("runtime_manifest", None)
    apply_normalization_manifest(config, project_root=PROJECT_ROOT)
    attach_runtime_manifest(config)
    if saved_runtime_manifest != config["runtime_manifest"]:
        raise ValueError("saved run runtime manifest differs from the current implementation")
    return config


def audit_run(directory):
    directory = Path(directory)
    config = load_run_config(directory / "config.json")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    selections = summary["validation_subsets"]
    dataset = load_dataset_split(config, "validation")
    train_rows = read_rows(directory / "train_log.csv")
    if [int(row["episode"]) for row in train_rows] != list(range(1, 401)):
        raise ValueError("comparison run must contain exactly 400 ordered episodes")
    plan = training_sampling_plan(config, 400)
    for row, (pressure, severity) in zip(train_rows, plan, strict=True):
        if row["pressure_type"] != pressure or not math.isclose(float(row["severity"]), severity, abs_tol=1e-9):
            raise ValueError("training scenario or severity differs from its episode plan")
    for row in read_rows(directory / "update_log.csv"):
        for field in ("policy_loss", "value_loss", "entropy", "loss", "approx_kl", "gradient_norm"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"non-finite PPO statistic: {field}")
    main = selections["target"]
    diagnostic = selections["diagnostic"]
    if set(main["instance_indices"]) & set(diagnostic["instance_indices"]):
        raise ValueError("main and diagnostic validation overlap")
    for selection in selections.values():
        if subset_snapshot(dataset, selection["instance_indices"], role=selection["role"])["subset_sha256"] != selection["subset_sha256"]:
            raise ValueError("saved validation subset differs from the dataset")
    checkpoint = torch.load(summary["best_checkpoint"], map_location="cpu", weights_only=False)["metadata"]
    if checkpoint["validation_subset_sha256"] != main["subset_sha256"]:
        raise ValueError("checkpoint was selected on a different subset")
    if checkpoint["checkpoint_episode"] not in {200, 400}:
        raise ValueError("diagnostic validation affected checkpoint selection")
    checks = {}
    for filename, split, indices, subset_hash, stage, episodes in (
        ("sampled_validation_instance_metrics.csv", "validation", main["instance_indices"], main["subset_sha256"], "validation", (200,400)),
        ("sampled_diagnostic_instance_metrics.csv", "validation", diagnostic["instance_indices"], diagnostic["subset_sha256"], "validation", (400,)),
        ("final_sampled_instance_metrics.csv", "test", list(range(20)), summary["final_sampled"]["subset_sha256"], "final_test", (None,)),
    ):
        source = load_dataset_split(config, split)
        wanted_seeds = {source.manifest["files"][index]["seed"] for index in indices}
        points = {point.key for point in formal_preferences(config, stage)}
        expected = {(episode, seed, key) for episode in episodes for seed in wanted_seeds for key in points}
        rows = read_rows(directory / filename)
        observed = [(int(row["validation_episode"]) if row.get("validation_episode") else None,
                     int(row["seed"]), row["preference_key"]) for row in rows]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError(f"missing, duplicate or wrong evaluation cells: {filename}")
        for row in rows:
            if row["subset_sha256"] != subset_hash or row["sampling_repeat"] != "0":
                raise ValueError("subset or repeat differs from the comparison contract")
            if row["normalization_manifest_sha256"] != config["objective_scalarizer"]["normalization_manifest_sha256"]:
                raise ValueError("evaluation used a different scale manifest")
            for name, digest in protocol_hashes(config).items():
                if row[name] != digest:
                    raise ValueError(f"evaluation {name} differs from the dataset protocol")
            if row["heuristic_comparison_valid"] == "False" and any(row[field] for field in (
                "relative_heuristic_gap_percent", "makespan_heuristic_gap_percent",
                "reconfiguration_cost_heuristic_gap_percent", "worker_load_variance_heuristic_gap_percent")):
                raise ValueError("invalid heuristic comparison produced a numeric gap")
        checks[filename] = {"cell_count": len(rows), "instance_count": len(wanted_seeds),
                            "feasibility_counts": dict(Counter(row["feasibility_status"] for row in rows))}
    return {"passed": True, "training_episode_count": len(train_rows), "evaluation_checks": checks,
            "best_episode": checkpoint["checkpoint_episode"], "subset_sha256": main["subset_sha256"]}


def main():
    root = project_path("result/audits/data_v2")
    report = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    results = {name: audit_run(run["run_directory"]) for name, run in report["runs"].items()}
    if len({value["subset_sha256"] for value in results.values()}) != 1:
        raise ValueError("comparison arms used different main validation subsets")
    write_json(root / "run_acceptance.json", {"legacy_files_verified": verify_legacy_files(), "runs": results})
    from scripts.data_v2_comparison import append_runtime_stats
    append_runtime_stats(root / "comparison.json")
    print(f"Run audit passed: {root / 'run_acceptance.json'}")


if __name__ == "__main__":
    main()
