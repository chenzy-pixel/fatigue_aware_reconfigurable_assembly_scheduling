"""Paired ablation comparisons from a completed, source-backed evaluation batch."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from configs import load_config, project_path
from configs.formal_preferences import formal_preferences
from data import load_dataset_split
from data.dataset import sha256_file
from data.selection import subset_snapshot
from result.provenance import effective_config_snapshot, dataset_manifest_snapshot
from utils import configured_formal_evaluation_sampling_seeds, derive_evaluation_sampling_seed, SAMPLED_EVALUATION_RNG_VERSION
from result.io import write_csv, write_json
from result.metrics import evaluation_quality_metric, quality_metric_sha256
from scripts.ablation_protocol import ROLE_CONFIGS, latest_evaluation_manifest, protocol_profile

PAIRS = (
    ("graph_propagation", "universal", "no_graph"),
    ("objective_experts", "universal", "shared_head"),
    ("fatigue_flow", "full_flow", "neutral_flow"),
    ("fatigue_cost", "full_cost", "neutral_cost"),
    ("fatigue_variance", "full_variance", "neutral_variance"),
)
FIELDS = (
    "flow_time_objective", "reconfiguration_cost", "worker_load_variance",
    "completed_reconfigurations", "completed_reconfigurations_per_operation",
    "worker_reconfiguration_busy_minutes", "mean_interstage_idle_minutes",
    "max_consecutive_worker_stages", "fatigue_monitor_peak",
    "fatigue_monitor_over_limit_worker_ratio", "fatigue_monitor_over_limit_minutes",
    "fatigue_monitor_over_limit_area", "fatigue_monitor_over_limit_time_ratio",
    "fatigue_monitor_over_limit_area_ratio", "machine_waiting_for_worker_time", "forced_recovery_wait_count",
)
CONTROL_FIELDS = (
    "algorithm_seed", "generator_version", "dataset_manifest_sha256", "subset_sha256",
    "normalization_manifest_sha256", "quality_metric_sha256", "terminal_failure_penalty_configured",
    "objective_scalarizer_type", "objective_scalarizer_rho", "objective_scale_flow",
    "objective_scale_cost", "objective_scale_variance", "sampling_rng_version",
    "sampling_seed", "derived_sampling_seed",
)


def _true(value) -> bool:
    return str(value).lower() in {"true", "1"}


def _key(row: dict) -> tuple[str, str, str]:
    return row["instance_id"], str(row["sampling_repeat"]), row["preference_key"]


def paired_rows(baseline: list[dict], variant: list[dict], experiment: str) -> tuple[list[dict], dict]:
    left, right = ({_key(row): row for row in rows} for rows in (baseline, variant))
    if len(left) != len(baseline) or len(right) != len(variant):
        raise ValueError(f"duplicate evaluation cell: {experiment}")
    if not left or left.keys() != right.keys():
        raise ValueError(f"unmatched evaluation cells: {experiment}")
    deltas = defaultdict(list)
    rows = []
    for key in sorted(left):
        first, second = left[key], right[key]
        for field in CONTROL_FIELDS:
            if first.get(field) in (None, "") or second.get(field) in (None, ""):
                raise ValueError(f"missing {field}: {experiment}, {key}")
            if first.get(field) != second.get(field):
                raise ValueError(f"unmatched {field}: {experiment}, {key}")
        row = {"experiment": experiment, "instance_id": key[0], "sampling_repeat": key[1],
            "preference_key": key[2], "baseline_success": _true(first["task_succeeded"]),
            "variant_success": _true(second["task_succeeded"]),
            "baseline_physical_safe": _true(first.get("physical_safety_pass", False)),
            "variant_physical_safe": _true(second.get("physical_safety_pass", False))}
        if row["baseline_success"] and row["variant_success"]:
            for field in FIELDS:
                if first.get(field) not in (None, "") and second.get(field) not in (None, ""):
                    delta = float(second[field]) - float(first[field])
                    row[f"delta_{field}"] = delta
                    deltas[field].append(delta)
        rows.append(row)
    summary = {"experiment": experiment, "cell_count": len(rows),
        "baseline_completed": sum(r["baseline_success"] for r in rows),
        "variant_completed": sum(r["variant_success"] for r in rows),
        "baseline_physical_safe": sum(r["baseline_physical_safe"] for r in rows),
        "variant_physical_safe": sum(r["variant_physical_safe"] for r in rows),
        "common_success_count": sum(r["baseline_success"] and r["variant_success"] for r in rows),
        **{f"mean_delta_{field}": mean(values) for field, values in deltas.items()}}
    return rows, summary


def _parameter_count(path: str) -> int:
    state = torch.load(path, map_location="cpu", weights_only=True)["network"]
    return sum(tensor.numel() for name, tensor in state.items() if not name.endswith(".directions"))


def validate_role_evaluation(role: str, entry: dict, rows: list[dict], metrics: dict, seed: int) -> dict:
    """Verify role identity, source files, and the complete formal test matrix."""
    def require(condition: bool, detail: str) -> None:
        if not condition:
            raise ValueError(f"{role}: {detail}")

    require(entry.get("config") == ROLE_CONFIGS[role], "role config mismatch")
    expected = load_config(ROLE_CONFIGS[role])
    expected["seed"] = seed
    directory = Path(entry["evaluation_run"])
    config = json.loads((directory / "config.json").read_text(encoding="utf-8-sig"))
    require(protocol_profile(config) == protocol_profile(expected), "evaluation protocol mismatch")
    require(int(config["seed"]) == seed, "evaluation seed mismatch")
    training = Path(entry["training_run"])
    training_config = json.loads((training / "config.json").read_text(encoding="utf-8-sig"))
    summary = json.loads((training / "summary.json").read_text(encoding="utf-8-sig"))
    require(protocol_profile(training_config) == protocol_profile(expected), "training protocol mismatch")
    require(int(training_config["seed"]) == seed and int(summary["episodes"]) == expected["training"]["episodes"]
            and summary["checkpoint_selection"]["has_best"], "training run incomplete or wrong seed")
    checkpoint = Path(entry["checkpoint"])
    require(checkpoint.resolve() == (training / "best_checkpoint.pt").resolve(), "best checkpoint source mismatch")
    checkpoint_hash = sha256_file(checkpoint)
    provenance = metrics["provenance"]
    require(checkpoint_hash == summary["provenance"]["checkpoint_sha256"] == provenance["checkpoint_sha256"],
            "checkpoint hash mismatch")
    config_hash = effective_config_snapshot(config)["sha256"]
    require(config_hash == provenance["effective_config_sha256"], "effective config hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    require(protocol_profile(metadata["effective_config"]) == protocol_profile(expected)
            and int(metadata["algorithm_seed"]) == seed, "checkpoint protocol mismatch")
    selectors = {"encoder_variant": expected["network"].get("encoder_variant", "hetero_gnn"),
                 "actor_head_variant": expected["network"].get("actor_head_variant", "objective_experts"),
                 "fatigue_mode": expected["environment"].get("fatigue_mode", "full")}
    require(all(payload["network_spec"].get(key, default) == selectors[key] for key, default in
                (("encoder_variant", "hetero_gnn"), ("actor_head_variant", "objective_experts")))
            and payload["network_spec"]["observation_schema_version"] == 6
            and metadata["runtime_manifest"].get("fatigue_mode", "full") == selectors["fatigue_mode"],
            "checkpoint variant mismatch")
    dataset = load_dataset_split(expected, "test")
    indices = list(range(len(dataset)))
    selection = subset_snapshot(dataset, indices, role="evaluation")
    dataset_hash = dataset_manifest_snapshot(dataset.manifest_path)["sha256"]
    require(metrics.get("dataset") == "test" and metrics.get("policy") == "ppo"
            and metrics.get("decode_mode") == "sampled" and metrics.get("result_role") == "formal_sampled",
            "formal test evaluation required")
    require(metrics.get("instance_indices") == indices and metrics.get("subset_sha256") == selection["subset_sha256"]
            and provenance.get("evaluation_subset_sha256") == selection["subset_sha256"]
            and metrics.get("dataset_manifest_sha256") == dataset_hash == provenance.get("dataset_manifest_sha256"),
            "test dataset or subset mismatch")
    preferences = formal_preferences(expected, "final_test")
    sampling_seeds = configured_formal_evaluation_sampling_seeds(expected, "final_test")
    require(metrics.get("sampling_seeds") == sampling_seeds
            and int(metrics.get("repeat_count", 0)) == len(sampling_seeds), "sampling repeat mismatch")
    instance_ids = {record.instance.instance_id for record in dataset}
    expected_cells = {(instance_id, str(repeat), point.key) for instance_id in instance_ids
                      for repeat in range(len(sampling_seeds)) for point in preferences}
    require(len(rows) == len(expected_cells) and {_key(row) for row in rows} == expected_cells,
            "incomplete or duplicate evaluation matrix")
    require(int(metrics.get("cell_count", metrics["instance_count"])) == len(rows)
            and int(metrics.get("unique_instance_count", metrics["instance_count"])) == len(dataset)
            and int(metrics.get("preference_count", 1)) == len(preferences), "aggregate cell count mismatch")
    controls = {"algorithm_seed": seed, "generator_version": expected["generator"]["version"],
                "dataset_manifest_sha256": dataset_hash, "subset_sha256": selection["subset_sha256"],
                "normalization_manifest_sha256": expected["objective_scalarizer"]["normalization_manifest_sha256"],
                "quality_metric_sha256": quality_metric_sha256(evaluation_quality_metric(expected)),
                "terminal_failure_penalty_configured": expected["reward"]["terminal_failure_penalty"],
                "objective_scalarizer_type": expected["objective_scalarizer"]["type"],
                "objective_scalarizer_rho": expected["objective_scalarizer"]["rho"],
                "sampling_rng_version": SAMPLED_EVALUATION_RNG_VERSION,
                "checkpoint_sha256": checkpoint_hash, "effective_config_sha256": config_hash,
                "dataset": "test", **selectors,
                **{f"objective_scale_{key}": value for key, value in expected["objective_scalarizer"]["scales"].items()}}
    require(provenance.get("normalization_manifest_sha256") == controls["normalization_manifest_sha256"]
            and metrics.get("quality_metric_sha256") == provenance.get("quality_metric_sha256") == controls["quality_metric_sha256"], "aggregate protocol mismatch")
    required = ("task_succeeded", "active_constraint_pass", "physical_safety_pass")
    finite_fields = ("flow_time_objective", "reconfiguration_cost", "worker_load_variance",
                     "fatigue_monitor_peak", "fatigue_monitor_over_limit_minutes", "fatigue_monitor_over_limit_area")
    for row in rows:
        for key, value in controls.items():
            actual = row.get(key)
            match = (actual not in (None, "") and float(actual) == float(value)) if isinstance(value, (int, float)) else actual == value
            require(match, f"row {key} mismatch")
        require(all(key in row for key in FIELDS) and all(row.get(key) not in (None, "") for key in required),
                "missing monitor or safety fields")
        require(all(row.get(key) not in (None, "") and math.isfinite(float(row[key])) for key in finite_fields),
                "nonfinite objective or monitor fields")
        require(int(row["sampling_seed"]) == sampling_seeds[int(row["sampling_repeat"])], "row sampling seed mismatch")
        key = row["preference_key"] if role in ("universal", "no_graph", "shared_head") else None
        require(int(row["derived_sampling_seed"]) == derive_evaluation_sampling_seed(
            int(row["sampling_seed"]), row["instance_id"], key), "derived sampling seed mismatch")
    return {"evaluation_run": str(directory), "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash, "effective_config_sha256": config_hash,
            "dataset_manifest_sha256": dataset_hash}


def summarize(manifest_path: Path, output: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (manifest.get("status") != "complete" or manifest.get("schema") != "matched_ablation_evaluation_v1"
            or set(manifest.get("runs", {})) != set(ROLE_CONFIGS)):
        raise ValueError("ablation evaluation batch must be complete")
    loaded = {}
    sources = {}
    for role in {name for pair in PAIRS for name in pair[1:]}:
        entry = manifest["runs"][role]
        directory = Path(entry["evaluation_run"])
        with (directory / "instance_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8-sig"))
        sources[role] = validate_role_evaluation(role, entry, rows, metrics, int(manifest["seed"]))
        loaded[role] = (rows, metrics)
    comparisons, summaries = [], []
    for experiment, baseline, variant in PAIRS:
        left, old_metrics = loaded[baseline]
        right, new_metrics = loaded[variant]
        for field in ("dataset_manifest_sha256", "subset_sha256", "quality_metric_sha256", "normalization_manifest_sha256"):
            if old_metrics.get(field) != new_metrics.get(field):
                raise ValueError(f"aggregate {field} mismatch: {experiment}")
        cells, row = paired_rows(left, right, experiment)
        for cell in cells:
            cell.update(baseline=baseline, variant=variant)
        row.update(baseline=baseline, variant=variant,
            baseline_parameters=_parameter_count(manifest["runs"][baseline]["checkpoint"]),
            variant_parameters=_parameter_count(manifest["runs"][variant]["checkpoint"]),
            baseline_inference_seconds=old_metrics.get("total_inference_time_seconds"),
            variant_inference_seconds=new_metrics.get("total_inference_time_seconds"))
        comparisons.extend(cells)
        summaries.append(row)
    output.mkdir(parents=True, exist_ok=False)
    # Different variants may expose different monitor columns; retain their union.
    columns = sorted({key for row in comparisons for key in row})
    write_csv(output / "paired_cells.csv", [{key: row.get(key) for key in columns} for row in comparisons])
    columns = sorted({key for row in summaries for key in row})
    write_csv(output / "summary.csv", [{key: row.get(key) for key in columns} for row in summaries])
    write_json(output / "manifest.json", {"seed": manifest["seed"], "evaluation_manifest": str(manifest_path), "sources": sources})
    lines = [f"# Seed {manifest['seed']} ablation comparisons", "",
        "Objective differences are variant minus baseline on common successful cells. Completion counts include all matched cells.",
        "These are descriptive results from one training seed. Neutral-fatigue exposure is a counterfactual audit of its realized schedule.", "",
        "| Experiment | Cells | Completed: baseline / variant | Common success | Δ Flow | Δ Cost | Δ Variance | Δ Fatigue area |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        values = ["—" if row.get(f"mean_delta_{field}") is None else f"{row['mean_delta_' + field]:+.4f}" for field in
            ("flow_time_objective", "reconfiguration_cost", "worker_load_variance", "fatigue_monitor_over_limit_area")]
        lines.append(f"| {row['experiment']} | {row['cell_count']} | {row['baseline_completed']} / {row['variant_completed']} | {row['common_success_count']} | " + " | ".join(values) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize five paired ablation comparisons")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = args.manifest or latest_evaluation_manifest(args.seed)
    output = args.output_dir or project_path("result/analysis") / manifest.stem
    print(summarize(manifest, output))


if __name__ == "__main__":
    main()
