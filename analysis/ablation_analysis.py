"""Paired seed-11 ablation comparison on identical evaluation cells."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from result.io import write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "result" / "runs"
OUTPUT = ROOT / "result" / "analysis" / "ablation_seed11"
PAIRS = (
    ("graph_propagation", "universal", "no_graph"),
    ("objective_experts", "universal", "shared_head"),
    ("fatigue_flow", "full_flow", "neutral_flow"),
    ("fatigue_cost", "full_cost", "neutral_cost"),
    ("fatigue_variance", "full_variance", "neutral_variance"),
)
CHECKPOINTS = {
    "universal": "universal_seed11_ep2000",
    "no_graph": "ablation_no_graph_seed11_ep2000",
    "shared_head": "ablation_shared_head_seed11_ep2000",
    "full_flow": "e1_flow_relative_time_seed11_ep1000_rerun_20260928_223621",
    "neutral_flow": "neutral_flow_seed11_ep1000",
    "full_cost": "e1_cost_seed11_ep1000_rerun_20260928_223621",
    "neutral_cost": "neutral_cost_seed11_ep1000",
    "full_variance": "e1_variance_seed11_ep1000_rerun_20260928_223621",
    "neutral_variance": "neutral_variance_seed11_ep1000",
}
FIELDS = (
    "flow_time_objective", "reconfiguration_cost", "worker_load_variance",
    "completed_reconfigurations", "completed_reconfigurations_per_operation",
    "completed_reconfigurations_per_minute", "worker_reconfiguration_busy_minutes",
    "mean_interstage_idle_minutes", "max_consecutive_worker_stages",
    "fatigue_monitor_peak", "fatigue_monitor_over_limit_worker_ratio",
    "fatigue_monitor_over_limit_minutes", "fatigue_monitor_over_limit_area",
    "fatigue_monitor_over_limit_time_ratio", "fatigue_monitor_over_limit_area_ratio",
    "machine_waiting_for_worker_time", "forced_recovery_wait_count",
)


def _read(label: str) -> tuple[list[dict[str, str]], dict]:
    directory = EVAL_ROOT / f"ablation_eval_{label}_seed11"
    with (directory / "instance_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    aggregate = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    return rows, aggregate


def _key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["instance_id"], row["sampling_repeat"], row["preference_key"])


def _parameter_count(label: str) -> int:
    checkpoint = EVAL_ROOT / CHECKPOINTS[label] / "best_checkpoint.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["network"]
    return sum(tensor.numel() for name, tensor in state.items()
               if isinstance(tensor, torch.Tensor) and not name.endswith(".directions"))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparisons = []
    summary = []
    sources: dict[str, dict[str, str | None]] = {}
    for experiment, baseline, variant in PAIRS:
        old, old_aggregate = _read(baseline)
        new, new_aggregate = _read(variant)
        for label, aggregate in ((baseline, old_aggregate), (variant, new_aggregate)):
            provenance = aggregate.get("provenance", {})
            sources[label] = {
                "checkpoint_sha256": provenance.get("checkpoint_sha256"),
                "effective_config_sha256": provenance.get("effective_config_sha256"),
                "dataset_manifest_sha256": provenance.get("dataset_manifest_sha256"),
            }
        old_cells, new_cells = ({_key(row): row for row in rows} for rows in (old, new))
        if len(old_cells) != len(old) or len(new_cells) != len(new):
            raise ValueError(f"duplicate evaluation cell: {experiment}")
        if old_cells.keys() != new_cells.keys():
            raise ValueError(f"unmatched evaluation cells: {experiment}")
        old_manifest = old_aggregate.get("dataset_manifest_sha256")
        new_manifest = new_aggregate.get("dataset_manifest_sha256")
        if old_manifest and new_manifest and old_manifest != new_manifest:
            raise ValueError(f"test manifest changed: {experiment}")
        paired_deltas: dict[str, list[float]] = defaultdict(list)
        completion = [0, 0]
        for key in sorted(old_cells):
            first, second = old_cells[key], new_cells[key]
            first_success = str(first["task_succeeded"]).lower() == "true"
            second_success = str(second["task_succeeded"]).lower() == "true"
            completion[0] += first_success
            completion[1] += second_success
            row = {"experiment": experiment, "baseline": baseline, "variant": variant,
                   "instance_id": key[0], "sampling_repeat": key[1],
                   "preference_key": key[2], "baseline_success": first_success,
                   "variant_success": second_success}
            if first_success and second_success:
                for field in FIELDS:
                    if first.get(field) not in (None, "") and second.get(field) not in (None, ""):
                        delta = float(second[field]) - float(first[field])
                        row[f"delta_{field}"] = delta
                        paired_deltas[field].append(delta)
            comparisons.append(row)
        row = {"experiment": experiment, "baseline": baseline, "variant": variant,
               "cell_count": len(old_cells), "baseline_completed": completion[0],
               "variant_completed": completion[1],
               "baseline_parameters": _parameter_count(baseline),
               "variant_parameters": _parameter_count(variant),
               "baseline_inference_seconds": old_aggregate.get("total_inference_time_seconds"),
               "variant_inference_seconds": new_aggregate.get("total_inference_time_seconds"),
               "common_success_count": sum(1 for cell in comparisons if cell["experiment"] == experiment
                                           and cell["baseline_success"] and cell["variant_success"])}
        row.update({f"mean_delta_{field}": mean(values) for field, values in paired_deltas.items() if values})
        summary.append(row)
    write_csv(OUTPUT / "paired_cells.csv", comparisons)
    write_csv(OUTPUT / "summary.csv", summary)
    write_json(OUTPUT / "manifest.json", {"training_seed": 11, "pairs": PAIRS,
                                          "metrics": FIELDS, "test_cells": len(comparisons),
                                          "sources": sources})
    lines = ["# Seed11 ablation comparisons", "",
             "Differences are variant minus baseline on the common successful cells.",
             "Completion counts use all matched cells. This is descriptive evidence from one training seed.",
             "Fatigue-neutral exposure is audited on its idealized execution timeline.", "",
             "| Experiment | Cells | Completed (base / variant) | Common success | Δ Flow | Δ Cost | Δ Variance | Δ Reconfigs | Δ Over-limit area |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        def value(name: str) -> str:
            item = row.get(f"mean_delta_{name}")
            return "—" if item is None else f"{item:+.4f}"
        lines.append("| {experiment} | {cell_count} | {baseline_completed} / {variant_completed} | "
                     "{common_success_count} | {flow} | {cost} | {variance} | {reconfigs} | {exposure} |".format(
                         **row, flow=value("flow_time_objective"),
                         cost=value("reconfiguration_cost"),
                         variance=value("worker_load_variance"),
                         reconfigs=value("completed_reconfigurations"),
                         exposure=value("fatigue_monitor_over_limit_area")))
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
