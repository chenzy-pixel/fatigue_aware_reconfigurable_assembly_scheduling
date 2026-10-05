"""Per-instance empirical Pareto/HV analysis of complete V8 sampled grids."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from configs import load_config
from configs.formal_preferences import formal_preferences
from data.dataset import sha256_file
from environment import PreferenceContext
from result.io import write_csv, write_json
from result.metrics import EVALUATION_SCHEMA_VERSION
from result.provenance import effective_config_snapshot
from utils import derive_evaluation_sampling_seed
from .pareto_analysis import dominates, hypervolume_3d, nondominated_indices, normalize_objectives, vectors_equal

OBJECTIVE_FIELDS = ("flow_time_objective", "reconfiguration_cost", "worker_load_variance")
PROTOCOL = "v8_sampled_grid_pareto_v1"


def _flag(value: Any) -> bool:
    if value is True or str(value).lower() in {"true", "1"}:
        return True
    if value is False or str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _finite(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _unique(points: Sequence[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    result = []
    for point in points:
        if not any(vectors_equal(point, previous) for previous in result):
            result.append(point)
    return result


def _coverage(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> float | None:
    return (sum(any(dominates(a, b) or vectors_equal(a, b) for a in first) for b in second) / len(second)) if second else None


def parse_run_arguments(values: Sequence[str]) -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        label, separator, path = value.partition("=")
        if not separator:
            path = label
            label = Path(path).name
        if not label or not path:
            raise ValueError("--run-dir requires [LABEL=]PATH")
        runs[label].append(Path(path))
    return dict(runs)


def _load_run(directory: Path) -> tuple[dict, dict, list[dict[str, str]], dict]:
    config_path = directory / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = load_config(config_path)
    if config["preference"]["quality"]["mode"] != "universal_sobol_v1":
        raise ValueError("V8 Pareto analysis requires a Universal evaluation")
    if config.get("environment", {}).get("fatigue_mode", "full") != "full":
        raise ValueError("fatigue-neutral trajectories cannot enter the safe-problem Pareto set")
    if (directory / "metrics.json").is_file():
        metadata_path, csv_path = directory / "metrics.json", directory / "instance_metrics.csv"
        metrics = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata_path, csv_path = directory / "summary.json", directory / "final_sampled_instance_metrics.csv"
        metrics = json.loads(metadata_path.read_text(encoding="utf-8"))["final_sampled"]
    if not isinstance(metrics, dict) or metrics.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("V8 analysis requires the current evaluation schema")
    if not metrics.get("evaluation_complete", True) or metrics.get("sampling_truncated_count", 0):
        raise ValueError("incomplete evaluations cannot enter Pareto/HV analysis")
    provenance = metrics.get("provenance", {})
    if provenance.get("formal_evaluation_stage") != "final_test":
        raise ValueError("V8 Pareto analysis requires the final_test preference grid")
    if metrics.get("decode_mode") != "sampled":
        raise ValueError("V8 Pareto analysis requires sampled decoding")
    if provenance.get("effective_config_sha256") != effective_config_snapshot(raw_config)["sha256"]:
        raise ValueError("evaluation effective configuration hash mismatch")
    if provenance.get("objective_scales") != config["objective_scalarizer"]["scales"]:
        raise ValueError("evaluation objective scales mismatch")
    if provenance.get("normalization_manifest_sha256") != config["objective_scalarizer"].get("normalization_manifest_sha256"):
        raise ValueError("evaluation normalization manifest mismatch")
    if not provenance.get("dataset_manifest_sha256"):
        raise ValueError("evaluation has no dataset manifest hash")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    source = {
        "directory": str(directory.resolve()),
        "config_sha256": sha256_file(config_path),
        "metadata_sha256": sha256_file(metadata_path),
        "rows_sha256": sha256_file(csv_path),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "network_weights_sha256": provenance.get("network_weights_sha256"),
        "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
    }
    return config, metrics, rows, source


def analyze_runs(
    runs: Mapping[str, str | Path | Sequence[str | Path]], output_dir: str | Path,
) -> dict[str, Any]:
    """Analyze matched candidate budgets separately for each instance and seed.

    Each repeat produces its own 66-candidate HV; pooled-repeat HV uses all
    sampled candidates for the same instance. Failed/unsafe trajectories are
    counted for completion and excluded from the empirical feasible front.
    """
    if not runs:
        raise ValueError("at least one V8 run is required")
    candidates: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    sources = []
    fronts: dict[tuple[str, int, str], list[tuple[float, float, float]]] = {}
    seen_runs: set[tuple[str, int]] = set()
    shared_identity = None
    shared_config = None
    seeds_by_training_seed: dict[int, list[int]] = {}
    for method, directories in runs.items():
        paths = [directories] if isinstance(directories, (str, Path)) else list(directories)
        if not paths:
            raise ValueError(f"method has no runs: {method}")
        for raw_path in paths:
            config, metrics, rows, source = _load_run(Path(raw_path))
            seed = int(config["seed"])
            if (method, seed) in seen_runs:
                raise ValueError(f"duplicate method/training seed: {method}/{seed}")
            seen_runs.add((method, seed))
            grid = formal_preferences(config, "final_test")
            keys = {point.key for point in grid}
            if len(keys) != 66:
                raise ValueError("formal V8 final analysis requires 66 preferences")
            repeats = int(metrics["repeat_count"])
            sampling_seeds = list(metrics["sampling_seeds"])
            if repeats < 1 or len(sampling_seeds) != repeats or len(set(sampling_seeds)) != repeats:
                raise ValueError("invalid sampled repeat/seed metadata")
            if seed in seeds_by_training_seed and sampling_seeds != seeds_by_training_seed[seed]:
                raise ValueError("methods use unmatched evaluation sampling seeds")
            seeds_by_training_seed[seed] = sampling_seeds
            ids = {row["instance_id"] for row in rows}
            if len(ids) != int(metrics["instance_count"]):
                raise ValueError("instance count differs from evaluation metadata")
            identity = (metrics["dataset"], source["dataset_manifest_sha256"],
                        tuple(sorted(ids)), tuple(sorted(keys)), repeats,
                        tuple(config["objective_scalarizer"]["scales"][name] for name in ("flow", "cost", "variance")),
                        config["objective_scalarizer"].get("normalization_manifest_sha256"))
            if shared_identity is None:
                shared_identity, shared_config = identity, config
            elif identity != shared_identity:
                raise ValueError("runs have unmatched dataset, instances, preferences, repeat budget, or scales")
            expected_cells = {(instance_id, key, repeat) for instance_id in ids for key in keys for repeat in range(repeats)}
            observed_cells = {(r["instance_id"], r["preference_key"], int(r["sampling_repeat"])) for r in rows}
            if len(observed_cells) != len(rows) or observed_cells != expected_cells:
                raise ValueError("duplicate or incomplete instance/preference/repeat cells")
            scales = identity[5]
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for raw in rows:
                repeat = int(raw["sampling_repeat"])
                sampling_seed = int(sampling_seeds[repeat])
                if int(raw["sampling_seed"]) != sampling_seed:
                    raise ValueError("row sampling seed disagrees with repeat metadata")
                weights = [float(raw[f"preference_{name}"]) for name in ("flow", "cost", "variance")]
                if PreferenceContext.from_input(weights).key != raw["preference_key"]:
                    raise ValueError("row preference weights disagree with preference key")
                if int(raw["derived_sampling_seed"]) != derive_evaluation_sampling_seed(
                    sampling_seed, raw["instance_id"], raw["preference_key"]
                ):
                    raise ValueError("row derived sampling seed disagrees with its evaluation cell")
                succeeded = _flag(raw["task_succeeded"])
                if _flag(raw.get("sampling_truncated", raw["truncated"])):
                    raise ValueError("externally truncated evaluations cannot enter Pareto/HV analysis")
                if (_flag(raw["task_failed"]) == succeeded
                        or not _flag(raw["terminated"])):
                    raise ValueError("inconsistent task completion flags")
                safe = (int(raw["schedule_violation_count"]) == 0 and
                        _finite(raw.get("fatigue_monitor_peak", raw.get("maximum_worker_fatigue")), "fatigue peak")
                        <= _finite(raw["safe_fatigue_limit"], "fatigue limit") + 1e-9)
                if raw.get("fatigue_mode", "full") != "full":
                    raise ValueError("neutral row in full-fatigue evaluation")
                point = normalize_objectives([_finite(raw[name], name) for name in OBJECTIVE_FIELDS], scales)
                row = {**raw, "method": method, "algorithm_seed": seed,
                       "sampling_repeat": repeat, "candidate_feasible": succeeded and safe,
                       "is_pareto": False,
                       **dict(zip(("normalized_flow", "normalized_cost", "normalized_variance"), point)),
                       "_point": point}
                candidates.append(row)
                grouped[raw["instance_id"]].append(row)
            for instance_id, group in sorted(grouped.items()):
                feasible = [row for row in group if row["candidate_feasible"]]
                points = [row["_point"] for row in feasible]
                indices = nondominated_indices(points)
                for index in indices:
                    feasible[index]["is_pareto"] = True
                unique_front = _unique([points[index] for index in indices])
                fronts[(method, seed, instance_id)] = unique_front
                repeat_hvs = [hypervolume_3d(row["_point"] for row in feasible if row["sampling_repeat"] == repeat)
                              for repeat in range(repeats)]
                instances.append({
                    "method": method, "algorithm_seed": seed, "instance_id": instance_id,
                    "candidate_count": len(group), "completed_count": sum(_flag(r["task_succeeded"]) for r in group),
                    "feasible_count": len(feasible), "completion_rate": sum(_flag(r["task_succeeded"]) for r in group) / len(group),
                    "unique_front_size": len(unique_front), "pooled_hypervolume": hypervolume_3d(unique_front),
                    "mean_repeat_hypervolume": mean(repeat_hvs), "repeat_hypervolumes": json.dumps(repeat_hvs),
                })
            sources.append({"method": method, "algorithm_seed": seed, **source})
    methods = sorted(runs)
    seed_sets = [{seed for method_key, seed in seen_runs if method_key == method} for method in methods]
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("methods have unmatched training seeds")
    seed_summary = []
    for method, seed in sorted(seen_runs):
        group = [r for r in instances if r["method"] == method and r["algorithm_seed"] == seed]
        seed_summary.append({"method": method, "algorithm_seed": seed, "instance_count": len(group),
                             **{name: mean(r[field] for r in group) for name, field in (
                                 ("mean_completion_rate", "completion_rate"),
                                 ("mean_pooled_hypervolume", "pooled_hypervolume"),
                                 ("mean_repeat_hypervolume", "mean_repeat_hypervolume"),
                                 ("mean_unique_front_size", "unique_front_size"))}})
    paired = []
    by_instance = {(r["method"], r["algorithm_seed"], r["instance_id"]): r for r in instances}
    for first, second in combinations(methods, 2):
        for seed in sorted(seed_sets[0]):
            for instance_id in shared_identity[2]:
                a, b = fronts[(first, seed, instance_id)], fronts[(second, seed, instance_id)]
                paired.append({"baseline": first, "variant": second, "algorithm_seed": seed, "instance_id": instance_id,
                               "delta_pooled_hypervolume": by_instance[(second, seed, instance_id)]["pooled_hypervolume"] - by_instance[(first, seed, instance_id)]["pooled_hypervolume"],
                               "coverage_baseline_over_variant": _coverage(a, b),
                               "coverage_variant_over_baseline": _coverage(b, a),
                               "union_hypervolume": hypervolume_3d([*a, *b])})
    method_summary = {}
    for method in methods:
        group = [r for r in seed_summary if r["method"] == method]
        method_summary[method] = {"training_seed_count": len(group), **{
            field: {"mean": mean(r[field] for r in group), "std": stdev(r[field] for r in group) if len(group) > 1 else None}
            for field in ("mean_completion_rate", "mean_pooled_hypervolume", "mean_repeat_hypervolume")}}
    summary = {
        "protocol": PROTOCOL, "dataset": shared_identity[0], "independent_test": shared_identity[0] == "test",
        "dataset_manifest_sha256": shared_identity[1], "instance_count": len(shared_identity[2]),
        "preference_count": len(shared_identity[3]), "repeat_count": shared_identity[4],
        "normalization": "J/(scale+J)", "objective_scales": shared_config["objective_scalarizer"]["scales"],
        "reference_point": [1.0, 1.0, 1.0], "methods": method_summary,
        "coverage_definition": "fraction of unique front points weakly dominated by the other method",
        "candidate_count": len(candidates), "front_row_count": sum(r["is_pareto"] for r in candidates),
        "sources": sources,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for row in candidates:
        row.pop("_point")
    write_csv(output / "candidates.csv", candidates)
    write_csv(output / "pareto_front.csv", [r for r in candidates if r["is_pareto"]])
    write_csv(output / "instance_summary.csv", instances)
    write_csv(output / "seed_summary.csv", seed_summary)
    write_csv(output / "paired_instances.csv", paired)
    write_json(output / "summary.json", summary)
    lines = ["# V8 sampled Pareto analysis", "",
             f"Dataset: {summary['dataset']}; independent instances: {summary['instance_count']}; "
             f"preferences: 66; repeats: {summary['repeat_count']}.", "",
             "Each front belongs to one scheduling instance. Completion uses every evaluation cell; "
             "only successful, safe trajectories enter the empirical front. Empty feasible sets have HV=0.",
             "Pooled HV combines repeats for the same instance; mean-repeat HV reports each repeat's candidate set separately. "
             "Training-seed means are the units of the reported mean/std. Coverage counts weak dominance, including equal points.", "",
             "| Method | Seed | Mean completion | Mean pooled HV | Mean repeat HV |",
             "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['method']} | {r['algorithm_seed']} | {r['mean_completion_rate']:.4f} | "
              f"{r['mean_pooled_hypervolume']:.6f} | {r['mean_repeat_hypervolume']:.6f} |" for r in seed_summary]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
