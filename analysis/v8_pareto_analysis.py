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
from configs.formal_preferences import formal_preferences, objective_scales
from data.dataset import sha256_file
from environment import PreferenceContext
from result.io import write_csv, write_json
from result.metrics import evaluation_quality_metric, quality_metric_sha256, result_schema_version
from environment.workload import objective_names, is_modern_workload, metrics_workload_objective
from .workload_dominance import resolve_workload_evaluation_config, rescore_workload_row, workload_identity, labelled_workload_dominance
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


def _validate_explicit_scoring(directory: Path, config: dict, metrics: dict, rows: list[dict]) -> dict:
    """Check the saved scoring goal while retaining the executing training goal."""
    path = directory / "evaluation_config.json"
    if metrics.get("evaluation_goal") != "explicit_frozen_workload" or not path.is_file():
        raise ValueError("explicit evaluation requires its saved evaluation_config.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    scoring = load_config(path)
    if not is_modern_workload(scoring):
        raise ValueError("explicit workload evaluation requires a modern workload config")
    digest = effective_config_snapshot(raw)["sha256"]
    if metrics.get("evaluation_config_sha256") != digest:
        raise ValueError("evaluation scoring configuration hash mismatch")
    if metrics.get("training_evaluation_schema_version") != result_schema_version(config):
        raise ValueError("evaluation training result schema mismatch")
    if metrics.get("evaluation_schema_version") != result_schema_version(scoring):
        raise ValueError("evaluation scoring result schema mismatch")
    for name, expected in {
        "evaluation_objective_scales": scoring["objective_scalarizer"]["scales"],
        "evaluation_normalization_manifest_sha256": scoring["objective_scalarizer"].get("normalization_manifest_sha256"),
        **workload_identity(scoring, prefix="evaluation_"),
    }.items():
        if metrics.get(name) != expected:
            raise ValueError(f"evaluation scoring {name} mismatch")
    metric = evaluation_quality_metric(scoring)
    if metrics.get("quality_metric") != metric or metrics.get("quality_metric_sha256") != quality_metric_sha256(metric):
        raise ValueError("evaluation scoring quality metric mismatch")
    training_metric = evaluation_quality_metric(config)
    training_quality = metrics.get("training_goal_quality", {})
    training_digest = quality_metric_sha256(training_metric)
    if (training_quality.get("quality_metric") != training_metric
            or training_quality.get("quality_metric_sha256") != training_digest
            or metrics.get("provenance", {}).get("quality_metric_sha256") != training_digest):
        raise ValueError("evaluation original training quality metric mismatch")
    from .pareto_analysis import _validate_row_protocol
    for row in rows:
        training_row = dict(row)
        for name in ("workload_objective_value", "preference_quality_score", "actual_preference_quality_score",
                     "quality_score", "quality_metric_sha256", "result_schema_version"):
            if f"training_{name}" in row:
                training_row[name] = row[f"training_{name}"]
        for name in ("result_schema_version", "quality_metric_sha256", "preference_quality_score"):
            if f"training_{name}" not in row:
                raise ValueError(f"evaluation row is missing its original training_{name}")
        if not is_modern_workload(config):
            training_row["workload_objective_value"] = row["worker_load_variance"]
        _validate_row_protocol(training_row, config)
        if row.get("result_schema_version") != result_schema_version(scoring):
            raise ValueError("evaluation row scoring result schema mismatch")
        if row.get("quality_metric_sha256") != quality_metric_sha256(metric):
            raise ValueError("evaluation row scoring quality hash mismatch")
        for name, expected in {
            **workload_identity(scoring, prefix="evaluation_"),
            "evaluation_flow_mode": metric["flow_mode"],
            "evaluation_normalization_manifest_sha256": scoring["objective_scalarizer"].get("normalization_manifest_sha256"),
        }.items():
            actual = row.get(name)
            if name == "evaluation_workload_q":
                actual = None if actual in {None, "", "None", "null"} else float(actual)
            if actual != expected:
                raise ValueError(f"evaluation row {name} mismatch")
        for name in objective_names(scoring):
            actual = float(row[f"evaluation_objective_scale_{name}"])
            if not math.isclose(actual, scoring["objective_scalarizer"]["scales"][name], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"evaluation row scoring {name} scale mismatch")
        recomputed = rescore_workload_row(row, scoring)
        for name in ("workload_objective_value", "preference_quality_score", "quality_score", "actual_preference_quality_score"):
            actual = float(row[name])
            if not math.isfinite(actual) or not math.isclose(actual, recomputed[name], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"evaluation row scoring {name} disagrees with its actual objectives")
    return {
        "evaluation_config_sha256": sha256_file(path),
        "evaluation_effective_config_sha256": digest,
        "evaluation_scales": scoring["objective_scalarizer"]["scales"],
        "evaluation_workload": workload_identity(scoring),
    }


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
    explicit_scoring = isinstance(metrics, dict) and (
        metrics.get("evaluation_goal") is not None or (directory / "evaluation_config.json").is_file()
    )
    expected_schema = result_schema_version(config)
    if not isinstance(metrics, dict) or (not explicit_scoring and metrics.get("evaluation_schema_version") != expected_schema):
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
    from environment.types import flow_mode
    if provenance.get("flow_mode", "raw_v1") != flow_mode(config):
        raise ValueError("evaluation Flow mode mismatch")
    if not provenance.get("dataset_manifest_sha256"):
        raise ValueError("evaluation has no dataset manifest hash")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scoring_source = _validate_explicit_scoring(directory, config, metrics, rows) if explicit_scoring else {}
    source = {
        "directory": str(directory.resolve()),
        "config_sha256": sha256_file(config_path),
        "metadata_sha256": sha256_file(metadata_path),
        "rows_sha256": sha256_file(csv_path),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "network_weights_sha256": provenance.get("network_weights_sha256"),
        "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
        "flow_mode": flow_mode(config),
        "training_scales": config["objective_scalarizer"]["scales"],
        "reward_version": config["reward"]["mode"],
        "training_workload": workload_identity(config) if is_modern_workload(config) else {"workload_mode": "variance"},
        **scoring_source,
    }
    return config, metrics, rows, source


def analyze_runs(
    runs: Mapping[str, str | Path | Sequence[str | Path]], output_dir: str | Path,
    *, evaluation_flow_mode: str | None = None, evaluation_config: Mapping[str, Any] | None = None,
    own_goal: bool = False,
) -> dict[str, Any]:
    """Analyze matched candidate budgets separately for each instance and seed.

    Each repeat produces its own 66-candidate HV; pooled-repeat HV uses all
    sampled candidates for the same instance. Failed/unsafe trajectories are
    counted for completion and excluded from the empirical feasible front.
    """
    if not runs:
        raise ValueError("at least one V8 run is required")
    from environment.types import FLOW_RAW, FLOW_EXCESS, flow_mode
    if evaluation_flow_mode not in {None, FLOW_RAW, FLOW_EXCESS}:
        raise ValueError("unknown evaluation_flow_mode")
    evaluation_scales = ((368.3143,353.27,2.2629) if evaluation_flow_mode == FLOW_EXCESS
                         else (1089.15,353.27,2.2629))
    lower_bounds = {}
    candidates: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    sources = []
    fronts: dict[tuple[str, int, str], list[tuple[float, float, float]]] = {}
    seen_runs: set[tuple[str, int]] = set()
    shared_identity = None
    shared_config = None
    seeds_by_training_seed: dict[int, list[int]] = {}
    prepared_runs = {}
    for method, directories in runs.items():
        for path in ([directories] if isinstance(directories, (str, Path)) else list(directories)):
            prepared_runs[(method, str(path))] = _load_run(Path(path))
    workload_comparison = evaluation_config is not None or any(
        is_modern_workload(loaded[0]) or "evaluation_workload" in loaded[3]
        for loaded in prepared_runs.values()
    )
    shared_scoring_config = None
    for method, directories in runs.items():
        paths = [directories] if isinstance(directories, (str, Path)) else list(directories)
        if not paths:
            raise ValueError(f"method has no runs: {method}")
        for raw_path in paths:
            config, metrics, rows, source = prepared_runs[(method, str(raw_path))]
            scoring_config = (resolve_workload_evaluation_config(config, evaluation_config=evaluation_config,
                own_goal=own_goal, flow_mode=evaluation_flow_mode) if workload_comparison else config)
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
                        objective_scales(config),
                        config["objective_scalarizer"].get("normalization_manifest_sha256"))
            if evaluation_flow_mode is not None:
                # Training identities are validated per run above; compare shared physical/evaluation identity.
                identity = (*identity[:5], evaluation_scales, json.dumps(config.get("environment",{}),sort_keys=True))
                if not lower_bounds and evaluation_flow_mode == FLOW_EXCESS:
                    from data import load_dataset_split
                    from environment import AssemblySchedulingEnv
                    dataset = load_dataset_split(config,metrics["dataset"])
                    if dataset_manifest_snapshot_hash(dataset) != source["dataset_manifest_sha256"]:
                        raise ValueError("cannot derive excess from an unverified instance manifest")
                    env = AssemblySchedulingEnv(config)
                    for record in dataset:
                        if record.instance.instance_id in ids:
                            env.reset(record.instance,build_observation=False)
                            lower_bounds[record.instance.instance_id] = env._total_processing_lower_bound_ticks*env.resolution
            if workload_comparison:
                identity = (*identity[:5], objective_scales(scoring_config),
                    (json.dumps(workload_identity(scoring_config), sort_keys=True),
                     scoring_config["objective_scalarizer"].get("normalization_manifest_sha256"),
                     json.dumps(config.get("environment", {}), sort_keys=True)))
                if flow_mode(scoring_config) == FLOW_EXCESS and not lower_bounds:
                    from data import load_dataset_split
                    from environment import AssemblySchedulingEnv
                    dataset = load_dataset_split(config, metrics["dataset"])
                    if dataset_manifest_snapshot_hash(dataset) != source["dataset_manifest_sha256"]:
                        raise ValueError("cannot derive excess from an unverified instance manifest")
                    environment = AssemblySchedulingEnv(config)
                    for record in dataset:
                        if record.instance.instance_id in ids:
                            environment.reset(record.instance, build_observation=False)
                            lower_bounds[record.instance.instance_id] = environment._total_processing_lower_bound_ticks * environment.resolution
            if shared_identity is None:
                shared_identity, shared_config = identity, config
                shared_scoring_config = scoring_config
            elif identity != shared_identity:
                raise ValueError("runs have unmatched dataset, instances, preferences, repeat budget, or scales")
            expected_cells = {(instance_id, key, repeat) for instance_id in ids for key in keys for repeat in range(repeats)}
            observed_cells = {(r["instance_id"], r["preference_key"], int(r["sampling_repeat"])) for r in rows}
            if len(observed_cells) != len(rows) or observed_cells != expected_cells:
                raise ValueError("duplicate or incomplete instance/preference/repeat cells")
            scales = identity[5]
            load_sources = {}
            if workload_comparison and any(row.get("worker_loads") in {None, ""} for row in rows if _flag(row["task_succeeded"])):
                from .workload_dominance import historical_load_sources
                load_sources = historical_load_sources(raw_path, config, metrics["dataset"],
                    [row for row in rows if _flag(row["task_succeeded"])])
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for raw in rows:
                repeat = int(raw["sampling_repeat"])
                sampling_seed = int(sampling_seeds[repeat])
                if int(raw["sampling_seed"]) != sampling_seed:
                    raise ValueError("row sampling seed disagrees with repeat metadata")
                weights = [float(raw[f"preference_{name}"]) for name in objective_names(config)]
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
                objectives = [_finite(raw[name], name) for name in OBJECTIVE_FIELDS]
                selected_flow_mode = flow_mode(scoring_config) if workload_comparison else evaluation_flow_mode
                if selected_flow_mode == FLOW_EXCESS and succeeded:
                    lb = lower_bounds.get(raw["instance_id"])
                    if lb is None or objectives[0] < lb-1e-8:
                        raise ValueError("successful excess derivation has invalid processing lower bound")
                    expected_excess = max(0.0, objectives[0]-lb)
                    if raw.get("flow_excess_objective") not in {None,""} and not math.isclose(
                            _finite(raw["flow_excess_objective"],"flow_excess_objective"),expected_excess,abs_tol=1e-7):
                        raise ValueError("excess report disagrees with verified instance lower bound")
                    raw = {**raw, "flow_excess_objective": expected_excess}
                    objectives[0] = expected_excess
                if workload_comparison and succeeded and safe:
                    if is_modern_workload(config):
                        from .pareto_analysis import _validate_row_protocol
                        _validate_row_protocol(raw, config)
                    raw = rescore_workload_row(raw, scoring_config,
                        **load_sources.get((raw["instance_id"], str(raw.get("candidate_id", "")), str(raw.get("sampling_repeat", 0))), {}))
                    objectives[2] = metrics_workload_objective(raw, scoring_config)
                    objectives[0] = _finite(raw["flow_excess_objective"] if flow_mode(scoring_config) == FLOW_EXCESS else raw["flow_time_objective"], "evaluation Flow")
                point = normalize_objectives(objectives, scales) if not workload_comparison or succeeded and safe else (None, None, None)
                row = {**raw, "method": method, "algorithm_seed": seed,
                       "sampling_repeat": repeat, "candidate_feasible": succeeded and safe,
                       "is_pareto": False,
                       **dict(zip(tuple(f"normalized_{name}" for name in objective_names(scoring_config)), point)),
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
        "normalization": "J/(scale+J)", "objective_scales": dict(zip(objective_names(shared_scoring_config),shared_identity[5])),
        "evaluation_flow_mode": evaluation_flow_mode or flow_mode(shared_scoring_config),
        "evaluation_goal": "own_training_goal" if own_goal else "unified_q05" if workload_comparison and evaluation_config is None else "explicit_goal" if workload_comparison else "legacy_variance",
        "reference_point": [1.0, 1.0, 1.0], "methods": method_summary,
        "coverage_definition": "fraction of unique front points weakly dominated by the other method",
        "candidate_count": len(candidates), "front_row_count": sum(r["is_pareto"] for r in candidates),
        "sources": sources,
    }
    if workload_comparison:
        summary.update(workload_identity(shared_scoring_config, prefix="evaluation_"))
        summary["evaluation_normalization_manifest_sha256"] = shared_scoring_config["objective_scalarizer"].get("normalization_manifest_sha256")
        summary["labelled_worker_dominance"] = labelled_workload_dominance(candidates)
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


def dataset_manifest_snapshot_hash(dataset) -> str:
    from data.dataset import canonical_json_bytes,sha256_bytes
    return sha256_bytes(canonical_json_bytes(dataset.manifest))


def analyze_dual_flow_runs(runs, output_dir):
    """One common raw primary metric and one common excess supplementary metric."""
    from environment.types import FLOW_RAW,FLOW_EXCESS
    output=Path(output_dir)
    primary=analyze_runs(runs,output/"raw_primary",evaluation_flow_mode=FLOW_RAW)
    supplementary=analyze_runs(runs,output/"excess_supplementary",evaluation_flow_mode=FLOW_EXCESS)
    summary={"primary":primary,"supplementary":supplementary}
    write_json(output/"dual_flow_summary.json",summary)
    return summary
