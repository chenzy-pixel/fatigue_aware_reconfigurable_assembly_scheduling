"""Pareto analysis for the current sampled multiobjective experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from configs import load_config, project_path
from configs.formal_preferences import formal_preferences, objective_scales
from environment import PreferenceContext, terminal_quality_score
from result.io import write_csv, write_json
from result.metrics import EVALUATION_SCHEMA_VERSION, evaluation_quality_metric, quality_metric_sha256

RELATIVE_TOLERANCE = 1e-9
REFERENCE_POINT = (1.0, 1.0, 1.0)
OBJECTIVE_FIELDS = ("flow_time_objective", "reconfiguration_cost", "worker_load_variance")
PROTOCOL_VERSION = "sampled_multiobjective_analysis_v2"

def _relative_slack(first: float, second: float, tolerance: float) -> float:
    return tolerance * max(1.0, abs(first), abs(second))


def vectors_equal(
    first: Sequence[float],
    second: Sequence[float],
    *,
    tolerance: float = RELATIVE_TOLERANCE,
) -> bool:
    """Return whether two objective vectors are equal within relative tolerance."""

    if len(first) != len(second):
        return False
    return all(
        abs(left - right) <= _relative_slack(left, right, tolerance)
        for left, right in zip(first, second, strict=True)
    )


def dominates(
    first: Sequence[float],
    second: Sequence[float],
    *,
    tolerance: float = RELATIVE_TOLERANCE,
) -> bool:
    """Return whether ``first`` Pareto-dominates ``second`` for minimization."""

    if len(first) != len(second):
        raise ValueError("objective vectors must have the same dimension")
    weakly_better = True
    strictly_better = False
    for left, right in zip(first, second, strict=True):
        slack = _relative_slack(left, right, tolerance)
        if left > right + slack:
            weakly_better = False
            break
        if left < right - slack:
            strictly_better = True
    return weakly_better and strictly_better


def nondominated_indices(
    points: Sequence[Sequence[float]],
    *,
    tolerance: float = RELATIVE_TOLERANCE,
) -> list[int]:
    """Return stable indices of all nondominated points, retaining equal points."""

    return [
        index
        for index, point in enumerate(points)
        if not any(
            other_index != index
            and dominates(other, point, tolerance=tolerance)
            for other_index, other in enumerate(points)
        )
    ]


def normalize_objectives(
    objectives: Sequence[float],
    scales: Sequence[float],
) -> tuple[float, float, float]:
    """Apply the canonical monotone bounded transform ``x / (scale + x)``."""

    if len(objectives) != 3 or len(scales) != 3:
        raise ValueError("normalization requires three objectives and three scales")
    normalized: list[float] = []
    for objective, scale in zip(objectives, scales, strict=True):
        value = float(objective)
        denominator_scale = float(scale)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("objectives must be finite and nonnegative")
        if not math.isfinite(denominator_scale) or denominator_scale <= 0.0:
            raise ValueError("objective scales must be finite and positive")
        normalized.append(value / (denominator_scale + value))
    return tuple(normalized)  # type: ignore[return-value]


def hypervolume_3d(points: Iterable[Sequence[float]], *, reference: Sequence[float] = REFERENCE_POINT) -> float:
    """Compute the exact union of dominated boxes by an x sweep."""
    bounds = tuple(float(value) for value in reference)
    if len(bounds) != 3 or not all(math.isfinite(value) for value in bounds):
        raise ValueError("the hypervolume reference point must be finite and three-dimensional")
    unique = set()
    for raw in points:
        point = tuple(float(value) for value in raw)
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            raise ValueError("hypervolume points must be finite and three-dimensional")
        if any(value > bound + _relative_slack(value, bound, RELATIVE_TOLERANCE) for value, bound in zip(point, bounds)):
            raise ValueError("a hypervolume point lies beyond the reference point")
        unique.add(tuple(min(value, bound) for value, bound in zip(point, bounds)))
    ordered = sorted(unique)
    xs = sorted({point[0] for point in ordered} | {bounds[0]})
    volume = 0.0
    for left, right in zip(xs, xs[1:]):
        active = sorted((y, z) for x, y, z in ordered if x <= left)
        area = 0.0
        minimum_z = bounds[2]
        for index, (y, z) in enumerate(active):
            minimum_z = min(minimum_z, z)
            next_y = active[index + 1][0] if index + 1 < len(active) else bounds[1]
            area += (next_y - y) * (bounds[2] - minimum_z)
        volume += (right - left) * area
    return volume


def _flag(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() in {"true", "1", "yes"}


def _preference(row: Mapping[str, Any]) -> PreferenceContext:
    values = []
    for name in ("flow", "cost", "variance"):
        value = row.get(f"w_{name}")
        if value in {None, ""}:
            value = row.get(f"preference_{name}")
        if value is None or value == "":
            raise ValueError("candidate is missing preference coordinates")
        values.append(float(value))
    return PreferenceContext.from_input(values)


def valid_candidate(row: Mapping[str, Any]) -> bool:
    try:
        objectives = [float(row[name]) for name in OBJECTIVE_FIELDS]
        return bool(
            _flag(row.get("terminated")) and not _flag(row.get("truncated"))
            and int(float(row.get("schedule_violation_count", 0))) == 0
            and float(row.get("maximum_worker_fatigue", 0)) <= float(row.get("safe_fatigue_limit", math.inf)) + 1e-9
            and all(math.isfinite(value) and value >= 0 for value in objectives)
        )
    except (TypeError, ValueError):
        return False


def _validate_row_protocol(row: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    scalarizer = config["objective_scalarizer"]
    required = {
        "result_schema_version": EVALUATION_SCHEMA_VERSION,
        "experiment_suite_version": config["experiment_suite_version"],
        "reward_version": config["reward"]["mode"],
        "objective_scalarizer_type": scalarizer["type"],
        "normalization_manifest_sha256": scalarizer["normalization_manifest_sha256"],
        "quality_metric_sha256": quality_metric_sha256(evaluation_quality_metric(config)),
    }
    for field, expected in required.items():
        if str(row.get(field)) != str(expected):
            raise ValueError(f"candidate {field} does not match the current experiment")
    for field, expected in {
        "objective_scalarizer_rho": scalarizer["rho"],
        **{f"objective_scale_{name}": scalarizer["scales"][name] for name in ("flow", "cost", "variance")},
    }.items():
        value = float(row[field])
        if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"candidate {field} does not match the current experiment")
    for field in ("dataset", "algorithm_seed", "instance_id", "schedule_violation_count", "maximum_worker_fatigue", "safe_fatigue_limit"):
        if row.get(field) in {None, ""}:
            raise ValueError(f"candidate is missing {field}")
    if row["arm"] == "ppo":
        if row.get("decode_mode") != "sampled":
            raise ValueError("formal PPO candidates require sampled decoding")
        if row.get("policy_execution_version") != config["training"]["policy_execution_version"]:
            raise ValueError("candidate policy execution does not match the current experiment")
        if row.get("policy_precision") != config["training"]["policy_precision"]:
            raise ValueError("candidate policy precision does not match the current experiment")
    normalize_objectives([float(row[field]) for field in OBJECTIVE_FIELDS], objective_scales(config))
    quality = terminal_quality_score(
        *(float(row[field]) for field in OBJECTIVE_FIELDS), dict(config),
        preference=_preference(row).preference, terminal_failure=_flag(row.get("truncated")),
    )
    if not math.isclose(float(row["preference_quality_score"]), quality, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("candidate preference quality does not match its objectives and preference")


def analyze_rows(rows: Sequence[Mapping[str, Any]], config=None, *, stage="final_test", arms=None):
    """Validate the preference/repeat matrix and analyze each scheduling instance."""
    config = load_config("configs/default.json") if config is None else config
    preferences = formal_preferences(config, stage)
    expected = {point.key for point in preferences}
    scales = objective_scales(config)
    groups = defaultdict(list)
    for original in rows:
        row = dict(original)
        arm = str(row.get("arm", "ppo"))
        row["arm"] = arm
        if arms is not None and arm not in arms:
            continue
        _validate_row_protocol(row, config)
        row["preference_key"] = _preference(row).key
        for name, value in zip(("flow", "cost", "variance"), _preference(row).as_tuple()):
            row[f"w_{name}"] = value
        groups[(str(row["dataset"]), int(row["algorithm_seed"]), str(row["instance_id"]))].append(row)
    if not groups:
        raise ValueError("no candidate rows were supplied")
    required_arms = tuple(arms) if arms is not None else tuple(sorted({row["arm"] for group in groups.values() for row in group}))
    annotated, instance_summary = [], []
    formal = config["training"]["formal_evaluation"]
    ppo_repeats = int(formal["validation_repeats" if stage == "validation" else "final_test_repeats"])
    for (dataset, seed, instance_id), group in sorted(groups.items()):
        annotation_start = len(annotated)
        result = {"dataset": dataset, "algorithm_seed": seed, "instance_id": instance_id}
        all_points, all_safe_rows = [], []
        for arm in required_arms:
            cells = [row for row in group if row["arm"] == arm]
            repeats = ppo_repeats if arm == "ppo" else 1
            observed = set()
            for row in cells:
                repeat_value = row.get("sampling_repeat")
                repeat = int(float(repeat_value)) if repeat_value not in {None, ""} else 0
                key = (row["preference_key"], repeat)
                if key in observed:
                    raise ValueError(f"duplicate candidate cell: {instance_id}/{arm}/{key}")
                observed.add(key)
            wanted = {(key, repeat) for key in expected for repeat in range(repeats)}
            if observed != wanted:
                raise ValueError(f"{instance_id}/{arm} requires {len(expected)} preferences x {repeats} repeats; incomplete or unexpected cells")
            safe = [row for row in cells if valid_candidate(row)]
            points = [normalize_objectives([float(row[name]) for name in OBJECTIVE_FIELDS], scales) for row in safe]
            front = set(nondominated_indices(points))
            all_points.extend(points)
            all_safe_rows.extend(safe)
            success_quality = []
            for key in sorted(expected):
                values = [float(row["preference_quality_score"]) for row in safe if row["preference_key"] == key]
                success_quality.append(fmean(values) if values else math.inf)
            result.update({
                f"{arm}_hypervolume": hypervolume_3d(points),
                f"{arm}_front_size": len(front),
                f"{arm}_completion_rate": len(safe) / len(cells),
                f"{arm}_valid_candidates": len(safe),
                f"{arm}_preference_balanced_quality": fmean(success_quality),
            })
            for index, row in enumerate(safe):
                annotated.append({**row, "valid_candidate": True, "is_arm_pareto": index in front})
            annotated.extend({**row, "valid_candidate": False, "is_arm_pareto": False} for row in cells if not valid_candidate(row))
        result["union_hypervolume"] = hypervolume_3d(all_points)
        union_front = set(nondominated_indices(all_points))
        union_cells = {
            (row["arm"], row["preference_key"], str(row.get("sampling_repeat")))
            for index, row in enumerate(all_safe_rows) if index in union_front
        }
        for row in annotated[annotation_start:]:
            row["is_union_pareto"] = (row["arm"], row["preference_key"], str(row.get("sampling_repeat"))) in union_cells
        for arm in required_arms:
            result[f"{arm}_union_contribution"] = sum(all_safe_rows[index]["arm"] == arm for index in union_front)
        instance_summary.append(result)
    seed_groups = defaultdict(list)
    for row in instance_summary:
        seed_groups[(row["dataset"], row["algorithm_seed"])].append(row)
    seed_summary = []
    for (dataset, seed), values in sorted(seed_groups.items()):
        summary = {"dataset": dataset, "algorithm_seed": seed, "instance_count": len(values)}
        for key in values[0]:
            if key not in {"dataset", "algorithm_seed", "instance_id"}:
                summary[f"mean_{key}"] = fmean(float(row[key]) for row in values)
        seed_summary.append(summary)
    summary = {
        "analysis_protocol": PROTOCOL_VERSION,
        "result_schema_version": EVALUATION_SCHEMA_VERSION,
        "formal_evaluation_stage": stage,
        "preference_count": len(expected),
        "ppo_repeats": ppo_repeats,
        "objective_scales": dict(zip(("flow", "cost", "variance"), scales)),
        "normalization_manifest_sha256": config["objective_scalarizer"]["normalization_manifest_sha256"],
        "arms": list(required_arms),
        "candidate_count": len(annotated),
        "instance_seed_count": len(instance_summary),
        "dataset_seed_count": len(seed_summary),
        "budget_note": "PPO uses the configured sampled repeats per preference; MO-ALNS selects one endpoint after its configured search budget per preference.",
    }
    return annotated, instance_summary, seed_summary, summary


def analyze_candidate_files(paths, output_dir, config=None, *, stage="final_test", arms=None, analyzer=None):
    rows = []
    for path in paths:
        with project_path(path).open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if analyzer is None:
        annotated, instances, seeds, summary = analyze_rows(rows, config, stage=stage, arms=arms)
    else:
        annotated, instances, seeds, summary = analyzer(rows, config, stage=stage)
    output = project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "candidates.csv", annotated)
    write_csv(output / "pareto_front.csv", [row for row in annotated if row["is_arm_pareto"]])
    write_csv(output / "instance_summary.csv", instances)
    write_csv(output / "seed_summary.csv", seeds)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--stage", choices=("validation", "final_test"), default="final_test")
    args = parser.parse_args()
    print(json.dumps(analyze_candidate_files(args.candidate_csv, args.output_dir, load_config(args.config), stage=args.stage), indent=2))


if __name__ == "__main__":
    main()
