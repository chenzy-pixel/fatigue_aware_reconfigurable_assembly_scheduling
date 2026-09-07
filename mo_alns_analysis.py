"""Three-arm empirical Pareto analysis for E1, E2 and MO-ALNS results.

This intentionally does *not* overwrite the E1/E2 equal-budget protocol: the
report labels MO-ALNS as a solver-budget arm because each of its 22 endpoints
is searched with an internal environment-evaluation budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from scipy.stats import spearmanr, wilcoxon

from configs import project_path
from pareto_analysis import hypervolume_3d, nondominated_indices, normalize_objectives
from result.io import write_csv, write_json


PROTOCOL_VERSION = "e1_e2_mo_alns_solver_budget_v1"
ARMS = ("e1", "e2", "mo_alns")
OBJECTIVE_FIELDS = (
    "flow_time_objective",
    "reconfiguration_cost",
    "worker_load_variance",
)
SCALES = (1200.0, 1000.0, 50.0)
CANONICAL = (0.5, 0.3, 0.2)


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected a finite numeric value, got {value!r}")
    return result


def _objectives(row: Mapping[str, Any]) -> tuple[float, float, float]:
    return tuple(_finite(row[field]) for field in OBJECTIVE_FIELDS)  # type: ignore[return-value]


def valid_candidate(row: Mapping[str, Any]) -> bool:
    try:
        fatigue_safe = _finite(row.get("maximum_worker_fatigue", 0.0)) <= _finite(row.get("safe_fatigue_limit", math.inf)) + 1e-9
        return (
            _flag(row.get("terminated"))
            and not _flag(row.get("truncated"))
            and int(row.get("schedule_violation_count", 0)) == 0
            and fatigue_safe
            and all(math.isfinite(value) for value in _objectives(row))
        )
    except (TypeError, ValueError):
        return False


def _canonical_row(rows: Sequence[Mapping[str, Any]], arm: str) -> Mapping[str, Any] | None:
    if arm == "e1":
        matches = [row for row in rows if str(row.get("candidate_source")) == "greedy"]
    else:
        matches = [
            row
            for row in rows
            if all(
                math.isclose(_finite(row[f"w_{name}"]), target, rel_tol=0.0, abs_tol=1e-9)
                for name, target in zip(("flow", "cost", "variance"), CANONICAL, strict=True)
            )
        ]
    return matches[0] if len(matches) == 1 else None


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "mean": fmean(finite) if finite else None,
        "std": pstdev(finite) if len(finite) > 1 else 0.0 if finite else None,
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
    }


def _spearman(rows: Sequence[Mapping[str, Any]], weight: str, objective: str) -> float:
    x = [_finite(row[f"w_{weight}"]) for row in rows]
    y = [_finite(row[objective]) for row in rows]
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else math.nan


def _paired(first: Sequence[float], second: Sequence[float], *, higher: bool) -> dict[str, Any]:
    pairs = [(float(left), float(right)) for left, right in zip(first, second, strict=True) if math.isfinite(left) and math.isfinite(right)]
    delta = [right - left for left, right in pairs]
    oriented = delta if higher else [-value for value in delta]
    nonzero = [value for value in delta if not math.isclose(value, 0.0, abs_tol=1e-12)]
    test = wilcoxon(nonzero, method="exact") if nonzero else None
    return {
        "first": _summary(left for left, _ in pairs),
        "second": _summary(right for _, right in pairs),
        "second_minus_first": _summary(delta),
        "win_tie_loss": {
            "wins": sum(value > 1e-12 for value in oriented),
            "ties": sum(abs(value) <= 1e-12 for value in oriented),
            "losses": sum(value < -1e-12 for value in oriented),
        },
        "exact_wilcoxon": {
            "nonzero_pairs": len(nonzero),
            "statistic": float(test.statistic) if test is not None else 0.0,
            "p_value": float(test.pvalue) if test is not None else 1.0,
        },
    }


def analyze_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for value in rows:
        row = dict(value)
        arm = str(row.get("arm", ""))
        if arm not in ARMS:
            continue
        groups[(str(row["dataset"]), int(row["algorithm_seed"]), str(row["instance_id"]))].append(row)
    if not groups:
        raise ValueError("no E1/E2/MO-ALNS candidate rows were supplied")

    annotated: list[dict[str, Any]] = []
    instance_summary: list[dict[str, Any]] = []
    for (dataset, seed, instance_id), group in sorted(groups.items()):
        by_arm = {arm: [row for row in group if row["arm"] == arm] for arm in ARMS}
        for arm in ARMS:
            if len(by_arm[arm]) != 22:
                raise ValueError(f"{dataset}/{seed}/{instance_id}/{arm} requires 22 endpoints, found {len(by_arm[arm])}")
        valid = {arm: [row for row in by_arm[arm] if valid_candidate(row)] for arm in ARMS}
        if any(not values for values in valid.values()):
            missing = [arm for arm, values in valid.items() if not values]
            raise ValueError(f"{dataset}/{seed}/{instance_id} has no safe candidate for {missing}")
        arm_front_ids: dict[str, set[str]] = {}
        arm_hv: dict[str, float] = {}
        arm_unique: dict[str, int] = {}
        arm_vectors: dict[str, int] = {}
        for arm, candidates in valid.items():
            normalized = [normalize_objectives(_objectives(row), SCALES) for row in candidates]
            front = nondominated_indices(normalized)
            arm_front_ids[arm] = {str(candidates[index]["candidate_id"]) for index in front}
            arm_hv[arm] = hypervolume_3d(normalized[index] for index in front)
            arm_unique[arm] = len({str(row.get("action_trace_sha256")) for row in candidates})
            arm_vectors[arm] = len({tuple(round(value, 9) for value in _objectives(row)) for row in candidates})
        all_valid = [row for arm in ARMS for row in valid[arm]]
        union_points = [normalize_objectives(_objectives(row), SCALES) for row in all_valid]
        union_indices = set(nondominated_indices(union_points))
        union_hv = hypervolume_3d(union_points[index] for index in sorted(union_indices))
        contribution = {arm: 0 for arm in ARMS}
        for index, row in enumerate(all_valid):
            is_union = index in union_indices
            if is_union:
                contribution[str(row["arm"])] += 1
            annotated.append({
                **row,
                "valid_candidate": True,
                "is_arm_pareto": str(row["candidate_id"]) in arm_front_ids[str(row["arm"])],
                "is_union_pareto": is_union,
            })
        annotated.extend(
            {**row, "valid_candidate": False, "is_arm_pareto": False, "is_union_pareto": False}
            for arm in ARMS for row in by_arm[arm] if not valid_candidate(row)
        )
        canonical = {arm: _canonical_row(valid[arm], arm) for arm in ARMS}
        row: dict[str, Any] = {
            "dataset": dataset,
            "algorithm_seed": seed,
            "instance_id": instance_id,
            "union_hypervolume": union_hv,
        }
        for arm in ARMS:
            row.update({
                f"{arm}_hypervolume": arm_hv[arm],
                f"{arm}_front_size": len(arm_front_ids[arm]),
                f"{arm}_valid_candidates": len(valid[arm]),
                f"{arm}_union_contribution": contribution[arm],
                f"{arm}_unique_action_traces": arm_unique[arm],
                f"{arm}_unique_objective_vectors": arm_vectors[arm],
                f"{arm}_canonical_quality_score": _finite(canonical[arm]["quality_score"]) if canonical[arm] is not None else math.nan,
            })
        for weight, objective in zip(("flow", "cost", "variance"), OBJECTIVE_FIELDS, strict=True):
            row[f"mo_alns_{weight}_weight_objective_spearman"] = _spearman(valid["mo_alns"], weight, objective)
        row["mo_alns_environment_evaluations"] = _summary(
            _finite(value.get("environment_evaluation_count", math.nan)) for value in by_arm["mo_alns"]
        )["mean"]
        row["mo_alns_solve_time_seconds"] = _summary(
            _finite(value.get("solve_time_seconds", math.nan)) for value in by_arm["mo_alns"]
        )["mean"]
        instance_summary.append(row)

    seed_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in instance_summary:
        seed_groups[(str(row["dataset"]), int(row["algorithm_seed"]))].append(row)
    seed_summary: list[dict[str, Any]] = []
    for (dataset, seed), values in sorted(seed_groups.items()):
        numeric = [
            key
            for key in values[0]
            if key not in {"dataset", "algorithm_seed", "instance_id"}
        ]
        seed_summary.append({
            "dataset": dataset,
            "algorithm_seed": seed,
            "instance_count": len(values),
            **{
                f"mean_{field}": _summary(_finite(row[field]) for row in values if row.get(field) is not None)["mean"]
                for field in numeric
            },
        })
    statistics: dict[str, Any] = {}
    for dataset in sorted({str(row["dataset"]) for row in seed_summary}):
        values = [row for row in seed_summary if row["dataset"] == dataset]
        comparisons = {}
        for first, second in (("e1", "mo_alns"), ("e2", "mo_alns")):
            comparisons[f"{second}_vs_{first}"] = {
                "hypervolume": _paired(
                    [_finite(row[f"mean_{first}_hypervolume"]) for row in values],
                    [_finite(row[f"mean_{second}_hypervolume"]) for row in values],
                    higher=True,
                ),
                "canonical_quality": _paired(
                    [_finite(row[f"mean_{first}_canonical_quality_score"]) for row in values],
                    [_finite(row[f"mean_{second}_canonical_quality_score"]) for row in values],
                    higher=False,
                ),
                "union_contribution": _paired(
                    [_finite(row[f"mean_{first}_union_contribution"]) for row in values],
                    [_finite(row[f"mean_{second}_union_contribution"]) for row in values],
                    higher=True,
                ),
            }
        statistics[dataset] = {"algorithm_seed_count": len(values), "pairwise": comparisons}
    summary = {
        "analysis_protocol": PROTOCOL_VERSION,
        "true_pareto_claim": False,
        "candidate_row_count": len(annotated),
        "instance_count": len(instance_summary),
        "seed_count": len(seed_summary),
        "arms": list(ARMS),
        "budget_note": "E1/E2 submit 22 rollout candidates; MO-ALNS selects 22 endpoints after a configured internal environment-evaluation budget per preference.",
        "statistics": statistics,
    }
    return annotated, instance_summary, seed_summary, summary


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    with project_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# E1/E2/MO-ALNS empirical Pareto analysis",
        "",
        "This report compares empirical candidate sets; it does not certify a true Pareto frontier.",
        "",
        f"- Candidate rows: {summary['candidate_row_count']}",
        f"- Instance/seed summaries: {summary['instance_count']}/{summary['seed_count']}",
        "- MO-ALNS is labelled as a solver-budget arm, not an equal-rollout-budget arm.",
        "",
    ]
    for dataset, values in summary["statistics"].items():
        lines.extend([f"## {dataset}", ""])
        for name, comparison in values["pairwise"].items():
            delta = comparison["hypervolume"]["second_minus_first"]
            lines.append(f"- {name} mean hypervolume difference: {delta['mean']}.")
        lines.append("")
    return "\n".join(lines)


def analyze_candidate_files(paths: Sequence[str | Path], output_dir: str | Path) -> dict[str, Any]:
    rows = [row for path in paths for row in _read_csv(path)]
    annotated, instance_summary, seed_summary, summary = analyze_rows(rows)
    output = project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "candidates.csv", annotated)
    write_csv(output / "pareto_front.csv", [row for row in annotated if _flag(row["is_union_pareto"])])
    write_csv(output / "instance_summary.csv", instance_summary)
    write_csv(output / "seed_summary.csv", seed_summary)
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E1/E2/MO-ALNS candidate fronts")
    parser.add_argument("--e1-e2-candidate-csv", action="append", required=True)
    parser.add_argument("--mo-alns-candidate-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = analyze_candidate_files(
        [*args.e1_e2_candidate_csv, *args.mo_alns_candidate_csv], args.output_dir
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results: {project_path(args.output_dir)}")


if __name__ == "__main__":
    main()
