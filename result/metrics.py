from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any


EVALUATION_SCHEMA_VERSION = "2.1.0"


def compare_lexicographic(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    tolerance: float = 1e-6,
) -> int:
    """Return -1 if first is better, 1 if second is better, and 0 for a tie."""
    if first["truncated"] != second["truncated"]:
        return 1 if first["truncated"] else -1
    if first["truncated"]:
        unfinished_difference = (
            first["unfinished_orders"] - second["unfinished_orders"]
        )
        if unfinished_difference:
            return -1 if unfinished_difference < 0 else 1
    fields = (
        "flow_time_objective",
        "reconfiguration_cost",
        "worker_load_variance",
    )
    for field in fields:
        difference = float(first[field]) - float(second[field])
        if abs(difference) > tolerance:
            return -1 if difference < 0 else 1
    return 0


def relative_gap_percent(
    value: float | None,
    reference: float | None,
    *,
    tolerance: float = 1e-12,
) -> float | None:
    """Return a minimization gap in percent; negative values are better."""
    if value is None or reference is None:
        return None
    actual = float(value)
    baseline = float(reference)
    if (
        not math.isfinite(actual)
        or not math.isfinite(baseline)
        or abs(baseline) <= tolerance
    ):
        return None
    return 100.0 * (actual - baseline) / baseline


def summarize_values(
    values: Iterable[float | int | None],
) -> dict[str, float | int | None]:
    """Summarize finite observations with sample standard deviation."""
    observations: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("cannot summarize a non-finite value")
        observations.append(number)
    if not observations:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": len(observations),
        "mean": float(statistics.fmean(observations)),
        "std": (
            float(statistics.stdev(observations))
            if len(observations) > 1
            else 0.0
        ),
    }


def aggregate_evaluation_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    policy: str,
    manifest: str,
) -> dict[str, Any]:
    completed = [
        row
        for row in rows
        if bool(row["terminated"]) and not bool(row["truncated"])
    ]
    completed_metrics = {
        "makespan": summarize_values(
            row["makespan"] for row in completed
        ),
        "total_flow_time": summarize_values(
            row["total_flow_time"] for row in completed
        ),
    }
    all_instance_metrics = {
        "flow_time_objective": summarize_values(
            row["flow_time_objective"] for row in rows
        ),
        "reconfiguration_cost": summarize_values(
            row["reconfiguration_cost"] for row in rows
        ),
        "worker_load_variance": summarize_values(
            row["worker_load_variance"] for row in rows
        ),
        "inference_time_seconds": summarize_values(
            row["inference_time_seconds"] for row in rows
        ),
        "solve_time_seconds": summarize_values(
            row["solve_time_seconds"] for row in rows
        ),
        "inference_time_per_decision_ms": summarize_values(
            row["inference_time_per_decision_ms"] for row in rows
        ),
        "maximum_worker_fatigue": summarize_values(
            row.get("maximum_worker_fatigue") for row in rows
        ),
        "mean_peak_worker_fatigue": summarize_values(
            row.get("mean_peak_worker_fatigue") for row in rows
        ),
        "safe_fatigue_limit": summarize_values(
            row.get("safe_fatigue_limit") for row in rows
        ),
        "fatigue_masked_action_ratio": summarize_values(
            row.get("fatigue_masked_action_ratio") for row in rows
        ),
        "worker_competition_event_count": summarize_values(
            row.get("worker_competition_event_count") for row in rows
        ),
        "worker_matching_deficit_event_count": summarize_values(
            row.get("worker_matching_deficit_event_count") for row in rows
        ),
        "resource_admission_masked_action_count": summarize_values(
            row.get("resource_admission_masked_action_count") for row in rows
        ),
        "resource_admission_masked_action_ratio": summarize_values(
            row.get("resource_admission_masked_action_ratio") for row in rows
        ),
        "minimum_worker_alternatives": summarize_values(
            row.get("minimum_worker_alternatives") for row in rows
        ),
        "matching_preserving_worker_action_count": summarize_values(
            row.get("matching_preserving_worker_action_count") for row in rows
        ),
        "candidate_recovery_advance_count": summarize_values(
            row.get("candidate_recovery_advance_count") for row in rows
        ),
        "machine_waiting_for_worker_time": summarize_values(
            row.get("machine_waiting_for_worker_time") for row in rows
        ),
        "completed_reconfigurations": summarize_values(
            row.get("completed_reconfigurations") for row in rows
        ),
        "worker_switch_ratio": summarize_values(
            row.get("worker_switch_ratio") for row in rows
        ),
        "unfinished_orders": summarize_values(
            row.get("unfinished_orders") for row in rows
        ),
        "feasibility_proxy_return": summarize_values(
            row.get("feasibility_proxy_return") for row in rows
        ),
    }
    gap_metrics = {
        "relative_heuristic_gap_percent": summarize_values(
            row["relative_heuristic_gap_percent"] for row in rows
        ),
        "makespan_heuristic_gap_percent": summarize_values(
            row["makespan_heuristic_gap_percent"] for row in rows
        ),
        "reconfiguration_cost_heuristic_gap_percent": summarize_values(
            row["reconfiguration_cost_heuristic_gap_percent"]
            for row in rows
        ),
        "worker_load_variance_heuristic_gap_percent": summarize_values(
            row["worker_load_variance_heuristic_gap_percent"]
            for row in rows
        ),
    }
    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "dataset": dataset,
        "manifest": manifest,
        "policy": policy,
        "instance_count": len(rows),
        "completed_count": len(completed),
        "completion_rate": len(completed) / len(rows) if rows else 0.0,
        "truncated_count": sum(
            bool(row["truncated"]) for row in rows
        ),
        "schedule_violation_count": sum(
            int(row["schedule_violation_count"]) for row in rows
        ),
        "decision_count": sum(int(row["decisions"]) for row in rows),
        "total_inference_time_seconds": sum(
            float(row["inference_time_seconds"]) for row in rows
        ),
        "total_solve_time_seconds": sum(
            float(row["solve_time_seconds"]) for row in rows
        ),
        "completed_metrics": completed_metrics,
        "all_instance_metrics": all_instance_metrics,
        "gap_metrics": gap_metrics,
    }


def evaluation_selection_key(
    aggregate: dict[str, Any],
) -> tuple[float, float, float, float]:
    metrics = aggregate["all_instance_metrics"]

    def mean(name: str) -> float:
        value = metrics[name]["mean"]
        return math.inf if value is None else float(value)

    return (
        -float(aggregate["completion_rate"]),
        mean("flow_time_objective"),
        mean("reconfiguration_cost"),
        mean("worker_load_variance"),
    )
