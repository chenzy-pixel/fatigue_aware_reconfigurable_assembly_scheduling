from __future__ import annotations

import math
from collections import Counter
import statistics
from collections.abc import Iterable, Mapping
from copy import deepcopy
import hashlib
import json
from typing import Any


EVALUATION_SCHEMA_VERSION = "8.0.0"
QUALITY_METRIC_VERSION = "canonical_bounded_quality_v2"
CURRENT_RUNTIME_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "current_worker_matching_deficit",
    "maximum_worker_matching_deficit",
    "wait_total_ticks",
    "wait_total_time",
    "production_wait_ticks",
    "production_wait_time",
    "worker_wait_ticks",
    "worker_wait_time",
    "wait_min_estimated_deadline_slack_ticks",
)


def result_schema_version(config: Mapping[str, Any]) -> str:
    """Return the sole active result schema and reject per-run overrides."""
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise TypeError("config.evaluation must be an object")
    configured = evaluation.get("result_schema_version")
    if configured is not None and str(configured) != EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            f"only result schema {EVALUATION_SCHEMA_VERSION} is supported"
        )
    return EVALUATION_SCHEMA_VERSION
CANONICAL_QUALITY_METRIC: dict[str, Any] = {
    "version": QUALITY_METRIC_VERSION,
    "flow_scale": 1089.15,
    "cost_scale": 353.27,
    "variance_scale": 2.2629,
    "quality_weights": {
        "flow": 0.5,
        "cost": 0.3,
        "variance": 0.2,
    },
}
LEGACY_QUALITY_METRIC: dict[str, Any] = {
    **CANONICAL_QUALITY_METRIC,
    "version": "canonical_bounded_quality_v1",
    "flow_scale": 1200.0,
    "cost_scale": 1000.0,
    "variance_scale": 50.0,
}


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def evaluation_quality_metric(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable paper-quality metric, with canonical reference defaults."""

    evaluation = config.get("evaluation")
    if evaluation is None:
        return deepcopy(CANONICAL_QUALITY_METRIC)
    if not isinstance(evaluation, Mapping):
        raise TypeError("config.evaluation must be an object")
    raw = evaluation.get("quality_metric")
    if raw is None:
        return deepcopy(CANONICAL_QUALITY_METRIC)
    if not isinstance(raw, Mapping):
        raise TypeError("config.evaluation.quality_metric must be an object")
    metric = deepcopy(dict(raw))
    expected_keys = set(CANONICAL_QUALITY_METRIC)
    if set(metric) != expected_keys:
        raise ValueError(
            "evaluation.quality_metric must contain exactly "
            f"{sorted(expected_keys)}"
        )
    weights = metric.get("quality_weights")
    if not isinstance(weights, Mapping) or set(weights) != {
        "flow",
        "cost",
        "variance",
    }:
        raise ValueError(
            "evaluation.quality_metric.quality_weights must contain exactly "
            "flow/cost/variance"
        )
    normalized = {
        "version": str(metric["version"]),
        "flow_scale": float(metric["flow_scale"]),
        "cost_scale": float(metric["cost_scale"]),
        "variance_scale": float(metric["variance_scale"]),
        "quality_weights": {
            name: float(weights[name])
            for name in ("flow", "cost", "variance")
        },
    }
    scales = tuple(
        normalized[name]
        for name in ("flow_scale", "cost_scale", "variance_scale")
    )
    weight_values = tuple(normalized["quality_weights"].values())
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("evaluation quality scales must be finite and positive")
    if any(not math.isfinite(value) or value < 0.0 for value in weight_values):
        raise ValueError("evaluation quality weights must be finite and nonnegative")
    if sum(weight_values) <= 0.0:
        raise ValueError("evaluation quality weights must have a positive sum")
    reference = (LEGACY_QUALITY_METRIC if normalized["version"] == LEGACY_QUALITY_METRIC["version"]
                 else CANONICAL_QUALITY_METRIC)
    if normalized != reference:
        raise ValueError(
            f"{reference['version']} is immutable and must use "
            f"flow/cost/variance scales {reference['flow_scale']}/{reference['cost_scale']}/{reference['variance_scale']} "
            "and weights 0.5/0.3/0.2"
        )
    return normalized


def quality_metric_sha256(metric: Mapping[str, Any]) -> str:
    normalized = evaluation_quality_metric(
        {"evaluation": {"quality_metric": dict(metric)}}
    )
    return hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()


def compare_lexicographic(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    tolerance: float = 1e-6,
) -> int:
    """Return -1 if first is better, 1 if second is better, and 0 for a tie."""
    if first.get("sampling_truncated", first.get("truncated", False)) or second.get("sampling_truncated", second.get("truncated", False)):
        raise ValueError("cannot rank externally truncated schedules")
    if first["task_failed"] != second["task_failed"]:
        return 1 if first["task_failed"] else -1
    if first["task_failed"]:
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
        return {"count": 0, "mean": None, "median": None, "std": None}
    return {
        "count": len(observations),
        "mean": float(statistics.fmean(observations)),
        "median": float(statistics.median(observations)),
        "std": (
            float(statistics.stdev(observations))
            if len(observations) > 1
            else 0.0
        ),
    }


def summarize_upper_tail(
    values: Iterable[float | int | None],
    *,
    tail_fraction: float,
) -> dict[str, float | int | None]:
    observations = sorted(
        float(value) for value in values if value is not None
    )
    if not observations:
        return {"count": 0, "quantile": None, "cvar": None, "max": None}
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if any(not math.isfinite(value) for value in observations):
        raise ValueError("cannot summarize a non-finite tail value")
    rank = max(0, math.ceil((1.0 - tail_fraction) * len(observations)) - 1)
    tail_count = max(1, math.ceil(tail_fraction * len(observations)))
    tail = observations[-tail_count:]
    return {
        "count": len(observations),
        "quantile": observations[rank],
        "cvar": float(statistics.fmean(tail)),
        "max": observations[-1],
    }


def aggregate_evaluation_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    policy: str,
    manifest: str,
    quality_metric: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_metric = evaluation_quality_metric(
        {}
        if quality_metric is None
        else {"evaluation": {"quality_metric": dict(quality_metric)}}
    )
    metric_hash = quality_metric_sha256(normalized_metric)
    if any(row.get("result_schema_version") not in {None, EVALUATION_SCHEMA_VERSION} for row in rows):
        raise ValueError(f"only result schema {EVALUATION_SCHEMA_VERSION} can be aggregated by this evaluator")
    row_hashes = {
        str(row["quality_metric_sha256"])
        for row in rows
        if row.get("quality_metric_sha256") is not None
    }
    if len(row_hashes) > 1:
        raise ValueError("cannot aggregate rows with different quality metrics")
    if row_hashes and row_hashes != {metric_hash}:
        raise ValueError(
            "row quality metric hash does not match the aggregate metric"
        )
    for field in ("result_schema_version", "generator_version", "dataset_manifest_sha256", "subset_sha256",
                  "generator_config_sha256", "environment_config_sha256", "distribution_contract_sha256",
                  "normalization_manifest_sha256", "experiment_suite_version", "objective_scalarizer_type",
                  "objective_scalarizer_rho", "objective_scale_flow", "objective_scale_cost", "objective_scale_variance"):
        values = {str(row.get(field)) for row in rows}
        if len(values) > 1:
            raise ValueError(f"cannot aggregate rows with different {field}")
    worker_timing_totals = {
        name: sum(float(row.get(name) or 0.0) for row in rows)
        for name in ("observation_time_seconds", "environment_step_time_seconds",
                     "terminal_metrics_time_seconds", "worker_service_time_seconds", "reset_time_seconds")
        if any(row.get(name) is not None for row in rows)
    }
    sampling_truncated_count = sum(bool(row.get("sampling_truncated", row.get("truncated", False))) for row in rows)
    if sampling_truncated_count:
        return {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluation_complete": False,
            "selection_eligible": False,
            "quality_metric_version": normalized_metric["version"],
            "quality_metric": normalized_metric,
            "quality_metric_sha256": metric_hash,
            "dataset": dataset, "manifest": manifest, "policy": policy,
            "instance_count": len(rows),
            "completed_count": sum(bool(row["task_succeeded"]) for row in rows),
            "task_failed_count": sum(bool(row["task_failed"]) for row in rows),
            "terminated_count": sum(bool(row["terminated"]) for row in rows),
            "truncated_count": sampling_truncated_count,
            "sampling_truncated_count": sampling_truncated_count,
            "completion_coverage": (len(rows) - sampling_truncated_count) / len(rows),
            "completion_rate": None,
            "schedule_violation_count": sum(int(row["schedule_violation_count"]) for row in rows),
            "decision_count": sum(int(row["decisions"]) for row in rows),
            "total_inference_time_seconds": sum(float(row["inference_time_seconds"]) for row in rows),
            "total_solve_time_seconds": sum(float(row["solve_time_seconds"]) for row in rows),
            "worker_timing_totals": worker_timing_totals,
            "completed_metrics": {}, "all_instance_metrics": {},
            "preference_quality_by_key": {}, "preference_balanced_quality_score": None,
            "gap_metrics": {}, "tail_metrics": {},
            "by_pressure_type": {}, "by_feasibility_status": {},
            "failure_reasons": dict(Counter(failure_reason(row) for row in rows if row["task_failed"])),
        }
    completed = [
        row
        for row in rows
        if successful_row(row)
    ]
    completed_metrics = {
        "quality_score": summarize_values(
            row.get("quality_score") for row in completed
        ),
        "preference_quality_score": summarize_values(
            row.get("preference_quality_score") for row in completed
        ),
        "makespan": summarize_values(
            row["makespan"] for row in completed
        ),
        "total_flow_time": summarize_values(
            row["total_flow_time"] for row in completed
        ),
        "flow_time_objective": summarize_values(
            row["flow_time_objective"] for row in completed
        ),
        "reconfiguration_cost": summarize_values(
            row["reconfiguration_cost"] for row in completed
        ),
        "worker_load_variance": summarize_values(
            row["worker_load_variance"] for row in completed
        ),
    }
    all_instance_metrics = {
        "quality_score": summarize_values(
            row.get("quality_score") for row in rows
        ),
        "preference_quality_score": summarize_values(
            row.get("preference_quality_score") for row in rows
        ),
        "heuristic_quality_score": summarize_values(
            row.get("heuristic_quality_score") for row in rows
        ),
        "reward_quality_score": summarize_values(
            row.get("reward_quality_score") for row in rows
        ),
        "heuristic_reward_quality_score": summarize_values(
            row.get("heuristic_reward_quality_score") for row in rows
        ),
        "flow_time_objective": summarize_values(
            row["flow_time_objective"] for row in rows
        ),
        "flow_excess_objective": summarize_values(row.get("flow_excess_objective") for row in rows),
        "reward_objective_flow": summarize_values(row.get("reward_objective_flow") for row in rows),
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
        "minimum_worker_alternatives": summarize_values(
            row.get("minimum_worker_alternatives") for row in rows
        ),
        "wait_total_time": summarize_values(
            row.get("wait_total_time") for row in rows
        ),
        "production_wait_time": summarize_values(
            row.get("production_wait_time") for row in rows
        ),
        "worker_wait_time": summarize_values(
            row.get("worker_wait_time") for row in rows
        ),
        **{
            name: summarize_values(row.get(name) for row in rows)
            for name in (
                "ranker_top_selection_rate",
                "context_override_rate",
                "production_pair_plus_wait_state_count",
                "production_decision_state_count",
                "production_pair_plus_wait_ratio",
                "worker_pair_plus_wait_state_count",
                "worker_decision_state_count",
                "worker_pair_plus_wait_ratio",
                "reconfiguration_reuse_count",
                *CURRENT_RUNTIME_DIAGNOSTIC_FIELDS,
            )
        },
        **{
            name: summarize_values(row.get(name) for row in rows)
            for name in (
                "direct_process_action_count",
                "commit_reconfig_action_count",
                "worker_assign_action_count",
                "wait_action_count",
                "production_wait_action_count",
                "worker_wait_action_count",
            )
        },
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
        "initial_progress": summarize_values(
            row.get("initial_progress") for row in rows
        ),
        "initial_preference_quality_score": summarize_values(
            row.get("initial_preference_quality_score") for row in rows
        ),
        "operation_progress": summarize_values(
            row.get("operation_progress") for row in rows
        ),
        "single_stage_proxy_return": summarize_values(
            row.get("single_stage_proxy_return") for row in rows
        ),
        **{
            name: summarize_values(row.get(name) for row in rows)
            for name in (
                "forced_action_state_count",
                "forced_production_count",
                "forced_worker_count",
                "forced_pair_count",
                "forced_wait_count",
                "forced_production_pair_count",
                "forced_production_wait_count",
                "forced_worker_pair_count",
                "forced_worker_wait_count",
                "forced_pair_wait_physically_unavailable_count",
                "forced_wait_pair_physically_unavailable_count",
                "forced_wait_dis_count",
                "forced_wait_ins_count",
                "forced_mixed_wait_stage_count",
                "forced_phase_handoff_count",
                "forced_recovery_wait_count",
                "forced_future_event_wait_count",
                "forced_action_chain_count",
                "longest_forced_action_chain",
                "mean_forced_action_chain_length",
            )
        },
    }
    for field in ("flow_excess_objective", "reward_objective_flow"):
        completed_metrics[field] = summarize_values(row.get(field) for row in completed)
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
        "quality_score_heuristic_gap_percent": summarize_values(
            relative_gap_percent(
                row.get("quality_score"), row.get("heuristic_quality_score")
            )
            for row in rows if row.get("heuristic_comparison_valid", True)
        ),
    }
    tail_metrics = {
        "worker_load_variance": summarize_upper_tail(
            (row.get("worker_load_variance") for row in rows),
            tail_fraction=0.10,
        ),
        "maximum_worker_fatigue": summarize_upper_tail(
            (row.get("maximum_worker_fatigue") for row in rows),
            tail_fraction=0.10,
        ),
        "forced_action_chain": summarize_upper_tail(
            (row.get("longest_forced_action_chain") for row in rows),
            tail_fraction=0.05,
        ),
    }
    preference_keys = sorted(
        {
            str(row["preference_key"])
            for row in rows
            if row.get("preference_key") is not None
        }
    )
    preference_quality_means: dict[str, float] = {}
    for key in preference_keys:
        values = [
            float(row["preference_quality_score"])
            for row in completed
            if str(row.get("preference_key")) == key
            and row.get("preference_quality_score") is not None
            and math.isfinite(float(row["preference_quality_score"]))
        ]
        preference_quality_means[key] = (
            sum(values) / len(values) if values else math.inf
        )
    preference_balanced_quality = (
        sum(preference_quality_means.values()) / len(preference_quality_means)
        if preference_quality_means
        and all(math.isfinite(value) for value in preference_quality_means.values())
        else math.inf
    )

    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_complete": True,
        "selection_eligible": True,
        "completion_coverage": 1.0 if rows else 0.0,
        "sampling_truncated_count": 0,
        "task_failed_count": sum(bool(row["task_failed"]) for row in rows),
        "terminated_count": sum(bool(row["terminated"]) for row in rows),
        "quality_metric_version": normalized_metric["version"],
        "quality_metric": normalized_metric,
        "quality_metric_sha256": metric_hash,
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
        "worker_timing_totals": worker_timing_totals,
        "completed_metrics": completed_metrics,
        "all_instance_metrics": all_instance_metrics,
        "preference_quality_by_key": preference_quality_means,
        "preference_balanced_quality_score": preference_balanced_quality,
        "gap_metrics": gap_metrics,
        "by_pressure_type": grouped_outcomes(rows, "pressure_type"),
        "by_feasibility_status": grouped_outcomes(rows, "feasibility_status"),
        "failure_reasons": dict(Counter(failure_reason(row) for row in rows if not successful_row(row))),
        "tail_metrics": tail_metrics,
    }


def evaluation_selection_key(
    aggregate: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Return completion first, then preference-balanced quality."""

    if not aggregate.get("evaluation_complete", True) or aggregate.get("sampling_truncated_count", 0):
        raise ValueError("incomplete evaluation cannot select a checkpoint")
    quality = float(
        aggregate.get("preference_balanced_quality_score", math.inf)
    )

    return (
        -float(aggregate["completion_rate"]),
        quality,
        0.0,
        0.0,
    )


def successful_row(row: dict[str, Any]) -> bool:
    return (bool(row["task_succeeded"]) and not bool(row.get("sampling_truncated", row["truncated"]))
            and not row.get("schedule_violation_count", 0)
            and float(row.get("maximum_worker_fatigue", 0)) <= float(row.get("safe_fatigue_limit", math.inf)) + 1e-9)


def failure_reason(row: dict[str, Any]) -> str:
    if row.get("schedule_violation_count", 0):
        return "schedule_violation"
    if float(row.get("maximum_worker_fatigue", 0)) > float(row.get("safe_fatigue_limit", math.inf)) + 1e-9:
        return "unsafe_fatigue"
    return str(row.get("termination_reason", row.get("terminal_reason", "unknown")))


def grouped_outcomes(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups = {}
    for value in sorted({str(row.get(field, "unknown")) for row in rows}):
        selected = [row for row in rows if str(row.get(field, "unknown")) == value]
        successful = [row for row in selected if successful_row(row)]
        failures = [row for row in selected if not successful_row(row)]
        groups[value] = {
            "count": len(selected), "completed_count": len(successful),
            "completion_rate": len(successful) / len(selected),
            "completed_metrics": {name: summarize_values(row.get(name) for row in successful)
                                  for name in ("flow_time_objective", "reconfiguration_cost", "worker_load_variance", "preference_quality_score")},
            "failure_reasons": dict(Counter(failure_reason(row) for row in failures)),
            "failure_progress": summarize_values(row.get("operation_progress") for row in failures),
        }
    return groups
