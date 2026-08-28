from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from copy import deepcopy
import hashlib
import json
from typing import Any


EVALUATION_SCHEMA_VERSION = "4.1.0"
PREFERENCE_EVALUATION_SCHEMA_VERSION = "4.2.0"
HIERARCHICAL_PREFERENCE_EVALUATION_SCHEMA_VERSION = "4.3.0"
SAFE_PRODUCTION_PREFERENCE_EVALUATION_SCHEMA_VERSION = "4.4.0"
NEUTRAL_GATE_SAFE_VARIANCE_EVALUATION_SCHEMA_VERSION = "4.5.0"
SAFE_MONOTONE_FLOW_GATE_EVALUATION_SCHEMA_VERSION = "4.6.0"
COUNTERFACTUAL_PREFERENCE_CONSISTENCY_EVALUATION_SCHEMA_VERSION = "4.7.0"
E1_WARMSTART_SAFE_GATE_EVALUATION_SCHEMA_VERSION = "4.8.0"
E1_WARMSTART_SAFE_GATE_V2_1_EVALUATION_SCHEMA_VERSION = "4.9.0"
TIERED_TRAINING_GATES_EVALUATION_SCHEMA_VERSION = "5.0.0"
MO_ALNS_EVALUATION_SCHEMA_VERSION = "5.1.0"

# Scalar diagnostics intentionally shared by evaluation, training and Pareto
# persistence.  Keeping the registry here prevents a new experiment field from
# being silently defaulted to zero by one of the CSV writers.
MATCHING_RECOVERY_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "current_worker_matching_deficit",
    "maximum_worker_matching_deficit",
    "deficit_reducing_worker_action_candidate_count",
    "deficit_reducing_worker_action_count",
    "matching_deficit_recovery_advance_count",
    "current_matching_admission_masked_action_count",
    "future_installation_admission_candidate_count",
    "future_installation_admission_masked_action_count",
    "future_installation_admission_masked_action_ratio",
    "future_installation_matching_deficit_after_commit",
    "maximum_projected_installation_deficit",
    "temporal_oracle_call_count",
    "temporal_oracle_cache_hit_count",
    "temporal_oracle_searched_nodes",
    "temporal_oracle_feasible_count",
    "temporal_oracle_infeasible_count",
    "temporal_oracle_unknown_count",
    "temporal_worker_action_rescued_count",
    "temporal_future_installation_rescued_count",
    "temporal_delayed_disassembly_rescued_count",
)
PREFERENCE_POLICY_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "ranker_top_decision_count",
    "ranker_top_selected_count",
    "ranker_top_selection_rate",
    "context_override_count",
    "context_override_rate",
    "preference_override_count",
    "preference_override_rate",
    "mean_preference_logit_std",
    "production_ranker_top_decision_count",
    "production_preference_override_count",
    "production_preference_override_rate",
    "production_mean_preference_logit_std",
    "worker_ranker_top_decision_count",
    "worker_preference_override_count",
    "worker_preference_override_rate",
    "worker_mean_preference_logit_std",
    "production_conditional_preference_override_count",
    "production_conditional_preference_override_rate",
    "worker_variance_preference_override_count",
    "worker_variance_preference_override_rate",
    "worker_direct_preference_flow_logit_max_abs",
    "worker_direct_preference_cost_logit_max_abs",
    "worker_direct_preference_variance_logit_max_abs",
    "unsafe_worker_preference_selection_count",
    "production_gate_state_count",
    "production_gate_commit_selected_count",
    "production_gate_defer_selected_count",
    "mean_production_gate_commit_probability",
    "mean_production_gate_defer_probability",
    "mean_production_gate_logit_margin",
    "mean_production_gate_base_commit_probability",
    "mean_production_gate_base_defer_probability",
    "mean_production_gate_commit_logit_boost",
    "production_gate_residual_active_count",
    "production_gate_base_defer_to_final_commit_flip_count",
    "counterfactual_eligible_state_count",
    "counterfactual_high_flow_commit_flip_count",
    "counterfactual_high_flow_commit_flip_rate",
    "mean_counterfactual_state_residual_scale",
    "max_counterfactual_state_residual_scale",
    "counterfactual_low_flow_identity_violation_count",
    "counterfactual_monotonicity_violation_count",
    "centered_gate_dual_legal_state_count",
    "centered_gate_flow_cost_flip_count",
    "centered_gate_flow_variance_flip_count",
    "centered_gate_flow_cost_flip_rate",
    "centered_gate_flow_variance_flip_rate",
    "centered_gate_extreme_flip_rate",
    "centered_gate_monotonicity_violation_count",
)
QUALITY_METRIC_VERSION = "canonical_bounded_quality_v1"
CANONICAL_QUALITY_METRIC: dict[str, Any] = {
    "version": QUALITY_METRIC_VERSION,
    "flow_scale": 1200.0,
    "cost_scale": 1000.0,
    "variance_scale": 50.0,
    "quality_weights": {
        "flow": 0.5,
        "cost": 0.3,
        "variance": 0.2,
    },
}


def result_schema_version(config: Mapping[str, Any]) -> str:
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise TypeError("config.evaluation must be an object")
    version = str(
        evaluation.get("result_schema_version", EVALUATION_SCHEMA_VERSION)
    )
    if version not in {
        EVALUATION_SCHEMA_VERSION,
        PREFERENCE_EVALUATION_SCHEMA_VERSION,
        HIERARCHICAL_PREFERENCE_EVALUATION_SCHEMA_VERSION,
        SAFE_PRODUCTION_PREFERENCE_EVALUATION_SCHEMA_VERSION,
        NEUTRAL_GATE_SAFE_VARIANCE_EVALUATION_SCHEMA_VERSION,
        SAFE_MONOTONE_FLOW_GATE_EVALUATION_SCHEMA_VERSION,
        COUNTERFACTUAL_PREFERENCE_CONSISTENCY_EVALUATION_SCHEMA_VERSION,
        E1_WARMSTART_SAFE_GATE_EVALUATION_SCHEMA_VERSION,
        E1_WARMSTART_SAFE_GATE_V2_1_EVALUATION_SCHEMA_VERSION,
        TIERED_TRAINING_GATES_EVALUATION_SCHEMA_VERSION,
        MO_ALNS_EVALUATION_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported evaluation result schema {version!r}")
    return version


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
    """Return the immutable paper-quality metric, with a legacy fallback."""

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
    if normalized != CANONICAL_QUALITY_METRIC:
        raise ValueError(
            f"{QUALITY_METRIC_VERSION} is immutable and must use "
            "flow/cost/variance scales 1200/1000/50 and weights 0.5/0.3/0.2"
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


def aggregate_preference_diagnostics(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Aggregate top-1 preference diagnostics by ranked decision count."""

    decision_count = 0
    override_count = 0
    weighted_logit_std = 0.0
    head_totals: dict[str, list[float | int]] = {
        "production": [0, 0, 0.0],
        "worker": [0, 0, 0.0],
    }
    production_conditional_overrides = 0
    worker_variance_overrides = 0
    worker_direct_flow_max_abs = 0.0
    worker_direct_cost_max_abs = 0.0
    worker_direct_variance_max_abs = 0.0
    unsafe_worker_preference_selection_count = 0
    gate_state_count = 0
    gate_commit_selected_count = 0
    gate_defer_selected_count = 0
    gate_commit_probability_sum = 0.0
    gate_defer_probability_sum = 0.0
    gate_logit_margin_sum = 0.0
    gate_base_commit_probability_sum = 0.0
    gate_base_defer_probability_sum = 0.0
    gate_commit_logit_boost_sum = 0.0
    gate_residual_active_count = 0
    gate_flip_count = 0
    counterfactual_eligible_count = 0
    counterfactual_high_flow_flip_count = 0
    counterfactual_state_scale_sum = 0.0
    counterfactual_state_scale_max = 0.0
    counterfactual_identity_violation_count = 0
    counterfactual_monotonicity_violation_count = 0
    centered_dual_legal_count = 0
    centered_flow_cost_flip_count = 0
    centered_flow_variance_flip_count = 0
    centered_monotonicity_violation_count = 0
    for row in rows:
        count = int(row.get("ranker_top_decision_count", 0) or 0)
        overrides = int(row.get("preference_override_count", 0) or 0)
        mean_logit_std = float(
            row.get("mean_preference_logit_std", 0.0) or 0.0
        )
        if count < 0 or overrides < 0 or overrides > count:
            raise ValueError("preference diagnostic counts are inconsistent")
        if not math.isfinite(mean_logit_std) or mean_logit_std < 0.0:
            raise ValueError("preference logit standard deviation is invalid")
        decision_count += count
        override_count += overrides
        weighted_logit_std += count * mean_logit_std
        for head in head_totals:
            head_count = int(
                row.get(f"{head}_ranker_top_decision_count", 0) or 0
            )
            head_overrides = int(
                row.get(f"{head}_preference_override_count", 0) or 0
            )
            head_mean_std = float(
                row.get(f"{head}_mean_preference_logit_std", 0.0) or 0.0
            )
            if (
                head_count < 0
                or head_overrides < 0
                or head_overrides > head_count
                or not math.isfinite(head_mean_std)
                or head_mean_std < 0.0
            ):
                raise ValueError(
                    f"{head} preference diagnostic values are inconsistent"
                )
            head_totals[head][0] += head_count
            head_totals[head][1] += head_overrides
            head_totals[head][2] += head_count * head_mean_std
        production_conditional_overrides += int(
            row.get("production_conditional_preference_override_count", 0)
            or 0
        )
        worker_variance_overrides += int(
            row.get("worker_variance_preference_override_count", 0) or 0
        )
        worker_direct_flow_max_abs = max(
            worker_direct_flow_max_abs,
            float(
                row.get(
                    "worker_direct_preference_flow_logit_max_abs", 0.0
                )
                or 0.0
            ),
        )
        worker_direct_cost_max_abs = max(
            worker_direct_cost_max_abs,
            float(
                row.get(
                    "worker_direct_preference_cost_logit_max_abs", 0.0
                )
                or 0.0
            ),
        )
        worker_direct_variance_max_abs = max(
            worker_direct_variance_max_abs,
            float(
                row.get(
                    "worker_direct_preference_variance_logit_max_abs", 0.0
                )
                or 0.0
            ),
        )
        unsafe_worker_preference_selection_count += int(
            row.get("unsafe_worker_preference_selection_count", 0) or 0
        )
        row_gate_count = int(row.get("production_gate_state_count", 0) or 0)
        if row_gate_count < 0:
            raise ValueError("production gate decision count is invalid")
        gate_state_count += row_gate_count
        gate_commit_selected_count += int(
            row.get("production_gate_commit_selected_count", 0) or 0
        )
        gate_defer_selected_count += int(
            row.get("production_gate_defer_selected_count", 0) or 0
        )
        gate_commit_probability_sum += row_gate_count * float(
            row.get("mean_production_gate_commit_probability", 0.0) or 0.0
        )
        gate_defer_probability_sum += row_gate_count * float(
            row.get("mean_production_gate_defer_probability", 0.0) or 0.0
        )
        gate_logit_margin_sum += row_gate_count * float(
            row.get("mean_production_gate_logit_margin", 0.0) or 0.0
        )
        gate_base_commit_probability_sum += row_gate_count * float(
            row.get("mean_production_gate_base_commit_probability", 0.0) or 0.0
        )
        gate_base_defer_probability_sum += row_gate_count * float(
            row.get("mean_production_gate_base_defer_probability", 0.0) or 0.0
        )
        gate_commit_logit_boost_sum += row_gate_count * float(
            row.get("mean_production_gate_commit_logit_boost", 0.0) or 0.0
        )
        gate_residual_active_count += int(
            row.get("production_gate_residual_active_count", 0) or 0
        )
        gate_flip_count += int(
            row.get("production_gate_base_defer_to_final_commit_flip_count", 0) or 0
        )
        row_counterfactual_eligible = int(
            row.get("counterfactual_eligible_state_count", 0) or 0
        )
        row_counterfactual_flips = int(
            row.get("counterfactual_high_flow_commit_flip_count", 0) or 0
        )
        if (
            row_counterfactual_eligible < 0
            or row_counterfactual_flips < 0
            or row_counterfactual_flips > row_counterfactual_eligible
        ):
            raise ValueError("counterfactual gate diagnostics are inconsistent")
        counterfactual_eligible_count += row_counterfactual_eligible
        counterfactual_high_flow_flip_count += row_counterfactual_flips
        row_scale = float(
            row.get("mean_counterfactual_state_residual_scale", 0.0) or 0.0
        )
        row_max_scale = float(
            row.get("max_counterfactual_state_residual_scale", 0.0) or 0.0
        )
        if (
            not math.isfinite(row_scale)
            or not math.isfinite(row_max_scale)
            or row_scale < 0.0
            or row_max_scale < row_scale
        ):
            raise ValueError("counterfactual state scale diagnostics are invalid")
        counterfactual_state_scale_sum += row_gate_count * row_scale
        counterfactual_state_scale_max = max(counterfactual_state_scale_max, row_max_scale)
        counterfactual_identity_violation_count += int(
            row.get("counterfactual_low_flow_identity_violation_count", 0) or 0
        )
        counterfactual_monotonicity_violation_count += int(
            row.get("counterfactual_monotonicity_violation_count", 0) or 0
        )
        row_centered_dual = int(
            row.get("centered_gate_dual_legal_state_count", 0) or 0
        )
        row_flow_cost_flips = int(
            row.get("centered_gate_flow_cost_flip_count", 0) or 0
        )
        row_flow_variance_flips = int(
            row.get("centered_gate_flow_variance_flip_count", 0) or 0
        )
        if (
            row_centered_dual < 0
            or not 0 <= row_flow_cost_flips <= row_centered_dual
            or not 0 <= row_flow_variance_flips <= row_centered_dual
        ):
            raise ValueError("centered gate diagnostics are inconsistent")
        centered_dual_legal_count += row_centered_dual
        centered_flow_cost_flip_count += row_flow_cost_flips
        centered_flow_variance_flip_count += row_flow_variance_flips
        centered_monotonicity_violation_count += int(
            row.get("centered_gate_monotonicity_violation_count", 0) or 0
        )
    result = {
        "ranker_top_decision_count": decision_count,
        "preference_override_count": override_count,
        "preference_override_rate": (
            override_count / decision_count if decision_count else 0.0
        ),
        "mean_preference_logit_std": (
            weighted_logit_std / decision_count if decision_count else 0.0
        ),
    }
    for head, totals in head_totals.items():
        head_count = int(totals[0])
        head_overrides = int(totals[1])
        head_weighted_std = float(totals[2])
        result[f"{head}_ranker_top_decision_count"] = head_count
        result[f"{head}_preference_override_count"] = head_overrides
        result[f"{head}_preference_override_rate"] = (
            head_overrides / head_count if head_count else 0.0
        )
        result[f"{head}_mean_preference_logit_std"] = (
            head_weighted_std / head_count if head_count else 0.0
        )
    production_count = int(head_totals["production"][0])
    worker_count = int(head_totals["worker"][0])
    result.update(
        {
            "production_conditional_preference_override_count": (
                production_conditional_overrides
            ),
            "production_conditional_preference_override_rate": (
                production_conditional_overrides / production_count
                if production_count
                else 0.0
            ),
            "worker_variance_preference_override_count": (
                worker_variance_overrides
            ),
            "worker_variance_preference_override_rate": (
                worker_variance_overrides / worker_count
                if worker_count
                else 0.0
            ),
            "worker_direct_preference_flow_logit_max_abs": (
                worker_direct_flow_max_abs
            ),
            "worker_direct_preference_cost_logit_max_abs": (
                worker_direct_cost_max_abs
            ),
            "worker_direct_preference_variance_logit_max_abs": (
                worker_direct_variance_max_abs
            ),
            "unsafe_worker_preference_selection_count": (
                unsafe_worker_preference_selection_count
            ),
            "production_gate_state_count": gate_state_count,
            "production_gate_commit_selected_count": (
                gate_commit_selected_count
            ),
            "production_gate_defer_selected_count": gate_defer_selected_count,
            "mean_production_gate_commit_probability": (
                gate_commit_probability_sum / gate_state_count
                if gate_state_count
                else 0.0
            ),
            "mean_production_gate_defer_probability": (
                gate_defer_probability_sum / gate_state_count
                if gate_state_count
                else 0.0
            ),
            "mean_production_gate_logit_margin": (
                gate_logit_margin_sum / gate_state_count
                if gate_state_count
                else 0.0
            ),
            "mean_production_gate_base_commit_probability": (
                gate_base_commit_probability_sum / gate_state_count if gate_state_count else 0.0
            ),
            "mean_production_gate_base_defer_probability": (
                gate_base_defer_probability_sum / gate_state_count if gate_state_count else 0.0
            ),
            "mean_production_gate_commit_logit_boost": (
                gate_commit_logit_boost_sum / gate_state_count if gate_state_count else 0.0
            ),
            "production_gate_residual_active_count": gate_residual_active_count,
            "production_gate_base_defer_to_final_commit_flip_count": gate_flip_count,
            "counterfactual_eligible_state_count": counterfactual_eligible_count,
            "counterfactual_high_flow_commit_flip_count": (
                counterfactual_high_flow_flip_count
            ),
            "counterfactual_high_flow_commit_flip_rate": (
                counterfactual_high_flow_flip_count
                / counterfactual_eligible_count
                if counterfactual_eligible_count
                else 0.0
            ),
            "mean_counterfactual_state_residual_scale": (
                counterfactual_state_scale_sum / gate_state_count
                if gate_state_count
                else 0.0
            ),
            "max_counterfactual_state_residual_scale": (
                counterfactual_state_scale_max
            ),
            "counterfactual_low_flow_identity_violation_count": (
                counterfactual_identity_violation_count
            ),
            "counterfactual_monotonicity_violation_count": (
                counterfactual_monotonicity_violation_count
            ),
            "centered_gate_dual_legal_state_count": centered_dual_legal_count,
            "centered_gate_flow_cost_flip_count": centered_flow_cost_flip_count,
            "centered_gate_flow_variance_flip_count": (
                centered_flow_variance_flip_count
            ),
            "centered_gate_flow_cost_flip_rate": (
                centered_flow_cost_flip_count / centered_dual_legal_count
                if centered_dual_legal_count
                else 0.0
            ),
            "centered_gate_flow_variance_flip_rate": (
                centered_flow_variance_flip_count / centered_dual_legal_count
                if centered_dual_legal_count
                else 0.0
            ),
            "centered_gate_extreme_flip_rate": (
                min(centered_flow_cost_flip_count, centered_flow_variance_flip_count)
                / centered_dual_legal_count
                if centered_dual_legal_count
                else 0.0
            ),
            "centered_gate_monotonicity_violation_count": (
                centered_monotonicity_violation_count
            ),
        }
    )
    return result


def aggregate_matching_recovery_diagnostics(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Aggregate E2.3 matching diagnostics with count-weighted ratios."""

    values = list(rows)

    def total(name: str) -> int:
        result = sum(int(row.get(name, 0) or 0) for row in values)
        if result < 0:
            raise ValueError(f"{name} must be non-negative")
        return result

    def maximum(name: str) -> int:
        result = max(
            (int(row.get(name, 0) or 0) for row in values),
            default=0,
        )
        if result < 0:
            raise ValueError(f"{name} must be non-negative")
        return result

    future_candidates = total(
        "future_installation_admission_candidate_count"
    )
    future_masked = total(
        "future_installation_admission_masked_action_count"
    )
    if future_masked > future_candidates:
        raise ValueError(
            "future installation admission counts are inconsistent"
        )
    maximum_projected = maximum("maximum_projected_installation_deficit")
    return {
        "current_worker_matching_deficit": maximum(
            "current_worker_matching_deficit"
        ),
        "maximum_worker_matching_deficit": maximum(
            "maximum_worker_matching_deficit"
        ),
        "deficit_reducing_worker_action_candidate_count": total(
            "deficit_reducing_worker_action_candidate_count"
        ),
        "deficit_reducing_worker_action_count": total(
            "deficit_reducing_worker_action_count"
        ),
        "matching_deficit_recovery_advance_count": total(
            "matching_deficit_recovery_advance_count"
        ),
        "current_matching_admission_masked_action_count": total(
            "current_matching_admission_masked_action_count"
        ),
        "future_installation_admission_candidate_count": future_candidates,
        "future_installation_admission_masked_action_count": future_masked,
        "future_installation_admission_masked_action_ratio": (
            future_masked / future_candidates if future_candidates else 0.0
        ),
        "future_installation_matching_deficit_after_commit": (
            maximum_projected
        ),
        "maximum_projected_installation_deficit": maximum_projected,
        **{
            name: total(name)
            for name in (
                "temporal_oracle_call_count",
                "temporal_oracle_cache_hit_count",
                "temporal_oracle_searched_nodes",
                "temporal_oracle_feasible_count",
                "temporal_oracle_infeasible_count",
                "temporal_oracle_unknown_count",
                "temporal_worker_action_rescued_count",
                "temporal_future_installation_rescued_count",
                "temporal_delayed_disassembly_rescued_count",
            )
        },
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
    schema_version: str = EVALUATION_SCHEMA_VERSION,
) -> dict[str, Any]:
    if schema_version not in {
        EVALUATION_SCHEMA_VERSION,
        PREFERENCE_EVALUATION_SCHEMA_VERSION,
        HIERARCHICAL_PREFERENCE_EVALUATION_SCHEMA_VERSION,
        SAFE_PRODUCTION_PREFERENCE_EVALUATION_SCHEMA_VERSION,
        NEUTRAL_GATE_SAFE_VARIANCE_EVALUATION_SCHEMA_VERSION,
        SAFE_MONOTONE_FLOW_GATE_EVALUATION_SCHEMA_VERSION,
            COUNTERFACTUAL_PREFERENCE_CONSISTENCY_EVALUATION_SCHEMA_VERSION,
            E1_WARMSTART_SAFE_GATE_EVALUATION_SCHEMA_VERSION,
            E1_WARMSTART_SAFE_GATE_V2_1_EVALUATION_SCHEMA_VERSION,
            TIERED_TRAINING_GATES_EVALUATION_SCHEMA_VERSION,
            MO_ALNS_EVALUATION_SCHEMA_VERSION,
        }:
        raise ValueError(f"unsupported evaluation result schema {schema_version!r}")
    normalized_metric = evaluation_quality_metric(
        {}
        if quality_metric is None
        else {"evaluation": {"quality_metric": dict(quality_metric)}}
    )
    metric_hash = quality_metric_sha256(normalized_metric)
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
        "quality_score": summarize_values(
            row.get("quality_score") for row in rows
        ),
        "heuristic_quality_score": summarize_values(
            row.get("heuristic_quality_score") for row in rows
        ),
        "reward_quality_score": summarize_values(
            row.get("reward_quality_score") for row in rows
        ),
        "preference_quality_score": summarize_values(
            row.get("preference_quality_score") for row in rows
        ),
        "heuristic_reward_quality_score": summarize_values(
            row.get("heuristic_reward_quality_score") for row in rows
        ),
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
        **{
            name: summarize_values(row.get(name) for row in rows)
            for name in (
                "maximum_worker_matching_deficit",
                "current_worker_matching_deficit",
                "deficit_reducing_worker_action_candidate_count",
                "deficit_reducing_worker_action_count",
                "matching_deficit_recovery_advance_count",
                "current_matching_admission_masked_action_count",
                "future_installation_admission_candidate_count",
                "future_installation_admission_masked_action_count",
                "future_installation_admission_masked_action_ratio",
                "future_installation_matching_deficit_after_commit",
                "maximum_projected_installation_deficit",
                "temporal_oracle_call_count",
                "temporal_oracle_cache_hit_count",
                "temporal_oracle_searched_nodes",
                "temporal_oracle_feasible_count",
                "temporal_oracle_infeasible_count",
                "temporal_oracle_unknown_count",
                "temporal_worker_action_rescued_count",
                "temporal_future_installation_rescued_count",
                "temporal_delayed_disassembly_rescued_count",
            )
        },
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
        "production_defer_recovery_improvement_count": summarize_values(
            row.get("production_defer_recovery_improvement_count")
            for row in rows
        ),
        "production_defer_wait_time": summarize_values(
            row.get("production_defer_wait_time") for row in rows
        ),
        **{
            name: summarize_values(row.get(name) for row in rows)
            for name in (
                "ranker_top_selection_rate",
                "context_override_rate",
                "preference_override_count",
                "preference_override_rate",
                "mean_preference_logit_std",
                "production_pair_plus_defer_state_count",
                "production_decision_state_count",
                "production_pair_plus_defer_ratio",
                "worker_pair_plus_advance_state_count",
                "worker_decision_state_count",
                "worker_pair_plus_advance_ratio",
                "mean_commit_set_logit",
                "conditional_worker_wait_opportunity_count",
                "conditional_worker_wait_selected_count",
                "conditional_worker_wait_total_ticks",
                "conditional_worker_wait_pair_gain_sum",
                "conditional_worker_wait_fatigue_improvement_sum",
                "conditional_worker_wait_duration_improvement_ticks_sum",
                "conditional_worker_wait_max_consecutive_observed",
                "reconfiguration_reuse_count",
                "qualification_scarcity_regret",
            )
        },
        **{
            name: summarize_values(row.get(name) for row in rows)
            for name in (
                "direct_process_action_count",
                "commit_reconfig_action_count",
                "defer_production_action_count",
                "worker_assign_action_count",
                "advance_event_action_count",
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
        "feasibility_proxy_return": summarize_values(
            row.get("feasibility_proxy_return") for row in rows
        ),
        **{
            name: summarize_values(row.get(name) for row in rows)
            for name in (
                "forced_action_state_count",
                "forced_production_count",
                "forced_worker_count",
                "forced_pair_count",
                "forced_advance_count",
                "forced_production_pair_count",
                "forced_production_advance_count",
                "forced_worker_pair_count",
                "forced_worker_advance_count",
                "forced_pair_advance_blocked_non_delay_count",
                "forced_worker_pair_non_delay_count",
                "forced_pair_advance_physically_unavailable_count",
                "forced_advance_pair_physically_unavailable_count",
                "forced_wait_dis_count",
                "forced_wait_ins_count",
                "forced_mixed_wait_stage_count",
                "forced_phase_handoff_count",
                "forced_recovery_advance_count",
                "forced_future_event_advance_count",
                "forced_action_chain_count",
                "longest_forced_action_chain",
                "mean_forced_action_chain_length",
            )
        },
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
        "quality_score_heuristic_gap_percent": summarize_values(
            relative_gap_percent(
                row.get("quality_score"), row.get("heuristic_quality_score")
            )
            for row in rows
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
    preference_diagnostics = aggregate_preference_diagnostics(rows)
    matching_recovery_diagnostics = (
        aggregate_matching_recovery_diagnostics(rows)
    )
    return {
        "evaluation_schema_version": schema_version,
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
        "completed_metrics": completed_metrics,
        "all_instance_metrics": all_instance_metrics,
        "gap_metrics": gap_metrics,
        "tail_metrics": tail_metrics,
        **preference_diagnostics,
        **matching_recovery_diagnostics,
    }


def evaluation_selection_key(
    aggregate: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Return the M1 completion-constrained aligned-quality key.

    The four-field shape is retained for log/checkpoint compatibility.  Only
    completion and the mean per-instance Q12 score participate in selection.
    """
    metrics = aggregate["all_instance_metrics"]

    def mean(name: str) -> float:
        value = metrics.get(name, {}).get("mean")
        return math.inf if value is None else float(value)

    quality = mean("quality_score")
    if not math.isfinite(quality):
        flow = mean("flow_time_objective")
        cost = mean("reconfiguration_cost")
        variance = mean("worker_load_variance")
        if all(math.isfinite(value) for value in (flow, cost, variance)):
            quality = (
                0.5 * flow / (1200.0 + flow)
                + 0.3 * cost / (1000.0 + cost)
                + 0.2 * variance / (50.0 + variance)
            )

    return (
        -float(aggregate["completion_rate"]),
        quality,
        0.0,
        0.0,
    )
