from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pareto_analysis import hypervolume_3d, nondominated_indices, normalize_objectives


OBJECTIVE_FIELDS = (
    "flow_time_objective",
    "reconfiguration_cost",
    "worker_load_variance",
)
ENDPOINTS = {
    (1.0, 0.0, 0.0): "flow",
    (0.0, 1.0, 0.0): "cost",
    (0.0, 0.0, 1.0): "variance",
}
EXPECTED_SIMPLEX_GRID = frozenset(
    (flow / 10.0, cost / 10.0, (10 - flow - cost) / 10.0)
    for flow in range(11)
    for cost in range(11 - flow)
)


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    replicates: int
    seed: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "replicates": self.replicates,
            "seed": self.seed,
        }


def paired_instance_block_bootstrap(
    candidate: Sequence[float],
    incumbent: Sequence[float],
    *,
    replicates: int = 10_000,
    seed: int = 20260811,
) -> BootstrapInterval:
    """Paired bootstrap over instances; each value already contains all 66 lambdas."""

    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(incumbent, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or left.size < 2:
        raise ValueError("paired bootstrap requires aligned instance vectors")
    if replicates != 10_000:
        raise ValueError("V8 promotion requires exactly 10,000 bootstrap replicates")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("bootstrap inputs must be finite")
    delta = left - right
    rng = np.random.default_rng(int(seed))
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        count = min(1000, replicates - start)
        indices = rng.integers(0, delta.size, size=(count, delta.size))
        means[start : start + count] = delta[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return BootstrapInterval(float(delta.mean()), float(lower), float(upper), replicates, int(seed))


def _preference_tuple(row: Mapping[str, Any]) -> tuple[float, float, float]:
    raw = row.get("preference")
    if not isinstance(raw, Mapping):
        raise ValueError("promotion row is missing preference mapping")
    values = tuple(float(raw[name]) for name in ("flow", "cost", "variance"))
    if (
        any(not math.isfinite(value) or value < 0.0 for value in values)
        or not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-8)
    ):
        raise ValueError("promotion row contains an invalid simplex preference")
    return values  # type: ignore[return-value]


def _grid_key(values: Sequence[float]) -> tuple[float, float, float]:
    return tuple(round(float(value), 10) for value in values)  # type: ignore[return-value]


def _validated_blocks(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    blocks: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        blocks[str(row["instance_id"])].append(row)
    if not blocks:
        raise ValueError("promotion rows cannot be empty")
    for instance_id, block in blocks.items():
        if len(block) != 66:
            raise ValueError(f"instance {instance_id} does not contain all 66 preferences")
        keys = {str(row["preference_key"]) for row in block}
        if len(keys) != 66:
            raise ValueError(f"instance {instance_id} contains duplicate preferences")
        preferences = {_grid_key(_preference_tuple(row)) for row in block}
        expected = {_grid_key(value) for value in EXPECTED_SIMPLEX_GRID}
        if preferences != expected:
            raise ValueError(
                f"instance {instance_id} does not contain the fixed step-0.1 simplex grid"
            )
    return dict(sorted(blocks.items()))


def _safety_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_completion: float,
) -> tuple[bool, dict[str, Any]]:
    by_preference: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_preference[str(row["preference_key"])].append(row)
    rates = {
        key: sum(bool(row["terminated"]) and not bool(row["truncated"]) for row in values)
        / len(values)
        for key, values in by_preference.items()
    }
    violations = sum(int(row.get("schedule_violation_count", 0)) for row in rows)
    unsafe = sum(
        float(row.get("maximum_worker_fatigue", math.inf))
        > float(row.get("safe_fatigue_limit", -math.inf)) + 1e-9
        for row in rows
    )
    failed_instances = len(
        {
            str(row["instance_id"])
            for row in rows
            if not bool(row["terminated"]) or bool(row["truncated"])
        }
    )
    passed = bool(
        min(rates.values()) >= minimum_completion
        and violations == 0
        and unsafe == 0
    )
    return passed, {
        "minimum_completion_rate": min(rates.values()),
        "completion_rate_by_preference": dict(sorted(rates.items())),
        "failed_instance_count": failed_instances,
        "failed_instance_count_is_diagnostic": True,
        "schedule_violation_count": violations,
        "physical_safety_violation_count": unsafe,
    }


def _endpoint_gate(
    rows: Sequence[Mapping[str, Any]],
    upper_bounds: Mapping[str, float],
) -> tuple[bool, dict[str, Any]]:
    if set(upper_bounds) != set(ENDPOINTS.values()) or any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in upper_bounds.values()
    ):
        raise ValueError("endpoint prediction upper bounds are incomplete or invalid")
    values: dict[str, list[float]] = {name: [] for name in ENDPOINTS.values()}
    for row in rows:
        preference = _preference_tuple(row)
        for endpoint, objective in ENDPOINTS.items():
            if all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(preference, endpoint)):
                values[objective].append(float(row[OBJECTIVE_FIELDS[list(ENDPOINTS.values()).index(objective)]]))
    means = {name: float(np.mean(items)) if items else math.inf for name, items in values.items()}
    passed = all(means[name] <= float(upper_bounds[name]) for name in ENDPOINTS.values())
    return passed, {"means": means, "prediction_upper_bounds": dict(upper_bounds)}


def _primary_scores(blocks: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[float]:
    result = []
    for block in blocks.values():
        values = [float(row["preference_quality_score"]) for row in block]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("primary scalarized scores must be finite")
        result.append(float(np.mean(values)))
    return result


def _hypervolumes(
    blocks: Mapping[str, Sequence[Mapping[str, Any]]],
    scales: Sequence[float],
) -> list[float]:
    result = []
    for block in blocks.values():
        points = [
            normalize_objectives(
                tuple(float(row[name]) for name in OBJECTIVE_FIELDS), scales
            )
            for row in block
        ]
        front = [points[index] for index in nondominated_indices(points)]
        result.append(hypervolume_3d(front, reference=(1.0, 1.0, 1.0)))
    return result


def compare_preference_conditioned_checkpoints(
    candidate_rows: Sequence[Mapping[str, Any]],
    incumbent_rows: Sequence[Mapping[str, Any]],
    *,
    scales: Sequence[float],
    endpoint_prediction_upper_bounds: Mapping[str, float],
    audit: bool = False,
    bootstrap_seed: int = 20260811,
) -> dict[str, Any]:
    """Apply the V8 safety, endpoint, primary-score, and paired-HV hierarchy."""

    candidate = _validated_blocks(candidate_rows)
    incumbent = _validated_blocks(incumbent_rows)
    if tuple(candidate) != tuple(incumbent):
        raise ValueError("candidate and incumbent must contain identical instance blocks")
    required_instances = 200 if audit else 50
    if len(candidate) != required_instances:
        raise ValueError(f"V8 comparison requires {required_instances} instances")
    safety, safety_detail = _safety_gate(
        candidate_rows,
        minimum_completion=0.98 if audit else 0.95,
    )
    endpoint, endpoint_detail = _endpoint_gate(
        candidate_rows, endpoint_prediction_upper_bounds
    )
    result: dict[str, Any] = {
        "protocol": "preference_conditioned_pareto_v8",
        "instance_count": len(candidate),
        "preference_count": 66,
        "bootstrap_unit": "instance_block_with_all_66_preferences",
        "safety_gate_pass": safety,
        "safety": safety_detail,
        "endpoint_gate_pass": endpoint,
        "endpoints": endpoint_detail,
        "accepted": False,
    }
    if not safety or not endpoint:
        result["decision"] = "reject_candidate_gate_failure"
        return result
    primary = paired_instance_block_bootstrap(
        _primary_scores(candidate),
        _primary_scores(incumbent),
        seed=bootstrap_seed,
    )
    result["primary_delta"] = primary.as_dict()
    if primary.upper < 0.0:
        result.update(accepted=True, decision="accept_candidate_primary_ci_below_zero")
        return result
    if primary.lower > 0.0:
        result["decision"] = "reject_candidate_primary_ci_above_zero"
        return result
    hv = paired_instance_block_bootstrap(
        _hypervolumes(candidate, scales),
        _hypervolumes(incumbent, scales),
        seed=bootstrap_seed + 1,
    )
    result["hypervolume_delta"] = hv.as_dict()
    if hv.lower > 0.0:
        result.update(accepted=True, decision="accept_candidate_hv_ci_above_zero")
    else:
        result["decision"] = "keep_incumbent_hv_not_significant"
    return result
