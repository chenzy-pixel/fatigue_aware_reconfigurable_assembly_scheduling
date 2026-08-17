"""Extract empirical C0/E1 Pareto fronts from a protocol-v2 audit.

The primary analysis is performed separately for every fixed validation
instance.  A test-set-mean front is emitted only as an auxiliary diagnostic;
objective vectors from different scheduling instances are never pooled into a
single optimization problem.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from result.io import write_csv, write_json


ANALYSIS_SCHEMA_VERSION = "1.0.0"
EXPECTED_AUDIT_PROTOCOL = "v7_e1_protocol_v2"
EXPECTED_RESULT_SCHEMA = "4.1.0"
EXPECTED_ARMS = ("c0", "e1")
EXPECTED_SAMPLING_SEEDS = (100011, 100012, 100013)
SERIAL_MODES = ("greedy_serial", "sampled_serial")
OBJECTIVE_FIELDS = (
    "flow_time_objective",
    "reconfiguration_cost",
    "worker_load_variance",
)
NORMALIZED_FIELDS = (
    "normalized_flow_time_objective",
    "normalized_reconfiguration_cost",
    "normalized_worker_load_variance",
)
REFERENCE_POINT = (1.0, 1.0, 1.0)
RELATIVE_TOLERANCE = 1e-9


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_bool(value: Any, *, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be true or false, got {value!r}")


def _parse_int(value: Any, *, field: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer, got {value!r}") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return int(number)


def _parse_finite(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


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


def hypervolume_3d(
    points: Iterable[Sequence[float]],
    *,
    reference: Sequence[float] = REFERENCE_POINT,
) -> float:
    """Return exact dominated hypervolume for three-dimensional minimization.

    The implementation partitions the union of axis-aligned boxes on all point
    coordinates.  Audit fronts contain at most a handful of points per instance,
    so this exact grid method is both transparent and fast.
    """

    reference_tuple = tuple(float(value) for value in reference)
    if len(reference_tuple) != 3:
        raise ValueError("the hypervolume reference point must be three-dimensional")
    if any(not math.isfinite(value) for value in reference_tuple):
        raise ValueError("the hypervolume reference point must be finite")

    unique: list[tuple[float, float, float]] = []
    for raw_point in points:
        point = tuple(float(value) for value in raw_point)
        if len(point) != 3:
            raise ValueError("hypervolume points must be three-dimensional")
        if any(not math.isfinite(value) for value in point):
            raise ValueError("hypervolume points must be finite")
        if any(
            value > bound + _relative_slack(value, bound, RELATIVE_TOLERANCE)
            for value, bound in zip(point, reference_tuple, strict=True)
        ):
            raise ValueError("a hypervolume point lies beyond the reference point")
        clipped = tuple(
            min(value, bound)
            for value, bound in zip(point, reference_tuple, strict=True)
        )
        if not any(vectors_equal(clipped, existing) for existing in unique):
            unique.append(clipped)  # type: ignore[arg-type]
    if not unique:
        return 0.0

    coordinates = [
        sorted({point[dimension] for point in unique} | {reference_tuple[dimension]})
        for dimension in range(3)
    ]
    volume = 0.0
    for x_lower, x_upper in zip(coordinates[0], coordinates[0][1:]):
        for y_lower, y_upper in zip(coordinates[1], coordinates[1][1:]):
            for z_lower, z_upper in zip(coordinates[2], coordinates[2][1:]):
                covered = any(
                    point[0] <= x_lower
                    and point[1] <= y_lower
                    and point[2] <= z_lower
                    for point in unique
                )
                if covered:
                    volume += (
                        (x_upper - x_lower)
                        * (y_upper - y_lower)
                        * (z_upper - z_lower)
                    )
    return float(volume)


def _point_groups(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda row: (
            *row["objectives"],
            row["arm"],
            row["decode_mode"],
            row["sampling_seed"] or -1,
            row["action_trace_sha256"],
        ),
    ):
        matching = next(
            (
                group
                for group in groups
                if vectors_equal(group["objectives"], candidate["objectives"])
            ),
            None,
        )
        if matching is None:
            groups.append(
                {
                    "objectives": candidate["objectives"],
                    "normalized_objectives": candidate["normalized_objectives"],
                    "members": [candidate],
                }
            )
        else:
            matching["members"].append(candidate)
    return groups


def _validate_audit(
    audit: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[float, float, float], dict[str, int]]:
    _require(
        audit.get("audit_protocol_version") == EXPECTED_AUDIT_PROTOCOL,
        f"audit protocol must be {EXPECTED_AUDIT_PROTOCOL}",
    )
    _require(
        audit.get("result_schema_version") == EXPECTED_RESULT_SCHEMA,
        f"audit result schema must be {EXPECTED_RESULT_SCHEMA}",
    )
    _require(bool(audit.get("all_checks_passed")), "audit checks did not pass")
    _require(
        tuple(audit.get("sampling_seeds", ())) == EXPECTED_SAMPLING_SEEDS,
        f"sampling seeds must be {EXPECTED_SAMPLING_SEEDS}",
    )
    arms = audit.get("arms")
    _require(isinstance(arms, Mapping), "audit.arms must be an object")

    manifest_hashes: set[str] = set()
    metric_hashes: set[str] = set()
    quality_metrics: list[dict[str, Any]] = []
    algorithm_seeds: dict[str, int] = {}
    datasets: set[str] = set()
    instance_counts: set[int] = set()
    provenance: dict[str, Any] = {}
    for arm in EXPECTED_ARMS:
        _require(arm in arms, f"audit is missing arm {arm}")
        arm_payload = arms[arm]
        _require(isinstance(arm_payload, Mapping), f"audit arm {arm} is invalid")
        greedy = arm_payload.get("greedy")
        _require(isinstance(greedy, Mapping), f"audit arm {arm} has no greedy result")
        _require(
            greedy.get("evaluation_schema_version") == EXPECTED_RESULT_SCHEMA,
            f"arm {arm} evaluation schema is incompatible",
        )
        _require(
            bool(arm_payload.get("sampled_global_rng_unchanged")),
            f"arm {arm} sampled evaluation changed global RNG state",
        )
        datasets.add(str(greedy.get("dataset")))
        instance_counts.add(int(greedy.get("instance_count", -1)))
        manifest_hashes.add(str(greedy.get("dataset_manifest_sha256")))
        metric_hashes.add(str(greedy.get("quality_metric_sha256")))
        metric = greedy.get("quality_metric")
        _require(isinstance(metric, Mapping), f"arm {arm} has no quality metric")
        quality_metrics.append(dict(metric))
        checkpoint_metadata = arm_payload.get("checkpoint_metadata")
        _require(
            isinstance(checkpoint_metadata, Mapping),
            f"arm {arm} has no checkpoint metadata",
        )
        seed = checkpoint_metadata.get(
            "algorithm_seed", checkpoint_metadata.get("seed")
        )
        algorithm_seeds[arm] = int(seed)
        arm_provenance = arm_payload.get("provenance")
        _require(
            isinstance(arm_provenance, Mapping),
            f"arm {arm} has no provenance",
        )
        _require(
            arm_provenance.get("dataset_manifest_sha256")
            == greedy.get("dataset_manifest_sha256"),
            f"arm {arm} provenance manifest hash does not match evaluation",
        )
        _require(
            arm_provenance.get("quality_metric_sha256")
            == greedy.get("quality_metric_sha256"),
            f"arm {arm} provenance quality hash does not match evaluation",
        )
        provenance[arm] = {
            "checkpoint": arm_payload.get("checkpoint"),
            "checkpoint_sha256": arm_provenance.get("checkpoint_sha256"),
            "network_weights_sha256": arm_provenance.get(
                "network_weights_sha256"
            ),
            "effective_config_sha256": arm_provenance.get(
                "effective_config_sha256"
            ),
            "source_state_sha256": arm_provenance.get("source_state_sha256"),
            "algorithm_seed": algorithm_seeds[arm],
        }

    _require(datasets == {"validation"}, "audit must use the validation split")
    _require(len(instance_counts) == 1, "arms have different instance counts")
    _require(len(manifest_hashes) == 1, "arms use different dataset manifests")
    _require(len(metric_hashes) == 1, "arms use different quality metrics")
    _require(
        all(metric == quality_metrics[0] for metric in quality_metrics[1:]),
        "arms contain different quality metric definitions",
    )
    _require(
        set(algorithm_seeds.values()) == {11},
        "strict paired analysis requires algorithm seed 11 for both arms",
    )
    metric = quality_metrics[0]
    scales = (
        _parse_finite(metric.get("flow_scale"), field="quality_metric.flow_scale"),
        _parse_finite(metric.get("cost_scale"), field="quality_metric.cost_scale"),
        _parse_finite(
            metric.get("variance_scale"), field="quality_metric.variance_scale"
        ),
    )
    return (
        {
            "dataset": next(iter(datasets)),
            "instance_count": next(iter(instance_counts)),
            "dataset_manifest_sha256": next(iter(manifest_hashes)),
            "quality_metric_sha256": next(iter(metric_hashes)),
            "quality_metric": metric,
            "provenance": provenance,
        },
        scales,
        algorithm_seeds,
    )


def _parallel_copy_count(rows: Sequence[dict[str, str]]) -> int:
    serial_keys = {
        (
            row.get("arm"),
            row.get("instance_id"),
            row.get("sampling_seed"),
            row.get("action_trace_sha256"),
        )
        for row in rows
        if row.get("mode") == "sampled_serial"
    }
    count = 0
    for row in rows:
        mode = str(row.get("mode", ""))
        if mode in SERIAL_MODES:
            continue
        _require(
            mode.startswith("sampled_parallel_"),
            f"unexpected audit mode {mode!r}",
        )
        key = (
            row.get("arm"),
            row.get("instance_id"),
            row.get("sampling_seed"),
            row.get("action_trace_sha256"),
        )
        _require(
            key in serial_keys,
            "parallel audit row has no action-trace-identical serial counterpart",
        )
        count += 1
    return count


def _prepare_candidates(
    rows: Sequence[dict[str, str]],
    *,
    scales: Sequence[float],
    algorithm_seeds: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("instance_metrics.csv contains no rows")
    required = {
        "arm",
        "mode",
        "instance_id",
        "seed",
        "pressure_type",
        "cost_profile",
        "terminated",
        "truncated",
        "schedule_violation_count",
        "action_trace_sha256",
        "sampling_seed",
        *OBJECTIVE_FIELDS,
    }
    missing = sorted(required - set(rows[0]))
    _require(not missing, f"instance_metrics.csv is missing columns: {missing}")
    parallel_copies = _parallel_copy_count(rows)

    candidates: list[dict[str, Any]] = []
    invalid_reason_counts: dict[str, int] = defaultdict(int)
    invalid_rows = 0
    serial_rows = [row for row in rows if row.get("mode") in SERIAL_MODES]
    for row in serial_rows:
        arm = str(row["arm"])
        _require(arm in EXPECTED_ARMS, f"unexpected arm {arm!r}")
        terminated = _parse_bool(row["terminated"], field="terminated")
        truncated = _parse_bool(row["truncated"], field="truncated")
        violations = _parse_int(
            row["schedule_violation_count"], field="schedule_violation_count"
        )
        reasons: list[str] = []
        if not terminated:
            reasons.append("not_terminated")
        if truncated:
            reasons.append("truncated")
        if violations != 0:
            reasons.append("schedule_violation")
        if reasons:
            invalid_rows += 1
            for reason in reasons:
                invalid_reason_counts[reason] += 1
            continue

        mode = str(row["mode"])
        decode_mode = "greedy" if mode == "greedy_serial" else "sampled"
        sampling_seed = (
            None
            if decode_mode == "greedy"
            else _parse_int(row["sampling_seed"], field="sampling_seed")
        )
        if decode_mode == "sampled":
            _require(
                sampling_seed in EXPECTED_SAMPLING_SEEDS,
                f"unexpected sampling seed {sampling_seed}",
            )
        trace = str(row["action_trace_sha256"]).strip()
        _require(trace, "a serial candidate is missing action_trace_sha256")
        objectives = tuple(
            _parse_finite(row[field], field=field) for field in OBJECTIVE_FIELDS
        )
        normalized = normalize_objectives(objectives, scales)
        label = "greedy" if sampling_seed is None else f"sampled_{sampling_seed}"
        candidate_id = f"{arm}|{row['instance_id']}|{label}|{trace}"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "instance_id": str(row["instance_id"]),
                "instance_seed": _parse_int(row["seed"], field="seed"),
                "pressure_type": str(row["pressure_type"]),
                "cost_profile": str(row["cost_profile"]),
                "arm": arm,
                "algorithm_seed": int(algorithm_seeds[arm]),
                "source_mode": mode,
                "decode_mode": decode_mode,
                "sampling_seed": sampling_seed,
                "action_trace_sha256": trace,
                "objectives": objectives,
                "normalized_objectives": normalized,
            }
        )

    design_keys: set[tuple[str, str, str, int | None]] = set()
    for candidate in candidates:
        key = (
            candidate["arm"],
            candidate["instance_id"],
            candidate["decode_mode"],
            candidate["sampling_seed"],
        )
        _require(key not in design_keys, f"duplicate serial design cell {key}")
        design_keys.add(key)
    trace_keys = [
        (
            candidate["arm"],
            candidate["instance_id"],
            candidate["action_trace_sha256"],
        )
        for candidate in candidates
    ]
    _require(
        len(trace_keys) == len(set(trace_keys)),
        "serial candidates contain duplicate action traces within an arm/instance",
    )

    instances = sorted({candidate["instance_id"] for candidate in candidates})
    for instance_id in instances:
        for arm in EXPECTED_ARMS:
            _require(
                any(
                    candidate["instance_id"] == instance_id
                    and candidate["arm"] == arm
                    for candidate in candidates
                ),
                f"instance {instance_id} is missing arm {arm}",
            )
    return (
        sorted(
            candidates,
            key=lambda row: (
                row["instance_id"],
                row["arm"],
                row["decode_mode"],
                row["sampling_seed"] or -1,
            ),
        ),
        {
            "raw_row_count": len(rows),
            "serial_row_count": len(serial_rows),
            "parallel_copy_rows_verified_and_excluded": parallel_copies,
            "invalid_serial_row_count": invalid_rows,
            "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
            "valid_candidate_count": len(candidates),
            "unique_action_trace_count": len(set(trace_keys)),
        },
    )


def _set_front_flags(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    front_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_instance[candidate["instance_id"]].append(candidate)

    for instance_id in sorted(by_instance):
        instance_candidates = by_instance[instance_id]
        union_groups = _point_groups(instance_candidates)
        union_front_indices = set(
            nondominated_indices([group["objectives"] for group in union_groups])
        )
        candidate_to_union: dict[str, tuple[bool, str | None, int]] = {}
        front_index = 0
        contribution_counts = {"c0": 0, "e1": 0, "shared": 0}
        front_normalized: list[tuple[float, float, float]] = []
        for group_index, group in enumerate(union_groups):
            is_front = group_index in union_front_indices
            point_id = None
            if is_front:
                front_index += 1
                point_id = f"{instance_id}|p{front_index:02d}"
                front_normalized.append(group["normalized_objectives"])
                arms = sorted({member["arm"] for member in group["members"]})
                contribution = arms[0] if len(arms) == 1 else "shared"
                contribution_counts[contribution] += 1
                members = sorted(
                    group["members"], key=lambda member: member["candidate_id"]
                )
                front_rows.append(
                    {
                        "pareto_point_id": point_id,
                        "instance_id": instance_id,
                        "instance_seed": members[0]["instance_seed"],
                        "pressure_type": members[0]["pressure_type"],
                        "cost_profile": members[0]["cost_profile"],
                        "contribution": contribution,
                        "arms": ";".join(arms),
                        "source_count": len(members),
                        "candidate_ids": ";".join(
                            member["candidate_id"] for member in members
                        ),
                        "decode_modes": ";".join(
                            sorted({member["decode_mode"] for member in members})
                        ),
                        "sampling_seeds": ";".join(
                            str(seed)
                            for seed in sorted(
                                {
                                    member["sampling_seed"]
                                    for member in members
                                    if member["sampling_seed"] is not None
                                }
                            )
                        ),
                        "action_trace_sha256": ";".join(
                            member["action_trace_sha256"] for member in members
                        ),
                        **dict(zip(OBJECTIVE_FIELDS, group["objectives"], strict=True)),
                        **dict(
                            zip(
                                NORMALIZED_FIELDS,
                                group["normalized_objectives"],
                                strict=True,
                            )
                        ),
                    }
                )
            for member in group["members"]:
                candidate_to_union[member["candidate_id"]] = (
                    is_front,
                    point_id,
                    len(group["members"]),
                )

        arm_front_ids: set[str] = set()
        arm_front_sizes: dict[str, int] = {}
        arm_unique_sizes: dict[str, int] = {}
        arm_hv: dict[str, float] = {}
        for arm in EXPECTED_ARMS:
            arm_candidates = [
                candidate
                for candidate in instance_candidates
                if candidate["arm"] == arm
            ]
            arm_groups = _point_groups(arm_candidates)
            arm_indices = set(
                nondominated_indices([group["objectives"] for group in arm_groups])
            )
            arm_front_sizes[arm] = len(arm_indices)
            arm_unique_sizes[arm] = len(arm_groups)
            for group_index in arm_indices:
                arm_front_ids.update(
                    member["candidate_id"]
                    for member in arm_groups[group_index]["members"]
                )
            arm_hv[arm] = hypervolume_3d(
                group["normalized_objectives"] for group in arm_groups
            )

        union_hv = hypervolume_3d(
            group["normalized_objectives"] for group in union_groups
        )
        _require(
            union_hv + RELATIVE_TOLERANCE >= max(arm_hv.values()),
            f"union hypervolume invariant failed for {instance_id}",
        )
        hv_delta = arm_hv["e1"] - arm_hv["c0"]
        hv_scale = max(1.0, abs(arm_hv["e1"]), abs(arm_hv["c0"]))
        if abs(hv_delta) <= RELATIVE_TOLERANCE * hv_scale:
            hv_winner = "tie"
        else:
            hv_winner = "e1" if hv_delta > 0.0 else "c0"

        front_spans = []
        for dimension in range(3):
            values = [point[dimension] for point in front_normalized]
            front_spans.append(max(values) - min(values) if values else 0.0)
        first = instance_candidates[0]
        instance_rows.append(
            {
                "instance_id": instance_id,
                "instance_seed": first["instance_seed"],
                "pressure_type": first["pressure_type"],
                "cost_profile": first["cost_profile"],
                "candidate_count": len(instance_candidates),
                "c0_candidate_count": sum(
                    candidate["arm"] == "c0" for candidate in instance_candidates
                ),
                "e1_candidate_count": sum(
                    candidate["arm"] == "e1" for candidate in instance_candidates
                ),
                "unique_objective_count": len(union_groups),
                "c0_unique_objective_count": arm_unique_sizes["c0"],
                "e1_unique_objective_count": arm_unique_sizes["e1"],
                "union_front_size": len(union_front_indices),
                "c0_front_size": arm_front_sizes["c0"],
                "e1_front_size": arm_front_sizes["e1"],
                "c0_exclusive_front_points": contribution_counts["c0"],
                "e1_exclusive_front_points": contribution_counts["e1"],
                "shared_front_points": contribution_counts["shared"],
                "normalized_flow_front_span": front_spans[0],
                "normalized_cost_front_span": front_spans[1],
                "normalized_variance_front_span": front_spans[2],
                "c0_hypervolume": arm_hv["c0"],
                "e1_hypervolume": arm_hv["e1"],
                "union_hypervolume": union_hv,
                "e1_minus_c0_hypervolume": hv_delta,
                "hypervolume_winner": hv_winner,
            }
        )

        for candidate in instance_candidates:
            is_union, point_id, duplicate_count = candidate_to_union[
                candidate["candidate_id"]
            ]
            candidate_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "instance_id": candidate["instance_id"],
                    "instance_seed": candidate["instance_seed"],
                    "pressure_type": candidate["pressure_type"],
                    "cost_profile": candidate["cost_profile"],
                    "arm": candidate["arm"],
                    "algorithm_seed": candidate["algorithm_seed"],
                    "source_mode": candidate["source_mode"],
                    "decode_mode": candidate["decode_mode"],
                    "sampling_seed": candidate["sampling_seed"],
                    "action_trace_sha256": candidate["action_trace_sha256"],
                    **dict(
                        zip(OBJECTIVE_FIELDS, candidate["objectives"], strict=True)
                    ),
                    **dict(
                        zip(
                            NORMALIZED_FIELDS,
                            candidate["normalized_objectives"],
                            strict=True,
                        )
                    ),
                    "is_arm_pareto": candidate["candidate_id"] in arm_front_ids,
                    "is_union_pareto": is_union,
                    "union_pareto_point_id": point_id,
                    "objective_duplicate_count": duplicate_count,
                }
            )
    return candidate_rows, front_rows, instance_rows


def _aggregate_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    scales: Sequence[float],
) -> tuple[list[dict[str, Any]], int]:
    groups: dict[tuple[str, str, int | None], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for candidate in candidates:
        key = (
            candidate["arm"],
            candidate["decode_mode"],
            candidate["sampling_seed"],
        )
        groups[key][candidate["instance_id"]] = candidate
    if not groups:
        raise ValueError("there are no candidate groups to aggregate")
    common_instances = set.intersection(
        *(set(instance_map) for instance_map in groups.values())
    )
    _require(common_instances, "candidate groups have no common instances")

    aggregates: list[dict[str, Any]] = []
    for (arm, decode_mode, sampling_seed), instance_map in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2] or -1,
        ),
    ):
        selected = [instance_map[instance] for instance in sorted(common_instances)]
        objectives = tuple(
            statistics.fmean(candidate["objectives"][dimension] for candidate in selected)
            for dimension in range(3)
        )
        normalized = normalize_objectives(objectives, scales)
        aggregates.append(
            {
                "aggregate_candidate_id": (
                    f"{arm}|{decode_mode}|"
                    f"{sampling_seed if sampling_seed is not None else 'none'}"
                ),
                "arm": arm,
                "algorithm_seed": selected[0]["algorithm_seed"],
                "decode_mode": decode_mode,
                "sampling_seed": sampling_seed,
                "instance_count": len(selected),
                "objectives": objectives,
                "normalized_objectives": normalized,
            }
        )
    front_indices = set(
        nondominated_indices([aggregate["objectives"] for aggregate in aggregates])
    )
    output = []
    for index, aggregate in enumerate(aggregates):
        output.append(
            {
                "aggregate_candidate_id": aggregate["aggregate_candidate_id"],
                "arm": aggregate["arm"],
                "algorithm_seed": aggregate["algorithm_seed"],
                "decode_mode": aggregate["decode_mode"],
                "sampling_seed": aggregate["sampling_seed"],
                "instance_count": aggregate["instance_count"],
                **dict(zip(OBJECTIVE_FIELDS, aggregate["objectives"], strict=True)),
                **dict(
                    zip(
                        NORMALIZED_FIELDS,
                        aggregate["normalized_objectives"],
                        strict=True,
                    )
                ),
                "is_auxiliary_mean_pareto": index in front_indices,
            }
        )
    return output, len(common_instances)


def _summary_statistics(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _build_summary(
    *,
    audit_dir: Path,
    audit_info: Mapping[str, Any],
    scales: Sequence[float],
    ingestion: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    front_rows: Sequence[Mapping[str, Any]],
    instance_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    aggregate_common_instances: int,
) -> dict[str, Any]:
    front_sizes = [int(row["union_front_size"]) for row in instance_rows]
    single_point_count = sum(size == 1 for size in front_sizes)
    contribution_counts = {
        name: sum(row["contribution"] == name for row in front_rows)
        for name in ("c0", "e1", "shared")
    }
    winners = {
        name: sum(row["hypervolume_winner"] == name for row in instance_rows)
        for name in ("e1", "tie", "c0")
    }
    hv = {
        arm: _summary_statistics(
            [float(row[f"{arm}_hypervolume"]) for row in instance_rows]
        )
        for arm in EXPECTED_ARMS
    }
    hv["union"] = _summary_statistics(
        [float(row["union_hypervolume"]) for row in instance_rows]
    )
    hv["e1_minus_c0"] = _summary_statistics(
        [float(row["e1_minus_c0_hypervolume"]) for row in instance_rows]
    )
    arm_counts = {
        arm: sum(row["arm"] == arm for row in candidate_rows)
        for arm in EXPECTED_ARMS
    }
    aggregate_front = [
        row["aggregate_candidate_id"]
        for row in aggregate_rows
        if bool(row["is_auxiliary_mean_pareto"])
    ]
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "study_design": {
            "type": "exploratory_single_algorithm_seed_fixed_validation",
            "algorithm_seed": 11,
            "dataset": "validation",
            "paired_arms": list(EXPECTED_ARMS),
            "sampling_seeds": list(EXPECTED_SAMPLING_SEEDS),
            "inferential_claim": False,
            "true_pareto_claim": False,
        },
        "input": {
            "audit_directory": str(audit_dir.resolve()),
            "audit_json_sha256": _sha256(audit_dir / "audit.json"),
            "instance_metrics_csv_sha256": _sha256(
                audit_dir / "instance_metrics.csv"
            ),
            "audit_protocol_version": EXPECTED_AUDIT_PROTOCOL,
            "result_schema_version": EXPECTED_RESULT_SCHEMA,
            **dict(audit_info),
        },
        "objective_definition": {
            "sense": "minimize",
            "fields": list(OBJECTIVE_FIELDS),
            "relative_dominance_tolerance": RELATIVE_TOLERANCE,
            "normalization": "x / (scale + x)",
            "scales": dict(zip(OBJECTIVE_FIELDS, scales, strict=True)),
            "hypervolume_reference_point": list(REFERENCE_POINT),
        },
        "ingestion": dict(ingestion),
        "candidate_design": {
            "instance_count": len(instance_rows),
            "candidate_count": len(candidate_rows),
            "arm_candidate_counts": arm_counts,
            "serial_modes": list(SERIAL_MODES),
            "aggregate_common_instance_count": aggregate_common_instances,
        },
        "empirical_front": {
            "total_unique_front_point_count": len(front_rows),
            "front_size": _summary_statistics(front_sizes),
            "single_point_instance_count": single_point_count,
            "single_point_instance_fraction": (
                single_point_count / len(instance_rows) if instance_rows else 0.0
            ),
            "point_contribution_counts": contribution_counts,
            "mean_normalized_front_span": {
                field: statistics.fmean(float(row[field]) for row in instance_rows)
                for field in (
                    "normalized_flow_front_span",
                    "normalized_cost_front_span",
                    "normalized_variance_front_span",
                )
            },
        },
        "hypervolume": {**hv, "instance_win_tie_loss": winners},
        "auxiliary_aggregate_mean_front": {
            "candidate_count": len(aggregate_rows),
            "front_size": len(aggregate_front),
            "front_candidate_ids": aggregate_front,
            "interpretation": (
                "Auxiliary system-level diagnostic only; it is not a pooled "
                "instance-level Pareto front."
            ),
        },
        "limitations": [
            "Only one algorithm seed (11) is compared.",
            "The fixed validation split is used; no held-out test claim is made.",
            "The fronts are empirical subsets of observed rollouts, not certified true Pareto fronts.",
            "No inferential significance test is reported.",
            "Historical C0 checkpoints trained under a different promotion protocol are excluded.",
        ],
    }


def _format_number(value: float) -> str:
    return f"{value:.6f}"


def _build_report(summary: Mapping[str, Any]) -> str:
    design = summary["candidate_design"]
    front = summary["empirical_front"]
    hypervolume = summary["hypervolume"]
    aggregate = summary["auxiliary_aggregate_mean_front"]
    ingestion = summary["ingestion"]
    contributions = front["point_contribution_counts"]
    wins = hypervolume["instance_win_tie_loss"]
    front_size = front["front_size"]
    return "\n".join(
        [
            "# C0/E1 empirical Pareto engineering validation",
            "",
            "> Exploratory paired seed-11 validation analysis. These are empirical "
            "rollout fronts, not certified true Pareto fronts, and no multi-seed "
            "or inferential claim is made.",
            "",
            "## Study design",
            "",
            f"- Dataset: fixed validation manifest ({design['instance_count']} instances).",
            f"- Candidates: {design['candidate_count']} feasible unique schedules; "
            f"C0={design['arm_candidate_counts']['c0']}, "
            f"E1={design['arm_candidate_counts']['e1']}.",
            "- Decode coverage per arm: one greedy schedule and three sampled "
            "schedules (100011, 100012, 100013) per instance.",
            f"- Invalid serial rows excluded: {ingestion['invalid_serial_row_count']}; "
            f"parallel audit copies verified and excluded: "
            f"{ingestion['parallel_copy_rows_verified_and_excluded']}.",
            "- Objectives (all minimized): flow-time objective, reconfiguration "
            "cost, and worker-load variance.",
            "",
            "## Per-instance empirical fronts",
            "",
            f"- Mean front size: {_format_number(front_size['mean'])}; median: "
            f"{_format_number(front_size['median'])}; range: "
            f"{int(front_size['minimum'])}-{int(front_size['maximum'])}.",
            f"- Single-point fronts: {front['single_point_instance_count']}/"
            f"{design['instance_count']} "
            f"({100.0 * front['single_point_instance_fraction']:.1f}%).",
            f"- Unique front-point contributions: C0-only={contributions['c0']}, "
            f"E1-only={contributions['e1']}, shared={contributions['shared']}.",
            "",
            "The instance-level results are the primary evidence. A front is "
            "constructed only among schedules for the same scheduling instance, so "
            "difficulty differences across instances cannot create false dominance.",
            "",
            "## Normalized hypervolume",
            "",
            "| Metric | C0 | E1 | Union |",
            "|---|---:|---:|---:|",
            f"| Mean | {_format_number(hypervolume['c0']['mean'])} | "
            f"{_format_number(hypervolume['e1']['mean'])} | "
            f"{_format_number(hypervolume['union']['mean'])} |",
            f"| Median | {_format_number(hypervolume['c0']['median'])} | "
            f"{_format_number(hypervolume['e1']['median'])} | "
            f"{_format_number(hypervolume['union']['median'])} |",
            "",
            f"Across the {design['instance_count']} paired instances, E1 has larger "
            f"HV on {wins['e1']}, ties on {wins['tie']}, and has smaller HV on "
            f"{wins['c0']}. Mean E1-C0 HV is "
            f"{_format_number(hypervolume['e1_minus_c0']['mean'])}.",
            "",
            "HV uses the canonical bounded transform `x / (scale + x)` with "
            "flow/cost/variance scales 1200/1000/50 and reference point (1,1,1).",
            "",
            "## Auxiliary aggregate-mean diagnostic",
            "",
            f"The mean-objective diagnostic contains {aggregate['candidate_count']} "
            f"candidates and {aggregate['front_size']} nondominated point(s): "
            f"{', '.join(aggregate['front_candidate_ids'])}.",
            "",
            "This auxiliary front summarizes whole-policy mean behavior only. It "
            "must not be interpreted as the Pareto front of any individual scheduling "
            "instance. In particular, a single aggregate point can coexist with "
            "multiple trade-off solutions on individual instances.",
            "",
            "## Limitations",
            "",
            *[f"- {limitation}" for limitation in summary["limitations"]],
            "",
            "## Reproducibility",
            "",
            f"- Audit protocol: `{summary['input']['audit_protocol_version']}`.",
            f"- Evaluation schema: `{summary['input']['result_schema_version']}`.",
            f"- Dataset manifest SHA-256: "
            f"`{summary['input']['dataset_manifest_sha256']}`.",
            f"- Quality metric SHA-256: "
            f"`{summary['input']['quality_metric_sha256']}`.",
            "",
        ]
    )


def _render_plots(
    candidate_rows: Sequence[Mapping[str, Any]],
    instance_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    matplotlib_config = output_dir / ".matplotlib-cache"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config.resolve()))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"c0": "#2F6B9A", "e1": "#D97706"}
    markers = {"c0": "o", "e1": "^"}
    neutral = "#5B6470"
    grid = "#DDE2E7"
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#69727D",
            "axes.labelcolor": "#252B33",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )

    labels = {
        NORMALIZED_FIELDS[0]: "Normalized flow-time objective",
        NORMALIZED_FIELDS[1]: "Normalized reconfiguration cost",
        NORMALIZED_FIELDS[2]: "Normalized worker-load variance",
    }
    ranges: dict[str, tuple[float, float]] = {}
    for field in NORMALIZED_FIELDS:
        values = [float(row[field]) for row in candidate_rows]
        lower, upper = min(values), max(values)
        padding = max(0.01, 0.06 * (upper - lower))
        ranges[field] = (max(0.0, lower - padding), min(1.0, upper + padding))

    pairs = (
        (NORMALIZED_FIELDS[0], NORMALIZED_FIELDS[1]),
        (NORMALIZED_FIELDS[0], NORMALIZED_FIELDS[2]),
        (NORMALIZED_FIELDS[1], NORMALIZED_FIELDS[2]),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))
    for axis, (x_field, y_field) in zip(axes, pairs, strict=True):
        for arm in EXPECTED_ARMS:
            dominated = [
                row
                for row in candidate_rows
                if row["arm"] == arm and not bool(row["is_union_pareto"])
            ]
            front = [
                row
                for row in candidate_rows
                if row["arm"] == arm and bool(row["is_union_pareto"])
            ]
            axis.scatter(
                [float(row[x_field]) for row in dominated],
                [float(row[y_field]) for row in dominated],
                s=24,
                marker=markers[arm],
                facecolors="none",
                edgecolors=colors[arm],
                linewidths=0.8,
                alpha=0.20,
                zorder=1,
            )
            greedy_front = [row for row in front if row["decode_mode"] == "greedy"]
            sampled_front = [
                row for row in front if row["decode_mode"] == "sampled"
            ]
            axis.scatter(
                [float(row[x_field]) for row in greedy_front],
                [float(row[y_field]) for row in greedy_front],
                s=48,
                marker=markers[arm],
                facecolors=colors[arm],
                edgecolors="#20262D",
                linewidths=0.8,
                alpha=0.92,
                zorder=3,
            )
            axis.scatter(
                [float(row[x_field]) for row in sampled_front],
                [float(row[y_field]) for row in sampled_front],
                s=48,
                marker=markers[arm],
                facecolors="white",
                edgecolors=colors[arm],
                linewidths=1.4,
                alpha=0.95,
                zorder=3,
            )
        axis.set_xlabel(labels[x_field])
        axis.set_ylabel(labels[y_field])
        axis.set_xlim(*ranges[x_field])
        axis.set_ylim(*ranges[y_field])
        axis.grid(True, color=grid, linewidth=0.7, alpha=0.75, zorder=0)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[arm],
            color="none",
            markerfacecolor=colors[arm],
            markeredgecolor="#20262D",
            markersize=7,
            label=arm.upper(),
        )
        for arm in EXPECTED_ARMS
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=neutral,
                markeredgecolor="#20262D",
                markersize=6,
                label="Greedy Pareto candidate",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="white",
                markeredgecolor=neutral,
                markersize=6,
                label="Sampled Pareto candidate",
            ),
        ]
    )
    figure.suptitle("Per-instance empirical Pareto candidates", y=0.99)
    figure.text(
        0.5,
        0.935,
        (
            f"Validation | {len(instance_rows)} instances | "
            f"{len(candidate_rows)} schedules | canonical bounded normalization"
        ),
        ha="center",
        va="top",
        color=neutral,
        fontsize=10,
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.14, top=0.78, wspace=0.30)
    figure.savefig(output_dir / "pareto_projections.pdf", bbox_inches="tight")
    figure.savefig(
        output_dir / "pareto_projections.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)

    c0_values = [float(row["c0_hypervolume"]) for row in instance_rows]
    e1_values = [float(row["e1_hypervolume"]) for row in instance_rows]
    lower = min(c0_values + e1_values)
    upper = max(c0_values + e1_values)
    padding = max(0.002, 0.08 * (upper - lower))
    lower = max(0.0, lower - padding)
    upper = min(1.0, upper + padding)
    hv_figure, axis = plt.subplots(figsize=(6.2, 5.3))
    axis.plot(
        [lower, upper],
        [lower, upper],
        color=neutral,
        linewidth=1.1,
        linestyle="--",
        label="Equal hypervolume",
        zorder=1,
    )
    axis.scatter(
        c0_values,
        e1_values,
        s=56,
        facecolors="white",
        edgecolors=colors["c0"],
        linewidths=1.5,
        alpha=0.95,
        zorder=2,
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("C0 normalized hypervolume")
    axis.set_ylabel("E1 normalized hypervolume")
    axis.grid(True, color=grid, linewidth=0.7, alpha=0.75, zorder=0)
    axis.legend(frameon=False, loc="lower right")
    hv_figure.suptitle("Per-instance C0/E1 hypervolume comparison", y=0.985)
    hv_figure.text(
        0.5,
        0.925,
        (
            f"Validation | n={len(instance_rows)} paired instances | "
            "points above the diagonal favour E1"
        ),
        ha="center",
        va="top",
        color=neutral,
        fontsize=10,
    )
    hv_figure.subplots_adjust(left=0.16, right=0.96, bottom=0.13, top=0.86)
    hv_figure.savefig(output_dir / "hypervolume_comparison.pdf", bbox_inches="tight")
    hv_figure.savefig(
        output_dir / "hypervolume_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(hv_figure)


def analyze_audit(
    audit_dir: str | Path,
    output_dir: str | Path,
    *,
    render_plots: bool = True,
) -> dict[str, Any]:
    """Run the deterministic empirical Pareto analysis and write its artifacts."""

    audit_path = Path(audit_dir)
    output_path = Path(output_dir)
    audit_json_path = audit_path / "audit.json"
    metrics_path = audit_path / "instance_metrics.csv"
    if not audit_json_path.is_file():
        raise FileNotFoundError(audit_json_path)
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    audit = _read_json(audit_json_path)
    audit_info, scales, algorithm_seeds = _validate_audit(audit)
    raw_rows = _read_csv(metrics_path)
    candidates, ingestion = _prepare_candidates(
        raw_rows, scales=scales, algorithm_seeds=algorithm_seeds
    )
    candidate_rows, front_rows, instance_rows = _set_front_flags(candidates)
    aggregate_rows, aggregate_common_instances = _aggregate_candidates(
        candidates, scales=scales
    )
    summary = _build_summary(
        audit_dir=audit_path,
        audit_info=audit_info,
        scales=scales,
        ingestion=ingestion,
        candidate_rows=candidate_rows,
        front_rows=front_rows,
        instance_rows=instance_rows,
        aggregate_rows=aggregate_rows,
        aggregate_common_instances=aggregate_common_instances,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    write_csv(output_path / "candidates.csv", list(candidate_rows))
    write_csv(output_path / "pareto_front.csv", list(front_rows))
    write_csv(output_path / "instance_summary.csv", list(instance_rows))
    write_csv(output_path / "aggregate_candidates.csv", list(aggregate_rows))
    write_json(output_path / "summary.json", summary)
    (output_path / "report.md").write_text(
        _build_report(summary), encoding="utf-8"
    )
    if render_plots:
        _render_plots(candidate_rows, instance_rows, output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze_audit(args.audit_dir, args.output_dir)
    print(
        json.dumps(
            {
                "output_directory": str(args.output_dir.resolve()),
                "instance_count": summary["candidate_design"]["instance_count"],
                "candidate_count": summary["candidate_design"]["candidate_count"],
                "front_point_count": summary["empirical_front"][
                    "total_unique_front_point_count"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
