"""Actual labelled-worker load reconstruction, rescoring and empirical dominance.

The historical inputs remain immutable. OWA needs every worker's actual DIS/INS
minutes; neither a variance nor a scalar workload value identifies that vector.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from environment.workload import workload_config, workload_value
from environment.types import metrics_flow_objective
from result.metrics import evaluation_quality_metric, quality_metric_sha256


def _flag(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1"}


def _loads(value: Any) -> dict[str, float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping) or not value:
        raise ValueError("worker_loads must be a nonempty mapping of worker IDs to actual minutes")
    result = {str(key): float(load) for key, load in value.items()}
    if any(not math.isfinite(load) or load < 0 for load in result.values()):
        raise ValueError("actual worker loads must be finite and nonnegative")
    return result


def actual_worker_loads(row: Mapping[str, Any], *, worker_ids: Sequence[str] | None = None,
                        reconfigurations: Sequence[Mapping[str, Any]] | None = None,
                        logs_complete: bool = False) -> dict[str, float]:
    """Get loads, or reconstruct only from explicitly complete terminal logs."""
    actual = _loads(row.get("worker_loads"))
    if actual is not None:
        if worker_ids is not None and set(actual) != {str(worker) for worker in worker_ids}:
            raise ValueError("actual loads omit or add labelled workers")
        return actual
    if reconfigurations is None or worker_ids is None or not logs_complete:
        raise ValueError("actual worker loads unavailable; complete reconfiguration logs and worker IDs are required")
    if not _flag(row.get("terminated")) or _flag(row.get("sampling_truncated", row.get("truncated", False))):
        raise ValueError("historical load reconstruction requires a complete terminal trajectory")
    roster = tuple(str(worker) for worker in worker_ids)
    if not roster or len(set(roster)) != len(roster):
        raise ValueError("worker roster must be nonempty and unique")
    actual = dict.fromkeys(roster, 0.0)
    seen = set()
    for segment in reconfigurations:
        if str(segment.get("stage")) not in {"DIS", "INS"}:
            raise ValueError("reconfiguration labor logs require DIS or INS stages")
        worker = str(segment["worker_id"])
        if worker not in actual:
            raise ValueError("reconfiguration log contains an unknown worker")
        start, end = float(segment["start"]), float(segment["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("invalid actual labor interval")
        if row.get("makespan") not in {None, ""} and end > float(row["makespan"]) + 1e-9:
            raise ValueError("labor interval extends beyond the terminal trajectory")
        key = (worker, segment.get("machine_id"), segment.get("operation_id"), segment["stage"], start, end)
        if key in seen:
            raise ValueError("duplicate labor interval")
        seen.add(key)
        actual[worker] += end - start
    if row.get("completed_reconfigurations") not in {None, ""}:
        if sum(str(segment["stage"]) == "INS" for segment in reconfigurations) != int(row["completed_reconfigurations"]):
            raise ValueError("reconfiguration logs do not cover all completed installations")
    if row.get("worker_load_variance") not in {None, ""}:
        variance = workload_value(list(actual.values()), {"workload_objective": {"mode": "variance"}})
        if not math.isclose(variance, float(row["worker_load_variance"]), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("reconstructed loads disagree with the historical actual variance")
    return actual


def workload_identity(config: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    definition = workload_config(config)
    return {f"{prefix}workload_mode": definition["mode"], f"{prefix}workload_q": definition["q"],
            f"{prefix}workload_rule": definition["weights"], f"{prefix}workload_order": definition["order"],
            f"{prefix}workload_version": definition["version"]}


def rescore_workload_row(row: Mapping[str, Any], evaluation_config: Mapping[str, Any],
                         **load_sources: Any) -> dict[str, Any]:
    """Score an executed trajectory under an explicit, frozen evaluation goal."""
    from environment import bounded_quality_score, terminal_quality_score
    from result.metrics import result_schema_version
    from environment.workload import objective_names
    loads = actual_worker_loads(row, **load_sources)
    value = workload_value(list(loads.values()), evaluation_config)
    variance = workload_value(list(loads.values()), {"workload_objective": {"mode": "variance"}})
    if row.get("worker_load_variance") not in {None, ""} and not math.isclose(float(row["worker_load_variance"]), variance, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("actual worker loads disagree with the recorded variance diagnostic")
    metric = evaluation_quality_metric(evaluation_config)
    weights = [float(row.get(f"preference_{name}", row.get(f"w_{name}", 0))) for name in ("flow", "cost")]
    weights.append(float(row.get("preference_workload", row.get("preference_variance", row.get("w_workload", row.get("w_variance", 0))))))
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise ValueError("workload rescoring requires the executed preference coordinates")
    result = dict(row)
    for name in ("workload_objective_value", "preference_quality_score", "actual_preference_quality_score", "quality_score", "quality_metric_sha256", "result_schema_version", "heuristic_quality_score"):
        if name in row:
            result.setdefault(f"training_{name}", row[name])
    result.update({"worker_loads": json.dumps(loads, sort_keys=True), "worker_load_total": math.fsum(loads.values()),
                   "worker_load_max": max(loads.values()), "workload_objective_value": value,
                   "worker_load_variance": variance,
                   "result_schema_version": result_schema_version(evaluation_config),
                   "evaluation_goal": "explicit_frozen_workload", **workload_identity(evaluation_config, prefix="evaluation_"),
                   "evaluation_flow_mode": metric.get("flow_mode", "raw_v1"),
                   "evaluation_normalization_manifest_sha256": evaluation_config["objective_scalarizer"].get("normalization_manifest_sha256"),
                   "quality_metric_sha256": quality_metric_sha256(metric), "quality_metric_version": metric["version"],
                   "quality_score": bounded_quality_score(metrics_flow_objective(dict(row), dict(evaluation_config)), float(row["reconfiguration_cost"]), value, metric),
                   "preference_quality_score": terminal_quality_score(metrics_flow_objective(dict(row), dict(evaluation_config)),
                        float(row["reconfiguration_cost"]), value, dict(evaluation_config), preference=weights,
                        terminal_failure=_flag(row.get("task_failed"))),
                   "actual_preference_quality_score": bounded_quality_score(metrics_flow_objective(dict(row), dict(evaluation_config)),
                        float(row["reconfiguration_cost"]), value, dict(evaluation_config), preference=weights),
                   **{f"evaluation_objective_scale_{name}": evaluation_config["objective_scalarizer"]["scales"][name]
                      for name in objective_names(evaluation_config)}})
    heuristic_loads = _loads(row.get("heuristic_worker_loads"))
    result["heuristic_quality_score"] = None
    result["heuristic_reward_quality_score"] = None
    result["workload_objective_value_heuristic_gap_percent"] = None
    if heuristic_loads is not None and _flag(row.get("heuristic_comparison_valid")):
        from result.metrics import relative_gap_percent
        heuristic_value = workload_value(list(heuristic_loads.values()), evaluation_config)
        result["heuristic_workload_objective_value"] = heuristic_value
        from environment.types import FLOW_EXCESS, flow_mode
        heuristic_flow = float(row["heuristic_flow_time"])
        if flow_mode(dict(evaluation_config)) == FLOW_EXCESS:
            heuristic_flow = max(0.0, heuristic_flow - float(row["flow_processing_lower_bound"]))
        result["heuristic_quality_score"] = bounded_quality_score(heuristic_flow,
            float(row["heuristic_reconfiguration_cost"]), heuristic_value, metric)
        result["workload_objective_value_heuristic_gap_percent"] = relative_gap_percent(value, heuristic_value)
    return result


def resolve_workload_evaluation_config(config: Mapping[str, Any], *, evaluation_config: Mapping[str, Any] | None = None,
                                       own_goal: bool = False, flow_mode: str | None = None) -> dict[str, Any]:
    """Choose own training goal or the canonical frozen q=.5 comparison goal."""
    from configs import load_config
    from environment.types import FLOW_RAW, FLOW_EXCESS, flow_mode as config_flow_mode
    from copy import deepcopy
    if evaluation_config is not None:
        if flow_mode is not None and config_flow_mode(dict(evaluation_config)) != flow_mode:
            raise ValueError("explicit evaluation config Flow mode differs from the requested scoring mode")
        return deepcopy(dict(evaluation_config))
    if own_goal:
        if flow_mode is not None and config_flow_mode(dict(config)) != flow_mode:
            raise ValueError("own training goal cannot override Flow mode; provide a matching frozen evaluation config")
        return deepcopy(dict(config))
    mode = config_flow_mode(dict(config)) if flow_mode is None else flow_mode
    if mode not in {FLOW_RAW, FLOW_EXCESS}:
        raise ValueError("unknown evaluation Flow mode")
    path = "configs/controls/raw.json" if mode == FLOW_RAW else "configs/default.json"
    return load_config(path)


def historical_load_sources(directory, config, dataset_name, rows):
    """Read verified rosters and candidate-specific historical logs without rewriting them."""
    import csv
    from pathlib import Path
    from data import load_dataset_split
    from result.provenance import dataset_manifest_snapshot
    dataset = load_dataset_split(config, dataset_name)
    manifest_hash = dataset_manifest_snapshot(dataset.manifest_path)["sha256"]
    if any(row.get("dataset_manifest_sha256", manifest_hash) != manifest_hash for row in rows):
        raise ValueError("historical candidate manifest differs from the verified instance dataset")
    rosters = {record.instance.instance_id: tuple(worker.id for worker in record.instance.workers) for record in dataset}
    path = Path(directory) / "reconfigurations.csv"
    logs = defaultdict(list)
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for segment in csv.DictReader(stream):
                logs[str(segment["instance_id"])].append(segment)
    result = {}
    for row in rows:
        iid = str(row["instance_id"])
        if iid not in rosters:
            raise ValueError("historical candidate references an unverified instance")
        candidate = str(row.get("candidate_id", ""))
        segments = logs.get(iid, [])
        if candidate and any(segment.get("candidate_id") for segment in segments):
            segments = [segment for segment in segments if segment.get("candidate_id") == candidate]
        else:
            if any(segment.get("endpoint_preference_key") or segment.get("preference_key") for segment in segments):
                segments = [segment for segment in segments if segment.get("endpoint_preference_key", segment.get("preference_key")) == row.get("preference_key")]
            if any(segment.get("sampling_repeat") not in {None, ""} for segment in segments):
                segments = [segment for segment in segments if int(segment["sampling_repeat"]) == int(row.get("sampling_repeat") or 0)]
            # Multiple preference cells require a preference label on the logs.
            elif len({str(other.get("preference_key", "")) for other in rows if str(other["instance_id"]) == iid}) > 1:
                segments = []
        if not path.is_file() or (not segments and int(float(row.get("completed_reconfigurations") or 0)) > 0):
            raise ValueError("complete candidate-specific reconfiguration logs are unavailable")
        result[(iid, candidate, str(row.get("sampling_repeat", 0)))] = {
            "worker_ids": rosters[iid], "reconfigurations": segments, "logs_complete": True}
    return result


def labelled_workload_dominance(rows: Sequence[Mapping[str, Any]], *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Compare flow, cost and every labelled worker for the same physical instance."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("dominance tolerance must be finite and nonnegative")
    groups = defaultdict(list)
    missing = []
    unsafe = []
    for index, row in enumerate(rows):
        candidate = str(row.get("candidate_id", row.get("action_trace_sha256", index)))
        policy = str(row.get("method", row.get("policy", row.get("arm", "unknown"))))
        identity = f"{row.get('instance_id', '')}:{policy}:{row.get('algorithm_seed', '')}:{candidate}:repeat_{row.get('sampling_repeat', 0)}"
        safe = (_flag(row.get("task_succeeded")) and _flag(row.get("terminated")) and not _flag(row.get("task_failed"))
                and not _flag(row.get("sampling_truncated", row.get("truncated", False)))
                and int(float(row.get("schedule_violation_count", 0))) == 0
                and row.get("fatigue_mode", "full") == "full"
                and float(row.get("fatigue_monitor_peak", row.get("maximum_worker_fatigue", math.inf)))
                <= float(row.get("safe_fatigue_limit", -math.inf)) + tolerance)
        if not safe:
            unsafe.append(identity)
            continue
        try:
            loads = actual_worker_loads(row)
        except ValueError:
            missing.append(identity)
            continue
        physical = tuple(str(row.get(name, "")) for name in ("dataset_manifest_sha256", "instance_id", "environment_config_sha256", "distribution_contract_sha256"))
        if not physical[0] or not physical[1] or not physical[2]:
            raise ValueError("labelled dominance requires verified dataset, instance and physical environment identity")
        vector = (float(row["flow_time_objective"]), float(row["reconfiguration_cost"]), *[loads[worker] for worker in sorted(loads)])
        if any(not math.isfinite(value) or value < 0 for value in vector):
            raise ValueError("dominance objectives must be finite and nonnegative")
        groups[(*physical, tuple(sorted(loads)))].append((identity, policy, vector))
    within, across = [], []
    for group in groups.values():
        for better_id, better_policy, better in group:
            for worse_id, worse_policy, worse in group:
                if better_id == worse_id:
                    continue
                slacks = [tolerance * max(1.0, abs(a), abs(b)) for a, b in zip(better, worse)]
                if all(a <= b + slack for a, b, slack in zip(better, worse, slacks)) and any(a < b - slack for a, b, slack in zip(better, worse, slacks)):
                    witness = {"dominator_id": better_id, "dominated_id": worse_id,
                               "dominator_policy": better_policy, "dominated_policy": worse_policy}
                    (within if better_policy == worse_policy else across).append(witness)
    evaluated = sum(len(group) for group in groups.values())
    def pool_summary(witnesses):
        dominated = len({pair["dominated_id"] for pair in witnesses})
        return {"dominated_candidate_count": dominated, "eligible_candidate_count": evaluated,
                "dominated_candidate_ratio": dominated / evaluated if evaluated else None,
                "pair_count": len(witnesses), "witnesses": witnesses}
    policy_counts = defaultdict(int)
    for group in groups.values():
        for _, policy, _ in group:
            policy_counts[policy] += 1
    per_policy = {}
    for policy, count in sorted(policy_counts.items()):
        own = {pair["dominated_id"] for pair in within if pair["dominated_policy"] == policy}
        pooled = {pair["dominated_id"] for pair in within + across if pair["dominated_policy"] == policy}
        per_policy[policy] = {"eligible_candidate_count": count,
                              "within_policy_dominated_count": len(own), "within_policy_dominated_ratio": len(own) / count,
                              "pooled_dominated_count": len(pooled), "pooled_dominated_ratio": len(pooled) / count}
    return {"diagnostic": "labelled_worker_load_dominance_v1", "candidate_count": len(rows),
            "task_failed_count": sum(_flag(row.get("task_failed")) for row in rows),
            "completion_rate": (evaluated + len(missing)) / len(rows) if rows else None,
            "successful_legal_with_loads_count": evaluated, "excluded_failed_or_unsafe_count": len(unsafe),
            "successful_legal_missing_loads_count": len(missing), "missing_load_candidate_ids": missing,
            "load_coverage": evaluated / (evaluated + len(missing)) if evaluated + len(missing) else 0.0,
            "physical_group_count": len(groups),
            "within_policy": pool_summary(within), "across_policy": pool_summary(across),
            "pooled": pool_summary(within + across), "per_policy": per_policy,
            "interpretation": "Counts describe this empirical candidate pool; zero observed pairs is not a global dominance guarantee."}


def analyze_workload_pool(rows: Sequence[Mapping[str, Any]], *, evaluation_config: Mapping[str, Any] | None = None,
                          training_configs: Mapping[str, Mapping[str, Any]] | None = None,
                          worker_rosters: Mapping[str, Sequence[str]] | None = None,
                          complete_reconfiguration_logs: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]] | None = None,
                          require_matched_sampling_budget: bool = True) -> dict[str, Any]:
    """Compare a candidate pool at frozen raw-Flow q=.5, without changing source artifacts.

    Log keys are (method, instance_id, candidate_id). Supplying a log entry is an
    explicit assertion that it covers the complete recorded terminal trajectory.
    Training configurations are validated individually before common rescoring.
    """
    from configs import load_config
    from configs.formal_preferences import objective_scales
    from analysis.pareto_analysis import valid_candidate, _validate_row_protocol, normalize_objectives, hypervolume_3d, nondominated_indices
    scoring = (load_config("configs/controls/raw.json") if evaluation_config is None
               else dict(evaluation_config))
    metric = evaluation_quality_metric(scoring)
    grouped = defaultdict(list)
    annotated = []
    budget_cells = defaultdict(set)
    methods_by_instance = defaultdict(set)
    sampled_seeds = defaultdict(set)
    physical_by_unit = {}
    physical_fields = ("dataset_manifest_sha256", "generator_config_sha256", "environment_config_sha256", "distribution_contract_sha256")
    for original in rows:
        row = dict(original)
        method = str(row.get("method", row.get("arm", "unknown")))
        row["method"] = method
        if training_configs is not None:
            if method not in training_configs:
                raise ValueError("candidate method has no source training configuration")
            _validate_row_protocol(row, training_configs[method])
        iid, seed = str(row["instance_id"]), int(row["algorithm_seed"])
        unit = (seed, iid)
        if any(row.get(name) in {None, "", "unknown"} for name in physical_fields):
            raise ValueError("workload pool requires a complete verified physical identity")
        if worker_rosters is not None:
            if iid not in worker_rosters:
                raise ValueError("workload pool physical identity has no verified worker roster")
            roster = tuple(sorted(str(worker) for worker in worker_rosters[iid]))
            if not roster or len(set(roster)) != len(roster):
                raise ValueError("workload pool physical identity requires a unique nonempty worker roster")
            if row.get("worker_loads") is not None and row.get("worker_loads") != "":
                actual_worker_loads(row, worker_ids=roster)
        else:
            roster = tuple(sorted(actual_worker_loads(row)))
        physical = (*[str(row[name]) for name in physical_fields], row.get("fatigue_mode", "full"), roster)
        if unit in physical_by_unit and physical != physical_by_unit[unit]:
            raise ValueError("candidate methods have unmatched physical identity or labelled worker roster")
        physical_by_unit[unit] = physical
        cell = (str(row["preference_key"]), int(row.get("sampling_repeat") or 0))
        if cell in budget_cells[(method, *unit)]:
            raise ValueError("duplicate candidate sampling cell")
        budget_cells[(method, *unit)].add(cell)
        methods_by_instance[unit].add(method)
        if row.get("sampling_seed") not in {None, ""}:
            sampled_seeds[(method, *unit)].add(int(row["sampling_seed"]))
        if _flag(row.get("sampling_truncated", row.get("truncated", False))):
            raise ValueError("externally truncated candidates cannot enter workload Pareto/HV")
        if valid_candidate(row):
            load_sources = {}
            if worker_rosters is not None:
                load_sources["worker_ids"] = worker_rosters[iid]
            if complete_reconfiguration_logs is not None:
                key = (method, iid, str(row["candidate_id"]))
                if key in complete_reconfiguration_logs:
                    load_sources.update(reconfigurations=complete_reconfiguration_logs[key], logs_complete=True)
            row = rescore_workload_row(row, scoring, **load_sources)
        else:
            # Failures remain counted; their unavailable historical loads are not inferred.
            row["evaluation_candidate_feasible"] = False
        grouped[(method, *unit)].append(row)
        annotated.append(row)
    methods = {str(row["method"]) for row in annotated}
    if require_matched_sampling_budget:
        for unit, present in methods_by_instance.items():
            if present != methods:
                raise ValueError("methods have unmatched instance/seed coverage")
            cells = [budget_cells[(method, *unit)] for method in sorted(methods)]
            if any(current != cells[0] for current in cells[1:]):
                raise ValueError("methods use unmatched preference/repeat sampling budgets")
            rng = [sampled_seeds[(method, *unit)] for method in sorted(methods)]
            if any(current != rng[0] for current in rng[1:]):
                raise ValueError("sampled methods use unmatched evaluation seeds")
    summaries = []
    scales = objective_scales(scoring)
    for (method, seed, iid), group in sorted(grouped.items()):
        safe = [row for row in group if valid_candidate(row)]
        points = [normalize_objectives((metrics_flow_objective(row, scoring), float(row["reconfiguration_cost"]),
                    float(row["workload_objective_value"])), scales) for row in safe]
        front = set(nondominated_indices(points))
        for index, row in enumerate(safe):
            row["is_workload_pareto"] = index in front
        summaries.append({"method": method, "algorithm_seed": seed, "instance_id": iid,
            "candidate_count": len(group), "successful_legal_count": len(safe), "completion_rate": len(safe) / len(group),
            "hypervolume": hypervolume_3d(points), "front_size": len(front)})
    return {"protocol": "common_workload_pool_v1", "evaluation_goal": "unified_raw_flow_q05" if evaluation_config is None else "explicit_frozen_goal",
            "evaluation_quality_metric": metric, "evaluation_quality_metric_sha256": quality_metric_sha256(metric),
            "matched_sampling_budget_required": require_matched_sampling_budget,
            "candidates": annotated, "instances": summaries,
            "labelled_worker_dominance": labelled_workload_dominance(annotated)}
