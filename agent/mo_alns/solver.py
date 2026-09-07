"""Environment-decoded multi-objective adaptive large-neighbourhood search."""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from agent.baselines import HeuristicPolicy
from data.models import AssemblyInstance, OperationSpec
from environment import (
    AssemblySchedulingEnv,
    CANONICAL_PREFERENCE,
    DecisionType,
    PreferenceInput,
    PreferenceVector,
    normalize_preference,
    simplex_lattice,
)
from environment.types import OperationState, ReconfigurationStage
from utils import action_trace_sha256

from .archive import ParetoArchive, augmented_tchebycheff, candidate_better, normalized_objectives
from .types import (
    CandidateEvaluation,
    GridSearchResult,
    MOALNSSolution,
    SearchResult,
    metrics_objectives,
    preference_key,
)


DEFAULT_SETTINGS: dict[str, Any] = {
    "max_evaluations_per_preference": 300,
    "max_proposals_multiplier": 10,
    "destroy_fraction": [0.10, 0.30],
    "minimum_removed_operations": 2,
    "regret_position_candidates": 6,
    "regret_resource_variants": 3,
    "operator_segment_length": 25,
    "operator_reaction": 0.20,
    "operator_minimum_probability": 0.05,
    "operator_scores": {
        "archive_addition": 8.0,
        "scalar_best": 5.0,
        "current_improvement": 3.0,
        "accepted": 1.0,
    },
    "temperature_calibration_samples": 20,
    "temperature_target_acceptance": 0.50,
    "temperature_fallback": 0.01,
    "temperature_final_ratio": 0.01,
    "stagnation_evaluations": 50,
    "reheat_fraction": 0.25,
    "maximum_reheats": 2,
    "matching_repair": True,
}

DESTROY_OPERATORS = (
    "random",
    "worst_flow_order",
    "high_reconfiguration_segment",
    "fatigue_critical_worker",
    "load_imbalance_worker",
    "same_wave_related",
)
REPAIR_OPERATORS = (
    "regret_2",
    "regret_3",
    "earliest_finish",
    "minimum_reconfiguration_cost",
    "lowest_predicted_fatigue",
    "minimum_load_variance",
)
INITIAL_RULES = (
    "existing_heuristic",
    "earliest_finish",
    "short_flow",
    "configuration_reuse",
    "minimum_reconfiguration_cost",
    "lowest_predicted_fatigue",
    "minimum_load_variance",
    "same_wave_randomized",
)


def _merge_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("mo_alns", {})
    if not isinstance(raw, Mapping):
        raise TypeError("config.mo_alns must be an object")
    result = dict(DEFAULT_SETTINGS)
    for key, value in raw.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = {**dict(result[key]), **dict(value)}
        else:
            result[key] = value
    evaluations = int(result["max_evaluations_per_preference"])
    if evaluations < len(INITIAL_RULES):
        raise ValueError("max_evaluations_per_preference must cover all initial rules")
    fraction = tuple(float(value) for value in result["destroy_fraction"])
    if len(fraction) != 2 or not 0 < fraction[0] <= fraction[1] <= 1:
        raise ValueError("mo_alns.destroy_fraction must be an increasing range in (0, 1]")
    if int(result["regret_position_candidates"]) < 1 or int(result["regret_resource_variants"]) < 1:
        raise ValueError("regret candidate limits must be positive")
    if not 0 < float(result["operator_reaction"]) <= 1:
        raise ValueError("operator_reaction must be in (0, 1]")
    if not 0.0 <= float(result["operator_minimum_probability"]) < 1.0 / len(DESTROY_OPERATORS):
        raise ValueError("operator_minimum_probability must be in [0, 1 / operator_count)")
    if not 0 < float(result["temperature_target_acceptance"]) < 1:
        raise ValueError("temperature_target_acceptance must be in (0, 1)")
    return result


def derive_mo_alns_seed(
    algorithm_seed: int,
    instance_id: str,
    preference: PreferenceVector,
    dataset_name: str = "",
) -> int:
    """Use a seed independent of scheduling order and process topology."""

    payload = (
        f"mo_alns_solver_v1\0{int(algorithm_seed)}\0{dataset_name}\0{instance_id}\0"
        f"{preference_key(preference)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _pending_reconfiguration(env: AssemblySchedulingEnv, machine_index: int):
    machine_id = env.machines[machine_index].spec.id
    candidates = [
        value
        for value in env.reconfigurations.values()
        if value.machine_id == machine_id
        and value.stage
        in {ReconfigurationStage.WAIT_DIS, ReconfigurationStage.WAIT_INS}
    ]
    return min(candidates, key=lambda value: value.id) if candidates else None


def _rank_index(values: Sequence[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(values)}


def _choose_production_action(env: AssemblySchedulingEnv, solution: MOALNSSolution, mask: np.ndarray) -> int:
    terminal = env.production_defer_action
    legal_pairs = [int(value) for value in np.flatnonzero(~mask) if int(value) != terminal]
    if not legal_pairs:
        return terminal
    operation_positions = _rank_index(solution.operation_order)
    ready = [
        (operation_positions[operation.spec.id], index, operation)
        for index, operation in enumerate(env.operations)
        if operation.state == OperationState.READY
    ]
    if ready and not bool(mask[terminal]):
        _, primary_index, primary_operation = min(ready)
        preferred_machine = solution.machine_rankings[primary_operation.spec.id][0]
        preferred_legal = any(
            env.decode_production_action(action)
            == (primary_index, env.instance.machine_index[preferred_machine])
            for action in legal_pairs
        )
        if solution.production_wait[primary_operation.spec.id] and not preferred_legal:
            return terminal
    scored: list[tuple[int, int, int]] = []
    for action in legal_pairs:
        operation_index, machine_index = env.decode_production_action(action)
        operation_id = env.operations[operation_index].spec.id
        machine_id = env.machines[machine_index].spec.id
        scored.append(
            (
                operation_positions[operation_id],
                _rank_index(solution.machine_rankings[operation_id])[machine_id],
                action,
            )
        )
    return min(scored)[-1]


def _choose_worker_action(env: AssemblySchedulingEnv, solution: MOALNSSolution, mask: np.ndarray) -> int:
    terminal = env.worker_advance_action
    legal_pairs = [int(value) for value in np.flatnonzero(~mask) if int(value) != terminal]
    if not legal_pairs:
        return terminal
    operation_positions = _rank_index(solution.operation_order)
    pending = []
    for machine_index in range(len(env.machines)):
        reconfiguration = _pending_reconfiguration(env, machine_index)
        if reconfiguration is not None:
            pending.append((operation_positions[reconfiguration.operation_id], machine_index, reconfiguration))
    if pending and not bool(mask[terminal]):
        _, primary_machine, primary = min(pending, key=lambda value: (value[0], value[2].id))
        rankings = (
            solution.disassembly_worker_rankings
            if primary.stage == ReconfigurationStage.WAIT_DIS
            else solution.installation_worker_rankings
        )
        waits = (
            solution.disassembly_wait
            if primary.stage == ReconfigurationStage.WAIT_DIS
            else solution.installation_wait
        )
        preferred_worker = rankings[primary.operation_id][0]
        preferred_legal = any(
            env.decode_worker_action(action)
            == (primary_machine, env.instance.worker_index[preferred_worker])
            for action in legal_pairs
        )
        if waits[primary.operation_id] and not preferred_legal:
            return terminal
    scored: list[tuple[int, int, int, int]] = []
    for action in legal_pairs:
        machine_index, worker_index = env.decode_worker_action(action)
        reconfiguration = _pending_reconfiguration(env, machine_index)
        if reconfiguration is None:
            raise RuntimeError("a legal worker action has no pending reconfiguration")
        rankings = (
            solution.disassembly_worker_rankings
            if reconfiguration.stage == ReconfigurationStage.WAIT_DIS
            else solution.installation_worker_rankings
        )
        worker_id = env.workers[worker_index].spec.id
        scored.append(
            (
                operation_positions[reconfiguration.operation_id],
                _rank_index(rankings[reconfiguration.operation_id])[worker_id],
                machine_index,
                action,
            )
        )
    return min(scored)[-1]


def _candidate_diagnostics(env: AssemblySchedulingEnv) -> dict[str, Any]:
    instance = env.instance
    if instance is None:
        raise RuntimeError("environment has no instance")
    operation_by_id = {operation.id: operation for operation in instance.operations}
    order_by_id = {order.id: order for order in instance.orders}
    order_flow: dict[str, float] = {}
    metrics = env.metrics()
    completion = metrics["completion_times"]
    for order in instance.orders:
        end = completion[order.id]
        if end is None:
            end = env.current_time + instance.unfinished_order_penalty
        order_flow[order.id] = max(0.0, float(end) - order.release_time)

    worker_labor = {worker.id: worker.labor_cost_per_minute for worker in instance.workers}
    machine_downtime = {machine.id: machine.downtime_cost_per_minute for machine in instance.machines}
    operation_cost = defaultdict(float)
    worker_tasks: dict[str, list[str]] = defaultdict(list)
    log_by_reconfiguration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in env.reconfiguration_log:
        log_by_reconfiguration[str(record["reconfiguration_id"])].append(record)
        worker_tasks[str(record["worker_id"])].append(str(record["operation_id"]))
    for reconfiguration in env.reconfigurations.values():
        records = log_by_reconfiguration.get(reconfiguration.id, [])
        fixed = sum(float(record["fixed_cost"]) for record in records)
        labor = sum(
            float(record["duration"]) * worker_labor[str(record["worker_id"])]
            for record in records
        )
        downtime = 0.0
        if reconfiguration.installation_end_tick is not None:
            downtime = (
                (reconfiguration.installation_end_tick - reconfiguration.lock_tick)
                * instance.resolution
                * machine_downtime[reconfiguration.machine_id]
            )
        operation_cost[reconfiguration.operation_id] += fixed + labor + downtime
    return {
        "order_flow": dict(order_flow),
        "operation_reconfiguration_cost": dict(operation_cost),
        "worker_tasks": {key: tuple(value) for key, value in worker_tasks.items()},
        "worker_load": {worker.spec.id: float(worker.load) for worker in env.workers},
        "worker_peak_fatigue": {worker.spec.id: float(worker.peak_fatigue) for worker in env.workers},
        "operation_wave": {
            operation.id: order_by_id[operation.order_id].wave
            for operation in instance.operations
        },
        "operation_module": {
            operation.id: operation.required_module for operation in instance.operations
        },
    }


def decode_solution(
    config: Mapping[str, Any],
    instance: AssemblyInstance,
    solution: MOALNSSolution,
    preference: PreferenceInput,
    *,
    capture_logs: bool = False,
) -> CandidateEvaluation:
    """Replay one encoded solution exclusively through ``AssemblySchedulingEnv``."""

    effective_preference = normalize_preference(preference)
    solution.validate(instance)
    started = time.perf_counter()
    env = AssemblySchedulingEnv(dict(config))
    env.reset(instance, preference=effective_preference, build_observation=False)
    actions: list[int] = []
    realized: dict[str, Any] = {
        "machines": {},
        "disassembly_workers": {},
        "installation_workers": {},
        "production_wait_operations": [],
        "worker_wait_operations": [],
    }
    while not (env.terminated or env.truncated):
        mask = env.get_action_mask()
        phase = env.decision_type
        if phase == DecisionType.PRODUCTION:
            action = _choose_production_action(env, solution, mask)
            if action == env.production_defer_action:
                ready = [
                    operation for operation in env.operations if operation.state == OperationState.READY
                ]
                if ready:
                    operation_positions = _rank_index(solution.operation_order)
                    realized["production_wait_operations"].append(
                        min(ready, key=lambda value: operation_positions[value.spec.id]).spec.id
                    )
            else:
                operation_index, machine_index = env.decode_production_action(action)
                realized["machines"][env.operations[operation_index].spec.id] = env.machines[machine_index].spec.id
        elif phase == DecisionType.WORKER:
            action = _choose_worker_action(env, solution, mask)
            if action == env.worker_advance_action:
                pending = [
                    _pending_reconfiguration(env, machine_index)
                    for machine_index in range(len(env.machines))
                ]
                pending = [value for value in pending if value is not None]
                if pending:
                    positions = _rank_index(solution.operation_order)
                    realized["worker_wait_operations"].append(
                        min(pending, key=lambda value: positions[value.operation_id]).operation_id
                    )
            else:
                machine_index, worker_index = env.decode_worker_action(action)
                reconfiguration = _pending_reconfiguration(env, machine_index)
                if reconfiguration is None:
                    raise RuntimeError("selected worker action lost its reconfiguration")
                key = (
                    "disassembly_workers"
                    if reconfiguration.stage == ReconfigurationStage.WAIT_DIS
                    else "installation_workers"
                )
                realized[key][reconfiguration.operation_id] = env.workers[worker_index].spec.id
        else:
            raise RuntimeError("non-terminal rollout reached an invalid decision phase")
        actions.append(int(action))
        env.step(action, build_observation=False)
    metrics = env.metrics()
    metrics["decisions"] = len(actions)
    metrics["inference_time_seconds"] = 0.0
    metrics["solve_time_seconds"] = time.perf_counter() - started
    metrics["inference_time_per_decision_ms"] = 0.0
    metrics["schedule_violations"] = env.validate_schedule()
    metrics["action_trace_sha256"] = action_trace_sha256(actions)
    objectives = metrics_objectives(metrics)
    feasible = bool(
        metrics["terminated"]
        and not metrics["truncated"]
        and not metrics["schedule_violations"]
        and int(metrics["completed_orders"]) == int(metrics["total_orders"])
        and float(metrics["maximum_worker_fatigue"])
        <= float(metrics["safe_fatigue_limit"]) + 1e-9
        and int(metrics.get("current_worker_matching_deficit", 0)) == 0
    )
    return CandidateEvaluation(
        solution=solution.clone(),
        preference=effective_preference,
        metrics=metrics,
        objectives=objectives,
        normalized_objectives=normalized_objectives(objectives),
        tchebycheff=augmented_tchebycheff(objectives, effective_preference),
        feasible=feasible,
        action_trace_sha256=str(metrics["action_trace_sha256"]),
        realized=realized,
        diagnostics=_candidate_diagnostics(env),
        schedule_log=tuple(dict(row) for row in env.schedule_log) if capture_logs else (),
        reconfiguration_log=(
            tuple(dict(row) for row in env.reconfiguration_log) if capture_logs else ()
        ),
    )


def _operation_lookup(instance: AssemblyInstance) -> dict[str, OperationSpec]:
    return {operation.id: operation for operation in instance.operations}


def _topological_order(instance: AssemblyInstance, scores: Mapping[str, tuple[Any, ...]]) -> tuple[str, ...]:
    remaining = {order.id: list(order.operations) for order in instance.orders}
    result: list[str] = []
    while remaining:
        candidates = [operations[0] for operations in remaining.values() if operations]
        selected = min(candidates, key=lambda operation: (scores[operation.id], operation.id))
        result.append(selected.id)
        values = remaining[selected.order_id]
        values.pop(0)
        if not values:
            del remaining[selected.order_id]
    return tuple(result)


def _ranking_with_first(values: Sequence[str], first: str) -> tuple[str, ...]:
    return (first, *(value for value in values if value != first))


def _heuristic_trace_seed(
    config: Mapping[str, Any],
    instance: AssemblyInstance,
    preference: PreferenceVector,
) -> MOALNSSolution:
    """Capture the repository heuristic as an ALNS seed without changing its policy."""

    env = AssemblySchedulingEnv(dict(config))
    env.reset(instance, preference=preference, build_observation=False)
    policy = HeuristicPolicy()
    operation_order: list[str] = []
    machine_first: dict[str, str] = {}
    disassembly_first: dict[str, str] = {}
    installation_first: dict[str, str] = {}
    production_wait: dict[str, bool] = {operation.id: False for operation in instance.operations}
    disassembly_wait = dict(production_wait)
    installation_wait = dict(production_wait)
    while not (env.terminated or env.truncated):
        phase = env.decision_type
        action = policy.select_action(env)
        if phase == DecisionType.PRODUCTION:
            if action == env.production_defer_action:
                ready = [operation for operation in env.operations if operation.state == OperationState.READY]
                if ready:
                    production_wait[min(ready, key=lambda value: value.spec.id).spec.id] = True
            else:
                operation_index, machine_index = env.decode_production_action(action)
                operation_id = env.operations[operation_index].spec.id
                if operation_id not in operation_order:
                    operation_order.append(operation_id)
                machine_first[operation_id] = env.machines[machine_index].spec.id
        else:
            if action == env.worker_advance_action:
                pending = [
                    _pending_reconfiguration(env, index) for index in range(len(env.machines))
                ]
                pending = [value for value in pending if value is not None]
                if pending:
                    selected = min(pending, key=lambda value: value.operation_id)
                    target = (
                        disassembly_wait
                        if selected.stage == ReconfigurationStage.WAIT_DIS
                        else installation_wait
                    )
                    target[selected.operation_id] = True
            else:
                machine_index, worker_index = env.decode_worker_action(action)
                reconfiguration = _pending_reconfiguration(env, machine_index)
                if reconfiguration is not None:
                    target = (
                        disassembly_first
                        if reconfiguration.stage == ReconfigurationStage.WAIT_DIS
                        else installation_first
                    )
                    target[reconfiguration.operation_id] = env.workers[worker_index].spec.id
        env.step(action, build_observation=False)
    all_operations = [operation.id for operation in instance.operations]
    operation_order.extend(operation_id for operation_id in all_operations if operation_id not in operation_order)
    machines = [machine.id for machine in instance.machines]
    workers = [worker.id for worker in instance.workers]
    return MOALNSSolution(
        operation_order=tuple(operation_order),
        machine_rankings={
            operation_id: _ranking_with_first(machines, machine_first.get(operation_id, machines[0]))
            for operation_id in all_operations
        },
        disassembly_worker_rankings={
            operation_id: _ranking_with_first(workers, disassembly_first.get(operation_id, workers[0]))
            for operation_id in all_operations
        },
        installation_worker_rankings={
            operation_id: _ranking_with_first(workers, installation_first.get(operation_id, workers[0]))
            for operation_id in all_operations
        },
        production_wait=production_wait,
        disassembly_wait=disassembly_wait,
        installation_wait=installation_wait,
        origin="existing_heuristic",
    )


def _rule_solution(
    instance: AssemblyInstance,
    rule: str,
    rng: random.Random,
) -> MOALNSSolution:
    operations = _operation_lookup(instance)
    machines = tuple(machine.id for machine in instance.machines)
    workers = tuple(worker.id for worker in instance.workers)
    machine_specs = {machine.id: machine for machine in instance.machines}
    worker_specs = {worker.id: worker for worker in instance.workers}
    order_specs = {order.id: order for order in instance.orders}

    def operation_score(operation: OperationSpec) -> tuple[Any, ...]:
        minimum_processing = min(
            operation.base_processing_time
            * machine_specs[machine_id].module_parameters[operation.required_module].processing_speed_factor
            for machine_id in machines
            if operation.required_module in machine_specs[machine_id].module_parameters
        )
        order = order_specs[operation.order_id]
        compatible_initial = sum(
            machine_specs[machine_id].initial_module == operation.required_module
            for machine_id in machines
        )
        if rule == "short_flow":
            return (minimum_processing, order.release_time, operation.id)
        if rule == "configuration_reuse":
            return (-compatible_initial, order.release_time, operation.id)
        if rule == "minimum_reconfiguration_cost":
            cost = min(
                instance.module_costs[operation.required_module].fixed_installation_cost
                + (
                    0.0
                    if machine_specs[machine_id].initial_module == operation.required_module
                    else instance.module_costs[machine_specs[machine_id].initial_module].fixed_disassembly_cost
                )
                for machine_id in machines
                if operation.required_module in machine_specs[machine_id].module_parameters
            )
            return (cost, minimum_processing, operation.id)
        if rule == "lowest_predicted_fatigue":
            fatigue = min(
                worker.initial_fatigue
                for worker in worker_specs.values()
                if operation.required_module in worker.qualified_modules
            )
            return (fatigue, minimum_processing, operation.id)
        if rule == "minimum_load_variance":
            return (operation.sequence, operation.id)
        if rule == "same_wave_randomized":
            return (order.wave, rng.random(), operation.id)
        return (order.release_time, minimum_processing, operation.sequence, operation.id)

    order = _topological_order(instance, {operation_id: operation_score(operation) for operation_id, operation in operations.items()})
    machine_rankings: dict[str, tuple[str, ...]] = {}
    dis_rankings: dict[str, tuple[str, ...]] = {}
    ins_rankings: dict[str, tuple[str, ...]] = {}
    for index, operation_id in enumerate(order):
        operation = operations[operation_id]

        def machine_score(machine_id: str) -> tuple[Any, ...]:
            machine = machine_specs[machine_id]
            if operation.required_module not in machine.module_parameters:
                return (1, math.inf, machine_id)
            parameter = machine.module_parameters[operation.required_module]
            processing = operation.base_processing_time * parameter.processing_speed_factor
            reconfiguration = 0.0 if machine.initial_module == operation.required_module else 1.0
            fixed_cost = instance.module_costs[operation.required_module].fixed_installation_cost
            if machine.initial_module != operation.required_module:
                fixed_cost += instance.module_costs[machine.initial_module].fixed_disassembly_cost
            if rule == "configuration_reuse":
                return (reconfiguration, processing, machine_id)
            if rule == "minimum_reconfiguration_cost":
                return (fixed_cost, reconfiguration, processing, machine_id)
            return (processing + reconfiguration, reconfiguration, machine_id)

        machine_rankings[operation_id] = tuple(sorted(machines, key=machine_score))

        def worker_score(worker_id: str, *, installation: bool) -> tuple[Any, ...]:
            worker = worker_specs[worker_id]
            qualified = operation.required_module in worker.qualified_modules
            rotate = (index + workers.index(worker_id)) % max(1, len(workers))
            if rule == "minimum_load_variance":
                return (0 if qualified else 1, rotate, worker.initial_fatigue, worker_id)
            if rule == "minimum_reconfiguration_cost":
                return (0 if qualified else 1, worker.labor_cost_per_minute, worker.initial_fatigue, worker_id)
            if rule == "lowest_predicted_fatigue":
                return (0 if qualified else 1, worker.initial_fatigue, worker.labor_cost_per_minute, worker_id)
            return (0 if qualified else 1, worker.initial_fatigue, worker.labor_cost_per_minute, worker_id)

        dis_rankings[operation_id] = tuple(sorted(workers, key=lambda value: worker_score(value, installation=False)))
        ins_rankings[operation_id] = tuple(sorted(workers, key=lambda value: worker_score(value, installation=True)))
    waits = {
        operation_id: rule in {"configuration_reuse", "same_wave_randomized"}
        for operation_id in operations
    }
    worker_waits = {
        operation_id: rule == "lowest_predicted_fatigue" for operation_id in operations
    }
    return MOALNSSolution(
        operation_order=order,
        machine_rankings=machine_rankings,
        disassembly_worker_rankings=dis_rankings,
        installation_worker_rankings=ins_rankings,
        production_wait=waits,
        disassembly_wait=worker_waits,
        installation_wait=dict(worker_waits),
        origin=rule,
    )


def _promote(values: tuple[str, ...], selected: str) -> tuple[str, ...]:
    return (selected, *(value for value in values if value != selected))


def matching_safe_repair(solution: MOALNSSolution, candidate: CandidateEvaluation) -> MOALNSSolution:
    """Promote resources actually admitted by the environment's safety masks."""

    repaired = solution.clone(origin="matching_safe_repair")
    changed = False
    for operation_id, machine_id in candidate.realized.get("machines", {}).items():
        values = repaired.machine_rankings[operation_id]
        promoted = _promote(values, str(machine_id))
        changed = changed or promoted != values
        repaired.machine_rankings[operation_id] = promoted
    for key, rankings in (
        ("disassembly_workers", repaired.disassembly_worker_rankings),
        ("installation_workers", repaired.installation_worker_rankings),
    ):
        for operation_id, worker_id in candidate.realized.get(key, {}).items():
            values = rankings[operation_id]
            promoted = _promote(values, str(worker_id))
            changed = changed or promoted != values
            rankings[operation_id] = promoted
    return repaired if changed else solution


def _removed_count(instance: AssemblyInstance, settings: Mapping[str, Any], rng: random.Random) -> int:
    lower, upper = (float(value) for value in settings["destroy_fraction"])
    return min(
        len(instance.operations),
        max(int(settings["minimum_removed_operations"]), int(math.ceil(rng.uniform(lower, upper) * len(instance.operations)))),
    )


def _fill_removed(selected: Iterable[str], solution: MOALNSSolution, count: int) -> tuple[str, ...]:
    result: list[str] = []
    for operation_id in selected:
        if operation_id not in result:
            result.append(operation_id)
        if len(result) == count:
            return tuple(result)
    for operation_id in solution.operation_order:
        if operation_id not in result:
            result.append(operation_id)
        if len(result) == count:
            return tuple(result)
    return tuple(result)


def destroy_operations(
    name: str,
    solution: MOALNSSolution,
    candidate: CandidateEvaluation,
    instance: AssemblyInstance,
    settings: Mapping[str, Any],
    rng: random.Random,
) -> tuple[str, ...]:
    """Choose operations to remove using one of the six project-specific rules."""

    count = _removed_count(instance, settings, rng)
    diagnostics = candidate.diagnostics
    if name == "random":
        return tuple(rng.sample(list(solution.operation_order), count))
    if name == "worst_flow_order":
        flow = diagnostics.get("order_flow", {})
        orders = sorted(instance.orders, key=lambda order: (-float(flow.get(order.id, 0.0)), order.id))
        selected = [operation.id for order in orders for operation in order.operations]
        return _fill_removed(selected, solution, count)
    if name == "high_reconfiguration_segment":
        costs = diagnostics.get("operation_reconfiguration_cost", {})
        anchor = max(solution.operation_order, key=lambda value: (float(costs.get(value, 0.0)), value))
        actual_machines = candidate.realized.get("machines", {})
        machine_id = actual_machines.get(anchor)
        if machine_id is not None:
            machine_segment = [
                operation_id
                for operation_id in solution.operation_order
                if actual_machines.get(operation_id) == machine_id
            ]
            if machine_segment:
                anchor_index = machine_segment.index(anchor)
                start = max(0, min(len(machine_segment) - count, anchor_index - count // 2))
                return _fill_removed(machine_segment[start : start + count], solution, count)
        index = solution.operation_order.index(anchor)
        start = max(0, min(len(solution.operation_order) - count, index - count // 2))
        return tuple(solution.operation_order[start : start + count])
    if name == "fatigue_critical_worker":
        peaks = diagnostics.get("worker_peak_fatigue", {})
        worker = max(peaks, key=lambda value: (float(peaks[value]), value)) if peaks else ""
        tasks = diagnostics.get("worker_tasks", {}).get(worker, ())
        return _fill_removed(tasks, solution, count)
    if name == "load_imbalance_worker":
        loads = diagnostics.get("worker_load", {})
        worker = max(loads, key=lambda value: (float(loads[value]), value)) if loads else ""
        tasks = diagnostics.get("worker_tasks", {}).get(worker, ())
        return _fill_removed(tasks, solution, count)
    if name == "same_wave_related":
        anchor = rng.choice(list(solution.operation_order))
        waves = diagnostics.get("operation_wave", {})
        modules = diagnostics.get("operation_module", {})
        same_wave = [operation_id for operation_id in solution.operation_order if waves.get(operation_id) == waves.get(anchor)]
        related = sorted(
            same_wave,
            key=lambda value: (
                0 if modules.get(value) == modules.get(anchor) else 1,
                abs(solution.operation_order.index(value) - solution.operation_order.index(anchor)),
                value,
            ),
        )
        return _fill_removed(related, solution, count)
    raise ValueError(f"unknown destroy operator {name!r}")


def _valid_positions(order: Sequence[str], operation_id: str, instance: AssemblyInstance) -> tuple[int, int]:
    operation = _operation_lookup(instance)[operation_id]
    positions = {value: index for index, value in enumerate(order)}
    values = next(order_spec.operations for order_spec in instance.orders if order_spec.id == operation.order_id)
    sequence_index = next(index for index, value in enumerate(values) if value.id == operation_id)
    # A direct predecessor may itself be absent during repair.  The nearest
    # *present* transitive predecessor/successor still bounds the insertion;
    # otherwise an early insertion of operation 3 could temporarily put it
    # before operation 1 and make operation 2 impossible to insert later.
    predecessor_positions = [
        positions[value.id] for value in values[:sequence_index] if value.id in positions
    ]
    successor_positions = [
        positions[value.id] for value in values[sequence_index + 1 :] if value.id in positions
    ]
    lower = max(predecessor_positions, default=-1) + 1
    upper = min(successor_positions, default=len(order))
    if lower > upper:
        raise RuntimeError("partial order cannot accept a precedence-feasible insertion")
    return lower, upper


def _sample_positions(lower: int, upper: int, limit: int) -> tuple[int, ...]:
    values = tuple(range(lower, upper + 1))
    if len(values) <= limit:
        return values
    if limit == 1:
        return (values[len(values) // 2],)
    indices = {round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)}
    return tuple(values[index] for index in sorted(indices))


def _insert(order: Sequence[str], operation_id: str, position: int) -> tuple[str, ...]:
    values = list(order)
    values.insert(position, operation_id)
    return tuple(values)


def _complete_partial_order(
    partial: Sequence[str],
    original_order: Sequence[str],
    instance: AssemblyInstance,
) -> tuple[str, ...]:
    """Fill an incomplete encoding through a fresh precedence-safe topological sort.

    During regret repair two adjacent operations of the same order can both be
    absent.  Rebuilding from priority keys, instead of repeatedly splicing the
    partially completed list, avoids a transient predecessor/successor inversion.
    """

    partial_positions = {operation_id: index for index, operation_id in enumerate(partial)}
    original_positions = {operation_id: index for index, operation_id in enumerate(original_order)}
    scale = max(1, len(original_order))
    scores: dict[str, tuple[float, int]] = {}
    for operation in instance.operations:
        if operation.id in partial_positions:
            scores[operation.id] = (2.0 * partial_positions[operation.id], original_positions[operation.id])
        else:
            scores[operation.id] = (
                2.0 * original_positions[operation.id] * max(1, len(partial)) / scale + 1.0,
                original_positions[operation.id],
            )
    return _topological_order(instance, scores)


def _rank_for_mode(
    solution: MOALNSSolution,
    operation_id: str,
    instance: AssemblyInstance,
    mode: str,
    candidate: CandidateEvaluation | None,
) -> MOALNSSolution:
    result = solution.clone(origin=mode)
    operation = _operation_lookup(instance)[operation_id]
    machine_specs = {machine.id: machine for machine in instance.machines}
    worker_specs = {worker.id: worker for worker in instance.workers}

    def machine_score(machine_id: str) -> tuple[float, ...]:
        machine = machine_specs[machine_id]
        if operation.required_module not in machine.module_parameters:
            return (1.0, math.inf)
        parameter = machine.module_parameters[operation.required_module]
        process = operation.base_processing_time * parameter.processing_speed_factor
        recon = 0.0 if machine.initial_module == operation.required_module else 1.0
        cost = instance.module_costs[operation.required_module].fixed_installation_cost
        if recon:
            cost += instance.module_costs[machine.initial_module].fixed_disassembly_cost
        if mode == "minimum_reconfiguration_cost":
            return (0.0, cost, process)
        return (0.0, process + recon, cost)

    result.machine_rankings[operation_id] = tuple(sorted(result.machine_rankings[operation_id], key=machine_score))
    load = candidate.diagnostics.get("worker_load", {}) if candidate is not None else {}
    peak_fatigue = (
        candidate.diagnostics.get("worker_peak_fatigue", {}) if candidate is not None else {}
    )

    def worker_score(worker_id: str) -> tuple[float, ...]:
        worker = worker_specs[worker_id]
        qualified = operation.required_module in worker.qualified_modules
        if mode == "lowest_predicted_fatigue":
            return (
                0.0 if qualified else 1.0,
                float(peak_fatigue.get(worker_id, worker.initial_fatigue)),
                worker.labor_cost_per_minute,
            )
        if mode == "minimum_load_variance":
            return (0.0 if qualified else 1.0, float(load.get(worker_id, 0.0)), worker.initial_fatigue)
        if mode == "minimum_reconfiguration_cost":
            return (0.0 if qualified else 1.0, worker.labor_cost_per_minute, worker.initial_fatigue)
        return (0.0 if qualified else 1.0, worker.initial_fatigue, worker.labor_cost_per_minute)

    result.disassembly_worker_rankings[operation_id] = tuple(sorted(result.disassembly_worker_rankings[operation_id], key=worker_score))
    result.installation_worker_rankings[operation_id] = tuple(sorted(result.installation_worker_rankings[operation_id], key=worker_score))
    result.production_wait[operation_id] = mode in {"lowest_predicted_fatigue", "minimum_reconfiguration_cost"}
    result.disassembly_wait[operation_id] = mode == "lowest_predicted_fatigue"
    result.installation_wait[operation_id] = mode == "lowest_predicted_fatigue"
    return result


@dataclass
class _RepairOutcome:
    solution: MOALNSSolution
    probes: tuple[CandidateEvaluation, ...] = ()


class _CandidateEvaluator:
    def __init__(
        self,
        config: Mapping[str, Any],
        instance: AssemblyInstance,
        preference: PreferenceVector,
        maximum_evaluations: int,
    ) -> None:
        self.config = config
        self.instance = instance
        self.preference = preference
        self.maximum_evaluations = maximum_evaluations
        self.cache: dict[str, CandidateEvaluation] = {}
        self.evaluation_count = 0
        self.cache_hits = 0

    def evaluate(self, solution: MOALNSSolution) -> CandidateEvaluation | None:
        digest = solution.digest()
        cached = self.cache.get(digest)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.evaluation_count >= self.maximum_evaluations:
            return None
        candidate = decode_solution(self.config, self.instance, solution, self.preference)
        self.evaluation_count += 1
        stripped = candidate.without_logs()
        self.cache[digest] = stripped
        return stripped


def _repair_greedy(
    name: str,
    base: MOALNSSolution,
    removed: Sequence[str],
    instance: AssemblyInstance,
    candidate: CandidateEvaluation,
    *,
    original_order: Sequence[str] | None = None,
) -> _RepairOutcome:
    original = tuple(base.operation_order if original_order is None else original_order)
    core = tuple(value for value in original if value not in set(removed))
    solution = base.clone(origin=name)
    solution.operation_order = core
    for operation_id in removed:
        lower, upper = _valid_positions(solution.operation_order, operation_id, instance)
        if name == "earliest_finish":
            position = lower
        elif name == "minimum_reconfiguration_cost":
            position = upper
        elif name == "lowest_predicted_fatigue":
            position = upper
        else:
            position = (lower + upper) // 2
        solution.operation_order = _insert(solution.operation_order, operation_id, position)
        solution = _rank_for_mode(solution, operation_id, instance, name, candidate)
    solution.operation_order = _complete_partial_order(solution.operation_order, original, instance)
    return _RepairOutcome(solution)


def _repair_regret(
    name: str,
    base: MOALNSSolution,
    removed: Sequence[str],
    instance: AssemblyInstance,
    evaluator: _CandidateEvaluator,
    candidate: CandidateEvaluation,
    settings: Mapping[str, Any],
) -> _RepairOutcome:
    original = tuple(base.operation_order)
    remaining = list(removed)
    working = base.clone(origin=name)
    working.operation_order = tuple(value for value in original if value not in set(remaining))
    probes: list[CandidateEvaluation] = []
    regret_rank = 2 if name == "regret_2" else 3
    while remaining:
        choices: list[tuple[float, CandidateEvaluation, MOALNSSolution, str]] = []
        for operation_id in tuple(remaining):
            lower, upper = _valid_positions(working.operation_order, operation_id, instance)
            options: list[tuple[CandidateEvaluation, MOALNSSolution]] = []
            for position in _sample_positions(lower, upper, int(settings["regret_position_candidates"])):
                partial = working.clone(origin=name)
                partial.operation_order = _insert(working.operation_order, operation_id, position)
                for variant in range(int(settings["regret_resource_variants"])):
                    mode = (
                        "earliest_finish"
                        if variant == 1
                        else "minimum_reconfiguration_cost"
                        if variant == 2
                        else "regret"
                    )
                    ranked_partial = _rank_for_mode(
                        partial, operation_id, instance, mode, candidate
                    )
                    option = ranked_partial.clone(origin=name)
                    option.operation_order = _complete_partial_order(
                        option.operation_order,
                        original,
                        instance,
                    )
                    evaluated = evaluator.evaluate(option)
                    if evaluated is None:
                        break
                    probes.append(evaluated)
                    options.append((evaluated, ranked_partial))
                if evaluator.evaluation_count >= evaluator.maximum_evaluations:
                    break
            if not options:
                continue
            options.sort(
                key=lambda value: (
                    0 if value[0].feasible else 1,
                    value[0].tchebycheff if value[0].feasible else value[0].completion_rank,
                    value[0].action_trace_sha256,
                )
            )
            best_candidate, best_partial = options[0]
            values = [
                value[0].tchebycheff if value[0].feasible else 1.0 + value[0].completion_rank[0]
                for value in options
            ]
            comparison = values[min(regret_rank - 1, len(values) - 1)]
            regret = comparison - values[0]
            choices.append((regret, best_candidate, best_partial, operation_id))
            if evaluator.evaluation_count >= evaluator.maximum_evaluations:
                break
        if not choices:
            # Budget exhaustion: retain a deterministic, fully valid repair path.
            fallback = _repair_greedy(
                "earliest_finish",
                working,
                remaining,
                instance,
                candidate,
                original_order=original,
            )
            return _RepairOutcome(fallback.solution, tuple(probes))
        _, _, selected_partial, selected_operation = max(
            choices,
            key=lambda value: (value[0], value[3]),
        )
        working = selected_partial
        remaining.remove(selected_operation)
    working.operation_order = _complete_partial_order(working.operation_order, original, instance)
    return _RepairOutcome(working, tuple(probes))


def repair_solution(
    name: str,
    base: MOALNSSolution,
    removed: Sequence[str],
    instance: AssemblyInstance,
    candidate: CandidateEvaluation,
    evaluator: _CandidateEvaluator,
    settings: Mapping[str, Any],
) -> _RepairOutcome:
    if name in {"regret_2", "regret_3"}:
        return _repair_regret(name, base, removed, instance, evaluator, candidate, settings)
    if name not in REPAIR_OPERATORS:
        raise ValueError(f"unknown repair operator {name!r}")
    return _repair_greedy(name, base, removed, instance, candidate)


def _roulette(weights: Mapping[str, float], rng: random.Random) -> str:
    names = tuple(sorted(weights))
    values = [max(0.0, float(weights[name])) for name in names]
    total = sum(values)
    if total <= 0.0:
        return names[0]
    target = rng.random() * total
    cumulative = 0.0
    for name, value in zip(names, values, strict=True):
        cumulative += value
        if target <= cumulative:
            return name
    return names[-1]


def _update_weights(
    weights: dict[str, float],
    scores: Mapping[str, float],
    uses: Mapping[str, int],
    reaction: float,
    floor: float,
) -> None:
    for name in weights:
        if int(uses.get(name, 0)):
            average = float(scores.get(name, 0.0)) / int(uses[name])
            weights[name] = max(floor, (1.0 - reaction) * weights[name] + reaction * average)
    raw_total = sum(max(0.0, float(value)) for value in weights.values())
    remaining_probability = 1.0 - floor * len(weights)
    if raw_total <= 0.0:
        for name in weights:
            weights[name] = 1.0 / len(weights)
        return
    for name, value in weights.items():
        weights[name] = floor + remaining_probability * max(0.0, float(value)) / raw_total


class MOALNSSolver:
    """Strong, fully environment-decoded multi-objective ALNS baseline."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        algorithm_seed: int | None = None,
        dataset_name: str = "",
    ) -> None:
        self.config = dict(config)
        self.settings = _merge_settings(config)
        self.algorithm_seed = int(config.get("seed", 0) if algorithm_seed is None else algorithm_seed)
        self.dataset_name = str(dataset_name)

    def _initial_solutions(
        self,
        instance: AssemblyInstance,
        preference: PreferenceVector,
        rng: random.Random,
    ) -> tuple[MOALNSSolution, ...]:
        result: list[MOALNSSolution] = []
        for rule in INITIAL_RULES:
            if rule == "existing_heuristic":
                result.append(_heuristic_trace_seed(self.config, instance, preference))
            else:
                result.append(_rule_solution(instance, rule, rng))
        return tuple(result)

    def solve(
        self,
        instance: AssemblyInstance,
        preference: PreferenceInput,
    ) -> SearchResult:
        started = time.perf_counter()
        effective_preference = normalize_preference(preference)
        rng = random.Random(
            derive_mo_alns_seed(
                self.algorithm_seed,
                instance.instance_id,
                effective_preference,
                self.dataset_name,
            )
        )
        maximum = int(self.settings["max_evaluations_per_preference"])
        evaluator = _CandidateEvaluator(self.config, instance, effective_preference, maximum)
        archive = ParetoArchive()
        search_log: list[dict[str, Any]] = []
        initial: list[CandidateEvaluation] = []
        for solution in self._initial_solutions(instance, effective_preference, rng):
            candidate = evaluator.evaluate(solution)
            if candidate is None:
                break
            archive.update(candidate)
            initial.append(candidate)
            search_log.append({
                "kind": "initial",
                "origin": solution.origin,
                "evaluation": evaluator.evaluation_count,
                "feasible": candidate.feasible,
                "tchebycheff": candidate.tchebycheff,
                "archive_size": len(archive.entries),
            })
        if not initial:
            raise RuntimeError("MO-ALNS had no budget to evaluate an initial solution")
        current = min(initial, key=lambda value: (0 if value.feasible else 1, value.tchebycheff if value.feasible else value.completion_rank))
        feasible_initial = [value for value in initial if value.feasible]
        initial_best = min((value.tchebycheff for value in feasible_initial), default=None)
        scalar_best = min((value.tchebycheff for value in archive.entries), default=math.inf)

        destroy_weights = {name: 1.0 for name in DESTROY_OPERATORS}
        repair_weights = {name: 1.0 for name in REPAIR_OPERATORS}
        destroy_scores = Counter()
        repair_scores = Counter()
        destroy_uses = Counter()
        repair_uses = Counter()
        positive_deltas: list[float] = []
        temperature: float | None = None
        initial_temperature: float | None = None
        cooling_per_evaluation: float | None = None
        last_temperature_evaluation = evaluator.evaluation_count
        completed_neighbourhoods = 0
        stale_evaluations = 0
        reheats = 0
        proposals = 0
        max_proposals = maximum * int(self.settings["max_proposals_multiplier"])
        segment = int(self.settings["operator_segment_length"])
        score_config = self.settings["operator_scores"]

        while evaluator.evaluation_count < maximum and proposals < max_proposals:
            proposals += 1
            destroy_name = _roulette(destroy_weights, rng)
            repair_name = _roulette(repair_weights, rng)
            destroy_uses[destroy_name] += 1
            repair_uses[repair_name] += 1
            removed = destroy_operations(
                destroy_name,
                current.solution,
                current,
                instance,
                self.settings,
                rng,
            )
            evaluations_before = evaluator.evaluation_count
            outcome = repair_solution(
                repair_name,
                current.solution,
                removed,
                instance,
                current,
                evaluator,
                self.settings,
            )
            probe_archive_added = False
            for probe in outcome.probes:
                probe_archive_added = archive.update(probe) or probe_archive_added
            candidate = evaluator.evaluate(outcome.solution)
            if candidate is None:
                break
            if bool(self.settings["matching_repair"]):
                repaired = matching_safe_repair(outcome.solution, candidate)
                if repaired.digest() != outcome.solution.digest():
                    re_evaluated = evaluator.evaluate(repaired)
                    if re_evaluated is not None:
                        candidate = re_evaluated
            completed_neighbourhoods += 1
            evaluation_delta = evaluator.evaluation_count - evaluations_before
            archive_added = archive.update(candidate) or probe_archive_added
            was_scalar_best = candidate.feasible and candidate.tchebycheff < scalar_best - 1e-12
            if was_scalar_best:
                scalar_best = candidate.tchebycheff
            improved_current = candidate_better(candidate, current)
            delta = (
                candidate.tchebycheff - current.tchebycheff
                if candidate.feasible and current.feasible
                else math.inf
            )
            calibration_samples = int(self.settings["temperature_calibration_samples"])
            if (
                completed_neighbourhoods <= calibration_samples
                and math.isfinite(delta)
                and delta > 1e-12
            ):
                positive_deltas.append(delta)
            just_calibrated = False
            if temperature is None and completed_neighbourhoods >= calibration_samples:
                if positive_deltas:
                    temperature = max(
                        1e-12,
                        -float(np.median(np.asarray(positive_deltas)))
                        / math.log(float(self.settings["temperature_target_acceptance"])),
                    )
                else:
                    temperature = float(self.settings["temperature_fallback"])
                initial_temperature = temperature
                cooling_per_evaluation = float(self.settings["temperature_final_ratio"]) ** (
                    1.0 / max(1, maximum - evaluator.evaluation_count)
                )
                last_temperature_evaluation = evaluator.evaluation_count
                just_calibrated = True
            accepted = improved_current
            if not accepted and candidate.feasible and current.feasible and temperature is not None and math.isfinite(delta):
                accepted = rng.random() < math.exp(-max(0.0, delta) / max(temperature, 1e-12))
            if accepted:
                current = candidate

            score = 0.0
            if archive_added:
                score = max(score, float(score_config["archive_addition"]))
            if was_scalar_best:
                score = max(score, float(score_config["scalar_best"]))
            if improved_current:
                score = max(score, float(score_config["current_improvement"]))
            if accepted:
                score = max(score, float(score_config["accepted"]))
            destroy_scores[destroy_name] += score
            repair_scores[repair_name] += score
            stale_evaluations = (
                0
                if archive_added or was_scalar_best
                else stale_evaluations + evaluation_delta
            )

            if temperature is not None:
                if not just_calibrated and cooling_per_evaluation is not None:
                    elapsed = max(0, evaluator.evaluation_count - last_temperature_evaluation)
                    temperature *= cooling_per_evaluation ** elapsed
                    last_temperature_evaluation = evaluator.evaluation_count
                if stale_evaluations >= int(self.settings["stagnation_evaluations"]) and reheats < int(self.settings["maximum_reheats"]):
                    baseline = initial_temperature if initial_temperature is not None else float(self.settings["temperature_fallback"])
                    temperature = max(temperature, float(self.settings["reheat_fraction"]) * baseline)
                    reheats += 1
                    stale_evaluations = 0
            if proposals % segment == 0:
                _update_weights(
                    destroy_weights,
                    destroy_scores,
                    destroy_uses,
                    float(self.settings["operator_reaction"]),
                    float(self.settings["operator_minimum_probability"]),
                )
                _update_weights(
                    repair_weights,
                    repair_scores,
                    repair_uses,
                    float(self.settings["operator_reaction"]),
                    float(self.settings["operator_minimum_probability"]),
                )
                destroy_scores.clear()
                repair_scores.clear()
                destroy_uses.clear()
                repair_uses.clear()
            search_log.append({
                "kind": "neighbour",
                "proposal": proposals,
                "evaluation": evaluator.evaluation_count,
                "destroy_operator": destroy_name,
                "repair_operator": repair_name,
                "removed_operation_count": len(removed),
                "feasible": candidate.feasible,
                "tchebycheff": candidate.tchebycheff,
                "accepted": accepted,
                "archive_added": archive_added,
                "scalar_best": was_scalar_best,
                "temperature": temperature,
                "archive_size": len(archive.entries),
            })

        selected = archive.best(effective_preference) if archive.entries else current
        return SearchResult(
            preference=effective_preference,
            selected=selected,
            archive=archive.snapshots(),
            initial_best_tchebycheff=initial_best,
            environment_evaluations=evaluator.evaluation_count,
            cache_hits=evaluator.cache_hits,
            proposal_count=proposals,
            search_time_seconds=time.perf_counter() - started,
            operator_statistics={
                "destroy_weights": dict(destroy_weights),
                "repair_weights": dict(repair_weights),
                "reheats": reheats,
                "temperature": temperature,
            },
            search_log=tuple(search_log),
        )

    def solve_grid(
        self,
        instance: AssemblyInstance,
        preferences: Sequence[PreferenceInput] | None = None,
    ) -> GridSearchResult:
        points = (
            tuple(simplex_lattice(5, include=(CANONICAL_PREFERENCE,)))
            if preferences is None
            else tuple(normalize_preference(value) for value in preferences)
        )
        searches = tuple(self.solve(instance, point) for point in points)
        archive = ParetoArchive()
        for search in searches:
            archive.extend(search.archive)
        endpoints: dict[str, CandidateEvaluation] = {}
        for search in searches:
            endpoints[preference_key(search.preference)] = (
                archive.best(search.preference) if archive.entries else search.selected
            )
        return GridSearchResult(
            preferences=points,
            endpoints=endpoints,
            archive=archive.snapshots(),
            searches=searches,
        )
