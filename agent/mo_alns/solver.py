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

from configs.formal_preferences import formal_preferences

from .budget import SearchBudget, BudgetExhausted
from .projection import SerialPredictor, MODES, production_score, worker_score

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
    "operator_segment_length": 5,
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
    "max_seconds_per_preference": None,
    "max_evaluations_per_neighbourhood": 6,
    "max_proxy_evaluations_per_neighbourhood": 24,
    "adaptive_weights": True,
    "no_new_evaluation_proposals": 100,
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
    "minimum_workload_delta",
)
INITIAL_RULES = (
    "existing_heuristic",
    "earliest_finish",
    "short_flow",
    "configuration_reuse",
    "minimum_reconfiguration_cost",
    "lowest_predicted_fatigue",
    "minimum_workload_delta",
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
    if result.get("protocol", "mo_alns_solver_budget_v3") != "mo_alns_solver_budget_v3":
        raise ValueError("legacy MO-ALNS protocol requires explicit migration to v3")
    evaluations = result["max_evaluations_per_preference"]
    if evaluations is not None and (type(evaluations) is not int or evaluations < 1):
        raise ValueError("evaluation budget must be a positive integer or null")
    if evaluations is None and result.get("max_seconds_per_preference") is None:
        raise ValueError("at least one search budget is required")
    local = result["max_evaluations_per_neighbourhood"]
    if type(local) is not int or not 1 <= local <= 6:
        raise ValueError("local evaluation budget must be in [1, 6]")
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
        f"mo_alns_solver_v2\0{int(algorithm_seed)}\0{dataset_name}\0{instance_id}\0"
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
    terminal = env.wait_action
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
                production_score(env, operation_index, machine_index, solution.resource_modes[operation_id]) if operation_id in solution.resource_modes else (),
                _rank_index(solution.machine_rankings[operation_id])[machine_id],
                action,
            )
        )
    return min(scored)[-1]


def _choose_worker_action(env: AssemblySchedulingEnv, solution: MOALNSSolution, mask: np.ndarray) -> int:
    terminal = env.wait_action
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
                worker_score(env, machine_index, worker_index, solution.resource_modes[reconfiguration.operation_id]) if reconfiguration.operation_id in solution.resource_modes else (),
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
    budget: SearchBudget | None = None,
    policy: Any = None,
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
        if budget is not None:
            budget.check_time()
        mask = env.get_action_mask()
        phase = env.decision_type
        forced = (policy.select_action(env) if policy is not None else
                  solution.replay_actions[len(actions)] if solution.replay_actions is not None and len(actions) < len(solution.replay_actions) else None)
        if solution.replay_actions is not None and policy is None and forced is None:
            raise RuntimeError("saved action trace ended before environment termination")
        if forced is not None and (forced >= len(mask) or mask[forced]):
            raise RuntimeError("saved/policy action is illegal")
        if phase == DecisionType.PRODUCTION:
            action = forced if forced is not None else _choose_production_action(env, solution, mask)
            if action == env.wait_action:
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
            action = forced if forced is not None else _choose_worker_action(env, solution, mask)
            if action == env.wait_action:
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
        if budget is not None:
            budget.environment_steps += 1
            budget.check_time()
    if solution.replay_actions is not None and policy is None and len(actions) != len(solution.replay_actions):
        raise RuntimeError("saved action trace has trailing actions")
    metrics = env.metrics()
    metrics["decisions"] = len(actions)
    metrics["inference_time_seconds"] = 0.0
    metrics["solve_time_seconds"] = time.perf_counter() - started
    metrics["inference_time_per_decision_ms"] = 0.0
    metrics["schedule_violations"] = env.validate_schedule()
    metrics["action_trace_sha256"] = action_trace_sha256(actions)
    objectives = metrics_objectives(metrics, config)
    feasible = bool(
        metrics["task_succeeded"]
        and not metrics["truncated"]
        and not metrics["schedule_violations"]
        and int(metrics["completed_orders"]) == int(metrics["total_orders"])
        and float(metrics["maximum_worker_fatigue"])
        <= float(metrics["safe_fatigue_limit"]) + 1e-9
        and int(metrics.get("current_worker_matching_deficit", 0)) == 0
    )
    result = CandidateEvaluation(
        solution=solution.clone(),
        preference=effective_preference,
        metrics=metrics,
        objectives=objectives,
        normalized_objectives=normalized_objectives(objectives, config),
        tchebycheff=augmented_tchebycheff(objectives, effective_preference, config),
        feasible=feasible,
        action_trace_sha256=str(metrics["action_trace_sha256"]),
        realized=realized,
        diagnostics=_candidate_diagnostics(env),
        actions=tuple(actions),
        schedule_log=tuple(dict(row) for row in env.schedule_log) if capture_logs else (),
        reconfiguration_log=(
            tuple(dict(row) for row in env.reconfiguration_log) if capture_logs else ()
        ),
    )
    if budget is not None:
        budget.check_time()
    return result


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
            if action == env.wait_action:
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
            if action == env.wait_action:
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
        if rule == "minimum_workload_delta":
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
            if rule == "minimum_workload_delta":
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
    result.replay_actions = None
    result.resource_modes[operation_id] = "earliest_finish" if mode == "regret" else mode
    return result


@dataclass
class _RepairOutcome:
    solution: MOALNSSolution
    probes: tuple[CandidateEvaluation, ...] = ()


# Bounded repair and accounting are implemented in search.py.
from .search import CandidateEvaluator as _CandidateEvaluator, repair_solution


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
    """Independent preference searches with explicitly accounted anytime budgets."""
    def __init__(self, config, *, algorithm_seed=None, dataset_name=""):
        self.config = dict(config)
        self.settings = _merge_settings(config)
        self.algorithm_seed = int(config.get("seed", 0) if algorithm_seed is None else algorithm_seed)
        self.dataset_name = str(dataset_name)

    def solve(self, instance, preference, *, budget=None):
        from .search import solve
        return solve(self, instance, normalize_preference(preference), budget=budget)

    def solve_grid(self, instance, preferences=None):
        points = (tuple(p.preference for p in formal_preferences(self.config, "final_test"))
                  if preferences is None else tuple(normalize_preference(p) for p in preferences))
        searches = tuple(self.solve(instance, p) for p in points)
        archive = ParetoArchive(self.config)
        for search in searches:
            archive.extend(search.archive)
        # Independent request endpoints: the pooled archive is a separate artifact.
        return GridSearchResult(points, {preference_key(s.preference): s.selected for s in searches},
                                archive.snapshots(), searches)
