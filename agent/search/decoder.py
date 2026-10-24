"""Decode priorities through the scheduling environment and seed populations."""
from __future__ import annotations
import math
import random
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence
import numpy as np
from data.models import AssemblyInstance, OperationSpec
from environment import AssemblySchedulingEnv, DecisionType, PreferenceInput, PreferenceVector, normalize_preference
from environment.types import OperationState, ReconfigurationStage
from utils import action_trace_sha256
from .budget import SearchBudget
from .scoring import production_score, worker_score
from .archive import normalized_objectives, augmented_tchebycheff
from .types import CandidateEvaluation, SearchSolution, metrics_objectives

INITIAL_RULES = (
    "existing_heuristic",
    "earliest_finish",
    "short_flow",
    "configuration_reuse",
    "minimum_reconfiguration_cost",
    "lowest_predicted_fatigue",
    "minimum_workload_delta",
    "module_group_randomized",
)


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


def _choose_production_action(env: AssemblySchedulingEnv, solution: SearchSolution, mask: np.ndarray) -> int:
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
    # Operation priority is the first lexicographic key. Later operations cannot
    # win, even if their dynamic resource score is better; avoid projecting them.
    first_position = min(
        operation_positions[env.operations[env.decode_production_action(a)[0]].spec.id]
        for a in legal_pairs
    )
    scored = []
    for action in legal_pairs:
        operation_index, machine_index = env.decode_production_action(action)
        operation_id = env.operations[operation_index].spec.id
        if operation_positions[operation_id] != first_position:
            continue
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


def _choose_worker_action(env: AssemblySchedulingEnv, solution: SearchSolution, mask: np.ndarray) -> int:
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
    first_position = min(
        operation_positions[_pending_reconfiguration(env, env.decode_worker_action(a)[0]).operation_id]
        for a in legal_pairs
    )
    scored = []
    for action in legal_pairs:
        machine_index, worker_index = env.decode_worker_action(action)
        reconfiguration = _pending_reconfiguration(env, machine_index)
        if reconfiguration is None:
            raise RuntimeError("a legal worker action has no pending reconfiguration")
        if operation_positions[reconfiguration.operation_id] != first_position:
            continue
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
        "operation_module": {
            operation.id: operation.required_module for operation in instance.operations
        },
    }


def decode_solution(
    config: Mapping[str, Any],
    instance: AssemblyInstance,
    solution: SearchSolution,
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


def _rule_solution(
    instance: AssemblyInstance,
    rule: str,
    rng: random.Random,
) -> SearchSolution:
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
        if rule == "module_group_randomized":
            return (operation.required_module, rng.random(), operation.id)
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
        operation_id: rule in {"configuration_reuse", "module_group_randomized"}
        for operation_id in operations
    }
    worker_waits = {
        operation_id: rule == "lowest_predicted_fatigue" for operation_id in operations
    }
    return SearchSolution(
        operation_order=order,
        machine_rankings=machine_rankings,
        disassembly_worker_rankings=dis_rankings,
        installation_worker_rankings=ins_rankings,
        production_wait=waits,
        disassembly_wait=worker_waits,
        installation_wait=dict(worker_waits),
        origin=rule,
    )
