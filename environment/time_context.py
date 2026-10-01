"""Soft completion estimates for order chains and known WAIT transitions."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .state import ReconfigurationRuntime
from .dynamics import quantize_to_ticks
from .types import MachineState, OperationState, ReconfigurationStage

if TYPE_CHECKING:
    from .env import AssemblySchedulingEnv

TIME_CONTEXT_VERSION = "order_chain_action_context_v1"
ORDER_TIME_FEATURE = "estimated_order_slack_norm"
WORKER_WAIT_FEATURE = "stage_wait_time_norm"
WAIT_TIME_FEATURES = (
    "minimum_order_slack_after_wait_norm",
    "minimum_order_slack_delta_if_wait_norm",
)
TIME_CONTEXT_FEATURE_SCHEMA = {
    "order": [ORDER_TIME_FEATURE],
    "production": [ORDER_TIME_FEATURE],
    "worker": [ORDER_TIME_FEATURE, WORKER_WAIT_FEATURE],
    "wait": ["wait_duration_norm", *WAIT_TIME_FEATURES],
}


@dataclass
class _Resources:
    machines: list[tuple[int, str]]
    workers: list[tuple[int, float]]

    def copy(self) -> "_Resources":
        return _Resources(self.machines.copy(), self.workers.copy())


class OrderTimeEstimator:
    """Project one order at a time, honoring physical work already committed.

    Future choices reserve resources within the estimated chain. Other orders'
    undecided assignments are not a known schedule and are not hard constraints.
    """

    def __init__(self, env: "AssemblySchedulingEnv"):
        self.env = env
        self.operation_indices = env.instance.operation_index
        self.machine_indices = env.instance.machine_index
        self.workers = [env._worker_fatigue_at_availability(i) for i in range(len(env.workers))]
        self.machines = []
        for i, machine in enumerate(env.machines):
            if machine.state == MachineState.IDLE:
                available = (env.current_tick, machine.current_module)
            elif machine.state == MachineState.PROCESSING:
                available = (max(env.current_tick, machine.busy_until_tick or env.current_tick),
                             machine.current_module)
            else:
                reconfiguration = env._active_reconfiguration(machine.spec.id)
                if reconfiguration is None:
                    raise RuntimeError("active machine lacks its reconfiguration")
                resources = _Resources([], self.workers.copy())
                end = self._reconfiguration_finish(i, reconfiguration, resources)
                op = self.operation_indices[reconfiguration.operation_id]
                available = (end + env.estimate_processing_ticks(op, i), reconfiguration.target_module)
            self.machines.append(available)

    def _stage_finish(
        self, machine_index: int, module: str, installation: bool,
        earliest: int, resources: _Resources,
    ) -> int:
        env = self.env
        temporary = ReconfigurationRuntime(
            id="time_projection", machine_id=env.machines[machine_index].spec.id,
            operation_id="", source_module=env.instance.no_module_state if installation else module,
            target_module=module if installation else env.instance.no_module_state,
            lock_tick=env.current_tick,
            stage=ReconfigurationStage.WAIT_INS if installation else ReconfigurationStage.WAIT_DIS,
        )
        recovery = env.instance.fatigue.idle_recovery_rate_per_minute
        choices = []
        for i, worker in enumerate(env.workers):
            if module not in worker.spec.qualified_modules:
                continue
            available, fatigue = resources.workers[i]
            start = max(earliest, available)

            def safe(tick: int) -> tuple[bool, int]:
                return env._safe_stage_projection_at_tick(
                    temporary, worker, available_tick=available, available_fatigue=fatigue,
                    recovery_rate=recovery, tick=tick,
                )

            allowed, duration = safe(start)
            if not allowed:
                if recovery <= 0:
                    continue
                upper = max(start, available + math.ceil(fatigue / recovery / env.resolution) + 1)
                if not safe(upper)[0]:
                    continue
                lower = start
                while lower < upper:
                    middle = (lower + upper) // 2
                    if safe(middle)[0]:
                        upper = middle
                    else:
                        lower = middle + 1
                start = lower
                _, duration = safe(start)
            start_fatigue = max(0.0, fatigue - recovery * (start - available) * env.resolution)
            after = start_fatigue + env._stage_accumulation_rate(temporary) * duration * env.resolution
            choices.append((start + duration, i, after))
        if not choices:
            # A finite pessimistic sentinel keeps unusable stages visible to the
            # context network without changing action feasibility.
            return earliest + env.horizon_tick + 1
        end, worker, fatigue = min(choices)
        resources.workers[worker] = (end, min(1.0, fatigue))
        return end

    def _transition_finish(
        self, machine: int, source: str, target: str, earliest: int, resources: _Resources
    ) -> int:
        if source == target:
            return earliest
        if source != self.env.instance.no_module_state:
            earliest = self._stage_finish(machine, source, False, earliest, resources)
        return self._stage_finish(machine, target, True, earliest, resources)

    def _reconfiguration_finish(
        self, machine: int, reconfiguration: ReconfigurationRuntime, resources: _Resources
    ) -> int:
        env = self.env
        stage = reconfiguration.stage
        if stage == ReconfigurationStage.WAIT_DIS:
            end = self._stage_finish(machine, reconfiguration.source_module, False, env.current_tick, resources)
            return self._stage_finish(machine, reconfiguration.target_module, True, end, resources)
        if stage == ReconfigurationStage.DIS:
            end = max(env.current_tick, reconfiguration.disassembly_end_tick or env.current_tick)
            return self._stage_finish(machine, reconfiguration.target_module, True, end, resources)
        if stage == ReconfigurationStage.WAIT_INS:
            return self._stage_finish(machine, reconfiguration.target_module, True, env.current_tick, resources)
        if stage == ReconfigurationStage.INS:
            return max(env.current_tick, reconfiguration.installation_end_tick or env.current_tick)
        return env.current_tick

    def finish_ticks(self) -> dict[str, int]:
        env = self.env
        result = {}
        for order in env.instance.orders:
            resources = _Resources(self.machines.copy(), self.workers.copy())
            cursor = max(env.current_tick, quantize_to_ticks(order.release_time, env.resolution))
            for spec in order.operations:
                op = self.operation_indices[spec.id]
                runtime = env.operations[op]
                if runtime.state == OperationState.DONE:
                    continue
                if runtime.state == OperationState.PROCESSING:
                    machine = env.machines[self.machine_indices[runtime.machine_id]]
                    cursor = max(cursor, machine.busy_until_tick or env.current_tick)
                    continue
                if runtime.state == OperationState.LOCKED:
                    machine = self.machine_indices[runtime.machine_id]
                    reconfiguration = env._active_reconfiguration(runtime.machine_id)
                    cursor = max(cursor, self._reconfiguration_finish(machine, reconfiguration, resources))
                    cursor += env.estimate_processing_ticks(op, machine)
                    resources.machines[machine] = (cursor, spec.required_module)
                    continue
                choices = []
                for machine, machine_runtime in enumerate(env.machines):
                    if spec.required_module not in machine_runtime.spec.module_parameters:
                        continue
                    available, source = resources.machines[machine]
                    candidate = resources.copy()
                    start = self._transition_finish(
                        machine, source, spec.required_module, max(cursor, available), candidate
                    )
                    end = start + env.estimate_processing_ticks(op, machine)
                    choices.append((end, machine, candidate))
                if choices:
                    cursor, machine, resources = min(choices, key=lambda item: (item[0], item[1]))
                    resources.machines[machine] = (cursor, spec.required_module)
                else:
                    cursor += env.horizon_tick + 1
            result[order.id] = cursor
        return result


def project_wait_state(env: "AssemblySchedulingEnv", wait_ticks: int) -> "AssemblySchedulingEnv":
    """Copy mutable kernel state and apply known events without an env action."""
    projected = copy(env)
    for name in ("operations", "machines", "workers"):
        setattr(projected, name, [copy(value) for value in getattr(env, name)])
    projected.reconfigurations = {key: copy(value) for key, value in env.reconfigurations.items()}
    for name in ("_machine_reconfiguration", "_order_released", "_order_completion_tick",
                 "_active_committed_worker_tasks", "_post_reconfiguration_process_count"):
        setattr(projected, name, dict(getattr(env, name)))
    projected._events = env._events.copy()
    projected.schedule_log = []
    projected.reconfiguration_log = []
    projected._committed_worker_loads = env._committed_worker_loads.copy()
    projected._invalidate_resource_snapshot()
    next_tick = env.current_tick + wait_ticks
    projected._advance_interval(next_tick)
    projected.current_tick = next_tick
    projected._process_events_at_current_tick()
    return projected
