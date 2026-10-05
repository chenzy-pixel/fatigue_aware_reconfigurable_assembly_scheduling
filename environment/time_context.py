"""Soft completion estimates for order chains and known WAIT transitions."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

from .state import ReconfigurationRuntime
from .resource_projection import ResourceProjector, ProjectedResources
from .dynamics import quantize_to_ticks
from .types import OperationState

if TYPE_CHECKING:
    from .env import AssemblySchedulingEnv

TIME_CONTEXT_VERSION = "order_chain_action_context_v2"
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
    loads: np.ndarray | None = None
    materialized: set[int] | None = None

    def copy(self) -> "_Resources":
        return _Resources(self.machines.copy(), self.workers.copy(),
                          self.loads.copy() if self.loads is not None else None,
                          set(self.materialized or ()))


class OrderTimeEstimator:
    """Estimate each order using the shared sequential resource projector."""

    def __init__(self, env: "AssemblySchedulingEnv"):
        self.env = env
        self.projector = ResourceProjector(env)
        state = self.projector.initial_resources()
        self.workers, self.loads = state.workers, state.loads
        self.operation_indices = env.instance.operation_index
        self.machine_indices = env.instance.machine_index
        self.machines = [(env.current_tick, m.current_module) for m in env.machines]

    def _state(self, resources: _Resources) -> ProjectedResources:
        return ProjectedResources(resources.workers.copy(),
                                  (self.loads if resources.loads is None else resources.loads).copy())

    def _save(self, resources: _Resources, state: ProjectedResources) -> None:
        resources.workers = state.workers.copy()
        resources.loads = state.loads.copy()

    def _transition_finish(self, machine: int, source: str, target: str, earliest: int, resources: _Resources) -> int:
        route = self.projector.transition(machine, source, target, earliest, self._state(resources))
        if route is None:
            return max(earliest, self.env.horizon_tick)+self.env.horizon_tick+1
        self._save(resources, route.resources)
        return route.end_tick

    def _reconfiguration_finish(self, machine: int, reconfiguration: ReconfigurationRuntime, resources: _Resources) -> int:
        route = self.projector.active(machine, self._state(resources))
        if route is None:
            return max(self.env.current_tick, self.env.horizon_tick)+self.env.horizon_tick+1
        self._save(resources, route.resources)
        return route.end_tick

    def _release(self, mi: int, resources: _Resources) -> tuple[int, str]:
        if resources.materialized is None:
            resources.materialized = set()
        if mi not in resources.materialized:
            tick, module, state = self.projector.machine_release(mi, self._state(resources))
            self._save(resources, state)
            resources.machines[mi] = (tick if tick is not None else 2*self.env.horizon_tick+1, module)
            resources.materialized.add(mi)
        return resources.machines[mi]

    def finish_ticks(self) -> dict[str, int]:
        env = self.env
        result = {}
        for order in env.instance.orders:
            resources = _Resources(self.machines.copy(), self.workers.copy(), self.loads.copy(), set())
            cursor = max(env.current_tick, quantize_to_ticks(order.release_time, env.resolution))
            for spec in order.operations:
                oi = self.operation_indices[spec.id]
                operation = env.operations[oi]
                if operation.state == OperationState.DONE:
                    continue
                if operation.state == OperationState.PROCESSING:
                    mi = self.machine_indices[operation.machine_id]
                    cursor = max(cursor, env.machines[mi].busy_until_tick)
                    resources.machines[mi] = (cursor, spec.required_module)
                    resources.materialized.add(mi)
                    continue
                if operation.state == OperationState.LOCKED:
                    mi = self.machine_indices[operation.machine_id]
                    rec = env._active_reconfiguration(operation.machine_id)
                    cursor = max(cursor, self._reconfiguration_finish(mi, rec, resources))
                    cursor += env.estimate_processing_ticks(oi, mi)
                    resources.machines[mi] = (cursor, spec.required_module)
                    resources.materialized.add(mi)
                    continue
                choices = []
                for mi, machine in enumerate(env.machines):
                    if spec.required_module not in machine.spec.module_parameters:
                        continue
                    branch = resources.copy()
                    available, source = self._release(mi, branch)
                    start = self._transition_finish(mi, source, spec.required_module, max(cursor, available), branch)
                    choices.append((start+env.estimate_processing_ticks(oi, mi), mi, branch))
                if choices:
                    cursor, mi, resources = min(choices, key=lambda item: (item[0], item[1]))
                    resources.machines[mi] = (cursor, spec.required_module)
                else:
                    cursor += env.horizon_tick+1
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
