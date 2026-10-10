"""Soft completion estimates for order chains and known WAIT transitions."""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence
import numpy as np

from .state import ReconfigurationRuntime
from .observation_index import observation_operation_index
from .resource_projection import ResourceProjector, ProjectedResources, projection_cache_scope
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


@dataclass
class _TimeResources:
    machines: list[tuple[int, str]]
    workers: tuple[tuple[int, float], ...]
    materialized: set[int] | None = None


class OrderTimeEstimator:
    """Estimate each order using the shared sequential resource projector."""

    def __init__(self, env: "AssemblySchedulingEnv"):
        self.env = env
        self.projector = ResourceProjector(env)
        self.workers = [env._worker_fatigue_at_availability(i)
                        for i in range(len(env.workers))]
        # Full-state compatibility helpers materialize an isolated load array
        # lazily. The production time-only chain never requests it.
        self._loads: np.ndarray | None = None
        self.operation_indices = observation_operation_index(env.instance)
        self.machine_indices = env.instance.machine_index
        self.machines = [(env.current_tick, m.current_module) for m in env.machines]

    @property
    def loads(self) -> np.ndarray:
        if self._loads is None:
            self._loads = self.env._committed_worker_loads.copy()
        return self._loads

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

    def _transition_finish_time(self, machine: int, source: str, target: str,
                                earliest: int, resources: _TimeResources) -> int:
        route = self.projector.transition_time(
            machine, source, target, earliest, resources.workers
        )
        if route is None:
            return max(earliest, self.env.horizon_tick) + self.env.horizon_tick + 1
        resources.workers = route.workers
        return route.end_tick

    def _reconfiguration_finish_time(self, machine: int,
                                      reconfiguration: ReconfigurationRuntime,
                                      resources: _TimeResources) -> int:
        route = self.projector.active_time(machine, resources.workers)
        if route is None:
            return max(self.env.current_tick, self.env.horizon_tick) + self.env.horizon_tick + 1
        resources.workers = route.workers
        return route.end_tick

    def _release_time(self, machine: int, resources: _TimeResources) -> tuple[int, str]:
        if resources.materialized is None:
            resources.materialized = set()
        if machine not in resources.materialized:
            tick, module, workers = self.projector.machine_release_time(
                machine, resources.workers
            )
            resources.workers = workers
            resources.machines[machine] = (
                tick if tick is not None else 2 * self.env.horizon_tick + 1,
                module,
            )
            resources.materialized.add(machine)
        return resources.machines[machine]

    def _release(self, mi: int, resources: _Resources) -> tuple[int, str]:
        if resources.materialized is None:
            resources.materialized = set()
        if mi not in resources.materialized:
            tick, module, state = self.projector.machine_release(mi, self._state(resources))
            self._save(resources, state)
            resources.machines[mi] = (tick if tick is not None else 2*self.env.horizon_tick+1, module)
            resources.materialized.add(mi)
        return resources.machines[mi]

    def finish_ticks(self, order_ids: Sequence[str] | None = None) -> dict[str, int]:
        requested = None if order_ids is None else frozenset(order_ids)
        if requested is not None:
            unknown = requested - {order.id for order in self.env.instance.orders}
            if unknown:
                raise KeyError(f"unknown order ids: {sorted(unknown)}")
        with projection_cache_scope(self.env):
            return self._finish_ticks(requested)

    def _finish_ticks(self, order_ids: frozenset[str] | None = None) -> dict[str, int]:
        env = self.env
        result = {}
        initial_workers = tuple(self.workers)
        for order in env.instance.orders:
            if order_ids is not None and order.id not in order_ids:
                continue
            resources = _TimeResources(self.machines.copy(), initial_workers, set())
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
                    cursor = max(cursor, self._reconfiguration_finish_time(mi, rec, resources))
                    cursor += env.estimate_processing_ticks(oi, mi)
                    resources.machines[mi] = (cursor, spec.required_module)
                    resources.materialized.add(mi)
                    continue
                choices = []
                for mi, machine in enumerate(env.machines):
                    if spec.required_module not in machine.spec.module_parameters:
                        continue
                    branch = _TimeResources(
                        resources.machines.copy(), resources.workers,
                        set(resources.materialized or ()),
                    )
                    available, source = self._release_time(mi, branch)
                    start = self._transition_finish_time(
                        mi, source, spec.required_module, max(cursor, available), branch
                    )
                    choices.append((start + env.estimate_processing_ticks(oi, mi), mi, branch))
                if choices:
                    cursor, mi, resources = min(choices, key=lambda item: (item[0], item[1]))
                    resources.machines[mi] = (cursor, spec.required_module)
                else:
                    cursor += env.horizon_tick+1
            result[order.id] = cursor
        return result


def project_wait_state(env: "AssemblySchedulingEnv", wait_ticks: int, *,
                       settle_terminal: bool = False, certificate: dict | None = None) -> "AssemblySchedulingEnv":
    """Copy mutable kernel state and apply known events without an env action."""
    projected = copy(env)
    if settle_terminal:
        # Static reset-time arrays are only read by the WAIT kernel. Runtime
        # records are copied below; caches are cleared before any transition.
        shared = {"config", "_static_relations", "_static_edge_indices",
                  "_machine_module_constant_features", "_processing_lower_bound_ticks"}
        replaced = {"operations", "machines", "workers", "reconfigurations", "_events",
                    "schedule_log", "reconfiguration_log", "_committed_worker_loads",
                    "_machine_reconfiguration", "_order_released", "_order_completion_tick",
                    "_active_committed_worker_tasks", "_post_reconfiguration_process_count"}
        invalidated = {"_production_resource_profile_cache", "_candidate_projection_cache",
                       "_projection_transition_cache",
                       "_stage_projection_cache", "_order_finish_tick_cache",
                       "_wait_certificate_cache", "_action_mask_cache"}
        for name, value in vars(env).items():
            if name in shared or name in replaced or name in invalidated or name.startswith("_capability_"):
                continue
            if isinstance(value, set):
                # Diagnostic set members are immutable ticks/action tuples.
                setattr(projected, name, value.copy())
            elif isinstance(value, (dict, list)):
                setattr(projected, name, deepcopy(value))
            elif isinstance(value, np.ndarray):
                isolated = value.copy()
                isolated.flags.writeable = value.flags.writeable
                setattr(projected, name, isolated)
    for name in ("operations", "machines", "workers"):
        setattr(projected, name, [copy(value) for value in getattr(env, name)])
    projected.reconfigurations = {key: copy(value) for key, value in env.reconfigurations.items()}
    for name in ("_machine_reconfiguration", "_order_released", "_order_completion_tick",
                 "_active_committed_worker_tasks", "_post_reconfiguration_process_count"):
        setattr(projected, name, dict(getattr(env, name)))
    projected._events = ([(*event[:-1], dict(event[-1])) for event in env._events]
                         if settle_terminal else env._events.copy())
    # Execution logs contain scalar fields. Copy each record once; retain a
    # deep-copy fallback for extensions containing nested mutable values.
    for name in ("schedule_log", "reconfiguration_log"):
        records = getattr(env, name)
        isolated = [deepcopy(record) if any(isinstance(value, (dict, list, set, np.ndarray))
                                           for value in record.values()) else dict(record)
                    for record in records] if settle_terminal else []
        setattr(projected, name, isolated)
    projected._committed_worker_loads = env._committed_worker_loads.copy()
    projected._invalidate_resource_snapshot()
    if settle_terminal:
        certified = dict(certificate) if certificate is not None else dict(projected._wait_certificate())
        if not certified.get("allowed") or int(certified.get("wait_ticks", -1)) != wait_ticks:
            raise ValueError("WAIT projection requires a matching legal certificate")
        projected._execute_wait(certified)
        projected._resolve_terminal_or_deadlock()
        projected._invalidate_resource_snapshot()
        return projected
    next_tick = env.current_tick + wait_ticks
    projected._advance_interval(next_tick)
    projected.current_tick = next_tick
    projected._process_events_at_current_tick()
    return projected
