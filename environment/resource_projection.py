"""Pure sequential resource projections used by observation features."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import math
from typing import TYPE_CHECKING, Sequence
import numpy as np

from .dynamics import EPSILON, quantize_to_ticks
from .observation_index import observation_operation_index
from .types import MachineState, OperationState, ReconfigurationStage

if TYPE_CHECKING:
    from .env import AssemblySchedulingEnv


@dataclass
class ProjectedResources:
    workers: list[tuple[int, float]]
    loads: np.ndarray

    def copy(self) -> "ProjectedResources":
        return ProjectedResources(self.workers.copy(), self.loads.copy())


@dataclass(frozen=True)
class ProjectedStage:
    worker_index: int
    module: str
    installation: bool
    start_tick: int
    end_tick: int
    start_fatigue: float
    end_fatigue: float
    committed: bool = False

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick


@dataclass
class ReconfigurationProjection:
    start_tick: int
    end_tick: int
    stages: tuple[ProjectedStage, ...]
    resources: ProjectedResources
    baseline_loads: np.ndarray
    safe_disassembly_workers: int = 0
    safe_installation_workers: int = 0


@dataclass(frozen=True)
class TimeReconfigurationProjection:
    """A reconfiguration result containing only scheduling state."""

    start_tick: int
    end_tick: int
    stages: tuple[ProjectedStage, ...]
    workers: tuple[tuple[int, float], ...]
    safe_disassembly_workers: int = 0
    safe_installation_workers: int = 0


@dataclass
class CandidateProjection:
    processing_start_tick: int | None
    finish_tick: int | None
    resource_ready_tick: int
    path: ReconfigurationProjection | None
    resources: ProjectedResources
    baseline_loads: np.ndarray
    machine_available_tick: int
    source_module: str


@dataclass
class ProjectionCache:
    state_version: int
    capacity: int
    entries: dict = field(default_factory=dict)
    time_entries: dict = field(default_factory=dict)


@contextmanager
def projection_cache_scope(env: "AssemblySchedulingEnv"):
    """Share bounded projections within one read-only observation/chain query."""
    previous = env._projection_transition_cache
    if previous is not None and previous.state_version == env._state_version:
        yield
        return
    capacity = env.config.get("training", {}).get("resource_projection_cache_entries", 1024)
    if not capacity:
        yield
        return
    cache = ProjectionCache(env._state_version, capacity)
    env._projection_transition_cache = cache
    try:
        yield
    finally:
        if env._projection_transition_cache is cache:
            env._projection_transition_cache = (
                previous if previous is not None and previous.state_version == env._state_version else None)


def _copy_projection(path: ReconfigurationProjection | None) -> ReconfigurationProjection | None:
    return (None if path is None else
            replace(path, resources=path.resources.copy(), baseline_loads=path.baseline_loads.copy()))


class ResourceProjector:
    """Honor real commitments, then enumerate one candidate's safe service paths.

    Unassigned tasks on other machines are not reserved as a hypothetical global
    schedule. Each branch owns its worker state and never mutates the kernel.
    """

    def __init__(self, env: "AssemblySchedulingEnv"):
        self.env = env

    def initial_resources(self) -> ProjectedResources:
        return ProjectedResources([self.env._worker_fatigue_at_availability(i)
                                   for i in range(len(self.env.workers))],
                                  self.env._committed_worker_loads.copy())

    def stage_options(self, machine_index: int, module: str, installation: bool,
                      earliest_tick: int, resources: ProjectedResources) -> list[tuple[ProjectedStage, ProjectedResources]]:
        options = []
        resolution = self.env.resolution
        for stage in self._stage_descriptors(machine_index, module, installation, earliest_tick, resources.workers):
            branch = resources.copy()
            branch.workers[stage.worker_index] = (stage.end_tick, stage.end_fatigue)
            branch.loads[stage.worker_index] += stage.duration_ticks*resolution
            options.append((stage, branch))
        return options

    def _stage_descriptors(self, machine_index: int, module: str, installation: bool,
                           earliest_tick: int, workers: Sequence[tuple[int, float]]) -> list[ProjectedStage]:
        """Search every safe worker without materializing speculative load arrays."""
        env = self.env
        resolution = env.resolution
        parameters = env.machines[machine_index].spec.module_parameters[module]
        fatigue_spec = env.instance.fatigue
        base = parameters.installation_base_time if installation else parameters.disassembly_base_time
        coefficient = fatigue_spec.installation_time_coefficient if installation else fatigue_spec.disassembly_time_coefficient
        rate = fatigue_spec.installation_accumulation_rate_per_minute if installation else fatigue_spec.disassembly_accumulation_rate_per_minute
        recovery = fatigue_spec.idle_recovery_rate_per_minute
        options = []
        for wi, worker in enumerate(env.workers):
            if module not in worker.spec.qualified_modules:
                continue
            available, fatigue = workers[wi]
            lower = max(int(earliest_tick), available)

            def at(tick: int) -> tuple[bool, int, float, float]:
                start_fatigue = max(0.0, fatigue - recovery*(tick-available)*resolution)
                duration = max(1, quantize_to_ticks(base*(1+coefficient*start_fatigue), resolution))
                after = start_fatigue + rate*duration*resolution
                return after <= fatigue_spec.maximum_safe_fatigue + EPSILON, duration, start_fatigue, after

            safe, _, _, _ = at(lower)
            if not safe:
                if recovery <= 0:
                    continue
                upper = max(lower, available + math.ceil(fatigue/recovery/resolution)+1)
                if not at(upper)[0]:
                    continue
                while lower < upper:
                    middle = (lower+upper)//2
                    if at(middle)[0]:
                        upper = middle
                    else:
                        lower = middle+1
            _, duration, before, after = at(lower)
            stage = ProjectedStage(wi, module, installation, lower, lower+duration, before, min(1.0, after))
            options.append(stage)
        return options

    def transition(self, machine_index: int, source: str, target: str, earliest_tick: int,
                   resources: ProjectedResources) -> ReconfigurationProjection | None:
        env = self.env
        cache = env._projection_transition_cache
        # Empty and single-stage transitions are cheap; avoid key/copy overhead.
        if (cache is None or cache.state_version != env._state_version
                or source == target or source == env.instance.no_module_state
                or target == env.instance.no_module_state):
            return self._transition_uncached(machine_index, source, target, earliest_tick, resources)
        key = (machine_index, source, target, earliest_tick, tuple(resources.workers),
               resources.loads.dtype.str, resources.loads.shape, resources.loads.tobytes())
        if key in cache.entries:
            return _copy_projection(cache.entries[key])
        result = self._transition_uncached(machine_index, source, target, earliest_tick, resources)
        if len(cache.entries) >= cache.capacity:
            cache.entries.pop(next(iter(cache.entries)))
        # active() and order branches may modify returned paths. Store an isolated copy.
        cache.entries[key] = _copy_projection(result)
        return result

    def _select_route(self, machine_index: int, source: str, target: str,
                      earliest_tick: int,
                      workers: Sequence[tuple[int, float]]
                      ) -> tuple[tuple[ProjectedStage, ...], int, int] | None:
        """Select a safe DIS/INS route without materializing resource loads."""
        if source == target:
            return (), 0, 0
        env = self.env
        branches: list[tuple[tuple[ProjectedStage, ...], Sequence[tuple[int, float]]]] = [((), workers)]
        if source != env.instance.no_module_state:
            branches = []
            for stage in self._stage_descriptors(
                machine_index, source, False, earliest_tick, workers
            ):
                branch_workers = list(workers)
                branch_workers[stage.worker_index] = (stage.end_tick, stage.end_fatigue)
                branches.append(((stage,), branch_workers))
        routes: list[tuple[ProjectedStage, ...]] = []
        for prefix, branch_workers in branches:
            start = prefix[-1].end_tick if prefix else earliest_tick
            if target == env.instance.no_module_state:
                routes.append(prefix)
            else:
                routes.extend(
                    prefix + (stage,)
                    for stage in self._stage_descriptors(
                        machine_index, target, True, start, branch_workers
                    )
                )
        if not routes:
            return None

        def route_key(route: tuple[ProjectedStage, ...]) -> tuple[int, int, int]:
            end = route[-1].end_tick if route else earliest_tick
            dis = next((stage.worker_index for stage in route if not stage.installation), -1)
            ins = next((stage.worker_index for stage in route if stage.installation), -1)
            return end, dis, ins

        stages = min(routes, key=route_key)
        dis_count = len({
            stage.worker_index
            for route in routes
            for stage in route
            if not stage.installation and stage.start_tick == earliest_tick
        })
        if stages and not stages[0].installation:
            selected_dis = stages[0]
            installation_count = len({
                route[-1].worker_index
                for route in routes
                if route and route[0] == selected_dis
                and route[-1].installation
                and route[-1].start_tick == selected_dis.end_tick
            })
        else:
            installation_count = len({
                route[-1].worker_index
                for route in routes
                if route and route[-1].installation
                and route[-1].start_tick == earliest_tick
            })
        return stages, dis_count, installation_count

    def _transition_uncached(self, machine_index: int, source: str, target: str, earliest_tick: int,
                             resources: ProjectedResources) -> ReconfigurationProjection | None:
        # Reuse an immutable timing route populated by the order query. Loads
        # are still materialized from this caller's resources, never from the
        # time result. Full and time cache values retain independent types.
        cache = self.env._projection_transition_cache
        timing_key = None
        if (cache is not None and cache.state_version == self.env._state_version
                and source != target and self.env.instance.no_module_state not in (source, target)):
            timing_key = (machine_index, source, target, earliest_tick, tuple(resources.workers))
        if timing_key is None or timing_key not in cache.time_entries:
            selected = self._select_route(
                machine_index, source, target, earliest_tick, resources.workers
            )
        else:
            timed = cache.time_entries[timing_key]
            selected = (None if timed is None else
                        (timed.stages, timed.safe_disassembly_workers, timed.safe_installation_workers))
        if selected is None:
            return None
        stages, dis_count, installation_count = selected
        state = resources.copy()
        resolution = self.env.resolution
        for stage in stages:
            state.workers[stage.worker_index] = (stage.end_tick, stage.end_fatigue)
            state.loads[stage.worker_index] += stage.duration_ticks*resolution
        return ReconfigurationProjection(earliest_tick, stages[-1].end_tick if stages else earliest_tick,
                                         stages, state, resources.loads.copy(), dis_count, installation_count)

    def transition_time(self, machine_index: int, source: str, target: str,
                        earliest_tick: int,
                        workers: Sequence[tuple[int, float]]
                        ) -> TimeReconfigurationProjection | None:
        """Project a route using only machine timing and worker fatigue."""
        env = self.env
        cache = env._projection_transition_cache
        use_cache = (cache is not None and cache.state_version == env._state_version
                     and source != target and env.instance.no_module_state not in (source, target))
        if use_cache:
            key = (machine_index, source, target, earliest_tick, tuple(workers))
            cached = cache.time_entries.get(key)
            if cached is not None or key in cache.time_entries:
                return cached
        selected = self._select_route(machine_index, source, target, earliest_tick, workers)
        if selected is None:
            result = None
        else:
            stages, dis_count, installation_count = selected
            if stages:
                updated = list(workers)
                for stage in stages:
                    updated[stage.worker_index] = (stage.end_tick, stage.end_fatigue)
                updated_workers = tuple(updated)
            else:
                updated_workers = tuple(workers)
            result = TimeReconfigurationProjection(
                earliest_tick,
                stages[-1].end_tick if stages else earliest_tick,
                stages,
                updated_workers,
                dis_count,
                installation_count,
            )
        if use_cache:
            if len(cache.time_entries) >= cache.capacity:
                cache.time_entries.pop(next(iter(cache.time_entries)))
            cache.time_entries[key] = result
        return result

    def active(self, machine_index: int, resources: ProjectedResources) -> ReconfigurationProjection | None:
        env = self.env
        rec = env._active_reconfiguration(env.machines[machine_index].spec.id)
        if rec is None:
            raise RuntimeError("projection: active machine lacks a reconfiguration")
        if rec.stage == ReconfigurationStage.WAIT_DIS:
            return self.transition(machine_index, rec.source_module, rec.target_module, env.current_tick, resources)
        if rec.stage == ReconfigurationStage.WAIT_INS:
            return self.transition(machine_index, env.instance.no_module_state, rec.target_module, env.current_tick, resources)
        installation = rec.stage == ReconfigurationStage.INS
        worker_id = rec.installation_worker_id if installation else rec.disassembly_worker_id
        wi = next(i for i, w in enumerate(env.workers) if w.spec.id == worker_id)
        prefix = "installation" if installation else "disassembly"
        end = max(env.current_tick, getattr(rec, prefix+"_end_tick"))
        # The actual task is already included in initial availability and committed loads.
        _, after = env._worker_fatigue_at_availability(wi)
        fixed = ProjectedStage(wi, rec.target_module if installation else rec.source_module,
                               installation, env.current_tick, end, env.workers[wi].fatigue, after, True)
        if installation:
            return ReconfigurationProjection(env.current_tick, end, (fixed,), resources.copy(), resources.loads.copy())
        suffix = self.transition(machine_index, env.instance.no_module_state, rec.target_module, end, resources)
        if suffix is None:
            return None
        suffix.start_tick = env.current_tick
        suffix.stages = (fixed,)+suffix.stages
        return suffix

    def active_time(self, machine_index: int,
                    workers: Sequence[tuple[int, float]]
                    ) -> TimeReconfigurationProjection | None:
        """Project an active reconfiguration without load materialization."""
        env = self.env
        rec = env._active_reconfiguration(env.machines[machine_index].spec.id)
        if rec is None:
            raise RuntimeError("projection: active machine lacks a reconfiguration")
        if rec.stage == ReconfigurationStage.WAIT_DIS:
            return self.transition_time(machine_index, rec.source_module, rec.target_module,
                                         env.current_tick, workers)
        if rec.stage == ReconfigurationStage.WAIT_INS:
            return self.transition_time(machine_index, env.instance.no_module_state,
                                         rec.target_module, env.current_tick, workers)
        installation = rec.stage == ReconfigurationStage.INS
        worker_id = rec.installation_worker_id if installation else rec.disassembly_worker_id
        wi = next(i for i, worker in enumerate(env.workers) if worker.spec.id == worker_id)
        prefix = "installation" if installation else "disassembly"
        end = max(env.current_tick, getattr(rec, prefix + "_end_tick"))
        _, after = env._worker_fatigue_at_availability(wi)
        fixed = ProjectedStage(
            wi,
            rec.target_module if installation else rec.source_module,
            installation,
            env.current_tick,
            end,
            env.workers[wi].fatigue,
            after,
            True,
        )
        if installation:
            return TimeReconfigurationProjection(
                env.current_tick, end, (fixed,), tuple(workers)
            )
        suffix = self.transition_time(
            machine_index, env.instance.no_module_state, rec.target_module,
            end, workers
        )
        if suffix is None:
            return None
        return TimeReconfigurationProjection(
            env.current_tick,
            suffix.end_tick,
            (fixed,) + suffix.stages,
            suffix.workers,
            suffix.safe_disassembly_workers,
            suffix.safe_installation_workers,
        )

    def machine_release_time(self, machine_index: int,
                             workers: Sequence[tuple[int, float]]
                             ) -> tuple[int | None, str, tuple[tuple[int, float], ...]]:
        """Release a machine using only worker timing/fatigue state."""
        env = self.env
        machine = env.machines[machine_index]
        if machine.state == MachineState.IDLE:
            return env.current_tick, machine.current_module, tuple(workers)
        if machine.state == MachineState.PROCESSING:
            return max(env.current_tick, machine.busy_until_tick), machine.current_module, tuple(workers)
        path = self.active_time(machine_index, workers)
        rec = env._active_reconfiguration(machine.spec.id)
        if path is None:
            return None, rec.target_module, tuple(workers)
        oi = observation_operation_index(env.instance)[rec.operation_id]
        return (path.end_tick + env.estimate_processing_ticks(oi, machine_index),
                rec.target_module, path.workers)

    def machine_release(self, mi: int, resources: ProjectedResources) -> tuple[int | None, str, ProjectedResources]:
        env = self.env
        machine = env.machines[mi]
        if machine.state == MachineState.IDLE:
            return env.current_tick, machine.current_module, resources.copy()
        if machine.state == MachineState.PROCESSING:
            return max(env.current_tick, machine.busy_until_tick), machine.current_module, resources.copy()
        path = self.active(mi, resources)
        rec = env._active_reconfiguration(machine.spec.id)
        if path is None:
            return None, rec.target_module, resources.copy()
        oi = observation_operation_index(env.instance)[rec.operation_id]
        return path.end_tick+env.estimate_processing_ticks(oi, mi), rec.target_module, path.resources

    def candidate(self, oi: int, mi: int) -> CandidateProjection:
        env = self.env
        op, machine = env.operations[oi], env.machines[mi]
        state = self.initial_resources()
        if op.state == OperationState.PROCESSING and op.machine_id == machine.spec.id:
            return CandidateProjection(op.start_tick, machine.busy_until_tick, op.start_tick, None, state,
                                       state.loads.copy(), env.current_tick, machine.current_module)
        if op.state == OperationState.LOCKED and op.machine_id == machine.spec.id:
            path = self.active(mi, state)
            start = path.end_tick if path is not None else None
            ready = path.stages[0].start_tick if path and path.stages else env.current_tick if path else env.horizon_tick+1
            return CandidateProjection(start, start+env.estimate_processing_ticks(oi, mi) if start is not None else None,
                                       ready, path, path.resources if path else state, state.loads.copy(),
                                       env.current_tick, machine.current_module)
        available, source, state = self.machine_release(mi, state)
        baseline = state.loads.copy()
        if available is None:
            return CandidateProjection(None, None, env.horizon_tick+1, None, state, baseline,
                                       env.horizon_tick+1, source)
        release = quantize_to_ticks(env._order_by_id(op.spec.order_id).release_time, env.resolution)
        earliest = max(available, release)
        path = self.transition(mi, source, op.spec.required_module, earliest, state)
        start = path.end_tick if path else None
        ready = path.stages[0].start_tick if path and path.stages else earliest if path else env.horizon_tick+1
        return CandidateProjection(start, start+env.estimate_processing_ticks(oi, mi) if start is not None else None,
                                   ready, path, path.resources if path else state, baseline, available, source)

    def path_costs(self, mi: int, path: ReconfigurationProjection | None) -> tuple[float, float, float, float, float]:
        """Remaining DIS fixed fee, INS fixed fee, labor, downtime, variance delta."""
        if path is None:
            return 0., 0., 0., 0., 0.
        env = self.env
        dis = ins = labor = 0.0
        for stage in path.stages:
            labor += stage.duration_ticks*env.resolution*env.workers[stage.worker_index].spec.labor_cost_per_minute
            if stage.committed:
                continue
            costs = env.instance.module_costs[stage.module]
            if stage.installation:
                ins += costs.fixed_installation_cost
            else:
                dis += costs.fixed_disassembly_cost
        downtime = ((path.end_tick-path.start_tick)*env.resolution*env.machines[mi].spec.downtime_cost_per_minute
                    if path.stages else 0.0)
        delta = float(np.var(path.resources.loads)-np.var(path.baseline_loads))
        return dis, ins, labor, downtime, delta
