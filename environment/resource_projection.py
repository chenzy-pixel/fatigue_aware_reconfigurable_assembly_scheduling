"""Pure sequential resource projections used by observation features."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
import numpy as np

from .dynamics import EPSILON, quantize_to_ticks
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
        env = self.env
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
            available, fatigue = resources.workers[wi]
            lower = max(int(earliest_tick), available)

            def at(tick: int) -> tuple[bool, int, float, float]:
                start_fatigue = max(0.0, fatigue - recovery*(tick-available)*env.resolution)
                duration = max(1, quantize_to_ticks(base*(1+coefficient*start_fatigue), env.resolution))
                after = start_fatigue + rate*duration*env.resolution
                return after <= fatigue_spec.maximum_safe_fatigue + EPSILON, duration, start_fatigue, after

            safe, _, _, _ = at(lower)
            if not safe:
                if recovery <= 0:
                    continue
                upper = max(lower, available + math.ceil(fatigue/recovery/env.resolution)+1)
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
            branch = resources.copy()
            branch.workers[wi] = (stage.end_tick, stage.end_fatigue)
            branch.loads[wi] += duration*env.resolution
            options.append((stage, branch))
        return options

    def transition(self, machine_index: int, source: str, target: str, earliest_tick: int,
                   resources: ProjectedResources) -> ReconfigurationProjection | None:
        if source == target:
            return ReconfigurationProjection(earliest_tick, earliest_tick, (), resources.copy(), resources.loads.copy())
        env = self.env
        branches = [((), resources.copy())]
        if source != env.instance.no_module_state:
            branches = [((stage,), state) for stage, state in self.stage_options(machine_index, source, False, earliest_tick, resources)]
        routes = []
        for prefix, state in branches:
            start = prefix[-1].end_tick if prefix else earliest_tick
            if target == env.instance.no_module_state:
                routes.append((prefix, state))
            else:
                routes.extend((prefix+(stage,), updated) for stage, updated in self.stage_options(machine_index, target, True, start, state))
        if not routes:
            return None

        def key(route):
            stages, _ = route
            end = stages[-1].end_tick if stages else earliest_tick
            dis = next((s.worker_index for s in stages if not s.installation), -1)
            ins = next((s.worker_index for s in stages if s.installation), -1)
            return end, dis, ins

        stages, state = min(routes, key=key)
        dis_count = len({s.worker_index for prefix, _ in routes for s in prefix
                         if not s.installation and s.start_tick == earliest_tick})
        # Count installers at the selected disassembly endpoint, with its updated fatigue.
        if stages and not stages[0].installation:
            selected_dis = stages[0]
            installation_count = len({r[-1].worker_index for r, _ in routes
                                      if r[0] == selected_dis and r[-1].installation
                                      and r[-1].start_tick == selected_dis.end_tick})
        else:
            installation_count = len({r[-1].worker_index for r, _ in routes
                                      if r and r[-1].installation and r[-1].start_tick == earliest_tick})
        return ReconfigurationProjection(earliest_tick, stages[-1].end_tick if stages else earliest_tick,
                                         stages, state, resources.loads.copy(), dis_count, installation_count)

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
        oi = env.instance.operation_index[rec.operation_id]
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
