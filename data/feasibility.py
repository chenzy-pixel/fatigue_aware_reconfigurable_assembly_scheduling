from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from data.models import AssemblyInstance


EPSILON = 1e-9
PRECHECK_VERSION = "necessary_conditions_v1"


def quantize_minutes_to_ticks(minutes: float, resolution: float) -> int:
    """Ceil a non-negative duration to the instance event grid."""

    if minutes < 0.0 or not math.isfinite(minutes):
        raise ValueError("duration must be finite and non-negative")
    if resolution <= 0.0 or not math.isfinite(resolution):
        raise ValueError("resolution must be finite and positive")
    return int(math.ceil((minutes - EPSILON) / resolution))


def maximum_matching_size(
    edges: Sequence[Sequence[int]], worker_count: int
) -> int:
    """Return the cardinality of a deterministic bipartite matching."""

    matched_task = [-1] * int(worker_count)

    def augment(task_index: int, seen: set[int]) -> bool:
        for worker_index in edges[task_index]:
            candidate = int(worker_index)
            if candidate in seen:
                continue
            seen.add(candidate)
            previous = matched_task[candidate]
            if previous < 0 or augment(previous, seen):
                matched_task[candidate] = task_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(edges)))


@dataclass(frozen=True)
class StaticFeasibilityAnalysis:
    horizon_tick: int
    release_ticks: dict[str, int]
    minimum_processing_ticks: dict[str, int]
    compatible_machine_edges: dict[str, tuple[int, ...]]
    qualified_worker_edges: dict[str, tuple[int, ...]]
    total_effective_load: float
    module_loads: dict[str, float]
    worker_qualification_density: float

    def load_metrics(self) -> dict[str, Any]:
        return {
            "total_effective_load": self.total_effective_load,
            "module_loads": dict(self.module_loads),
            "max_module_load": max(self.module_loads.values(), default=0.0),
            "worker_qualification_density": self.worker_qualification_density,
        }


@dataclass(frozen=True)
class FeasibilityPrecheckReport:
    passed: bool
    version: str
    reason_codes: tuple[str, ...]
    reason_details: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "version": self.version,
            "reason_codes": list(self.reason_codes),
            "reason_details": [dict(value) for value in self.reason_details],
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class _MandatoryWorkerTask:
    module: str
    deadline_tick: int
    duration_tick: int
    worker_edges: tuple[int, ...]
    compulsory_start_tick: int | None
    compulsory_end_tick: int | None


def analyze_static_feasibility(
    instance: AssemblyInstance,
) -> StaticFeasibilityAnalysis:
    """Build reusable optimistic resource bounds without constructing an env."""

    resolution = float(instance.resolution)
    horizon_tick = quantize_minutes_to_ticks(instance.horizon, resolution)
    release_ticks = {
        order.id: quantize_minutes_to_ticks(order.release_time, resolution)
        for order in instance.orders
    }
    compatible_machine_edges: dict[str, tuple[int, ...]] = {}
    minimum_processing_ticks: dict[str, int] = {}
    for operation in instance.operations:
        compatible = tuple(
            machine_index
            for machine_index, machine in enumerate(instance.machines)
            if operation.required_module in machine.module_parameters
        )
        compatible_machine_edges[operation.id] = compatible
        if compatible:
            minimum_processing_ticks[operation.id] = min(
                max(
                    1,
                    quantize_minutes_to_ticks(
                        operation.base_processing_time
                        * instance.machines[machine_index]
                        .module_parameters[operation.required_module]
                        .processing_speed_factor,
                        resolution,
                    ),
                )
                for machine_index in compatible
            )
        else:
            minimum_processing_ticks[operation.id] = horizon_tick + 1

    qualified_worker_edges = {
        module: tuple(
            worker_index
            for worker_index, worker in enumerate(instance.workers)
            if module in worker.qualified_modules
        )
        for module in instance.modules
    }
    total_minimum = sum(minimum_processing_ticks.values())
    total_capacity = max(1, len(instance.machines) * horizon_tick)
    module_loads: dict[str, float] = {}
    for module in instance.modules:
        compatible_count = sum(
            module in machine.module_parameters for machine in instance.machines
        )
        module_work = sum(
            minimum_processing_ticks[operation.id]
            for operation in instance.operations
            if operation.required_module == module
        )
        capacity = compatible_count * horizon_tick
        module_loads[module] = (
            module_work / capacity if capacity > 0 else math.inf
        )
    qualification_count = sum(
        module in worker.qualified_modules
        for worker in instance.workers
        for module in instance.modules
    )
    qualification_capacity = len(instance.workers) * len(instance.modules)
    return StaticFeasibilityAnalysis(
        horizon_tick=horizon_tick,
        release_ticks=release_ticks,
        minimum_processing_ticks=minimum_processing_ticks,
        compatible_machine_edges=compatible_machine_edges,
        qualified_worker_edges=qualified_worker_edges,
        total_effective_load=total_minimum / total_capacity,
        module_loads=module_loads,
        worker_qualification_density=(
            qualification_count / qualification_capacity
            if qualification_capacity
            else 0.0
        ),
    )


def _mandatory_installation_tasks(
    instance: AssemblyInstance,
    analysis: StaticFeasibilityAnalysis,
) -> tuple[_MandatoryWorkerTask, ...]:
    initial_modules = {machine.initial_module for machine in instance.machines}
    demanded_modules = {
        operation.required_module for operation in instance.operations
    }
    tasks: list[_MandatoryWorkerTask] = []
    resolution = float(instance.resolution)
    safe_limit = float(instance.fatigue.maximum_safe_fatigue)
    accumulation = float(
        instance.fatigue.installation_accumulation_rate_per_minute
    )
    for module in sorted(demanded_modules - initial_modules):
        deadlines: list[int] = []
        for order in instance.orders:
            suffix = 0
            for operation in reversed(order.operations):
                suffix += analysis.minimum_processing_ticks[operation.id]
                if operation.required_module == module:
                    deadlines.append(analysis.horizon_tick - suffix)
        deadline = min(deadlines, default=-1)
        worker_durations: list[tuple[int, int]] = []
        compatible_installation_times = [
            parameters.installation_base_time
            for machine in instance.machines
            if (parameters := machine.module_parameters.get(module)) is not None
        ]
        if compatible_installation_times and deadline >= 0:
            base_ticks = max(
                1,
                quantize_minutes_to_ticks(
                    min(compatible_installation_times), resolution
                ),
            )
            projected_fatigue = accumulation * base_ticks * resolution
            if projected_fatigue <= safe_limit + EPSILON:
                worker_durations.extend(
                    (worker_index, base_ticks)
                    for worker_index in analysis.qualified_worker_edges.get(
                        module, ()
                    )
                )
        edges = tuple(worker for worker, _ in worker_durations)
        duration = min(
            (value for _, value in worker_durations),
            default=analysis.horizon_tick + 1,
        )
        latest_start = deadline - duration
        earliest_finish = duration
        compulsory_start = latest_start if latest_start < earliest_finish else None
        compulsory_end = earliest_finish if latest_start < earliest_finish else None
        tasks.append(
            _MandatoryWorkerTask(
                module=module,
                deadline_tick=deadline,
                duration_tick=duration,
                worker_edges=edges,
                compulsory_start_tick=compulsory_start,
                compulsory_end_tick=compulsory_end,
            )
        )
    return tuple(tasks)


def cheap_feasibility_precheck(
    instance: AssemblyInstance,
    static_analysis: StaticFeasibilityAnalysis | None = None,
) -> FeasibilityPrecheckReport:
    """Reject only failures of optimistic necessary feasibility conditions."""

    analysis = static_analysis or analyze_static_feasibility(instance)
    details: list[dict[str, Any]] = []

    for operation in instance.operations:
        if not analysis.compatible_machine_edges.get(operation.id):
            details.append(
                {"code": "no_machine_edge", "operation_id": operation.id}
            )
    for module in sorted(
        {operation.required_module for operation in instance.operations}
    ):
        if not analysis.qualified_worker_edges.get(module):
            details.append({"code": "no_worker_edge", "module": module})

    critical_path_max = 0
    for order in instance.orders:
        lower_bound = analysis.release_ticks[order.id] + sum(
            analysis.minimum_processing_ticks[operation.id]
            for operation in order.operations
        )
        critical_path_max = max(critical_path_max, lower_bound)
        if lower_bound > analysis.horizon_tick:
            details.append(
                {
                    "code": "critical_path_over_horizon",
                    "order_id": order.id,
                    "lower_bound_tick": lower_bound,
                    "horizon_tick": analysis.horizon_tick,
                }
            )

    release_cuts = sorted({0, *analysis.release_ticks.values()})
    maximum_machine_load_ratio = 0.0
    maximum_module_load_ratio = 0.0
    for cut in release_cuts:
        remaining = max(0, analysis.horizon_tick - cut)
        released_later = [
            operation
            for order in instance.orders
            if analysis.release_ticks[order.id] >= cut
            for operation in order.operations
        ]
        total_work = sum(
            analysis.minimum_processing_ticks[operation.id]
            for operation in released_later
        )
        total_capacity = len(instance.machines) * remaining
        ratio = total_work / total_capacity if total_capacity else math.inf
        maximum_machine_load_ratio = max(maximum_machine_load_ratio, ratio)
        if total_work > total_capacity:
            details.append(
                {
                    "code": "machine_capacity_over_horizon",
                    "release_cut_tick": cut,
                    "minimum_work_ticks": total_work,
                    "capacity_ticks": total_capacity,
                }
            )
        for module in instance.modules:
            module_work = sum(
                analysis.minimum_processing_ticks[operation.id]
                for operation in released_later
                if operation.required_module == module
            )
            compatible_count = sum(
                module in machine.module_parameters
                for machine in instance.machines
            )
            capacity = compatible_count * remaining
            module_ratio = module_work / capacity if capacity else math.inf
            maximum_module_load_ratio = max(
                maximum_module_load_ratio, module_ratio
            )
            if module_work > capacity:
                details.append(
                    {
                        "code": "module_capacity_over_horizon",
                        "module": module,
                        "release_cut_tick": cut,
                        "minimum_work_ticks": module_work,
                        "capacity_ticks": capacity,
                    }
                )

    mandatory_tasks = _mandatory_installation_tasks(instance, analysis)
    for task in mandatory_tasks:
        if not task.worker_edges or task.duration_tick > task.deadline_tick:
            details.append(
                {
                    "code": "mandatory_task_no_safe_edge",
                    "module": task.module,
                    "deadline_tick": task.deadline_tick,
                    "minimum_duration_tick": task.duration_tick,
                }
            )

    matching_cuts = sorted(
        {
            int(task.compulsory_start_tick)
            for task in mandatory_tasks
            if task.compulsory_start_tick is not None
        }
    )
    maximum_mandatory_deficit = 0
    for cut in matching_cuts:
        simultaneous = [
            task
            for task in mandatory_tasks
            if task.compulsory_start_tick is not None
            and task.compulsory_end_tick is not None
            and task.compulsory_start_tick <= cut < task.compulsory_end_tick
        ]
        edges = [list(task.worker_edges) for task in simultaneous]
        matching = maximum_matching_size(edges, len(instance.workers))
        deficit = len(simultaneous) - matching
        maximum_mandatory_deficit = max(maximum_mandatory_deficit, deficit)
        if deficit > 0:
            details.append(
                {
                    "code": "mandatory_matching_deficit",
                    "tick": cut,
                    "task_modules": [task.module for task in simultaneous],
                    "task_count": len(simultaneous),
                    "matching_size": matching,
                }
            )

    codes = tuple(dict.fromkeys(str(detail["code"]) for detail in details))
    return FeasibilityPrecheckReport(
        passed=not codes,
        version=PRECHECK_VERSION,
        reason_codes=codes,
        reason_details=tuple(details),
        metrics={
            "critical_path_lower_bound_tick": critical_path_max,
            "horizon_tick": analysis.horizon_tick,
            "maximum_machine_load_lower_bound_ratio": (
                maximum_machine_load_ratio
            ),
            "maximum_module_load_lower_bound_ratio": maximum_module_load_ratio,
            "mandatory_task_count": len(mandatory_tasks),
            "maximum_mandatory_matching_deficit": maximum_mandatory_deficit,
        },
    )
