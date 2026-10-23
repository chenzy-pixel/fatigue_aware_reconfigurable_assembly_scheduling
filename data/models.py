from __future__ import annotations

import math
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FatigueSpec:
    maximum_safe_fatigue: float
    disassembly_time_coefficient: float
    installation_time_coefficient: float
    disassembly_accumulation_rate_per_minute: float
    installation_accumulation_rate_per_minute: float
    idle_recovery_rate_per_minute: float


@dataclass(frozen=True)
class ModuleCostSpec:
    fixed_disassembly_cost: float
    fixed_installation_cost: float


@dataclass(frozen=True)
class MachineModuleSpec:
    installation_base_time: float
    disassembly_base_time: float
    processing_speed_factor: float


@dataclass(frozen=True)
class MachineSpec:
    id: str
    initial_module: str
    downtime_cost_per_minute: float
    module_parameters: dict[str, MachineModuleSpec]


@dataclass(frozen=True)
class WorkerSpec:
    id: str
    qualified_modules: tuple[str, ...]
    labor_cost_per_minute: float
    initial_fatigue: float


@dataclass(frozen=True)
class OperationSpec:
    id: str
    order_id: str
    sequence: int
    required_module: str
    base_processing_time: float


@dataclass(frozen=True)
class OrderSpec:
    id: str
    release_time: float
    operations: tuple[OperationSpec, ...]


@dataclass(frozen=True)
class AssemblyInstance:
    schema_version: str
    instance_id: str
    instance_type: str
    resolution: float
    modules: tuple[str, ...]
    no_module_state: str
    machines: tuple[MachineSpec, ...]
    workers: tuple[WorkerSpec, ...]
    orders: tuple[OrderSpec, ...]
    fatigue: FatigueSpec
    module_costs: dict[str, ModuleCostSpec]
    unfinished_order_penalty: float

    @property
    def operations(self) -> tuple[OperationSpec, ...]:
        return tuple(operation for order in self.orders for operation in order.operations)

    @property
    def operation_index(self) -> dict[str, int]:
        return {operation.id: index for index, operation in enumerate(self.operations)}

    @property
    def machine_index(self) -> dict[str, int]:
        return {machine.id: index for index, machine in enumerate(self.machines)}

    @property
    def worker_index(self) -> dict[str, int]:
        return {worker.id: index for index, worker in enumerate(self.workers)}


def _as_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _require_non_negative(value: Any, name: str) -> float:
    return _as_float(value, name)


def _require_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def parse_instance_dict(
    raw: dict[str, Any],
    *,
    minimum_compatible_machines: int = 2,
    minimum_qualified_workers: int = 2,
) -> AssemblyInstance:
    if raw.get("schema_version") != "3.0.0" or "horizon" in raw.get("time", {}) or "waves" in raw:
        raise ValueError("data schema 3 required; use archived code for historical instances")
    fatigue_raw = raw["fatigue"]
    fatigue = FatigueSpec(
        maximum_safe_fatigue=_as_float(
            fatigue_raw["maximum_safe_fatigue"], "maximum_safe_fatigue"
        ),
        disassembly_time_coefficient=_as_float(
            fatigue_raw["disassembly_time_coefficient"],
            "disassembly_time_coefficient",
        ),
        installation_time_coefficient=_as_float(
            fatigue_raw["installation_time_coefficient"],
            "installation_time_coefficient",
        ),
        disassembly_accumulation_rate_per_minute=_as_float(
            fatigue_raw["disassembly_accumulation_rate_per_minute"],
            "disassembly_accumulation_rate_per_minute",
        ),
        installation_accumulation_rate_per_minute=_as_float(
            fatigue_raw["installation_accumulation_rate_per_minute"],
            "installation_accumulation_rate_per_minute",
        ),
        idle_recovery_rate_per_minute=_as_float(
            fatigue_raw["idle_recovery_rate_per_minute"],
            "idle_recovery_rate_per_minute",
        ),
    )
    module_costs = {
        module: ModuleCostSpec(
            fixed_disassembly_cost=_as_float(
                values["fixed_disassembly_cost"],
                f"{module}.fixed_disassembly_cost",
            ),
            fixed_installation_cost=_as_float(
                values["fixed_installation_cost"],
                f"{module}.fixed_installation_cost",
            ),
        )
        for module, values in raw["modules"].items()
    }
    machines = []
    for machine in raw["machines"]:
        parameters = {
            module: MachineModuleSpec(
                installation_base_time=_as_float(
                    values["installation_base_time"],
                    f"{machine['id']}.{module}.installation_base_time",
                ),
                disassembly_base_time=_as_float(
                    values["disassembly_base_time"],
                    f"{machine['id']}.{module}.disassembly_base_time",
                ),
                processing_speed_factor=_as_float(
                    values["processing_speed_factor"],
                    f"{machine['id']}.{module}.processing_speed_factor",
                ),
            )
            for module, values in machine["module_parameters"].items()
        }
        machines.append(
            MachineSpec(
                id=str(machine["id"]),
                initial_module=str(machine["initial_module"]),
                downtime_cost_per_minute=_as_float(
                    machine["downtime_cost_per_minute"],
                    f"{machine['id']}.downtime_cost_per_minute",
                ),
                module_parameters=parameters,
            )
        )
    workers = tuple(
        WorkerSpec(
            id=str(worker["id"]),
            qualified_modules=tuple(str(value) for value in worker["qualified_modules"]),
            labor_cost_per_minute=_as_float(
                worker["labor_cost_per_minute"],
                f"{worker['id']}.labor_cost_per_minute",
            ),
            initial_fatigue=_as_float(
                worker.get("initial_fatigue", fatigue_raw.get("initial_fatigue", 0.0)),
                f"{worker['id']}.initial_fatigue",
            ),
        )
        for worker in raw["workers"]
    )
    orders = []
    for order in raw["orders"]:
        operations = tuple(
            OperationSpec(
                id=str(operation["id"]),
                order_id=str(order["id"]),
                sequence=int(operation["sequence"]),
                required_module=str(operation["required_module"]),
                base_processing_time=_as_float(
                    operation["base_processing_time"],
                    f"{operation['id']}.base_processing_time",
                ),
            )
            for operation in order["operations"]
        )
        orders.append(
            OrderSpec(
                id=str(order["id"]),
                release_time=_as_float(
                    order["release_time"], f"{order['id']}.release_time"
                ),
                operations=operations,
            )
        )
    instance = AssemblyInstance(
        schema_version=str(raw["schema_version"]),
        instance_id=str(raw["instance_id"]),
        instance_type=str(raw["instance_type"]),
        resolution=_as_float(raw["time"]["resolution"], "time.resolution"),
        modules=tuple(str(value) for value in raw["sets"]["modules"]),
        no_module_state=str(raw["sets"]["no_module_state"]),
        machines=tuple(machines),
        workers=workers,
        orders=tuple(orders),
        fatigue=fatigue,
        module_costs=module_costs,
        unfinished_order_penalty=_as_float(
            raw["truncation"]["unfinished_order_penalty_per_order"],
            "unfinished_order_penalty_per_order",
        ),
    )
    validate_instance(
        instance,
        minimum_compatible_machines=minimum_compatible_machines,
        minimum_qualified_workers=minimum_qualified_workers,
    )
    return instance


def load_instance_yaml(path: str | Path) -> AssemblyInstance:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_instance_dict(raw.get("instance",raw))


def load_instance_json(
    path: str | Path,
    *,
    minimum_compatible_machines: int = 2,
    minimum_qualified_workers: int = 2,
) -> AssemblyInstance:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return parse_instance_dict(
        raw,
        minimum_compatible_machines=minimum_compatible_machines,
        minimum_qualified_workers=minimum_qualified_workers,
    )


def instance_to_dict(instance: AssemblyInstance) -> dict[str, Any]:
    """Return the canonical machine-readable representation of an instance."""
    return {
        "schema_version": instance.schema_version,
        "instance_id": instance.instance_id,
        "instance_type": instance.instance_type,
        "time": {
            "unit": "minute",
            "resolution": instance.resolution,
        },
        "sets": {
            "modules": list(instance.modules),
            "no_module_state": instance.no_module_state,
            "machines": [machine.id for machine in instance.machines],
            "workers": [worker.id for worker in instance.workers],
            "orders": [order.id for order in instance.orders],
        },
        "fatigue": {
            "initial_fatigue": (
                instance.workers[0].initial_fatigue if instance.workers else 0.0
            ),
            "maximum_safe_fatigue": instance.fatigue.maximum_safe_fatigue,
            "disassembly_time_coefficient": (
                instance.fatigue.disassembly_time_coefficient
            ),
            "installation_time_coefficient": (
                instance.fatigue.installation_time_coefficient
            ),
            "disassembly_accumulation_rate_per_minute": (
                instance.fatigue.disassembly_accumulation_rate_per_minute
            ),
            "installation_accumulation_rate_per_minute": (
                instance.fatigue.installation_accumulation_rate_per_minute
            ),
            "idle_recovery_rate_per_minute": (
                instance.fatigue.idle_recovery_rate_per_minute
            ),
        },
        "modules": {
            module: {
                "fixed_disassembly_cost": costs.fixed_disassembly_cost,
                "fixed_installation_cost": costs.fixed_installation_cost,
            }
            for module, costs in instance.module_costs.items()
        },
        "machines": [
            {
                "id": machine.id,
                "initial_module": machine.initial_module,
                "downtime_cost_per_minute": machine.downtime_cost_per_minute,
                "module_parameters": {
                    module: {
                        "installation_base_time": parameters.installation_base_time,
                        "disassembly_base_time": parameters.disassembly_base_time,
                        "processing_speed_factor": parameters.processing_speed_factor,
                    }
                    for module, parameters in machine.module_parameters.items()
                },
            }
            for machine in instance.machines
        ],
        "workers": [
            {
                "id": worker.id,
                "qualified_modules": list(worker.qualified_modules),
                "labor_cost_per_minute": worker.labor_cost_per_minute,
                "initial_fatigue": worker.initial_fatigue,
            }
            for worker in instance.workers
        ],
        "orders": [
            {
                "id": order.id,
                "release_time": order.release_time,
                "operations": [
                    {
                        "id": operation.id,
                        "sequence": operation.sequence,
                        "required_module": operation.required_module,
                        "base_processing_time": operation.base_processing_time,
                    }
                    for operation in order.operations
                ],
            }
            for order in instance.orders
        ],
        "truncation": {
            "unfinished_order_penalty_per_order": instance.unfinished_order_penalty,
        },
    }


def save_instance_pickle(instance: AssemblyInstance, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(instance, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_instance_pickle(path: str | Path) -> AssemblyInstance:
    with Path(path).open("rb") as handle:
        instance = pickle.load(handle)
    if not isinstance(instance, AssemblyInstance):
        raise TypeError("pickle does not contain an AssemblyInstance")
    validate_instance(instance)
    return instance


def validate_instance(
    instance: AssemblyInstance,
    *,
    minimum_compatible_machines: int = 2,
    minimum_qualified_workers: int = 2,
) -> None:
    if minimum_compatible_machines < 1 or minimum_qualified_workers < 1:
        raise ValueError("minimum resource counts must be positive")
    for name, identifiers in (
        ("module", instance.modules),
        ("machine", tuple(m.id for m in instance.machines)),
        ("worker", tuple(w.id for w in instance.workers)),
    ):
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{name} ids must be nonempty and unique")
    if not instance.orders:
        raise ValueError("instance must contain orders")
    if instance.no_module_state in instance.modules:
        raise ValueError("no_module_state must be distinct from real modules")
    modules = set(instance.modules)
    if not modules:
        raise ValueError("instance must define at least one module")
    _require_positive(instance.resolution, "time.resolution")
    if instance.schema_version != "3.0.0":
        raise ValueError("data schema 3 required")
    _require_non_negative(
        instance.unfinished_order_penalty,
        "unfinished_order_penalty_per_order",
    )
    fatigue_values = {
        "maximum_safe_fatigue": instance.fatigue.maximum_safe_fatigue,
        "disassembly_time_coefficient": (
            instance.fatigue.disassembly_time_coefficient
        ),
        "installation_time_coefficient": (
            instance.fatigue.installation_time_coefficient
        ),
        "disassembly_accumulation_rate_per_minute": (
            instance.fatigue.disassembly_accumulation_rate_per_minute
        ),
        "installation_accumulation_rate_per_minute": (
            instance.fatigue.installation_accumulation_rate_per_minute
        ),
        "idle_recovery_rate_per_minute": (
            instance.fatigue.idle_recovery_rate_per_minute
        ),
    }
    for name, value in fatigue_values.items():
        _require_non_negative(value, name)
    if not 0 < instance.fatigue.maximum_safe_fatigue <= 1:
        raise ValueError("maximum_safe_fatigue must be in (0, 1]")
    if set(instance.module_costs) != modules:
        raise ValueError("module cost table must cover every module")
    for module, costs in instance.module_costs.items():
        _require_non_negative(
            costs.fixed_disassembly_cost,
            f"{module}.fixed_disassembly_cost",
        )
        _require_non_negative(
            costs.fixed_installation_cost,
            f"{module}.fixed_installation_cost",
        )
    for machine in instance.machines:
        if len(machine.module_parameters) != 3:
            raise ValueError(f"{machine.id} must support exactly three modules")
        if machine.initial_module not in machine.module_parameters:
            raise ValueError(f"{machine.id} initial module is not compatible")
        _require_non_negative(
            machine.downtime_cost_per_minute,
            f"{machine.id}.downtime_cost_per_minute",
        )
        for module, parameters in machine.module_parameters.items():
            if module not in modules:
                raise ValueError(
                    f"{machine.id} supports unknown module {module}"
                )
            _require_positive(
                parameters.installation_base_time,
                f"{machine.id}.{module}.installation_base_time",
            )
            _require_positive(
                parameters.disassembly_base_time,
                f"{machine.id}.{module}.disassembly_base_time",
            )
            _require_positive(
                parameters.processing_speed_factor,
                f"{machine.id}.{module}.processing_speed_factor",
            )
    for worker in instance.workers:
        _require_non_negative(
            worker.labor_cost_per_minute,
            f"{worker.id}.labor_cost_per_minute",
        )
        initial_fatigue = _require_non_negative(
            worker.initial_fatigue,
            f"{worker.id}.initial_fatigue",
        )
        if initial_fatigue > instance.fatigue.maximum_safe_fatigue:
            raise ValueError(
                f"{worker.id}.initial_fatigue exceeds maximum_safe_fatigue"
            )
        unknown = set(worker.qualified_modules) - modules
        if unknown:
            raise ValueError(
                f"{worker.id} is qualified for unknown modules "
                f"{sorted(unknown)}"
            )
    for module in modules:
        machine_count = sum(
            module in machine.module_parameters for machine in instance.machines
        )
        worker_count = sum(
            module in worker.qualified_modules for worker in instance.workers
        )
        if (
            machine_count < minimum_compatible_machines
            or worker_count < minimum_qualified_workers
        ):
            raise ValueError(
                f"{module} requires at least {minimum_compatible_machines} "
                "compatible machines and at least "
                f"{minimum_qualified_workers} qualified workers"
            )
    order_ids: set[str] = set()
    operation_ids: set[str] = set()
    precedence_edges: list[tuple[str, str]] = []
    for order in instance.orders:
        if order.id in order_ids:
            raise ValueError(f"duplicate order id {order.id}")
        order_ids.add(order.id)
        _require_non_negative(
            order.release_time,
            f"{order.id}.release_time",
        )
        if order.release_time != 0:
            raise ValueError("batch orders must release at zero")
        if not order.operations:
            raise ValueError(f"{order.id} has no operations")
        expected_sequences = list(range(1, len(order.operations) + 1))
        actual_sequences = [operation.sequence for operation in order.operations]
        if actual_sequences != expected_sequences:
            raise ValueError(f"{order.id} operation sequence is not contiguous")
        for operation in order.operations:
            if operation.id in operation_ids:
                raise ValueError(f"duplicate operation id {operation.id}")
            operation_ids.add(operation.id)
            if operation.order_id != order.id:
                raise ValueError(
                    f"{operation.id} references order {operation.order_id}, "
                    f"expected {order.id}"
                )
            if operation.required_module not in modules:
                raise ValueError(f"{operation.id} requires an unknown module")
            _require_positive(
                operation.base_processing_time,
                f"{operation.id}.base_processing_time",
            )
            if not any(
                operation.required_module in machine.module_parameters
                for machine in instance.machines
            ):
                raise ValueError(f"{operation.id} has no compatible machine")
        precedence_edges.extend(
            (predecessor.id, successor.id)
            for predecessor, successor in zip(
                order.operations,
                order.operations[1:],
            )
        )
    demanded_modules = {
        operation.required_module for operation in instance.operations
    }
    for module in demanded_modules:
        if not any(
            module in worker.qualified_modules for worker in instance.workers
        ):
            raise ValueError(f"{module} has no qualified worker")
    successors = {operation_id: [] for operation_id in operation_ids}
    indegree = {operation_id: 0 for operation_id in operation_ids}
    for predecessor, successor in precedence_edges:
        successors[predecessor].append(successor)
        indegree[successor] += 1
    ready = [
        operation_id
        for operation_id, degree in indegree.items()
        if degree == 0
    ]
    visited = 0
    while ready:
        operation_id = ready.pop()
        visited += 1
        for successor in successors[operation_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if visited != len(operation_ids):
        raise ValueError("operation precedence graph contains a cycle")


def instance_time_scale(instance: AssemblyInstance) -> float:
    from environment.dynamics import quantize_to_ticks
    r = instance.resolution
    times = {o.id: min(max(1, quantize_to_ticks(o.base_processing_time*m.module_parameters[o.required_module].processing_speed_factor,r))*r for m in instance.machines if o.required_module in m.module_parameters) for o in instance.operations}
    bounds = [r, sum(times.values())/len(instance.machines)]
    bounds += [sum(times[o.id] for o in j.operations) for j in instance.orders]
    bounds += [sum(times[o.id] for o in instance.operations if o.required_module==a)/sum(a in m.module_parameters for m in instance.machines) for a in instance.modules]
    return quantize_to_ticks(max(bounds),r)*r
