from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from data.models import (
    AssemblyInstance,
    MachineSpec,
    OperationSpec,
    WorkerSpec,
)
from environment.types import (
    CAPABLE_EDGE,
    CAN_DISASSEMBLE_EDGE,
    CAN_INSTALL_EDGE,
    LOCKED_EDGE,
    PRECEDES_EDGE,
    DecisionType,
    EdgeStore,
    EdgeType,
    EventType,
    MachineState,
    Observation,
    OperationState,
    ReconfigurationStage,
    RewardVector,
    WorkerState,
    bounded_quality_score,
)


EPSILON = 1e-9
ACTIVE_RECONFIGURATION_STAGES = (
    ReconfigurationStage.WAIT_DIS,
    ReconfigurationStage.DIS,
    ReconfigurationStage.WAIT_INS,
    ReconfigurationStage.INS,
)


def quantize_to_ticks(minutes: float, resolution: float) -> int:
    """Ceil a duration to the event grid with floating point protection."""
    if minutes < 0 or not math.isfinite(minutes):
        raise ValueError("duration must be finite and non-negative")
    return int(math.ceil((minutes - EPSILON) / resolution))


def ticks_to_minutes(ticks: int, resolution: float) -> float:
    return float(ticks) * resolution


def _as_edge_index(pairs: list[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    ordered = sorted(pairs)
    return np.asarray(ordered, dtype=np.int64).T


def _maximum_matching_size(edges: list[list[int]], worker_count: int) -> int:
    matched_task = [-1] * worker_count

    def augment(task_index: int, seen: set[int]) -> bool:
        for worker_index in edges[task_index]:
            if worker_index in seen:
                continue
            seen.add(worker_index)
            previous = matched_task[worker_index]
            if previous < 0 or augment(previous, seen):
                matched_task[worker_index] = task_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(edges)))


@dataclass
class OperationRuntime:
    spec: OperationSpec
    state: OperationState
    machine_id: str | None = None
    start_tick: int | None = None
    end_tick: int | None = None


@dataclass
class MachineRuntime:
    spec: MachineSpec
    state: MachineState
    current_module: str
    busy_until_tick: int | None = None
    locked_operation_id: str | None = None
    source_module: str | None = None
    target_module: str | None = None


@dataclass
class WorkerRuntime:
    spec: WorkerSpec
    state: WorkerState
    fatigue: float
    peak_fatigue: float = 0.0
    load: float = 0.0
    busy_until_tick: int | None = None


@dataclass
class ReconfigurationRuntime:
    id: str
    machine_id: str
    operation_id: str
    source_module: str
    target_module: str
    lock_tick: int
    stage: ReconfigurationStage = ReconfigurationStage.WAIT_DIS
    disassembly_worker_id: str | None = None
    installation_worker_id: str | None = None
    disassembly_start_tick: int | None = None
    disassembly_end_tick: int | None = None
    installation_start_tick: int | None = None
    installation_end_tick: int | None = None


@dataclass(frozen=True)
class WorkerTaskSnapshot:
    """A worker-requiring reconfiguration stage at the current tick."""

    task_id: str
    machine_index: int
    stage: ReconfigurationStage
    module: str


@dataclass(frozen=True)
class ResourceFeasibilitySnapshot:
    """Shared worker feasibility view used by masks, features, and metrics."""

    tasks: tuple[WorkerTaskSnapshot, ...]
    safe_edges: tuple[tuple[int, ...], ...]
    matching_size: int
    safe_idle_workers: tuple[int, ...]
    minimum_worker_alternatives: int


@dataclass(frozen=True)
class ProductionCandidateProfile:
    resource_ready_tick: int
    predicted_finish_tick: int
    safe_disassembly_workers: int
    safe_installation_workers: int
    matching_deficit_after_commit: int
    horizon_slack_ticks: int
    admissible: bool


class AssemblySchedulingEnv:
    """Single-instance discrete-event environment with two decision phases."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.instance: AssemblyInstance | None = None
        self.current_tick = 0
        self.horizon_tick = 0
        self.decision_type = DecisionType.TERMINAL
        self.terminated = False
        self.truncated = False
        self.terminal_reason: str | None = None
        self.operations: list[OperationRuntime] = []
        self.machines: list[MachineRuntime] = []
        self.workers: list[WorkerRuntime] = []
        self.reconfigurations: dict[str, ReconfigurationRuntime] = {}
        self._machine_reconfiguration: dict[str, str] = {}
        self._static_edge_indices: dict[EdgeType, np.ndarray] = {}
        self._events: list[tuple[int, int, int, EventType, dict[str, Any]]] = []
        self._event_serial = 0
        self._decision_count = 0
        self._zero_time_actions = 0
        self._flow_integral = 0.0
        self._flow_penalty = 0.0
        self._reconfiguration_cost = 0.0
        self._maximum_fatigue_seen = 0.0
        self._fatigue_candidate_actions: set[tuple[int, str, str, str]] = set()
        self._fatigue_masked_actions: set[tuple[int, str, str, str]] = set()
        self._worker_competition_ticks: set[int] = set()
        self._worker_matching_deficit_ticks: set[int] = set()
        self._resource_admission_candidates: set[tuple[int, int, int]] = set()
        self._resource_admission_masked: set[tuple[int, int, int]] = set()
        self._matching_preserving_worker_actions: set[
            tuple[int, str, str, str]
        ] = set()
        self._candidate_recovery_advance_count = 0
        self._minimum_worker_alternatives_seen: int | None = None
        self._resource_snapshot_cache: ResourceFeasibilitySnapshot | None = None
        self._candidate_profile_cache: dict[
            tuple[int, int], ProductionCandidateProfile
        ] = {}
        self._stage_projection_cache: dict[
            tuple[int, int, str, bool, int], tuple[int, int] | None
        ] = {}
        self._cumulative_reward = np.zeros(3, dtype=np.float64)
        self._order_released: dict[str, bool] = {}
        self._order_completion_tick: dict[str, int] = {}
        self.schedule_log: list[dict[str, Any]] = []
        self.reconfiguration_log: list[dict[str, Any]] = []

    @property
    def resolution(self) -> float:
        self._require_instance()
        return self.instance.resolution

    @property
    def current_time(self) -> float:
        return ticks_to_minutes(self.current_tick, self.resolution)

    @property
    def production_action_size(self) -> int:
        return len(self.operations) * len(self.machines) + 1

    @property
    def worker_action_size(self) -> int:
        return len(self.machines) * len(self.workers) + 1

    @property
    def advance_action(self) -> int:
        if self.decision_type == DecisionType.PRODUCTION:
            return self.production_action_size - 1
        if self.decision_type == DecisionType.WORKER:
            return self.worker_action_size - 1
        raise RuntimeError("terminal state has no action")

    @property
    def worker_resource_control(self) -> dict[str, Any]:
        settings = self.config.get("environment", {}).get(
            "worker_resource_control",
            {"mode": "legacy_postcheck"},
        )
        if not isinstance(settings, dict):
            raise TypeError("environment.worker_resource_control must be a mapping")
        mode = str(settings.get("mode", "legacy_postcheck"))
        if mode not in {"legacy_postcheck", "matching_admission_v1"}:
            raise ValueError(
                "worker_resource_control.mode must be 'legacy_postcheck' or "
                "'matching_admission_v1'"
            )
        return settings

    @property
    def matching_admission_enabled(self) -> bool:
        return str(self.worker_resource_control.get("mode")) == (
            "matching_admission_v1"
        )

    def _resource_setting(self, name: str, default: bool) -> bool:
        return bool(self.worker_resource_control.get(name, default))

    def _invalidate_resource_snapshot(self) -> None:
        self._resource_snapshot_cache = None
        self._candidate_profile_cache = {}
        self._stage_projection_cache = {}

    def reset(self, instance: AssemblyInstance) -> Observation:
        self.instance = instance
        self.current_tick = 0
        self.horizon_tick = quantize_to_ticks(instance.horizon, instance.resolution)
        self.decision_type = DecisionType.PRODUCTION
        self.terminated = False
        self.truncated = False
        self.terminal_reason = None
        self.operations = [
            OperationRuntime(operation, OperationState.UNRELEASED)
            for operation in instance.operations
        ]
        self.machines = [
            MachineRuntime(machine, MachineState.IDLE, machine.initial_module)
            for machine in instance.machines
        ]
        self.workers = [
            WorkerRuntime(
                worker,
                WorkerState.IDLE,
                worker.initial_fatigue,
                worker.initial_fatigue,
            )
            for worker in instance.workers
        ]
        self.reconfigurations = {}
        self._machine_reconfiguration = {}
        self._static_edge_indices = self._build_static_edge_indices()
        self._events = []
        self._event_serial = 0
        self._decision_count = 0
        self._zero_time_actions = 0
        self._flow_integral = 0.0
        self._flow_penalty = 0.0
        self._reconfiguration_cost = 0.0
        self._maximum_fatigue_seen = max(
            worker.initial_fatigue for worker in instance.workers
        )
        self._fatigue_candidate_actions = set()
        self._fatigue_masked_actions = set()
        self._worker_competition_ticks = set()
        self._worker_matching_deficit_ticks = set()
        self._resource_admission_candidates = set()
        self._resource_admission_masked = set()
        self._matching_preserving_worker_actions = set()
        self._candidate_recovery_advance_count = 0
        self._minimum_worker_alternatives_seen = None
        self._invalidate_resource_snapshot()
        self._cumulative_reward = np.zeros(3, dtype=np.float64)
        self._order_released = {order.id: False for order in instance.orders}
        self._order_completion_tick = {}
        self.schedule_log = []
        self.reconfiguration_log = []
        for order in instance.orders:
            release_tick = quantize_to_ticks(order.release_time, instance.resolution)
            self._push_event(
                release_tick,
                EventType.ORDER_RELEASE,
                {"order_id": order.id},
                priority=1,
            )
        self._process_events_at_current_tick()
        self._resolve_terminal_or_deadlock()
        return self.observe()

    def observe(self) -> Observation:
        self._require_instance()
        module_values = (self.instance.no_module_state, *self.instance.modules)
        operation_states = tuple(OperationState)
        machine_states = tuple(MachineState)
        worker_states = tuple(WorkerState)
        operation_features = []
        for operation in self.operations:
            module_one_hot = [
                float(operation.spec.required_module == module)
                for module in self.instance.modules
            ]
            state_one_hot = [
                float(operation.state == state) for state in operation_states
            ]
            order = self._order_by_id(operation.spec.order_id)
            operation_features.append(
                module_one_hot
                + state_one_hot
                + [
                    order.release_time / self.instance.horizon,
                    operation.spec.base_processing_time / 14.0,
                    operation.spec.sequence / 5.0,
                ]
            )
        machine_features = []
        for machine in self.machines:
            current_one_hot = [
                float(machine.current_module == module) for module in module_values
            ]
            state_one_hot = [
                float(machine.state == state) for state in machine_states
            ]
            target = machine.target_module or machine.current_module
            target_one_hot = [float(target == module) for module in module_values]
            remaining = max(0, (machine.busy_until_tick or self.current_tick) - self.current_tick)
            machine_features.append(
                current_one_hot
                + state_one_hot
                + target_one_hot
                + [
                    ticks_to_minutes(remaining, self.resolution)
                    / self.instance.horizon,
                    machine.spec.downtime_cost_per_minute / 5.0,
                ]
                + [
                    float(module in machine.spec.module_parameters)
                    for module in self.instance.modules
                ]
            )
        worker_features = []
        for worker in self.workers:
            state_one_hot = [
                float(worker.state == state) for state in worker_states
            ]
            remaining = max(0, (worker.busy_until_tick or self.current_tick) - self.current_tick)
            worker_features.append(
                [worker.fatigue]
                + state_one_hot
                + [
                    ticks_to_minutes(remaining, self.resolution)
                    / self.instance.horizon
                ]
                + [
                    float(module in worker.spec.qualified_modules)
                    for module in self.instance.modules
                ]
                + [
                    worker.load / self.instance.horizon,
                    worker.spec.labor_cost_per_minute,
                ]
            )
        active_orders = sum(
            self._order_released[order.id]
            and order.id not in self._order_completion_tick
            for order in self.instance.orders
        )
        ready_operations = sum(
            operation.state == OperationState.READY for operation in self.operations
        )
        pending_reconfigurations = sum(
            reconfiguration.stage
            in {ReconfigurationStage.WAIT_DIS, ReconfigurationStage.WAIT_INS}
            for reconfiguration in self.reconfigurations.values()
        )
        completed_operations = sum(
            operation.state == OperationState.DONE for operation in self.operations
        )
        resource_snapshot = self._resource_feasibility_snapshot()
        if resource_snapshot.tasks:
            safe_worker_indices = {
                worker_index
                for edge in resource_snapshot.safe_edges
                for worker_index in edge
            }
        else:
            safe_worker_indices = set(resource_snapshot.safe_idle_workers)
        matching_deficit = max(
            0,
            len(resource_snapshot.tasks) - resource_snapshot.matching_size,
        )
        candidate_slacks: list[float] = []
        for operation_index, operation in enumerate(self.operations):
            if operation.state != OperationState.READY:
                continue
            for machine_index, machine in enumerate(self.machines):
                if (
                    machine.state == MachineState.IDLE
                    and machine.current_module != self.instance.no_module_state
                    and operation.spec.required_module
                    in machine.spec.module_parameters
                ):
                    profile = self._production_candidate_profile(
                        operation_index,
                        machine_index,
                    )
                    candidate_slacks.append(
                        max(
                            -1.0,
                            min(
                                1.0,
                                profile.horizon_slack_ticks
                                / max(1, self.horizon_tick),
                            ),
                        )
                    )
        global_features = np.asarray(
            [
                self.current_time / self.instance.horizon,
                active_orders / max(1, len(self.instance.orders)),
                ready_operations / max(1, len(self.operations)),
                pending_reconfigurations / max(1, len(self.machines)),
                completed_operations / max(1, len(self.operations)),
                float(self.decision_type == DecisionType.PRODUCTION),
                float(self.decision_type == DecisionType.WORKER),
                len(safe_worker_indices) / max(1, len(self.workers)),
                matching_deficit / max(1, len(resource_snapshot.tasks)),
                resource_snapshot.minimum_worker_alternatives
                / max(1, len(self.workers)),
                min(candidate_slacks) if candidate_slacks else 0.0,
            ],
            dtype=np.float32,
        )
        relations = self._build_graph_relations()
        return Observation(
            operations=np.asarray(operation_features, dtype=np.float32),
            machines=np.asarray(machine_features, dtype=np.float32),
            workers=np.asarray(worker_features, dtype=np.float32),
            global_features=global_features,
            decision_type=self.decision_type,
            global_feature_names=(
                "current_time_norm",
                "active_order_ratio",
                "ready_operation_ratio",
                "pending_reconfiguration_ratio",
                "completed_operation_ratio",
                "production_decision",
                "worker_decision",
                "safe_idle_worker_ratio",
                "worker_matching_deficit_norm",
                "minimum_worker_alternative_ratio",
                "minimum_candidate_horizon_slack",
            ),
            node_ids={
                "operation": tuple(
                    operation.spec.id for operation in self.operations
                ),
                "machine": tuple(machine.spec.id for machine in self.machines),
                "worker": tuple(worker.spec.id for worker in self.workers),
            },
            relations=relations,
        )

    def _build_static_edge_indices(self) -> dict[EdgeType, np.ndarray]:
        self._require_instance()
        operation_index = self.instance.operation_index
        precedence_pairs: list[tuple[int, int]] = []
        for order in self.instance.orders:
            for predecessor, successor in zip(
                order.operations, order.operations[1:]
            ):
                precedence_pairs.append(
                    (
                        operation_index[predecessor.id],
                        operation_index[successor.id],
                    )
                )
        capability_pairs = [
            (operation_index_value, machine_index)
            for operation_index_value, operation in enumerate(self.operations)
            for machine_index, machine in enumerate(self.machines)
            if operation.spec.required_module in machine.spec.module_parameters
        ]
        installation_pairs = [
            (worker_index, operation_index_value)
            for worker_index, worker in enumerate(self.workers)
            for operation_index_value, operation in enumerate(self.operations)
            if operation.spec.required_module in worker.spec.qualified_modules
        ]
        return {
            PRECEDES_EDGE: _as_edge_index(precedence_pairs),
            CAPABLE_EDGE: _as_edge_index(capability_pairs),
            CAN_INSTALL_EDGE: _as_edge_index(installation_pairs),
        }

    def _build_graph_relations(self) -> dict[EdgeType, EdgeStore]:
        self._require_instance()
        precedence_index = self._static_edge_indices[PRECEDES_EDGE]
        precedence = EdgeStore(
            edge_index=precedence_index.copy(),
            edge_features=np.ones(
                (precedence_index.shape[1], 1), dtype=np.float32
            ),
            feature_names=("precedence",),
        )

        capability_index = self._static_edge_indices[CAPABLE_EDGE]
        capability_features = []
        for operation_index, machine_index in capability_index.T:
            operation = self.operations[int(operation_index)]
            machine = self.machines[int(machine_index)]
            profile = self._production_candidate_profile(
                int(operation_index), int(machine_index)
            )
            capability_features.append(
                [
                    min(
                        2.0,
                        max(
                            0.0,
                            self.estimate_processing_ticks(
                                int(operation_index), int(machine_index)
                            )
                            / self.horizon_tick,
                        ),
                    ),
                    float(
                        machine.current_module
                        == operation.spec.required_module
                    ),
                    min(
                        2.0,
                        max(
                            0.0,
                            self.estimate_earliest_start_tick(
                                int(operation_index), int(machine_index)
                            )
                            / self.horizon_tick,
                        ),
                    ),
                    min(
                        2.0,
                        max(0.0, profile.resource_ready_tick / self.horizon_tick),
                    ),
                    min(
                        2.0,
                        max(0.0, profile.predicted_finish_tick / self.horizon_tick),
                    ),
                    profile.safe_disassembly_workers / max(1, len(self.workers)),
                    profile.safe_installation_workers / max(1, len(self.workers)),
                    profile.matching_deficit_after_commit
                    / max(1, len(self.workers)),
                    max(
                        -1.0,
                        min(
                            1.0,
                            profile.horizon_slack_ticks / max(1, self.horizon_tick),
                        ),
                    ),
                ]
            )
        capability = EdgeStore(
            edge_index=capability_index.copy(),
            edge_features=np.asarray(
                capability_features, dtype=np.float32
            ).reshape(-1, 9),
            feature_names=(
                "processing_time_norm",
                "configuration_match",
                "earliest_start_time_norm",
                "resource_ready_time_norm",
                "predicted_finish_time_norm",
                "safe_disassembly_worker_ratio",
                "safe_installation_worker_ratio",
                "matching_deficit_after_commit_norm",
                "horizon_slack_norm",
            ),
            bidirectional=True,
        )

        locked = self._build_locked_edges()

        installation_index = self._static_edge_indices[CAN_INSTALL_EDGE]
        installation = EdgeStore(
            edge_index=installation_index.copy(),
            edge_features=np.ones(
                (installation_index.shape[1], 1), dtype=np.float32
            ),
            feature_names=("qualified",),
        )

        disassembly_pairs: list[tuple[int, int]] = []
        disassembly_features: list[list[float]] = []
        for worker_index, worker in enumerate(self.workers):
            for machine_index, machine in enumerate(self.machines):
                if (
                    machine.current_module == self.instance.no_module_state
                    or machine.current_module
                    not in worker.spec.qualified_modules
                ):
                    continue
                disassembly_pairs.append((worker_index, machine_index))
                disassembly_features.append(
                    [
                        1.0,
                        self._worker_disassembly_ticks(machine, worker)
                        / self.horizon_tick,
                    ]
                )
        disassembly = EdgeStore(
            edge_index=_as_edge_index(disassembly_pairs),
            edge_features=np.asarray(
                disassembly_features, dtype=np.float32
            ).reshape(-1, 2),
            feature_names=("qualified", "disassembly_time_norm"),
        )

        return {
            PRECEDES_EDGE: precedence,
            CAPABLE_EDGE: capability,
            LOCKED_EDGE: locked,
            CAN_INSTALL_EDGE: installation,
            CAN_DISASSEMBLE_EDGE: disassembly,
        }

    def _build_locked_edges(self) -> EdgeStore:
        self._require_instance()
        module_values = (self.instance.no_module_state, *self.instance.modules)
        feature_names = (
            tuple(f"source_module_{module}" for module in module_values)
            + tuple(f"target_module_{module}" for module in module_values)
            + tuple(f"stage_{stage.value}" for stage in ACTIVE_RECONFIGURATION_STAGES)
            + ("stage_elapsed_time_norm",)
        )
        records: list[tuple[tuple[int, int], list[float]]] = []
        for reconfiguration in self.reconfigurations.values():
            if reconfiguration.stage not in ACTIVE_RECONFIGURATION_STAGES:
                continue
            pair = (
                self.instance.operation_index[reconfiguration.operation_id],
                self.instance.machine_index[reconfiguration.machine_id],
            )
            source_one_hot = [
                float(reconfiguration.source_module == module)
                for module in module_values
            ]
            target_one_hot = [
                float(reconfiguration.target_module == module)
                for module in module_values
            ]
            stage_one_hot = [
                float(reconfiguration.stage == stage)
                for stage in ACTIVE_RECONFIGURATION_STAGES
            ]
            elapsed = (
                self.current_tick
                - self._reconfiguration_stage_start_tick(reconfiguration)
            ) / self.horizon_tick
            records.append(
                (
                    pair,
                    source_one_hot
                    + target_one_hot
                    + stage_one_hot
                    + [elapsed],
                )
            )
        records.sort(key=lambda record: record[0])
        return EdgeStore(
            edge_index=_as_edge_index([pair for pair, _ in records]),
            edge_features=np.asarray(
                [features for _, features in records], dtype=np.float32
            ).reshape(-1, len(feature_names)),
            feature_names=feature_names,
            bidirectional=True,
        )

    def get_action_mask(self) -> np.ndarray:
        """Return True for illegal actions and False for feasible actions."""
        if self.decision_type == DecisionType.TERMINAL:
            return np.ones(1, dtype=bool)
        if self.decision_type == DecisionType.PRODUCTION:
            mask = np.ones(self.production_action_size, dtype=bool)
            for operation_index, operation in enumerate(self.operations):
                if operation.state != OperationState.READY:
                    continue
                for machine_index, machine in enumerate(self.machines):
                    if (
                        machine.state == MachineState.IDLE
                        and machine.current_module != self.instance.no_module_state
                        and operation.spec.required_module
                        in machine.spec.module_parameters
                    ):
                        direct = (
                            machine.current_module
                            == operation.spec.required_module
                        )
                        admissible = direct or not self.matching_admission_enabled
                        if not direct and self.matching_admission_enabled:
                            key = (
                                self.current_tick,
                                operation_index,
                                machine_index,
                            )
                            self._resource_admission_candidates.add(key)
                            admissible = self._production_candidate_profile(
                                operation_index,
                                machine_index,
                            ).admissible
                            if not admissible:
                                self._resource_admission_masked.add(key)
                        if admissible:
                            mask[
                                self.encode_production_action(
                                    operation_index, machine_index
                                )
                            ] = False
            if self._production_advance_allowed():
                mask[-1] = False
            return mask
        mask = np.ones(self.worker_action_size, dtype=bool)
        legal_worker_pairs = 0
        for machine_index, machine in enumerate(self.machines):
            reconfiguration = self._pending_reconfiguration(machine.spec.id)
            if reconfiguration is None:
                continue
            for worker_index, worker in enumerate(self.workers):
                legal = self._worker_can_start(reconfiguration, worker)
                if (
                    legal
                    and self.matching_admission_enabled
                    and self._resource_setting(
                        "preserve_matching_on_worker_action", True
                    )
                ):
                    legal = self._worker_action_preserves_matching(
                        reconfiguration,
                        worker_index,
                    )
                if legal:
                    mask[
                        self.encode_worker_action(machine_index, worker_index)
                    ] = False
                    legal_worker_pairs += 1
                    self._matching_preserving_worker_actions.add(
                        (
                            self.current_tick,
                            reconfiguration.id,
                            reconfiguration.stage.value,
                            worker.spec.id,
                        )
                    )
        non_delay = bool(
            self.matching_admission_enabled
            and self._resource_setting("non_delay_worker_dispatch", True)
        )
        if self._has_strict_future() and not (
            non_delay and legal_worker_pairs > 0
        ):
            mask[-1] = False
        return mask

    def encode_production_action(
        self, operation_index: int, machine_index: int
    ) -> int:
        return operation_index * len(self.machines) + machine_index

    def decode_production_action(self, action: int) -> tuple[int, int]:
        if action < 0 or action >= self.production_action_size - 1:
            raise ValueError("not a production pair action")
        return divmod(action, len(self.machines))

    def encode_worker_action(self, machine_index: int, worker_index: int) -> int:
        return machine_index * len(self.workers) + worker_index

    def decode_worker_action(self, action: int) -> tuple[int, int]:
        if action < 0 or action >= self.worker_action_size - 1:
            raise ValueError("not a worker pair action")
        return divmod(action, len(self.workers))

    def step(
        self, action: int
    ) -> tuple[Observation, RewardVector, bool, bool, dict[str, Any]]:
        if self.decision_type == DecisionType.TERMINAL:
            raise RuntimeError("cannot step a terminal environment")
        mask = self.get_action_mask()
        if action < 0 or action >= len(mask) or mask[action]:
            raise ValueError(
                f"illegal {self.decision_type.value} action {action} at t={self.current_time}"
            )
        before_tick = self.current_tick
        before = self._objective_vector()
        completed_orders_before = len(self._order_completion_tick)
        potential_before = self.feasibility_potential()
        quality_before = bounded_quality_score(
            *before,
            self.config["reward"],
        )
        phase = self.decision_type
        if phase == DecisionType.WORKER:
            self._record_worker_pressure_snapshot()
        self._invalidate_resource_snapshot()
        if phase == DecisionType.PRODUCTION:
            if action == self.advance_action:
                self.decision_type = DecisionType.WORKER
            else:
                operation_index, machine_index = self.decode_production_action(action)
                self._execute_production_action(operation_index, machine_index)
        else:
            if action == self.advance_action:
                self._advance_to_next_event()
            else:
                machine_index, worker_index = self.decode_worker_action(action)
                self._execute_worker_action(machine_index, worker_index)
        self._decision_count += 1
        if self.current_tick == before_tick:
            self._zero_time_actions += 1
        else:
            self._zero_time_actions = 0
        if (
            self._decision_count >= self.config["environment"]["max_decisions"]
            or self._zero_time_actions
            >= self.config["environment"]["max_zero_time_actions"]
        ):
            self._apply_truncation("decision_limit")
        self._resolve_terminal_or_deadlock()
        self._invalidate_resource_snapshot()
        after = self._objective_vector()
        completed_orders_after = len(self._order_completion_tick)
        quality_after = bounded_quality_score(
            *after,
            self.config["reward"],
        )
        shaping_config = self.config["reward"].get(
            "feasibility_shaping", {}
        )
        shaping_enabled = bool(shaping_config.get("enabled", False))
        shaping_coefficient = float(shaping_config.get("coefficient", 0.0))
        if shaping_coefficient < 0.0:
            raise ValueError("feasibility shaping coefficient must be non-negative")
        potential_after = self.feasibility_potential()
        feasibility_shaping = (
            shaping_coefficient * (potential_after - potential_before)
            if shaping_enabled
            else 0.0
        )
        reward = RewardVector(
            flow=-(after[0] - before[0]),
            cost=-(after[1] - before[1]),
            variance=-(after[2] - before[2]),
            completion_progress=(
                (completed_orders_after - completed_orders_before)
                / len(self.instance.orders)
            ),
            completion_bonus=float(self.terminated and not self.truncated),
            quality=-(quality_after - quality_before),
            truncation=-float(self.truncated),
            unfinished=(
                -(
                    len(self.instance.orders) - completed_orders_after
                )
                / len(self.instance.orders)
                if self.truncated
                else 0.0
            ),
            feasibility_shaping=feasibility_shaping,
        )
        self._cumulative_reward += np.asarray(
            [reward.flow, reward.cost, reward.variance], dtype=np.float64
        )
        info = {
            "time": self.current_time,
            "decision_type": self.decision_type.value,
            "action_phase": phase.value,
            "terminal_reason": self.terminal_reason,
        }
        return self.observe(), reward, self.terminated, self.truncated, info

    def estimate_processing_ticks(
        self, operation_index: int, machine_index: int
    ) -> int:
        operation = self.operations[operation_index]
        machine = self.machines[machine_index]
        module_parameters = machine.spec.module_parameters[
            operation.spec.required_module
        ]
        minutes = (
            operation.spec.base_processing_time
            * module_parameters.processing_speed_factor
        )
        return max(1, quantize_to_ticks(minutes, self.resolution))

    def estimate_earliest_start_tick(
        self, operation_index: int, machine_index: int
    ) -> int:
        """Optimistic tick at which an operation can start on a machine.

        Existing machine commitments are honored. Future worker contention,
        worker busy states, and fatigue recovery are deliberately ignored.
        """
        operation = self.operations[operation_index]
        machine = self.machines[machine_index]
        reconfiguration = self._active_reconfiguration(machine.spec.id)
        if (
            reconfiguration is not None
            and reconfiguration.operation_id == operation.spec.id
        ):
            return (
                self.current_tick
                + self._remaining_reconfiguration_ticks(reconfiguration)
            )
        available_tick, available_module = self._machine_committed_release(
            machine_index
        )
        return available_tick + self._optimistic_reconfiguration_ticks(
            machine,
            available_module,
            operation.spec.required_module,
        )

    def estimate_reconfiguration_ticks(
        self, operation_index: int, machine_index: int
    ) -> int:
        operation = self.operations[operation_index]
        machine = self.machines[machine_index]
        return self._optimistic_reconfiguration_ticks(
            machine,
            machine.current_module,
            operation.spec.required_module,
        )

    def _machine_committed_release(
        self, machine_index: int
    ) -> tuple[int, str]:
        machine = self.machines[machine_index]
        if machine.state == MachineState.IDLE:
            return self.current_tick, machine.current_module
        if machine.state == MachineState.PROCESSING:
            return (
                max(self.current_tick, machine.busy_until_tick or self.current_tick),
                machine.current_module,
            )
        reconfiguration = self._active_reconfiguration(machine.spec.id)
        if reconfiguration is None:
            raise RuntimeError(
                f"machine {machine.spec.id} has no active reconfiguration"
            )
        processing_start = (
            self.current_tick
            + self._remaining_reconfiguration_ticks(reconfiguration)
        )
        locked_operation_index = self.instance.operation_index[
            reconfiguration.operation_id
        ]
        processing_ticks = self.estimate_processing_ticks(
            locked_operation_index, machine_index
        )
        return (
            processing_start + processing_ticks,
            reconfiguration.target_module,
        )

    def _remaining_reconfiguration_ticks(
        self, reconfiguration: ReconfigurationRuntime
    ) -> int:
        machine = self._machine_by_id(reconfiguration.machine_id)
        if reconfiguration.stage == ReconfigurationStage.WAIT_DIS:
            return self._optimistic_module_stage_ticks(
                machine, reconfiguration.source_module, installation=False
            ) + self._optimistic_module_stage_ticks(
                machine, reconfiguration.target_module, installation=True
            )
        if reconfiguration.stage == ReconfigurationStage.DIS:
            return max(
                0, (machine.busy_until_tick or self.current_tick) - self.current_tick
            ) + self._optimistic_module_stage_ticks(
                machine, reconfiguration.target_module, installation=True
            )
        if reconfiguration.stage == ReconfigurationStage.WAIT_INS:
            return self._optimistic_module_stage_ticks(
                machine, reconfiguration.target_module, installation=True
            )
        if reconfiguration.stage == ReconfigurationStage.INS:
            return max(
                0, (machine.busy_until_tick or self.current_tick) - self.current_tick
            )
        if reconfiguration.stage == ReconfigurationStage.DONE:
            return 0
        raise ValueError(
            f"unsupported reconfiguration stage {reconfiguration.stage}"
        )

    def _optimistic_reconfiguration_ticks(
        self,
        machine: MachineRuntime,
        source_module: str,
        target_module: str,
    ) -> int:
        if source_module == target_module:
            return 0
        ticks = 0
        if source_module != self.instance.no_module_state:
            ticks += self._optimistic_module_stage_ticks(
                machine, source_module, installation=False
            )
        if target_module != self.instance.no_module_state:
            ticks += self._optimistic_module_stage_ticks(
                machine, target_module, installation=True
            )
        return ticks

    def _optimistic_module_stage_ticks(
        self,
        machine: MachineRuntime,
        module: str,
        *,
        installation: bool,
    ) -> int:
        qualified_workers = [
            worker
            for worker in self.workers
            if module in worker.spec.qualified_modules
        ]
        if not qualified_workers:
            raise RuntimeError(f"module {module} has no qualified worker")
        fatigue = min(worker.fatigue for worker in qualified_workers)
        module_parameters = machine.spec.module_parameters[module]
        if installation:
            base = module_parameters.installation_base_time
            coefficient = self.instance.fatigue.installation_time_coefficient
        else:
            base = module_parameters.disassembly_base_time
            coefficient = self.instance.fatigue.disassembly_time_coefficient
        return max(
            1,
            quantize_to_ticks(
                base * (1.0 + coefficient * fatigue), self.resolution
            ),
        )

    def _worker_disassembly_ticks(
        self, machine: MachineRuntime, worker: WorkerRuntime
    ) -> int:
        module_parameters = machine.spec.module_parameters[
            machine.current_module
        ]
        minutes = module_parameters.disassembly_base_time * (
            1.0
            + self.instance.fatigue.disassembly_time_coefficient
            * worker.fatigue
        )
        return max(1, quantize_to_ticks(minutes, self.resolution))

    def _reconfiguration_stage_start_tick(
        self, reconfiguration: ReconfigurationRuntime
    ) -> int:
        starts = {
            ReconfigurationStage.WAIT_DIS: reconfiguration.lock_tick,
            ReconfigurationStage.DIS: reconfiguration.disassembly_start_tick,
            ReconfigurationStage.WAIT_INS: reconfiguration.disassembly_end_tick,
            ReconfigurationStage.INS: reconfiguration.installation_start_tick,
        }
        start_tick = starts.get(reconfiguration.stage)
        if start_tick is None:
            raise RuntimeError(
                f"missing start tick for stage {reconfiguration.stage}"
            )
        return start_tick

    def _active_reconfiguration(
        self, machine_id: str
    ) -> ReconfigurationRuntime | None:
        reconfiguration_id = self._machine_reconfiguration.get(machine_id)
        if reconfiguration_id is None:
            return None
        reconfiguration = self.reconfigurations[reconfiguration_id]
        if reconfiguration.stage == ReconfigurationStage.DONE:
            return None
        return reconfiguration

    def projected_worker_fatigue(
        self, machine_index: int, worker_index: int
    ) -> float:
        machine = self.machines[machine_index]
        reconfiguration = self._pending_reconfiguration(machine.spec.id)
        if reconfiguration is None:
            return math.inf
        worker = self.workers[worker_index]
        duration_ticks = self._stage_duration_ticks(reconfiguration, worker)
        rate = self._stage_accumulation_rate(reconfiguration)
        return worker.fatigue + rate * ticks_to_minutes(
            duration_ticks, self.resolution
        )

    def metrics(self) -> dict[str, Any]:
        self._require_instance()
        completed_orders = len(self._order_completion_tick)
        completed_operations = sum(
            operation.state == OperationState.DONE for operation in self.operations
        )
        completion_times = {
            order.id: (
                ticks_to_minutes(self._order_completion_tick[order.id], self.resolution)
                if order.id in self._order_completion_tick
                else None
            )
            for order in self.instance.orders
        }
        completed_reconfigurations = [
            value
            for value in self.reconfigurations.values()
            if value.stage == ReconfigurationStage.DONE
        ]
        switches = [
            int(
                value.installation_worker_id is not None
                and value.disassembly_worker_id != value.installation_worker_id
            )
            for value in completed_reconfigurations
        ]
        switch_ratio = (
            float(sum(switches) / len(switches)) if switches else None
        )
        return {
            "instance_id": self.instance.instance_id,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "terminal_reason": self.terminal_reason,
            "time": self.current_time,
            "completed_orders": completed_orders,
            "total_orders": len(self.instance.orders),
            "completed_operations": completed_operations,
            "total_operations": len(self.operations),
            "unfinished_orders": len(self.instance.orders) - completed_orders,
            "total_flow_time": self._flow_integral
            if self.terminated
            else None,
            "censored_flow_time": self._flow_integral,
            "flow_time_objective": self._flow_integral + self._flow_penalty,
            "reconfiguration_cost": self._reconfiguration_cost,
            "worker_load_variance": self._load_variance(),
            "maximum_worker_fatigue": self._maximum_fatigue_seen,
            "safe_fatigue_limit": (
                self.instance.fatigue.maximum_safe_fatigue
            ),
            "worker_peak_fatigue": {
                worker.spec.id: worker.peak_fatigue for worker in self.workers
            },
            "mean_peak_worker_fatigue": (
                float(np.mean([worker.peak_fatigue for worker in self.workers]))
                if self.workers
                else 0.0
            ),
            "fatigue_masked_action_count": len(self._fatigue_masked_actions),
            "fatigue_masked_action_ratio": (
                len(self._fatigue_masked_actions)
                / len(self._fatigue_candidate_actions)
                if self._fatigue_candidate_actions
                else 0.0
            ),
            "worker_competition_event_count": len(
                self._worker_competition_ticks
            ),
            "worker_matching_deficit_event_count": len(
                self._worker_matching_deficit_ticks
            ),
            "resource_admission_masked_action_count": len(
                self._resource_admission_masked
            ),
            "resource_admission_masked_action_ratio": (
                len(self._resource_admission_masked)
                / len(self._resource_admission_candidates)
                if self._resource_admission_candidates
                else 0.0
            ),
            "minimum_worker_alternatives": (
                self._minimum_worker_alternatives_seen
                if self._minimum_worker_alternatives_seen is not None
                else len(self.workers)
            ),
            "matching_preserving_worker_action_count": len(
                self._matching_preserving_worker_actions
            ),
            "candidate_recovery_advance_count": (
                self._candidate_recovery_advance_count
            ),
            "machine_waiting_for_worker_time": (
                self._machine_waiting_for_worker_time()
            ),
            "completed_reconfigurations": len(completed_reconfigurations),
            "worker_switch_indicators": switches,
            "worker_switch_ratio": switch_ratio,
            "completion_times": completion_times,
            "cumulative_reward": {
                "flow": float(self._cumulative_reward[0]),
                "cost": float(self._cumulative_reward[1]),
                "variance": float(self._cumulative_reward[2]),
            },
            "quality_score": bounded_quality_score(
                self._flow_integral + self._flow_penalty,
                self._reconfiguration_cost,
                self._load_variance(),
                self.config["reward"],
            ),
        }

    def validate_schedule(self) -> list[str]:
        """Return invariant violations; an empty list means the rollout is feasible."""
        violations: list[str] = []
        by_machine: dict[str, list[tuple[float, float, str]]] = {
            machine.spec.id: [] for machine in self.machines
        }
        by_worker: dict[str, list[tuple[float, float, str]]] = {
            worker.spec.id: [] for worker in self.workers
        }
        for record in self.schedule_log:
            by_machine[record["machine_id"]].append(
                (record["start"], record["end"], record["operation_id"])
            )
        for record in self.reconfiguration_log:
            by_machine[record["machine_id"]].append(
                (record["start"], record["end"], record["stage"])
            )
            by_worker[record["worker_id"]].append(
                (record["start"], record["end"], record["stage"])
            )
        for resource, intervals in (*by_machine.items(), *by_worker.items()):
            ordered = sorted(intervals)
            for previous, current in zip(ordered, ordered[1:]):
                if current[0] < previous[1] - EPSILON:
                    violations.append(
                        f"{resource} overlap: {previous[2]} and {current[2]}"
                    )
        runtime_by_id = {operation.spec.id: operation for operation in self.operations}
        for order in self.instance.orders:
            for predecessor, successor in zip(order.operations, order.operations[1:]):
                before = runtime_by_id[predecessor.id]
                after = runtime_by_id[successor.id]
                if (
                    before.end_tick is not None
                    and after.start_tick is not None
                    and after.start_tick < before.end_tick
                ):
                    violations.append(
                        f"precedence violated: {predecessor.id} -> {successor.id}"
                    )
        for worker in self.workers:
            if worker.fatigue > self.instance.fatigue.maximum_safe_fatigue + EPSILON:
                violations.append(f"{worker.spec.id} fatigue exceeds safety bound")
        return violations

    def _record_worker_pressure_snapshot(self) -> None:
        pending = [
            value
            for value in self.reconfigurations.values()
            if value.stage
            in {
                ReconfigurationStage.WAIT_DIS,
                ReconfigurationStage.WAIT_INS,
            }
        ]
        safe_edges: list[list[int]] = []
        for reconfiguration in pending:
            module = (
                reconfiguration.source_module
                if reconfiguration.stage == ReconfigurationStage.WAIT_DIS
                else reconfiguration.target_module
            )
            task_edges: list[int] = []
            for worker_index, worker in enumerate(self.workers):
                if (
                    worker.state != WorkerState.IDLE
                    or module not in worker.spec.qualified_modules
                ):
                    continue
                key = (
                    self.current_tick,
                    reconfiguration.id,
                    reconfiguration.stage.value,
                    worker.spec.id,
                )
                self._fatigue_candidate_actions.add(key)
                if self._worker_can_start(reconfiguration, worker):
                    task_edges.append(worker_index)
                else:
                    self._fatigue_masked_actions.add(key)
            safe_edges.append(task_edges)
        if pending and _maximum_matching_size(
            safe_edges, len(self.workers)
        ) < len(pending):
            self._worker_competition_ticks.add(self.current_tick)

    def _machine_waiting_for_worker_time(self) -> float:
        wait_ticks = 0
        for reconfiguration in self.reconfigurations.values():
            disassembly_start = (
                reconfiguration.disassembly_start_tick
                if reconfiguration.disassembly_start_tick is not None
                else self.current_tick
            )
            wait_ticks += max(
                0, disassembly_start - reconfiguration.lock_tick
            )
            if reconfiguration.disassembly_end_tick is not None:
                installation_start = (
                    reconfiguration.installation_start_tick
                    if reconfiguration.installation_start_tick is not None
                    else self.current_tick
                )
                wait_ticks += max(
                    0,
                    installation_start
                    - reconfiguration.disassembly_end_tick,
                )
        return ticks_to_minutes(wait_ticks, self.resolution)

    def _execute_production_action(
        self, operation_index: int, machine_index: int
    ) -> None:
        operation = self.operations[operation_index]
        machine = self.machines[machine_index]
        if machine.current_module == operation.spec.required_module:
            self._start_processing(operation, machine)
            return
        reconfiguration_id = f"REC_{len(self.reconfigurations) + 1}"
        reconfiguration = ReconfigurationRuntime(
            id=reconfiguration_id,
            machine_id=machine.spec.id,
            operation_id=operation.spec.id,
            source_module=machine.current_module,
            target_module=operation.spec.required_module,
            lock_tick=self.current_tick,
        )
        self.reconfigurations[reconfiguration_id] = reconfiguration
        self._machine_reconfiguration[machine.spec.id] = reconfiguration_id
        operation.state = OperationState.LOCKED
        operation.machine_id = machine.spec.id
        machine.state = MachineState.WAIT_DIS
        machine.locked_operation_id = operation.spec.id
        machine.source_module = machine.current_module
        machine.target_module = operation.spec.required_module

    def _execute_worker_action(
        self, machine_index: int, worker_index: int
    ) -> None:
        machine = self.machines[machine_index]
        worker = self.workers[worker_index]
        reconfiguration = self._pending_reconfiguration(machine.spec.id)
        if reconfiguration is None:
            raise RuntimeError("worker action has no pending reconfiguration")
        duration_ticks = self._stage_duration_ticks(reconfiguration, worker)
        end_tick = self.current_tick + duration_ticks
        if reconfiguration.stage == ReconfigurationStage.WAIT_DIS:
            reconfiguration.stage = ReconfigurationStage.DIS
            reconfiguration.disassembly_worker_id = worker.spec.id
            reconfiguration.disassembly_start_tick = self.current_tick
            reconfiguration.disassembly_end_tick = end_tick
            worker.state = WorkerState.DIS
            machine.state = MachineState.DIS
            fixed_cost = self.instance.module_costs[
                reconfiguration.source_module
            ].fixed_disassembly_cost
            event_type = EventType.DIS_COMPLETE
            stage_name = "DIS"
        else:
            reconfiguration.stage = ReconfigurationStage.INS
            reconfiguration.installation_worker_id = worker.spec.id
            reconfiguration.installation_start_tick = self.current_tick
            reconfiguration.installation_end_tick = end_tick
            worker.state = WorkerState.INS
            machine.state = MachineState.INS
            fixed_cost = self.instance.module_costs[
                reconfiguration.target_module
            ].fixed_installation_cost
            event_type = EventType.INS_COMPLETE
            stage_name = "INS"
        worker.busy_until_tick = end_tick
        machine.busy_until_tick = end_tick
        self._reconfiguration_cost += fixed_cost
        self._push_event(
            end_tick,
            event_type,
            {
                "reconfiguration_id": reconfiguration.id,
                "worker_id": worker.spec.id,
            },
        )
        self.reconfiguration_log.append(
            {
                "reconfiguration_id": reconfiguration.id,
                "operation_id": reconfiguration.operation_id,
                "machine_id": machine.spec.id,
                "worker_id": worker.spec.id,
                "stage": stage_name,
                "source_module": reconfiguration.source_module,
                "target_module": reconfiguration.target_module,
                "start": self.current_time,
                "end": ticks_to_minutes(end_tick, self.resolution),
                "duration": ticks_to_minutes(duration_ticks, self.resolution),
                "fixed_cost": fixed_cost,
            }
        )

    def _start_processing(
        self, operation: OperationRuntime, machine: MachineRuntime
    ) -> None:
        operation_index = self.operations.index(operation)
        machine_index = self.machines.index(machine)
        duration_ticks = self.estimate_processing_ticks(
            operation_index, machine_index
        )
        end_tick = self.current_tick + duration_ticks
        operation.state = OperationState.PROCESSING
        operation.machine_id = machine.spec.id
        operation.start_tick = self.current_tick
        machine.state = MachineState.PROCESSING
        machine.busy_until_tick = end_tick
        self._push_event(
            end_tick,
            EventType.PROCESS_COMPLETE,
            {
                "operation_id": operation.spec.id,
                "machine_id": machine.spec.id,
            },
        )
        self.schedule_log.append(
            {
                "order_id": operation.spec.order_id,
                "operation_id": operation.spec.id,
                "sequence": operation.spec.sequence,
                "required_module": operation.spec.required_module,
                "machine_id": machine.spec.id,
                "start": self.current_time,
                "end": ticks_to_minutes(end_tick, self.resolution),
                "duration": ticks_to_minutes(duration_ticks, self.resolution),
            }
        )

    def _advance_to_next_event(self) -> None:
        event_ticks = [event[0] for event in self._events if event[0] > self.current_tick]
        recovery_tick = self._earliest_recovery_tick()
        candidate_recovery_tick = self._earliest_candidate_recovery_tick()
        candidates = event_ticks + (
            [recovery_tick] if recovery_tick is not None else []
        ) + (
            [candidate_recovery_tick]
            if candidate_recovery_tick is not None
            else []
        )
        if not candidates:
            self._truncate_at_horizon("deadlock")
            return
        next_tick = min(candidates)
        if (
            candidate_recovery_tick is not None
            and next_tick == candidate_recovery_tick
        ):
            self._candidate_recovery_advance_count += 1
        if next_tick > self.horizon_tick:
            self._truncate_at_horizon("horizon")
            return
        self._advance_interval(next_tick)
        self.current_tick = next_tick
        self._process_events_at_current_tick()
        self._invalidate_resource_snapshot()
        self.decision_type = DecisionType.PRODUCTION

    def _advance_interval(self, next_tick: int) -> None:
        delta_ticks = next_tick - self.current_tick
        if delta_ticks <= 0:
            return
        delta = ticks_to_minutes(delta_ticks, self.resolution)
        active_orders = sum(
            self._order_released[order.id]
            and order.id not in self._order_completion_tick
            for order in self.instance.orders
        )
        self._flow_integral += active_orders * delta
        downtime_states = {
            MachineState.WAIT_DIS,
            MachineState.DIS,
            MachineState.WAIT_INS,
            MachineState.INS,
        }
        self._reconfiguration_cost += delta * sum(
            machine.spec.downtime_cost_per_minute
            for machine in self.machines
            if machine.state in downtime_states
        )
        self._reconfiguration_cost += delta * sum(
            worker.spec.labor_cost_per_minute
            for worker in self.workers
            if worker.state in {WorkerState.DIS, WorkerState.INS}
        )
        recovery = self.instance.fatigue.idle_recovery_rate_per_minute * delta
        for worker in self.workers:
            if worker.state == WorkerState.IDLE:
                worker.fatigue = max(0.0, worker.fatigue - recovery)

    def _process_events_at_current_tick(self) -> None:
        current_events: list[tuple[int, int, int, EventType, dict[str, Any]]] = []
        while self._events and self._events[0][0] == self.current_tick:
            current_events.append(heapq.heappop(self._events))
        for _, _, _, event_type, payload in sorted(
            current_events, key=lambda value: (value[1], value[2])
        ):
            if event_type == EventType.ORDER_RELEASE:
                self._release_order(payload["order_id"])
            elif event_type == EventType.PROCESS_COMPLETE:
                self._complete_processing(payload["operation_id"], payload["machine_id"])
            elif event_type == EventType.DIS_COMPLETE:
                self._complete_disassembly(
                    payload["reconfiguration_id"], payload["worker_id"]
                )
            elif event_type == EventType.INS_COMPLETE:
                self._complete_installation(
                    payload["reconfiguration_id"], payload["worker_id"]
                )
        self._refresh_ready_operations()

    def _release_order(self, order_id: str) -> None:
        self._order_released[order_id] = True
        order = self._order_by_id(order_id)
        for index, operation in enumerate(order.operations):
            runtime = self._operation_by_id(operation.id)
            runtime.state = (
                OperationState.READY
                if index == 0
                else OperationState.BLOCKED
            )

    def _complete_processing(self, operation_id: str, machine_id: str) -> None:
        operation = self._operation_by_id(operation_id)
        machine = self._machine_by_id(machine_id)
        operation.state = OperationState.DONE
        operation.end_tick = self.current_tick
        machine.state = MachineState.IDLE
        machine.busy_until_tick = None
        machine.locked_operation_id = None
        machine.source_module = None
        machine.target_module = None
        order = self._order_by_id(operation.spec.order_id)
        if all(
            self._operation_by_id(spec.id).state == OperationState.DONE
            for spec in order.operations
        ):
            self._order_completion_tick[order.id] = self.current_tick

    def _complete_disassembly(
        self, reconfiguration_id: str, worker_id: str
    ) -> None:
        reconfiguration = self.reconfigurations[reconfiguration_id]
        machine = self._machine_by_id(reconfiguration.machine_id)
        worker = self._worker_by_id(worker_id)
        duration = ticks_to_minutes(
            reconfiguration.disassembly_end_tick
            - reconfiguration.disassembly_start_tick,
            self.resolution,
        )
        worker.fatigue = min(
            1.0,
            worker.fatigue
            + self.instance.fatigue.disassembly_accumulation_rate_per_minute
            * duration,
        )
        worker.peak_fatigue = max(worker.peak_fatigue, worker.fatigue)
        self._maximum_fatigue_seen = max(
            self._maximum_fatigue_seen, worker.fatigue
        )
        worker.load += duration
        worker.state = WorkerState.IDLE
        worker.busy_until_tick = None
        machine.current_module = self.instance.no_module_state
        machine.state = MachineState.WAIT_INS
        machine.busy_until_tick = None
        reconfiguration.stage = ReconfigurationStage.WAIT_INS

    def _complete_installation(
        self, reconfiguration_id: str, worker_id: str
    ) -> None:
        reconfiguration = self.reconfigurations[reconfiguration_id]
        machine = self._machine_by_id(reconfiguration.machine_id)
        worker = self._worker_by_id(worker_id)
        duration = ticks_to_minutes(
            reconfiguration.installation_end_tick
            - reconfiguration.installation_start_tick,
            self.resolution,
        )
        worker.fatigue = min(
            1.0,
            worker.fatigue
            + self.instance.fatigue.installation_accumulation_rate_per_minute
            * duration,
        )
        worker.peak_fatigue = max(worker.peak_fatigue, worker.fatigue)
        self._maximum_fatigue_seen = max(
            self._maximum_fatigue_seen, worker.fatigue
        )
        worker.load += duration
        worker.state = WorkerState.IDLE
        worker.busy_until_tick = None
        machine.current_module = reconfiguration.target_module
        machine.state = MachineState.IDLE
        machine.busy_until_tick = None
        reconfiguration.stage = ReconfigurationStage.DONE
        self._machine_reconfiguration.pop(machine.spec.id, None)
        locked_operation = self._operation_by_id(reconfiguration.operation_id)
        self._start_processing(locked_operation, machine)

    def _refresh_ready_operations(self) -> None:
        for order in self.instance.orders:
            if not self._order_released[order.id]:
                continue
            for index, operation_spec in enumerate(order.operations):
                operation = self._operation_by_id(operation_spec.id)
                if operation.state != OperationState.BLOCKED:
                    continue
                if index == 0:
                    operation.state = OperationState.READY
                else:
                    predecessor = self._operation_by_id(order.operations[index - 1].id)
                    if predecessor.state == OperationState.DONE:
                        operation.state = OperationState.READY

    def _current_worker_tasks(self) -> tuple[WorkerTaskSnapshot, ...]:
        tasks: list[WorkerTaskSnapshot] = []
        for reconfiguration in self.reconfigurations.values():
            if reconfiguration.stage not in {
                ReconfigurationStage.WAIT_DIS,
                ReconfigurationStage.WAIT_INS,
            }:
                continue
            module = (
                reconfiguration.source_module
                if reconfiguration.stage == ReconfigurationStage.WAIT_DIS
                else reconfiguration.target_module
            )
            tasks.append(
                WorkerTaskSnapshot(
                    task_id=reconfiguration.id,
                    machine_index=self.instance.machine_index[
                        reconfiguration.machine_id
                    ],
                    stage=reconfiguration.stage,
                    module=module,
                )
            )
        return tuple(sorted(tasks, key=lambda value: value.task_id))

    def _task_reconfiguration(
        self,
        task: WorkerTaskSnapshot,
    ) -> ReconfigurationRuntime:
        if task.task_id in self.reconfigurations:
            return self.reconfigurations[task.task_id]
        machine = self.machines[task.machine_index]
        return ReconfigurationRuntime(
            id=task.task_id,
            machine_id=machine.spec.id,
            operation_id="",
            source_module=(
                task.module
                if task.stage == ReconfigurationStage.WAIT_DIS
                else self.instance.no_module_state
            ),
            target_module=(
                task.module
                if task.stage == ReconfigurationStage.WAIT_INS
                else self.instance.no_module_state
            ),
            lock_tick=self.current_tick,
            stage=task.stage,
        )

    def _safe_edges_for_tasks(
        self,
        tasks: tuple[WorkerTaskSnapshot, ...],
    ) -> tuple[tuple[int, ...], ...]:
        edges: list[tuple[int, ...]] = []
        for task in tasks:
            reconfiguration = self._task_reconfiguration(task)
            safe = tuple(
                worker_index
                for worker_index, worker in enumerate(self.workers)
                if self._worker_can_start(reconfiguration, worker)
            )
            edges.append(safe)
        return tuple(edges)

    def _resource_feasibility_snapshot(
        self,
    ) -> ResourceFeasibilitySnapshot:
        if self._resource_snapshot_cache is not None:
            return self._resource_snapshot_cache
        tasks = self._current_worker_tasks()
        safe_edges = self._safe_edges_for_tasks(tasks)
        matching_size = _maximum_matching_size(
            [list(edge) for edge in safe_edges],
            len(self.workers),
        )
        safe_idle_workers = tuple(
            worker_index
            for worker_index, worker in enumerate(self.workers)
            if worker.state == WorkerState.IDLE
        )
        minimum_alternatives = (
            min((len(edge) for edge in safe_edges), default=len(self.workers))
        )
        snapshot = ResourceFeasibilitySnapshot(
            tasks=tasks,
            safe_edges=safe_edges,
            matching_size=matching_size,
            safe_idle_workers=safe_idle_workers,
            minimum_worker_alternatives=minimum_alternatives,
        )
        self._resource_snapshot_cache = snapshot
        if tasks:
            if matching_size < len(tasks):
                self._worker_matching_deficit_ticks.add(self.current_tick)
            if self._minimum_worker_alternatives_seen is None:
                self._minimum_worker_alternatives_seen = minimum_alternatives
            else:
                self._minimum_worker_alternatives_seen = min(
                    self._minimum_worker_alternatives_seen,
                    minimum_alternatives,
                )
        return snapshot

    def _worker_action_preserves_matching(
        self,
        reconfiguration: ReconfigurationRuntime,
        worker_index: int,
    ) -> bool:
        snapshot = self._resource_feasibility_snapshot()
        task_index = next(
            (
                index
                for index, task in enumerate(snapshot.tasks)
                if task.task_id == reconfiguration.id
            ),
            None,
        )
        if task_index is None or worker_index not in snapshot.safe_edges[task_index]:
            return False
        remaining_edges = [
            [candidate for candidate in edge if candidate != worker_index]
            for index, edge in enumerate(snapshot.safe_edges)
            if index != task_index
        ]
        return _maximum_matching_size(
            remaining_edges,
            len(self.workers),
        ) == len(remaining_edges)

    def _worker_fatigue_at_availability(
        self,
        worker_index: int,
    ) -> tuple[int, float]:
        worker = self.workers[worker_index]
        if worker.state == WorkerState.IDLE:
            return self.current_tick, worker.fatigue
        available_tick = worker.busy_until_tick
        if available_tick is None:
            return self.current_tick, worker.fatigue
        accumulation = 0.0
        for reconfiguration in self.reconfigurations.values():
            if (
                reconfiguration.stage == ReconfigurationStage.DIS
                and reconfiguration.disassembly_worker_id == worker.spec.id
                and reconfiguration.disassembly_start_tick is not None
                and reconfiguration.disassembly_end_tick is not None
            ):
                accumulation = (
                    self.instance.fatigue.disassembly_accumulation_rate_per_minute
                    * ticks_to_minutes(
                        reconfiguration.disassembly_end_tick
                        - reconfiguration.disassembly_start_tick,
                        self.resolution,
                    )
                )
                break
            if (
                reconfiguration.stage == ReconfigurationStage.INS
                and reconfiguration.installation_worker_id == worker.spec.id
                and reconfiguration.installation_start_tick is not None
                and reconfiguration.installation_end_tick is not None
            ):
                accumulation = (
                    self.instance.fatigue.installation_accumulation_rate_per_minute
                    * ticks_to_minutes(
                        reconfiguration.installation_end_tick
                        - reconfiguration.installation_start_tick,
                        self.resolution,
                    )
                )
                break
        return available_tick, min(1.0, worker.fatigue + accumulation)

    def _earliest_safe_stage_projection(
        self,
        machine_index: int,
        worker_index: int,
        module: str,
        *,
        installation: bool,
        earliest_tick: int,
    ) -> tuple[int, int] | None:
        key = (
            machine_index,
            worker_index,
            module,
            installation,
            int(earliest_tick),
        )
        if key in self._stage_projection_cache:
            return self._stage_projection_cache[key]
        worker = self.workers[worker_index]
        if module not in worker.spec.qualified_modules:
            self._stage_projection_cache[key] = None
            return None
        available_tick, available_fatigue = self._worker_fatigue_at_availability(
            worker_index
        )
        start_tick = max(int(earliest_tick), available_tick)
        recovery_rate = self.instance.fatigue.idle_recovery_rate_per_minute
        machine = self.machines[machine_index]
        stage = (
            ReconfigurationStage.WAIT_INS
            if installation
            else ReconfigurationStage.WAIT_DIS
        )
        temporary = ReconfigurationRuntime(
            id="projection",
            machine_id=machine.spec.id,
            operation_id="",
            source_module=(
                self.instance.no_module_state if installation else module
            ),
            target_module=(
                module if installation else self.instance.no_module_state
            ),
            lock_tick=self.current_tick,
            stage=stage,
        )
        for tick in range(start_tick, self.horizon_tick + 1):
            recovery_minutes = ticks_to_minutes(
                max(0, tick - available_tick),
                self.resolution,
            )
            fatigue = max(
                0.0,
                available_fatigue - recovery_rate * recovery_minutes,
            )
            duration_ticks = self._stage_duration_ticks(
                temporary,
                worker,
                fatigue_override=fatigue,
            )
            predicted = fatigue + self._stage_accumulation_rate(
                temporary
            ) * ticks_to_minutes(duration_ticks, self.resolution)
            if (
                predicted
                <= self.instance.fatigue.maximum_safe_fatigue + EPSILON
                and tick + duration_ticks <= self.horizon_tick
            ):
                result = (tick, duration_ticks)
                self._stage_projection_cache[key] = result
                return result
        self._stage_projection_cache[key] = None
        return None

    def _production_candidate_profile(
        self,
        operation_index: int,
        machine_index: int,
    ) -> ProductionCandidateProfile:
        key = (int(operation_index), int(machine_index))
        if key in self._candidate_profile_cache:
            return self._candidate_profile_cache[key]
        operation = self.operations[operation_index]
        machine = self.machines[machine_index]
        processing_ticks = self.estimate_processing_ticks(
            operation_index,
            machine_index,
        )
        if machine.current_module == operation.spec.required_module:
            finish_tick = self.current_tick + processing_ticks
            profile = ProductionCandidateProfile(
                resource_ready_tick=self.current_tick,
                predicted_finish_tick=finish_tick,
                safe_disassembly_workers=len(self.workers),
                safe_installation_workers=len(self.workers),
                matching_deficit_after_commit=0,
                horizon_slack_ticks=self.horizon_tick - finish_tick,
                admissible=finish_tick <= self.horizon_tick,
            )
            self._candidate_profile_cache[key] = profile
            return profile

        candidate_task = WorkerTaskSnapshot(
            task_id=f"candidate:{operation_index}:{machine_index}",
            machine_index=machine_index,
            stage=ReconfigurationStage.WAIT_DIS,
            module=machine.current_module,
        )
        snapshot = self._resource_feasibility_snapshot()
        candidate_edges = self._safe_edges_for_tasks((candidate_task,))[0]
        combined_edges = [list(edge) for edge in snapshot.safe_edges]
        combined_edges.append(list(candidate_edges))
        matching_size = _maximum_matching_size(
            combined_edges,
            len(self.workers),
        )
        matching_deficit = len(combined_edges) - matching_size

        disassembly_projections = [
            projection
            for worker_index in range(len(self.workers))
            if (
                projection := self._earliest_safe_stage_projection(
                    machine_index,
                    worker_index,
                    machine.current_module,
                    installation=False,
                    earliest_tick=self.current_tick,
                )
            )
            is not None
        ]
        if not disassembly_projections:
            resource_ready_tick = self.horizon_tick + 1
            predicted_finish_tick = self.horizon_tick + 1
            safe_installation_workers = 0
        else:
            resource_ready_tick, disassembly_ticks = min(
                disassembly_projections,
                key=lambda value: (value[0] + value[1], value[0]),
            )
            disassembly_end = resource_ready_tick + disassembly_ticks
            installation_projections = [
                projection
                for worker_index in range(len(self.workers))
                if (
                    projection := self._earliest_safe_stage_projection(
                        machine_index,
                        worker_index,
                        operation.spec.required_module,
                        installation=True,
                        earliest_tick=disassembly_end,
                    )
                )
                is not None
            ]
            safe_installation_workers = sum(
                projection[0] == disassembly_end
                for projection in installation_projections
            )
            if installation_projections:
                installation_start, installation_ticks = min(
                    installation_projections,
                    key=lambda value: (value[0] + value[1], value[0]),
                )
                predicted_finish_tick = (
                    installation_start + installation_ticks + processing_ticks
                )
            else:
                predicted_finish_tick = self.horizon_tick + 1

        require_full_matching = self._resource_setting(
            "require_full_matching", True
        )
        admissible = bool(
            candidate_edges
            and (not require_full_matching or matching_deficit == 0)
            and resource_ready_tick == self.current_tick
            and predicted_finish_tick <= self.horizon_tick
        )
        profile = ProductionCandidateProfile(
            resource_ready_tick=resource_ready_tick,
            predicted_finish_tick=predicted_finish_tick,
            safe_disassembly_workers=len(candidate_edges),
            safe_installation_workers=safe_installation_workers,
            matching_deficit_after_commit=matching_deficit,
            horizon_slack_ticks=self.horizon_tick - predicted_finish_tick,
            admissible=admissible,
        )
        self._candidate_profile_cache[key] = profile
        return profile

    def _earliest_candidate_recovery_tick(self) -> int | None:
        if not (
            self.matching_admission_enabled
            and self._resource_setting("candidate_recovery_advance", True)
        ):
            return None
        candidates: list[int] = []
        for operation_index, operation in enumerate(self.operations):
            if operation.state != OperationState.READY:
                continue
            for machine_index, machine in enumerate(self.machines):
                if (
                    machine.state != MachineState.IDLE
                    or machine.current_module == self.instance.no_module_state
                    or operation.spec.required_module
                    not in machine.spec.module_parameters
                    or machine.current_module == operation.spec.required_module
                ):
                    continue
                profile = self._production_candidate_profile(
                    operation_index,
                    machine_index,
                )
                if (
                    self.current_tick < profile.resource_ready_tick
                    <= self.horizon_tick
                    and profile.predicted_finish_tick <= self.horizon_tick
                ):
                    candidates.append(profile.resource_ready_tick)
        return min(candidates) if candidates else None

    def feasibility_potential(self) -> float:
        if self.terminated or self.truncated:
            return 0.0
        completed = sum(
            operation.state == OperationState.DONE
            for operation in self.operations
        )
        operation_progress = completed / max(1, len(self.operations))
        snapshot = self._resource_feasibility_snapshot()
        resource_margin = (
            1.0
            if not snapshot.tasks
            else snapshot.minimum_worker_alternatives
            / max(1, len(self.workers))
        )
        slacks: list[float] = []
        for operation_index, operation in enumerate(self.operations):
            if operation.state != OperationState.READY:
                continue
            for machine_index, machine in enumerate(self.machines):
                if (
                    machine.state == MachineState.IDLE
                    and machine.current_module != self.instance.no_module_state
                    and operation.spec.required_module
                    in machine.spec.module_parameters
                ):
                    profile = self._production_candidate_profile(
                        operation_index,
                        machine_index,
                    )
                    slacks.append(
                        max(
                            0.0,
                            min(
                                1.0,
                                profile.horizon_slack_ticks
                                / max(1, self.horizon_tick),
                            ),
                        )
                    )
        slack_health = min(slacks) if slacks else 0.0
        weights = self.config.get("reward", {}).get(
            "feasibility_shaping", {}
        ).get(
            "weights",
            {
                "operation_progress": 0.50,
                "resource_margin": 0.25,
                "horizon_slack": 0.25,
            },
        )
        operation_weight = float(weights.get("operation_progress", 0.50))
        resource_weight = float(weights.get("resource_margin", 0.25))
        slack_weight = float(weights.get("horizon_slack", 0.25))
        if min(operation_weight, resource_weight, slack_weight) < 0.0:
            raise ValueError("feasibility shaping weights must be non-negative")
        if not math.isclose(
            operation_weight + resource_weight + slack_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("feasibility shaping weights must sum to one")
        return (
            operation_weight * operation_progress
            + resource_weight * resource_margin
            + slack_weight * slack_health
        )

    def _worker_can_start(
        self,
        reconfiguration: ReconfigurationRuntime,
        worker: WorkerRuntime,
    ) -> bool:
        if worker.state != WorkerState.IDLE:
            return False
        module = (
            reconfiguration.source_module
            if reconfiguration.stage == ReconfigurationStage.WAIT_DIS
            else reconfiguration.target_module
        )
        if module not in worker.spec.qualified_modules:
            return False
        duration_ticks = self._stage_duration_ticks(reconfiguration, worker)
        predicted = worker.fatigue + self._stage_accumulation_rate(
            reconfiguration
        ) * ticks_to_minutes(duration_ticks, self.resolution)
        return (
            predicted
            <= self.instance.fatigue.maximum_safe_fatigue + EPSILON
        )

    def _stage_duration_ticks(
        self,
        reconfiguration: ReconfigurationRuntime,
        worker: WorkerRuntime,
        *,
        fatigue_override: float | None = None,
    ) -> int:
        machine = self._machine_by_id(reconfiguration.machine_id)
        fatigue = worker.fatigue if fatigue_override is None else fatigue_override
        if reconfiguration.stage == ReconfigurationStage.WAIT_DIS:
            base = machine.spec.module_parameters[
                reconfiguration.source_module
            ].disassembly_base_time
            coefficient = self.instance.fatigue.disassembly_time_coefficient
        elif reconfiguration.stage == ReconfigurationStage.WAIT_INS:
            base = machine.spec.module_parameters[
                reconfiguration.target_module
            ].installation_base_time
            coefficient = self.instance.fatigue.installation_time_coefficient
        else:
            raise ValueError("duration requested for a non-waiting stage")
        return max(
            1,
            quantize_to_ticks(base * (1.0 + coefficient * fatigue), self.resolution),
        )

    def _stage_accumulation_rate(
        self, reconfiguration: ReconfigurationRuntime
    ) -> float:
        if reconfiguration.stage == ReconfigurationStage.WAIT_DIS:
            return (
                self.instance.fatigue.disassembly_accumulation_rate_per_minute
            )
        return self.instance.fatigue.installation_accumulation_rate_per_minute

    def _earliest_recovery_tick(self) -> int | None:
        pending = [
            value
            for value in self.reconfigurations.values()
            if value.stage
            in {ReconfigurationStage.WAIT_DIS, ReconfigurationStage.WAIT_INS}
        ]
        if not pending:
            return None
        recovery_rate = self.instance.fatigue.idle_recovery_rate_per_minute
        for tick in range(self.current_tick + 1, self.horizon_tick + 1):
            elapsed = ticks_to_minutes(tick - self.current_tick, self.resolution)
            for reconfiguration in pending:
                module = (
                    reconfiguration.source_module
                    if reconfiguration.stage == ReconfigurationStage.WAIT_DIS
                    else reconfiguration.target_module
                )
                for worker in self.workers:
                    if (
                        worker.state != WorkerState.IDLE
                        or module not in worker.spec.qualified_modules
                    ):
                        continue
                    fatigue = max(0.0, worker.fatigue - recovery_rate * elapsed)
                    duration_ticks = self._stage_duration_ticks(
                        reconfiguration,
                        worker,
                        fatigue_override=fatigue,
                    )
                    predicted = fatigue + self._stage_accumulation_rate(
                        reconfiguration
                    ) * ticks_to_minutes(duration_ticks, self.resolution)
                    if (
                        predicted
                        <= self.instance.fatigue.maximum_safe_fatigue + EPSILON
                    ):
                        return tick
        return None

    def _production_advance_allowed(self) -> bool:
        pending = any(
            value.stage
            in {ReconfigurationStage.WAIT_DIS, ReconfigurationStage.WAIT_INS}
            for value in self.reconfigurations.values()
        )
        return pending or self._has_strict_future()

    def _has_strict_future(self) -> bool:
        if any(event[0] > self.current_tick for event in self._events):
            return True
        if self._earliest_recovery_tick() is not None:
            return True
        return self._earliest_candidate_recovery_tick() is not None

    def _pending_reconfiguration(
        self, machine_id: str
    ) -> ReconfigurationRuntime | None:
        reconfiguration_id = self._machine_reconfiguration.get(machine_id)
        if reconfiguration_id is None:
            return None
        reconfiguration = self.reconfigurations[reconfiguration_id]
        if reconfiguration.stage not in {
            ReconfigurationStage.WAIT_DIS,
            ReconfigurationStage.WAIT_INS,
        }:
            return None
        return reconfiguration

    def _resolve_terminal_or_deadlock(self) -> None:
        if self.terminated or self.truncated:
            self.decision_type = DecisionType.TERMINAL
            return
        if all(operation.state == OperationState.DONE for operation in self.operations):
            self.terminated = True
            self.terminal_reason = "completed"
            self.decision_type = DecisionType.TERMINAL
            return
        if self.current_tick >= self.horizon_tick:
            self._apply_truncation("horizon")
            return
        if self.decision_type != DecisionType.TERMINAL:
            mask = self.get_action_mask()
            if bool(mask.all()):
                self._truncate_at_horizon("deadlock")

    def _truncate_at_horizon(self, reason: str) -> None:
        if self.terminated or self.truncated:
            return
        if self.current_tick < self.horizon_tick:
            self._advance_interval(self.horizon_tick)
            self.current_tick = self.horizon_tick
        self._apply_truncation(reason)

    def _apply_truncation(self, reason: str) -> None:
        if self.truncated:
            return
        self._settle_partial_worker_tasks()
        self._clip_active_log_intervals()
        unfinished = len(self.instance.orders) - len(self._order_completion_tick)
        self._flow_penalty = (
            unfinished * self.instance.unfinished_order_penalty
        )
        self.truncated = True
        self.terminal_reason = reason
        self.decision_type = DecisionType.TERMINAL

    def _settle_partial_worker_tasks(self) -> None:
        for reconfiguration in self.reconfigurations.values():
            if reconfiguration.stage == ReconfigurationStage.DIS:
                worker_id = reconfiguration.disassembly_worker_id
                start_tick = reconfiguration.disassembly_start_tick
                end_tick = reconfiguration.disassembly_end_tick
                accumulation_rate = (
                    self.instance.fatigue.disassembly_accumulation_rate_per_minute
                )
            elif reconfiguration.stage == ReconfigurationStage.INS:
                worker_id = reconfiguration.installation_worker_id
                start_tick = reconfiguration.installation_start_tick
                end_tick = reconfiguration.installation_end_tick
                accumulation_rate = (
                    self.instance.fatigue.installation_accumulation_rate_per_minute
                )
            else:
                continue
            if worker_id is None or start_tick is None or end_tick is None:
                raise RuntimeError(
                    f"incomplete runtime data for {reconfiguration.id}"
                )
            worked_ticks = max(
                0, min(self.current_tick, end_tick) - start_tick
            )
            duration = ticks_to_minutes(worked_ticks, self.resolution)
            worker = self._worker_by_id(worker_id)
            worker.load += duration
            worker.fatigue = min(
                1.0, worker.fatigue + accumulation_rate * duration
            )
            worker.peak_fatigue = max(worker.peak_fatigue, worker.fatigue)
            self._maximum_fatigue_seen = max(
                self._maximum_fatigue_seen, worker.fatigue
            )

    def _clip_active_log_intervals(self) -> None:
        terminal_time = self.current_time
        for record in (*self.schedule_log, *self.reconfiguration_log):
            if record["end"] <= terminal_time + EPSILON:
                continue
            record["planned_end"] = record["end"]
            record["end"] = terminal_time
            record["duration"] = max(0.0, terminal_time - record["start"])
            record["truncated"] = True

    def _objective_vector(self) -> tuple[float, float, float]:
        return (
            self._flow_integral + self._flow_penalty,
            self._reconfiguration_cost,
            self._load_variance(),
        )

    def _load_variance(self) -> float:
        loads = np.asarray([worker.load for worker in self.workers], dtype=np.float64)
        return float(np.var(loads)) if len(loads) else 0.0

    def _push_event(
        self,
        tick: int,
        event_type: EventType,
        payload: dict[str, Any],
        *,
        priority: int = 0,
    ) -> None:
        self._event_serial += 1
        heapq.heappush(
            self._events,
            (tick, priority, self._event_serial, event_type, payload),
        )

    def _order_by_id(self, order_id: str):
        return next(order for order in self.instance.orders if order.id == order_id)

    def _operation_by_id(self, operation_id: str) -> OperationRuntime:
        return next(
            operation
            for operation in self.operations
            if operation.spec.id == operation_id
        )

    def _machine_by_id(self, machine_id: str) -> MachineRuntime:
        return next(
            machine for machine in self.machines if machine.spec.id == machine_id
        )

    def _worker_by_id(self, worker_id: str) -> WorkerRuntime:
        return next(worker for worker in self.workers if worker.spec.id == worker_id)

    def _require_instance(self) -> None:
        if self.instance is None:
            raise RuntimeError("reset must be called before using the environment")
