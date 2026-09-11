from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from data.feasibility import maximum_matching_size as _maximum_matching_size
from data.models import (
    AssemblyInstance,
)
from environment.actions import ActionCodec
from environment.dynamics import EPSILON, quantize_to_ticks, ticks_to_minutes
from environment.state import (
    MachineRuntime,
    OperationRuntime,
    ReconfigurationRuntime,
    WorkerTaskSnapshot,
    WorkerRuntime,
)
from environment.preference import (
    PreferenceContext,
    PreferenceContextInput,
    PreferenceInput,
    PreferenceVector,
    default_preference,
    default_preference_context,
    normalize_preference,
)
from environment.types import (
    CAPABLE_EDGE,
    CAN_DISASSEMBLE_EDGE,
    CAN_INSTALL_EDGE,
    LOCKED_EDGE,
    MACHINE_MODULE_EDGE,
    OPERATION_ORDER_EDGE,
    ORDER_WAVE_EDGE,
    PRECEDES_EDGE,
    REQUIRES_MODULE_EDGE,
    SERVICE_CANDIDATE_EDGE,
    WAVE_MODULE_EDGE,
    WORKER_MODULE_EDGE,
    DecisionType,
    EdgeStore,
    EdgeType,
    EventType,
    HeterogeneousGraphObservation,
    MachineState,
    Observation,
    OperationState,
    ReconfigurationStage,
    RewardVector,
    WorkerState,
    bounded_quality_score,
    objective_scalarizer_config,
)


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


@dataclass(frozen=True)
class ResourceFeasibilitySnapshot:
    """Shared worker feasibility view used by masks, features, and metrics."""

    tasks: tuple[WorkerTaskSnapshot, ...]
    safe_edges: tuple[tuple[int, ...], ...]
    matching_size: int
    safe_idle_workers: tuple[int, ...]
    minimum_worker_alternatives: int


@dataclass(frozen=True)
class ProductionResourceProfile:
    """Operation-independent resource projection for a machine/module pair."""

    resource_ready_tick: int
    processing_start_tick: int | None
    safe_disassembly_workers: int
    safe_installation_workers: int
    matching_deficit_after_commit: int


class AssemblySchedulingEnv:
    """Single-instance discrete-event environment with two decision phases."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.preference_context = default_preference_context(config)
        # Compatibility alias retained for archived analysis code.
        self.preference: PreferenceVector = self.preference_context.preference
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
        self._committed_worker_loads = np.empty(0, dtype=np.float64)
        self._active_committed_worker_tasks: dict[
            tuple[str, str], tuple[int, float]
        ] = {}
        self._worker_assignment_count = 0
        self._worker_assignment_variance_reward_sum = 0.0
        self._worker_assignment_variance_reward_abs_sum = 0.0
        self._worker_assignment_nonzero_variance_reward_count = 0
        self.reconfigurations: dict[str, ReconfigurationRuntime] = {}
        self._machine_reconfiguration: dict[str, str] = {}
        self._static_edge_indices: dict[EdgeType, np.ndarray] = {}
        self._static_relations: dict[EdgeType, EdgeStore] = {}
        self._capability_operation_indices = np.empty(0, dtype=np.int64)
        self._capability_machine_indices = np.empty(0, dtype=np.int64)
        self._capability_target_module_indices = np.empty(0, dtype=np.int64)
        self._capability_group_ids = np.empty(0, dtype=np.int64)
        self._capability_unique_group_ids = np.empty(0, dtype=np.int64)
        self._capability_processing_ticks = np.empty(0, dtype=np.int64)
        self._machine_module_constant_features = np.empty(
            (0, 3), dtype=np.float32
        )
        self._wave_module_operation_indices: tuple[np.ndarray, ...] = ()
        self._events: list[tuple[int, int, int, EventType, dict[str, Any]]] = []
        self._event_serial = 0
        self._decision_count = 0
        self._zero_time_actions = 0
        self._forced_action_counts: dict[str, int] = {}
        self._current_forced_action_chain = 0
        self._longest_forced_action_chain = 0
        self._forced_action_chain_count = 0
        self._last_action_mask_analysis: dict[str, Any] | None = None
        self._flow_integral = 0.0
        self._flow_penalty = 0.0
        self._reconfiguration_cost = 0.0
        self._maximum_fatigue_seen = 0.0
        self._fatigue_candidate_actions: set[tuple[int, str, str, str]] = set()
        self._fatigue_masked_actions: set[tuple[int, str, str, str]] = set()
        self._worker_competition_ticks: set[int] = set()
        self._worker_matching_deficit_ticks: set[int] = set()
        self._maximum_worker_matching_deficit = 0
        self._wait_action_count = 0
        self._production_wait_action_count = 0
        self._worker_wait_action_count = 0
        self._production_wait_ticks = 0
        self._worker_wait_ticks = 0
        self._wait_reason_counts: dict[str, int] = {}
        self._wait_mask_reason_counts: dict[str, int] = {}
        self._wait_masked_states: set[tuple[int, str]] = set()
        self._wait_min_estimated_deadline_slack_ticks: int | None = None
        self._last_wait_certificate: dict[str, Any] | None = None
        self._first_unrecoverable_deadlock_diagnostic: dict[str, Any] | None = None
        self._reconfiguration_reuse_count = 0
        self._post_reconfiguration_process_count: dict[str, int] = {}
        self._action_type_counts: dict[str, int] = {}
        self._minimum_worker_alternatives_seen: int | None = None
        self._resource_snapshot_cache: ResourceFeasibilitySnapshot | None = None
        self._production_resource_profile_cache: dict[
            tuple[int, str], ProductionResourceProfile
        ] = {}
        self._stage_projection_cache: dict[
            tuple[int, int, str, bool, int], tuple[int, int] | None
        ] = {}
        self._reconfiguration_duration_improvement_cache_version = -1
        self._reconfiguration_duration_improvement_cache: int | None = None
        self._state_version = 0
        self._action_mask_cache_version = -1
        self._action_mask_cache: np.ndarray | None = None
        self._observation_cache_version = -1
        self._observation_cache: Observation | None = None
        self._cumulative_reward = np.zeros(3, dtype=np.float64)
        self._initial_objectives = (0.0, 0.0, 0.0)
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
    def _action_codec(self) -> ActionCodec:
        return ActionCodec(
            operation_count=len(self.operations),
            machine_count=len(self.machines),
            worker_count=len(self.workers),
        )

    @property
    def production_action_size(self) -> int:
        return self._action_codec.production_size

    @property
    def worker_action_size(self) -> int:
        return self._action_codec.worker_size

    @property
    def wait_action(self) -> int:
        """Return the phase-specific WAIT slot."""
        if self.decision_type == DecisionType.PRODUCTION:
            return self._action_codec.production_wait
        if self.decision_type == DecisionType.WORKER:
            return self._action_codec.worker_wait
        raise RuntimeError("terminal state has no action")

    def _invalidate_resource_snapshot(self) -> None:
        self._state_version += 1
        self._resource_snapshot_cache = None
        self._production_resource_profile_cache = {}
        self._stage_projection_cache = {}
        self._reconfiguration_duration_improvement_cache_version = -1
        self._reconfiguration_duration_improvement_cache = None
        self._action_mask_cache_version = -1
        self._action_mask_cache = None
        self._observation_cache_version = -1
        self._observation_cache = None

    def reset(
        self,
        instance: AssemblyInstance,
        *,
        preference: PreferenceContextInput | None = None,
        build_observation: bool = True,
    ) -> Observation | None:
        self.preference_context = (
            default_preference_context(self.config)
            if preference is None
            else PreferenceContext.from_input(preference)
        )
        self.preference = self.preference_context.preference
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
        self._committed_worker_loads = np.zeros(
            len(self.workers), dtype=np.float64
        )
        self._active_committed_worker_tasks = {}
        self._worker_assignment_count = 0
        self._worker_assignment_variance_reward_sum = 0.0
        self._worker_assignment_variance_reward_abs_sum = 0.0
        self._worker_assignment_nonzero_variance_reward_count = 0
        self.reconfigurations = {}
        self._machine_reconfiguration = {}
        self._static_edge_indices = self._build_static_edge_indices()
        self._initialize_static_graph_cache()
        self._events = []
        self._event_serial = 0
        self._decision_count = 0
        self._zero_time_actions = 0
        self._forced_action_counts = {}
        self._current_forced_action_chain = 0
        self._longest_forced_action_chain = 0
        self._forced_action_chain_count = 0
        self._last_action_mask_analysis = None
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
        self._maximum_worker_matching_deficit = 0
        self._wait_action_count = 0
        self._production_wait_action_count = 0
        self._worker_wait_action_count = 0
        self._production_wait_ticks = 0
        self._worker_wait_ticks = 0
        self._wait_reason_counts = {}
        self._wait_mask_reason_counts = {}
        self._wait_masked_states = set()
        self._wait_min_estimated_deadline_slack_ticks = None
        self._last_wait_certificate = None
        self._first_unrecoverable_deadlock_diagnostic = None
        self._reconfiguration_reuse_count = 0
        self._post_reconfiguration_process_count = {}
        self._action_type_counts = {}
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
        self._initial_objectives = self._objective_vector()
        # ``build_observation=False`` is used by callers that intentionally
        # defer graph/resource feature construction until the first explicit
        # ``observe()`` call.  Deadlock resolution must still inspect the
        # action mask, but its speculative resource projections should not
        # consume the observation cache's first-build budget.
        if not build_observation:
            self._production_resource_profile_cache.clear()
        return self.observe() if build_observation else None

    def observe(self) -> Observation:
        self._require_instance()
        if (
            self._observation_cache is not None
            and self._observation_cache_version == self._state_version
        ):
            return self._observation_cache.copy()
        horizon = float(self.instance.horizon)
        total_operations = max(1, len(self.operations))
        total_machines = max(1, len(self.machines))
        total_orders = max(1, len(self.instance.orders))
        scalarizer = objective_scalarizer_config(self.config)
        cost_scale = float(scalarizer["scales"]["cost"])
        variance_scale = float(scalarizer["scales"]["variance"])
        safe_fatigue = float(self.instance.fatigue.maximum_safe_fatigue)
        module_values = (self.instance.no_module_state, *self.instance.modules)
        operation_states = tuple(OperationState)
        machine_states = tuple(MachineState)
        worker_states = tuple(WorkerState)
        runtime_by_id = {
            operation.spec.id: operation for operation in self.operations
        }
        maximum_order_operations = max(
            1, max(len(order.operations) for order in self.instance.orders)
        )

        operation_feature_names = (
            tuple(f"required_module_{module}" for module in self.instance.modules)
            + tuple(f"state_{state.value}" for state in operation_states)
            + (
                "order_release_time_norm",
                "base_processing_time_norm",
                "sequence_norm",
                "is_last_operation",
            )
        )
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
                    operation.spec.base_processing_time / horizon,
                    operation.spec.sequence / maximum_order_operations,
                    float(operation.spec.sequence == len(order.operations)),
                ]
            )

        machine_feature_names = (
            tuple(f"current_module_{module}" for module in module_values)
            + tuple(f"state_{state.value}" for state in machine_states)
            + tuple(f"target_module_{module}" for module in module_values)
            + ("remaining_busy_time_norm", "downtime_cost_rate_norm")
            + tuple(
                f"supports_module_{module}" for module in self.instance.modules
            )
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
                    machine.spec.downtime_cost_per_minute * horizon
                    / cost_scale,
                ]
                + [
                    float(module in machine.spec.module_parameters)
                    for module in self.instance.modules
                ]
            )

        worker_feature_names = (
            ("fatigue_ratio",)
            + tuple(f"state_{state.value}" for state in worker_states)
            + ("remaining_busy_time_norm",)
            + tuple(
                f"qualified_module_{module}" for module in self.instance.modules
            )
            + ("load_norm", "labor_cost_rate_norm")
        )
        worker_features = []
        for worker in self.workers:
            state_one_hot = [
                float(worker.state == state) for state in worker_states
            ]
            remaining = max(0, (worker.busy_until_tick or self.current_tick) - self.current_tick)
            worker_features.append(
                [worker.fatigue / safe_fatigue]
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
                    worker.spec.labor_cost_per_minute * horizon / cost_scale,
                ]
            )

        order_feature_names = (
            "released",
            "completed",
            "age_norm",
            "completion_ratio",
            "remaining_operation_ratio",
            "remaining_workload_norm",
        )
        order_features = []
        for order in self.instance.orders:
            released = self._order_released[order.id]
            completed = order.id in self._order_completion_tick
            done_count = sum(
                runtime_by_id[operation.id].state == OperationState.DONE
                for operation in order.operations
            )
            remaining_operations = [
                operation
                for operation in order.operations
                if runtime_by_id[operation.id].state != OperationState.DONE
            ]
            age = (
                max(0.0, self.current_time - order.release_time)
                if released
                else 0.0
            )
            order_features.append(
                [
                    float(released),
                    float(completed),
                    age / horizon,
                    done_count / max(1, len(order.operations)),
                    len(remaining_operations) / total_operations,
                    sum(
                        operation.base_processing_time
                        for operation in remaining_operations
                    )
                    / horizon,
                ]
            )

        module_feature_names = (
            "fixed_disassembly_cost_norm",
            "fixed_installation_cost_norm",
            "released_remaining_operation_ratio",
            "released_remaining_workload_norm",
            "future_remaining_operation_ratio",
            "future_remaining_workload_norm",
            "ready_operation_ratio",
            "installed_machine_ratio",
            "target_machine_ratio",
        )
        module_features = []
        for module in self.instance.modules:
            remaining = [
                operation
                for operation in self.operations
                if operation.spec.required_module == module
                and operation.state != OperationState.DONE
            ]
            released_remaining = [
                operation
                for operation in remaining
                if self._order_released[operation.spec.order_id]
            ]
            future_remaining = [
                operation
                for operation in remaining
                if not self._order_released[operation.spec.order_id]
            ]
            ready = [
                operation
                for operation in remaining
                if operation.state == OperationState.READY
            ]
            costs = self.instance.module_costs[module]
            module_features.append(
                [
                    costs.fixed_disassembly_cost / cost_scale,
                    costs.fixed_installation_cost / cost_scale,
                    len(released_remaining) / total_operations,
                    sum(
                        operation.spec.base_processing_time
                        for operation in released_remaining
                    )
                    / horizon,
                    len(future_remaining) / total_operations,
                    sum(
                        operation.spec.base_processing_time
                        for operation in future_remaining
                    )
                    / horizon,
                    len(ready) / total_operations,
                    sum(
                        machine.current_module == module
                        for machine in self.machines
                    )
                    / total_machines,
                    sum(
                        (machine.target_module or machine.current_module) == module
                        for machine in self.machines
                    )
                    / total_machines,
                ]
            )

        wave_ids = tuple(self.instance.waves)
        wave_feature_names = (
            "release_start_norm",
            "release_end_norm",
            "time_until_release_norm",
            "released_order_ratio",
            "active_order_ratio",
            "completed_order_ratio",
            "remaining_operation_ratio",
            "remaining_workload_norm",
        )
        wave_features = []
        for wave_id in wave_ids:
            wave_orders = [
                order for order in self.instance.orders if order.wave == wave_id
            ]
            release_interval = self.instance.waves[wave_id].get(
                "release_interval"
            )
            if release_interval is None:
                release_interval = (
                    [
                        min(order.release_time for order in wave_orders),
                        max(order.release_time for order in wave_orders),
                    ]
                    if wave_orders
                    else [0.0, 0.0]
                )
            released_count = sum(
                self._order_released[order.id] for order in wave_orders
            )
            completed_count = sum(
                order.id in self._order_completion_tick for order in wave_orders
            )
            active_count = sum(
                self._order_released[order.id]
                and order.id not in self._order_completion_tick
                for order in wave_orders
            )
            remaining_specs = [
                operation
                for order in wave_orders
                for operation in order.operations
                if runtime_by_id[operation.id].state != OperationState.DONE
            ]
            wave_features.append(
                [
                    float(release_interval[0]) / horizon,
                    float(release_interval[1]) / horizon,
                    max(0.0, float(release_interval[0]) - self.current_time)
                    / horizon,
                    released_count / max(1, len(wave_orders)),
                    active_count / max(1, len(wave_orders)),
                    completed_count / max(1, len(wave_orders)),
                    len(remaining_specs) / total_operations,
                    sum(
                        operation.base_processing_time
                        for operation in remaining_specs
                    )
                    / horizon,
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
                    profile = self._production_resource_profile(
                        machine_index,
                        operation.spec.required_module,
                    )
                    predicted_finish_tick = (
                        profile.processing_start_tick
                        + self.estimate_processing_ticks(
                            operation_index, machine_index
                        )
                        if profile.processing_start_tick is not None
                        else self.horizon_tick + 1
                    )
                    candidate_slacks.append(
                        max(
                            -1.0,
                            min(
                                1.0,
                                (self.horizon_tick - predicted_finish_tick)
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
        action_set_features, action_set_feature_names = (
            self._build_action_set_features(relations)
        )
        observation = Observation(
            node_features={
                "operation": np.asarray(operation_features, dtype=np.float32),
                "machine": np.asarray(machine_features, dtype=np.float32),
                "worker": np.asarray(worker_features, dtype=np.float32),
                "order": np.asarray(order_features, dtype=np.float32),
                "module": np.asarray(module_features, dtype=np.float32),
                "wave": np.asarray(wave_features, dtype=np.float32),
            },
            global_features=global_features,
            decision_type=self.decision_type,
            preference=self.preference_context.as_array(),
            node_feature_names={
                "operation": operation_feature_names,
                "machine": machine_feature_names,
                "worker": worker_feature_names,
                "order": order_feature_names,
                "module": module_feature_names,
                "wave": wave_feature_names,
            },
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
                "order": tuple(order.id for order in self.instance.orders),
                "module": tuple(self.instance.modules),
                "wave": wave_ids,
            },
            relations=relations,
            action_set_features=action_set_features,
            action_set_feature_names=action_set_feature_names,
        )
        self._observation_cache = observation.copy()
        self._observation_cache_version = self._state_version
        return observation

    def _build_static_edge_indices(self) -> dict[EdgeType, np.ndarray]:
        self._require_instance()
        operation_index = self.instance.operation_index
        order_index = {
            order.id: index for index, order in enumerate(self.instance.orders)
        }
        module_index = {
            module: index for index, module in enumerate(self.instance.modules)
        }
        wave_index = {
            wave: index for index, wave in enumerate(self.instance.waves)
        }
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
        operation_order_pairs = [
            (operation_index[operation.id], order_index[order.id])
            for order in self.instance.orders
            for operation in order.operations
        ]
        order_wave_pairs = [
            (order_index[order.id], wave_index[order.wave])
            for order in self.instance.orders
        ]
        operation_module_pairs = [
            (
                operation_index[operation.id],
                module_index[operation.required_module],
            )
            for operation in self.instance.operations
        ]
        machine_module_pairs = [
            (machine_index, module_index[module])
            for machine_index, machine in enumerate(self.machines)
            for module in self.instance.modules
            if module in machine.spec.module_parameters
        ]
        worker_module_pairs = [
            (worker_index, module_index[module])
            for worker_index, worker in enumerate(self.workers)
            for module in self.instance.modules
            if module in worker.spec.qualified_modules
        ]
        wave_module_pairs = [
            (wave_index[wave], module_index[module])
            for wave in self.instance.waves
            for module in self.instance.modules
        ]
        return {
            PRECEDES_EDGE: _as_edge_index(precedence_pairs),
            CAPABLE_EDGE: _as_edge_index(capability_pairs),
            CAN_INSTALL_EDGE: _as_edge_index(installation_pairs),
            OPERATION_ORDER_EDGE: _as_edge_index(operation_order_pairs),
            ORDER_WAVE_EDGE: _as_edge_index(order_wave_pairs),
            REQUIRES_MODULE_EDGE: _as_edge_index(operation_module_pairs),
            MACHINE_MODULE_EDGE: _as_edge_index(machine_module_pairs),
            WORKER_MODULE_EDGE: _as_edge_index(worker_module_pairs),
            WAVE_MODULE_EDGE: _as_edge_index(wave_module_pairs),
        }

    def _initialize_static_graph_cache(self) -> None:
        """Build immutable-by-convention graph structures for this instance."""

        def unit_relation(
            edge_type: EdgeType,
            feature_name: str,
            *,
            bidirectional: bool,
        ) -> EdgeStore:
            edge_index = self._static_edge_indices[edge_type]
            return EdgeStore(
                edge_index=edge_index.copy(),
                edge_features=np.ones(
                    (edge_index.shape[1], 1), dtype=np.float32
                ),
                feature_names=(feature_name,),
                bidirectional=bidirectional,
            )

        self._static_relations = {
            PRECEDES_EDGE: unit_relation(
                PRECEDES_EDGE, "precedence", bidirectional=False
            ),
            CAN_INSTALL_EDGE: unit_relation(
                CAN_INSTALL_EDGE, "qualified", bidirectional=False
            ),
            OPERATION_ORDER_EDGE: unit_relation(
                OPERATION_ORDER_EDGE,
                "belongs_to_order",
                bidirectional=True,
            ),
            ORDER_WAVE_EDGE: unit_relation(
                ORDER_WAVE_EDGE,
                "belongs_to_wave",
                bidirectional=True,
            ),
            REQUIRES_MODULE_EDGE: unit_relation(
                REQUIRES_MODULE_EDGE,
                "requires_module",
                bidirectional=True,
            ),
            WORKER_MODULE_EDGE: unit_relation(
                WORKER_MODULE_EDGE,
                "qualified_for_module",
                bidirectional=True,
            ),
        }

        capability_index = self._static_edge_indices[CAPABLE_EDGE]
        self._capability_operation_indices = capability_index[0].copy()
        self._capability_machine_indices = capability_index[1].copy()
        module_index = {
            module: index for index, module in enumerate(self.instance.modules)
        }
        self._capability_target_module_indices = np.asarray(
            [
                module_index[self.operations[int(index)].spec.required_module]
                for index in self._capability_operation_indices
            ],
            dtype=np.int64,
        )
        module_count = len(self.instance.modules)
        self._capability_group_ids = (
            self._capability_machine_indices * module_count
            + self._capability_target_module_indices
        )
        self._capability_unique_group_ids = np.unique(
            self._capability_group_ids
        )
        self._capability_processing_ticks = np.asarray(
            [
                self.estimate_processing_ticks(int(operation), int(machine))
                for operation, machine in capability_index.T
            ],
            dtype=np.int64,
        )

        machine_module_index = self._static_edge_indices[MACHINE_MODULE_EDGE]
        self._machine_module_constant_features = np.asarray(
            [
                [
                    self.machines[int(machine)].spec.module_parameters[
                        self.instance.modules[int(module)]
                    ].installation_base_time
                    / self.instance.horizon,
                    self.machines[int(machine)].spec.module_parameters[
                        self.instance.modules[int(module)]
                    ].disassembly_base_time
                    / self.instance.horizon,
                    self.machines[int(machine)].spec.module_parameters[
                        self.instance.modules[int(module)]
                    ].processing_speed_factor,
                ]
                for machine, module in machine_module_index.T
            ],
            dtype=np.float32,
        ).reshape(-1, 3)

        order_wave = {order.id: order.wave for order in self.instance.orders}
        wave_ids = tuple(self.instance.waves)
        wave_module_index = self._static_edge_indices[WAVE_MODULE_EDGE]
        self._wave_module_operation_indices = tuple(
            np.asarray(
                [
                    operation_index
                    for operation_index, operation in enumerate(self.operations)
                    if order_wave[operation.spec.order_id]
                    == wave_ids[int(wave_index)]
                    and operation.spec.required_module
                    == self.instance.modules[int(module_index_value)]
                ],
                dtype=np.int64,
            )
            for wave_index, module_index_value in wave_module_index.T
        )

    def _build_capability_relation(self) -> EdgeStore:
        """Build dynamic capability features from machine/module profiles."""

        edge_index = self._static_edge_indices[CAPABLE_EDGE]
        edge_count = edge_index.shape[1]
        feature_names = (
            "processing_time_norm",
            "configuration_match",
            "earliest_start_time_norm",
            "resource_ready_time_norm",
            "predicted_finish_time_norm",
            "safe_disassembly_worker_ratio",
            "safe_installation_worker_ratio",
            "matching_deficit_after_commit_norm",
            "horizon_slack_norm",
            "reconfiguration_time_norm",
            "fixed_disassembly_cost_norm",
            "fixed_installation_cost_norm",
            "estimated_labor_cost_norm",
            "estimated_downtime_cost_norm",
            "estimated_worker_load_variance_delta_norm",
        )
        if edge_count == 0:
            return EdgeStore(
                edge_index=edge_index.copy(),
                edge_features=np.empty((0, len(feature_names)), dtype=np.float32),
                feature_names=feature_names,
                bidirectional=True,
            )

        module_count = len(self.instance.modules)
        group_count = len(self.machines) * module_count
        configuration_match = np.zeros(group_count, dtype=np.float64)
        earliest_start = np.zeros(group_count, dtype=np.float64)
        resource_ready = np.zeros(group_count, dtype=np.float64)
        processing_start = np.zeros(group_count, dtype=np.float64)
        projection_missing = np.zeros(group_count, dtype=bool)
        safe_disassembly = np.zeros(group_count, dtype=np.float64)
        safe_installation = np.zeros(group_count, dtype=np.float64)
        matching_deficit = np.zeros(group_count, dtype=np.float64)
        reconfiguration_ticks = np.zeros(group_count, dtype=np.float64)
        fixed_disassembly = np.zeros(group_count, dtype=np.float64)
        fixed_installation = np.zeros(group_count, dtype=np.float64)
        labor_cost = np.zeros(group_count, dtype=np.float64)
        downtime_cost = np.zeros(group_count, dtype=np.float64)
        load_variance_delta = np.zeros(group_count, dtype=np.float64)

        for group_id_value in self._capability_unique_group_ids:
            group_id = int(group_id_value)
            machine_index, module_index = divmod(group_id, module_count)
            machine = self.machines[machine_index]
            target_module = self.instance.modules[module_index]
            resource_profile = self._production_resource_profile(
                machine_index, target_module
            )
            matches = machine.current_module == target_module
            configuration_match[group_id] = float(matches)
            available_tick, available_module = self._machine_committed_release(
                machine_index
            )
            earliest_start[group_id] = (
                available_tick
                + self._optimistic_reconfiguration_ticks(
                    machine, available_module, target_module
                )
            )
            resource_ready[group_id] = resource_profile.resource_ready_tick
            if resource_profile.processing_start_tick is None:
                projection_missing[group_id] = True
                processing_start[group_id] = self.horizon_tick + 1
            else:
                processing_start[group_id] = (
                    resource_profile.processing_start_tick
                )
            safe_disassembly[group_id] = (
                resource_profile.safe_disassembly_workers
            )
            safe_installation[group_id] = (
                resource_profile.safe_installation_workers
            )
            matching_deficit[group_id] = (
                resource_profile.matching_deficit_after_commit
            )
            reconfiguration_ticks[group_id] = (
                self._optimistic_reconfiguration_ticks(
                    machine, machine.current_module, target_module
                )
            )
            source_cost = self.instance.module_costs.get(machine.current_module)
            fixed_disassembly[group_id] = (
                0.0
                if matches or source_cost is None
                else source_cost.fixed_disassembly_cost
            )
            fixed_installation[group_id] = (
                0.0
                if matches
                else self.instance.module_costs[
                    target_module
                ].fixed_installation_cost
            )
            labor_cost[group_id], downtime_cost[group_id] = (
                self._estimate_candidate_reconfiguration_costs(
                    machine, target_module
                )
            )
            load_variance_delta[group_id] = (
                self._estimate_candidate_load_variance_delta(
                    machine_index, target_module
                )
            )

        group_ids = self._capability_group_ids
        operation_indices = self._capability_operation_indices
        machine_indices = self._capability_machine_indices
        earliest_start_edges = earliest_start[group_ids].copy()
        for machine_index, machine in enumerate(self.machines):
            reconfiguration = self._active_reconfiguration(machine.spec.id)
            if reconfiguration is None:
                continue
            operation_index = self.instance.operation_index[
                reconfiguration.operation_id
            ]
            locked_edge = (
                (operation_indices == operation_index)
                & (machine_indices == machine_index)
            )
            earliest_start_edges[locked_edge] = (
                self.current_tick
                + self._remaining_reconfiguration_ticks(reconfiguration)
            )

        predicted_finish = (
            processing_start[group_ids] + self._capability_processing_ticks
        )
        predicted_finish[projection_missing[group_ids]] = self.horizon_tick + 1
        horizon_slack = self.horizon_tick - predicted_finish
        worker_count = max(1, len(self.workers))
        horizon_tick = max(1, self.horizon_tick)
        cost_scale = float(
            objective_scalarizer_config(self.config)["scales"]["cost"]
        )
        features = np.empty((edge_count, len(feature_names)), dtype=np.float32)
        features[:, 0] = np.clip(
            self._capability_processing_ticks / horizon_tick, 0.0, 2.0
        )
        features[:, 1] = configuration_match[group_ids]
        features[:, 2] = np.clip(
            earliest_start_edges / horizon_tick, 0.0, 2.0
        )
        features[:, 3] = np.clip(
            resource_ready[group_ids] / horizon_tick, 0.0, 2.0
        )
        features[:, 4] = np.clip(predicted_finish / horizon_tick, 0.0, 2.0)
        features[:, 5] = safe_disassembly[group_ids] / worker_count
        features[:, 6] = safe_installation[group_ids] / worker_count
        features[:, 7] = matching_deficit[group_ids] / worker_count
        features[:, 8] = np.clip(horizon_slack / horizon_tick, -1.0, 1.0)
        features[:, 9] = reconfiguration_ticks[group_ids] / horizon_tick
        features[:, 10] = fixed_disassembly[group_ids] / cost_scale
        features[:, 11] = fixed_installation[group_ids] / cost_scale
        features[:, 12] = labor_cost[group_ids] / cost_scale
        features[:, 13] = downtime_cost[group_ids] / cost_scale
        variance_scale = float(
            objective_scalarizer_config(self.config)["scales"]["variance"]
        )
        features[:, 14] = load_variance_delta[group_ids] / variance_scale
        return EdgeStore(
            edge_index=edge_index.copy(),
            edge_features=features,
            feature_names=feature_names,
            bidirectional=True,
        )

    def _build_graph_relations(self) -> dict[EdgeType, EdgeStore]:
        self._require_instance()
        cost_scale = float(
            objective_scalarizer_config(self.config)["scales"]["cost"]
        )
        precedence = self._static_relations[PRECEDES_EDGE].copy()
        capability = self._build_capability_relation()

        locked = self._build_locked_edges()

        installation = self._static_relations[CAN_INSTALL_EDGE].copy()

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

        operation_order = self._static_relations[OPERATION_ORDER_EDGE].copy()
        order_wave = self._static_relations[ORDER_WAVE_EDGE].copy()
        requires_module = self._static_relations[REQUIRES_MODULE_EDGE].copy()

        machine_module_index = self._static_edge_indices[MACHINE_MODULE_EDGE]
        machine_module_features = np.empty(
            (machine_module_index.shape[1], 5), dtype=np.float32
        )
        for edge, (machine_index, module_index) in enumerate(
            machine_module_index.T
        ):
            machine = self.machines[int(machine_index)]
            module = self.instance.modules[int(module_index)]
            target_module = machine.target_module or machine.current_module
            machine_module_features[edge, 0] = float(
                machine.current_module == module
            )
            machine_module_features[edge, 1] = float(target_module == module)
        machine_module_features[:, 2:] = (
            self._machine_module_constant_features
        )
        machine_module = EdgeStore(
            edge_index=machine_module_index.copy(),
            edge_features=machine_module_features,
            feature_names=(
                "currently_installed",
                "target_installed",
                "installation_base_time_norm",
                "disassembly_base_time_norm",
                "processing_speed_factor",
            ),
            bidirectional=True,
        )

        worker_module = self._static_relations[WORKER_MODULE_EDGE].copy()

        wave_module_index = self._static_edge_indices[WAVE_MODULE_EDGE]
        wave_module_features: list[list[float]] = []
        total_operations = max(1, len(self.operations))
        for matching_indices in self._wave_module_operation_indices:
            matching = [
                self.operations[int(index)]
                for index in matching_indices
                if self.operations[int(index)].state != OperationState.DONE
            ]
            released = [
                operation
                for operation in matching
                if self._order_released[operation.spec.order_id]
            ]
            future = [
                operation
                for operation in matching
                if not self._order_released[operation.spec.order_id]
            ]
            wave_module_features.append(
                [
                    len(matching) / total_operations,
                    sum(value.spec.base_processing_time for value in matching)
                    / self.instance.horizon,
                    len(released) / total_operations,
                    len(future) / total_operations,
                ]
            )
        wave_module = EdgeStore(
            edge_index=wave_module_index.copy(),
            edge_features=np.asarray(
                wave_module_features, dtype=np.float32
            ).reshape(-1, 4),
            feature_names=(
                "remaining_operation_ratio",
                "remaining_workload_norm",
                "released_operation_ratio",
                "future_operation_ratio",
            ),
            bidirectional=True,
        )

        service_pairs: list[tuple[int, int]] = []
        service_features: list[list[float]] = []
        variance_scale = float(
            objective_scalarizer_config(self.config)["scales"]["variance"]
        )
        safe_fatigue = float(self.instance.fatigue.maximum_safe_fatigue)
        current_load_variance = self._committed_load_variance()
        remaining_workload_by_module = {
            module: sum(
                operation.spec.base_processing_time
                for operation in self.operations
                if operation.state != OperationState.DONE
                and operation.spec.required_module == module
            )
            for module in self.instance.modules
        }
        total_remaining_workload = max(
            EPSILON,
            sum(remaining_workload_by_module.values()),
        )
        qualified_worker_count = {
            module: sum(
                module in worker.spec.qualified_modules
                for worker in self.workers
            )
            for module in self.instance.modules
        }
        for machine_index, machine in enumerate(self.machines):
            reconfiguration = self._pending_reconfiguration(machine.spec.id)
            if reconfiguration is None:
                continue
            module = (
                reconfiguration.source_module
                if reconfiguration.stage == ReconfigurationStage.WAIT_DIS
                else reconfiguration.target_module
            )
            for worker_index, worker in enumerate(self.workers):
                if module not in worker.spec.qualified_modules:
                    continue
                duration_ticks = self._stage_duration_ticks(
                    reconfiguration, worker
                )
                duration = ticks_to_minutes(duration_ticks, self.resolution)
                projected_fatigue = worker.fatigue + (
                    self._stage_accumulation_rate(reconfiguration) * duration
                )
                fixed_cost = (
                    self.instance.module_costs[
                        reconfiguration.source_module
                    ].fixed_disassembly_cost
                    if reconfiguration.stage == ReconfigurationStage.WAIT_DIS
                    else self.instance.module_costs[
                        reconfiguration.target_module
                    ].fixed_installation_cost
                )
                projected_variance = self.projected_worker_load_variance(
                    machine_index,
                    worker_index,
                )
                labor_cost = duration * worker.spec.labor_cost_per_minute
                downtime_cost = (
                    duration * machine.spec.downtime_cost_per_minute
                )
                values = [
                    float(
                        reconfiguration.stage
                        == ReconfigurationStage.WAIT_DIS
                    ),
                    float(
                        reconfiguration.stage
                        == ReconfigurationStage.WAIT_INS
                    ),
                    duration / self.instance.horizon,
                    projected_fatigue / safe_fatigue,
                    fixed_cost / cost_scale,
                    labor_cost / cost_scale,
                    downtime_cost / cost_scale,
                    (projected_variance - current_load_variance)
                    / variance_scale,
                ]
                service_pairs.append((machine_index, worker_index))
                service_features.append(values)
        service_feature_names = (
            "stage_dis",
            "stage_ins",
            "stage_duration_norm",
            "projected_fatigue_ratio",
            "fixed_cost_norm",
            "incremental_labor_cost_norm",
            "incremental_downtime_cost_norm",
            "incremental_load_variance_norm",
        )
        service_candidate = EdgeStore(
            edge_index=_as_edge_index(service_pairs),
            edge_features=np.asarray(
                service_features, dtype=np.float32
            ).reshape(-1, len(service_feature_names)),
            feature_names=service_feature_names,
            bidirectional=True,
        )

        return {
            PRECEDES_EDGE: precedence,
            CAPABLE_EDGE: capability,
            LOCKED_EDGE: locked,
            CAN_INSTALL_EDGE: installation,
            CAN_DISASSEMBLE_EDGE: disassembly,
            OPERATION_ORDER_EDGE: operation_order,
            ORDER_WAVE_EDGE: order_wave,
            REQUIRES_MODULE_EDGE: requires_module,
            MACHINE_MODULE_EDGE: machine_module,
            WORKER_MODULE_EDGE: worker_module,
            WAVE_MODULE_EDGE: wave_module,
            SERVICE_CANDIDATE_EDGE: service_candidate,
        }

    def _build_action_set_features(
        self,
        relations: dict[EdgeType, EdgeStore],
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        del relations
        certificate = self._wait_certificate()
        wait_ticks = int(certificate.get("wait_ticks", 0))
        wait_minutes = ticks_to_minutes(wait_ticks, self.resolution)
        scalarizer = objective_scalarizer_config(self.config)
        active_orders = [
            order
            for order in self.instance.orders
            if self._order_released[order.id]
            and order.id not in self._order_completion_tick
        ]
        active_ids = {order.id for order in active_orders}
        active_workload = sum(
            operation.spec.base_processing_time
            for operation in self.operations
            if operation.spec.order_id in active_ids
            and operation.state != OperationState.DONE
        )
        next_tick = certificate.get("next_tick")
        next_events = (
            [event for event in self._events if event[0] == int(next_tick)]
            if next_tick is not None
            else []
        )
        new_ready = sum(event[3] == EventType.PROCESS_COMPLETE for event in next_events)
        released_machines = sum(
            event[3]
            in {EventType.PROCESS_COMPLETE, EventType.DIS_COMPLETE, EventType.INS_COMPLETE}
            for event in next_events
        )
        recovering = sum(
            worker.state == WorkerState.IDLE and worker.fatigue > EPSILON
            for worker in self.workers
        )
        recovery_gain = (
            self.instance.fatigue.idle_recovery_rate_per_minute
            * wait_minutes
            * recovering
            / max(1, len(self.workers))
        )
        future_release_ticks = [
            event[0]
            for event in self._events
            if event[3] == EventType.ORDER_RELEASE and event[0] > self.current_tick
        ]
        future_release_delta = (
            (min(future_release_ticks) - self.current_tick) / max(1, self.horizon_tick)
            if future_release_ticks
            else 1.0
        )
        reusable = sum(
            operation.state == OperationState.READY
            and any(
                machine.state == MachineState.IDLE
                and machine.current_module == operation.spec.required_module
                for machine in self.machines
            )
            for operation in self.operations
        )
        downtime_states = {
            MachineState.WAIT_DIS,
            MachineState.DIS,
            MachineState.WAIT_INS,
            MachineState.INS,
        }
        flow_delta = len(active_orders) * wait_minutes
        cost_delta = wait_minutes * (
            sum(
                machine.spec.downtime_cost_per_minute
                for machine in self.machines
                if machine.state in downtime_states
            )
            + sum(
                worker.spec.labor_cost_per_minute
                for worker in self.workers
                if worker.state in {WorkerState.DIS, WorkerState.INS}
            )
        )
        names = (
            "estimated_flow_objective_delta_if_wait",
            "estimated_cost_objective_delta_if_wait",
            "estimated_load_variance_delta_if_wait",
            "wait_duration_norm",
            "next_event_delta_norm",
            "active_order_ratio",
            "active_order_workload_norm",
            "new_ready_operation_ratio",
            "released_machine_ratio",
            "recovering_worker_ratio",
            "worker_recovery_gain",
            "future_wave_release_delta_norm",
            "configuration_reuse_gain",
        )
        values = np.asarray(
            [
                flow_delta / scalarizer["scales"]["flow"],
                cost_delta / scalarizer["scales"]["cost"],
                0.0,
                wait_ticks / max(1, self.horizon_tick),
                wait_ticks / max(1, self.horizon_tick),
                len(active_orders) / max(1, len(self.instance.orders)),
                active_workload / max(EPSILON, self.instance.horizon),
                new_ready / max(1, len(self.operations)),
                released_machines / max(1, len(self.machines)),
                recovering / max(1, len(self.workers)),
                recovery_gain,
                future_release_delta,
                reusable / max(1, len(self.operations)),
            ],
            dtype=np.float32,
        )
        return values, names

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
        if (
            self._action_mask_cache is not None
            and self._action_mask_cache_version == self._state_version
        ):
            return self._action_mask_cache.copy()
        if self.decision_type == DecisionType.TERMINAL:
            self._last_action_mask_analysis = {
                "state_version": self._state_version,
                "phase": self.decision_type.value,
            }
            return self._cache_action_mask(np.ones(1, dtype=bool))
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
                        mask[
                            self.encode_production_action(
                                operation_index, machine_index
                            )
                        ] = False
            legal_pair_count = int(np.count_nonzero(~mask[:-1]))
            wait_certificate = self._wait_certificate()
            if wait_certificate["allowed"]:
                mask[self.wait_action] = False
            self._record_wait_mask_certificate(wait_certificate)
            self._last_action_mask_analysis = {
                "state_version": self._state_version,
                "phase": self.decision_type.value,
                "legal_pair_count": legal_pair_count,
                "wait_allowed": bool(wait_certificate["allowed"]),
                "wait": dict(wait_certificate),
            }
            return self._cache_action_mask(mask)
        mask = np.ones(self.worker_action_size, dtype=bool)
        legal_worker_pairs = 0
        for machine_index, machine in enumerate(self.machines):
            reconfiguration = self._pending_reconfiguration(machine.spec.id)
            if reconfiguration is None:
                continue
            for worker_index, worker in enumerate(self.workers):
                if self._worker_can_start(reconfiguration, worker):
                    mask[
                        self.encode_worker_action(machine_index, worker_index)
                    ] = False
                    legal_worker_pairs += 1
        snapshot = self._resource_feasibility_snapshot()
        matching_deficit = len(snapshot.tasks) - snapshot.matching_size
        self._maximum_worker_matching_deficit = max(
            self._maximum_worker_matching_deficit,
            matching_deficit,
        )
        wait_certificate = self._wait_certificate()
        if wait_certificate["allowed"]:
            mask[self.wait_action] = False
        self._record_wait_mask_certificate(wait_certificate)
        self._last_action_mask_analysis = {
            "state_version": self._state_version,
            "phase": self.decision_type.value,
            "legal_pair_count": legal_worker_pairs,
            "wait_allowed": bool(wait_certificate["allowed"]),
            "wait": dict(wait_certificate),
            "matching_deficit": matching_deficit,
        }
        return self._cache_action_mask(mask)

    def _cache_action_mask(self, mask: np.ndarray) -> np.ndarray:
        """Cache one immutable-by-convention mask for the current state."""

        cached = np.asarray(mask, dtype=np.bool_).copy()
        self._action_mask_cache_version = self._state_version
        self._action_mask_cache = cached
        return cached.copy()

    def forced_action_diagnostic(
        self,
        action_mask: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Classify a state whose mask contains exactly one legal action.

        The classification separates physical pair availability from progress
        opportunities. It is independent of forced-action compression.
        """

        mask = (
            self.get_action_mask()
            if action_mask is None
            else np.asarray(action_mask, dtype=np.bool_)
        )
        legal_actions = np.flatnonzero(~mask)
        if legal_actions.size != 1:
            return None
        action = int(legal_actions[0])
        analysis = self._last_action_mask_analysis or {}
        if (
            analysis.get("state_version") != self._state_version
            or analysis.get("phase") != self.decision_type.value
        ):
            # The supplied mask did not originate from the latest state.  A
            # fresh mask keeps the diagnostic causal and internally aligned.
            mask = self.get_action_mask()
            legal_actions = np.flatnonzero(~mask)
            if legal_actions.size != 1:
                return None
            action = int(legal_actions[0])
            analysis = self._last_action_mask_analysis or {}

        action_kind = "wait" if action == len(mask) - 1 else "pair"
        action_type = self._action_type(self.decision_type, action)
        stage_tags: set[str] = set()
        if self.decision_type == DecisionType.WORKER and action_kind == "pair":
            machine_index, _ = self.decode_worker_action(action)
            reconfiguration = self._pending_reconfiguration(
                self.machines[machine_index].spec.id
            )
            if reconfiguration is not None:
                stage_tags.add(reconfiguration.stage.value)
        else:
            stage_tags.update(
                reconfiguration.stage.value
                for reconfiguration in self.reconfigurations.values()
                if reconfiguration.stage
                in {
                    ReconfigurationStage.WAIT_DIS,
                    ReconfigurationStage.WAIT_INS,
                }
            )

        phase_handoff = bool(
            self.decision_type == DecisionType.PRODUCTION
            and action_kind == "wait"
            and self._has_pending_worker_task()
        )
        recovery = bool(
            action_kind == "wait"
            and self._forced_wait_has_recovery_candidate()
        )
        future_event = bool(
            action_kind == "wait"
            and any(event[0] > self.current_tick for event in self._events)
        )
        return {
            "phase": self.decision_type.value,
            "action": action,
            "action_kind": action_kind,
            "action_type": action_type,
            "stage_tags": tuple(sorted(stage_tags)),
            "wait_physically_unavailable": action_kind == "pair",
            "pair_physically_unavailable": action_kind == "wait",
            "phase_handoff": phase_handoff,
            "recovery": recovery,
            "future_event": future_event,
            "wait_reason": (analysis.get("wait") or {}).get("reason"),
        }

    def _forced_wait_has_recovery_candidate(self) -> bool:
        """Return whether a WAIT-only state is waiting on worker recovery."""

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
            for worker in self.workers:
                if (
                    worker.state == WorkerState.IDLE
                    and module in worker.spec.qualified_modules
                    and not self._worker_can_start(reconfiguration, worker)
                ):
                    return True
        return self._earliest_worker_pair_recovery_tick() is not None

    def _record_forced_action_diagnostic(
        self,
        diagnostic: dict[str, Any] | None,
    ) -> None:
        if diagnostic is None:
            self._current_forced_action_chain = 0
            return

        if self._current_forced_action_chain == 0:
            self._forced_action_chain_count += 1
        self._current_forced_action_chain += 1
        self._longest_forced_action_chain = max(
            self._longest_forced_action_chain,
            self._current_forced_action_chain,
        )

        def increment(name: str) -> None:
            self._forced_action_counts[name] = (
                self._forced_action_counts.get(name, 0) + 1
            )

        phase = str(diagnostic["phase"]).lower()
        action_kind = str(diagnostic["action_kind"])
        increment("forced_action_state_count")
        increment(f"forced_{phase}_count")
        increment(f"forced_{action_kind}_count")
        increment(f"forced_{phase}_{action_kind}_count")
        increment(f"forced_{str(diagnostic['action_type']).lower()}_count")
        if diagnostic["wait_physically_unavailable"]:
            increment("forced_pair_wait_physically_unavailable_count")
        if diagnostic["pair_physically_unavailable"]:
            increment("forced_wait_pair_physically_unavailable_count")
        if diagnostic["phase_handoff"]:
            increment("forced_phase_handoff_count")
        if diagnostic["recovery"]:
            increment("forced_recovery_wait_count")
        if diagnostic["future_event"]:
            increment("forced_future_event_wait_count")
        stage_tags = set(diagnostic["stage_tags"])
        if ReconfigurationStage.WAIT_DIS.value in stage_tags:
            increment("forced_wait_dis_count")
        if ReconfigurationStage.WAIT_INS.value in stage_tags:
            increment("forced_wait_ins_count")
        if len(stage_tags) > 1:
            increment("forced_mixed_wait_stage_count")

    def _forced_action_metrics(self) -> dict[str, int | float]:
        count_fields = (
            "forced_action_state_count",
            "forced_production_count",
            "forced_worker_count",
            "forced_pair_count",
            "forced_wait_count",
            "forced_production_pair_count",
            "forced_production_wait_count",
            "forced_worker_pair_count",
            "forced_worker_wait_count",
            "forced_pair_wait_physically_unavailable_count",
            "forced_wait_pair_physically_unavailable_count",
            "forced_wait_dis_count",
            "forced_wait_ins_count",
            "forced_mixed_wait_stage_count",
            "forced_phase_handoff_count",
            "forced_recovery_wait_count",
            "forced_future_event_wait_count",
            "forced_direct_process_count",
            "forced_commit_reconfig_count",
            "forced_worker_assign_count",
        )
        result: dict[str, int | float] = {
            name: int(self._forced_action_counts.get(name, 0))
            for name in count_fields
        }
        result.update(
            {
                "forced_action_chain_count": self._forced_action_chain_count,
                "longest_forced_action_chain": (
                    self._longest_forced_action_chain
                ),
                "mean_forced_action_chain_length": (
                    result["forced_action_state_count"]
                    / self._forced_action_chain_count
                    if self._forced_action_chain_count > 0
                    else 0.0
                ),
            }
        )
        return result

    def _action_type(self, phase: DecisionType, action: int) -> str:
        if phase == DecisionType.PRODUCTION:
            if action == self._action_codec.production_wait:
                return "WAIT"
            operation_index, machine_index = self.decode_production_action(action)
            operation = self.operations[operation_index]
            machine = self.machines[machine_index]
            if machine.current_module == operation.spec.required_module:
                return "DIRECT_PROCESS"
            return "COMMIT_RECONFIG"
        if phase == DecisionType.WORKER:
            if action == self._action_codec.worker_wait:
                return "WAIT"
            return "WORKER_ASSIGN"
        raise RuntimeError("terminal state has no action type")

    def encode_production_action(
        self, operation_index: int, machine_index: int
    ) -> int:
        return self._action_codec.encode_production(
            operation_index, machine_index
        )

    def decode_production_action(self, action: int) -> tuple[int, int]:
        return self._action_codec.decode_production(action)

    def encode_worker_action(self, machine_index: int, worker_index: int) -> int:
        return self._action_codec.encode_worker(machine_index, worker_index)

    def decode_worker_action(self, action: int) -> tuple[int, int]:
        return self._action_codec.decode_worker(action)

    def step(
        self,
        action: int,
        *,
        build_observation: bool = True,
    ) -> tuple[Observation | None, RewardVector, bool, bool, dict[str, Any]]:
        if self.decision_type == DecisionType.TERMINAL:
            raise RuntimeError("cannot step a terminal environment")
        mask = self.get_action_mask()
        if action < 0 or action >= len(mask) or mask[action]:
            raise ValueError(
                f"illegal {self.decision_type.value} action {action} at t={self.current_time}"
            )
        self._record_forced_action_diagnostic(
            self.forced_action_diagnostic(mask)
        )
        before_tick = self.current_tick
        before = self._objective_vector()
        completed_orders_before = len(self._order_completion_tick)
        potential_before = self.feasibility_potential()
        quality_before = bounded_quality_score(
            *before,
            self.config,
            preference=self.preference,
        )
        phase = self.decision_type
        action_type = self._action_type(phase, action)
        is_wait_action = action == self.wait_action
        wait_certificate = (
            dict((self._last_action_mask_analysis or {}).get("wait") or {})
            if is_wait_action
            else {}
        )
        action_outcome: dict[str, Any] = {}
        if phase == DecisionType.WORKER:
            self._record_worker_pressure_snapshot()
        self._invalidate_resource_snapshot()
        if is_wait_action:
            action_outcome = self._execute_wait(wait_certificate)
            wait_ticks = int(action_outcome.get("wait_ticks", 0))
            wait_reason = str(action_outcome.get("wait_reason", "unknown"))
            self._wait_action_count += 1
            self._wait_reason_counts[wait_reason] = (
                self._wait_reason_counts.get(wait_reason, 0) + 1
            )
            if phase == DecisionType.PRODUCTION:
                self._production_wait_action_count += 1
                self._production_wait_ticks += wait_ticks
            else:
                self._worker_wait_action_count += 1
                self._worker_wait_ticks += wait_ticks
        elif phase == DecisionType.PRODUCTION:
            operation_index, machine_index = self.decode_production_action(action)
            self._execute_production_action(operation_index, machine_index)
        else:
            machine_index, worker_index = self.decode_worker_action(action)
            self._execute_worker_action(machine_index, worker_index)
        self._action_type_counts[action_type] = (
            self._action_type_counts.get(action_type, 0) + 1
        )
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
            self.config,
            preference=self.preference,
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
            preference_key=self.preference_context.key,
        )
        if phase == DecisionType.WORKER and not is_wait_action:
            self._worker_assignment_count += 1
            self._worker_assignment_variance_reward_sum += reward.variance
            self._worker_assignment_variance_reward_abs_sum += abs(
                reward.variance
            )
            if not math.isclose(
                reward.variance, 0.0, rel_tol=0.0, abs_tol=1e-12
            ):
                self._worker_assignment_nonzero_variance_reward_count += 1
        self._cumulative_reward += np.asarray(
            [reward.flow, reward.cost, reward.variance], dtype=np.float64
        )
        info = {
            "time": self.current_time,
            "decision_type": self.decision_type.value,
            "action_phase": phase.value,
            "action_type": action_type,
            "wait_reason": action_outcome.get("wait_reason"),
            "wait_ticks": int(action_outcome.get("wait_ticks", 0)),
            "wait_time": ticks_to_minutes(
                int(action_outcome.get("wait_ticks", 0)),
                self.resolution,
            ),
            "wait_certificate": wait_certificate or None,
            "terminal_reason": self.terminal_reason,
            "preference_key": self.preference_context.key,
        }
        observation = self.observe() if build_observation else None
        return observation, reward, self.terminated, self.truncated, info

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

    def _estimate_candidate_reconfiguration_costs(
        self,
        machine: MachineRuntime,
        target_module: str,
    ) -> tuple[float, float]:
        """Return optimistic labor and downtime costs for an action edge."""
        if machine.current_module == target_module:
            return 0.0, 0.0

        stage_specs = []
        if machine.current_module in machine.spec.module_parameters:
            stage_specs.append(
                (
                    machine.current_module,
                    machine.spec.module_parameters[
                        machine.current_module
                    ].disassembly_base_time,
                    self.instance.fatigue.disassembly_time_coefficient,
                )
            )
        stage_specs.append(
            (
                target_module,
                machine.spec.module_parameters[
                    target_module
                ].installation_base_time,
                self.instance.fatigue.installation_time_coefficient,
            )
        )
        labor_cost = 0.0
        downtime_minutes = 0.0
        for module, base_time, coefficient in stage_specs:
            candidates: list[tuple[float, float]] = []
            for worker in self.workers:
                if module not in worker.spec.qualified_modules:
                    continue
                duration_ticks = max(
                    1,
                    quantize_to_ticks(
                        base_time * (1.0 + coefficient * worker.fatigue),
                        self.resolution,
                    ),
                )
                duration = ticks_to_minutes(duration_ticks, self.resolution)
                candidates.append(
                    (duration, duration * worker.spec.labor_cost_per_minute)
                )
            if not candidates:
                continue
            downtime_minutes += min(value[0] for value in candidates)
            labor_cost += min(value[1] for value in candidates)
        return (
            labor_cost,
            downtime_minutes * machine.spec.downtime_cost_per_minute,
        )

    def _estimate_candidate_load_variance_delta(
        self,
        machine_index: int,
        target_module: str,
    ) -> float:
        """Minimum safe committed-load variance increment for a reconfiguration."""

        machine = self.machines[machine_index]
        if machine.current_module == target_module:
            return 0.0
        current = self._committed_load_variance()
        disassembly = [
            (worker_index, projection)
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
        candidates: list[float] = []
        for disassembly_worker, (start, duration_ticks) in disassembly:
            disassembly_end = start + duration_ticks
            installation = [
                (worker_index, projection)
                for worker_index in range(len(self.workers))
                if (
                    projection := self._earliest_safe_stage_projection(
                        machine_index,
                        worker_index,
                        target_module,
                        installation=True,
                        earliest_tick=disassembly_end,
                    )
                )
                is not None
            ]
            for installation_worker, (_, installation_ticks) in installation:
                loads = self._committed_worker_loads.copy()
                loads[disassembly_worker] += ticks_to_minutes(
                    duration_ticks, self.resolution
                )
                loads[installation_worker] += ticks_to_minutes(
                    installation_ticks, self.resolution
                )
                candidates.append(float(np.var(loads)) - current)
        # An unavailable reconstruction is already masked by the shared
        # feasibility projection. A neutral feature is safest for diagnostics.
        return min(candidates, default=0.0)

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

    def projected_worker_load_variance(
        self, machine_index: int, worker_index: int
    ) -> float:
        machine = self.machines[machine_index]
        reconfiguration = self._pending_reconfiguration(machine.spec.id)
        if reconfiguration is None:
            return math.inf
        worker = self.workers[worker_index]
        duration_ticks = self._stage_duration_ticks(reconfiguration, worker)
        projected_loads = self._committed_worker_loads.copy()
        projected_loads[worker_index] += ticks_to_minutes(
            duration_ticks,
            self.resolution,
        )
        return float(np.var(projected_loads))

    def metrics(self) -> dict[str, Any]:
        self._require_instance()
        completed_orders = len(self._order_completion_tick)
        completed_operations = sum(
            operation.state == OperationState.DONE
            for operation in self.operations
        )
        completion_times = {
            order.id: (
                ticks_to_minutes(
                    self._order_completion_tick[order.id], self.resolution
                )
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
                and value.disassembly_worker_id
                != value.installation_worker_id
            )
            for value in completed_reconfigurations
        ]
        resource_snapshot = self._resource_feasibility_snapshot()
        current_matching_deficit = (
            len(resource_snapshot.tasks) - resource_snapshot.matching_size
        )
        self._maximum_worker_matching_deficit = max(
            self._maximum_worker_matching_deficit,
            current_matching_deficit,
        )
        total_wait_ticks = self._production_wait_ticks + self._worker_wait_ticks
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
            "total_flow_time": self._flow_integral if self.terminated else None,
            "censored_flow_time": self._flow_integral,
            "flow_time_objective": self._flow_integral + self._flow_penalty,
            "reconfiguration_cost": self._reconfiguration_cost,
            "worker_load_variance": self._load_variance(),
            "maximum_worker_fatigue": self._maximum_fatigue_seen,
            "safe_fatigue_limit": self.instance.fatigue.maximum_safe_fatigue,
            "worker_peak_fatigue": {
                worker.spec.id: worker.peak_fatigue
                for worker in self.workers
            },
            "mean_peak_worker_fatigue": (
                float(
                    np.mean(
                        [worker.peak_fatigue for worker in self.workers]
                    )
                )
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
            "current_worker_matching_deficit": current_matching_deficit,
            "maximum_worker_matching_deficit": (
                self._maximum_worker_matching_deficit
            ),
            "minimum_worker_alternatives": (
                self._minimum_worker_alternatives_seen
                if self._minimum_worker_alternatives_seen is not None
                else len(self.workers)
            ),
            "worker_assignment_count": self._worker_assignment_count,
            "worker_assignment_variance_reward_sum": (
                self._worker_assignment_variance_reward_sum
            ),
            "worker_assignment_variance_reward_abs_sum": (
                self._worker_assignment_variance_reward_abs_sum
            ),
            "worker_assignment_nonzero_variance_reward_count": (
                self._worker_assignment_nonzero_variance_reward_count
            ),
            "action_type_counts": dict(self._action_type_counts),
            "direct_process_action_count": self._action_type_counts.get(
                "DIRECT_PROCESS", 0
            ),
            "commit_reconfig_action_count": self._action_type_counts.get(
                "COMMIT_RECONFIG", 0
            ),
            "worker_assign_action_count": self._action_type_counts.get(
                "WORKER_ASSIGN", 0
            ),
            "wait_action_count": self._wait_action_count,
            "production_wait_action_count": self._production_wait_action_count,
            "worker_wait_action_count": self._worker_wait_action_count,
            "wait_total_ticks": total_wait_ticks,
            "wait_total_time": ticks_to_minutes(
                total_wait_ticks, self.resolution
            ),
            "production_wait_ticks": self._production_wait_ticks,
            "production_wait_time": ticks_to_minutes(
                self._production_wait_ticks, self.resolution
            ),
            "worker_wait_ticks": self._worker_wait_ticks,
            "worker_wait_time": ticks_to_minutes(
                self._worker_wait_ticks, self.resolution
            ),
            "wait_reason_counts": dict(sorted(self._wait_reason_counts.items())),
            "wait_mask_reason_counts": dict(
                sorted(self._wait_mask_reason_counts.items())
            ),
            "wait_min_estimated_deadline_slack_ticks": (
                self._wait_min_estimated_deadline_slack_ticks
            ),
            "first_unrecoverable_deadlock_diagnostic": (
                dict(self._first_unrecoverable_deadlock_diagnostic)
                if self._first_unrecoverable_deadlock_diagnostic is not None
                else None
            ),
            "reconfiguration_reuse_count": self._reconfiguration_reuse_count,
            **self._forced_action_metrics(),
            "machine_waiting_for_worker_time": (
                self._machine_waiting_for_worker_time()
            ),
            "completed_reconfigurations": len(completed_reconfigurations),
            "worker_switch_indicators": switches,
            "worker_switch_ratio": (
                float(sum(switches) / len(switches)) if switches else None
            ),
            "completion_times": completion_times,
            "cumulative_reward": {
                "flow": float(self._cumulative_reward[0]),
                "cost": float(self._cumulative_reward[1]),
                "variance": float(self._cumulative_reward[2]),
            },
            "initial_objectives": {
                "flow": float(self._initial_objectives[0]),
                "cost": float(self._initial_objectives[1]),
                "variance": float(self._initial_objectives[2]),
            },
            "quality_score": bounded_quality_score(
                self._flow_integral + self._flow_penalty,
                self._reconfiguration_cost,
                self._load_variance(),
                self.config,
            ),
            "preference_quality_score": bounded_quality_score(
                self._flow_integral + self._flow_penalty,
                self._reconfiguration_cost,
                self._load_variance(),
                self.config,
                preference=self.preference,
            ),
            "preference": self.preference.as_dict(),
            "preference_context": self.preference_context.as_dict(),
            "preference_key": self.preference_context.key,
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

    def _execute_wait(self, certificate: dict[str, Any]) -> dict[str, Any]:
        """Apply the exact progress transition certified by the action mask."""

        if not certificate or not bool(certificate.get("allowed", False)):
            raise RuntimeError("WAIT action has no valid progress certificate")
        progress_kind = str(certificate["progress_kind"])
        wait_ticks = int(certificate["wait_ticks"])
        if progress_kind == "worker_phase_handoff":
            if self.decision_type != DecisionType.PRODUCTION:
                raise RuntimeError("worker phase handoff requires production phase")
            self.decision_type = DecisionType.WORKER
            return {
                "wait_reason": progress_kind,
                "wait_ticks": 0,
            }

        next_tick = int(certificate["next_tick"])
        if next_tick <= self.current_tick or wait_ticks != next_tick - self.current_tick:
            raise RuntimeError("WAIT certificate does not advance time consistently")
        self._advance_interval(next_tick)
        self.current_tick = next_tick
        self._process_events_at_current_tick()
        self._invalidate_resource_snapshot()
        self.decision_type = DecisionType.PRODUCTION
        return {
            "wait_reason": progress_kind,
            "wait_ticks": wait_ticks,
        }

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
        duration = ticks_to_minutes(duration_ticks, self.resolution)
        committed_key = (reconfiguration.id, stage_name)
        if committed_key in self._active_committed_worker_tasks:
            raise RuntimeError(
                f"duplicate committed worker task {committed_key}"
            )
        self._committed_worker_loads[worker_index] += duration
        self._active_committed_worker_tasks[committed_key] = (
            worker_index,
            duration,
        )
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
                "duration": duration,
                "fixed_cost": fixed_cost,
            }
        )

    def _start_processing(
        self, operation: OperationRuntime, machine: MachineRuntime
    ) -> None:
        if machine.spec.id in self._post_reconfiguration_process_count:
            prior_uses = self._post_reconfiguration_process_count[
                machine.spec.id
            ]
            if prior_uses >= 1:
                self._reconfiguration_reuse_count += 1
            self._post_reconfiguration_process_count[machine.spec.id] = (
                prior_uses + 1
            )
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
        self._finish_committed_worker_task(
            reconfiguration.id,
            "DIS",
            worker.spec.id,
            duration,
        )
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
        self._finish_committed_worker_task(
            reconfiguration.id,
            "INS",
            worker.spec.id,
            duration,
        )
        worker.state = WorkerState.IDLE
        worker.busy_until_tick = None
        machine.current_module = reconfiguration.target_module
        machine.state = MachineState.IDLE
        machine.busy_until_tick = None
        reconfiguration.stage = ReconfigurationStage.DONE
        self._post_reconfiguration_process_count[machine.spec.id] = 0
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

    def _worker_fatigue_at_tick(
        self, worker_index: int, tick: int
    ) -> float | None:
        worker = self.workers[worker_index]
        if worker.state == WorkerState.IDLE:
            available_tick = self.current_tick
            available_fatigue = worker.fatigue
        else:
            available_tick, available_fatigue = (
                self._worker_fatigue_at_availability(worker_index)
            )
            if available_tick > tick:
                return None
        recovery = (
            self.instance.fatigue.idle_recovery_rate_per_minute
            * ticks_to_minutes(max(0, tick - available_tick), self.resolution)
        )
        return max(0.0, available_fatigue - recovery)

    def _projected_stage_duration_ticks(
        self,
        task: WorkerTaskSnapshot,
        worker_index: int,
        tick: int,
    ) -> int:
        worker = self.workers[worker_index]
        fatigue = self._worker_fatigue_at_tick(worker_index, tick)
        if fatigue is None:
            return self.horizon_tick + 1
        temporary = ReconfigurationRuntime(
            id=task.task_id,
            machine_id=self.machines[task.machine_index].spec.id,
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
        return self._stage_duration_ticks(
            temporary, worker, fatigue_override=fatigue
        )

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
        if start_tick <= self.horizon_tick:
            lower = start_tick
            upper = self.horizon_tick
            upper_safe, _ = self._safe_stage_projection_at_tick(
                temporary,
                worker,
                available_tick=available_tick,
                available_fatigue=available_fatigue,
                recovery_rate=recovery_rate,
                tick=upper,
            )
            if upper_safe:
                while lower < upper:
                    middle = (lower + upper) // 2
                    safe, _ = self._safe_stage_projection_at_tick(
                        temporary,
                        worker,
                        available_tick=available_tick,
                        available_fatigue=available_fatigue,
                        recovery_rate=recovery_rate,
                        tick=middle,
                    )
                    if safe:
                        upper = middle
                    else:
                        lower = middle + 1
                safe, duration_ticks = self._safe_stage_projection_at_tick(
                    temporary,
                    worker,
                    available_tick=available_tick,
                    available_fatigue=available_fatigue,
                    recovery_rate=recovery_rate,
                    tick=lower,
                )
                if safe and lower + duration_ticks <= self.horizon_tick:
                    result = (lower, duration_ticks)
                    self._stage_projection_cache[key] = result
                    return result
        self._stage_projection_cache[key] = None
        return None

    def _safe_stage_projection_at_tick(
        self,
        reconfiguration: ReconfigurationRuntime,
        worker: WorkerRuntime,
        *,
        available_tick: int,
        available_fatigue: float,
        recovery_rate: float,
        tick: int,
    ) -> tuple[bool, int]:
        recovery_minutes = ticks_to_minutes(
            max(0, tick - available_tick), self.resolution
        )
        fatigue = max(
            0.0,
            available_fatigue - recovery_rate * recovery_minutes,
        )
        duration_ticks = self._stage_duration_ticks(
            reconfiguration,
            worker,
            fatigue_override=fatigue,
        )
        predicted = fatigue + self._stage_accumulation_rate(
            reconfiguration
        ) * ticks_to_minutes(duration_ticks, self.resolution)
        return (
            predicted
            <= self.instance.fatigue.maximum_safe_fatigue + EPSILON,
            duration_ticks,
        )

    def _production_resource_profile(
        self,
        machine_index: int,
        target_module: str,
    ) -> ProductionResourceProfile:
        """Return the shared resource projection for one machine/module pair."""

        key = (int(machine_index), str(target_module))
        if key in self._production_resource_profile_cache:
            return self._production_resource_profile_cache[key]
        profile = self._compute_production_resource_profile(
            machine_index, target_module
        )
        self._production_resource_profile_cache[key] = profile
        return profile

    def _compute_production_resource_profile(
        self,
        machine_index: int,
        target_module: str,
    ) -> ProductionResourceProfile:
        """Compute an uncached machine/module resource projection."""

        machine = self.machines[machine_index]
        if machine.current_module == target_module:
            return ProductionResourceProfile(
                resource_ready_tick=self.current_tick,
                processing_start_tick=self.current_tick,
                safe_disassembly_workers=len(self.workers),
                safe_installation_workers=len(self.workers),
                matching_deficit_after_commit=0,
            )

        candidate_task = WorkerTaskSnapshot(
            task_id=f"candidate:{machine_index}:{target_module}",
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
            processing_start_tick = None
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
                        target_module,
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
                processing_start_tick = (
                    installation_start + installation_ticks
                )
            else:
                processing_start_tick = None

        return ProductionResourceProfile(
            resource_ready_tick=resource_ready_tick,
            processing_start_tick=processing_start_tick,
            safe_disassembly_workers=len(candidate_edges),
            safe_installation_workers=safe_installation_workers,
            matching_deficit_after_commit=matching_deficit,
        )

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
                    profile = self._production_resource_profile(
                        machine_index,
                        operation.spec.required_module,
                    )
                    predicted_finish_tick = (
                        profile.processing_start_tick
                        + self.estimate_processing_ticks(
                            operation_index, machine_index
                        )
                        if profile.processing_start_tick is not None
                        else self.horizon_tick + 1
                    )
                    slacks.append(
                        max(
                            0.0,
                            min(
                                1.0,
                                (self.horizon_tick - predicted_finish_tick)
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
        fatigue = worker.fatigue if fatigue_override is None else fatigue_override
        base, coefficient = self._stage_duration_parameters(reconfiguration)
        return max(
            1,
            quantize_to_ticks(base * (1.0 + coefficient * fatigue), self.resolution),
        )

    def _stage_duration_parameters(
        self,
        reconfiguration: ReconfigurationRuntime,
    ) -> tuple[float, float]:
        """Return the base duration and fatigue coefficient for one stage."""

        machine = self._machine_by_id(reconfiguration.machine_id)
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
        return float(base), float(coefficient)

    def _stage_accumulation_rate(
        self, reconfiguration: ReconfigurationRuntime
    ) -> float:
        if reconfiguration.stage == ReconfigurationStage.WAIT_DIS:
            return (
                self.instance.fatigue.disassembly_accumulation_rate_per_minute
            )
        return self.instance.fatigue.installation_accumulation_rate_per_minute

    def _earliest_worker_pair_recovery_tick(self) -> int | None:
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
                        or self._worker_can_start(reconfiguration, worker)
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

    def _has_pending_worker_task(self) -> bool:
        return any(
            value.stage
            in {ReconfigurationStage.WAIT_DIS, ReconfigurationStage.WAIT_INS}
            for value in self.reconfigurations.values()
        )

    def _wait_opportunity(self) -> tuple[int, str] | None:
        """Return the earliest deterministic transition that WAIT can reach."""

        if self.current_tick >= self.horizon_tick:
            return None
        if (
            self.decision_type == DecisionType.PRODUCTION
            and self._has_pending_worker_task()
        ):
            return self.current_tick, "worker_phase_handoff"

        candidates: list[tuple[int, int, str]] = []
        future_events = [
            event for event in self._events if event[0] > self.current_tick
        ]
        if future_events:
            event_tick = min(event[0] for event in future_events)
            event_types = sorted(
                {
                    event[3].value
                    for event in future_events
                    if event[0] == event_tick
                }
            )
            candidates.append(
                (
                    event_tick,
                    0,
                    "external_event:" + "+".join(event_types),
                )
            )
        recovery_tick = self._earliest_worker_pair_recovery_tick()
        if recovery_tick is not None:
            candidates.append(
                (recovery_tick, 1, "worker_recovery_feasible")
            )
        improvement_tick = (
            self._earliest_reconfiguration_duration_improvement_tick()
        )
        if improvement_tick is not None:
            candidates.append(
                (
                    improvement_tick,
                    2,
                    "reconfiguration_duration_improved",
                )
            )
        feasible = [
            candidate
            for candidate in candidates
            if self.current_tick < candidate[0] <= self.horizon_tick
        ]
        if not feasible:
            return None
        tick, _, reason = min(feasible)
        return tick, reason

    def _wait_certificate(self) -> dict[str, Any]:
        """Certify that WAIT reaches a deterministic state transition."""

        opportunity = self._wait_opportunity()
        estimate = self._remaining_completion_estimate_ticks()
        estimated_completion_tick = self.current_tick + estimate
        estimated_deadline_slack_ticks = (
            self.horizon_tick - estimated_completion_tick
        )
        if opportunity is None:
            certificate = {
                "allowed": False,
                "reason": "no_state_progress",
                "progress_kind": "",
                "next_tick": None,
                "wait_ticks": 0,
                "remaining_completion_estimate_ticks": estimate,
                "estimated_completion_tick": estimated_completion_tick,
                "horizon_tick": self.horizon_tick,
                "estimated_deadline_slack_ticks": (
                    estimated_deadline_slack_ticks
                ),
            }
        else:
            next_tick, progress_kind = opportunity
            wait_ticks = next_tick - self.current_tick
            phase_handoff = progress_kind == "worker_phase_handoff"
            allowed = bool(
                (phase_handoff and next_tick == self.current_tick)
                or (
                    not phase_handoff
                    and self.current_tick < next_tick <= self.horizon_tick
                )
            )
            certificate = {
                "allowed": allowed,
                "reason": progress_kind if allowed else "no_state_progress",
                "progress_kind": progress_kind,
                "next_tick": next_tick,
                "wait_ticks": wait_ticks,
                "remaining_completion_estimate_ticks": estimate,
                "estimated_completion_tick": estimated_completion_tick,
                "horizon_tick": self.horizon_tick,
                "estimated_deadline_slack_ticks": (
                    estimated_deadline_slack_ticks
                ),
            }
        self._last_wait_certificate = dict(certificate)
        return certificate

    def _record_wait_mask_certificate(
        self,
        certificate: dict[str, Any],
    ) -> None:
        """Record one WAIT-mask decision per environment state."""

        reason = str(certificate["reason"])
        key = (self._state_version, reason)
        if key in self._wait_masked_states:
            return
        self._wait_masked_states.add(key)
        self._wait_mask_reason_counts[reason] = (
            self._wait_mask_reason_counts.get(reason, 0) + 1
        )
        slack = int(certificate["estimated_deadline_slack_ticks"])
        if (
            self._wait_min_estimated_deadline_slack_ticks is None
            or slack < self._wait_min_estimated_deadline_slack_ticks
        ):
            self._wait_min_estimated_deadline_slack_ticks = slack

    def _remaining_work_lower_bound_ticks(self) -> int:
        """Return an optimistic resource/precedence lower bound in ticks."""

        minimum_processing: dict[int, int] = {}
        for operation_index in range(len(self.operations)):
            values = self._capability_processing_ticks[
                self._capability_operation_indices == operation_index
            ]
            minimum_processing[operation_index] = (
                int(values.min()) if values.size else self.horizon_tick + 1
            )
        remaining_by_order: dict[str, int] = {}
        total_processing = 0
        for operation_index, operation in enumerate(self.operations):
            if operation.state == OperationState.DONE:
                continue
            ticks = minimum_processing[operation_index]
            if operation.state == OperationState.PROCESSING:
                machine = (
                    self._machine_by_id(operation.machine_id)
                    if operation.machine_id is not None
                    else None
                )
                if machine is not None and machine.busy_until_tick is not None:
                    ticks = max(0, machine.busy_until_tick - self.current_tick)
            total_processing += ticks
            remaining_by_order[operation.spec.order_id] = (
                remaining_by_order.get(operation.spec.order_id, 0) + ticks
            )
        order_chain_bound = max(remaining_by_order.values(), default=0)
        machine_bound = math.ceil(
            total_processing / max(1, len(self.machines))
        )
        active_reconfiguration_bound = max(
            (
                self._remaining_reconfiguration_ticks(reconfiguration)
                for reconfiguration in self.reconfigurations.values()
                if reconfiguration.stage != ReconfigurationStage.DONE
            ),
            default=0,
        )
        return int(
            max(order_chain_bound, machine_bound, active_reconfiguration_bound)
        )

    def _minimum_module_transition_ticks(
        self,
        source_module: str,
        target_module: str,
    ) -> int:
        """Return an optimistic per-chain module transition estimate.

        A transition is not charged when an idle or busy machine already carries
        the target module: that machine could complete its current work in
        parallel and then service the successor operation.
        """
        if source_module == target_module:
            return 0
        if any(
            machine.current_module == target_module
            for machine in self.machines
        ):
            return 0
        candidates: list[int] = []
        for machine in self.machines:
            parameters = machine.spec.module_parameters
            if target_module not in parameters:
                continue
            if (
                source_module != self.instance.no_module_state
                and source_module not in parameters
            ):
                continue
            if not any(
                target_module in worker.spec.qualified_modules
                for worker in self.workers
            ):
                continue
            if (
                source_module != self.instance.no_module_state
                and not any(
                    source_module in worker.spec.qualified_modules
                    for worker in self.workers
                )
            ):
                continue
            disassembly = (
                0
                if source_module == self.instance.no_module_state
                else max(
                    1,
                    quantize_to_ticks(
                        parameters[source_module].disassembly_base_time,
                        self.resolution,
                    ),
                )
            )
            installation = max(
                1,
                quantize_to_ticks(
                    parameters[target_module].installation_base_time,
                    self.resolution,
                ),
            )
            candidates.append(disassembly + installation)
        return min(candidates, default=self.horizon_tick + 1)

    def _remaining_completion_estimate_ticks(self) -> int:
        """Return a soft completion estimate including remaining module chains.

        Cross-order module transitions may be shared or overlap with processing,
        so this value is diagnostic guidance rather than a certified lower bound.
        """
        baseline = self._remaining_work_lower_bound_ticks()
        chain_bounds: list[int] = []
        for order in self.instance.orders:
            pending = [
                operation
                for operation in order.operations
                if self.operations[self.instance.operation_index[operation.id]].state
                != OperationState.DONE
            ]
            if not pending:
                continue
            chain = 0
            previous_module = self.instance.no_module_state
            for operation in pending:
                runtime = self.operations[
                    self.instance.operation_index[operation.id]
                ]
                if runtime.state == OperationState.PROCESSING:
                    machine = (
                        self._machine_by_id(runtime.machine_id)
                        if runtime.machine_id is not None
                        else None
                    )
                    chain += (
                        max(0, machine.busy_until_tick - self.current_tick)
                        if machine is not None and machine.busy_until_tick is not None
                        else 0
                    )
                    previous_module = operation.required_module
                    continue
                values = self._capability_processing_ticks[
                    self._capability_operation_indices
                    == self.instance.operation_index[operation.id]
                ]
                chain += (
                    int(values.min()) if values.size else self.horizon_tick + 1
                )
                chain += self._minimum_module_transition_ticks(
                    previous_module,
                    operation.required_module,
                )
                previous_module = operation.required_module
            chain_bounds.append(chain)
        return int(max(baseline, max(chain_bounds, default=0)))

    def _earliest_reconfiguration_duration_improvement_tick(
        self,
    ) -> int | None:
        """Find the first tick where idle recovery lowers a legal stage duration."""
        if (
            self._reconfiguration_duration_improvement_cache_version
            == self._state_version
        ):
            return self._reconfiguration_duration_improvement_cache
        self._reconfiguration_duration_improvement_cache_version = (
            self._state_version
        )
        self._reconfiguration_duration_improvement_cache = None
        recovery_rate = self.instance.fatigue.idle_recovery_rate_per_minute
        if recovery_rate <= 0.0 or self.current_tick >= self.horizon_tick:
            return None

        scan_end_tick = min(
            self.horizon_tick,
            min(
                [
                    event[0]
                    for event in self._events
                    if event[0] > self.current_tick
                ]
                or [self.horizon_tick]
            ),
        )
        mismatch_candidates: list[tuple[MachineRuntime, str, int]] = []
        for operation_index, operation in enumerate(self.operations):
            if operation.state != OperationState.READY:
                continue
            for machine_index, machine in enumerate(self.machines):
                if (
                    machine.state != MachineState.IDLE
                    or machine.current_module == self.instance.no_module_state
                    or machine.current_module == operation.spec.required_module
                    or operation.spec.required_module
                    not in machine.spec.module_parameters
                ):
                    continue
                current_duration = self._idle_worker_reconfiguration_ticks_at(
                    machine,
                    operation.spec.required_module,
                    self.current_tick,
                )
                if current_duration is not None:
                    mismatch_candidates.append(
                        (machine, operation.spec.required_module, current_duration)
                    )

        stage_candidates: list[
            tuple[ReconfigurationRuntime, WorkerRuntime, int]
        ] = []
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
            for worker in self.workers:
                if (
                    worker.state == WorkerState.IDLE
                    and module in worker.spec.qualified_modules
                    and self._worker_can_start(reconfiguration, worker)
                ):
                    stage_candidates.append(
                        (
                            reconfiguration,
                            worker,
                            self._stage_duration_ticks(
                                reconfiguration, worker
                            ),
                        )
                    )

        for tick in range(self.current_tick + 1, scan_end_tick + 1):
            for machine, target_module, current_duration in mismatch_candidates:
                duration = self._idle_worker_reconfiguration_ticks_at(
                    machine,
                    target_module,
                    tick,
                )
                if duration is not None and duration < current_duration:
                    self._reconfiguration_duration_improvement_cache = tick
                    return tick
            elapsed = ticks_to_minutes(
                tick - self.current_tick, self.resolution
            )
            for reconfiguration, worker, current_duration in stage_candidates:
                fatigue = max(
                    0.0,
                    worker.fatigue
                    - recovery_rate * elapsed,
                )
                duration = self._stage_duration_ticks(
                    reconfiguration,
                    worker,
                    fatigue_override=fatigue,
                )
                if duration < current_duration:
                    self._reconfiguration_duration_improvement_cache = tick
                    return tick
        return None

    def _idle_worker_reconfiguration_ticks_at(
        self,
        machine: MachineRuntime,
        target_module: str,
        tick: int,
    ) -> int | None:
        source_module = machine.current_module
        stage_specs = (
            (source_module, False),
            (target_module, True),
        )
        elapsed = ticks_to_minutes(tick - self.current_tick, self.resolution)
        recovery_rate = self.instance.fatigue.idle_recovery_rate_per_minute
        total_ticks = 0
        for module, installation in stage_specs:
            if module == self.instance.no_module_state:
                continue
            qualified_workers = [
                worker
                for worker in self.workers
                if worker.state == WorkerState.IDLE
                and module in worker.spec.qualified_modules
            ]
            if not qualified_workers:
                return None
            minimum_fatigue = min(
                max(0.0, worker.fatigue - recovery_rate * elapsed)
                for worker in qualified_workers
            )
            module_parameters = machine.spec.module_parameters[module]
            if installation:
                base_time = module_parameters.installation_base_time
                coefficient = self.instance.fatigue.installation_time_coefficient
            else:
                base_time = module_parameters.disassembly_base_time
                coefficient = self.instance.fatigue.disassembly_time_coefficient
            total_ticks += max(
                1,
                quantize_to_ticks(
                    base_time * (1.0 + coefficient * minimum_fatigue),
                    self.resolution,
                ),
            )
        return total_ticks

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
                self._record_unrecoverable_deadlock_diagnostic()
                self._truncate_at_horizon("unrecoverable_deadlock")

    def _record_unrecoverable_deadlock_diagnostic(self) -> None:
        if self._first_unrecoverable_deadlock_diagnostic is not None:
            return
        unfinished = [
            {
                "operation_id": operation.spec.id,
                "order_id": operation.spec.order_id,
                "required_module": operation.spec.required_module,
                "state": operation.state.value,
            }
            for operation in self.operations
            if operation.state != OperationState.DONE
        ]
        self._first_unrecoverable_deadlock_diagnostic = {
            "state_version": int(self._state_version),
            "time": float(self.current_time),
            "tick": int(self.current_tick),
            "unfinished_operations": unfinished,
            "machine_modules": {
                machine.spec.id: machine.current_module
                for machine in self.machines
            },
            "completion_estimate_ticks": (
                self._remaining_completion_estimate_ticks()
            ),
            "estimated_completion_slack_ticks": (
                self.horizon_tick
                - self.current_tick
                - self._remaining_completion_estimate_ticks()
            ),
            "certificate_reason": (
                self._last_wait_certificate or {}
            ).get("reason"),
            "wait_certificate": dict(self._last_wait_certificate or {}),
        }

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
            stage_name = (
                "DIS"
                if reconfiguration.stage == ReconfigurationStage.DIS
                else "INS"
            )
            self._settle_committed_worker_task(
                reconfiguration.id,
                stage_name,
                worker.spec.id,
                duration,
            )
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
            self._committed_load_variance(),
        )

    def _load_variance(self) -> float:
        loads = np.asarray([worker.load for worker in self.workers], dtype=np.float64)
        return float(np.var(loads)) if len(loads) else 0.0

    def _committed_load_variance(self) -> float:
        return (
            float(np.var(self._committed_worker_loads))
            if len(self._committed_worker_loads)
            else 0.0
        )

    def _finish_committed_worker_task(
        self,
        reconfiguration_id: str,
        stage_name: str,
        worker_id: str,
        duration: float,
    ) -> None:
        key = (reconfiguration_id, stage_name)
        committed = self._active_committed_worker_tasks.pop(key, None)
        if committed is None:
            raise RuntimeError(f"missing committed worker task {key}")
        worker_index, planned_duration = committed
        if self.workers[worker_index].spec.id != worker_id:
            raise RuntimeError(f"committed worker mismatch for {key}")
        if not math.isclose(
            planned_duration,
            duration,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"committed duration mismatch for {key}")

    def _settle_committed_worker_task(
        self,
        reconfiguration_id: str,
        stage_name: str,
        worker_id: str,
        worked_duration: float,
    ) -> None:
        key = (reconfiguration_id, stage_name)
        committed = self._active_committed_worker_tasks.pop(key, None)
        if committed is None:
            raise RuntimeError(f"missing committed worker task {key}")
        worker_index, planned_duration = committed
        if self.workers[worker_index].spec.id != worker_id:
            raise RuntimeError(f"committed worker mismatch for {key}")
        if worked_duration < -1e-12 or worked_duration > planned_duration + 1e-9:
            raise RuntimeError(f"invalid worked duration for {key}")
        self._committed_worker_loads[worker_index] -= (
            planned_duration - worked_duration
        )

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
