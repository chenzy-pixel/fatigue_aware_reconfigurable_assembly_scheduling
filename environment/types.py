from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .preference import (
    CANONICAL_PREFERENCE,
    PREFERENCE_NAMES,
    PreferenceInput,
    normalize_preference,
)


EdgeType = tuple[str, str, str]

PRECEDES_EDGE: EdgeType = ("operation", "precedes", "operation")
CAPABLE_EDGE: EdgeType = ("operation", "capable_on", "machine")
LOCKED_EDGE: EdgeType = ("operation", "locked_to", "machine")
CAN_INSTALL_EDGE: EdgeType = ("worker", "can_install", "operation")
CAN_DISASSEMBLE_EDGE: EdgeType = (
    "worker",
    "can_disassemble",
    "machine",
)
OPERATION_ORDER_EDGE: EdgeType = ("operation", "belongs_to", "order")
ORDER_WAVE_EDGE: EdgeType = ("order", "belongs_to", "wave")
REQUIRES_MODULE_EDGE: EdgeType = ("operation", "requires", "module")
MACHINE_MODULE_EDGE: EdgeType = ("machine", "supports", "module")
WORKER_MODULE_EDGE: EdgeType = ("worker", "qualified_for", "module")
WAVE_MODULE_EDGE: EdgeType = ("wave", "demands", "module")
SERVICE_CANDIDATE_EDGE: EdgeType = (
    "machine",
    "service_candidate",
    "worker",
)

ASSEMBLY_EDGE_TYPES: tuple[EdgeType, ...] = (
    PRECEDES_EDGE,
    CAPABLE_EDGE,
    LOCKED_EDGE,
    CAN_INSTALL_EDGE,
    CAN_DISASSEMBLE_EDGE,
    OPERATION_ORDER_EDGE,
    ORDER_WAVE_EDGE,
    REQUIRES_MODULE_EDGE,
    MACHINE_MODULE_EDGE,
    WORKER_MODULE_EDGE,
    WAVE_MODULE_EDGE,
    SERVICE_CANDIDATE_EDGE,
)

ASSEMBLY_NODE_TYPES: tuple[str, ...] = (
    "operation",
    "machine",
    "worker",
    "order",
    "module",
    "wave",
)


class OperationState(str, Enum):
    UNRELEASED = "UNRELEASED"
    BLOCKED = "BLOCKED"
    READY = "READY"
    LOCKED = "LOCKED"
    PROCESSING = "PROCESSING"
    DONE = "DONE"


class MachineState(str, Enum):
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    WAIT_DIS = "WAIT_DIS"
    DIS = "DIS"
    WAIT_INS = "WAIT_INS"
    INS = "INS"


class WorkerState(str, Enum):
    IDLE = "IDLE"
    DIS = "DIS"
    INS = "INS"


class DecisionType(str, Enum):
    PRODUCTION = "PRODUCTION"
    WORKER = "WORKER"
    TERMINAL = "TERMINAL"


class ReconfigurationStage(str, Enum):
    WAIT_DIS = "WAIT_DIS"
    DIS = "DIS"
    WAIT_INS = "WAIT_INS"
    INS = "INS"
    DONE = "DONE"


class EventType(str, Enum):
    ORDER_RELEASE = "ORDER_RELEASE"
    PROCESS_COMPLETE = "PROCESS_COMPLETE"
    DIS_COMPLETE = "DIS_COMPLETE"
    INS_COMPLETE = "INS_COMPLETE"


@dataclass(frozen=True)
class RewardVector:
    flow: float
    cost: float
    variance: float
    operation_progress: float = 0.0
    quality: float = 0.0
    feasibility_shaping: float = 0.0
    preference_key: str | None = None

    def scalarize(self, config: dict) -> float:
        """Return the configured single-stage training reward."""

        base = self.base_scalarize(config)
        shaping = config.get("feasibility_shaping", {})
        if not isinstance(shaping, dict):
            raise TypeError("reward.feasibility_shaping must be an object")
        if bool(shaping.get("enabled", False)):
            return base + self.feasibility_shaping
        return base

    def base_scalarize(self, config: dict) -> float:
        """Return progress increment plus preference-conditioned quality delta."""

        mode = str(config.get("mode", "single_stage_progress_quality_v1"))
        if mode != "single_stage_progress_quality_v1":
            raise ValueError(f"unknown reward mode {mode!r}")
        return self.operation_progress + self.quality

    def as_dict(self) -> dict[str, float | str]:
        result = {
            "flow": self.flow,
            "cost": self.cost,
            "variance": self.variance,
            "operation_progress": self.operation_progress,
            "quality": self.quality,
            "feasibility_shaping": self.feasibility_shaping,
        }
        if self.preference_key is not None:
            result["preference_key"] = self.preference_key
        return result


def objective_scalarizer_config(config: dict) -> dict:
    """Return one validated scalarizer definition from a full or local config."""

    raw = config.get("objective_scalarizer", config)
    if not isinstance(raw, dict):
        raise TypeError("objective_scalarizer must be an object")
    scales_value = raw.get("scales")
    if scales_value is None:
        scales = {
            name: float(raw[f"{name}_scale"])
            for name in PREFERENCE_NAMES
        }
    else:
        if not isinstance(scales_value, dict):
            raise TypeError("objective_scalarizer.scales must be an object")
        if set(scales_value) != set(PREFERENCE_NAMES):
            raise ValueError(
                "objective_scalarizer.scales must contain flow/cost/variance"
            )
        scales = {
            name: float(scales_value[name]) for name in PREFERENCE_NAMES
        }
    if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise ValueError("objective scalarizer scales must be finite and positive")
    default_kind = (
        "normalized_augmented_tchebycheff_v1"
        if str(raw.get("mode", "")) == "single_stage_progress_quality_v1"
        else "legacy_bounded_weighted_sum_v1"
    )
    kind = str(raw.get("type", default_kind))
    rho = float(raw.get("rho", 0.05))
    if kind == "normalized_augmented_tchebycheff_v1":
        if not np.isfinite(rho) or rho < 0.0:
            raise ValueError("objective_scalarizer.rho must be finite and non-negative")
    return {"type": kind, "scales": scales, "rho": rho}


def reward_config(config: dict) -> dict:
    """Return the reward subsection from a full or reward-local config."""

    raw = config.get("reward", config)
    if not isinstance(raw, dict):
        raise TypeError("reward must be an object")
    return raw


def bounded_quality_score(
    flow: float,
    cost: float,
    variance: float,
    config: dict,
    *,
    preference: PreferenceInput | None = None,
) -> float:
    """Return the configured normalized scalar objective (smaller is better)."""
    values = {
        "flow": float(flow),
        "cost": float(cost),
        "variance": float(variance),
    }
    scalarizer = objective_scalarizer_config(config)
    weights = (
        normalize_preference(preference).as_dict()
        if preference is not None
        else normalize_preference(
            config.get(
                "quality_weights",
                config.get("preference", {})
                .get("quality", {})
                .get("fixed", CANONICAL_PREFERENCE),
            )
        ).as_dict()
    )
    scales = scalarizer["scales"]
    weight_sum = sum(float(weights[name]) for name in values)
    if weight_sum <= 0.0:
        raise ValueError("quality weights must have a positive sum")
    normalized: dict[str, float] = {}
    for name, value in values.items():
        weight = float(weights[name])
        scale = scales[name]
        if value < 0.0:
            raise ValueError(f"{name} objective cannot be negative")
        if weight < 0.0:
            raise ValueError(f"{name} quality weight cannot be negative")
        if scale <= 0.0:
            raise ValueError(f"{name} quality scale must be positive")
        normalized[name] = value / (scale + value)
    if scalarizer["type"] == "normalized_augmented_tchebycheff_v1":
        rho = float(scalarizer["rho"])
        weighted = [
            float(weights[name]) * normalized[name]
            for name in PREFERENCE_NAMES
        ]
        return (max(weighted) + rho * sum(weighted)) / (1.0 + rho)
    if scalarizer["type"] != "legacy_bounded_weighted_sum_v1":
        raise ValueError(f"unknown objective scalarizer {scalarizer['type']!r}")
    return sum(
        float(weights[name]) * normalized[name] for name in PREFERENCE_NAMES
    ) / weight_sum


def terminal_quality_score(
    flow: float,
    cost: float,
    variance: float,
    config: dict,
    *,
    preference: PreferenceInput | None = None,
    terminal_failure: bool = False,
) -> float:
    """Return the formal terminal score, using one for any failed episode."""

    if terminal_failure:
        return 1.0
    return bounded_quality_score(
        flow,
        cost,
        variance,
        config,
        preference=preference,
    )


def proxy_return_from_metrics(
    metrics: dict,
    config: dict,
    *,
    preference: PreferenceInput | None = None,
) -> float:
    """Recompute ``P_T - P_0 - Q_T + Q_0`` from trajectory metrics."""

    reward = reward_config(config)
    mode = str(reward.get("mode", "single_stage_progress_quality_v1"))
    if mode != "single_stage_progress_quality_v1":
        raise ValueError(f"unknown reward mode {mode!r}")
    initial_progress = float(metrics.get("initial_progress", 0.0))
    terminal_progress = float(metrics["operation_progress"])
    initial_score_value = metrics.get("initial_preference_quality_score")
    if initial_score_value is None:
        initial = metrics.get(
            "initial_objectives",
            {"flow": 0.0, "cost": 0.0, "variance": 0.0},
        )
        if not isinstance(initial, dict):
            raise TypeError("metrics.initial_objectives must be an object")
        initial_score = bounded_quality_score(
            float(initial.get("flow", 0.0)),
            float(initial.get("cost", 0.0)),
            float(initial.get("variance", 0.0)),
            config,
            preference=preference,
        )
    else:
        initial_score = float(initial_score_value)
    terminal_score_value = metrics.get("preference_quality_score")
    terminal_score = (
        terminal_quality_score(
            float(metrics["flow_time_objective"]),
            float(metrics["reconfiguration_cost"]),
            float(metrics["worker_load_variance"]),
            config,
            preference=preference,
            terminal_failure=bool(metrics.get("task_failed", metrics["truncated"])),
        )
        if terminal_score_value is None
        else float(terminal_score_value)
    )
    return (
        terminal_progress
        - initial_progress
        - terminal_score
        + initial_score
    )


@dataclass(frozen=True)
class EdgeStore:
    edge_index: np.ndarray
    edge_features: np.ndarray
    feature_names: tuple[str, ...]
    bidirectional: bool = False

    def __post_init__(self) -> None:
        edge_index = np.asarray(self.edge_index, dtype=np.int64)
        edge_features = np.asarray(self.edge_features, dtype=np.float32)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if edge_features.ndim != 2:
            raise ValueError("edge_features must have shape (E, F)")
        if edge_index.shape[1] != edge_features.shape[0]:
            raise ValueError(
                "edge_index and edge_features must contain the same edge count"
            )
        if edge_features.shape[1] != len(self.feature_names):
            raise ValueError(
                "edge feature width must match the number of feature names"
            )
        if edge_index.size and np.any(edge_index < 0):
            raise ValueError("edge indices must be non-negative")
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "edge_features", edge_features)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def copy(self) -> "EdgeStore":
        return EdgeStore(
            edge_index=self.edge_index.copy(),
            edge_features=self.edge_features.copy(),
            feature_names=self.feature_names,
            bidirectional=self.bidirectional,
        )


@dataclass(frozen=True)
class HeterogeneousGraphObservation:
    node_features: dict[str, np.ndarray]
    global_features: np.ndarray
    decision_type: DecisionType
    preference: np.ndarray = field(
        default_factory=lambda: np.asarray(CANONICAL_PREFERENCE, dtype=np.float32)
    )
    node_feature_names: dict[str, tuple[str, ...]] = field(default_factory=dict)
    global_feature_names: tuple[str, ...] = field(default_factory=tuple)
    node_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    relations: dict[EdgeType, EdgeStore] = field(default_factory=dict)
    action_set_features: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    action_set_feature_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized_features = {
            str(node_type): np.asarray(features, dtype=np.float32)
            for node_type, features in self.node_features.items()
        }
        object.__setattr__(self, "node_features", normalized_features)
        object.__setattr__(
            self,
            "global_features",
            np.asarray(self.global_features, dtype=np.float32),
        )
        object.__setattr__(
            self,
            "preference",
            normalize_preference(self.preference).as_array(),
        )
        object.__setattr__(
            self,
            "node_feature_names",
            {
                str(node_type): tuple(names)
                for node_type, names in self.node_feature_names.items()
            },
        )
        object.__setattr__(
            self,
            "action_set_features",
            np.asarray(self.action_set_features, dtype=np.float32),
        )
        object.__setattr__(
            self,
            "action_set_feature_names",
            tuple(self.action_set_feature_names),
        )

    @property
    def operations(self) -> np.ndarray:
        return self.node_features["operation"]

    @property
    def machines(self) -> np.ndarray:
        return self.node_features["machine"]

    @property
    def workers(self) -> np.ndarray:
        return self.node_features["worker"]

    @property
    def orders(self) -> np.ndarray:
        return self.node_features["order"]

    @property
    def modules(self) -> np.ndarray:
        return self.node_features["module"]

    @property
    def waves(self) -> np.ndarray:
        return self.node_features["wave"]

    def copy(self) -> "HeterogeneousGraphObservation":
        return HeterogeneousGraphObservation(
            node_features={
                node_type: features.copy()
                for node_type, features in self.node_features.items()
            },
            global_features=self.global_features.copy(),
            decision_type=self.decision_type,
            preference=self.preference.copy(),
            node_feature_names={
                node_type: tuple(names)
                for node_type, names in self.node_feature_names.items()
            },
            global_feature_names=tuple(self.global_feature_names),
            node_ids={
                node_type: tuple(identifiers)
                for node_type, identifiers in self.node_ids.items()
            },
            relations={
                edge_type: edge_store.copy()
                for edge_type, edge_store in self.relations.items()
            },
            action_set_features=self.action_set_features.copy(),
            action_set_feature_names=tuple(self.action_set_feature_names),
        )

    @property
    def feature_dimensions(self) -> dict[str, int]:
        dimensions = {
            node_type: int(features.shape[-1])
            for node_type, features in self.node_features.items()
        }
        dimensions["global"] = int(self.global_features.shape[-1])
        return dimensions

    @property
    def edge_feature_dimensions(self) -> dict[EdgeType, int]:
        return {
            edge_type: int(edge_store.edge_features.shape[1])
            for edge_type, edge_store in self.relations.items()
        }

    def validate(self) -> None:
        node_features = self.node_features
        if self.global_features.ndim != 1:
            raise ValueError("global features must have shape (F,)")
        if self.preference.shape != (3,) or not np.all(np.isfinite(self.preference)):
            raise ValueError("preference must have shape (3,) with finite values")
        if self.action_set_features.ndim != 1:
            raise ValueError("action-set features must have shape (F,)")
        if len(self.action_set_feature_names) != self.action_set_features.shape[0]:
            raise ValueError(
                "action-set feature width must match the number of names"
            )
        if not np.all(np.isfinite(self.action_set_features)):
            raise ValueError("action-set features must be finite")
        if (
            self.global_feature_names
            and len(self.global_feature_names) != self.global_features.shape[0]
        ):
            raise ValueError(
                "global feature width must match the number of feature names"
            )
        expected_node_types = set(ASSEMBLY_NODE_TYPES)
        if set(node_features) != expected_node_types:
            raise ValueError(
                "node_features must contain exactly the six M1 node types"
            )
        if self.node_feature_names and (
            set(self.node_feature_names) != expected_node_types
        ):
            raise ValueError(
                "node_feature_names must contain exactly the six M1 node types"
            )
        if self.node_ids and set(self.node_ids) != expected_node_types:
            raise ValueError(
                "node_ids must contain exactly the six M1 node types"
            )
        for node_type, features in node_features.items():
            if features.ndim != 2:
                raise ValueError(f"{node_type} features must have shape (N, F)")
            identifiers = self.node_ids.get(node_type)
            if identifiers is not None:
                if len(identifiers) != features.shape[0]:
                    raise ValueError(
                        f"{node_type} node id count does not match feature rows"
                    )
                if len(set(identifiers)) != len(identifiers):
                    raise ValueError(f"{node_type} node ids must be unique")
            names = self.node_feature_names.get(node_type)
            if names is not None and len(names) != features.shape[1]:
                raise ValueError(
                    f"{node_type} feature width does not match feature names"
                )
            if not np.all(np.isfinite(features)):
                raise ValueError(f"{node_type} features must be finite")
        if self.relations and set(self.relations) != set(ASSEMBLY_EDGE_TYPES):
            raise ValueError(
                "relations must contain exactly the M1 graph edge types"
            )
        for edge_type, edge_store in self.relations.items():
            source_type, _, target_type = edge_type
            if source_type not in node_features or target_type not in node_features:
                raise ValueError(f"unknown node type in edge relation {edge_type}")
            if edge_store.num_edges == 0:
                continue
            if np.any(
                edge_store.edge_index[0] >= node_features[source_type].shape[0]
            ):
                raise ValueError(f"source edge index out of range for {edge_type}")
            if np.any(
                edge_store.edge_index[1] >= node_features[target_type].shape[0]
            ):
                raise ValueError(f"target edge index out of range for {edge_type}")
            source = edge_store.edge_index[0]
            target = edge_store.edge_index[1]
            order = np.lexsort((target, source))
            if not np.array_equal(order, np.arange(edge_store.num_edges)):
                raise ValueError(f"edge relation {edge_type} is not stably sorted")


@dataclass(frozen=True)
class PolicyObservation:
    """Compact MLP policy input without graph metadata unused by the network."""

    operations: np.ndarray
    machines: np.ndarray
    workers: np.ndarray
    global_features: np.ndarray
    decision_type: DecisionType
    preference: np.ndarray = field(
        default_factory=lambda: np.asarray(CANONICAL_PREFERENCE, dtype=np.float32)
    )
    global_feature_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operations", np.asarray(self.operations, dtype=np.float32)
        )
        object.__setattr__(
            self, "machines", np.asarray(self.machines, dtype=np.float32)
        )
        object.__setattr__(
            self, "workers", np.asarray(self.workers, dtype=np.float32)
        )
        object.__setattr__(
            self,
            "global_features",
            np.asarray(self.global_features, dtype=np.float32),
        )
        object.__setattr__(
            self,
            "preference",
            normalize_preference(self.preference).as_array(),
        )
        object.__setattr__(
            self, "global_feature_names", tuple(self.global_feature_names)
        )

    @classmethod
    def from_observation(
        cls,
        observation: "HeterogeneousGraphObservation | PolicyObservation",
    ) -> "PolicyObservation":
        if isinstance(observation, cls):
            return observation.copy()
        return cls(
            operations=observation.operations.copy(),
            machines=observation.machines.copy(),
            workers=observation.workers.copy(),
            global_features=observation.global_features.copy(),
            decision_type=observation.decision_type,
            preference=observation.preference.copy(),
            global_feature_names=tuple(observation.global_feature_names),
        )

    def copy(self) -> "PolicyObservation":
        return PolicyObservation(
            operations=self.operations.copy(),
            machines=self.machines.copy(),
            workers=self.workers.copy(),
            global_features=self.global_features.copy(),
            decision_type=self.decision_type,
            preference=self.preference.copy(),
            global_feature_names=tuple(self.global_feature_names),
        )

    @property
    def feature_dimensions(self) -> dict[str, int]:
        return {
            "operation": int(self.operations.shape[-1]),
            "machine": int(self.machines.shape[-1]),
            "worker": int(self.workers.shape[-1]),
            "global": int(self.global_features.shape[-1]),
        }


# Backward-compatible public name used by the existing MLP/PPO path.
Observation = HeterogeneousGraphObservation
