from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


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
    completion_progress: float = 0.0
    completion_bonus: float = 0.0
    quality: float = 0.0
    truncation: float = 0.0
    unfinished: float = 0.0
    feasibility_shaping: float = 0.0

    def scalarize(self, config: dict, phase: str | None = None) -> float:
        base = self.base_scalarize(config, phase)
        mode = str(config.get("mode", "legacy_weighted_sum"))
        if mode == "hierarchical_constrained_v1":
            return base + self.feasibility_shaping
        return base

    def base_scalarize(self, config: dict, phase: str | None = None) -> float:
        """Return the formal objective reward without training-only shaping."""
        mode = str(config.get("mode", "legacy_weighted_sum"))
        if mode == "hierarchical_constrained_v1":
            effective_phase = "feasibility" if phase is None else str(phase)
            truncation_weight = float(config.get("truncation_penalty", 0.0))
            unfinished_weight = float(
                config.get("unfinished_order_penalty", 0.0)
            )
            if truncation_weight < 0.0 or unfinished_weight < 0.0:
                raise ValueError(
                    "hierarchical terminal penalty weights must be non-negative"
                )
            base = (
                self.completion_progress
                + self.completion_bonus
                + truncation_weight * self.truncation
                + unfinished_weight * self.unfinished
            )
            if effective_phase == "feasibility":
                return base
            if effective_phase == "quality":
                budget = float(config["quality_budget"])
                if not 0.0 <= budget < 1.0:
                    raise ValueError("quality_budget must be in [0, 1)")
                return base + budget * self.quality
            raise ValueError(f"unknown hierarchical reward phase {effective_phase!r}")
        if mode != "legacy_weighted_sum":
            raise ValueError(f"unknown reward mode {mode!r}")
        return (
            config["flow_weight"] * self.flow / config["flow_scale"]
            + config["cost_weight"] * self.cost / config["cost_scale"]
            + config["variance_weight"]
            * self.variance
            / config["variance_scale"]
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "flow": self.flow,
            "cost": self.cost,
            "variance": self.variance,
            "completion_progress": self.completion_progress,
            "completion_bonus": self.completion_bonus,
            "quality": self.quality,
            "truncation": self.truncation,
            "unfinished": self.unfinished,
            "feasibility_shaping": self.feasibility_shaping,
        }


def bounded_quality_score(
    flow: float,
    cost: float,
    variance: float,
    config: dict,
) -> float:
    """Return the bounded weighted quality proxy in [0, 1)."""
    values = {
        "flow": float(flow),
        "cost": float(cost),
        "variance": float(variance),
    }
    weights = config.get(
        "quality_weights",
        {
            "flow": float(config.get("flow_weight", 1.0)),
            "cost": float(config.get("cost_weight", 1.0)),
            "variance": float(config.get("variance_weight", 1.0)),
        },
    )
    scales = {
        "flow": float(config["flow_scale"]),
        "cost": float(config["cost_scale"]),
        "variance": float(config["variance_scale"]),
    }
    weight_sum = sum(float(weights[name]) for name in values)
    if weight_sum <= 0.0:
        raise ValueError("quality weights must have a positive sum")
    score = 0.0
    for name, value in values.items():
        weight = float(weights[name])
        scale = scales[name]
        if value < 0.0:
            raise ValueError(f"{name} objective cannot be negative")
        if weight < 0.0:
            raise ValueError(f"{name} quality weight cannot be negative")
        if scale <= 0.0:
            raise ValueError(f"{name} quality scale must be positive")
        score += weight * value / (scale + value)
    return score / weight_sum


def proxy_return_from_metrics(
    metrics: dict,
    config: dict,
    phase: str | None = None,
) -> float:
    """Recompute the trajectory proxy return from terminal metrics."""
    mode = str(config.get("mode", "legacy_weighted_sum"))
    if mode == "legacy_weighted_sum":
        return -(
            float(config["flow_weight"])
            * float(metrics["flow_time_objective"])
            / float(config["flow_scale"])
            + float(config["cost_weight"])
            * float(metrics["reconfiguration_cost"])
            / float(config["cost_scale"])
            + float(config["variance_weight"])
            * float(metrics["worker_load_variance"])
            / float(config["variance_scale"])
        )
    if mode != "hierarchical_constrained_v1":
        raise ValueError(f"unknown reward mode {mode!r}")
    total_orders = int(metrics["total_orders"])
    if total_orders <= 0:
        raise ValueError("total_orders must be positive")
    completion_progress = float(metrics["completed_orders"]) / total_orders
    completion_bonus = float(
        bool(metrics["terminated"]) and not bool(metrics["truncated"])
    )
    truncated = float(bool(metrics["truncated"]))
    unfinished_fraction = (
        float(metrics["unfinished_orders"]) / total_orders
        if bool(metrics["truncated"])
        else 0.0
    )
    truncation_weight = float(config.get("truncation_penalty", 0.0))
    unfinished_weight = float(
        config.get("unfinished_order_penalty", 0.0)
    )
    if truncation_weight < 0.0 or unfinished_weight < 0.0:
        raise ValueError(
            "hierarchical terminal penalty weights must be non-negative"
        )
    effective_phase = "feasibility" if phase is None else str(phase)
    result = (
        completion_progress
        + completion_bonus
        - truncation_weight * truncated
        - unfinished_weight * unfinished_fraction
    )
    if effective_phase == "feasibility":
        return result
    if effective_phase != "quality":
        raise ValueError(f"unknown hierarchical reward phase {effective_phase!r}")
    quality = bounded_quality_score(
        float(metrics["flow_time_objective"]),
        float(metrics["reconfiguration_cost"]),
        float(metrics["worker_load_variance"]),
        config,
    )
    budget = float(config["quality_budget"])
    if not 0.0 <= budget < 1.0:
        raise ValueError("quality_budget must be in [0, 1)")
    return result - budget * quality


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
    node_feature_names: dict[str, tuple[str, ...]] = field(default_factory=dict)
    global_feature_names: tuple[str, ...] = field(default_factory=tuple)
    node_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    relations: dict[EdgeType, EdgeStore] = field(default_factory=dict)

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
            "node_feature_names",
            {
                str(node_type): tuple(names)
                for node_type, names in self.node_feature_names.items()
            },
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
    global_feature_names: tuple[str, ...] = field(default_factory=tuple)

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
            global_feature_names=tuple(observation.global_feature_names),
        )

    def copy(self) -> "PolicyObservation":
        return PolicyObservation(
            operations=self.operations.copy(),
            machines=self.machines.copy(),
            workers=self.workers.copy(),
            global_features=self.global_features.copy(),
            decision_type=self.decision_type,
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
