from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from environment import (
    ASSEMBLY_EDGE_TYPES,
    ASSEMBLY_NODE_TYPES,
    CAPABLE_EDGE,
    LOCKED_EDGE,
    MACHINE_MODULE_EDGE,
    OPERATION_ORDER_EDGE,
    ORDER_WAVE_EDGE,
    REQUIRES_MODULE_EDGE,
    SERVICE_CANDIDATE_EDGE,
    WAVE_MODULE_EDGE,
    WORKER_MODULE_EDGE,
    DecisionType,
    EdgeType,
    HeterogeneousGraphObservation,
    Observation,
    PolicyObservation,
)


PolicyInput = Observation | PolicyObservation


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )


class TypedActorCritic(nn.Module):
    """Typed entity encoders with two pair-scoring actor heads and one critic."""

    requires_graph_observation = False

    def __init__(self, feature_dimensions: dict[str, int], hidden_dim: int):
        super().__init__()
        self.feature_dimensions = {
            key: int(feature_dimensions[key])
            for key in ("operation", "machine", "worker", "global")
        }
        self.hidden_dim = int(hidden_dim)
        self.operation_encoder = _mlp(
            self.feature_dimensions["operation"], hidden_dim
        )
        self.machine_encoder = _mlp(
            self.feature_dimensions["machine"], hidden_dim
        )
        self.worker_encoder = _mlp(
            self.feature_dimensions["worker"], hidden_dim
        )
        self.global_encoder = _mlp(
            self.feature_dimensions["global"], hidden_dim
        )
        self.production_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.worker_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.production_defer = nn.Linear(hidden_dim, 1)
        self.worker_advance = nn.Linear(hidden_dim, 1)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def network_spec(self) -> dict[str, Any]:
        return {
            "encoder_type": "typed_mlp",
            "hidden_dim": self.hidden_dim,
            "policy_head_version": 5,
            "production_action_semantics": "pair_plus_defer_v1",
            "observation_schema_version": 3,
            "feature_dimensions": dict(self.feature_dimensions),
        }

    def forward(
        self,
        observation: PolicyInput,
        action_mask: np.ndarray | torch.Tensor,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = self.forward_batch(
            [observation],
            [action_mask],
            device=device,
        )
        action_count = int(np.asarray(action_mask).shape[0])
        return logits[0, :action_count], values[0]

    def forward_batch(
        self,
        observations: Sequence[PolicyInput],
        action_masks: Sequence[np.ndarray | torch.Tensor],
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate mixed, variable-size observations in padded typed batches."""
        if not observations:
            raise ValueError("observation batch cannot be empty")
        if len(observations) != len(action_masks):
            raise ValueError(
                "observation and action-mask batch sizes must match"
            )
        if any(
            observation.decision_type == DecisionType.TERMINAL
            for observation in observations
        ):
            raise ValueError("actor cannot evaluate a terminal observation")
        operation_features, operation_valid = self._pad_entity_features(
            observations,
            "operations",
            device=device,
        )
        machine_features, machine_valid = self._pad_entity_features(
            observations,
            "machines",
            device=device,
        )
        worker_features, worker_valid = self._pad_entity_features(
            observations,
            "workers",
            device=device,
        )
        global_features = torch.stack(
            [
                torch.as_tensor(
                    observation.global_features,
                    dtype=torch.float32,
                    device=device,
                )
                for observation in observations
            ],
            dim=0,
        )
        operation_embeddings = self.operation_encoder(operation_features)
        machine_embeddings = self.machine_encoder(machine_features)
        worker_embeddings = self.worker_encoder(worker_features)
        global_embeddings = self.global_encoder(global_features)

        action_counts = []
        masks = []
        for action_mask in action_masks:
            mask = torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=device,
            )
            if mask.ndim != 1:
                raise ValueError("each action mask must be one-dimensional")
            if bool(mask.all()):
                raise ValueError("at least one action must be feasible")
            action_counts.append(int(mask.shape[0]))
            masks.append(mask)
        maximum_action_count = max(action_counts)
        minimum_logit = torch.finfo(operation_embeddings.dtype).min
        logits = torch.full(
            (len(observations), maximum_action_count),
            minimum_logit,
            dtype=operation_embeddings.dtype,
            device=device,
        )
        groups: dict[
            tuple[DecisionType, int, int],
            list[int],
        ] = defaultdict(list)
        for index, observation in enumerate(observations):
            groups[
                (
                    observation.decision_type,
                    int(observation.machines.shape[0]),
                    int(observation.workers.shape[0]),
                )
            ].append(index)
        for (decision_type, machine_count, worker_count), indices in groups.items():
            batch_indices = torch.as_tensor(
                indices,
                dtype=torch.long,
                device=device,
            )
            group_global = global_embeddings.index_select(0, batch_indices)
            if decision_type == DecisionType.PRODUCTION:
                operation_count = max(
                    int(observations[index].operations.shape[0])
                    for index in indices
                )
                group_operations = operation_embeddings.index_select(
                    0,
                    batch_indices,
                )[:, :operation_count]
                group_machines = machine_embeddings.index_select(
                    0,
                    batch_indices,
                )[:, :machine_count]
                operation_pairs = group_operations[:, :, None, :].expand(
                    -1,
                    operation_count,
                    machine_count,
                    -1,
                )
                machine_pairs = group_machines[:, None, :, :].expand(
                    -1,
                    operation_count,
                    machine_count,
                    -1,
                )
                global_pairs = group_global[:, None, None, :].expand(
                    -1,
                    operation_count,
                    machine_count,
                    -1,
                )
                pair_logits = self.production_scorer(
                    torch.cat(
                        (operation_pairs, machine_pairs, global_pairs),
                        dim=-1,
                    )
                ).reshape(len(indices), -1)
                defer_logits = self.production_defer(
                    group_global
                ).squeeze(-1)
                for local_index, observation_index in enumerate(indices):
                    pair_count = (
                        int(observations[observation_index].operations.shape[0])
                        * machine_count
                    )
                    expected_count = pair_count + 1
                    if action_counts[observation_index] != expected_count:
                        raise ValueError(
                            "production action mask does not match entity counts"
                        )
                    logits[observation_index, :pair_count] = pair_logits[
                        local_index,
                        :pair_count,
                    ]
                    logits[observation_index, pair_count] = defer_logits[
                        local_index
                    ]
            elif decision_type == DecisionType.WORKER:
                group_machines = machine_embeddings.index_select(
                    0,
                    batch_indices,
                )[:, :machine_count]
                group_workers = worker_embeddings.index_select(
                    0,
                    batch_indices,
                )[:, :worker_count]
                machine_pairs = group_machines[:, :, None, :].expand(
                    -1,
                    machine_count,
                    worker_count,
                    -1,
                )
                worker_pairs = group_workers[:, None, :, :].expand(
                    -1,
                    machine_count,
                    worker_count,
                    -1,
                )
                global_pairs = group_global[:, None, None, :].expand(
                    -1,
                    machine_count,
                    worker_count,
                    -1,
                )
                pair_logits = self.worker_scorer(
                    torch.cat(
                        (machine_pairs, worker_pairs, global_pairs),
                        dim=-1,
                    )
                ).reshape(len(indices), -1)
                advance_logits = self.worker_advance(
                    group_global
                ).squeeze(-1)
                pair_count = machine_count * worker_count
                for local_index, observation_index in enumerate(indices):
                    expected_count = pair_count + 1
                    if action_counts[observation_index] != expected_count:
                        raise ValueError(
                            "worker action mask does not match entity counts"
                        )
                    logits[observation_index, :pair_count] = pair_logits[
                        local_index
                    ]
                    logits[observation_index, pair_count] = advance_logits[
                        local_index
                    ]
            else:
                raise ValueError(
                    f"unsupported decision type {decision_type}"
                )
        for index, mask in enumerate(masks):
            logits[index, : action_counts[index]] = logits[
                index,
                : action_counts[index],
            ].masked_fill(mask, minimum_logit)

        pooled = torch.cat(
            (
                self._masked_mean(operation_embeddings, operation_valid),
                self._masked_mean(machine_embeddings, machine_valid),
                self._masked_mean(worker_embeddings, worker_valid),
                global_embeddings,
            ),
            dim=-1,
        )
        values = self.critic(pooled).squeeze(-1)
        return logits, values

    @staticmethod
    def _pad_entity_features(
        observations: Sequence[PolicyInput],
        attribute: str,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        arrays = [getattr(observation, attribute) for observation in observations]
        maximum_count = max(int(array.shape[0]) for array in arrays)
        feature_width = int(arrays[0].shape[1])
        if any(
            array.ndim != 2 or int(array.shape[1]) != feature_width
            for array in arrays
        ):
            raise ValueError(
                f"{attribute} feature widths must match within a batch"
            )
        features = torch.zeros(
            (len(arrays), maximum_count, feature_width),
            dtype=torch.float32,
            device=device,
        )
        valid = torch.zeros(
            (len(arrays), maximum_count),
            dtype=torch.bool,
            device=device,
        )
        for index, array in enumerate(arrays):
            count = int(array.shape[0])
            features[index, :count] = torch.as_tensor(
                array,
                dtype=torch.float32,
                device=device,
            )
            valid[index, :count] = True
        return features, valid

    @staticmethod
    def _masked_mean(
        embeddings: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        weights = valid.unsqueeze(-1).to(embeddings.dtype)
        counts = weights.sum(dim=1).clamp_min(1.0)
        return (embeddings * weights).sum(dim=1) / counts


NODE_TYPES: tuple[str, ...] = ASSEMBLY_NODE_TYPES
BIDIRECTIONAL_EDGE_TYPES: frozenset[EdgeType] = frozenset(
    (
        CAPABLE_EDGE,
        LOCKED_EDGE,
        OPERATION_ORDER_EDGE,
        ORDER_WAVE_EDGE,
        REQUIRES_MODULE_EDGE,
        MACHINE_MODULE_EDGE,
        WORKER_MODULE_EDGE,
        WAVE_MODULE_EDGE,
        SERVICE_CANDIDATE_EDGE,
    )
)


def _relation_key(edge_type: EdgeType) -> str:
    return "__".join(edge_type)


def normalize_network_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize an actor-critic architecture configuration."""
    if not isinstance(config, Mapping):
        raise TypeError("network config must be a mapping")
    encoder_type = str(config.get("encoder_type", "typed_mlp"))
    if encoder_type not in {"typed_mlp", "hetero_gnn"}:
        raise ValueError(
            "network.encoder_type must be 'typed_mlp' or 'hetero_gnn'"
        )
    if "hidden_dim" not in config:
        raise ValueError("network.hidden_dim is required")
    hidden_dim = int(config["hidden_dim"])
    if hidden_dim < 1:
        raise ValueError("network.hidden_dim must be positive")
    normalized: dict[str, Any] = {
        "encoder_type": encoder_type,
        "hidden_dim": hidden_dim,
    }
    if encoder_type == "hetero_gnn":
        message_passing_layers = int(
            config.get("message_passing_layers", 2)
        )
        dropout = float(config.get("dropout", 0.0))
        if message_passing_layers < 1:
            raise ValueError(
                "network.message_passing_layers must be positive"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("network.dropout must be in [0, 1)")
        normalized.update(
            {
                "message_passing_layers": message_passing_layers,
                "dropout": dropout,
                "production_action_edge_features": bool(
                    config.get("production_action_edge_features", False)
                ),
                "worker_action_edge_features": bool(
                    config.get("worker_action_edge_features", False)
                ),
                "production_candidate_relative_features": bool(
                    config.get(
                        "production_candidate_relative_features", False
                    )
                ),
                "worker_candidate_relative_features": bool(
                    config.get("worker_candidate_relative_features", False)
                ),
            }
        )
    return normalized


def network_requires_graph_observation(
    config: Mapping[str, Any],
) -> bool:
    return normalize_network_config(config)["encoder_type"] == "hetero_gnn"


def assert_network_config_matches_spec(
    config: Mapping[str, Any],
    checkpoint_spec: Mapping[str, Any],
) -> None:
    configured = normalize_network_config(config)
    saved = normalize_network_config(checkpoint_spec)
    if configured != saved:
        raise ValueError(
            "network configuration does not match checkpoint architecture: "
            f"configured={configured}, checkpoint={saved}"
        )
    if configured["encoder_type"] == "hetero_gnn":
        configured_policy_head = int(config.get("policy_head_version", 5))
        saved_policy_head = int(
            checkpoint_spec.get("policy_head_version", 3)
        )
        if configured_policy_head != saved_policy_head:
            raise ValueError(
                "checkpoint policy head version is incompatible with the "
                "current network: "
                f"configured={configured_policy_head}, "
                f"checkpoint={saved_policy_head}; v4 and older checkpoints "
                "used production advance semantics and cannot be loaded as "
                "the v5 production defer policy"
            )
        configured_semantics = str(
            config.get(
                "production_action_semantics",
                "pair_plus_defer_v1",
            )
        )
        saved_semantics = str(
            checkpoint_spec.get(
                "production_action_semantics",
                "pair_plus_advance_v0",
            )
        )
        if configured_semantics != saved_semantics:
            raise ValueError(
                "checkpoint production action semantics are incompatible "
                "with the current network: "
                f"configured={configured_semantics}, "
                f"checkpoint={saved_semantics}"
            )
        for field in (
            "production_relative_feature_names",
            "worker_relative_feature_names",
        ):
            configured_names = tuple(config.get(field, ()))
            saved_names = tuple(checkpoint_spec.get(field, ()))
            if configured_names != saved_names:
                raise ValueError(
                    f"checkpoint {field} is incompatible with the current "
                    f"network: configured={configured_names}, "
                    f"checkpoint={saved_names}"
                )
        configured_context_mode = str(
            config.get("candidate_context_mode", "per_candidate_v3")
        )
        saved_context_mode = str(
            checkpoint_spec.get(
                "candidate_context_mode",
                "per_candidate_v3",
            )
        )
        if configured_context_mode != saved_context_mode:
            raise ValueError(
                "checkpoint candidate_context_mode is incompatible with "
                "the current network: "
                f"configured={configured_context_mode}, "
                f"checkpoint={saved_context_mode}"
            )
        configured_worker_sharing = str(
            config.get(
                "worker_relative_weight_sharing",
                "independent_v3",
            )
        )
        saved_worker_sharing = str(
            checkpoint_spec.get(
                "worker_relative_weight_sharing",
                "independent_v3",
            )
        )
        if configured_worker_sharing != saved_worker_sharing:
            raise ValueError(
                "checkpoint worker_relative_weight_sharing is "
                "incompatible with the current network: "
                f"configured={configured_worker_sharing}, "
                f"checkpoint={saved_worker_sharing}"
            )
    configured_features = config.get("feature_dimensions")
    saved_features = checkpoint_spec.get("feature_dimensions")
    configured_edges = config.get("edge_feature_dimensions")
    saved_edges = checkpoint_spec.get("edge_feature_dimensions")
    configured_schema = int(config.get("observation_schema_version", 1))
    saved_schema = int(checkpoint_spec.get("observation_schema_version", 1))
    if (
        configured_schema != saved_schema
        or configured_features != saved_features
        or configured_edges != saved_edges
    ):
        raise ValueError(
            "checkpoint observation schema is incompatible with the current "
            "environment: "
            f"configured_schema={configured_schema}, "
            f"checkpoint_schema={saved_schema}, "
            f"configured_features={configured_features}, "
            f"checkpoint_features={saved_features}, "
            f"configured_edges={configured_edges}, "
            f"checkpoint_edges={saved_edges}"
        )


def _infer_observation_dimensions(
    checkpoint: Mapping[str, Any],
    encoder_type: str,
) -> tuple[dict[str, int] | None, dict[EdgeType, int] | None]:
    state = checkpoint.get("network")
    if not isinstance(state, Mapping):
        return None, None
    if encoder_type == "typed_mlp":
        prefixes = {
            "operation": "operation_encoder.0.weight",
            "machine": "machine_encoder.0.weight",
            "worker": "worker_encoder.0.weight",
            "global": "global_encoder.0.weight",
        }
    else:
        checkpoint_node_types = (
            NODE_TYPES
            if "node_projectors.order.0.weight" in state
            else ("operation", "machine", "worker")
        )
        prefixes = {
            node_type: f"node_projectors.{node_type}.0.weight"
            for node_type in checkpoint_node_types
        }
        prefixes["global"] = "global_encoder.0.weight"
    features: dict[str, int] = {}
    for name, key in prefixes.items():
        weight = state.get(key)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            return None, None
        features[name] = int(weight.shape[1])
    if encoder_type != "hetero_gnn":
        return features, None
    hidden_dim = int(state["node_projectors.operation.0.weight"].shape[0])
    edges: dict[EdgeType, int] = {}
    for edge_type in ASSEMBLY_EDGE_TYPES:
        key = (
            "message_layers.0.relation_transforms."
            f"{_relation_key(edge_type)}.weight"
        )
        weight = state.get(key)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            return features, None
        edges[edge_type] = int(weight.shape[1]) - hidden_dim
    return features, edges


def infer_checkpoint_network_spec(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Read a new network spec or infer the legacy TypedActorCritic format."""
    if "network_spec" in checkpoint:
        raw_spec = checkpoint["network_spec"]
        if not isinstance(raw_spec, Mapping):
            raise ValueError("checkpoint network_spec must be a mapping")
        spec = dict(raw_spec)
        architecture = normalize_network_config(spec)
        features, edges = _infer_observation_dimensions(
            checkpoint,
            architecture["encoder_type"],
        )
        spec.update(architecture)
        spec.setdefault("observation_schema_version", 1)
        if features is not None:
            spec.setdefault("feature_dimensions", features)
        if edges is not None:
            spec.setdefault("edge_feature_dimensions", edges)
        return spec
    state = checkpoint.get("network")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a network state")
    legacy_weight = state.get("operation_encoder.0.weight")
    if (
        not isinstance(legacy_weight, torch.Tensor)
        or legacy_weight.ndim != 2
    ):
        raise ValueError(
            "checkpoint has no network_spec and is not a recognized "
            "legacy TypedActorCritic checkpoint"
        )
    spec = {
        "encoder_type": "typed_mlp",
        "hidden_dim": int(legacy_weight.shape[0]),
        "observation_schema_version": 1,
    }
    features, _ = _infer_observation_dimensions(checkpoint, "typed_mlp")
    if features is not None:
        spec["feature_dimensions"] = features
    return spec


def _head(
    input_dim: int,
    hidden_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


RelationBatch = tuple[torch.Tensor, torch.Tensor, bool]


@dataclass(frozen=True)
class _GraphBatch:
    node_features: dict[str, torch.Tensor]
    node_slices: dict[str, list[tuple[int, int]]]
    relations: dict[EdgeType, RelationBatch]
    global_features: torch.Tensor


class HeterogeneousMessagePassingLayer(nn.Module):
    """One pure-PyTorch relation-aware residual message-passing layer."""

    def __init__(
        self,
        hidden_dim: int,
        edge_feature_dimensions: Mapping[EdgeType, int],
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.relation_transforms = nn.ModuleDict(
            {
                _relation_key(edge_type): nn.Linear(
                    self.hidden_dim
                    + int(edge_feature_dimensions[edge_type]),
                    self.hidden_dim,
                )
                for edge_type in ASSEMBLY_EDGE_TYPES
            }
        )
        self.layer_norms = nn.ModuleDict(
            {
                node_type: nn.LayerNorm(self.hidden_dim)
                for node_type in NODE_TYPES
            }
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        node_embeddings: dict[str, torch.Tensor],
        relations: Mapping[EdgeType, RelationBatch],
    ) -> dict[str, torch.Tensor]:
        aggregated = {
            node_type: torch.zeros_like(embedding)
            for node_type, embedding in node_embeddings.items()
        }
        degrees = {
            node_type: embedding.new_zeros(
                (embedding.shape[0], 1)
            )
            for node_type, embedding in node_embeddings.items()
        }
        for edge_type in ASSEMBLY_EDGE_TYPES:
            source_type, _, target_type = edge_type
            if (
                source_type not in node_embeddings
                or target_type not in node_embeddings
            ):
                continue
            edge_index, edge_features, bidirectional = relations[edge_type]
            if edge_index.shape[1] == 0:
                continue
            source_indices = edge_index[0]
            target_indices = edge_index[1]
            transform = self.relation_transforms[_relation_key(edge_type)]
            forward_messages = transform(
                torch.cat(
                    (
                        node_embeddings[source_type].index_select(
                            0, source_indices
                        ),
                        edge_features,
                    ),
                    dim=-1,
                )
            )
            aggregated[target_type] = aggregated[target_type].index_add(
                0,
                target_indices,
                forward_messages,
            )
            degrees[target_type] = degrees[target_type].index_add(
                0,
                target_indices,
                forward_messages.new_ones(
                    (forward_messages.shape[0], 1)
                ),
            )
            if bidirectional:
                reverse_messages = transform(
                    torch.cat(
                        (
                            node_embeddings[target_type].index_select(
                                0, target_indices
                            ),
                            edge_features,
                        ),
                        dim=-1,
                    )
                )
                aggregated[source_type] = aggregated[source_type].index_add(
                    0,
                    source_indices,
                    reverse_messages,
                )
                degrees[source_type] = degrees[source_type].index_add(
                    0,
                    source_indices,
                    reverse_messages.new_ones(
                        (reverse_messages.shape[0], 1)
                    ),
                )
        updated = {}
        for node_type in node_embeddings:
            mean_messages = aggregated[node_type] / degrees[
                node_type
            ].clamp_min(1.0)
            residual = node_embeddings[node_type] + mean_messages
            updated[node_type] = self.dropout(
                functional.relu(self.layer_norms[node_type](residual))
            )
        return updated


class HeteroGraphActorCritic(nn.Module):
    """Relation-aware actor-critic for the assembly heterogeneous graph."""

    requires_graph_observation = True

    def __init__(
        self,
        feature_dimensions: Mapping[str, int],
        edge_feature_dimensions: Mapping[EdgeType, int],
        hidden_dim: int,
        message_passing_layers: int,
        dropout: float,
        production_action_edge_features: bool = False,
        worker_action_edge_features: bool = False,
        production_candidate_relative_features: bool = False,
        worker_candidate_relative_features: bool = False,
    ):
        super().__init__()
        self.feature_dimensions = {
            key: int(value) for key, value in feature_dimensions.items()
        }
        self.edge_feature_dimensions = {
            edge_type: int(width)
            for edge_type, width in edge_feature_dimensions.items()
        }
        if set(self.feature_dimensions) != set((*NODE_TYPES, "global")):
            raise ValueError(
                "feature_dimensions must contain the six M1 node types and global"
            )
        if set(self.edge_feature_dimensions) != set(ASSEMBLY_EDGE_TYPES):
            raise ValueError(
                "edge_feature_dimensions must contain exactly the M1 relations"
            )
        self.hidden_dim = int(hidden_dim)
        self.message_passing_layer_count = int(message_passing_layers)
        self.dropout_probability = float(dropout)
        self.use_production_action_edge_features = bool(
            production_action_edge_features
        )
        self.use_worker_action_edge_features = bool(
            worker_action_edge_features
        )
        self.use_production_candidate_relative_features = bool(
            production_candidate_relative_features
        )
        self.use_worker_candidate_relative_features = bool(
            worker_candidate_relative_features
        )
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.message_passing_layer_count < 1:
            raise ValueError("message_passing_layers must be positive")
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.node_projectors = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(
                        self.feature_dimensions[node_type],
                        self.hidden_dim,
                    ),
                    nn.ReLU(),
                )
                for node_type in NODE_TYPES
            }
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(
                self.feature_dimensions["global"],
                self.hidden_dim,
            ),
            nn.ReLU(),
        )
        self.message_layers = nn.ModuleList(
            [
                HeterogeneousMessagePassingLayer(
                    self.hidden_dim,
                    self.edge_feature_dimensions,
                    self.dropout_probability,
                )
                for _ in range(self.message_passing_layer_count)
            ]
        )
        self.production_action_edge_encoder = nn.Sequential(
            nn.Linear(
                self.edge_feature_dimensions[CAPABLE_EDGE], self.hidden_dim
            ),
            nn.ReLU(),
        )
        self.worker_action_edge_encoder = nn.Sequential(
            nn.Linear(
                self.edge_feature_dimensions[SERVICE_CANDIDATE_EDGE],
                self.hidden_dim,
            ),
            nn.ReLU(),
        )
        self.production_scorer = _head(
            self.hidden_dim * 4,
            self.hidden_dim,
            self.dropout_probability,
        )
        self.worker_scorer = _head(
            self.hidden_dim * 5,
            self.hidden_dim,
            self.dropout_probability,
        )
        if self.use_production_candidate_relative_features:
            self.production_relative_ranker = nn.Linear(1, 1, bias=False)
            self._initialize_monotone_ranker(
                self.production_relative_ranker
            )
            self._initialize_zero_context_output(self.production_scorer)
            self.production_context_gate = nn.Parameter(
                torch.full((), -4.0)
            )
        else:
            self.production_relative_ranker = None
            self.register_parameter("production_context_gate", None)
        if self.use_worker_candidate_relative_features:
            self.worker_relative_ranker = nn.Linear(2, 1, bias=False)
            self._initialize_monotone_ranker(self.worker_relative_ranker)
            self._initialize_zero_context_output(self.worker_scorer)
            self.worker_context_gate = nn.Parameter(torch.full((), -4.0))
        else:
            self.worker_relative_ranker = None
            self.register_parameter("worker_context_gate", None)
        context_dim = self.hidden_dim * (len(NODE_TYPES) + 1)
        self.production_defer = _head(
            context_dim,
            self.hidden_dim,
            self.dropout_probability,
        )
        self.worker_advance = _head(
            context_dim,
            self.hidden_dim,
            self.dropout_probability,
        )
        self.critic = _head(
            context_dim,
            self.hidden_dim,
            self.dropout_probability,
        )

    def network_spec(self) -> dict[str, Any]:
        return {
            "encoder_type": "hetero_gnn",
            "hidden_dim": self.hidden_dim,
            "message_passing_layers": self.message_passing_layer_count,
            "dropout": self.dropout_probability,
            "production_action_edge_features": (
                self.use_production_action_edge_features
            ),
            "worker_action_edge_features": self.use_worker_action_edge_features,
            "production_candidate_relative_features": (
                self.use_production_candidate_relative_features
            ),
            "worker_candidate_relative_features": (
                self.use_worker_candidate_relative_features
            ),
            "policy_head_version": 5,
            "production_action_semantics": "pair_plus_defer_v1",
            "production_relative_feature_names": (
                ("processing_plus_reconfiguration_time_norm",)
                if self.use_production_candidate_relative_features
                else ()
            ),
            "worker_relative_feature_names": (
                (
                    "projected_fatigue_ratio",
                    "incremental_load_variance_norm",
                )
                if self.use_worker_candidate_relative_features
                else ()
            ),
            "candidate_context_mode": "common_offset_v4",
            "worker_relative_weight_sharing": "shared_mean_v4",
            "observation_schema_version": 3,
            "feature_dimensions": dict(self.feature_dimensions),
            "edge_feature_dimensions": dict(self.edge_feature_dimensions),
        }

    def forward(
        self,
        observation: HeterogeneousGraphObservation,
        action_mask: np.ndarray | torch.Tensor,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = self.forward_batch(
            [observation],
            [action_mask],
            device=device,
        )
        action_count = int(np.asarray(action_mask).shape[0])
        return logits[0, :action_count], values[0]

    def forward_batch(
        self,
        observations: Sequence[HeterogeneousGraphObservation],
        action_masks: Sequence[np.ndarray | torch.Tensor],
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not observations:
            raise ValueError("observation batch cannot be empty")
        if len(observations) != len(action_masks):
            raise ValueError(
                "observation and action-mask batch sizes must match"
            )
        if any(
            not isinstance(observation, HeterogeneousGraphObservation)
            for observation in observations
        ):
            raise TypeError(
                "HeteroGraphActorCritic requires full heterogeneous graph "
                "observations"
            )
        if any(
            observation.decision_type == DecisionType.TERMINAL
            for observation in observations
        ):
            raise ValueError("actor cannot evaluate a terminal observation")

        graph_batch = self._collate_graphs(observations, device=device)
        node_embeddings = {
            node_type: self.node_projectors[node_type](
                graph_batch.node_features[node_type]
            )
            for node_type in NODE_TYPES
        }
        for layer in self.message_layers:
            node_embeddings = layer(
                node_embeddings,
                graph_batch.relations,
            )
        global_embeddings = self.global_encoder(
            graph_batch.global_features
        )
        pooled = {
            node_type: self._pool_slices(
                node_embeddings[node_type],
                graph_batch.node_slices[node_type],
            )
            for node_type in NODE_TYPES
        }
        context = torch.cat(
            tuple(pooled[node_type] for node_type in NODE_TYPES)
            + (global_embeddings,),
            dim=-1,
        )
        values = self.critic(context).squeeze(-1)

        masks = [
            self._validate_action_mask(mask, device=device)
            for mask in action_masks
        ]
        action_counts = [int(mask.shape[0]) for mask in masks]
        maximum_action_count = max(action_counts)
        minimum_logit = torch.finfo(values.dtype).min
        logits = values.new_full(
            (len(observations), maximum_action_count),
            minimum_logit,
        )
        for batch_index, observation in enumerate(observations):
            operation_embeddings = self._node_slice(
                node_embeddings["operation"],
                graph_batch.node_slices["operation"][batch_index],
            )
            machine_embeddings = self._node_slice(
                node_embeddings["machine"],
                graph_batch.node_slices["machine"][batch_index],
            )
            worker_embeddings = self._node_slice(
                node_embeddings["worker"],
                graph_batch.node_slices["worker"][batch_index],
            )
            if observation.decision_type == DecisionType.PRODUCTION:
                pair_logits = self._production_logits(
                    observation,
                    operation_embeddings,
                    machine_embeddings,
                    global_embeddings[batch_index],
                    masks[batch_index],
                    device=device,
                )
                terminal_logit = self.production_defer(
                    context[batch_index]
                ).reshape(1)
            elif observation.decision_type == DecisionType.WORKER:
                pair_logits = self._worker_logits(
                    observation,
                    operation_embeddings,
                    machine_embeddings,
                    worker_embeddings,
                    global_embeddings[batch_index],
                    masks[batch_index],
                    device=device,
                )
                terminal_logit = self.worker_advance(
                    context[batch_index]
                ).reshape(1)
            else:
                raise ValueError(
                    f"unsupported decision type {observation.decision_type}"
                )
            unmasked_logits = torch.cat((pair_logits, terminal_logit))
            if unmasked_logits.shape[0] != action_counts[batch_index]:
                raise ValueError(
                    f"{observation.decision_type.value.lower()} action mask "
                    "does not match entity counts"
                )
            logits[batch_index, : action_counts[batch_index]] = (
                unmasked_logits.masked_fill(
                    masks[batch_index],
                    minimum_logit,
                )
            )
        return logits, values

    def _collate_graphs(
        self,
        observations: Sequence[HeterogeneousGraphObservation],
        *,
        device: torch.device | str,
    ) -> _GraphBatch:
        node_features: dict[str, torch.Tensor] = {}
        node_slices: dict[str, list[tuple[int, int]]] = {}
        for node_type in NODE_TYPES:
            arrays = [
                observation.node_features[node_type]
                for observation in observations
            ]
            expected_width = self.feature_dimensions[node_type]
            if any(
                array.ndim != 2
                or int(array.shape[1]) != expected_width
                for array in arrays
            ):
                raise ValueError(
                    f"{node_type} feature width does not match the network"
                )
            starts_and_ends = []
            offset = 0
            tensors = []
            for array in arrays:
                count = int(array.shape[0])
                starts_and_ends.append((offset, offset + count))
                offset += count
                tensors.append(
                    torch.as_tensor(
                        array,
                        dtype=torch.float32,
                        device=device,
                    )
                )
            node_features[node_type] = torch.cat(tensors, dim=0)
            node_slices[node_type] = starts_and_ends

        relations: dict[EdgeType, RelationBatch] = {}
        for edge_type in ASSEMBLY_EDGE_TYPES:
            source_type, _, target_type = edge_type
            index_parts = []
            feature_parts = []
            expected_bidirectional = edge_type in BIDIRECTIONAL_EDGE_TYPES
            for batch_index, observation in enumerate(observations):
                if set(observation.relations) != set(ASSEMBLY_EDGE_TYPES):
                    raise ValueError(
                        "graph observation must contain exactly the M1 "
                        "assembly relations"
                    )
                edge_store = observation.relations[edge_type]
                if edge_store.bidirectional != expected_bidirectional:
                    raise ValueError(
                        f"unexpected bidirectional flag for {edge_type}"
                    )
                expected_width = self.edge_feature_dimensions[edge_type]
                if edge_store.edge_features.shape != (
                    edge_store.num_edges,
                    expected_width,
                ):
                    raise ValueError(
                        f"edge feature width does not match for {edge_type}"
                    )
                edge_index = torch.as_tensor(
                    edge_store.edge_index,
                    dtype=torch.long,
                    device=device,
                ).clone()
                edge_index[0] += node_slices[source_type][batch_index][0]
                edge_index[1] += node_slices[target_type][batch_index][0]
                index_parts.append(edge_index)
                feature_parts.append(
                    torch.as_tensor(
                        edge_store.edge_features,
                        dtype=torch.float32,
                        device=device,
                    )
                )
            relations[edge_type] = (
                torch.cat(index_parts, dim=1),
                torch.cat(feature_parts, dim=0),
                expected_bidirectional,
            )
        global_features = torch.stack(
            [
                torch.as_tensor(
                    observation.global_features,
                    dtype=torch.float32,
                    device=device,
                )
                for observation in observations
            ]
        )
        if global_features.shape[1] != self.feature_dimensions["global"]:
            raise ValueError(
                "global feature width does not match the network"
            )
        return _GraphBatch(
            node_features=node_features,
            node_slices=node_slices,
            relations=relations,
            global_features=global_features,
        )

    def _production_logits(
        self,
        observation: HeterogeneousGraphObservation,
        operation_embeddings: torch.Tensor,
        machine_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        operation_count = operation_embeddings.shape[0]
        machine_count = machine_embeddings.shape[0]
        operation_pairs = operation_embeddings[:, None, :].expand(
            operation_count,
            machine_count,
            -1,
        )
        machine_pairs = machine_embeddings[None, :, :].expand(
            operation_count,
            machine_count,
            -1,
        )
        global_pairs = global_embedding[None, None, :].expand(
            operation_count,
            machine_count,
            -1,
        )
        pair_count = operation_count * machine_count
        if action_mask.shape[0] != pair_count + 1:
            raise ValueError(
                "production action mask does not match entity counts"
            )
        capable = observation.relations[CAPABLE_EDGE]
        dense_features = operation_pairs.new_zeros(
            (
                pair_count,
                self.edge_feature_dimensions[CAPABLE_EDGE],
            )
        )
        if capable.num_edges:
            edge_index = torch.as_tensor(
                capable.edge_index,
                dtype=torch.long,
                device=device,
            )
            action_indices = edge_index[0] * machine_count + edge_index[1]
            edge_features = torch.as_tensor(
                capable.edge_features,
                dtype=torch.float32,
                device=device,
            )
            dense_features = dense_features.index_copy(
                0, action_indices, edge_features
            )
        edge_embedding = operation_pairs.new_zeros(
            (pair_count, self.hidden_dim)
        )
        if self.use_production_action_edge_features:
            edge_embedding = self.production_action_edge_encoder(
                dense_features
            )
        edge_pairs = edge_embedding.reshape(
            operation_count, machine_count, self.hidden_dim
        )
        contextual_logits = self.production_scorer(
            torch.cat(
                (
                    operation_pairs,
                    machine_pairs,
                    global_pairs,
                    edge_pairs,
                ),
                dim=-1,
            )
        ).reshape(-1)
        if not self.use_production_candidate_relative_features:
            return contextual_logits
        if self.production_relative_ranker is None:
            raise RuntimeError("production relative ranker is not initialized")
        edge_names = capable.feature_names
        processing = dense_features[
            :, edge_names.index("processing_time_norm")
        ]
        reconfiguration = dense_features[
            :, edge_names.index("reconfiguration_time_norm")
        ]
        relative_features = self._standardize_candidate_features(
            (processing + reconfiguration).unsqueeze(-1),
            ~action_mask[:pair_count],
        )
        relative_logits = functional.linear(
            relative_features,
            -functional.softplus(
                self.production_relative_ranker.weight
            ),
        ).reshape(-1)
        common_context = self._common_candidate_context(
            contextual_logits,
            ~action_mask[:pair_count],
        )
        return relative_logits + torch.sigmoid(
            self.production_context_gate
        ) * common_context

    def _worker_logits(
        self,
        observation: HeterogeneousGraphObservation,
        operation_embeddings: torch.Tensor,
        machine_embeddings: torch.Tensor,
        worker_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        machine_count = machine_embeddings.shape[0]
        worker_count = worker_embeddings.shape[0]
        pair_count = machine_count * worker_count
        if action_mask.shape[0] != pair_count + 1:
            raise ValueError(
                "worker action mask does not match entity counts"
            )
        locked_store = observation.relations[LOCKED_EDGE]
        locked_operation_by_machine = operation_embeddings.new_zeros(
            (machine_count, self.hidden_dim)
        )
        locked_counts = torch.zeros(
            machine_count,
            dtype=torch.long,
            device=device,
        )
        if locked_store.num_edges:
            operation_indices = torch.as_tensor(
                locked_store.edge_index[0],
                dtype=torch.long,
                device=device,
            )
            machine_indices = torch.as_tensor(
                locked_store.edge_index[1],
                dtype=torch.long,
                device=device,
            )
            locked_counts = torch.bincount(
                machine_indices,
                minlength=machine_count,
            )
            duplicate_machines = torch.nonzero(
                locked_counts > 1,
                as_tuple=False,
            ).flatten()
            if duplicate_machines.numel():
                raise ValueError(
                    "worker policy requires at most one locked operation "
                    "per machine"
                )
            locked_operation_by_machine = (
                locked_operation_by_machine.index_copy(
                    0,
                    machine_indices,
                    operation_embeddings.index_select(
                        0, operation_indices
                    ),
                )
            )
        feasible_pairs = (~action_mask[:pair_count]).reshape(
            machine_count,
            worker_count,
        )
        feasible_machines = feasible_pairs.any(dim=1)
        if bool(torch.any(locked_counts[feasible_machines] != 1)):
            raise ValueError(
                "each machine with a feasible worker action must have "
                "exactly one locked operation"
            )
        operation_pairs = locked_operation_by_machine[:, None, :].expand(
            machine_count,
            worker_count,
            -1,
        )
        machine_pairs = machine_embeddings[:, None, :].expand(
            machine_count,
            worker_count,
            -1,
        )
        worker_pairs = worker_embeddings[None, :, :].expand(
            machine_count,
            worker_count,
            -1,
        )
        global_pairs = global_embedding[None, None, :].expand(
            machine_count,
            worker_count,
            -1,
        )
        service = observation.relations[SERVICE_CANDIDATE_EDGE]
        dense_features = operation_pairs.new_zeros(
            (
                pair_count,
                self.edge_feature_dimensions[SERVICE_CANDIDATE_EDGE],
            )
        )
        if service.num_edges:
            edge_index = torch.as_tensor(
                service.edge_index,
                dtype=torch.long,
                device=device,
            )
            action_indices = edge_index[0] * worker_count + edge_index[1]
            edge_features = torch.as_tensor(
                service.edge_features,
                dtype=torch.float32,
                device=device,
            )
            dense_features = dense_features.index_copy(
                0, action_indices, edge_features
            )
        action_edge_embedding = operation_pairs.new_zeros(
            (pair_count, self.hidden_dim)
        )
        if self.use_worker_action_edge_features:
            action_edge_embedding = self.worker_action_edge_encoder(
                dense_features
            )
        action_edge_pairs = action_edge_embedding.reshape(
            machine_count, worker_count, self.hidden_dim
        )
        contextual_logits = self.worker_scorer(
            torch.cat(
                (
                    operation_pairs,
                    machine_pairs,
                    worker_pairs,
                    global_pairs,
                    action_edge_pairs,
                ),
                dim=-1,
            )
        ).reshape(-1)
        if not self.use_worker_candidate_relative_features:
            return contextual_logits
        if self.worker_relative_ranker is None:
            raise RuntimeError("worker relative ranker is not initialized")
        edge_names = service.feature_names
        projected_fatigue = dense_features[
            :, edge_names.index("projected_fatigue_ratio")
        ]
        incremental_variance = dense_features[
            :, edge_names.index("incremental_load_variance_norm")
        ]
        relative_features = self._standardize_candidate_features(
            torch.stack(
                (projected_fatigue, incremental_variance),
                dim=-1,
            ),
            ~action_mask[:pair_count],
        )
        shared_raw_weight = self.worker_relative_ranker.weight.mean()
        shared_effective_weight = -functional.softplus(shared_raw_weight)
        relative_logits = (
            relative_features.sum(dim=-1) * shared_effective_weight
        )
        common_context = self._common_candidate_context(
            contextual_logits,
            ~action_mask[:pair_count],
        )
        return relative_logits + torch.sigmoid(
            self.worker_context_gate
        ) * common_context

    @staticmethod
    def _initialize_monotone_ranker(ranker: nn.Linear) -> None:
        effective_magnitude = torch.tensor(1.0)
        raw_value = torch.log(torch.expm1(effective_magnitude)).item()
        nn.init.constant_(ranker.weight, raw_value)

    @staticmethod
    def _initialize_zero_context_output(scorer: nn.Sequential) -> None:
        output = scorer[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("candidate context scorer must end in nn.Linear")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    @staticmethod
    def _common_candidate_context(
        contextual_logits: torch.Tensor,
        feasible: torch.Tensor,
    ) -> torch.Tensor:
        if bool(torch.any(feasible)):
            common = contextual_logits[feasible].mean()
        else:
            common = contextual_logits.new_zeros(())
        return common.expand_as(contextual_logits)

    def effective_relative_cost_weights(
        self,
    ) -> dict[str, tuple[float, ...]]:
        result: dict[str, tuple[float, ...]] = {}
        if self.production_relative_ranker is not None:
            result["production"] = tuple(
                float(value)
                for value in (
                    -functional.softplus(
                        self.production_relative_ranker.weight.detach()
                    )
                )
                .reshape(-1)
                .cpu()
            )
        if self.worker_relative_ranker is not None:
            shared = float(
                -functional.softplus(
                    self.worker_relative_ranker.weight.detach().mean()
                )
                .reshape(())
                .cpu()
            )
            result["worker"] = (shared, shared)
        return result

    @staticmethod
    def _standardize_candidate_features(
        features: torch.Tensor,
        feasible: torch.Tensor,
    ) -> torch.Tensor:
        """Scale raw candidate features within one legal action set."""
        if features.ndim != 2 or feasible.ndim != 1:
            raise ValueError("candidate features and feasibility have bad rank")
        if features.shape[0] != feasible.shape[0]:
            raise ValueError("candidate features and feasibility do not align")
        selected = features[feasible]
        if selected.shape[0] < 2:
            return torch.zeros_like(features)
        mean = selected.mean(dim=0, keepdim=True)
        scale = selected.std(dim=0, unbiased=False, keepdim=True)
        normalized = (features - mean) / scale.clamp_min(1e-6)
        varying = scale > 1e-6
        return torch.where(varying, normalized, torch.zeros_like(normalized))

    @staticmethod
    def _pool_slices(
        embeddings: torch.Tensor,
        slices: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        pooled = []
        for start, end in slices:
            if end == start:
                pooled.append(
                    embeddings.new_zeros((embeddings.shape[-1],))
                )
            else:
                pooled.append(embeddings[start:end].mean(dim=0))
        return torch.stack(pooled)

    @staticmethod
    def _node_slice(
        embeddings: torch.Tensor,
        bounds: tuple[int, int],
    ) -> torch.Tensor:
        return embeddings[bounds[0] : bounds[1]]

    @staticmethod
    def _validate_action_mask(
        action_mask: np.ndarray | torch.Tensor,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        mask = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=device,
        )
        if mask.ndim != 1:
            raise ValueError("each action mask must be one-dimensional")
        if bool(mask.all()):
            raise ValueError("at least one action must be feasible")
        return mask


ActorCriticNetwork = TypedActorCritic | HeteroGraphActorCritic


def build_actor_critic(
    observation: HeterogeneousGraphObservation,
    network_config: Mapping[str, Any],
) -> ActorCriticNetwork:
    """Build the configured actor-critic from an environment observation."""
    if not isinstance(observation, HeterogeneousGraphObservation):
        raise TypeError(
            "network construction requires a heterogeneous graph observation"
        )
    observation.validate()
    config = normalize_network_config(network_config)
    if config["encoder_type"] == "typed_mlp":
        return TypedActorCritic(
            {
                name: observation.feature_dimensions[name]
                for name in ("operation", "machine", "worker", "global")
            },
            config["hidden_dim"],
        )
    return HeteroGraphActorCritic(
        observation.feature_dimensions,
        observation.edge_feature_dimensions,
        config["hidden_dim"],
        config["message_passing_layers"],
        config["dropout"],
        config["production_action_edge_features"],
        config["worker_action_edge_features"],
        config["production_candidate_relative_features"],
        config["worker_candidate_relative_features"],
    )
