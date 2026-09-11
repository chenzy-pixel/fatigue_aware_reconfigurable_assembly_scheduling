from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from environment import (
    ASSEMBLY_EDGE_TYPES,
    ASSEMBLY_NODE_TYPES,
    CAPABLE_EDGE,
    MACHINE_MODULE_EDGE,
    OPERATION_ORDER_EDGE,
    ORDER_WAVE_EDGE,
    REQUIRES_MODULE_EDGE,
    WAVE_MODULE_EDGE,
    WORKER_MODULE_EDGE,
    LOCKED_EDGE,
    SERVICE_CANDIDATE_EDGE,
    DecisionType,
    EdgeType,
    HeterogeneousGraphObservation,
)


NODE_TYPES = ASSEMBLY_NODE_TYPES
OBJECTIVES = ("flow", "cost", "variance")
POLICY_HEAD_VERSION = 8
OBSERVATION_SCHEMA_VERSION = 5
EXPERT_WEIGHT_PARAMETERIZATION = "simplex_softplus_v8"
PREFERENCE_ENCODER_DIM = 32
RESIDUAL_STD_FLOOR = 1e-3

PRODUCTION_DIRECT_SCHEMA: dict[str, tuple[tuple[str, int], ...]] = {
    "flow": (
        ("processing_time_norm", -1),
        ("reconfiguration_time_norm", -1),
        ("predicted_finish_time_norm", -1),
        ("horizon_slack_norm", 1),
    ),
    "cost": (
        ("fixed_reconfiguration_cost_norm", -1),
        ("estimated_labor_cost_norm", -1),
        ("estimated_downtime_cost_norm", -1),
    ),
    "variance": (("estimated_worker_load_variance_delta_norm", -1),),
}
WORKER_DIRECT_SCHEMA: dict[str, tuple[tuple[str, int], ...]] = {
    "flow": (("stage_duration_norm", -1),),
    "cost": (
        ("fixed_cost_norm", -1),
        ("incremental_labor_cost_norm", -1),
        ("incremental_downtime_cost_norm", -1),
    ),
    "variance": (("incremental_load_variance_norm", -1),),
}
WAIT_DIRECT_SCHEMA: dict[str, tuple[tuple[str, int], ...]] = {
    "flow": (("estimated_flow_objective_delta_if_wait", -1),),
    "cost": (("estimated_cost_objective_delta_if_wait", -1),),
    "variance": (("estimated_load_variance_delta_if_wait", -1),),
}

BIDIRECTIONAL_EDGE_TYPES = frozenset(
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
    if not isinstance(config, Mapping):
        raise TypeError("network config must be a mapping")
    hidden_dim = int(config.get("hidden_dim", 128))
    layers = int(config.get("message_passing_layers", 2))
    dropout = float(config.get("dropout", 0.0))
    version = int(config.get("policy_head_version", POLICY_HEAD_VERSION))
    if version != POLICY_HEAD_VERSION:
        raise ValueError("V8 runtime accepts only policy_head_version=8")
    if hidden_dim <= 0 or layers <= 0:
        raise ValueError("hidden_dim and message_passing_layers must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("network.dropout must be in [0, 1)")
    gate = float(config.get("residual_gate_initial_logit", 0.0))
    if not np.isfinite(gate):
        raise ValueError("network.residual_gate_initial_logit must be finite")
    floor = float(config.get("residual_std_floor", RESIDUAL_STD_FLOOR))
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("network.residual_std_floor must be positive")
    manifest_sha = config.get("normalization_manifest_sha256")
    if manifest_sha is not None:
        manifest_sha = str(manifest_sha).lower()
        if len(manifest_sha) != 64 or any(
            character not in "0123456789abcdef" for character in manifest_sha
        ):
            raise ValueError("normalization_manifest_sha256 must be a 64-digit hex digest")
    return {
        "encoder_type": "hetero_gnn",
        "hidden_dim": hidden_dim,
        "message_passing_layers": layers,
        "dropout": dropout,
        "policy_head_version": version,
        "preference_embedding_dim": PREFERENCE_ENCODER_DIM,
        "residual_gate_initial_logit": gate,
        "residual_std_floor": floor,
        "expert_weight_parameterization": EXPERT_WEIGHT_PARAMETERIZATION,
        "normalization_manifest_sha256": manifest_sha,
    }


def network_requires_graph_observation(config: Mapping[str, Any]) -> bool:
    normalize_network_config(config)
    return True


def _schema_serializable(
    schema: Mapping[str, Sequence[tuple[str, int]]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        objective: [
            {"name": name, "direction": int(direction)}
            for name, direction in fields
        ]
        for objective, fields in schema.items()
    }


def infer_checkpoint_network_spec(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    raw = checkpoint.get("network_spec")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint does not contain a V8 network_spec")
    spec = dict(raw)
    if int(spec.get("policy_head_version", 0)) != POLICY_HEAD_VERSION:
        raise ValueError("V7 and earlier checkpoints are rejected by V8 runtime")
    if int(spec.get("observation_schema_version", 0)) != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("checkpoint observation schema is not V8 schema 5")
    if spec.get("expert_weight_parameterization") != EXPERT_WEIGHT_PARAMETERIZATION:
        raise ValueError("checkpoint does not use simplex_softplus_v8 experts")
    if int(spec.get("preference_embedding_dim", 0)) != PREFERENCE_ENCODER_DIM:
        raise ValueError("checkpoint preference encoder is not V8 3->32->ReLU->32")
    expected_schemas = {
        "production_direct_feature_schema": _schema_serializable(PRODUCTION_DIRECT_SCHEMA),
        "worker_direct_feature_schema": _schema_serializable(WORKER_DIRECT_SCHEMA),
        "wait_direct_feature_schema": _schema_serializable(WAIT_DIRECT_SCHEMA),
    }
    for name, expected in expected_schemas.items():
        if spec.get(name) != expected:
            raise ValueError(f"checkpoint {name} is incompatible with V8")
    normalize_network_config(spec)
    return spec


def assert_network_config_matches_spec(
    config: Mapping[str, Any], checkpoint_spec: Mapping[str, Any]
) -> None:
    configured = normalize_network_config(config)
    saved = infer_checkpoint_network_spec({"network_spec": checkpoint_spec})
    for name in (
        "hidden_dim",
        "message_passing_layers",
        "dropout",
        "policy_head_version",
        "preference_embedding_dim",
        "residual_gate_initial_logit",
        "residual_std_floor",
        "expert_weight_parameterization",
        "normalization_manifest_sha256",
    ):
        if configured[name] != saved.get(name):
            raise ValueError(
                f"checkpoint {name} is incompatible: "
                f"configured={configured[name]!r}, checkpoint={saved.get(name)!r}"
            )
    expected_schemas = {
        "production_direct_feature_schema": _schema_serializable(
            PRODUCTION_DIRECT_SCHEMA
        ),
        "worker_direct_feature_schema": _schema_serializable(WORKER_DIRECT_SCHEMA),
        "wait_direct_feature_schema": _schema_serializable(WAIT_DIRECT_SCHEMA),
    }
    for name, expected in expected_schemas.items():
        if saved.get(name) != expected:
            raise ValueError(f"checkpoint {name} is incompatible with V8")
    for name in (
        "feature_dimensions",
        "edge_feature_dimensions",
        "action_set_feature_names",
    ):
        if checkpoint_spec.get(name) != config.get(name):
            raise ValueError(f"checkpoint {name} is incompatible with this observation")


class SimplexMonotoneRanker(nn.Module):
    """Bias-free fixed-sign ranker whose positive weights lie on a simplex."""

    def __init__(self, directions: Sequence[int]):
        super().__init__()
        values = tuple(int(value) for value in directions)
        if not values or any(value not in {-1, 1} for value in values):
            raise ValueError("direct ranker directions must be non-empty and +/-1")
        self.theta = nn.Parameter(torch.zeros(len(values)))
        self.register_buffer("directions", torch.tensor(values, dtype=torch.float32))

    def normalized_weights(self) -> torch.Tensor:
        positive = F.softplus(self.theta)
        return positive / positive.sum()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.theta.numel():
            raise ValueError("direct feature matrix has an invalid shape")
        weights = self.normalized_weights() * self.directions
        return torch.tanh((features * weights).sum(dim=-1))


class ObjectiveExpert(nn.Module):
    def __init__(self, directions: Sequence[int], hidden_dim: int):
        super().__init__()
        self.direct_ranker = SimplexMonotoneRanker(directions)
        self.context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(
        self, action_embedding: torch.Tensor, direct_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direct = self.direct_ranker(direct_features)
        context = torch.tanh(self.context(action_embedding).squeeze(-1))
        return direct, context, direct + context


class ObjectiveExpertSet(nn.Module):
    def __init__(
        self,
        schema: Mapping[str, Sequence[tuple[str, int]]],
        hidden_dim: int,
    ):
        super().__init__()
        self.schema = {name: tuple(fields) for name, fields in schema.items()}
        self.experts = nn.ModuleDict(
            {
                objective: ObjectiveExpert(
                    [direction for _, direction in self.schema[objective]], hidden_dim
                )
                for objective in OBJECTIVES
            }
        )

    def forward(
        self,
        action_embedding: torch.Tensor,
        direct: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = [
            self.experts[name](action_embedding, direct[name])
            for name in OBJECTIVES
        ]
        return tuple(torch.stack(parts, dim=-1) for parts in zip(*values, strict=True))


RelationBatch = tuple[torch.Tensor, torch.Tensor, bool]


@dataclass(frozen=True)
class _GraphBatch:
    node_features: dict[str, torch.Tensor]
    node_slices: dict[str, list[tuple[int, int]]]
    relations: dict[EdgeType, RelationBatch]
    global_features: torch.Tensor


class HeterogeneousMessagePassingLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_feature_dimensions: Mapping[EdgeType, int],
        dropout: float,
    ):
        super().__init__()
        self.transforms = nn.ModuleDict(
            {
                _relation_key(edge_type): nn.Linear(
                    hidden_dim + int(edge_feature_dimensions[edge_type]), hidden_dim
                )
                for edge_type in ASSEMBLY_EDGE_TYPES
            }
        )
        self.norms = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in NODE_TYPES}
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        relations: Mapping[EdgeType, RelationBatch],
    ) -> dict[str, torch.Tensor]:
        total = {name: torch.zeros_like(value) for name, value in embeddings.items()}
        degree = {
            name: value.new_zeros((value.shape[0], 1))
            for name, value in embeddings.items()
        }
        for edge_type in ASSEMBLY_EDGE_TYPES:
            source_type, _, target_type = edge_type
            edge_index, edge_features, bidirectional = relations[edge_type]
            if edge_index.shape[1] == 0:
                continue
            source, target = edge_index
            transform = self.transforms[_relation_key(edge_type)]
            forward = transform(
                torch.cat((embeddings[source_type][source], edge_features), dim=-1)
            )
            total[target_type].index_add_(0, target, forward)
            degree[target_type].index_add_(
                0, target, forward.new_ones((forward.shape[0], 1))
            )
            if bidirectional:
                reverse = transform(
                    torch.cat((embeddings[target_type][target], edge_features), dim=-1)
                )
                total[source_type].index_add_(0, source, reverse)
                degree[source_type].index_add_(
                    0, source, reverse.new_ones((reverse.shape[0], 1))
                )
        return {
            name: self.dropout(
                F.relu(self.norms[name](value + total[name] / degree[name].clamp_min(1)))
            )
            for name, value in embeddings.items()
        }


def _positive_unit(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(0.0)
    return value / (1.0 + value)


def _signed_unit(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (value / (1.0 + value.abs()) + 1.0)


class HeteroGraphActorCritic(nn.Module):
    requires_graph_observation = True

    def __init__(
        self,
        feature_dimensions: Mapping[str, int],
        edge_feature_dimensions: Mapping[EdgeType, int],
        action_set_feature_names: Sequence[str],
        *,
        hidden_dim: int,
        message_passing_layers: int,
        dropout: float,
        residual_gate_initial_logit: float = 0.0,
        residual_std_floor: float = RESIDUAL_STD_FLOOR,
        normalization_manifest_sha256: str | None = None,
    ):
        super().__init__()
        self.feature_dimensions = {name: int(value) for name, value in feature_dimensions.items()}
        self.edge_feature_dimensions = {
            name: int(value) for name, value in edge_feature_dimensions.items()
        }
        self.action_set_feature_names = tuple(action_set_feature_names)
        self.hidden_dim = int(hidden_dim)
        self.message_passing_layer_count = int(message_passing_layers)
        self.dropout_probability = float(dropout)
        self.policy_head_version = POLICY_HEAD_VERSION
        self.expert_weight_parameterization = EXPERT_WEIGHT_PARAMETERIZATION
        self.residual_gate_initial_logit = float(residual_gate_initial_logit)
        self.residual_std_floor = float(residual_std_floor)
        self.normalization_manifest_sha256 = normalization_manifest_sha256
        if set(self.feature_dimensions) != set((*NODE_TYPES, "global")):
            raise ValueError("feature dimensions must contain six nodes and global")
        if set(self.edge_feature_dimensions) != set(ASSEMBLY_EDGE_TYPES):
            raise ValueError("edge feature dimensions do not match assembly graph")
        missing_wait = {
            field
            for fields in WAIT_DIRECT_SCHEMA.values()
            for field, _ in fields
            if field not in self.action_set_feature_names
        }
        if missing_wait:
            raise ValueError(f"WAIT feature schema is incomplete: {sorted(missing_wait)}")

        self.node_projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.feature_dimensions[name], hidden_dim), nn.ReLU()
                )
                for name in NODE_TYPES
            }
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(self.feature_dimensions["global"], hidden_dim), nn.ReLU()
        )
        self.message_layers = nn.ModuleList(
            [
                HeterogeneousMessagePassingLayer(
                    hidden_dim, self.edge_feature_dimensions, dropout
                )
                for _ in range(message_passing_layers)
            ]
        )
        graph_width = hidden_dim * (len(NODE_TYPES) + 1)
        self.graph_context_projector = nn.Sequential(
            nn.Linear(graph_width, hidden_dim), nn.ReLU()
        )
        self.preference_encoder = nn.Sequential(
            nn.Linear(3, PREFERENCE_ENCODER_DIM),
            nn.ReLU(),
            nn.Linear(PREFERENCE_ENCODER_DIM, PREFERENCE_ENCODER_DIM),
        )
        self.production_edge_encoder = nn.Sequential(
            nn.Linear(self.edge_feature_dimensions[CAPABLE_EDGE], hidden_dim), nn.ReLU()
        )
        self.worker_edge_encoder = nn.Sequential(
            nn.Linear(self.edge_feature_dimensions[SERVICE_CANDIDATE_EDGE], hidden_dim),
            nn.ReLU(),
        )
        self.production_action_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU()
        )
        self.worker_action_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim), nn.ReLU()
        )
        self.wait_feature_encoder = nn.Sequential(
            nn.Linear(len(self.action_set_feature_names), hidden_dim), nn.ReLU()
        )
        self.wait_action_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU()
        )
        self.production_experts = ObjectiveExpertSet(PRODUCTION_DIRECT_SCHEMA, hidden_dim)
        self.worker_experts = ObjectiveExpertSet(WORKER_DIRECT_SCHEMA, hidden_dim)
        self.production_wait_experts = ObjectiveExpertSet(WAIT_DIRECT_SCHEMA, hidden_dim)
        self.worker_wait_experts = ObjectiveExpertSet(WAIT_DIRECT_SCHEMA, hidden_dim)

        residual_width = hidden_dim * 2 + PREFERENCE_ENCODER_DIM * 2
        self.action_preference_projector = nn.Linear(hidden_dim, PREFERENCE_ENCODER_DIM)
        self.production_residual = self._residual_mlp(residual_width, hidden_dim)
        self.worker_residual = self._residual_mlp(residual_width, hidden_dim)
        self.production_residual_gate = nn.Parameter(
            torch.tensor(float(residual_gate_initial_logit))
        )
        self.worker_residual_gate = nn.Parameter(
            torch.tensor(float(residual_gate_initial_logit))
        )
        self.critic = nn.Sequential(
            nn.Linear(graph_width + PREFERENCE_ENCODER_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._latest_policy_decision_diagnostics: list[dict[str, Any]] = []

    @staticmethod
    def _residual_mlp(width: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(width, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def network_spec(self) -> dict[str, Any]:
        return {
            "encoder_type": "hetero_gnn",
            "hidden_dim": self.hidden_dim,
            "message_passing_layers": self.message_passing_layer_count,
            "dropout": self.dropout_probability,
            "policy_head_version": POLICY_HEAD_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "preference_embedding_dim": PREFERENCE_ENCODER_DIM,
            "expert_weight_parameterization": EXPERT_WEIGHT_PARAMETERIZATION,
            "direct_output_range": [-1.0, 1.0],
            "context_output_range": [-1.0, 1.0],
            "expert_output_range": [-2.0, 2.0],
            "production_direct_feature_schema": _schema_serializable(PRODUCTION_DIRECT_SCHEMA),
            "worker_direct_feature_schema": _schema_serializable(WORKER_DIRECT_SCHEMA),
            "wait_direct_feature_schema": _schema_serializable(WAIT_DIRECT_SCHEMA),
            "residual_gate_initial_logit": self.residual_gate_initial_logit,
            "residual_std_floor": self.residual_std_floor,
            "feature_dimensions": dict(self.feature_dimensions),
            "edge_feature_dimensions": dict(self.edge_feature_dimensions),
            "action_set_feature_names": self.action_set_feature_names,
            "normalization_manifest_sha256": self.normalization_manifest_sha256,
        }

    def encode_graph(
        self,
        observations: Sequence[HeterogeneousGraphObservation],
        *,
        device: torch.device | str,
    ) -> tuple[_GraphBatch, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        batch = self._collate_graphs(observations, device=device)
        embeddings = {
            name: self.node_projectors[name](batch.node_features[name])
            for name in NODE_TYPES
        }
        for layer in self.message_layers:
            embeddings = layer(embeddings, batch.relations)
        global_embeddings = self.global_encoder(batch.global_features)
        pooled = {
            name: self._pool_slices(embeddings[name], batch.node_slices[name])
            for name in NODE_TYPES
        }
        graph_context = torch.cat(
            tuple(pooled[name] for name in NODE_TYPES) + (global_embeddings,), dim=-1
        )
        return batch, embeddings, global_embeddings, graph_context

    def forward(
        self,
        observation: HeterogeneousGraphObservation,
        action_mask: np.ndarray | torch.Tensor,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = self.forward_batch([observation], [action_mask], device=device)
        return logits[0, : len(action_mask)], values[0]

    def forward_batch(
        self,
        observations: Sequence[HeterogeneousGraphObservation],
        action_masks: Sequence[np.ndarray | torch.Tensor],
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not observations or len(observations) != len(action_masks):
            raise ValueError("observation/action-mask batches must be non-empty and aligned")
        if any(not isinstance(item, HeterogeneousGraphObservation) for item in observations):
            raise TypeError("V8 requires heterogeneous graph observations")
        self._latest_policy_decision_diagnostics.clear()
        batch, embeddings, global_embeddings, graph_context = self.encode_graph(
            observations, device=device
        )
        preference = torch.stack(
            [torch.as_tensor(item.preference, dtype=torch.float32, device=device) for item in observations]
        )
        preference_embedding = self.preference_encoder(preference)
        values = self.critic(torch.cat((graph_context, preference_embedding), dim=-1)).squeeze(-1)
        masks = [self._validate_action_mask(mask, device=device) for mask in action_masks]
        result = values.new_full(
            (len(observations), max(mask.numel() for mask in masks)),
            torch.finfo(values.dtype).min,
        )
        graph_hidden = self.graph_context_projector(graph_context)
        for index, observation in enumerate(observations):
            nodes = {
                name: self._node_slice(embeddings[name], batch.node_slices[name][index])
                for name in NODE_TYPES
            }
            logits = self._phase_logits(
                observation,
                nodes,
                global_embeddings[index],
                graph_hidden[index],
                preference[index],
                preference_embedding[index],
                masks[index],
                device=device,
            )
            result[index, : masks[index].numel()] = logits.masked_fill(
                masks[index], torch.finfo(values.dtype).min
            )
        return result, values

    def _phase_logits(
        self,
        observation: HeterogeneousGraphObservation,
        nodes: Mapping[str, torch.Tensor],
        global_embedding: torch.Tensor,
        graph_hidden: torch.Tensor,
        preference: torch.Tensor,
        preference_embedding: torch.Tensor,
        mask: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        if observation.decision_type == DecisionType.PRODUCTION:
            action_embedding, direct = self._production_candidates(
                observation, nodes, global_embedding, mask, device=device
            )
            experts = self.production_experts
            wait_experts = self.production_wait_experts
            residual_mlp = self.production_residual
            gate = self.production_residual_gate
        elif observation.decision_type == DecisionType.WORKER:
            action_embedding, direct = self._worker_candidates(
                observation, nodes, global_embedding, mask, device=device
            )
            experts = self.worker_experts
            wait_experts = self.worker_wait_experts
            residual_mlp = self.worker_residual
            gate = self.worker_residual_gate
        else:
            raise ValueError("actor cannot evaluate a terminal observation")
        wait_raw = torch.as_tensor(
            observation.action_set_features, dtype=torch.float32, device=device
        ).reshape(1, -1)
        wait_embedding = self.wait_action_encoder(
            torch.cat((self.wait_feature_encoder(wait_raw), graph_hidden.reshape(1, -1)), dim=-1)
        )
        wait_direct = self._wait_direct(observation, wait_raw)
        direct_values, context_values, z = experts(action_embedding, direct)
        wait_d, wait_c, wait_z = wait_experts(wait_embedding, wait_direct)
        all_embeddings = torch.cat((action_embedding, wait_embedding), dim=0)
        all_d = torch.cat((direct_values, wait_d), dim=0)
        all_c = torch.cat((context_values, wait_c), dim=0)
        all_z = torch.cat((z, wait_z), dim=0)
        base = (all_z * preference.reshape(1, 3)).sum(dim=-1)
        interaction = self.action_preference_projector(all_embeddings) * preference_embedding
        residual_input = torch.cat(
            (
                all_embeddings,
                graph_hidden.reshape(1, -1).expand(all_embeddings.shape[0], -1),
                preference_embedding.reshape(1, -1).expand(all_embeddings.shape[0], -1),
                interaction,
            ),
            dim=-1,
        )
        raw_residual = torch.tanh(residual_mlp(residual_input).squeeze(-1))
        legal_base = base[~mask]
        scale = (
            legal_base.std(unbiased=False) if legal_base.numel() >= 2 else base.new_zeros(())
        ).clamp_min(self.residual_std_floor)
        residual = 2.0 * torch.sigmoid(gate) * scale * raw_residual
        final = base + residual
        self._record_components(
            observation.decision_type, mask, all_d, all_c, all_z, base, residual, final, preference
        )
        return final

    def _production_candidates(
        self,
        observation: HeterogeneousGraphObservation,
        nodes: Mapping[str, torch.Tensor],
        global_embedding: torch.Tensor,
        mask: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        operations, machines = nodes["operation"], nodes["machine"]
        operation_count, machine_count = operations.shape[0], machines.shape[0]
        pair_count = operation_count * machine_count
        if mask.numel() != pair_count + 1:
            raise ValueError("production mask width does not match candidates")
        dense = operations.new_zeros((pair_count, self.edge_feature_dimensions[CAPABLE_EDGE]))
        store = observation.relations[CAPABLE_EDGE]
        if store.num_edges:
            indices = torch.as_tensor(store.edge_index, dtype=torch.long, device=device)
            actions = indices[0] * machine_count + indices[1]
            dense = dense.index_copy(
                0, actions, torch.as_tensor(store.edge_features, dtype=torch.float32, device=device)
            )
        op = operations[:, None, :].expand(operation_count, machine_count, -1).reshape(pair_count, -1)
        machine = machines[None, :, :].expand(operation_count, machine_count, -1).reshape(pair_count, -1)
        global_part = global_embedding.reshape(1, -1).expand(pair_count, -1)
        action = self.production_action_encoder(
            torch.cat((op, machine, global_part, self.production_edge_encoder(dense)), dim=-1)
        )
        names = store.feature_names
        columns = {name: dense[:, names.index(name)] for name in names}
        columns["fixed_reconfiguration_cost_norm"] = (
            columns["fixed_disassembly_cost_norm"] + columns["fixed_installation_cost_norm"]
        )
        direct = self._direct_from_columns(PRODUCTION_DIRECT_SCHEMA, columns)
        return action, direct

    def _worker_candidates(
        self,
        observation: HeterogeneousGraphObservation,
        nodes: Mapping[str, torch.Tensor],
        global_embedding: torch.Tensor,
        mask: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        machines, workers = nodes["machine"], nodes["worker"]
        machine_count, worker_count = machines.shape[0], workers.shape[0]
        pair_count = machine_count * worker_count
        if mask.numel() != pair_count + 1:
            raise ValueError("worker mask width does not match candidates")
        locked = nodes["operation"].new_zeros((machine_count, self.hidden_dim))
        locked_store = observation.relations[LOCKED_EDGE]
        if locked_store.num_edges:
            indices = torch.as_tensor(locked_store.edge_index, dtype=torch.long, device=device)
            counts = torch.bincount(indices[1], minlength=machine_count)
            if bool(torch.any(counts > 1)):
                raise ValueError("a machine cannot lock multiple operations")
            locked = locked.index_copy(0, indices[1], nodes["operation"].index_select(0, indices[0]))
        service = observation.relations[SERVICE_CANDIDATE_EDGE]
        dense = machines.new_zeros((pair_count, self.edge_feature_dimensions[SERVICE_CANDIDATE_EDGE]))
        if service.num_edges:
            indices = torch.as_tensor(service.edge_index, dtype=torch.long, device=device)
            actions = indices[0] * worker_count + indices[1]
            dense = dense.index_copy(
                0, actions, torch.as_tensor(service.edge_features, dtype=torch.float32, device=device)
            )
        operation = locked[:, None, :].expand(machine_count, worker_count, -1).reshape(pair_count, -1)
        machine = machines[:, None, :].expand(machine_count, worker_count, -1).reshape(pair_count, -1)
        worker = workers[None, :, :].expand(machine_count, worker_count, -1).reshape(pair_count, -1)
        global_part = global_embedding.reshape(1, -1).expand(pair_count, -1)
        action = self.worker_action_encoder(
            torch.cat((operation, machine, worker, global_part, self.worker_edge_encoder(dense)), dim=-1)
        )
        names = service.feature_names
        columns = {name: dense[:, names.index(name)] for name in names}
        direct = self._direct_from_columns(WORKER_DIRECT_SCHEMA, columns)
        return action, direct

    def _wait_direct(
        self, observation: HeterogeneousGraphObservation, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        names = observation.action_set_feature_names
        columns = {name: features[:, names.index(name)] for name in names}
        return self._direct_from_columns(WAIT_DIRECT_SCHEMA, columns)

    @staticmethod
    def _direct_from_columns(
        schema: Mapping[str, Sequence[tuple[str, int]]],
        columns: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        signed = {
            "horizon_slack_norm",
            "estimated_worker_load_variance_delta_norm",
            "incremental_load_variance_norm",
            "estimated_load_variance_delta_if_wait",
        }
        for objective, fields in schema.items():
            transformed = [
                _signed_unit(columns[name]) if name in signed else _positive_unit(columns[name])
                for name, _ in fields
            ]
            result[objective] = torch.stack(transformed, dim=-1)
        return result

    def _collate_graphs(
        self,
        observations: Sequence[HeterogeneousGraphObservation],
        *,
        device: torch.device | str,
    ) -> _GraphBatch:
        node_features: dict[str, torch.Tensor] = {}
        node_slices: dict[str, list[tuple[int, int]]] = {}
        for node_type in NODE_TYPES:
            parts: list[torch.Tensor] = []
            slices: list[tuple[int, int]] = []
            offset = 0
            for observation in observations:
                array = observation.node_features[node_type]
                if array.shape[1] != self.feature_dimensions[node_type]:
                    raise ValueError(f"{node_type} feature width changed")
                count = array.shape[0]
                slices.append((offset, offset + count))
                offset += count
                parts.append(torch.as_tensor(array, dtype=torch.float32, device=device))
            node_features[node_type] = torch.cat(parts)
            node_slices[node_type] = slices
        relations: dict[EdgeType, RelationBatch] = {}
        for edge_type in ASSEMBLY_EDGE_TYPES:
            source_type, _, target_type = edge_type
            indices_parts: list[torch.Tensor] = []
            feature_parts: list[torch.Tensor] = []
            expected_bidirectional = edge_type in BIDIRECTIONAL_EDGE_TYPES
            for index, observation in enumerate(observations):
                store = observation.relations[edge_type]
                if store.bidirectional != expected_bidirectional:
                    raise ValueError(f"bidirectional flag changed for {edge_type}")
                indices = torch.as_tensor(store.edge_index, dtype=torch.long, device=device).clone()
                indices[0] += node_slices[source_type][index][0]
                indices[1] += node_slices[target_type][index][0]
                indices_parts.append(indices)
                feature_parts.append(torch.as_tensor(store.edge_features, dtype=torch.float32, device=device))
            relations[edge_type] = (
                torch.cat(indices_parts, dim=1), torch.cat(feature_parts), expected_bidirectional
            )
        global_features = torch.stack(
            [torch.as_tensor(item.global_features, dtype=torch.float32, device=device) for item in observations]
        )
        return _GraphBatch(node_features, node_slices, relations, global_features)

    def _record_components(
        self,
        phase: DecisionType,
        mask: torch.Tensor,
        direct: torch.Tensor,
        context: torch.Tensor,
        experts: torch.Tensor,
        base: torch.Tensor,
        residual: torch.Tensor,
        final: torch.Tensor,
        preference: torch.Tensor,
    ) -> None:
        legal = ~mask
        row: dict[str, Any] = {
            "decision_type": phase.value,
            "legal_pair_count": int((~mask[:-1]).sum().item()),
            "terminal_legal": bool(legal[-1].item()),
        }
        for name, values in (("direct", direct), ("context", context), ("expert", experts)):
            for index, objective in enumerate(OBJECTIVES):
                selected = values[legal, index]
                row[f"{name}_{objective}_mean"] = float(selected.mean().detach().cpu())
                row[f"{name}_{objective}_std"] = float(selected.std(unbiased=False).detach().cpu())
                row[f"{name}_{objective}_rms"] = float(selected.square().mean().sqrt().detach().cpu())
                row[f"{name}_{objective}_saturation_ratio"] = float((selected.abs() > 0.99).float().mean().detach().cpu())
                contribution = selected * preference[index]
                row[f"contribution_{objective}_rms"] = float(contribution.square().mean().sqrt().detach().cpu())
        base_rms = base[legal].square().mean().sqrt()
        residual_rms = residual[legal].square().mean().sqrt()
        row["residual_base_rms_ratio"] = float((residual_rms / base_rms.clamp_min(1e-12)).detach().cpu())
        pair_legal = ~mask[:-1]
        if bool(pair_legal.any()):
            row["relative_top_action"] = int(torch.nonzero(pair_legal).flatten()[base[:-1][pair_legal].argmax()].item())
            row["final_pair_top_action"] = int(torch.nonzero(pair_legal).flatten()[final[:-1][pair_legal].argmax()].item())
            row["context_overrode_top"] = row["relative_top_action"] != row["final_pair_top_action"]
        self._latest_policy_decision_diagnostics.append(row)

    def consume_policy_decision_diagnostics(self) -> list[dict[str, Any]]:
        result = list(self._latest_policy_decision_diagnostics)
        self._latest_policy_decision_diagnostics.clear()
        return result

    def effective_relative_cost_weights(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for phase, expert_set in (
            ("production", self.production_experts),
            ("worker", self.worker_experts),
            ("production_wait", self.production_wait_experts),
            ("worker_wait", self.worker_wait_experts),
        ):
            for objective in OBJECTIVES:
                expert = expert_set.experts[objective]
                weights = expert.direct_ranker.normalized_weights().detach().cpu()
                names = [name for name, _ in expert_set.schema[objective]]
                result[f"{phase}_{objective}"] = {
                    name: float(weight) for name, weight in zip(names, weights, strict=True)
                }
        return result

    def policy_head_diagnostics(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for expert, weights in self.effective_relative_cost_weights().items():
            values = np.asarray(tuple(weights.values()), dtype=np.float64)
            for name, value in weights.items():
                result[f"policy_head_weight_{expert}_{name}"] = value
            result[f"policy_head_weight_sum_{expert}"] = float(values.sum())
            entropy = float(-(values * np.log(np.maximum(values, 1e-30))).sum())
            result[f"policy_head_weight_entropy_{expert}"] = entropy
            result[f"policy_head_effective_feature_count_{expert}"] = float(np.exp(entropy))
        result["policy_head_gate_production_residual"] = float(torch.sigmoid(self.production_residual_gate.detach()).cpu())
        result["policy_head_gate_worker_residual"] = float(torch.sigmoid(self.worker_residual_gate.detach()).cpu())
        return result

    @staticmethod
    def _pool_slices(embeddings: torch.Tensor, slices: Sequence[tuple[int, int]]) -> torch.Tensor:
        return torch.stack(
            [embeddings[start:end].mean(0) if end > start else embeddings.new_zeros(embeddings.shape[-1]) for start, end in slices]
        )

    @staticmethod
    def _node_slice(embeddings: torch.Tensor, bounds: tuple[int, int]) -> torch.Tensor:
        return embeddings[bounds[0] : bounds[1]]

    @staticmethod
    def _validate_action_mask(
        mask: np.ndarray | torch.Tensor, *, device: torch.device | str
    ) -> torch.Tensor:
        value = torch.as_tensor(mask, dtype=torch.bool, device=device)
        if value.ndim != 1 or bool(value.all()):
            raise ValueError("action mask must be one-dimensional with one legal action")
        return value


ActorCriticNetwork = HeteroGraphActorCritic


def build_actor_critic(
    observation: HeterogeneousGraphObservation,
    network_config: Mapping[str, Any],
) -> ActorCriticNetwork:
    if not isinstance(observation, HeterogeneousGraphObservation):
        raise TypeError("V8 network construction requires a graph observation")
    observation.validate()
    config = normalize_network_config(network_config)
    return HeteroGraphActorCritic(
        observation.feature_dimensions,
        observation.edge_feature_dimensions,
        observation.action_set_feature_names,
        hidden_dim=config["hidden_dim"],
        message_passing_layers=config["message_passing_layers"],
        dropout=config["dropout"],
        residual_gate_initial_logit=config["residual_gate_initial_logit"],
        residual_std_floor=config["residual_std_floor"],
        normalization_manifest_sha256=config["normalization_manifest_sha256"],
    )
