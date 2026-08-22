from __future__ import annotations

import math
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
    PREFERENCE_NAMES,
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
        spec = {
            "encoder_type": "typed_mlp",
            "hidden_dim": self.hidden_dim,
            "policy_head_version": 5,
            "production_action_semantics": "pair_plus_defer_v1",
            "observation_schema_version": 3,
            "feature_dimensions": dict(self.feature_dimensions),
        }
        return spec

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

POLICY_HEAD_VERSION = 6
SUPPORTED_POLICY_HEAD_VERSIONS = (6, 7)
V6_PRODUCTION_RELATIVE_FEATURE_NAMES: tuple[str, ...] = (
    "processing_time_norm",
    "reconfiguration_time_norm",
    "fixed_reconfiguration_cost_norm",
    "estimated_labor_cost_norm",
    "estimated_downtime_cost_norm",
    "horizon_slack_norm",
)
V6_WORKER_RELATIVE_FEATURE_NAMES: tuple[str, ...] = (
    "stage_duration_norm",
    "projected_fatigue_ratio",
    "incremental_labor_cost_norm",
    "incremental_downtime_cost_norm",
    "incremental_load_variance_norm",
)
DEFAULT_PRODUCTION_RELATIVE_INITIAL_WEIGHTS: tuple[float, ...] = (
    0.30,
    0.30,
    0.20,
    0.20,
    0.20,
    0.30,
)
DEFAULT_WORKER_RELATIVE_INITIAL_WEIGHTS: tuple[float, ...] = (
    0.30,
    0.30,
    0.20,
    0.20,
    0.20,
)
PRODUCTION_RELATIVE_DIRECTIONS: tuple[float, ...] = (
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    1.0,
)
V7_PRODUCTION_FUTURE_FEATURE_NAMES: tuple[str, ...] = (
    "processing_time_norm",
    "reconfiguration_time_norm",
    "total_reconfiguration_cost_norm",
    "horizon_slack_norm",
    "future_configuration_reuse_value_norm",
    "configuration_opportunity_cost_norm",
)
V7_WORKER_FUTURE_FEATURE_NAMES: tuple[str, ...] = (
    "stage_duration_norm",
    "fatigue_headroom_ratio",
    "total_incremental_cost_norm",
    "incremental_load_variance_norm",
    "qualification_opportunity_cost_norm",
    "recovery_eta_norm",
    "remaining_service_capacity_norm",
)
PRODUCTION_RELATIVE_FEATURE_NAMES = V6_PRODUCTION_RELATIVE_FEATURE_NAMES
WORKER_RELATIVE_FEATURE_NAMES = V6_WORKER_RELATIVE_FEATURE_NAMES
WORKER_RELATIVE_DIRECTIONS: tuple[float, ...] = (-1.0,) * len(
    V6_WORKER_RELATIVE_FEATURE_NAMES
)
DEFAULT_CONTEXT_GATE_INITIAL_LOGIT = -4.0
V6_CANDIDATE_CONTEXT_MODE = "common_plus_gated_residual_v6"
V6_RELATIVE_WEIGHT_PARAMETERIZATION = "independent_softplus_signed_v6"
V6_WORKER_RELATIVE_WEIGHT_SHARING = "independent_softplus_v6"
V7_BOUNDED_CONTEXT_MODE = "bounded_ranker_scale_v7"
DIRECT_PREFERENCE_ACTION_SCORE_VERSION = "direct_main_rank_v1"
DIRECT_PREFERENCE_ACTION_SCORE_STANDARDIZATION = "legal_candidate_zscore"
SAFE_WORKER_VARIANCE_PREFERENCE_ACTION_SCORE_VERSION = (
    "direct_safe_worker_variance_rank_v2"
)
SAFE_WORKER_VARIANCE_SEPARATE_SCALE_VERSION = (
    "direct_safe_worker_variance_rank_v3"
)
DIRECT_PREFERENCE_ACTION_SCORE_SCOPES = (
    "all",
    "production_only",
    "production_plus_safe_worker_variance",
)
FLAT_PRODUCTION_ACTION_SEMANTICS = "pair_plus_defer_v1"
HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS = (
    "hierarchical_commit_then_pair_v2"
)
STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS = (
    "hierarchical_state_only_gate_then_pair_v3"
)
STATE_ONLY_PRODUCTION_GATE_VERSION = "state_only_action_set_gate_v1"
STATE_ONLY_MONOTONE_FLOW_COMMIT_GATE_VERSION = (
    "state_only_monotone_flow_commit_gate_v2"
)
STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION = (
    "state_only_counterfactual_monotone_flow_commit_gate_v3"
)
STATE_ONLY_PRODUCTION_GATE_TIE_BREAK = "commit"
PRODUCTION_ACTION_SET_FEATURE_NAMES: tuple[str, ...] = (
    "legal_candidate_count_norm",
    "configuration_match_rate",
    "minimum_reconfiguration_time_norm",
    "mean_reconfiguration_time_norm",
    "minimum_total_reconfiguration_cost_norm",
    "mean_total_reconfiguration_cost_norm",
    "minimum_horizon_slack_norm",
    "next_defer_event_distance_norm",
    "projected_legal_candidate_gain_norm",
)


def _relation_key(edge_type: EdgeType) -> str:
    return "__".join(edge_type)


def _positive_weight_tuple(
    config: Mapping[str, Any],
    name: str,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    values = tuple(float(value) for value in config.get(name, default))
    if len(values) != len(default):
        raise ValueError(
            f"network.{name} must contain exactly {len(default)} values"
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(
            f"network.{name} values must be finite and positive"
        )
    return values


def _relative_directions(names: Sequence[str]) -> tuple[float, ...]:
    positive = {
        "horizon_slack_norm",
        "future_configuration_reuse_value_norm",
        "fatigue_headroom_ratio",
        "remaining_service_capacity_norm",
    }
    return tuple(1.0 if name in positive else -1.0 for name in names)


def _validate_policy_head_config(config: Mapping[str, Any]) -> dict[str, Any]:
    version = int(config.get("policy_head_version", POLICY_HEAD_VERSION))
    if version not in SUPPORTED_POLICY_HEAD_VERSIONS:
        raise ValueError(
            "policy_head_version must be 6 or 7; older checkpoints require "
            "retraining or their historical source tree and v6/v7 weights "
            "are never converted"
        )
    semantics = str(
        config.get(
            "production_action_semantics",
            FLAT_PRODUCTION_ACTION_SEMANTICS,
        )
    )
    if semantics not in {
        FLAT_PRODUCTION_ACTION_SEMANTICS,
        HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS,
        STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS,
    }:
        raise ValueError(
            "network.production_action_semantics must be "
            f"{FLAT_PRODUCTION_ACTION_SEMANTICS!r} or "
            f"{HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS!r} or "
            f"{STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS!r}"
        )
    production_enabled = bool(
        config.get("production_candidate_relative_features", False)
    )
    worker_enabled = bool(
        config.get("worker_candidate_relative_features", False)
    )
    production_names = tuple(
        config.get(
            "production_relative_feature_names",
            PRODUCTION_RELATIVE_FEATURE_NAMES if production_enabled else (),
        )
    )
    worker_names = tuple(
        config.get(
            "worker_relative_feature_names",
            WORKER_RELATIVE_FEATURE_NAMES if worker_enabled else (),
        )
    )
    allowed_production = (
        (V6_PRODUCTION_RELATIVE_FEATURE_NAMES,)
        if version == 6
        else (
            V6_PRODUCTION_RELATIVE_FEATURE_NAMES,
            V7_PRODUCTION_FUTURE_FEATURE_NAMES,
        )
    )
    allowed_worker = (
        (V6_WORKER_RELATIVE_FEATURE_NAMES,)
        if version == 6
        else (V6_WORKER_RELATIVE_FEATURE_NAMES, V7_WORKER_FUTURE_FEATURE_NAMES)
    )
    if production_enabled and production_names not in allowed_production:
        raise ValueError(
            "network.production_relative_feature_names does not match a "
            f"policy-head v{version} schema"
        )
    if worker_enabled and worker_names not in allowed_worker:
        raise ValueError(
            "network.worker_relative_feature_names does not match a "
            f"policy-head v{version} schema"
        )
    context_mode = str(
        config.get("candidate_context_mode", V6_CANDIDATE_CONTEXT_MODE)
    )
    allowed_context_modes = (
        {V6_CANDIDATE_CONTEXT_MODE}
        if version == 6
        else {V6_CANDIDATE_CONTEXT_MODE, V7_BOUNDED_CONTEXT_MODE}
    )
    if context_mode not in allowed_context_modes:
        raise ValueError(
            "network.candidate_context_mode is incompatible with "
            f"policy-head v{version}"
        )
    parameterization = str(
        config.get(
            "relative_weight_parameterization",
            V6_RELATIVE_WEIGHT_PARAMETERIZATION,
        )
    )
    if parameterization != V6_RELATIVE_WEIGHT_PARAMETERIZATION:
        raise ValueError(
            "network.relative_weight_parameterization must be "
            f"{V6_RELATIVE_WEIGHT_PARAMETERIZATION!r}"
        )
    worker_sharing = str(
        config.get(
            "worker_relative_weight_sharing",
            V6_WORKER_RELATIVE_WEIGHT_SHARING,
        )
    )
    if worker_sharing != V6_WORKER_RELATIVE_WEIGHT_SHARING:
        raise ValueError(
            "network.worker_relative_weight_sharing must be "
            f"{V6_WORKER_RELATIVE_WEIGHT_SHARING!r}"
        )
    common_gate = float(
        config.get(
            "common_context_gate_initial_logit",
            DEFAULT_CONTEXT_GATE_INITIAL_LOGIT,
        )
    )
    residual_gate = float(
        config.get(
            "residual_context_gate_initial_logit",
            DEFAULT_CONTEXT_GATE_INITIAL_LOGIT,
        )
    )
    if not np.isfinite(common_gate) or not np.isfinite(residual_gate):
        raise ValueError("network context-gate initial logits must be finite")
    residual_scale_ratio = float(config.get("residual_scale_ratio", 2.0))
    if not np.isfinite(residual_scale_ratio) or residual_scale_ratio <= 0.0:
        raise ValueError("network.residual_scale_ratio must be positive")
    preference_action_score_raw = config.get("preference_action_score", {})
    if not isinstance(preference_action_score_raw, Mapping):
        raise TypeError("network.preference_action_score must be an object")
    preference_action_score_enabled = bool(
        preference_action_score_raw.get("enabled", False)
    )
    preference_action_score_version = str(
        preference_action_score_raw.get(
            "version", DIRECT_PREFERENCE_ACTION_SCORE_VERSION
        )
    )
    preference_action_score_shared_scale = bool(
        preference_action_score_raw.get("shared_scale", True)
    )
    preference_action_score_initial_scale = float(
        preference_action_score_raw.get("initial_scale", 1.0)
    )
    preference_action_score_minimum_scale = float(
        preference_action_score_raw.get("minimum_scale", 0.1)
    )
    preference_action_score_production_scale = dict(
        preference_action_score_raw.get("production_scale", {})
    )
    preference_action_score_worker_scale = dict(
        preference_action_score_raw.get("worker_scale", {})
    )
    preference_action_score_standardization = str(
        preference_action_score_raw.get(
            "standardization",
            DIRECT_PREFERENCE_ACTION_SCORE_STANDARDIZATION,
        )
    )
    preference_action_score_scope = str(
        preference_action_score_raw.get("scope", "all")
    ).strip().lower()
    if preference_action_score_enabled:
        if version != 7:
            raise ValueError(
                "direct preference-action scoring requires policy_head_version=7"
            )
        if str(config.get("preference_conditioning", "none")) != (
            "separate_encoder_v1"
        ):
            raise ValueError(
                "direct preference-action scoring requires preference conditioning"
            )
        if not production_enabled or not worker_enabled:
            raise ValueError(
                "direct preference-action scoring requires both relative rankers"
            )
        uses_separate_scales = not preference_action_score_shared_scale
        expected_preference_version = (
            SAFE_WORKER_VARIANCE_SEPARATE_SCALE_VERSION
            if uses_separate_scales
            else SAFE_WORKER_VARIANCE_PREFERENCE_ACTION_SCORE_VERSION
            if preference_action_score_scope
            == "production_plus_safe_worker_variance"
            else DIRECT_PREFERENCE_ACTION_SCORE_VERSION
        )
        if preference_action_score_version != expected_preference_version:
            raise ValueError(
                "network.preference_action_score.version must be "
                f"{expected_preference_version!r}"
            )
        if uses_separate_scales and preference_action_score_scope != (
            "production_plus_safe_worker_variance"
        ):
            raise ValueError(
                "separate preference-action scales require the safe worker "
                "variance scope"
            )
        if preference_action_score_standardization != (
            DIRECT_PREFERENCE_ACTION_SCORE_STANDARDIZATION
        ):
            raise ValueError(
                "network.preference_action_score.standardization must be "
                f"{DIRECT_PREFERENCE_ACTION_SCORE_STANDARDIZATION!r}"
            )
        if preference_action_score_scope not in DIRECT_PREFERENCE_ACTION_SCORE_SCOPES:
            raise ValueError(
                "network.preference_action_score.scope must be 'all', "
                "'production_only', or "
                "'production_plus_safe_worker_variance'"
            )
        if not uses_separate_scales and (
            not np.isfinite(preference_action_score_minimum_scale)
            or preference_action_score_minimum_scale < 0.0
        ):
            raise ValueError(
                "preference-action minimum_scale must be finite and non-negative"
            )
        if not uses_separate_scales and (
            not np.isfinite(preference_action_score_initial_scale)
            or preference_action_score_initial_scale
            <= preference_action_score_minimum_scale
        ):
            raise ValueError(
                "preference-action initial_scale must exceed minimum_scale"
            )
        if uses_separate_scales:
            for label, raw, initial, minimum, maximum in (
                (
                    "production",
                    preference_action_score_production_scale,
                    1.5,
                    0.5,
                    3.0,
                ),
                (
                    "worker",
                    preference_action_score_worker_scale,
                    1.0,
                    0.1,
                    2.0,
                ),
            ):
                if set(raw) != {"initial_scale", "minimum_scale", "maximum_scale"}:
                    raise ValueError(
                        f"preference-action {label}_scale must contain "
                        "initial_scale, minimum_scale, and maximum_scale"
                    )
                values = (float(raw["initial_scale"]), float(raw["minimum_scale"]), float(raw["maximum_scale"]))
                if not all(np.isfinite(value) for value in values) or not (
                    values[1] < values[0] < values[2]
                ):
                    raise ValueError(
                        f"preference-action {label}_scale must satisfy "
                        "minimum < initial < maximum"
                    )
    production_defaults = (
        DEFAULT_PRODUCTION_RELATIVE_INITIAL_WEIGHTS
        if production_names == V6_PRODUCTION_RELATIVE_FEATURE_NAMES
        else tuple(0.30 if index < 4 else 0.20 for index in range(len(production_names)))
    )
    worker_defaults = (
        DEFAULT_WORKER_RELATIVE_INITIAL_WEIGHTS
        if worker_names == V6_WORKER_RELATIVE_FEATURE_NAMES
        else tuple(0.30 if index < 2 else 0.20 for index in range(len(worker_names)))
    )
    production_commit_set_scorer = bool(
        config.get("production_commit_set_scorer", False)
    )
    production_gate_raw = config.get("production_gate", {})
    if not isinstance(production_gate_raw, Mapping):
        raise TypeError("network.production_gate must be an object")
    production_gate_version = str(
        production_gate_raw.get("version", "none")
    )
    production_gate_preference_conditioning = bool(
        production_gate_raw.get("preference_conditioning", False)
    )
    production_gate_tie_break = str(
        production_gate_raw.get("tie_break", "none")
    )
    production_gate_freeze_base_after_feasibility = bool(
        production_gate_raw.get("freeze_base_after_feasibility", False)
    )
    flow_commit_residual = dict(
        production_gate_raw.get("flow_commit_residual", {})
    )
    if semantics in {
        HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS,
        STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS,
    }:
        if version != 7:
            raise ValueError(
                "hierarchical production actions require policy_head_version=7"
            )
        if not production_commit_set_scorer:
            raise ValueError(
                "hierarchical production actions require "
                "network.production_commit_set_scorer=true"
            )
    if semantics == STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS:
        if production_gate_version not in {
            STATE_ONLY_PRODUCTION_GATE_VERSION,
            STATE_ONLY_MONOTONE_FLOW_COMMIT_GATE_VERSION,
            STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION,
        }:
            raise ValueError(
                "state-only hierarchical production actions require "
                "network.production_gate.version="
                f"{STATE_ONLY_PRODUCTION_GATE_VERSION!r}"
            )
        if production_gate_preference_conditioning:
            raise ValueError(
                "state-only production gate must not use preference conditioning"
            )
        if production_gate_tie_break != STATE_ONLY_PRODUCTION_GATE_TIE_BREAK:
            raise ValueError(
                "state-only production gate tie_break must be 'commit'"
            )
        if production_gate_version == STATE_ONLY_MONOTONE_FLOW_COMMIT_GATE_VERSION:
            if not production_gate_freeze_base_after_feasibility:
                raise ValueError("E2.5 requires freezing the base gate after feasibility")
            if set(flow_commit_residual) != {
                "enabled", "activation_threshold", "scale", "apply_during_feasibility"
            }:
                raise ValueError("E2.5 flow_commit_residual has an invalid schema")
            threshold = float(flow_commit_residual["activation_threshold"])
            scale = float(flow_commit_residual["scale"])
            if not bool(flow_commit_residual["enabled"]) or threshold != 0.2:
                raise ValueError("E2.5 residual must be enabled at threshold 0.2")
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("E2.5 residual scale must be finite and positive")
            if bool(flow_commit_residual["apply_during_feasibility"]):
                raise ValueError("E2.5 residual must be disabled during feasibility")
        elif (
            production_gate_version
            == STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION
        ):
            if not production_gate_freeze_base_after_feasibility:
                raise ValueError("E2.6 requires freezing the base gate after feasibility")
            if set(flow_commit_residual) != {
                "enabled",
                "version",
                "activation_threshold",
                "initial_scale",
                "maximum_scale",
                "apply_during_feasibility",
            }:
                raise ValueError("E2.6 flow_commit_residual has an invalid schema")
            threshold = float(flow_commit_residual["activation_threshold"])
            initial_scale = float(flow_commit_residual["initial_scale"])
            maximum_scale = float(flow_commit_residual["maximum_scale"])
            if not bool(flow_commit_residual["enabled"]) or threshold != 0.2:
                raise ValueError("E2.6 residual must be enabled at threshold 0.2")
            if str(flow_commit_residual["version"]) != "bounded_state_scale_v1":
                raise ValueError("E2.6 residual has an unsupported version")
            if not (
                np.isfinite(initial_scale)
                and np.isfinite(maximum_scale)
                and 0.0 < initial_scale < maximum_scale
            ):
                raise ValueError(
                    "E2.6 residual scales must satisfy 0 < initial < maximum"
                )
            if bool(flow_commit_residual["apply_during_feasibility"]):
                raise ValueError("E2.6 residual must be disabled during feasibility")
        elif production_gate_freeze_base_after_feasibility or flow_commit_residual:
            raise ValueError("E2.4 state-only gate does not accept E2.5 residual fields")
    elif production_gate_raw:
        raise ValueError(
            "network.production_gate is only valid for "
            "hierarchical_state_only_gate_then_pair_v3"
        )
    return {
        "policy_head_version": version,
        "production_action_semantics": semantics,
        "production_relative_feature_names": production_names,
        "worker_relative_feature_names": worker_names,
        "candidate_context_mode": context_mode,
        "production_commit_set_scorer": production_commit_set_scorer,
        "production_gate_version": production_gate_version,
        "production_gate_preference_conditioning": (
            production_gate_preference_conditioning
        ),
        "production_gate_tie_break": production_gate_tie_break,
        "production_gate_freeze_base_after_feasibility": (
            production_gate_freeze_base_after_feasibility
        ),
        "production_gate_flow_commit_residual": flow_commit_residual,
        "future_value_features": bool(
            config.get("future_value_features", False)
        ),
        "worker_common_context_enabled": bool(
            config.get("worker_common_context_enabled", True)
        ),
        "residual_scale_ratio": residual_scale_ratio,
        "preference_action_score_enabled": preference_action_score_enabled,
        "preference_action_score_version": preference_action_score_version,
        "preference_action_score_shared_scale": (
            preference_action_score_shared_scale
        ),
        "preference_action_score_initial_scale": (
            preference_action_score_initial_scale
        ),
        "preference_action_score_minimum_scale": (
            preference_action_score_minimum_scale
        ),
        "preference_action_score_production_scale": preference_action_score_production_scale,
        "preference_action_score_worker_scale": preference_action_score_worker_scale,
        "preference_action_score_standardization": (
            preference_action_score_standardization
        ),
        "preference_action_score_scope": preference_action_score_scope,
        "production_relative_initial_weights": (
            _positive_weight_tuple(
                config,
                "production_relative_initial_weights",
                production_defaults,
            )
            if production_enabled
            else production_defaults
        ),
        "worker_relative_initial_weights": (
            _positive_weight_tuple(
                config,
                "worker_relative_initial_weights",
                worker_defaults,
            )
            if worker_enabled
            else worker_defaults
        ),
        "common_context_gate_initial_logit": common_gate,
        "residual_context_gate_initial_logit": residual_gate,
    }


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
    preference_conditioning = str(
        config.get("preference_conditioning", "none")
    )
    if preference_conditioning not in {"none", "separate_encoder_v1"}:
        raise ValueError(
            "network.preference_conditioning must be 'none' or "
            "'separate_encoder_v1'"
        )
    if encoder_type != "hetero_gnn" and preference_conditioning != "none":
        raise ValueError(
            "preference conditioning is only supported by the hetero_gnn encoder"
        )
    if preference_conditioning != "none":
        normalized["preference_conditioning"] = preference_conditioning
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
        configured_policy_head = int(
            config.get("policy_head_version", POLICY_HEAD_VERSION)
        )
        saved_policy_head = int(
            checkpoint_spec.get("policy_head_version", 3)
        )
        if configured_policy_head != saved_policy_head:
            raise ValueError(
                "checkpoint policy head version is incompatible with the "
                "current network: "
                f"configured={configured_policy_head}, "
                f"checkpoint={saved_policy_head}; older policy heads are "
                "not automatically converted to v6 or v7"
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
            config.get("candidate_context_mode", V6_CANDIDATE_CONTEXT_MODE)
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
                V6_WORKER_RELATIVE_WEIGHT_SHARING,
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
        for field in (
            "relative_weight_parameterization",
            "production_relative_initial_weights",
            "worker_relative_initial_weights",
            "common_context_gate_initial_logit",
            "residual_context_gate_initial_logit",
        ):
            configured_value = config.get(field)
            saved_value = checkpoint_spec.get(field)
            if configured_value != saved_value:
                raise ValueError(
                    f"checkpoint {field} is incompatible with the current "
                    f"network: configured={configured_value}, "
                    f"checkpoint={saved_value}"
                )
        if configured_policy_head == 7:
            for field in (
                "production_commit_set_scorer",
                "future_value_features",
                "worker_common_context_enabled",
                "residual_scale_ratio",
                "action_set_feature_names",
            ):
                configured_defaults = {
                    "production_commit_set_scorer": False,
                    "future_value_features": False,
                    "worker_common_context_enabled": True,
                    "residual_scale_ratio": 2.0,
                }
                if field == "action_set_feature_names":
                    configured_value = (
                        PRODUCTION_ACTION_SET_FEATURE_NAMES
                        if bool(
                            config.get(
                                "production_commit_set_scorer", False
                            )
                        )
                        else ()
                    )
                else:
                    configured_value = config.get(
                        field, configured_defaults[field]
                    )
                saved_value = checkpoint_spec.get(field)
                if configured_value != saved_value:
                    raise ValueError(
                        f"checkpoint {field} is incompatible with the "
                        "current network: "
                        f"configured={configured_value}, "
                        f"checkpoint={saved_value}"
                    )
            configured_preference_action = dict(
                config.get("preference_action_score", {})
            )
            saved_preference_action = dict(
                checkpoint_spec.get("preference_action_score", {})
            )
            if configured_preference_action:
                configured_preference_action.setdefault("scope", "all")
            if saved_preference_action:
                saved_preference_action.setdefault("scope", "all")
            if configured_preference_action != saved_preference_action:
                raise ValueError(
                    "checkpoint preference_action_score is incompatible "
                    "with the current network: "
                    f"configured={configured_preference_action}, "
                    f"checkpoint={saved_preference_action}"
                )
            configured_production_gate = dict(
                config.get("production_gate", {})
            )
            saved_production_gate = dict(
                checkpoint_spec.get("production_gate", {})
            )
            # Frozen is runtime state, saved for recovery but not an
            # architecture/configuration change.
            saved_production_gate.pop("base_gate_frozen", None)
            configured_production_gate.pop("base_gate_frozen", None)
            saved_production_gate.pop("residual_active", None)
            configured_production_gate.pop("residual_active", None)
            if configured_production_gate != saved_production_gate:
                raise ValueError(
                    "checkpoint production_gate is incompatible with the "
                    "current network: "
                    f"configured={configured_production_gate}, "
                    f"checkpoint={saved_production_gate}"
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
    output_dim: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
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
        production_relative_initial_weights: Sequence[float] = (
            DEFAULT_PRODUCTION_RELATIVE_INITIAL_WEIGHTS
        ),
        worker_relative_initial_weights: Sequence[float] = (
            DEFAULT_WORKER_RELATIVE_INITIAL_WEIGHTS
        ),
        common_context_gate_initial_logit: float = (
            DEFAULT_CONTEXT_GATE_INITIAL_LOGIT
        ),
        residual_context_gate_initial_logit: float = (
            DEFAULT_CONTEXT_GATE_INITIAL_LOGIT
        ),
        policy_head_version: int = POLICY_HEAD_VERSION,
        production_action_semantics: str = FLAT_PRODUCTION_ACTION_SEMANTICS,
        production_relative_feature_names: Sequence[str] = (
            V6_PRODUCTION_RELATIVE_FEATURE_NAMES
        ),
        worker_relative_feature_names: Sequence[str] = (
            V6_WORKER_RELATIVE_FEATURE_NAMES
        ),
        candidate_context_mode: str = V6_CANDIDATE_CONTEXT_MODE,
        production_commit_set_scorer: bool = False,
        production_gate_version: str = "none",
        production_gate_preference_conditioning: bool = False,
        production_gate_tie_break: str = "none",
        production_gate_freeze_base_after_feasibility: bool = False,
        production_gate_flow_commit_residual: Mapping[str, Any] | None = None,
        future_value_features: bool = False,
        worker_common_context_enabled: bool = True,
        residual_scale_ratio: float = 2.0,
        preference_conditioning: str = "none",
        preference_action_score_enabled: bool = False,
        preference_action_score_version: str = (
            DIRECT_PREFERENCE_ACTION_SCORE_VERSION
        ),
        preference_action_score_shared_scale: bool = True,
        preference_action_score_initial_scale: float = 1.0,
        preference_action_score_minimum_scale: float = 0.1,
        preference_action_score_production_scale: Mapping[str, Any] | None = None,
        preference_action_score_worker_scale: Mapping[str, Any] | None = None,
        preference_action_score_standardization: str = (
            DIRECT_PREFERENCE_ACTION_SCORE_STANDARDIZATION
        ),
        preference_action_score_scope: str = "all",
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
        self.production_relative_initial_weights = tuple(
            float(value) for value in production_relative_initial_weights
        )
        self.worker_relative_initial_weights = tuple(
            float(value) for value in worker_relative_initial_weights
        )
        self.common_context_gate_initial_logit = float(
            common_context_gate_initial_logit
        )
        self.residual_context_gate_initial_logit = float(
            residual_context_gate_initial_logit
        )
        self.policy_head_version = int(policy_head_version)
        self.production_action_semantics = str(production_action_semantics)
        self.production_relative_feature_names = tuple(
            production_relative_feature_names
        )
        self.worker_relative_feature_names = tuple(worker_relative_feature_names)
        self.candidate_context_mode = str(candidate_context_mode)
        self.use_production_commit_set_scorer = bool(
            production_commit_set_scorer
        )
        self.production_gate_version = str(production_gate_version)
        self.production_gate_preference_conditioning = bool(
            production_gate_preference_conditioning
        )
        self.production_gate_tie_break = str(production_gate_tie_break)
        self.production_gate_freeze_base_after_feasibility = bool(
            production_gate_freeze_base_after_feasibility
        )
        self.production_gate_flow_commit_residual = dict(
            production_gate_flow_commit_residual or {}
        )
        self.production_state_gate_frozen = False
        self.production_flow_commit_residual_active = bool(
            self.production_gate_flow_commit_residual.get(
                "apply_during_feasibility", False
            )
        )
        self.use_future_value_features = bool(future_value_features)
        self.worker_common_context_enabled = bool(
            worker_common_context_enabled
        )
        self.residual_scale_ratio = float(residual_scale_ratio)
        self.preference_conditioning = str(preference_conditioning)
        self.preference_action_score_enabled = bool(
            preference_action_score_enabled
        )
        self.preference_action_score_version = str(
            preference_action_score_version
        )
        self.preference_action_score_shared_scale = bool(
            preference_action_score_shared_scale
        )
        self.preference_action_score_initial_scale = float(
            preference_action_score_initial_scale
        )
        self.preference_action_score_minimum_scale = float(
            preference_action_score_minimum_scale
        )
        self.preference_action_score_production_scale = dict(
            preference_action_score_production_scale or {}
        )
        self.preference_action_score_worker_scale = dict(
            preference_action_score_worker_scale or {}
        )
        self.preference_action_score_standardization = str(
            preference_action_score_standardization
        )
        self.preference_action_score_scope = str(
            preference_action_score_scope
        )
        if self.preference_conditioning not in {
            "none",
            "separate_encoder_v1",
        }:
            raise ValueError("unsupported preference conditioning mode")
        self.use_preference_conditioning = (
            self.preference_conditioning == "separate_encoder_v1"
        )
        if self.preference_action_score_enabled and not (
            self.use_preference_conditioning
            and self.use_production_candidate_relative_features
            and self.use_worker_candidate_relative_features
        ):
            raise ValueError(
                "direct preference-action scoring requires preference "
                "conditioning and both relative rankers"
            )
        if self.preference_action_score_scope not in (
            DIRECT_PREFERENCE_ACTION_SCORE_SCOPES
        ):
            raise ValueError("unsupported preference-action scoring scope")
        if (
            self.production_action_semantics
            == STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS
            and (
                self.production_gate_version not in {
                    STATE_ONLY_PRODUCTION_GATE_VERSION,
                    STATE_ONLY_MONOTONE_FLOW_COMMIT_GATE_VERSION,
                    STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION,
                }
                or self.production_gate_preference_conditioning
                or self.production_gate_tie_break
                != STATE_ONLY_PRODUCTION_GATE_TIE_BREAK
            )
        ):
            raise ValueError("invalid state-only production gate configuration")
        if (
            len(self.production_relative_initial_weights)
            != len(self.production_relative_feature_names)
        ):
            raise ValueError("production relative initial-weight width is invalid")
        if (
            len(self.worker_relative_initial_weights)
            != len(self.worker_relative_feature_names)
        ):
            raise ValueError("worker relative initial-weight width is invalid")
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
        self.preference_encoder = (
            _mlp(len(PREFERENCE_NAMES), self.hidden_dim)
            if self.use_preference_conditioning
            else None
        )
        if self.preference_action_score_enabled and self.preference_action_score_shared_scale:
            initial_magnitude = (
                self.preference_action_score_initial_scale
                - self.preference_action_score_minimum_scale
            )
            self.preference_action_scale_raw = nn.Parameter(
                torch.tensor(math.log(math.expm1(initial_magnitude)))
            )
        else:
            self.register_parameter("preference_action_scale_raw", None)
        if self.preference_action_score_enabled and not self.preference_action_score_shared_scale:
            self.production_preference_action_scale_raw = nn.Parameter(
                self._bounded_scale_raw(
                    self.preference_action_score_production_scale
                )
            )
            self.worker_preference_action_scale_raw = nn.Parameter(
                self._bounded_scale_raw(
                    self.preference_action_score_worker_scale
                )
            )
        else:
            self.register_parameter("production_preference_action_scale_raw", None)
            self.register_parameter("worker_preference_action_scale_raw", None)
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
            self.hidden_dim * (5 if self.use_preference_conditioning else 4),
            self.hidden_dim,
            self.dropout_probability,
        )
        self.worker_scorer = _head(
            self.hidden_dim * (6 if self.use_preference_conditioning else 5),
            self.hidden_dim,
            self.dropout_probability,
        )
        if self.use_production_candidate_relative_features:
            self.production_relative_ranker = nn.Linear(
                len(self.production_relative_feature_names), 1, bias=False
            )
            self._initialize_monotone_ranker(
                self.production_relative_ranker,
                self.production_relative_initial_weights,
            )
            self._initialize_zero_context_output(self.production_scorer)
            self.production_context_gate = nn.Parameter(
                torch.full((), self.common_context_gate_initial_logit)
            )
            self.production_residual_context_gate = nn.Parameter(
                torch.full((), self.residual_context_gate_initial_logit)
            )
        else:
            self.production_relative_ranker = None
            self.register_parameter("production_context_gate", None)
            self.register_parameter("production_residual_context_gate", None)
        if self.use_worker_candidate_relative_features:
            self.worker_relative_ranker = nn.Linear(
                len(self.worker_relative_feature_names), 1, bias=False
            )
            self._initialize_monotone_ranker(
                self.worker_relative_ranker,
                self.worker_relative_initial_weights,
            )
            self._initialize_zero_context_output(self.worker_scorer)
            self.worker_context_gate = nn.Parameter(
                torch.full((), self.common_context_gate_initial_logit)
            )
            self.worker_residual_context_gate = nn.Parameter(
                torch.full((), self.residual_context_gate_initial_logit)
            )
        else:
            self.worker_relative_ranker = None
            self.register_parameter("worker_context_gate", None)
            self.register_parameter("worker_residual_context_gate", None)
        context_dim = self.hidden_dim * (
            len(NODE_TYPES) + 1 + int(self.use_preference_conditioning)
        )
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
        self.production_commit_set = (
            _head(
                len(PRODUCTION_ACTION_SET_FEATURE_NAMES),
                self.hidden_dim,
                self.dropout_probability,
            )
            if (
                self.use_production_commit_set_scorer
                and self.production_action_semantics
                != STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS
            )
            else None
        )
        if self.production_commit_set is not None:
            self._initialize_zero_context_output(self.production_commit_set)
        self.production_state_gate = (
            _head(
                len(PRODUCTION_ACTION_SET_FEATURE_NAMES),
                self.hidden_dim,
                self.dropout_probability,
                output_dim=2,
            )
            if self.production_action_semantics
            == STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS
            else None
        )
        if self.production_state_gate is not None:
            self._initialize_zero_context_output(self.production_state_gate)
        self.production_flow_commit_residual_state = (
            _head(
                len(PRODUCTION_ACTION_SET_FEATURE_NAMES),
                self.hidden_dim,
                self.dropout_probability,
            )
            if self.production_gate_version
            == STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION
            else None
        )
        if self.production_flow_commit_residual_state is not None:
            self._initialize_bounded_state_scale_output(
                self.production_flow_commit_residual_state,
                initial_scale=float(
                    self.production_gate_flow_commit_residual["initial_scale"]
                ),
                maximum_scale=float(
                    self.production_gate_flow_commit_residual["maximum_scale"]
                ),
            )
        self._latest_policy_decision_diagnostics: list[dict[str, Any]] = []

    def network_spec(self) -> dict[str, Any]:
        spec = {
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
            "policy_head_version": self.policy_head_version,
            "production_action_semantics": self.production_action_semantics,
            "production_relative_feature_names": (
                self.production_relative_feature_names
                if self.use_production_candidate_relative_features
                else ()
            ),
            "worker_relative_feature_names": (
                self.worker_relative_feature_names
                if self.use_worker_candidate_relative_features
                else ()
            ),
            "relative_weight_parameterization": (
                V6_RELATIVE_WEIGHT_PARAMETERIZATION
            ),
            "production_relative_initial_weights": (
                self.production_relative_initial_weights
                if self.use_production_candidate_relative_features
                else ()
            ),
            "worker_relative_initial_weights": (
                self.worker_relative_initial_weights
                if self.use_worker_candidate_relative_features
                else ()
            ),
            "candidate_context_mode": self.candidate_context_mode,
            "worker_relative_weight_sharing": (
                V6_WORKER_RELATIVE_WEIGHT_SHARING
            ),
            "common_context_gate_initial_logit": (
                self.common_context_gate_initial_logit
            ),
            "residual_context_gate_initial_logit": (
                self.residual_context_gate_initial_logit
            ),
            "observation_schema_version": (
                5
                if self.use_preference_conditioning
                else 4
                if self.policy_head_version == 7
                else 3
            ),
            "feature_dimensions": dict(self.feature_dimensions),
            "edge_feature_dimensions": dict(self.edge_feature_dimensions),
        }
        if self.policy_head_version == 7:
            spec.update(
                {
                    "production_commit_set_scorer": (
                        self.use_production_commit_set_scorer
                    ),
                    "production_gate": (
                        (
                            {
                                "version": self.production_gate_version,
                                "preference_conditioning": self.production_gate_preference_conditioning,
                                "tie_break": self.production_gate_tie_break,
                                "freeze_base_after_feasibility": self.production_gate_freeze_base_after_feasibility,
                                "flow_commit_residual": dict(self.production_gate_flow_commit_residual),
                                "base_gate_frozen": self.production_state_gate_frozen,
                                "residual_active": self.production_flow_commit_residual_active,
                            }
                            if self.production_gate_version
                            in {
                                STATE_ONLY_MONOTONE_FLOW_COMMIT_GATE_VERSION,
                                STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION,
                            }
                            else {
                                "version": self.production_gate_version,
                                "preference_conditioning": self.production_gate_preference_conditioning,
                                "tie_break": self.production_gate_tie_break,
                            }
                        )
                        if self.production_action_semantics
                        == STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS
                        else {}
                    ),
                    "future_value_features": self.use_future_value_features,
                    "worker_common_context_enabled": (
                        self.worker_common_context_enabled
                    ),
                    "residual_scale_ratio": self.residual_scale_ratio,
                    "action_set_feature_names": (
                        PRODUCTION_ACTION_SET_FEATURE_NAMES
                        if self.use_production_commit_set_scorer
                        else ()
                    ),
                    "preference_action_score": (
                        {
                            "enabled": True,
                            "version": self.preference_action_score_version,
                            "shared_scale": (
                                self.preference_action_score_shared_scale
                            ),
                            "initial_scale": (
                                self.preference_action_score_initial_scale
                            ),
                            "minimum_scale": (
                                self.preference_action_score_minimum_scale
                            ),
                            **(
                                {
                                    "production_scale": dict(self.preference_action_score_production_scale),
                                    "worker_scale": dict(self.preference_action_score_worker_scale),
                                }
                                if not self.preference_action_score_shared_scale
                                else {}
                            ),
                            "standardization": (
                                self.preference_action_score_standardization
                            ),
                            "scope": self.preference_action_score_scope,
                        }
                        if self.preference_action_score_enabled
                        else {}
                    ),
                }
            )
        if self.use_preference_conditioning:
            spec.update(
                {
                    "preference_conditioning": self.preference_conditioning,
                    "preference_names": PREFERENCE_NAMES,
                }
            )
        return spec

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

        self._latest_policy_decision_diagnostics.clear()

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
        preference_values = torch.stack(
            [
                torch.as_tensor(
                    observation.preference,
                    dtype=torch.float32,
                    device=device,
                )
                for observation in observations
            ]
        )
        if self.preference_encoder is not None:
            if preference_values.shape != (len(observations), 3):
                raise ValueError("preference batch must have shape (B, 3)")
            if any(
                tuple(observation.preference_names) != PREFERENCE_NAMES
                for observation in observations
            ):
                raise ValueError(
                    "preference names must be flow/cost/variance in order"
                )
            preference_embeddings = self.preference_encoder(preference_values)
        else:
            preference_embeddings = global_embeddings.new_empty(
                (len(observations), 0)
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
            + (global_embeddings, preference_embeddings),
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
                production_gate_logits = None
                production_gate_base_logits = None
                production_gate_logit_boost = None
                production_gate_state_scale = None
                if (
                    self.production_action_semantics
                    == STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS
                ):
                    production_gate_base_logits = (
                        self._state_only_production_gate_logits(
                            observation,
                            dtype=global_embeddings.dtype,
                            device=device,
                        )
                    )
                    production_gate_state_scale = (
                        self._production_flow_commit_residual_scale(
                            observation,
                            dtype=global_embeddings.dtype,
                            device=device,
                        )
                    )
                    production_gate_logits, production_gate_logit_boost = (
                        self._final_production_gate_logits(
                            production_gate_base_logits,
                            preference_values[batch_index],
                            state_scale=production_gate_state_scale,
                        )
                    )
                pair_logits, commit_set_logit = self._production_logits(
                    observation,
                    operation_embeddings,
                    machine_embeddings,
                    global_embeddings[batch_index],
                    preference_embeddings[batch_index],
                    preference_values[batch_index],
                    masks[batch_index],
                    production_gate_logits=production_gate_logits,
                    production_gate_base_logits=production_gate_base_logits,
                    production_gate_logit_boost=production_gate_logit_boost,
                    production_gate_state_scale=production_gate_state_scale,
                    device=device,
                )
                if (
                    self.production_action_semantics
                    == HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS
                ):
                    terminal_logit = self.production_defer(
                        context[batch_index]
                    ).reshape(1)
                    unmasked_logits = self._hierarchical_production_logits(
                        pair_logits,
                        commit_set_logit,
                        terminal_logit.reshape(()),
                        masks[batch_index],
                    )
                elif (
                    self.production_action_semantics
                    == STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS
                ):
                    if production_gate_logits is None:
                        raise RuntimeError("state-only production gate is missing")
                    unmasked_logits = self._hierarchical_production_logits(
                        pair_logits,
                        production_gate_logits[0],
                        production_gate_logits[1],
                        masks[batch_index],
                    )
                else:
                    terminal_logit = self.production_defer(
                        context[batch_index]
                    ).reshape(1)
                    unmasked_logits = torch.cat((pair_logits, terminal_logit))
            elif observation.decision_type == DecisionType.WORKER:
                pair_logits = self._worker_logits(
                    observation,
                    operation_embeddings,
                    machine_embeddings,
                    worker_embeddings,
                    global_embeddings[batch_index],
                    preference_embeddings[batch_index],
                    preference_values[batch_index],
                    masks[batch_index],
                    device=device,
                )
                terminal_logit = self.worker_advance(
                    context[batch_index]
                ).reshape(1)
                unmasked_logits = torch.cat((pair_logits, terminal_logit))
            else:
                raise ValueError(
                    f"unsupported decision type {observation.decision_type}"
                )
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

    def _state_only_production_gate_logits(
        self,
        observation: HeterogeneousGraphObservation,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Return preference-independent commit/defer gate logits for E2.4."""

        if self.production_state_gate is None:
            raise RuntimeError("state-only production gate is not initialized")
        if tuple(observation.action_set_feature_names) != (
            PRODUCTION_ACTION_SET_FEATURE_NAMES
        ):
            raise ValueError(
                "production action-set feature schema does not match v7"
            )
        action_set_features = torch.as_tensor(
            observation.action_set_features,
            dtype=dtype,
            device=device,
        )
        if action_set_features.shape != (
            len(PRODUCTION_ACTION_SET_FEATURE_NAMES),
        ):
            raise ValueError("production action-set feature vector has wrong width")
        return self.production_state_gate(action_set_features).reshape(2)

    def _final_production_gate_logits(
        self,
        base_logits: torch.Tensor,
        preference: torch.Tensor,
        *,
        state_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply E2.5/E2.6's positive-only flow residual to commit."""

        if base_logits.shape != (2,) or preference.shape != (3,):
            raise ValueError("production gate logits or preference shape is invalid")
        boost = base_logits.new_zeros(())
        if (
            self.production_gate_version
            == STATE_ONLY_MONOTONE_FLOW_COMMIT_GATE_VERSION
            and self.production_flow_commit_residual_active
        ):
            threshold = float(
                self.production_gate_flow_commit_residual["activation_threshold"]
            )
            scale = float(self.production_gate_flow_commit_residual["scale"])
            boost = scale * torch.clamp(preference[0] - threshold, min=0.0)
        elif (
            self.production_gate_version
            == STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION
            and self.production_flow_commit_residual_active
        ):
            if state_scale is None or state_scale.ndim != 0:
                raise ValueError("E2.6 requires a scalar state residual scale")
            threshold = float(
                self.production_gate_flow_commit_residual["activation_threshold"]
            )
            boost = state_scale * torch.clamp(preference[0] - threshold, min=0.0)
        return base_logits + torch.stack((boost, boost.new_zeros(()))), boost

    def _production_flow_commit_residual_scale(
        self,
        observation: HeterogeneousGraphObservation,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Return E2.6's bounded non-negative state-dependent scale."""

        if (
            self.production_gate_version
            != STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION
        ):
            return torch.zeros((), dtype=dtype, device=device)
        if self.production_flow_commit_residual_state is None:
            raise RuntimeError("E2.6 state residual head is not initialized")
        if tuple(observation.action_set_feature_names) != (
            PRODUCTION_ACTION_SET_FEATURE_NAMES
        ):
            raise ValueError(
                "production action-set feature schema does not match v7"
            )
        features = torch.as_tensor(
            observation.action_set_features,
            dtype=dtype,
            device=device,
        )
        if features.shape != (len(PRODUCTION_ACTION_SET_FEATURE_NAMES),):
            raise ValueError("production action-set feature vector has wrong width")
        maximum_scale = float(
            self.production_gate_flow_commit_residual["maximum_scale"]
        )
        return maximum_scale * torch.sigmoid(
            self.production_flow_commit_residual_state(features).reshape(())
        )

    def counterfactual_production_gate_batch(
        self,
        observations: Sequence[HeterogeneousGraphObservation],
        action_masks: Sequence[np.ndarray | torch.Tensor],
        *,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        """Evaluate E2.6 low/high-flow gates on identical production states.

        The returned tensors align with ``observations``.  An entry is eligible
        exactly when both gate choices are legal and the frozen base gate would
        greedily defer.  The helper deliberately touches only the base/state
        gate path, so its auxiliary loss cannot update pair, worker, critic or
        graph-encoder parameters.
        """

        count = len(observations)
        if count != len(action_masks):
            raise ValueError("counterfactual observations and masks must align")
        if (
            self.production_gate_version
            != STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION
        ):
            zeros = torch.zeros((count,), dtype=torch.float32, device=device)
            return {
                "base_margin": zeros,
                "low_margin": zeros,
                "high_margin": zeros,
                "state_scale": zeros,
                "eligible": torch.zeros((count,), dtype=torch.bool, device=device),
                "high_flow_commit": torch.zeros((count,), dtype=torch.bool, device=device),
                "high_flow_flip": torch.zeros((count,), dtype=torch.bool, device=device),
                "low_flow_identity_violation": torch.zeros((count,), dtype=torch.bool, device=device),
                "monotonicity_violation": torch.zeros((count,), dtype=torch.bool, device=device),
            }
        parameter = next(self.parameters())
        low_preference = parameter.new_tensor((0.2, 0.4, 0.4))
        high_preference = parameter.new_tensor((1.0, 0.0, 0.0))
        base_margins: list[torch.Tensor] = []
        low_margins: list[torch.Tensor] = []
        high_margins: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        eligible_values: list[bool] = []
        high_commit_values: list[bool] = []
        identity_values: list[bool] = []
        monotonicity_values: list[bool] = []
        for observation, action_mask in zip(observations, action_masks):
            if observation.decision_type != DecisionType.PRODUCTION:
                zero = parameter.new_zeros(())
                base_margins.append(zero)
                low_margins.append(zero)
                high_margins.append(zero)
                scales.append(zero)
                eligible_values.append(False)
                high_commit_values.append(False)
                identity_values.append(False)
                monotonicity_values.append(False)
                continue
            mask = self._validate_action_mask(action_mask, device=device)
            base_logits = self._state_only_production_gate_logits(
                observation,
                dtype=parameter.dtype,
                device=device,
            )
            state_scale = self._production_flow_commit_residual_scale(
                observation,
                dtype=parameter.dtype,
                device=device,
            )
            low_logits, _ = self._final_production_gate_logits(
                base_logits,
                low_preference,
                state_scale=state_scale,
            )
            high_logits, _ = self._final_production_gate_logits(
                base_logits,
                high_preference,
                state_scale=state_scale,
            )
            base_margin = base_logits[0] - base_logits[1]
            low_margin = low_logits[0] - low_logits[1]
            high_margin = high_logits[0] - high_logits[1]
            commit_legal = bool((~mask[:-1]).any().detach().cpu())
            defer_legal = not bool(mask[-1].detach().cpu())
            eligible = commit_legal and defer_legal and bool(
                (base_margin < 0.0).detach().cpu()
            )
            high_commit = bool((high_margin >= 0.0).detach().cpu())
            base_margins.append(base_margin)
            low_margins.append(low_margin)
            high_margins.append(high_margin)
            scales.append(state_scale)
            eligible_values.append(eligible)
            high_commit_values.append(high_commit)
            identity_values.append(
                bool((low_margin - base_margin).abs().gt(1e-7).detach().cpu())
            )
            monotonicity_values.append(
                bool((high_margin < low_margin - 1e-7).detach().cpu())
            )
        eligible = torch.as_tensor(eligible_values, dtype=torch.bool, device=device)
        high_commit = torch.as_tensor(
            high_commit_values, dtype=torch.bool, device=device
        )
        return {
            "base_margin": torch.stack(base_margins),
            "low_margin": torch.stack(low_margins),
            "high_margin": torch.stack(high_margins),
            "state_scale": torch.stack(scales),
            "eligible": eligible,
            "high_flow_commit": high_commit,
            "high_flow_flip": eligible & high_commit,
            "low_flow_identity_violation": torch.as_tensor(
                identity_values, dtype=torch.bool, device=device
            ),
            "monotonicity_violation": torch.as_tensor(
                monotonicity_values, dtype=torch.bool, device=device
            ),
        }

    def set_production_state_gate_frozen(self, frozen: bool) -> None:
        """Freeze the E2.5 state-only base gate without changing its value."""

        self.production_state_gate_frozen = bool(frozen)
        if self.production_state_gate is not None:
            for parameter in self.production_state_gate.parameters():
                parameter.requires_grad_(not frozen)

    def set_production_flow_commit_residual_enabled(self, enabled: bool) -> None:
        """Enable the E2.5 residual only after the feasibility transition."""

        if self.production_gate_version not in {
            STATE_ONLY_MONOTONE_FLOW_COMMIT_GATE_VERSION,
            STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION,
        }:
            if enabled:
                raise ValueError("only E2.5 has a flow commit residual")
            return
        self.production_flow_commit_residual_active = bool(enabled)

    def _production_logits(
        self,
        observation: HeterogeneousGraphObservation,
        operation_embeddings: torch.Tensor,
        machine_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        preference_embedding: torch.Tensor,
        preference: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        production_gate_logits: torch.Tensor | None = None,
        production_gate_base_logits: torch.Tensor | None = None,
        production_gate_logit_boost: torch.Tensor | None = None,
        production_gate_state_scale: torch.Tensor | None = None,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        preference_pairs = preference_embedding[None, None, :].expand(
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
                    preference_pairs,
                ),
                dim=-1,
            )
        ).reshape(-1)
        commit_set_logit = contextual_logits.new_zeros(())
        if self.production_commit_set is not None:
            if tuple(observation.action_set_feature_names) != (
                PRODUCTION_ACTION_SET_FEATURE_NAMES
            ):
                raise ValueError(
                    "production action-set feature schema does not match v7"
                )
            action_set_features = torch.as_tensor(
                observation.action_set_features,
                dtype=contextual_logits.dtype,
                device=device,
            )
            commit_set_logit = self.production_commit_set(
                action_set_features
            ).squeeze(-1)
        if not self.use_production_candidate_relative_features:
            pair_logits = contextual_logits
            if (
                self.production_action_semantics
                == FLAT_PRODUCTION_ACTION_SEMANTICS
            ):
                pair_logits = pair_logits + commit_set_logit
            return pair_logits, commit_set_logit
        if self.production_relative_ranker is None:
            raise RuntimeError("production relative ranker is not initialized")
        edge_names = capable.feature_names
        processing = dense_features[
            :, edge_names.index("processing_time_norm")
        ]
        reconfiguration = dense_features[
            :, edge_names.index("reconfiguration_time_norm")
        ]
        fixed_reconfiguration_cost = (
            dense_features[
                :, edge_names.index("fixed_disassembly_cost_norm")
            ]
            + dense_features[
                :, edge_names.index("fixed_installation_cost_norm")
            ]
        )
        labor_cost = dense_features[
            :, edge_names.index("estimated_labor_cost_norm")
        ]
        downtime_cost = dense_features[
            :, edge_names.index("estimated_downtime_cost_norm")
        ]
        horizon_slack = dense_features[
            :, edge_names.index("horizon_slack_norm")
        ]
        production_columns = {
            "processing_time_norm": processing,
            "reconfiguration_time_norm": reconfiguration,
            "fixed_reconfiguration_cost_norm": fixed_reconfiguration_cost,
            "estimated_labor_cost_norm": labor_cost,
            "estimated_downtime_cost_norm": downtime_cost,
            "total_reconfiguration_cost_norm": (
                fixed_reconfiguration_cost + labor_cost + downtime_cost
            ),
            "horizon_slack_norm": horizon_slack,
        }
        for name in (
            "future_configuration_reuse_value_norm",
            "configuration_opportunity_cost_norm",
        ):
            if name in edge_names:
                production_columns[name] = dense_features[
                    :, edge_names.index(name)
                ]
        relative_features = self._standardize_candidate_features(
            torch.stack(
                tuple(
                    production_columns[name]
                    for name in self.production_relative_feature_names
                ),
                dim=-1,
            ),
            ~action_mask[:pair_count],
        )
        production_directions = relative_features.new_tensor(
            _relative_directions(self.production_relative_feature_names)
        )
        effective_weights = functional.softplus(
            self.production_relative_ranker.weight
        ) * production_directions
        relative_logits = functional.linear(
            relative_features, effective_weights
        ).reshape(-1)
        preference_logits = relative_logits.new_zeros(relative_logits.shape)
        if self.preference_action_score_enabled:
            preference_logits = self._direct_preference_logits(
                torch.stack(
                    (
                        processing + reconfiguration,
                        production_columns["total_reconfiguration_cost_norm"],
                    ),
                    dim=-1,
                ),
                ~action_mask[:pair_count],
                preference,
                scale=self.production_preference_action_scale(),
            )
        primary_logits = relative_logits + preference_logits
        common_context, raw_residual_context = self._candidate_context_components(
            contextual_logits,
            ~action_mask[:pair_count],
        )
        if self.candidate_context_mode == V7_BOUNDED_CONTEXT_MODE:
            raw_residual_context = contextual_logits - common_context
        residual_context = self._context_residual(
            primary_logits,
            raw_residual_context,
            ~action_mask[:pair_count],
            self.production_residual_context_gate,
        )
        final_logits = (
            primary_logits
            + torch.sigmoid(self.production_context_gate) * common_context
            + residual_context
        )
        if self.production_action_semantics == FLAT_PRODUCTION_ACTION_SEMANTICS:
            final_logits = final_logits + commit_set_logit
        self._record_policy_components(
            DecisionType.PRODUCTION,
            relative_logits,
            final_logits,
            action_mask,
            preference_logits=preference_logits,
            commit_set_logit=commit_set_logit,
            production_gate_logits=production_gate_logits,
            production_gate_base_logits=production_gate_base_logits,
            production_gate_logit_boost=production_gate_logit_boost,
            production_gate_state_scale=production_gate_state_scale,
        )
        return final_logits, commit_set_logit

    @staticmethod
    def _hierarchical_production_logits(
        pair_logits: torch.Tensor,
        commit_logit: torch.Tensor,
        defer_logit: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return flat joint log-probabilities for a two-stage production policy.

        The commit/defer gate is normalized independently of the conditional
        distribution over legal production pairs.  Returning the joint values
        as logits keeps the existing PPO action and rollout interfaces intact.
        """

        if pair_logits.ndim != 1:
            raise ValueError("production pair logits must be one-dimensional")
        if commit_logit.ndim != 0 or defer_logit.ndim != 0:
            raise ValueError("production gate logits must be scalar")
        if action_mask.shape != (pair_logits.shape[0] + 1,):
            raise ValueError("hierarchical production mask has the wrong shape")
        if action_mask.dtype != torch.bool:
            raise TypeError("hierarchical production mask must be boolean")
        if bool(action_mask.all()):
            raise ValueError("at least one hierarchical action must be feasible")

        minimum_logit = torch.finfo(pair_logits.dtype).min
        pair_legal = ~action_mask[:-1]
        commit_legal = pair_legal.any()
        defer_legal = ~action_mask[-1]
        gate_logits = torch.stack((commit_logit, defer_logit))
        gate_mask = torch.stack((~commit_legal, ~defer_legal))
        gate_log_probabilities = functional.log_softmax(
            gate_logits.masked_fill(gate_mask, minimum_logit),
            dim=0,
        )

        pair_joint = pair_logits.new_full(pair_logits.shape, minimum_logit)
        if bool(commit_legal):
            conditional_log_probabilities = functional.log_softmax(
                pair_logits.masked_fill(~pair_legal, minimum_logit),
                dim=0,
            )
            pair_joint = (
                conditional_log_probabilities + gate_log_probabilities[0]
            ).masked_fill(~pair_legal, minimum_logit)
        defer_joint = gate_log_probabilities[1].masked_fill(
            ~defer_legal,
            minimum_logit,
        )
        return torch.cat((pair_joint, defer_joint.reshape(1)))

    def _worker_logits(
        self,
        observation: HeterogeneousGraphObservation,
        operation_embeddings: torch.Tensor,
        machine_embeddings: torch.Tensor,
        worker_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        preference_embedding: torch.Tensor,
        preference: torch.Tensor,
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
        preference_pairs = preference_embedding[None, None, :].expand(
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
                    preference_pairs,
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
        stage_duration = dense_features[
            :, edge_names.index("stage_duration_norm")
        ]
        labor_cost = dense_features[
            :, edge_names.index("incremental_labor_cost_norm")
        ]
        downtime_cost = dense_features[
            :, edge_names.index("incremental_downtime_cost_norm")
        ]
        incremental_variance = dense_features[
            :, edge_names.index("incremental_load_variance_norm")
        ]
        worker_columns = {
            "stage_duration_norm": stage_duration,
            "projected_fatigue_ratio": projected_fatigue,
            "incremental_labor_cost_norm": labor_cost,
            "incremental_downtime_cost_norm": downtime_cost,
            "incremental_load_variance_norm": incremental_variance,
        }
        for name in (
            "fatigue_headroom_ratio",
            "total_incremental_cost_norm",
            "qualification_opportunity_cost_norm",
            "recovery_eta_norm",
            "remaining_service_capacity_norm",
        ):
            if name in edge_names:
                worker_columns[name] = dense_features[
                    :, edge_names.index(name)
                ]
        relative_features = self._standardize_candidate_features(
            torch.stack(
                tuple(
                    worker_columns[name]
                    for name in self.worker_relative_feature_names
                ),
                dim=-1,
            ),
            ~action_mask[:pair_count],
        )
        worker_directions = relative_features.new_tensor(
            _relative_directions(self.worker_relative_feature_names)
        )
        effective_weights = functional.softplus(
            self.worker_relative_ranker.weight
        ) * worker_directions
        relative_logits = functional.linear(
            relative_features, effective_weights
        ).reshape(-1)
        preference_logits = relative_logits.new_zeros(relative_logits.shape)
        direct_preference_components: torch.Tensor | None = None
        if (
            self.preference_action_score_enabled
            and self.preference_action_score_scope == "all"
        ):
            direct_preference_components = (
                self._direct_preference_logit_components(
                    torch.stack(
                        (
                            stage_duration,
                            labor_cost + downtime_cost,
                            incremental_variance,
                        ),
                        dim=-1,
                    ),
                    ~action_mask[:pair_count],
                    preference,
                    scale=self.worker_preference_action_scale(),
                )
            )
            preference_logits = direct_preference_components.sum(dim=-1)
        elif (
            self.preference_action_score_enabled
            and self.preference_action_score_scope
            == "production_plus_safe_worker_variance"
        ):
            direct_preference_components = (
                self._safe_worker_variance_preference_logit_components(
                    incremental_variance,
                    ~action_mask[:pair_count],
                    preference,
                )
            )
            preference_logits = direct_preference_components.sum(dim=-1)
        primary_logits = relative_logits + preference_logits
        common_context, raw_residual_context = self._candidate_context_components(
            contextual_logits,
            ~action_mask[:pair_count],
        )
        if self.candidate_context_mode == V7_BOUNDED_CONTEXT_MODE:
            raw_residual_context = contextual_logits - common_context
        residual_context = self._context_residual(
            primary_logits,
            raw_residual_context,
            ~action_mask[:pair_count],
            self.worker_residual_context_gate,
        )
        common_term_enabled = (
            self.policy_head_version == 6
            or (
                self.worker_common_context_enabled
                and bool(~action_mask[-1])
            )
        )
        common_term = (
            torch.sigmoid(self.worker_context_gate) * common_context
            if common_term_enabled
            else torch.zeros_like(common_context)
        )
        final_logits = (
            primary_logits
            + common_term
            + residual_context
        )
        self._record_policy_components(
            DecisionType.WORKER,
            relative_logits,
            final_logits,
            action_mask,
            preference_logits=preference_logits,
            direct_preference_components=direct_preference_components,
        )
        return final_logits

    @staticmethod
    def _initialize_monotone_ranker(
        ranker: nn.Linear,
        effective_magnitudes: Sequence[float],
    ) -> None:
        magnitudes = torch.as_tensor(
            tuple(float(value) for value in effective_magnitudes),
            dtype=ranker.weight.dtype,
        )
        if magnitudes.numel() != ranker.in_features:
            raise ValueError("ranker initialization width does not match inputs")
        if bool(torch.any(~torch.isfinite(magnitudes))) or bool(
            torch.any(magnitudes <= 0.0)
        ):
            raise ValueError("ranker initial magnitudes must be finite and positive")
        with torch.no_grad():
            ranker.weight.copy_(torch.log(torch.expm1(magnitudes)).reshape(1, -1))

    @staticmethod
    def _initialize_zero_context_output(scorer: nn.Sequential) -> None:
        output = scorer[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("candidate context scorer must end in nn.Linear")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    @staticmethod
    def _initialize_bounded_state_scale_output(
        scorer: nn.Sequential,
        *,
        initial_scale: float,
        maximum_scale: float,
    ) -> None:
        """Initialize a state-scale head to the preregistered constant value."""

        output = scorer[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("state-scale scorer must end in nn.Linear")
        probability = float(initial_scale) / float(maximum_scale)
        if not 0.0 < probability < 1.0:
            raise ValueError("state-scale initialization must be strictly bounded")
        with torch.no_grad():
            nn.init.zeros_(output.weight)
            output.bias.fill_(math.log(probability / (1.0 - probability)))

    @staticmethod
    def _candidate_context_components(
        contextual_logits: torch.Tensor,
        feasible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(torch.any(feasible)):
            common = contextual_logits[feasible].mean()
        else:
            common = contextual_logits.new_zeros(())
        residual = HeteroGraphActorCritic._standardize_candidate_features(
            contextual_logits.unsqueeze(-1), feasible
        ).squeeze(-1)
        return common.expand_as(contextual_logits), residual

    def _context_residual(
        self,
        relative_logits: torch.Tensor,
        raw_residual: torch.Tensor,
        feasible: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if self.candidate_context_mode == V6_CANDIDATE_CONTEXT_MODE:
            return torch.sigmoid(gate) * raw_residual
        if self.candidate_context_mode != V7_BOUNDED_CONTEXT_MODE:
            raise RuntimeError("unknown candidate context mode")
        selected = relative_logits[feasible]
        ranker_scale = (
            selected.std(unbiased=False)
            if selected.numel() >= 2
            else relative_logits.new_zeros(())
        ).clamp_min(1e-3)
        return (
            torch.sigmoid(gate)
            * self.residual_scale_ratio
            * ranker_scale
            * torch.tanh(raw_residual)
        )

    def _record_policy_components(
        self,
        decision_type: DecisionType,
        relative_logits: torch.Tensor,
        final_pair_logits: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        preference_logits: torch.Tensor | None = None,
        commit_set_logit: torch.Tensor | None = None,
        production_gate_logits: torch.Tensor | None = None,
        production_gate_base_logits: torch.Tensor | None = None,
        production_gate_logit_boost: torch.Tensor | None = None,
        production_gate_state_scale: torch.Tensor | None = None,
        direct_preference_components: torch.Tensor | None = None,
    ) -> None:
        feasible = ~action_mask[:-1]
        feasible_indices = torch.nonzero(feasible, as_tuple=False).flatten()
        if feasible_indices.numel():
            relative_top = int(
                feasible_indices[
                    torch.argmax(relative_logits[feasible_indices])
                ].detach().cpu()
            )
            final_top = int(
                feasible_indices[
                    torch.argmax(final_pair_logits[feasible_indices])
                ].detach().cpu()
            )
            preference_adjusted = relative_logits + (
                preference_logits
                if preference_logits is not None
                else torch.zeros_like(relative_logits)
            )
            preference_top = int(
                feasible_indices[
                    torch.argmax(preference_adjusted[feasible_indices])
                ].detach().cpu()
            )
        else:
            relative_top = -1
            final_top = -1
            preference_top = -1
        feasible_preference = (
            preference_logits[feasible_indices]
            if preference_logits is not None and feasible_indices.numel()
            else relative_logits.new_zeros((0,))
        )
        gate_commit_probability = 0.0
        gate_defer_probability = 0.0
        gate_logit_margin = 0.0
        gate_base_commit_probability = 0.0
        gate_base_defer_probability = 0.0
        gate_defer_to_commit_flip = False
        gate_logit_boost = 0.0
        counterfactual_eligible = False
        counterfactual_high_flow_flip = False
        counterfactual_low_flow_identity_violation = False
        counterfactual_monotonicity_violation = False
        counterfactual_high_flow_margin = 0.0
        counterfactual_state_scale = 0.0
        if production_gate_logits is not None:
            if production_gate_logits.shape != (2,):
                raise ValueError("production gate logits must have shape (2,)")
            gate_mask = torch.stack(
                (
                    ~feasible.any(),
                    action_mask[-1],
                )
            )
            minimum_logit = torch.finfo(production_gate_logits.dtype).min
            gate_probabilities = functional.softmax(
                production_gate_logits.masked_fill(gate_mask, minimum_logit),
                dim=0,
            )
            gate_commit_probability = float(
                gate_probabilities[0].detach().cpu()
            )
            gate_defer_probability = float(
                gate_probabilities[1].detach().cpu()
            )
            gate_logit_margin = float(
                (production_gate_logits[0] - production_gate_logits[1])
                .detach()
                .cpu()
            )
            if production_gate_base_logits is not None:
                if production_gate_base_logits.shape != (2,):
                    raise ValueError("base production gate logits must have shape (2,)")
                base_probabilities = functional.softmax(
                    production_gate_base_logits.masked_fill(gate_mask, minimum_logit),
                    dim=0,
                )
                gate_base_commit_probability = float(base_probabilities[0].detach().cpu())
                gate_base_defer_probability = float(base_probabilities[1].detach().cpu())
                gate_defer_to_commit_flip = bool(
                    torch.argmax(base_probabilities).detach().cpu() == 1
                    and torch.argmax(gate_probabilities).detach().cpu() == 0
                )
            if production_gate_logit_boost is not None:
                gate_logit_boost = float(production_gate_logit_boost.detach().cpu())
            if (
                self.production_gate_version
                == STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION
                and production_gate_base_logits is not None
            ):
                if production_gate_state_scale is None:
                    raise ValueError("E2.6 gate diagnostics require a state scale")
                base_margin = (
                    production_gate_base_logits[0]
                    - production_gate_base_logits[1]
                )
                low_gate_logits, _ = self._final_production_gate_logits(
                    production_gate_base_logits,
                    production_gate_base_logits.new_tensor((0.2, 0.4, 0.4)),
                    state_scale=production_gate_state_scale,
                )
                high_gate_logits, _ = self._final_production_gate_logits(
                    production_gate_base_logits,
                    production_gate_base_logits.new_tensor((1.0, 0.0, 0.0)),
                    state_scale=production_gate_state_scale,
                )
                low_margin = low_gate_logits[0] - low_gate_logits[1]
                high_margin = high_gate_logits[0] - high_gate_logits[1]
                counterfactual_state_scale = float(
                    production_gate_state_scale.detach().cpu()
                )
                counterfactual_high_flow_margin = float(
                    high_margin.detach().cpu()
                )
                counterfactual_eligible = bool(
                    feasible.any().detach().cpu()
                    and (~action_mask[-1]).detach().cpu()
                    and (base_margin < 0.0).detach().cpu()
                )
                counterfactual_high_flow_flip = bool(
                    counterfactual_eligible
                    and (high_margin >= 0.0).detach().cpu()
                )
                counterfactual_low_flow_identity_violation = bool(
                    (low_margin - base_margin).abs().gt(1e-7).detach().cpu()
                )
                counterfactual_monotonicity_violation = bool(
                    (high_margin < low_margin - 1e-7).detach().cpu()
                )

        direct_component_max_abs = [0.0, 0.0, 0.0]
        if direct_preference_components is not None:
            if direct_preference_components.ndim != 2 or (
                direct_preference_components.shape[0]
                != relative_logits.shape[0]
            ):
                raise ValueError("direct preference components have wrong shape")
            if direct_preference_components.shape[1] > 3:
                raise ValueError("direct preference components have too many columns")
            if feasible_indices.numel():
                values = direct_preference_components[feasible_indices]
                for index in range(values.shape[1]):
                    direct_component_max_abs[index] = float(
                        values[:, index].abs().max().detach().cpu()
                    )
        self._latest_policy_decision_diagnostics.append(
            {
                "decision_type": decision_type.value,
                "action_count": int(action_mask.shape[0]),
                "legal_pair_count": int(feasible_indices.numel()),
                "terminal_legal": bool((~action_mask[-1]).detach().cpu()),
                "relative_top_action": relative_top,
                "preference_top_action": preference_top,
                "final_pair_top_action": final_top,
                "preference_overrode_relative_top": (
                    relative_top != preference_top
                ),
                "context_overrode_top": preference_top != final_top,
                "preference_logit_std": (
                    float(
                        feasible_preference.std(unbiased=False).detach().cpu()
                    )
                    if feasible_preference.numel() >= 2
                    else 0.0
                ),
                "commit_set_logit": (
                    float(commit_set_logit.detach().cpu())
                    if commit_set_logit is not None
                    else 0.0
                ),
                "production_gate_state_count": int(
                    decision_type == DecisionType.PRODUCTION
                    and production_gate_logits is not None
                ),
                "production_gate_commit_probability": (
                    gate_commit_probability
                ),
                "production_gate_defer_probability": gate_defer_probability,
                "production_gate_logit_margin": gate_logit_margin,
                "production_gate_base_commit_probability": gate_base_commit_probability,
                "production_gate_base_defer_probability": gate_base_defer_probability,
                "production_gate_commit_logit_boost": gate_logit_boost,
                "production_gate_residual_active": (
                    decision_type == DecisionType.PRODUCTION and gate_logit_boost > 0.0
                ),
                "production_gate_base_defer_to_final_commit_flip": (
                    decision_type == DecisionType.PRODUCTION and gate_defer_to_commit_flip
                ),
                "counterfactual_eligible_state": int(counterfactual_eligible),
                "counterfactual_high_flow_commit_flip": int(
                    counterfactual_high_flow_flip
                ),
                "counterfactual_high_flow_margin": counterfactual_high_flow_margin,
                "counterfactual_state_residual_scale": counterfactual_state_scale,
                "counterfactual_low_flow_identity_violation": int(
                    counterfactual_low_flow_identity_violation
                ),
                "counterfactual_monotonicity_violation": int(
                    counterfactual_monotonicity_violation
                ),
                "production_conditional_preference_overrode_relative_top": (
                    decision_type == DecisionType.PRODUCTION
                    and relative_top != preference_top
                ),
                "worker_variance_preference_overrode_relative_top": (
                    decision_type == DecisionType.WORKER
                    and relative_top != preference_top
                    and direct_component_max_abs[2] > 0.0
                ),
                "worker_direct_preference_flow_logit_max_abs": (
                    direct_component_max_abs[0]
                    if decision_type == DecisionType.WORKER
                    else 0.0
                ),
                "worker_direct_preference_cost_logit_max_abs": (
                    direct_component_max_abs[1]
                    if decision_type == DecisionType.WORKER
                    else 0.0
                ),
                "worker_direct_preference_variance_logit_max_abs": (
                    direct_component_max_abs[2]
                    if decision_type == DecisionType.WORKER
                    else 0.0
                ),
            }
        )

    def consume_policy_decision_diagnostics(self) -> list[dict[str, Any]]:
        values = list(self._latest_policy_decision_diagnostics)
        self._latest_policy_decision_diagnostics.clear()
        return values

    def effective_relative_cost_weights(
        self,
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        if self.production_relative_ranker is not None:
            values = (
                functional.softplus(
                    self.production_relative_ranker.weight.detach()
                ).reshape(-1)
                * self.production_relative_ranker.weight.new_tensor(
                    _relative_directions(
                        self.production_relative_feature_names
                    )
                )
            ).cpu()
            result["production"] = dict(
                zip(
                    self.production_relative_feature_names,
                    (float(value) for value in values),
                )
            )
        if self.worker_relative_ranker is not None:
            values = (
                functional.softplus(
                    self.worker_relative_ranker.weight.detach()
                ).reshape(-1)
                * self.worker_relative_ranker.weight.new_tensor(
                    _relative_directions(self.worker_relative_feature_names)
                )
            ).cpu()
            result["worker"] = dict(
                zip(
                    self.worker_relative_feature_names,
                    (float(value) for value in values),
                )
            )
        return result

    def policy_head_diagnostics(self) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        for phase, weights in self.effective_relative_cost_weights().items():
            for name, value in weights.items():
                diagnostics[f"policy_head_weight_{phase}_{name}"] = float(value)
        gates = (
            ("production_common", self.production_context_gate),
            ("production_residual", self.production_residual_context_gate),
            ("worker_common", self.worker_context_gate),
            ("worker_residual", self.worker_residual_context_gate),
        )
        for name, parameter in gates:
            if parameter is not None:
                diagnostics[f"policy_head_gate_{name}"] = float(
                    torch.sigmoid(parameter.detach()).cpu()
                )
        if self.preference_action_score_enabled:
            diagnostics["policy_head_preference_action_scale"] = float(
                self.preference_action_scale().detach().cpu()
            )
            diagnostics["policy_head_preference_action_scale_production"] = float(
                self.production_preference_action_scale().detach().cpu()
            )
            diagnostics["policy_head_preference_action_scale_worker"] = float(
                self.worker_preference_action_scale().detach().cpu()
            )
        return diagnostics

    @staticmethod
    def _bounded_scale_raw(spec: Mapping[str, Any]) -> torch.Tensor:
        initial = float(spec["initial_scale"])
        minimum = float(spec["minimum_scale"])
        maximum = float(spec["maximum_scale"])
        probability = (initial - minimum) / (maximum - minimum)
        return torch.tensor(math.log(probability / (1.0 - probability)))

    def preference_action_scale(self) -> torch.Tensor:
        if self.preference_action_scale_raw is None:
            return next(self.parameters()).new_zeros(())
        return (
            self.preference_action_scale_raw.new_tensor(
                self.preference_action_score_minimum_scale
            )
            + functional.softplus(self.preference_action_scale_raw)
        )

    def _bounded_preference_action_scale(
        self,
        raw: torch.Tensor | None,
        spec: Mapping[str, Any],
    ) -> torch.Tensor:
        if raw is None:
            return self.preference_action_scale()
        return raw.new_tensor(float(spec["minimum_scale"])) + (
            raw.new_tensor(float(spec["maximum_scale"]))
            - raw.new_tensor(float(spec["minimum_scale"]))
        ) * torch.sigmoid(raw)

    def production_preference_action_scale(self) -> torch.Tensor:
        if self.preference_action_score_shared_scale:
            return self.preference_action_scale()
        return self._bounded_preference_action_scale(
            self.production_preference_action_scale_raw,
            self.preference_action_score_production_scale,
        )

    def worker_preference_action_scale(self) -> torch.Tensor:
        if self.preference_action_score_shared_scale:
            return self.preference_action_scale()
        return self._bounded_preference_action_scale(
            self.worker_preference_action_scale_raw,
            self.preference_action_score_worker_scale,
        )

    def _direct_preference_logits(
        self,
        objectives: torch.Tensor,
        feasible: torch.Tensor,
        preference: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._direct_preference_logit_components(
            objectives,
            feasible,
            preference,
            scale=scale,
        ).sum(dim=-1)

    def _direct_preference_logit_components(
        self,
        objectives: torch.Tensor,
        feasible: torch.Tensor,
        preference: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if objectives.ndim != 2 or objectives.shape[1] not in {2, 3}:
            raise ValueError(
                "direct preference objectives must have two or three columns"
            )
        if preference.shape != (3,):
            raise ValueError("direct preference weights must have shape (3,)")
        standardized = self._standardize_candidate_features(
            objectives, feasible
        )
        weights = preference[: objectives.shape[1]]
        multiplier = self.preference_action_scale() if scale is None else scale
        return multiplier * -(standardized * weights)

    def _safe_worker_variance_preference_logit_components(
        self,
        incremental_variance: torch.Tensor,
        feasible: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        """Return E2.4's variance-only direct term on an already safe mask."""

        if incremental_variance.ndim != 1:
            raise ValueError("worker incremental variance must be one-dimensional")
        if feasible.shape != incremental_variance.shape:
            raise ValueError("worker variance feasibility mask has wrong shape")
        if preference.shape != (3,):
            raise ValueError("direct preference weights must have shape (3,)")
        standardized_variance = self._standardize_candidate_features(
            incremental_variance.reshape(-1, 1), feasible
        ).reshape(-1)
        result = incremental_variance.new_zeros(
            (incremental_variance.shape[0], 3)
        )
        result[:, 2] = (
            -self.worker_preference_action_scale()
            * preference[2]
            * standardized_variance
        )
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
    policy_head = _validate_policy_head_config(network_config)
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
        policy_head["production_relative_initial_weights"],
        policy_head["worker_relative_initial_weights"],
        policy_head["common_context_gate_initial_logit"],
        policy_head["residual_context_gate_initial_logit"],
        policy_head["policy_head_version"],
        policy_head["production_action_semantics"],
        policy_head["production_relative_feature_names"],
        policy_head["worker_relative_feature_names"],
        policy_head["candidate_context_mode"],
        policy_head["production_commit_set_scorer"],
        policy_head["production_gate_version"],
        policy_head["production_gate_preference_conditioning"],
        policy_head["production_gate_tie_break"],
        policy_head["production_gate_freeze_base_after_feasibility"],
        policy_head["production_gate_flow_commit_residual"],
        policy_head["future_value_features"],
        policy_head["worker_common_context_enabled"],
        policy_head["residual_scale_ratio"],
        config.get("preference_conditioning", "none"),
        policy_head["preference_action_score_enabled"],
        policy_head["preference_action_score_version"],
        policy_head["preference_action_score_shared_scale"],
        policy_head["preference_action_score_initial_scale"],
        policy_head["preference_action_score_minimum_scale"],
        policy_head["preference_action_score_production_scale"],
        policy_head["preference_action_score_worker_scale"],
        policy_head["preference_action_score_standardization"],
        policy_head["preference_action_score_scope"],
    )
