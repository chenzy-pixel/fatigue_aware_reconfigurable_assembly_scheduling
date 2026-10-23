from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from environment.observation_schema import OBSERVATION_SCHEMA_VERSION, configured_global_objective_encoding
from environment.time_context import TIME_CONTEXT_VERSION
from .network_contract import message_identity, validate_message_identity
from .evaluation_sampling import BEST_OF_THREE, sample_aggregation, validate_sample_aggregation
from environment.types import flow_mode, flow_reward_version, FLOW_EXCESS
from environment.workload import objective_names, observation_schema_version, workload_config


_RUNTIME_MANIFEST: dict[str, Any] = {
    "core_profile": "v8_preference_conditioned_multiobjective",
    "production_action": "pair_plus_wait_v1",
    "worker_action": "pair_plus_wait_v1",
    "policy_head": 8,
    "candidate_ranker": "shared_preference_mlp_v1",
    "worker_feasibility": "instant_physical_pair_mask_v1",
    "wait_mask": "batch_progress_wait_v3",
    "observation_schema": OBSERVATION_SCHEMA_VERSION,
    "time_context": TIME_CONTEXT_VERSION,
    "reward": "single_stage_progress_quality_workload_failure_v6",
    "terminal_failure_penalty": 2.0,
    "training_protocol": "single_stage_lexicographic_workload_failure_v6",
}

_REMOVED_NETWORK_FIELDS = frozenset(
    {
        "encoder_type",
        "production_action_edge_features",
        "worker_action_edge_features",
        "production_candidate_relative_features",
        "worker_candidate_relative_features",
        "production_action_semantics",
        "worker_action_semantics",
        "production_relative_feature_names",
        "worker_relative_feature_names",
        "relative_weight_parameterization",
        "candidate_context_mode",
        "worker_relative_weight_sharing",
        "production_commit_set_scorer",
        "future_value_features",
        "preference_" + "conditioning",
        "preference_action_score",
        "production_gate",
        "centered_preference_adapter",
    }
)


def runtime_manifest(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return implementation identity and the configured task-failure penalty."""

    manifest = deepcopy(_RUNTIME_MANIFEST)
    manifest.update(data_schema=3,physics="batch_event_v1",distribution="factorial27_v1")
    manifest.update(message_identity())
    if config is not None:
        manifest["observation_schema"] = observation_schema_version(config)
        manifest["global_objective_encoding"] = configured_global_objective_encoding(config, manifest["observation_schema"])
        manifest["flow_mode"] = flow_mode(dict(config))
        manifest["reward"] = flow_reward_version(dict(config))
        manifest["normalization_manifest_sha256"] = config.get("objective_scalarizer", {}).get("normalization_manifest_sha256")
        manifest["workload_objective"] = workload_config(config)
        manifest["objective_names"] = list(objective_names(config))
        scales = config["objective_scalarizer"]["scales"]
        manifest["objective_scales"] = [float(scales[name]) for name in objective_names(config)]
        manifest["training_protocol"] = (
            "single_stage_lexicographic_workload_failure_v6"
            if manifest["flow_mode"] == FLOW_EXCESS else "single_stage_lexicographic_workload_failure_v5"
        )
        network = config.get("network", {})
        manifest["encoder_variant"] = network.get("encoder_variant", "hetero_gnn")
        manifest.update(validate_message_identity(network))
        manifest["actor_head_variant"] = network.get("actor_head_variant", "shared_preference")
        # Full defaults retain the historical manifest; explicit interventions
        # carry their identity in both the manifest and network checkpoint spec.
        for key in ("direct_features_enabled", "residual_enabled"):
            if key in network:
                if type(network[key]) is not bool:
                    raise ValueError(f"network.{key} must be a boolean")
                manifest[key] = network[key]
        manifest["fatigue_mode"] = config.get("environment", {}).get("fatigue_mode", "full")
        if manifest["fatigue_mode"] == "neutral":
            manifest["worker_feasibility"] = "qualification_and_availability_no_fatigue_v1"
        manifest["terminal_failure_penalty"] = float(
            config.get("reward", {}).get(
                "terminal_failure_penalty", manifest["terminal_failure_penalty"]
            )
        )
        if sample_aggregation(config) == BEST_OF_THREE:
            manifest["training_protocol"] += "_best_of_3_v1"
            manifest["sample_aggregation"] = BEST_OF_THREE
            manifest["sampling_budget"] = 3
        comparison = config.get("comparison")
        if comparison is not None:
            manifest["comparison"] = {key: comparison[key] for key in
                ("version", "method", "delivery_protocol", "training_steps", "training_budget") if key in comparison}
            method = comparison.get("method")
            if method in {"dcan", "rule_ddqn"}:
                manifest["encoder_variant"] = "dcan_assembly_v1" if method == "dcan" else "pooled_rule_state_v1"
                manifest["actor_head_variant"] = "learned_phase_candidate_v1" if method == "dcan" else "linear_rule_q_v1"
                manifest["message_function"] = "conditional_dual_and_service_attention_v1" if method == "dcan" else "not_applicable"
                manifest["message_aggregation"] = "preference_conditioned_attention_v1" if method == "dcan" else "mean_max_pooling_v1"
    return manifest


def validate_latest_only_config(config: Mapping[str, Any]) -> None:
    """Reject removed implementation selectors before a run can start."""

    if "runtime_manifest" in config:
        raise ValueError("runtime_manifest is generated by the repository")
    expected_reward = flow_reward_version(dict(config))
    if str(config.get("experiment_suite_version", "")) != expected_reward:
        raise ValueError(
            "experiment_suite_version must match the Flow reward contract"
        )
    network = config.get("network", {})
    if not isinstance(network, Mapping):
        raise TypeError("network config must be a mapping")
    removed = sorted(_REMOVED_NETWORK_FIELDS.intersection(network))
    if removed:
        raise ValueError(
            "latest-only configuration cannot select network implementations: "
            + ", ".join(removed)
        )
    if int(network.get("policy_head_version", 8)) != 8:
        raise ValueError("latest-only runtime accepts only policy_head_version=8")
    selected_schema = observation_schema_version(config)
    configured_global_objective_encoding(config, selected_schema)
    if int(network.get("observation_schema_version", selected_schema)) != selected_schema:
        raise ValueError(f"schema {OBSERVATION_SCHEMA_VERSION} requires retraining; network observation schema is incompatible")
    workload_config(config)
    if float(network.get("dropout", 0.0)) != 0.0:
        raise ValueError("PPO requires network.dropout = 0")
    if network.get("encoder_variant", "hetero_gnn") not in {"hetero_gnn", "node_mlp_pool"}:
        raise ValueError("unknown network.encoder_variant")
    validate_message_identity(network)
    if network.get("actor_head_variant", "shared_preference") != "shared_preference":
        raise ValueError("unknown network.actor_head_variant")
    scalarizer = config.get("objective_scalarizer", {})
    if scalarizer.get("type") != "normalized_augmented_tchebycheff_v1":
        raise ValueError("objective_scalarizer.type must be normalized_augmented_tchebycheff_v1")
    if scalarizer.get("scale_source") != "frozen_manifest":
        raise ValueError("objective_scalarizer.scale_source must be frozen_manifest")
    reward = config.get("reward", {})
    if not isinstance(reward, Mapping):
        raise TypeError("reward config must be a mapping")
    if "quality_weights" in reward:
        raise ValueError(
            "V8 objective preference must be configured only through "
            "preference.quality; reward.quality_weights is not accepted"
        )
    if str(reward.get("mode", "")) != expected_reward:
        raise ValueError(
            "runtime reward.mode must match flow_mode"
        )
    try:
        failure_penalty = float(reward.get("terminal_failure_penalty", -1.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "reward.terminal_failure_penalty must be finite and non-negative"
        ) from error
    if not math.isfinite(failure_penalty) or failure_penalty < 0.0:
        raise ValueError(
            "reward.terminal_failure_penalty must be finite and non-negative"
        )
    shaping = reward.get("feasibility_shaping", {})
    if not isinstance(shaping, Mapping):
        raise TypeError("reward.feasibility_shaping must be a mapping")
    if bool(shaping.get("enabled", False)):
        raise ValueError("single-stage workload reward requires feasibility shaping disabled")
    ppo = config.get("ppo", {})
    if not isinstance(ppo, Mapping):
        raise TypeError("ppo config must be a mapping")
    if float(ppo.get("gamma", 1.0)) != 1.0:
        raise ValueError("single-stage telescoping reward requires ppo.gamma = 1.0")
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise TypeError("training config must be a mapping")
    validate_sample_aggregation(config)
    if "two_stage" in training:
        raise ValueError("single-stage runtime rejects training.two_stage")
    environment = config.get("environment", {})
    if not isinstance(environment, Mapping):
        raise TypeError("environment config must be a mapping")
    for name in ("max_decisions", "max_zero_time_actions"):
        value = environment.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"environment.{name} must be a positive integer")
    if environment.get("fatigue_mode", "full") not in {"full", "neutral"}:
        raise ValueError("unknown environment.fatigue_mode")
    removed_environment = sorted(
        {"worker_resource_control", "production_defer"}.intersection(
            environment
        )
    )
    if removed_environment:
        raise ValueError(
            "latest-only configuration rejects removed environment controls: "
            + ", ".join(removed_environment)
        )


def attach_runtime_manifest(config: dict[str, Any]) -> dict[str, Any]:
    validate_latest_only_config(config)
    config.setdefault("network", {})["global_objective_encoding"] = configured_global_objective_encoding(config, 13)
    config["runtime_manifest"] = runtime_manifest(config)
    return config


def assert_checkpoint_fatigue_mode(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Reject cross-physics checkpoint reuse before evaluation or initialization."""
    saved = metadata.get("runtime_manifest", {})
    saved_mode = saved.get("fatigue_mode", "full") if isinstance(saved, Mapping) else "full"
    configured_mode = config.get("environment", {}).get("fatigue_mode", "full")
    if saved_mode != configured_mode:
        raise ValueError(f"checkpoint fatigue mode {saved_mode!r} does not match {configured_mode!r}")
