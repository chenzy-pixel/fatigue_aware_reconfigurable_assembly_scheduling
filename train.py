from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent.ppo import PPOAgent, RolloutBuffer, build_actor_critic
from agent.ppo.parallel import (
    EpisodeRollout,
    ParallelEpisodeRunner,
    TrainingRolloutBatch,
    forced_action_from_mask,
    training_base_instance_count,
    training_preference_group,
)
from configs import load_config, project_path
from configs.config import public_config
from data.dataset import (
    GeneratedInstanceRecord,
    OnlineInstanceDataset,
    load_dataset_split,
    validate_algorithm_seed,
)
from data.models import load_instance_yaml
from environment import (
    AssemblySchedulingEnv,
    CANONICAL_PREFERENCE,
    PreferenceVector,
    derive_episode_action_seed,
    preference_enabled,
    proxy_return_from_metrics,
    sample_episode_preference,
    simplex_lattice,
)
from eval import (
    evaluate_dataset,
    evaluate_dataset_parallel,
    evaluate_representative_diagnostic,
    load_configured_instance,
)
from result import (
    aggregate_evaluation_rows,
    aggregate_matching_recovery_diagnostics,
    aggregate_preference_diagnostics,
    build_provenance,
    create_run_directory,
    evaluation_selection_key,
    result_schema_version,
)
from result.io import write_config, write_csv, write_json
from result.metrics import (
    MATCHING_RECOVERY_DIAGNOSTIC_FIELDS,
    PREFERENCE_POLICY_DIAGNOSTIC_FIELDS,
)
from result.visdom_dashboard import (
    create_training_dashboard,
    override_visdom_enabled,
    resolve_visdom_settings,
)
from pareto_analysis import hypervolume_3d, nondominated_indices, normalize_objectives
from utils import set_seed


PARETO_PROMOTION_MODES = frozenset(
    {
        "pareto_guarded_e2_v1",
        "pareto_guarded_e2_3_v1",
        "pareto_guarded_e2_4_v1",
        "pareto_guarded_e2_5_v1",
        "pareto_guarded_e2_6_v1",
        "pareto_guarded_e2_7_development_v1",
    }
)
SINGLE_OBJECTIVE_PROMOTION_MODE = "single_objective_guarded_v1"
SINGLE_OBJECTIVE_METRICS = {
    "flow": "flow_time_objective",
    "cost": "reconfiguration_cost",
    "variance": "worker_load_variance",
}
POST_FEASIBILITY_RESIDUAL_GATE_VERSIONS = frozenset(
    {
        "state_only_monotone_flow_commit_gate_v2",
        "state_only_counterfactual_monotone_flow_commit_gate_v3",
    }
)


def _checkpoint_eligible_validation_event(event: str, promotion: str) -> bool:
    if promotion == SINGLE_OBJECTIVE_PROMOTION_MODE:
        return event == "accepted"
    return bool(
        event in {"promoted", "accepted"}
        or (
            event == "transition"
            and promotion
            not in {
                SINGLE_OBJECTIVE_PROMOTION_MODE,
                "pareto_guarded_e2_3_v1",
                "pareto_guarded_e2_4_v1",
                "pareto_guarded_e2_5_v1",
                "pareto_guarded_e2_6_v1",
                "pareto_guarded_e2_7_development_v1",
            }
        )
    )


def _validation_manifest_path(config: dict) -> Path:
    split = str(config["training"]["validation_split"])
    return project_path(config["paths"]["manifests_root"]) / split / "manifest.json"


def _validate_single_objective_validation_protocol(
    config: dict, *, smoke: bool, validation_limit: int | None
) -> None:
    """Require the publication-sized, deterministic validation manifest for formal runs."""
    if smoke or str(
        config["training"]["two_stage"].get("quality_checkpoint_promotion", "")
    ).strip().lower() != SINGLE_OBJECTIVE_PROMOTION_MODE:
        return
    settings = config["training"]["two_stage"][
        "single_objective_promotion"
    ]
    audit_limit = int(settings["audit_instance_limit"])
    if validation_limit != 100 or audit_limit != 500:
        raise ValueError(
            "single-objective validation must use 100 daily instances and "
            "a 500-instance audit"
        )
    manifest_path = _validation_manifest_path(config)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"single-objective validation manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    if (
        int(manifest.get("instance_count", len(files))) != 500
        or not isinstance(files, list)
        or len(files) != 500
    ):
        raise ValueError(
            "single-objective validation requires a 500-instance manifest; "
            f"found {manifest.get('instance_count')}"
        )


def _checkpoint_protocol_metadata(config: dict) -> dict[str, object]:
    manifest = _validation_manifest_path(config)
    provenance = build_provenance(
        config,
        dataset_manifest_path=manifest if manifest.is_file() else None,
        checkpoint_metadata={
            "experiment_suite_version": config.get(
                "experiment_suite_version", "legacy"
            )
        },
    )
    return {
        "effective_config": public_config(config),
        "source_state_sha256": provenance["source_state_sha256"],
        "effective_config_sha256": provenance["effective_config_sha256"],
        "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
        "experiment_suite_version": config.get(
            "experiment_suite_version", "legacy"
        ),
        "algorithm_seed": int(config["seed"]),
        "result_schema_version": result_schema_version(config),
        "worker_resource_control": dict(
            config["environment"].get("worker_resource_control", {})
        ),
        "provenance": provenance,
    }


def _run_relative_checkpoint(path: Path | None, run_directory: Path) -> str | None:
    """Persist portable checkpoint references while retaining old absolute readers."""

    if path is None:
        return None
    try:
        return str(path.relative_to(run_directory))
    except ValueError:
        return str(path)


def resolve_summary_checkpoint(summary_path: str | Path, value: str | None) -> Path | None:
    """Resolve E2.5 run-relative paths and legacy absolute summary paths."""

    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(summary_path).parent / path


def _pareto_promotion_settings(config: dict) -> dict[str, float | int]:
    raw = config["training"]["two_stage"].get("pareto_promotion")
    if not isinstance(raw, dict):
        raise ValueError("two_stage.pareto_promotion must be an object")
    settings: dict[str, float | int] = {
        "anchor_validate_every_updates": int(
            raw.get("anchor_validate_every_updates", 5)
        ),
        "full_grid_validate_every_updates": int(
            raw.get("full_grid_validate_every_updates", 20)
        ),
        "minimum_hv_improvement": float(
            raw.get("minimum_hv_improvement", 1e-4)
        ),
        "canonical_relative_tolerance": float(
            raw.get("canonical_relative_tolerance", 0.01)
        ),
        "canonical_absolute_tolerance": float(
            raw.get("canonical_absolute_tolerance", 1e-6)
        ),
        "fatigue_absolute_tolerance": float(
            raw.get("fatigue_absolute_tolerance", 1e-9)
        ),
        "required_full_grid_instance_count": int(
            raw.get("required_full_grid_instance_count", 0)
        ),
        "required_full_grid_preference_count": int(
            raw.get("required_full_grid_preference_count", 0)
        ),
        "required_full_grid_candidate_count": int(
            raw.get("required_full_grid_candidate_count", 0)
        ),
        "minimum_mean_unique_action_trace_count": float(
            raw.get("minimum_mean_unique_action_trace_count", 0.0)
        ),
        "minimum_mean_unique_objective_count": float(
            raw.get("minimum_mean_unique_objective_count", 0.0)
        ),
        "minimum_mean_nondominated_count": float(
            raw.get("minimum_mean_nondominated_count", 0.0)
        ),
        "required_counterfactual_instance_coverage": int(
            raw.get("required_counterfactual_instance_coverage", 0)
        ),
        "minimum_counterfactual_high_flow_flip_rate": float(
            raw.get("minimum_counterfactual_high_flow_flip_rate", 0.0)
        ),
        "minimum_centered_gate_extreme_flip_rate": float(
            raw.get("minimum_centered_gate_extreme_flip_rate", 0.0)
        ),
        "maximum_canonical_heuristic_relative_gap": float(
            raw.get("maximum_canonical_heuristic_relative_gap", 1.0)
        ),
        "maximum_preference_response_spearman_flow": float(
            raw.get("maximum_preference_response_spearman_flow", 0.0)
        ),
        "maximum_preference_response_spearman_cost": float(
            raw.get("maximum_preference_response_spearman_cost", 0.0)
        ),
        "maximum_preference_response_spearman_variance": float(
            raw.get("maximum_preference_response_spearman_variance", 0.0)
        ),
        "safety_guard_consecutive_failures": int(
            raw.get("safety_guard_consecutive_failures", 1)
        ),
        "safety_guard_learning_rate_decay_factor": float(
            raw.get("safety_guard_learning_rate_decay_factor", 0.5)
        ),
        "safety_guard_minimum_learning_rate": float(
            raw.get("safety_guard_minimum_learning_rate", 1e-5)
        ),
    }
    for name in (
        "anchor_validate_every_updates",
        "full_grid_validate_every_updates",
    ):
        if int(settings[name]) < 1:
            raise ValueError(f"pareto_promotion.{name} must be positive")
    for name in (
        "required_full_grid_instance_count",
        "required_full_grid_preference_count",
        "required_full_grid_candidate_count",
        "required_counterfactual_instance_coverage",
    ):
        if int(settings[name]) < 0:
            raise ValueError(f"pareto_promotion.{name} must be non-negative")
    for name in (
        "minimum_hv_improvement",
        "canonical_relative_tolerance",
        "canonical_absolute_tolerance",
        "fatigue_absolute_tolerance",
        "minimum_mean_unique_action_trace_count",
        "minimum_mean_unique_objective_count",
        "minimum_mean_nondominated_count",
        "minimum_counterfactual_high_flow_flip_rate",
        "minimum_centered_gate_extreme_flip_rate",
    ):
        value = float(settings[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"pareto_promotion.{name} must be finite and non-negative"
            )
    for name in (
        "maximum_preference_response_spearman_flow",
        "maximum_preference_response_spearman_cost",
        "maximum_preference_response_spearman_variance",
    ):
        value = float(settings[name])
        if not math.isfinite(value) or value < -1.0 or value > 1.0:
            raise ValueError(
                f"pareto_promotion.{name} must be finite and in [-1, 1]"
            )
    heuristic_gap = float(
        settings["maximum_canonical_heuristic_relative_gap"]
    )
    if not math.isfinite(heuristic_gap) or not -1.0 <= heuristic_gap <= 1.0:
        raise ValueError(
            "pareto_promotion.maximum_canonical_heuristic_relative_gap must "
            "be finite and in [-1, 1]"
        )
    if int(settings["safety_guard_consecutive_failures"]) < 1:
        raise ValueError(
            "pareto_promotion.safety_guard_consecutive_failures must be positive"
        )
    for name in (
        "safety_guard_learning_rate_decay_factor",
        "safety_guard_minimum_learning_rate",
    ):
        value = float(settings[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"pareto_promotion.{name} must be finite and positive"
            )
    if float(settings["safety_guard_learning_rate_decay_factor"]) > 1.0:
        raise ValueError(
            "pareto_promotion.safety_guard_learning_rate_decay_factor must be <= 1"
        )
    return settings


def _single_objective_name(config: dict) -> str:
    weights = config.get("reward", {}).get("quality_weights")
    if not isinstance(weights, dict) or set(weights) != set(
        SINGLE_OBJECTIVE_METRICS
    ):
        raise ValueError(
            "single-objective promotion requires reward.quality_weights "
            "with exactly flow/cost/variance"
        )
    values = {name: float(weights[name]) for name in SINGLE_OBJECTIVE_METRICS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("single-objective quality weights must be finite")
    active = [
        name
        for name, value in values.items()
        if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
    ]
    inactive_are_zero = all(
        math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12)
        for name, value in values.items()
        if name not in active
    )
    if len(active) != 1 or not inactive_are_zero:
        raise ValueError(
            "single-objective promotion requires strictly one-hot quality weights: "
            "one value equal to 1 and the others equal to 0"
        )
    return active[0]


def _single_objective_hard_gate(validation: dict) -> dict[str, bool]:
    completion_pass = bool(
        math.isfinite(float(validation.get("completion_rate", math.nan)))
        and float(validation["completion_rate"]) >= 1.0 - 1e-12
    )
    truncation_pass = int(validation.get("truncated_count", 0)) == 0
    violation_pass = int(validation.get("schedule_violation_count", 0)) == 0
    physical_pass = bool(validation.get("physical_safety_pass", True))
    return {
        "completion": completion_pass,
        "truncation": truncation_pass,
        "violation": violation_pass,
        "physical_safety": physical_pass,
        "all": completion_pass and truncation_pass and violation_pass and physical_pass,
    }


TIERED_TRAINING_GATES_VERSION = "tiered_training_gates_v1"


def _tiered_training_gates(config: dict) -> dict[str, Any] | None:
    """Return the opt-in E2.4--E2.7 gate protocol, or ``None`` for legacy runs.

    This is intentionally versioned and opt-in: historical experiment configs
    retain their original stop/rollback semantics when the field is absent.
    """

    raw = config.get("training", {}).get("gate_policy")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("training.gate_policy must be an object")
    if raw.get("version") != TIERED_TRAINING_GATES_VERSION:
        raise ValueError(
            "training.gate_policy.version must be "
            f"{TIERED_TRAINING_GATES_VERSION!r}"
        )
    required = {
        "version",
        "canonical_identity_tolerance",
        "policy_safety_action",
        "stage_timeout_action",
        "candidate_ranking",
        "final_acceptance_rule",
        "primary_precedence",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            "training.gate_policy missing required fields: " + ", ".join(missing)
        )
    tolerance = float(raw["canonical_identity_tolerance"])
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "training.gate_policy.canonical_identity_tolerance must be finite "
            "and non-negative"
        )
    if raw["policy_safety_action"] != "rollback_continue":
        raise ValueError("tiered gate policy only supports rollback_continue")
    if raw["stage_timeout_action"] != "force_transition":
        raise ValueError("tiered gate policy only supports force_transition")
    if raw["candidate_ranking"] != "completion_hypervolume_update":
        raise ValueError(
            "tiered gate policy candidate_ranking must be "
            "completion_hypervolume_update"
        )
    if raw["final_acceptance_rule"] != "any_candidate_passes":
        raise ValueError(
            "tiered gate policy final_acceptance_rule must be "
            "any_candidate_passes"
        )
    precedence = raw["primary_precedence"]
    if precedence != ["best_safe", "last_safe"]:
        raise ValueError(
            "tiered gate policy primary_precedence must be "
            "['best_safe', 'last_safe']"
        )
    return {
        "version": TIERED_TRAINING_GATES_VERSION,
        "canonical_identity_tolerance": tolerance,
        "policy_safety_action": raw["policy_safety_action"],
        "stage_timeout_action": raw["stage_timeout_action"],
        "candidate_ranking": raw["candidate_ranking"],
        "final_acceptance_rule": raw["final_acceptance_rule"],
        "primary_precedence": tuple(precedence),
    }


def _uses_tiered_training_gates(config: dict) -> bool:
    return _tiered_training_gates(config) is not None


def _pareto_anchor_preferences(config: dict) -> tuple[PreferenceVector, ...]:
    grouping = training_preference_group(config)
    if grouping is not None:
        anchors = tuple(grouping["anchors"])
    else:
        sampler = config.get("preference", {}).get("sampler", {})
        raw_anchors = sampler.get("anchors") if isinstance(sampler, dict) else None
        if not isinstance(raw_anchors, list):
            raise ValueError(
                "Pareto promotion without grouped training requires "
                "preference.sampler.anchors"
            )
        anchors = tuple(
            PreferenceVector(*map(float, values))
            for values in raw_anchors
        )
    if len(anchors) != 5:
        raise ValueError("Pareto anchor validation requires exactly five anchors")
    return anchors


@dataclass
class E2_7PreferenceStageController:
    """Metric-gated E2.7 adapter curriculum with resumable state."""

    enabled: bool
    final_update: int = 0
    gate_minimum_updates: int = 0
    gate_maximum_end_update: int = 0
    production_pair_minimum_updates: int = 0
    production_pair_maximum_end_update: int = 0
    required_consecutive_passes: int = 0
    minimum_gate_flip_rate: float = 0.0
    minimum_production_pair_correct_rate: float = 0.0
    maximum_auxiliary_loss: float = 0.0
    stage: str = "gate"
    stage_update_count: int = 0
    consecutive_passes: int = 0
    transition_history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict) -> "E2_7PreferenceStageController | None":
        raw = config["training"].get("preference_stage_schedule")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise TypeError("training.preference_stage_schedule must be an object")
        version = str(raw.get("version", ""))
        legacy_required = {
            "enabled",
            "version",
            "final_update",
            "gate_minimum_updates",
            "gate_maximum_end_update",
            "production_pair_minimum_updates",
            "production_pair_maximum_end_update",
            "required_consecutive_passes",
            "minimum_gate_flip_rate",
            "minimum_production_pair_correct_rate",
        }
        monitored_required = legacy_required | {"maximum_auxiliary_loss"}
        required = (
            monitored_required
            if version == "e1_centered_adapter_monitored_v4"
            else legacy_required
        )
        if set(raw) != required:
            raise ValueError(
                "training.preference_stage_schedule must contain exactly "
                f"{sorted(required)}"
            )
        if not bool(raw["enabled"]):
            return None
        if version not in {
            "e1_centered_adapter_metric_gated_v2",
            "e1_centered_adapter_metric_gated_v3",
            "e1_centered_adapter_monitored_v4",
        }:
            raise ValueError(
                "training.preference_stage_schedule.version must be "
                "'e1_centered_adapter_metric_gated_v2' or "
                "'e1_centered_adapter_metric_gated_v3' or "
                "'e1_centered_adapter_monitored_v4'"
            )
        values = {
            name: int(raw[name])
            for name in (
                "final_update",
                "gate_minimum_updates",
                "gate_maximum_end_update",
                "production_pair_minimum_updates",
                "production_pair_maximum_end_update",
                "required_consecutive_passes",
            )
        }
        local_stage_budgets = version == "e1_centered_adapter_monitored_v4"
        valid_bounds = (
            1 <= values["gate_minimum_updates"] <= values["gate_maximum_end_update"]
            and 1 <= values["production_pair_minimum_updates"]
            <= values["production_pair_maximum_end_update"]
            and 1 <= values["required_consecutive_passes"]
        )
        if local_stage_budgets:
            valid_bounds = valid_bounds and (
                values["gate_maximum_end_update"]
                + values["production_pair_maximum_end_update"]
                <= values["final_update"]
            )
        else:
            valid_bounds = valid_bounds and (
                values["gate_maximum_end_update"]
                < values["production_pair_maximum_end_update"]
                <= values["final_update"]
            )
        if not valid_bounds:
            raise ValueError("E2.7 metric-gated stage bounds are invalid")
        rates = {
            name: float(raw[name])
            for name in (
                "minimum_gate_flip_rate",
                "minimum_production_pair_correct_rate",
            )
        }
        if not all(math.isfinite(value) and 0.0 < value <= 1.0 for value in rates.values()):
            raise ValueError("E2.7 metric-gated stage rates must be within (0, 1]")
        counterfactual = config["ppo"].get(
            "counterfactual_preference_consistency", {}
        )
        maximum_auxiliary_loss = float(raw.get("maximum_auxiliary_loss", 0.0))
        if local_stage_budgets and (
            not math.isfinite(maximum_auxiliary_loss)
            or maximum_auxiliary_loss < 0.0
        ):
            raise ValueError(
                "E2.7 monitored v4 maximum_auxiliary_loss must be finite and non-negative"
            )
        if version in {
            "e1_centered_adapter_metric_gated_v3",
            "e1_centered_adapter_monitored_v4",
        }:
            if not isinstance(counterfactual, dict) or (
                counterfactual.get("version") != "centered_gate_pair_worker_v4"
            ):
                raise ValueError(
                    "E2.7 v3 stage schedule requires centered counterfactual v4"
                )
            gate = counterfactual.get("gate", {})
            if not isinstance(gate, dict) or not math.isclose(
                float(gate.get("minimum_extreme_flip_rate", 0.0)),
                rates["minimum_gate_flip_rate"],
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "E2.7 gate loss and stage gate flip thresholds must match"
                )
        controller = cls(
            enabled=True,
            **values,
            **rates,
            maximum_auxiliary_loss=maximum_auxiliary_loss,
        )
        controller._protocol_version = version
        return controller

    def apply(self, agent: PPOAgent) -> str:
        setter = getattr(agent.network, "set_centered_preference_stage", None)
        if setter is None:
            raise RuntimeError("E2.7 stage schedule requires a centered preference adapter")
        setter(self.stage)
        return self.stage

    def _is_v3(self) -> bool:
        return getattr(self, "_protocol_version", "e1_centered_adapter_metric_gated_v2") == (
            "e1_centered_adapter_metric_gated_v3"
        )

    def _is_v4(self) -> bool:
        return getattr(self, "_protocol_version", "") == (
            "e1_centered_adapter_monitored_v4"
        )

    def _validate_control_pool(self, losses: dict[str, Any], *, stage: str) -> None:
        if stage == "gate":
            eligible = losses.get("counterfactual_eligible_count")
            evaluated = losses.get("counterfactual_control_evaluated_state_count")
        else:
            eligible = losses.get("production_pair_eligible_count")
            evaluated = losses.get("production_pair_control_evaluated_state_count")
        if not isinstance(eligible, int) or not isinstance(evaluated, int):
            raise RuntimeError(f"E2.7 {stage} control counts must be integers")
        if eligible < 1 or evaluated != eligible:
            raise RuntimeError(f"E2.7 {stage} control pool is incomplete")

    def _stage_passed(self, losses: dict[str, Any]) -> bool:
        if self.stage == "gate":
            if self._is_v3() or self._is_v4():
                self._validate_control_pool(losses, stage="gate")
                flow_cost_rate = float(
                    losses.get("counterfactual_flow_cost_flip_rate", 0.0)
                )
                flow_variance_rate = float(
                    losses.get("counterfactual_flow_variance_flip_rate", 0.0)
                )
            else:
                eligible = float(losses.get("counterfactual_eligible_count", 0.0))
                flow_cost_rate = (
                    float(losses.get("counterfactual_flow_cost_flip_count", 0.0))
                    / eligible if eligible else 0.0
                )
                flow_variance_rate = (
                    float(losses.get("counterfactual_flow_variance_flip_count", 0.0))
                    / eligible if eligible else 0.0
                )
            constraint_pass = (
                float(losses.get("counterfactual_gate_loss", math.inf))
                <= self.maximum_auxiliary_loss
                if self._is_v4()
                else losses.get("counterfactual_constraint_status")
                == "constraint_satisfied"
            )
            return bool(
                constraint_pass
                and flow_cost_rate >= self.minimum_gate_flip_rate
                and flow_variance_rate >= self.minimum_gate_flip_rate
            )
        elif self.stage == "production_pair":
            if self._is_v3() or self._is_v4():
                self._validate_control_pool(losses, stage="production_pair")
            constraint_pass = (
                float(losses.get("production_pair_loss", math.inf))
                <= self.maximum_auxiliary_loss
                if self._is_v4()
                else losses.get("production_pair_constraint_status")
                == "constraint_satisfied"
            )
            return bool(
                constraint_pass
                and float(losses.get("production_pair_correct_rate", 0.0))
                >= self.minimum_production_pair_correct_rate
            )
        return False

    def propose(self, losses: dict[str, Any], *, update_id: int) -> dict[str, Any]:
        """Prepare, but do not commit, one metric-gated curriculum update."""

        if self.stage == "gate":
            maximum_update = self.gate_maximum_end_update
            next_stage = "production_pair"
            minimum_updates = self.gate_minimum_updates
        elif self.stage == "production_pair":
            maximum_update = self.production_pair_maximum_end_update
            next_stage = "worker_variance"
            minimum_updates = self.production_pair_minimum_updates
        else:
            return {
                "stage": self.stage,
                "stage_update_count": self.stage_update_count,
                "passed": False,
                "transition_requested": False,
                "failure": None,
            }
        passed = self._stage_passed(losses)
        stage_update_count = self.stage_update_count + 1
        consecutive_passes = self.consecutive_passes + 1 if passed else 0
        transition_requested = bool(
            stage_update_count >= minimum_updates
            and consecutive_passes >= self.required_consecutive_passes
        )
        timeout = bool(stage_update_count >= maximum_update and not transition_requested)
        if self._is_v4() and stage_update_count >= maximum_update:
            transition_requested = True
        return {
            "stage": self.stage,
            "next_stage": next_stage,
            "update_id": int(update_id),
            "stage_update_count": stage_update_count,
            "consecutive_passes": consecutive_passes,
            "maximum_update": maximum_update,
            "passed": passed,
            "transition_requested": transition_requested,
            "transition_reason": (
                "budget_exhausted" if self._is_v4() and timeout else
                "metrics_passed" if transition_requested else None
            ),
            "failure": (
                "preference_stage_failed"
                if not self._is_v4()
                and int(update_id) >= maximum_update
                and not transition_requested
                else None
            ),
        }

    def commit(
        self,
        proposal: dict[str, Any],
        *,
        transition_confirmed: bool = True,
    ) -> dict[str, Any]:
        """Commit a validated proposal and return any structured stage failure."""

        if proposal.get("stage") != self.stage:
            raise RuntimeError("E2.7 stage proposal no longer matches controller")
        if self.stage == "worker_variance":
            return {"transitioned": False, "failure": None}
        self.stage_update_count = int(proposal["stage_update_count"])
        self.consecutive_passes = int(proposal["consecutive_passes"])
        transitioned = False
        if bool(proposal["transition_requested"]):
            if transition_confirmed:
                self.transition_history.append(
                    {
                        "from": self.stage,
                        "to": proposal["next_stage"],
                        "update_id": int(proposal["update_id"]),
                        "stage_update_count": self.stage_update_count,
                        "reason": proposal.get("transition_reason", "metrics_passed"),
                    }
                )
                self.stage = str(proposal["next_stage"])
                self.stage_update_count = 0
                self.consecutive_passes = 0
                transitioned = True
            else:
                self.consecutive_passes = 0
        failed = (
            int(proposal["update_id"]) >= int(proposal["maximum_update"])
            and not transitioned
        )
        return {
            "transitioned": transitioned,
            "warning": (
                "budget_exhausted"
                if transitioned
                and proposal.get("transition_reason") == "budget_exhausted"
                else None
            ),
            "failure": (
                "preference_stage_failed"
                if failed and not self._is_v4()
                else None
            ),
        }

    def observe(self, losses: dict[str, Any], *, update_id: int) -> None:
        """Compatibility path for tests and legacy callers without validation."""

        outcome = self.commit(self.propose(losses, update_id=update_id))
        if outcome["failure"]:
            raise RuntimeError(
                "preference_stage_failed: "
                f"stage={self.stage}, update={update_id}, "
                f"consecutive_passes={self.consecutive_passes}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": getattr(
                self, "_protocol_version", "e1_centered_adapter_metric_gated_v2"
            ),
            "stage": self.stage,
            "stage_update_count": self.stage_update_count,
            "consecutive_passes": self.consecutive_passes,
            "transition_history": list(self.transition_history),
        }

    def restore(self, payload: object) -> None:
        """Restore the curriculum state from a strict protocol checkpoint."""
        if not isinstance(payload, dict):
            raise ValueError("E2.7 resume checkpoint lacks stage-controller state")
        expected = {
            "version",
            "stage",
            "stage_update_count",
            "consecutive_passes",
            "transition_history",
        }
        if set(payload) != expected or (
            payload.get("version")
            != getattr(self, "_protocol_version", "e1_centered_adapter_metric_gated_v2")
        ):
            raise ValueError("E2.7 resume checkpoint has invalid stage-controller state")
        stage = str(payload["stage"])
        if stage not in {"gate", "production_pair", "worker_variance"}:
            raise ValueError("E2.7 resume checkpoint has an invalid stage")
        stage_update_count = int(payload["stage_update_count"])
        consecutive_passes = int(payload["consecutive_passes"])
        history = payload["transition_history"]
        if (
            stage_update_count < 0
            or consecutive_passes < 0
            or not isinstance(history, list)
            or not all(isinstance(item, dict) for item in history)
        ):
            raise ValueError("E2.7 resume checkpoint has invalid stage counters")
        self.stage = stage
        self.stage_update_count = stage_update_count
        self.consecutive_passes = consecutive_passes
        self.transition_history = [dict(item) for item in history]


def _e2_7_preference_stage(config: dict, update_number: int) -> str | None:
    """Compatibility helper for callers that only need the initial stage."""

    controller = E2_7PreferenceStageController.from_config(config)
    if controller is None:
        return None
    if int(update_number) < 1 or int(update_number) > controller.final_update:
        raise ValueError("E2.7 update is outside the configured budget")
    return controller.stage


def _development_acceptance_enabled(config: dict) -> bool:
    raw = config["training"].get("development_acceptance", {})
    if not isinstance(raw, dict):
        raise TypeError("training.development_acceptance must be an object")
    return bool(raw.get("enabled", False))


def _accepted_checkpoint_path(run_directory: Path, config: dict) -> Path:
    if _development_acceptance_enabled(config):
        return run_directory / "development_accepted_pareto_checkpoint.pt"
    return run_directory / "accepted_checkpoint.pt"


def _restore_e2_7_resume_provenance(
    agent: PPOAgent,
    metadata: dict[str, Any],
    config: dict,
) -> dict[str, Any] | None:
    if not _development_acceptance_enabled(config):
        return None
    expected_suite = str(config.get("experiment_suite_version", "legacy"))
    if (
        "experiment_suite_version" in metadata
        and str(metadata["experiment_suite_version"]) != expected_suite
    ):
        raise ValueError(
            "E2.7 strict resume checkpoint has a different experiment protocol"
        )
    report = metadata.get("warm_start")
    if not isinstance(report, dict):
        raise ValueError("E2.7 resume checkpoint lacks warm-start provenance")
    warm_settings = config["training"].get("warm_start", {})
    required_source = project_path(
        warm_settings["required_source_checkpoint"]
    ).resolve()
    reported_source = Path(str(report.get("source_checkpoint", ""))).resolve()
    if reported_source != required_source:
        raise ValueError("E2.7 resume checkpoint has the wrong E1 source")
    expected_count = int(warm_settings["expected_shared_parameter_count"])
    if int(report.get("loaded_shared_parameter_count", -1)) != expected_count:
        raise ValueError("E2.7 resume checkpoint has invalid shared-parameter provenance")
    agent.restore_e1_teacher_from_warm_start_report(report)
    return dict(report)


def _e2_3_failure_replay_cells(
    config: dict,
) -> tuple[tuple[str, PreferenceVector], ...]:
    raw = config["training"].get("e2_3_failure_replay")
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise TypeError("training.e2_3_failure_replay must be an object")
    if not bool(raw.get("enabled", False)):
        return ()
    if str(raw.get("version")) != "e2_3_update200_low_flow_10_v1":
        raise ValueError("unsupported E2.3 failure replay version")
    if int(raw.get("every_updates", 0)) != 5:
        raise ValueError("E2.3 failure replay must run every five updates")
    cells = raw.get("cells")
    if not isinstance(cells, list) or len(cells) != 10:
        raise ValueError("E2.3 failure replay requires exactly ten cells")
    normalized = tuple(
        (str(cell[0]), PreferenceVector(*map(float, cell[1])))
        for cell in cells
    )
    if any(preference.flow > 0.2 + 1e-12 for _, preference in normalized):
        raise ValueError("E2.3 failure replay cells must all be low-flow")
    return normalized


def _evaluate_e2_3_failure_replay(
    config: dict,
    *,
    agent: PPOAgent,
    runner: ParallelEpisodeRunner,
    update_id: int,
) -> tuple[list[dict], dict[str, object]]:
    cells = _e2_3_failure_replay_cells(config)
    if not cells:
        return [], {"enabled": False, "pass": True}
    dataset = load_dataset_split(config, "validation")
    records_by_id = {record.instance.instance_id: record for record in dataset}
    rows: list[dict] = []
    for instance_id, preference in cells:
        if instance_id not in records_by_id:
            raise ValueError(f"E2.3 replay instance is missing: {instance_id}")
        rollout = runner.evaluate_records(
            agent,
            [records_by_id[instance_id]],
            max_parallelism=1,
            deterministic=True,
            preference=preference,
        )[0]
        metrics = rollout.metrics
        safe = bool(
            metrics.get("terminated", False)
            and not metrics.get("truncated", False)
            and not metrics.get("schedule_violations", [])
            and float(metrics.get("maximum_worker_fatigue", math.inf))
            <= float(metrics.get("safe_fatigue_limit", -math.inf)) + 1e-9
        )
        rows.append(
            {
                "update_id": int(update_id),
                "instance_id": instance_id,
                "preference_key": _preference_key(preference),
                "w_flow": preference.flow,
                "w_cost": preference.cost,
                "w_variance": preference.variance,
                "terminated": bool(metrics.get("terminated", False)),
                "truncated": bool(metrics.get("truncated", False)),
                "terminal_reason": metrics.get("terminal_reason"),
                "maximum_worker_fatigue": metrics.get(
                    "maximum_worker_fatigue"
                ),
                "schedule_violation_count": len(
                    metrics.get("schedule_violations", [])
                ),
                "production_defer_shield_masked_count": metrics.get(
                    "production_defer_shield_masked_count", 0
                ),
                "production_defer_shield_reason_counts": json.dumps(
                    metrics.get("production_defer_shield_reason_counts", {}),
                    sort_keys=True,
                ),
                "production_defer_shield_max_risk": metrics.get(
                    "production_defer_shield_max_risk", 0.0
                ),
                "production_defer_shield_max_wait_ticks": metrics.get(
                    "production_defer_shield_max_wait_ticks", 0
                ),
                "production_defer_shield_max_work_lower_bound_ticks": (
                    metrics.get(
                        "production_defer_shield_max_work_lower_bound_ticks", 0
                    )
                ),
                "production_defer_shield_min_deadline_slack_ticks": (
                    metrics.get(
                        "production_defer_shield_min_deadline_slack_ticks"
                    )
                ),
                "pass": safe,
                "action_trace_sha256": rollout.action_trace_sha256,
            }
        )
    return rows, {
        "enabled": True,
        "update_id": int(update_id),
        "candidate_count": len(rows),
        "completed_count": sum(bool(row["pass"]) for row in rows),
        "pass": all(bool(row["pass"]) for row in rows),
    }


def _e2_7_safety_replay_cell(
    config: dict,
) -> tuple[str, PreferenceVector] | None:
    """Return the mandatory pure-cost replay cell for strict E2.7 v2."""
    raw = config["training"].get("e2_7_safety_replay")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("training.e2_7_safety_replay must be an object")
    required = {"enabled", "version", "instance_id", "preference"}
    if set(raw) != required:
        raise ValueError("E2.7 safety replay has an invalid schema")
    if not bool(raw["enabled"]):
        return None
    if str(raw["version"]) != "validation_balanced_pure_cost_each_update_v1":
        raise ValueError("unsupported E2.7 safety replay version")
    preference = PreferenceVector(*map(float, raw["preference"]))
    if preference != PreferenceVector(0.0, 1.0, 0.0):
        raise ValueError("E2.7 safety replay must use the pure-cost preference")
    instance_id = str(raw["instance_id"])
    if instance_id != "validation_balanced_2000000":
        raise ValueError("E2.7 safety replay must target validation_balanced_2000000")
    return instance_id, preference


def _evaluate_e2_7_safety_replay(
    config: dict,
    *,
    agent: PPOAgent,
    runner: ParallelEpisodeRunner,
    update_id: int,
    stage: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cell = _e2_7_safety_replay_cell(config)
    if cell is None:
        return [], {"enabled": False, "pass": True}
    instance_id, preference = cell
    records = {
        record.instance.instance_id: record
        for record in load_dataset_split(config, "validation")
    }
    if instance_id not in records:
        raise ValueError(f"E2.7 safety replay instance is missing: {instance_id}")
    rollout = runner.evaluate_records(
        agent,
        [records[instance_id]],
        max_parallelism=1,
        deterministic=True,
        preference=preference,
    )[0]
    metrics = rollout.metrics
    physical_safety_pass = bool(
        not metrics.get("schedule_violations", [])
        and float(metrics.get("maximum_worker_fatigue", math.inf))
        <= float(metrics.get("safe_fatigue_limit", -math.inf)) + 1e-9
    )
    completion_pass = bool(metrics.get("terminated", False)) and not bool(
        metrics.get("truncated", False)
    )
    safe = bool(physical_safety_pass and completion_pass)
    row = {
        "update_id": int(update_id),
        "stage": str(stage),
        "instance_id": instance_id,
        "preference_key": _preference_key(preference),
        "w_flow": preference.flow,
        "w_cost": preference.cost,
        "w_variance": preference.variance,
        "terminated": bool(metrics.get("terminated", False)),
        "truncated": bool(metrics.get("truncated", False)),
        "terminal_reason": metrics.get("terminal_reason"),
        "completed_orders": metrics.get("completed_orders"),
        "total_orders": metrics.get("total_orders"),
        "schedule_violation_count": len(metrics.get("schedule_violations", [])),
        "maximum_worker_fatigue": metrics.get("maximum_worker_fatigue"),
        "unrecoverable_deadlock_diagnostic": json.dumps(
            metrics.get("first_unrecoverable_deadlock_diagnostic"),
            sort_keys=True,
        ),
        "physical_safety_pass": physical_safety_pass,
        "completion_pass": completion_pass,
        "pass": safe,
        "action_trace_sha256": rollout.action_trace_sha256,
    }
    return [row], {
        "enabled": True,
        "update_id": int(update_id),
        "stage": str(stage),
        "physical_safety_pass": physical_safety_pass,
        "completion_pass": completion_pass,
        "pass": safe,
        "failure_cell": None if physical_safety_pass else dict(row),
    }


def _save_rejected_candidate(
    config: dict,
    *,
    run_directory: Path,
    agent: PPOAgent,
    update_id: int,
    stage: str | None,
    failure_source: str,
    failure_cell: dict[str, Any] | None,
    preference_stage_controller: E2_7PreferenceStageController | None,
) -> dict[str, Any]:
    """Persist the unsafe online candidate before any safety rollback."""
    artifact = {
        "version": "rejected_candidate_v1",
        "update_id": int(update_id),
        "stage": stage,
        "failure_source": str(failure_source),
        "learning_rate": float(agent.learning_rate),
        "failure_cell": dict(failure_cell or {}),
        "action_trace_sha256": (failure_cell or {}).get("action_trace_sha256"),
        "preference_stage_controller": (
            preference_stage_controller.as_dict()
            if preference_stage_controller is not None
            else None
        ),
    }
    agent.save(
        run_directory / "latest_rejected_candidate.pt",
        metadata={
            **_checkpoint_protocol_metadata(config),
            "checkpoint_role": "latest_rejected_candidate",
            **artifact,
        },
    )
    write_json(run_directory / "latest_rejected_candidate.json", artifact)
    return artifact


def _restore_e2_7_rollback_checkpoint(
    agent: PPOAgent,
    checkpoint: Path,
    preference_stage_controller: E2_7PreferenceStageController | None,
    *,
    restore_stage_controller: bool = True,
) -> dict[str, Any]:
    """Restore model, optimizer, and curriculum as one E2.7 rollback state."""

    metadata = agent.load(checkpoint, load_optimizer=True)
    if preference_stage_controller is not None and restore_stage_controller:
        preference_stage_controller.restore(
            metadata.get("preference_stage_controller")
        )
        preference_stage_controller.apply(agent)
    return metadata


def _build_e2_7_safe_state_pool(
    config: dict,
    *,
    agent: PPOAgent,
    runner: ParallelEpisodeRunner,
) -> dict[str, dict[str, Any]] | None:
    if (
        agent.counterfactual_preference_consistency.get("version")
        not in {"centered_gate_pair_worker_v3", "centered_gate_pair_worker_v4"}
    ):
        return None
    cells = _e2_3_failure_replay_cells(config)
    if not cells:
        return None
    validation_records = list(load_dataset_split(config, "validation"))
    records_by_id = {
        record.instance.instance_id: record for record in validation_records
    }
    pools: dict[str, list[tuple[Any, np.ndarray]]] = {
        "gate": [],
        "production_pair": [],
        "worker_variance": [],
    }
    runner.evaluate_records(
        agent,
        validation_records,
        max_parallelism=min(runner.worker_count, len(validation_records)),
        deterministic=True,
        preference=PreferenceVector(*CANONICAL_PREFERENCE),
        dual_legal_state_sink=pools["gate"],
        maximum_captured_dual_legal_states=128,
        production_pair_state_sink=pools["production_pair"],
        worker_variance_state_sink=pools["worker_variance"],
        maximum_captured_preference_states=256,
    )
    canonical_counts = {name: len(states) for name, states in pools.items()}
    for instance_id, preference in cells:
        if instance_id not in records_by_id:
            raise ValueError(f"E2.3 replay instance is missing: {instance_id}")
        runner.evaluate_records(
            agent,
            [records_by_id[instance_id]],
            max_parallelism=1,
            deterministic=True,
            preference=preference,
            dual_legal_state_sink=pools["gate"],
            maximum_captured_dual_legal_states=256,
            production_pair_state_sink=pools["production_pair"],
            worker_variance_state_sink=pools["worker_variance"],
            maximum_captured_preference_states=512,
        )
    e2_3_counts = {
        name: len(states) - canonical_counts[name]
        for name, states in pools.items()
    }
    diagnostic_checkpoint = project_path(
        "result/runs/v7_2000_e2_3_safe_production_seed11/last_checkpoint.pt"
    )
    provenance = {
        "e1_validation_state_count": canonical_counts,
        "e2_3_failure_cell_state_count": e2_3_counts,
        "e2_3_failure_cell_count": len(cells),
        "e1_checkpoint": (
            agent.warm_start_report.get("source_checkpoint")
            if agent.warm_start_report
            else None
        ),
        "e1_checkpoint_sha256": (
            agent.warm_start_report.get("source_checkpoint_sha256")
            if agent.warm_start_report
            else None
        ),
        "e2_3_diagnostic_checkpoint": str(diagnostic_checkpoint.resolve()),
        "e2_3_diagnostic_checkpoint_sha256": _checkpoint_sha256(
            diagnostic_checkpoint
        ),
        "e2_3_weights_loaded": False,
    }
    return {
        name: agent.set_centered_state_pool(
            name,
            states,
            provenance={**provenance, "pool_kind": name},
        )
        for name, states in pools.items()
    }


def _evaluate_e2_7_full_grid_pair_response(
    config: dict,
    *,
    agent: PPOAgent,
    runner: ParallelEpisodeRunner,
    validation_parallel_envs: int,
    minimum_rate: float,
) -> dict[str, Any]:
    """Certify pair ordering on states captured from the full validation grid."""

    states: list[tuple[Any, np.ndarray]] = []
    records = list(load_dataset_split(config, "validation"))
    preferences = tuple(simplex_lattice(5, include=(CANONICAL_PREFERENCE,)))
    for preference in preferences:
        runner.evaluate_records(
            agent,
            records,
            max_parallelism=min(validation_parallel_envs, len(records)),
            deterministic=True,
            preference=preference,
            production_pair_state_sink=states,
            maximum_captured_preference_states=4096,
        )
    minimum = int(
        agent.counterfactual_preference_consistency["objectives"][
            "production_pair"
        ]["minimum_eligible_states"]
    )
    if len(states) < minimum:
        return {
            "scope": "full_grid_22_pair_response",
            "captured_state_count": len(states),
            "eligible_state_count": 0,
            "correct_count": 0,
            "correct_rate": 0.0,
            "pass": False,
            "reason": "insufficient_eligible_states",
        }
    chunks: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for start in range(0, len(states), minimum):
            observations = [state for state, _ in states[start : start + minimum]]
            masks = [mask for _, mask in states[start : start + minimum]]
            chunks.append(
                agent.network.centered_production_pair_counterfactual_batch(
                    observations,
                    masks,
                    device=agent.device,
                )
            )
    eligible = torch.cat([item["eligible"] for item in chunks])
    eligible_count = int(eligible.sum().detach().cpu())
    correct_count = int(
        sum(
            int(item["flow_correct"].sum().detach().cpu())
            + int(item["cost_correct"].sum().detach().cpu())
            for item in chunks
        )
    )
    rate = correct_count / (2.0 * eligible_count) if eligible_count else 0.0
    return {
        "scope": "full_grid_22_pair_response",
        "captured_state_count": len(states),
        "eligible_state_count": eligible_count,
        "correct_count": correct_count,
        "correct_rate": rate,
        "pass": bool(eligible_count >= minimum and rate >= minimum_rate),
        "reason": "passed" if eligible_count >= minimum and rate >= minimum_rate else "below_threshold",
    }


def _preference_key(preference: PreferenceVector) -> str:
    return "_".join(f"{value:.12g}" for value in preference.as_tuple())


def _row_is_physically_safe(row: dict, fatigue_tolerance: float) -> bool:
    """Check physical constraints only; completion is a separate concern."""

    return bool(
        int(row.get("schedule_violation_count", 0)) == 0
        and float(row.get("maximum_worker_fatigue", math.inf))
        <= float(row.get("safe_fatigue_limit", -math.inf))
        + float(fatigue_tolerance)
    )


def _row_is_complete(row: dict) -> bool:
    return bool(row.get("terminated", False)) and not bool(
        row.get("truncated", False)
    )


def _rows_are_physically_safe(rows: list[dict], fatigue_tolerance: float) -> bool:
    return bool(rows) and all(
        _row_is_physically_safe(row, fatigue_tolerance) for row in rows
    )


def _rows_are_complete(rows: list[dict]) -> bool:
    return bool(rows) and all(_row_is_complete(row) for row in rows)


def _rows_are_safe(rows: list[dict], fatigue_tolerance: float) -> bool:
    """Historical combined safety predicate retained for legacy reports."""

    return _rows_are_complete(rows) and _rows_are_physically_safe(
        rows, fatigue_tolerance
    )


def _tiered_safe_candidate_rank(snapshot: dict[str, Any]) -> tuple[float, float, int]:
    """Deterministic preference order for the final two-candidate protocol."""

    return (
        float(snapshot.get("completion_rate", 0.0)),
        float(snapshot.get("mean_hypervolume", 0.0)),
        int(snapshot.get("update_id", 0)),
    )


def _select_tiered_primary_role(
    candidate_reports: dict[str, dict[str, Any]]
) -> str | None:
    """Implement the documented any-pass rule with deterministic precedence."""

    return next(
        (
            role
            for role in ("best_safe", "last_safe")
            if candidate_reports.get(role, {}).get("acceptance_status") == "passed"
        ),
        None,
    )


def _save_tiered_safe_candidate(
    config: dict,
    *,
    agent: PPOAgent,
    snapshot: dict[str, Any],
    completed_episodes: int,
    last_safe_checkpoint: Path,
    best_safe_candidate_checkpoint: Path,
    best_rank: tuple[float, float, int] | None,
    preference_stage_controller: "E2_7PreferenceStageController | None",
) -> tuple[tuple[float, float, int] | None, bool]:
    """Track the newest safe audit and the best safe full-grid candidate.

    Completion is deliberately *not* a prerequisite here.  A 99% complete but
    physically safe fixed-grid audit is useful as a rollback anchor and must not
    be silently discarded.  Formal quality acceptance happens after training.
    """

    if not (
        bool(snapshot.get("evaluation_integrity_pass", False))
        and bool(snapshot.get("physical_safety_pass", False))
    ):
        return best_rank, False
    role_metadata = {
        **_checkpoint_protocol_metadata(config),
        "checkpoint_role": "last_safe",
        "safe_episode": int(completed_episodes),
        "pareto_snapshot": dict(snapshot),
        "preference_stage_controller": (
            preference_stage_controller.as_dict()
            if preference_stage_controller is not None
            else None
        ),
    }
    agent.save(last_safe_checkpoint, metadata=role_metadata)
    rank = _tiered_safe_candidate_rank(snapshot)
    is_full_grid = str(snapshot.get("scope")) == "full_grid_22"
    if is_full_grid and (best_rank is None or rank > best_rank):
        agent.save(
            best_safe_candidate_checkpoint,
            metadata={**role_metadata, "checkpoint_role": "best_safe_candidate"},
        )
        return rank, True
    return best_rank, False


def _tiered_hard_failure_reason(
    losses: dict[str, Any], config: dict
) -> str | None:
    """Return non-recoverable optimizer/canonical invariant failures.

    The environment raises illegal-action errors itself.  This guard covers the
    remaining numerical and canonical checks immediately after each PPO update.
    """

    for name, value in losses.items():
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                return f"non_finite_training_metric:{name}"
    policy = _tiered_training_gates(config)
    if policy is not None and "canonical_identity_max_abs_error" in losses:
        error = float(losses["canonical_identity_max_abs_error"])
        if (not math.isfinite(error)) or error > float(
            policy["canonical_identity_tolerance"]
        ):
            return "canonical_identity_failed"
    return None


def _write_tiered_hard_failure_summary(
    run_directory: Path,
    config: dict,
    *,
    reason: str,
    update_id: int,
    rejected_candidate: dict[str, Any] | None,
) -> None:
    """Write a minimal durable summary before a fail-fast hard gate raises."""

    write_json(
        run_directory / "summary.json",
        {
            "result_schema_version": result_schema_version(config),
            "training_status": "hard_gate_failed",
            "acceptance_status": "not_run",
            "failure_reason": reason,
            "failed_update_id": int(update_id),
            "latest_rejected_candidate": rejected_candidate,
        },
    )


def _finite_spearman(left: list[float], right: list[float]) -> float:
    """Return a finite tie-aware Spearman coefficient, or zero if undefined."""

    if len(left) != len(right) or len(left) < 2:
        return 0.0

    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        return result

    left_ranks = ranks(left)
    right_ranks = ranks(right)
    if np.std(left_ranks) <= 0.0 or np.std(right_ranks) <= 0.0:
        return 0.0
    coefficient = float(np.corrcoef(left_ranks, right_ranks)[0, 1])
    return coefficient if math.isfinite(coefficient) else 0.0


def _pareto_snapshot(
    rows: list[dict],
    *,
    config: dict,
    scope: str,
    update_id: int,
    completed_episodes: int,
    fatigue_tolerance: float,
    expected_instance_ids: tuple[str, ...] | None = None,
    expected_preference_keys: tuple[str, ...] | None = None,
) -> dict[str, object]:
    scales = tuple(
        float(config["evaluation"]["quality_metric"][name])
        for name in ("flow_scale", "cost_scale", "variance_scale")
    )
    by_instance: dict[str, list[dict]] = {}
    for row in rows:
        by_instance.setdefault(str(row["instance_id"]), []).append(row)
    # Pareto metrics deliberately continue to use fully completed physical-safe
    # candidates.  The three pass fields below expose completion, physical
    # safety, and evaluation integrity independently to the training protocol.
    safe_rows = [
        row
        for row in rows
        if _row_is_complete(row) and _row_is_physically_safe(row, fatigue_tolerance)
    ]
    hypervolumes: list[float] = []
    unique_counts: list[int] = []
    unique_action_trace_counts: list[int] = []
    nondominated_counts: list[int] = []
    response_values: dict[str, list[float]] = {
        "flow": [],
        "cost": [],
        "variance": [],
    }
    for all_instance_rows in by_instance.values():
        instance_rows = [row for row in all_instance_rows if row in safe_rows]
        unique: list[tuple[float, float, float]] = []
        for row in instance_rows:
            objectives = (
                float(row["flow_time_objective"]),
                float(row["reconfiguration_cost"]),
                float(row["worker_load_variance"]),
            )
            if objectives not in unique:
                unique.append(objectives)
        normalized = [normalize_objectives(value, scales) for value in unique]
        front_indices = nondominated_indices(normalized)
        front = [normalized[index] for index in front_indices]
        hypervolumes.append(hypervolume_3d(front))
        unique_counts.append(len(unique))
        unique_action_trace_counts.append(
            len(
                {
                    str(row.get("action_trace_sha256"))
                    for row in instance_rows
                    if row.get("action_trace_sha256")
                }
            )
        )
        nondominated_counts.append(len(front))
        for objective_name, weight_name, objective_key in (
            ("flow", "w_flow", "flow_time_objective"),
            ("cost", "w_cost", "reconfiguration_cost"),
            ("variance", "w_variance", "worker_load_variance"),
        ):
            weights = [
                float(row[weight_name])
                for row in instance_rows
                if weight_name in row and objective_key in row
            ]
            objectives = [
                float(row[objective_key])
                for row in instance_rows
                if weight_name in row and objective_key in row
            ]
            response_values[objective_name].append(
                _finite_spearman(weights, objectives)
            )
    canonical_key = _preference_key(PreferenceVector(*CANONICAL_PREFERENCE))
    canonical_rows = [
        row for row in safe_rows if str(row.get("preference_key")) == canonical_key
    ]
    canonical_values = [
        float(row["preference_quality_score"])
        for row in canonical_rows
        if math.isfinite(float(row["preference_quality_score"]))
    ]
    fatigue_margins = [
        float(row["safe_fatigue_limit"])
        - float(row["maximum_worker_fatigue"])
        for row in rows
    ]
    pair_counts = Counter(
        (str(row.get("instance_id")), str(row.get("preference_key")))
        for row in rows
    )
    observed_instance_ids = set(by_instance)
    observed_preference_keys = {
        str(row.get("preference_key")) for row in rows
    }
    expected_instances = set(
        expected_instance_ids
        if expected_instance_ids is not None
        else tuple(observed_instance_ids)
    )
    expected_preferences = set(
        expected_preference_keys
        if expected_preference_keys is not None
        else tuple(observed_preference_keys)
    )
    expected_pairs = {
        (instance_id, preference_key)
        for instance_id in expected_instances
        for preference_key in expected_preferences
    }
    observed_pairs = set(pair_counts)
    missing_candidate_count = len(expected_pairs - observed_pairs)
    unexpected_candidate_count = len(observed_pairs - expected_pairs)
    duplicate_candidate_count = sum(
        max(0, count - 1) for count in pair_counts.values()
    )
    pareto_settings = _pareto_promotion_settings(config)
    full_grid = scope == "full_grid_22"
    e2_4_mode = (
        str(
            config["training"]["two_stage"].get(
                "quality_checkpoint_promotion", ""
            )
        )
        in {
            "pareto_guarded_e2_4_v1",
            "pareto_guarded_e2_5_v1",
            "pareto_guarded_e2_6_v1",
            "pareto_guarded_e2_7_development_v1",
        }
    )
    e2_6_mode = (
        str(
            config["training"]["two_stage"].get(
                "quality_checkpoint_promotion", ""
            )
        )
        == "pareto_guarded_e2_6_v1"
    )
    e2_7_mode = (
        str(
            config["training"]["two_stage"].get(
                "quality_checkpoint_promotion", ""
            )
        )
        == "pareto_guarded_e2_7_development_v1"
    )
    required_instances = int(
        pareto_settings["required_full_grid_instance_count"]
    )
    required_preferences = int(
        pareto_settings["required_full_grid_preference_count"]
    )
    required_candidates = int(
        pareto_settings["required_full_grid_candidate_count"]
    )
    exact_expected_coverage = bool(
        len(rows) == len(expected_pairs)
        and observed_instance_ids == expected_instances
        and observed_preference_keys == expected_preferences
        and missing_candidate_count == 0
        and unexpected_candidate_count == 0
        and duplicate_candidate_count == 0
    )
    coverage_pass = bool(
        not (full_grid or e2_4_mode)
        or (
            exact_expected_coverage
            and (
                not full_grid
                or (
                    (
                        not required_instances
                        or len(expected_instances) == required_instances
                    )
                    and (
                        not required_preferences
                        or len(expected_preferences) == required_preferences
                    )
                    and (
                        not required_candidates
                        or len(rows) == required_candidates
                    )
                )
            )
        )
    )
    mean_unique_action_trace_count = (
        float(np.mean(unique_action_trace_counts))
        if unique_action_trace_counts
        else 0.0
    )
    mean_unique_objective_count = (
        float(np.mean(unique_counts)) if unique_counts else 0.0
    )
    mean_nondominated_count = (
        float(np.mean(nondominated_counts)) if nondominated_counts else 0.0
    )
    action_trace_pass = bool(
        mean_unique_action_trace_count
        >= float(pareto_settings["minimum_mean_unique_action_trace_count"])
        - 1e-12
    )
    unique_objective_pass = bool(
        mean_unique_objective_count
        >= float(pareto_settings["minimum_mean_unique_objective_count"])
        - 1e-12
    )
    nondominated_pass = bool(
        mean_nondominated_count
        >= float(pareto_settings["minimum_mean_nondominated_count"])
        - 1e-12
    )
    controllability_pass = bool(
        not full_grid
        or (action_trace_pass and unique_objective_pass and nondominated_pass)
    )
    preference_diagnostics = aggregate_preference_diagnostics(rows)
    worker_direct_preference_pass = bool(
        (
            abs(
                float(
                    preference_diagnostics[
                        "worker_direct_preference_flow_logit_max_abs"
                    ]
                )
            )
            <= 1e-12
            and abs(
                float(
                    preference_diagnostics[
                        "worker_direct_preference_cost_logit_max_abs"
                    ]
                )
            )
            <= 1e-12
            and int(
                preference_diagnostics[
                    "unsafe_worker_preference_selection_count"
                ]
            )
            == 0
        )
        if e2_4_mode
        else (
            int(preference_diagnostics["worker_preference_override_count"])
            == 0
            and abs(
                float(
                    preference_diagnostics[
                        "worker_mean_preference_logit_std"
                    ]
                )
            )
            <= 1e-12
        )
    )
    preference_response_pass = bool(
        not e2_4_mode
        or (
            float(np.mean(response_values["flow"]))
            <= float(
                pareto_settings[
                    "maximum_preference_response_spearman_flow"
                ]
            )
            + 1e-12
            and float(np.mean(response_values["cost"]))
            <= float(
                pareto_settings[
                    "maximum_preference_response_spearman_cost"
                ]
            )
            + 1e-12
            and float(np.mean(response_values["variance"]))
            <= float(
                pareto_settings[
                    "maximum_preference_response_spearman_variance"
                ]
            )
            + 1e-12
        )
    )
    low_flow_rows = [row for row in rows if float(row.get("w_flow", 1.0)) <= 0.2]
    low_flow_safety_pass = _rows_are_safe(low_flow_rows, fatigue_tolerance)
    counterfactual_key = _preference_key(
        PreferenceVector(0.2, 0.4, 0.4)
    )
    counterfactual_rows = [
        row
        for row in rows
        if str(row.get("preference_key")) == counterfactual_key
    ]
    canonical_heuristic_values = [
        float(row["heuristic_quality_score"])
        for row in canonical_rows
        if "heuristic_quality_score" in row
        and math.isfinite(float(row["heuristic_quality_score"]))
    ]
    counterfactual_eligible_by_instance = {
        instance_id: sum(
            int(row.get("counterfactual_eligible_state_count", 0) or 0)
            for row in instance_rows
        )
        for instance_id, instance_rows in (
            (
                instance_id,
                [
                    row
                    for row in counterfactual_rows
                    if str(row.get("instance_id")) == instance_id
                ],
            )
            for instance_id in expected_instances
        )
    }
    counterfactual_eligible_count = sum(
        counterfactual_eligible_by_instance.values()
    )
    counterfactual_flip_count = sum(
        int(row.get("counterfactual_high_flow_commit_flip_count", 0) or 0)
        for row in counterfactual_rows
    )
    counterfactual_instance_coverage = sum(
        count > 0 for count in counterfactual_eligible_by_instance.values()
    )
    counterfactual_flip_rate = (
        counterfactual_flip_count / counterfactual_eligible_count
        if counterfactual_eligible_count
        else 0.0
    )
    counterfactual_identity_violation_count = sum(
        int(row.get("counterfactual_low_flow_identity_violation_count", 0) or 0)
        for row in counterfactual_rows
    )
    counterfactual_monotonicity_violation_count = sum(
        int(row.get("counterfactual_monotonicity_violation_count", 0) or 0)
        for row in counterfactual_rows
    )
    counterfactual_gate_pass = bool(
        not e2_6_mode
        or (
            full_grid
            and counterfactual_instance_coverage
            >= int(pareto_settings["required_counterfactual_instance_coverage"])
            and counterfactual_flip_rate
            >= float(
                pareto_settings[
                    "minimum_counterfactual_high_flow_flip_rate"
                ]
            )
            - 1e-12
            and counterfactual_identity_violation_count == 0
            and counterfactual_monotonicity_violation_count == 0
        )
    )
    centered_dual_legal_count = sum(
        int(row.get("centered_gate_dual_legal_state_count", 0) or 0)
        for row in rows
    )
    centered_flow_cost_flip_count = sum(
        int(row.get("centered_gate_flow_cost_flip_count", 0) or 0)
        for row in rows
    )
    centered_flow_variance_flip_count = sum(
        int(row.get("centered_gate_flow_variance_flip_count", 0) or 0)
        for row in rows
    )
    centered_flow_cost_flip_rate = (
        centered_flow_cost_flip_count / centered_dual_legal_count
        if centered_dual_legal_count
        else 0.0
    )
    centered_flow_variance_flip_rate = (
        centered_flow_variance_flip_count / centered_dual_legal_count
        if centered_dual_legal_count
        else 0.0
    )
    # The development gate requires both cost- and variance-heavy extremes to
    # differ from the flow-heavy decision on a material fraction of the same
    # safe dual-legal states.  Taking the minimum prevents one responsive
    # contrast from hiding a collapsed second contrast.
    centered_extreme_flip_rate = min(
        centered_flow_cost_flip_rate,
        centered_flow_variance_flip_rate,
    )
    centered_monotonicity_violation_count = sum(
        int(row.get("centered_gate_monotonicity_violation_count", 0) or 0)
        for row in rows
    )
    centered_gate_pass = bool(
        not e2_7_mode
        or (
            full_grid
            and centered_dual_legal_count > 0
            and centered_extreme_flip_rate
            >= float(
                pareto_settings["minimum_centered_gate_extreme_flip_rate"]
            )
            - 1e-12
            and centered_monotonicity_violation_count == 0
        )
    )
    canonical_quality = (
        float(np.mean(canonical_values)) if canonical_values else math.inf
    )
    canonical_heuristic_quality = (
        float(np.mean(canonical_heuristic_values))
        if canonical_heuristic_values
        else math.inf
    )
    canonical_heuristic_relative_gap = (
        (canonical_quality - canonical_heuristic_quality)
        / canonical_heuristic_quality
        if math.isfinite(canonical_quality)
        and math.isfinite(canonical_heuristic_quality)
        and canonical_heuristic_quality > 0.0
        else math.inf
    )
    canonical_development_quality_pass = bool(
        not e2_7_mode
        or canonical_heuristic_relative_gap
        <= float(pareto_settings["maximum_canonical_heuristic_relative_gap"])
        + 1e-12
    )
    return {
        "scope": scope,
        "update_id": int(update_id),
        "completed_episodes": int(completed_episodes),
        "preference_count": len(
            {str(row.get("preference_key")) for row in rows}
        ),
        "instance_count": len(by_instance),
        "candidate_count": len(rows),
        "feasible_candidate_count": len(safe_rows),
        "filtered_candidate_count": len(rows) - len(safe_rows),
        "pareto_filter_version": "safe_completed_candidates_v1",
        "missing_candidate_count": missing_candidate_count,
        "unexpected_candidate_count": unexpected_candidate_count,
        "duplicate_candidate_count": duplicate_candidate_count,
        "evaluation_integrity_pass": exact_expected_coverage,
        "coverage_pass": coverage_pass,
        "controllability_pass": controllability_pass,
        "unique_action_trace_pass": action_trace_pass,
        "unique_objective_pass": unique_objective_pass,
        "nondominated_pass": nondominated_pass,
        "worker_direct_preference_pass": worker_direct_preference_pass,
        "preference_response_pass": preference_response_pass,
        "low_flow_candidate_count": len(low_flow_rows),
        "low_flow_completion_rate": (
            sum(bool(row.get("terminated", False)) and not bool(row.get("truncated", False)) for row in low_flow_rows) / len(low_flow_rows)
            if low_flow_rows else 0.0
        ),
        "low_flow_safety_pass": low_flow_safety_pass,
        "counterfactual_preference_key": counterfactual_key,
        "counterfactual_eligible_state_count": counterfactual_eligible_count,
        "counterfactual_high_flow_commit_flip_count": counterfactual_flip_count,
        "counterfactual_high_flow_commit_flip_rate": counterfactual_flip_rate,
        "counterfactual_instance_coverage": counterfactual_instance_coverage,
        "counterfactual_low_flow_identity_violation_count": (
            counterfactual_identity_violation_count
        ),
        "counterfactual_monotonicity_violation_count": (
            counterfactual_monotonicity_violation_count
        ),
        "counterfactual_gate_pass": counterfactual_gate_pass,
        "centered_gate_dual_legal_state_count": centered_dual_legal_count,
        "centered_gate_flow_cost_flip_count": centered_flow_cost_flip_count,
        "centered_gate_flow_variance_flip_count": (
            centered_flow_variance_flip_count
        ),
        "centered_gate_flow_cost_flip_rate": centered_flow_cost_flip_rate,
        "centered_gate_flow_variance_flip_rate": (
            centered_flow_variance_flip_rate
        ),
        "centered_gate_extreme_flip_rate": centered_extreme_flip_rate,
        "centered_gate_monotonicity_violation_count": (
            centered_monotonicity_violation_count
        ),
        "centered_gate_pass": centered_gate_pass,
        "canonical_heuristic_quality": canonical_heuristic_quality,
        "canonical_heuristic_relative_gap": canonical_heuristic_relative_gap,
        "canonical_development_quality_pass": (
            canonical_development_quality_pass
        ),
        **preference_diagnostics,
        **aggregate_matching_recovery_diagnostics(rows),
        "physical_safety_pass": _rows_are_physically_safe(
            rows, fatigue_tolerance
        ),
        "completion_pass": _rows_are_complete(rows),
        # Kept for readers of result schemas < 5.0.0.  New training guards use
        # physical_safety_pass, never this compatibility conjunction.
        "all_safe": _rows_are_safe(rows, fatigue_tolerance),
        "completion_rate": (
            sum(
                _row_is_complete(row)
                for row in rows
            )
            / len(rows)
            if rows
            else 0.0
        ),
        "schedule_violation_count": sum(
            int(row.get("schedule_violation_count", 0)) for row in rows
        ),
        "minimum_fatigue_margin": min(fatigue_margins, default=-math.inf),
        "mean_hypervolume": (
            float(np.mean(hypervolumes)) if hypervolumes else 0.0
        ),
        "mean_unique_action_trace_count": mean_unique_action_trace_count,
        "mean_unique_objective_count": mean_unique_objective_count,
        "mean_nondominated_count": mean_nondominated_count,
        "preference_response_spearman_flow": float(
            np.mean(response_values["flow"])
        ) if response_values["flow"] else 0.0,
        "preference_response_spearman_cost": float(
            np.mean(response_values["cost"])
        ) if response_values["cost"] else 0.0,
        "preference_response_spearman_variance": float(
            np.mean(response_values["variance"])
        ) if response_values["variance"] else 0.0,
        "canonical_quality": canonical_quality,
    }


def _evaluate_pareto_preferences(
    config: dict,
    *,
    preferences: tuple[PreferenceVector, ...],
    scope: str,
    ppo_agent: PPOAgent,
    runner: ParallelEpisodeRunner,
    dataset_name: str,
    instance_limit: int | None,
    validation_parallel_envs: int,
    update_id: int,
    completed_episodes: int,
    fatigue_tolerance: float,
    canonical_rows: list[dict] | None = None,
) -> tuple[list[dict], dict[str, object]]:
    rows: list[dict] = []
    canonical_key = _preference_key(PreferenceVector(*CANONICAL_PREFERENCE))
    for preference in preferences:
        key = _preference_key(preference)
        if key == canonical_key and canonical_rows is not None:
            preference_rows = [dict(row) for row in canonical_rows]
        elif validation_parallel_envs > 1:
            preference_rows, _ = evaluate_dataset_parallel(
                config,
                dataset_name=dataset_name,
                ppo_agent=ppo_agent,
                runner=runner,
                instance_limit=instance_limit,
                decode_mode="greedy",
                preference=preference,
            )
        else:
            preference_rows, _, _, _ = evaluate_dataset(
                config,
                dataset_name=dataset_name,
                policy_name="ppo",
                ppo_agent=ppo_agent,
                instance_limit=instance_limit,
                decode_mode="greedy",
                preference=preference,
            )
        for row in preference_rows:
            rows.append(
                {
                    **row,
                    "preference_key": key,
                    "pareto_scope": scope,
                    "w_flow": preference.flow,
                    "w_cost": preference.cost,
                    "w_variance": preference.variance,
                }
            )
    snapshot = _pareto_snapshot(
        rows,
        config=config,
        scope=scope,
        update_id=update_id,
        completed_episodes=completed_episodes,
        fatigue_tolerance=fatigue_tolerance,
        expected_instance_ids=tuple(
            str(row["instance_id"])
            for row in (
                canonical_rows
                if canonical_rows is not None
                else [
                    row
                    for row in rows
                    if str(row.get("preference_key")) == canonical_key
                ]
            )
        ),
        expected_preference_keys=tuple(
            _preference_key(preference) for preference in preferences
        ),
    )
    return rows, snapshot


def _e2_7_local_development_gate_pass(
    config: dict,
    snapshot: dict[str, object],
    controller: "TrainingPhaseController",
) -> bool:
    if not _development_acceptance_enabled(config):
        return False
    constraints = controller.quality_promotion_constraints
    reference_hv = controller.reference_e1_pareto_hv
    reference_canonical = controller.reference_e1_canonical_quality
    if reference_hv is None or reference_canonical is None:
        return False
    candidate_hv = float(snapshot.get("mean_hypervolume", -math.inf))
    candidate_canonical = float(snapshot.get("canonical_quality", math.inf))
    return bool(
        snapshot.get("scope") == "full_grid_22"
        and bool(snapshot.get("all_safe", False))
        and bool(snapshot.get("coverage_pass", False))
        and bool(snapshot.get("controllability_pass", False))
        and bool(snapshot.get("worker_direct_preference_pass", False))
        and bool(snapshot.get("preference_response_pass", False))
        and bool(snapshot.get("low_flow_safety_pass", False))
        and bool(snapshot.get("centered_gate_pass", False))
        and bool(snapshot.get("e2_3_failure_replay_pass", False))
        and bool(snapshot.get("canonical_development_quality_pass", False))
        and math.isfinite(candidate_hv)
        and candidate_hv
        >= float(reference_hv)
        + constraints["minimum_hv_improvement"]
        - 1e-12
        and math.isfinite(candidate_canonical)
        and candidate_canonical
        <= float(reference_canonical)
        * (1.0 + constraints["canonical_relative_tolerance"])
        + constraints["canonical_absolute_tolerance"]
        + 1e-12
    )


def _evaluate_e2_7_heldout_hv(
    config: dict,
    *,
    validation_snapshot: dict[str, object],
    ppo_agent: PPOAgent,
    runner: ParallelEpisodeRunner,
    validation_parallel_envs: int,
    update_id: int,
    completed_episodes: int,
    fatigue_tolerance: float,
) -> tuple[dict[str, object], list[dict]]:
    development = config["training"]["development_acceptance"]
    references = development["reference_e1_seed11"]
    preferences = tuple(simplex_lattice(5, include=(CANONICAL_PREFERENCE,)))
    comparisons: dict[str, dict[str, object]] = {}
    candidate_rows: list[dict] = []

    def comparison(
        split: str,
        snapshot: dict[str, object],
        reference_hv: float,
    ) -> dict[str, object]:
        candidate_hv = float(snapshot["mean_hypervolume"])
        split_pass = bool(
            snapshot.get("all_safe", False)
            and snapshot.get("coverage_pass", False)
            and float(snapshot.get("completion_rate", 0.0)) >= 1.0 - 1e-12
            and int(snapshot.get("schedule_violation_count", 0)) == 0
            and int(snapshot.get("filtered_candidate_count", 0)) == 0
            and candidate_hv > float(reference_hv) + 1e-12
        )
        return {
            "split": split,
            "candidate_mean_hypervolume": candidate_hv,
            "reference_e1_seed11_mean_hypervolume": float(reference_hv),
            "strict_improvement": candidate_hv - float(reference_hv),
            "candidate_count": int(snapshot.get("candidate_count", 0)),
            "feasible_candidate_count": int(
                snapshot.get("feasible_candidate_count", 0)
            ),
            "completion_rate": float(snapshot.get("completion_rate", 0.0)),
            "all_safe": bool(snapshot.get("all_safe", False)),
            "coverage_pass": bool(snapshot.get("coverage_pass", False)),
            "schedule_violation_count": int(
                snapshot.get("schedule_violation_count", 0)
            ),
            "pass": split_pass,
        }

    comparisons["validation"] = comparison(
        "validation",
        validation_snapshot,
        float(references["validation_mean_hypervolume"]),
    )
    for split in ("test", "ood", "stress"):
        rows, snapshot = _evaluate_pareto_preferences(
            config,
            preferences=preferences,
            scope="full_grid_22",
            ppo_agent=ppo_agent,
            runner=runner,
            dataset_name=split,
            instance_limit=20,
            validation_parallel_envs=validation_parallel_envs,
            update_id=update_id,
            completed_episodes=completed_episodes,
            fatigue_tolerance=fatigue_tolerance,
        )
        candidate_rows.extend(
            {**row, "heldout_split": split} for row in rows
        )
        comparisons[split] = comparison(
            split,
            snapshot,
            float(references[f"{split}_mean_hypervolume"]),
        )
    report = {
        "version": "e2_7_equal_budget_heldout_hv_v1",
        "update_id": int(update_id),
        "completed_episodes": int(completed_episodes),
        "instance_count_per_split": 20,
        "preference_count_per_instance": len(preferences),
        "candidate_count_per_split": 20 * len(preferences),
        "pareto_filter_version": "safe_completed_candidates_v1",
        "splits": comparisons,
        "pass": all(bool(value["pass"]) for value in comparisons.values()),
    }
    return report, candidate_rows


def _evaluate_tiered_final_candidate(
    config: dict,
    *,
    role: str,
    checkpoint: Path,
    agent: PPOAgent,
    runner: ParallelEpisodeRunner,
    phase_controller: "TrainingPhaseController",
    validation_split: str,
    validation_limit: int | None,
    validation_parallel_envs: int,
    update_id: int,
    completed_episodes: int,
    fatigue_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Freeze one safe candidate and run its formal, post-training audit."""

    metadata = agent.load(checkpoint, load_optimizer=True)
    rows, snapshot = _evaluate_pareto_preferences(
        config,
        preferences=tuple(simplex_lattice(5, include=(CANONICAL_PREFERENCE,))),
        scope="full_grid_22",
        ppo_agent=agent,
        runner=runner,
        dataset_name=validation_split,
        instance_limit=validation_limit,
        validation_parallel_envs=validation_parallel_envs,
        update_id=update_id,
        completed_episodes=completed_episodes,
        fatigue_tolerance=fatigue_tolerance,
    )
    artifact_rows = [dict(row) for row in rows]
    if _e2_3_failure_replay_cells(config):
        replay_rows, replay = _evaluate_e2_3_failure_replay(
            config, agent=agent, runner=runner, update_id=update_id
        )
        artifact_rows.extend({**row, "final_acceptance_role": role} for row in replay_rows)
        snapshot["e2_3_failure_replay"] = replay
        snapshot["e2_3_failure_replay_pass"] = bool(replay["pass"])
    else:
        snapshot["e2_3_failure_replay_pass"] = True

    e2_7_mode = (
        phase_controller.quality_checkpoint_promotion
        == "pareto_guarded_e2_7_development_v1"
    )
    heldout_report: dict[str, Any]
    if e2_7_mode:
        # The local validation prerequisites are tested once the candidate is
        # frozen; heldout never runs in the online training loop.
        if _e2_7_local_development_gate_pass(config, snapshot, phase_controller):
            heldout_report, heldout_rows = _evaluate_e2_7_heldout_hv(
                config,
                validation_snapshot=snapshot,
                ppo_agent=agent,
                runner=runner,
                validation_parallel_envs=validation_parallel_envs,
                update_id=update_id,
                completed_episodes=completed_episodes,
                fatigue_tolerance=fatigue_tolerance,
            )
            artifact_rows.extend(
                {**row, "final_acceptance_role": role} for row in heldout_rows
            )
            snapshot["heldout_hv_pass"] = bool(heldout_report["pass"])
        else:
            heldout_report = {
                "status": "not_run_validation_prerequisites_failed",
                "pass": False,
            }
            snapshot["heldout_hv_pass"] = False
    else:
        heldout_report = {"status": "not_configured", "pass": True}
        snapshot["heldout_hv_pass"] = True
    snapshot["heldout_hv_report"] = heldout_report
    decision = phase_controller.evaluate_final_pareto_snapshot(snapshot)
    report = {
        "version": "tiered_final_candidate_acceptance_v1",
        "role": role,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "checkpoint_metadata_role": metadata.get("checkpoint_role"),
        "acceptance_status": "passed" if decision["pass"] else "failed",
        "decision": decision,
        "heldout": heldout_report,
        "validation_snapshot": snapshot,
    }
    return report, artifact_rows


@dataclass
class TrainingPhaseController:
    enabled: bool
    completion_target: float = 1.0
    consecutive_required: int = 3
    quality_completion_floor: float = 1.0
    quality_checkpoint_promotion: str = "completion_only"
    phase: str = "legacy"
    consecutive_successes: int = 0
    phase_transition_episode: int | None = None
    accepted_quality_updates: int = 0
    rejected_quality_updates: int = 0
    not_promoted_quality_updates: int = 0
    accepted_quality_score: tuple[float, float, float, float] | None = None
    accepted_normalized_quality_score: float | None = None
    accepted_quality_episode: int | None = None
    quality_promotion_constraints: dict[str, float] = field(default_factory=dict)
    last_promotion_diagnostics: dict[str, object] = field(default_factory=dict)
    accepted_sampled_completion_rate: float | None = None
    accepted_sampled_fatigue_cvar90: float | None = None
    pending_quality_score: tuple[float, float, float, float] | None = None
    pending_normalized_quality_score: float | None = None
    pending_quality_episode: int | None = None
    accepted_pareto_hv: float | None = None
    accepted_pareto_canonical_quality: float | None = None
    reference_e1_pareto_hv: float | None = None
    reference_e1_canonical_quality: float | None = None
    development_consecutive_full_grid_passes: int = 0
    single_objective_name: str | None = None
    accepted_single_objective_value: float | None = None
    single_objective_window_size: int = 5
    single_objective_window_statistic: str = "median"
    single_objective_rollback_below_floor_consecutive: int = 2
    single_objective_candidate_improvement_epsilon: float = 1e-9
    single_objective_audit_instance_limit: int = 500
    single_objective_audit_completion_target: float = 0.98
    single_objective_audit_max_failed_instances: int = 10
    single_objective_audit_schedule_violation_target: int = 0
    single_objective_audit_physical_safety_required: bool = True
    single_objective_window_values: list[float] = field(default_factory=list)
    single_objective_window_episodes: list[int] = field(default_factory=list)
    single_objective_candidate_anchor_value: float | None = None
    single_objective_candidate_episode: int | None = None
    single_objective_audit_count: int = 0
    accepted_single_objective_failed_instances: int | None = None
    accepted_single_objective_window_value: float | None = None
    accepted_single_objective_audit_value: float | None = None
    last_single_objective_audit_diagnostics: dict[str, object] = field(
        default_factory=dict
    )

    @classmethod
    def from_config(cls, config: dict) -> "TrainingPhaseController":
        enabled = (
            str(config["reward"].get("mode", "legacy_weighted_sum"))
            == "hierarchical_constrained_v1"
        )
        if not enabled:
            return cls(enabled=False)
        if float(config["ppo"]["gamma"]) != 1.0:
            raise ValueError(
                "hierarchical constrained training requires ppo.gamma = 1.0"
            )
        settings = config["training"]["two_stage"]
        target = float(settings["completion_target"])
        required = int(settings["consecutive_validations"])
        floor = float(settings["quality_completion_floor"])
        promotion = str(
            settings.get("quality_checkpoint_promotion", "completion_only")
        ).strip().lower()
        if not 0.0 <= target <= 1.0:
            raise ValueError("two_stage.completion_target must be in [0, 1]")
        if required < 1:
            raise ValueError(
                "two_stage.consecutive_validations must be positive"
            )
        if not 0.0 <= floor <= 1.0:
            raise ValueError(
                "two_stage.quality_completion_floor must be in [0, 1]"
            )
        if promotion not in {
            "completion_only",
            "score_improving",
            "constrained_weighted",
            "aligned_quality",
            "balanced_guarded_v7",
            SINGLE_OBJECTIVE_PROMOTION_MODE,
            "pareto_guarded_e2_v1",
            "pareto_guarded_e2_3_v1",
            "pareto_guarded_e2_4_v1",
            "pareto_guarded_e2_5_v1",
            "pareto_guarded_e2_6_v1",
            "pareto_guarded_e2_7_development_v1",
        }:
            raise ValueError(
                "two_stage.quality_checkpoint_promotion must be "
                "'completion_only', 'score_improving', or "
                "'constrained_weighted', 'aligned_quality', "
                "'balanced_guarded_v7', 'single_objective_guarded_v1', "
                "'pareto_guarded_e2_v1', or "
                "'pareto_guarded_e2_3_v1', 'pareto_guarded_e2_4_v1', "
                "'pareto_guarded_e2_5_v1', 'pareto_guarded_e2_6_v1', or "
                "'pareto_guarded_e2_7_development_v1'"
            )
        constraints: dict[str, float] = {}
        if promotion in {"constrained_weighted", "balanced_guarded_v7"}:
            configured_constraints = settings.get(
                "quality_promotion_constraints"
            )
            if not isinstance(configured_constraints, dict):
                raise ValueError(
                    "two_stage.quality_promotion_constraints must be an object"
                )
            required_constraints = (
                (
                    "minimum_normalized_quality_improvement",
                    "variance_relative_tolerance",
                    "sampled_completion_drop_tolerance",
                    "fatigue_cvar_relative_tolerance",
                    "tail_fraction",
                )
                if promotion == "balanced_guarded_v7"
                else (
                    "flow_relative_tolerance",
                    "cost_relative_tolerance",
                    "variance_relative_tolerance",
                    "minimum_normalized_score_improvement",
                )
            )
            for name in required_constraints:
                if name not in configured_constraints:
                    raise ValueError(
                        "two_stage.quality_promotion_constraints is missing "
                        f"{name}"
                    )
                value = float(configured_constraints[name])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        "two_stage.quality_promotion_constraints."
                        f"{name} must be finite and non-negative"
                    )
                constraints[name] = value
        if promotion in PARETO_PROMOTION_MODES:
            constraints = {
                name: float(value)
                for name, value in _pareto_promotion_settings(config).items()
            }
        single_objective_name: str | None = None
        single_window_size = 5
        single_window_statistic = "median"
        single_rollback_count = 2
        single_candidate_epsilon = 1e-9
        single_audit_limit = 500
        single_audit_completion = 0.98
        single_audit_max_failed = 10
        single_audit_violation_target = 0
        single_audit_physical_required = True
        if promotion == SINGLE_OBJECTIVE_PROMOTION_MODE:
            single_objective_name = _single_objective_name(config)
            promotion_settings = settings.get("single_objective_promotion")
            if not isinstance(promotion_settings, dict):
                raise ValueError(
                    "single-objective promotion requires single_objective_promotion"
                )
            single_window_size = int(promotion_settings.get("window_size", 0))
            single_window_statistic = str(
                promotion_settings.get("window_statistic", "")
            ).strip().lower()
            single_rollback_count = int(
                promotion_settings.get("rollback_below_floor_consecutive", 0)
            )
            single_candidate_epsilon = float(
                promotion_settings.get(
                    "candidate_improvement_epsilon", math.nan
                )
            )
            single_audit_limit = int(
                promotion_settings.get("audit_instance_limit", 0)
            )
            single_audit_completion = float(
                promotion_settings.get("audit_completion_target", math.nan)
            )
            single_audit_max_failed = int(
                promotion_settings.get("audit_max_failed_instances", -1)
            )
            single_audit_violation_target = int(
                promotion_settings.get(
                    "audit_schedule_violation_target", -1
                )
            )
            single_audit_physical_required = bool(
                promotion_settings.get(
                    "audit_physical_safety_required", False
                )
            )
            if single_window_size < 1 or single_window_statistic != "median":
                raise ValueError("single-objective window must be a positive median window")
            if single_rollback_count < 1:
                raise ValueError("single-objective rollback count must be positive")
            if (
                not math.isfinite(single_candidate_epsilon)
                or single_candidate_epsilon < 0.0
                or single_audit_limit != 500
                or not math.isclose(
                    single_audit_completion,
                    0.98,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or single_audit_max_failed != 10
                or single_audit_violation_target != 0
                or not single_audit_physical_required
            ):
                raise ValueError(
                    "single-objective audit requires epsilon>=0, "
                    "500 instances, completion 0.98, at most 10 failures, "
                    "zero violations, and physical safety"
                )
            worker_control = config["environment"].get(
                "worker_resource_control", {}
            )
            if (
                not isinstance(worker_control, dict)
                or worker_control.get("mode")
                != "temporal_matching_admission_recovery_v3"
                or not isinstance(
                    worker_control.get("temporal_feasibility"), dict
                )
                or int(
                    worker_control["temporal_feasibility"].get(
                        "max_search_nodes", 0
                    )
                )
                != 50_000
                or worker_control["temporal_feasibility"].get(
                    "unknown_action"
                )
                != "allow"
            ):
                raise ValueError(
                    "single-objective promotion requires "
                    "temporal_matching_admission_recovery_v3 with the "
                    "50,000-node fail-open oracle"
                )
            shield = (
                config["environment"].get("production_defer", {}).get(
                    "shield", {}
                )
            )
            if (
                not isinstance(shield, dict)
                or not bool(shield.get("enabled", False))
                or shield.get("version")
                != "deadline_progress_viability_shield_v2"
                or not math.isclose(
                    float(shield.get("soft_risk_coefficient", math.nan)),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise ValueError(
                    "single-objective promotion requires the hard "
                    "deadline_progress_viability_shield_v2 with "
                    "soft_risk_coefficient=0"
                )
        if promotion in {
            "pareto_guarded_e2_4_v1",
            "pareto_guarded_e2_5_v1",
            "pareto_guarded_e2_6_v1",
            "pareto_guarded_e2_7_development_v1",
        }:
            worker_control = config["environment"].get(
                "worker_resource_control", {}
            )
            if (
                not isinstance(worker_control, dict)
                or worker_control.get("mode")
                != "matching_admission_recovery_v2"
                or not bool(worker_control.get("require_full_matching"))
                or not bool(worker_control.get("preserve_matching_on_worker_action"))
            ):
                raise ValueError(
                    "E2.4-E2.7 requires matching_admission_recovery_v2 with "
                    "full matching preservation"
                )
        if not bool(settings["quality_validate_every_update"]):
            raise ValueError(
                "hierarchical constrained training requires validation "
                "after every quality-phase update"
            )
        controller = cls(
            enabled=True,
            completion_target=target,
            consecutive_required=required,
            quality_completion_floor=floor,
            quality_checkpoint_promotion=promotion,
            quality_promotion_constraints=constraints,
            single_objective_name=single_objective_name,
            single_objective_window_size=single_window_size,
            single_objective_window_statistic=single_window_statistic,
            single_objective_rollback_below_floor_consecutive=single_rollback_count,
            single_objective_candidate_improvement_epsilon=(
                single_candidate_epsilon
            ),
            single_objective_audit_instance_limit=single_audit_limit,
            single_objective_audit_completion_target=(
                single_audit_completion
            ),
            single_objective_audit_max_failed_instances=(
                single_audit_max_failed
            ),
            single_objective_audit_schedule_violation_target=(
                single_audit_violation_target
            ),
            single_objective_audit_physical_safety_required=(
                single_audit_physical_required
            ),
            phase="feasibility",
        )
        if promotion == "pareto_guarded_e2_7_development_v1":
            development = config["training"].get("development_acceptance")
            if not isinstance(development, dict) or not bool(
                development.get("enabled", False)
            ):
                raise ValueError(
                    "E2.7 promotion requires training.development_acceptance"
                )
            reference = development.get("reference_e1_seed11")
            if not isinstance(reference, dict):
                raise ValueError(
                    "E2.7 development acceptance requires reference_e1_seed11"
                )
            controller.reference_e1_pareto_hv = float(
                reference["validation_mean_hypervolume"]
            )
            controller.reference_e1_canonical_quality = float(
                reference["validation_canonical_quality"]
            )
            for key in (
                "validation_mean_hypervolume",
                "test_mean_hypervolume",
                "ood_mean_hypervolume",
                "stress_mean_hypervolume",
            ):
                value = float(reference[key])
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"E2.7 E1 reference metric {key} must be finite and positive"
                    )
            if (
                not math.isfinite(controller.reference_e1_pareto_hv)
                or controller.reference_e1_pareto_hv <= 0.0
                or not math.isfinite(controller.reference_e1_canonical_quality)
                or controller.reference_e1_canonical_quality <= 0.0
            ):
                raise ValueError("E2.7 E1 reference metrics must be finite and positive")
        return controller

    def should_validate(self, regular_due: bool) -> bool:
        return self.phase == "quality" or regular_due

    def observe_validation(
        self,
        completion_rate: float,
        *,
        completed_episodes: int,
        score: tuple[float, float, float, float] | None = None,
        normalized_quality_score: float | None = None,
        truncated_count: int = 0,
        schedule_violation_count: int = 0,
        physical_safety_pass: bool = True,
    ) -> str:
        rate = float(completion_rate)
        truncations = int(truncated_count)
        violations = int(schedule_violation_count)
        self.last_promotion_diagnostics = {}
        if not self.enabled:
            return "legacy"
        if self.phase == "feasibility":
            feasibility_pass = bool(
                rate >= self.completion_target
                and (
                    self.quality_checkpoint_promotion
                    != SINGLE_OBJECTIVE_PROMOTION_MODE
                    or (truncations == 0 and violations == 0 and physical_safety_pass)
                )
            )
            if feasibility_pass:
                self.consecutive_successes += 1
            else:
                self.consecutive_successes = 0
            if self.consecutive_successes >= self.consecutive_required:
                self.phase = "quality"
                self.phase_transition_episode = int(completed_episodes)
                if self.quality_checkpoint_promotion not in (
                    PARETO_PROMOTION_MODES | {SINGLE_OBJECTIVE_PROMOTION_MODE}
                ):
                    self.accepted_quality_score = score
                    self.accepted_normalized_quality_score = (
                        normalized_quality_score
                    )
                    self.accepted_quality_episode = int(completed_episodes)
                self._record_constrained_promotion_diagnostics(
                    completion_rate=rate,
                    score=score,
                    normalized_quality_score=normalized_quality_score,
                    event="transition",
                    reason="transition_anchor",
                    anchor_score=None,
                    anchor_normalized_quality_score=None,
                    anchor_episode=None,
                )
                if (
                    self.quality_checkpoint_promotion
                    == SINGLE_OBJECTIVE_PROMOTION_MODE
                ):
                    self.last_promotion_diagnostics = {
                        "promotion_mode": SINGLE_OBJECTIVE_PROMOTION_MODE,
                        "promotion_event": "transition",
                        "promotion_decision_reason": "feasibility_phase_complete",
                        "promotion_target_objective": self.single_objective_name,
                        "promotion_candidate_objective_value": None,
                        "promotion_anchor_objective_value": None,
                        "promotion_completion_constraint_pass": (
                            rate >= self.completion_target
                        ),
                        "promotion_truncation_constraint_pass": truncations == 0,
                        "promotion_violation_constraint_pass": violations == 0,
                        "promotion_physical_safety_constraint_pass": bool(
                            physical_safety_pass
                        ),
                        "window_size": self.single_objective_window_size,
                        "window_statistic": self.single_objective_window_statistic,
                        "window_count": 0,
                        "window_objective_values": [],
                        "window_objective_episodes": [],
                        "window_objective_statistic": None,
                        "candidate_anchor_value": None,
                        "previous_candidate_anchor_value": None,
                        "accepted_window_median": None,
                        "accepted_failed_instance_count": None,
                        "audit_required": False,
                    }
                return "transition"
            return "feasibility"
        if self.quality_checkpoint_promotion in PARETO_PROMOTION_MODES:
            return "pareto_pending"
        if (
            self.quality_checkpoint_promotion
            == SINGLE_OBJECTIVE_PROMOTION_MODE
        ):
            return self._observe_single_objective_candidate(
                completion_rate=rate,
                completed_episodes=completed_episodes,
                score=score,
                truncated_count=truncations,
                schedule_violation_count=violations,
                physical_safety_pass=physical_safety_pass,
            )
        if self.quality_checkpoint_promotion == "balanced_guarded_v7":
            return self._observe_balanced_greedy_candidate(
                completion_rate=rate,
                completed_episodes=completed_episodes,
                score=score,
                normalized_quality_score=normalized_quality_score,
            )
        if self.quality_checkpoint_promotion == "aligned_quality":
            anchor = self.accepted_normalized_quality_score
            anchor_episode = self.accepted_quality_episode
            eligible = (
                rate >= self.quality_completion_floor
                and normalized_quality_score is not None
                and math.isfinite(float(normalized_quality_score))
            )
            improved = bool(
                eligible
                and (
                    anchor is None
                    or float(normalized_quality_score) < float(anchor) - 1e-12
                )
            )
            event = "promoted" if improved else "not_promoted"
            reason = (
                "quality_improved"
                if improved
                else (
                    "completion_below_floor"
                    if rate < self.quality_completion_floor
                    else "quality_not_improved"
                )
            )
            if improved:
                self.accepted_quality_updates += 1
                self.accepted_quality_score = score
                self.accepted_normalized_quality_score = float(
                    normalized_quality_score
                )
                self.accepted_quality_episode = int(completed_episodes)
            else:
                self.not_promoted_quality_updates += 1
            self.last_promotion_diagnostics = {
                "promotion_mode": "aligned_quality",
                "promotion_event": event,
                "promotion_decision_reason": reason,
                "promotion_candidate_normalized_quality_score": (
                    float(normalized_quality_score)
                    if normalized_quality_score is not None
                    and math.isfinite(float(normalized_quality_score))
                    else None
                ),
                "promotion_anchor_normalized_quality_score": anchor,
                "promotion_anchor_episode": anchor_episode,
                "promotion_completion_constraint_pass": (
                    rate >= self.quality_completion_floor
                ),
            }
            return event
        if rate >= self.quality_completion_floor:
            if (
                self.quality_checkpoint_promotion == "score_improving"
                and (
                    score is None
                    or (
                        self.accepted_quality_score is not None
                        and score >= self.accepted_quality_score
                    )
                )
            ):
                self.rejected_quality_updates += 1
                return "rejected"
            if self.quality_checkpoint_promotion == "constrained_weighted":
                event, reason = self._observe_constrained_candidate(
                    completion_rate=rate,
                    completed_episodes=completed_episodes,
                    score=score,
                    normalized_quality_score=normalized_quality_score,
                )
                if event == "rejected":
                    self.rejected_quality_updates += 1
                    return event
                self.accepted_quality_updates += 1
                return event
            self.accepted_quality_updates += 1
            if score is not None:
                self.accepted_quality_score = score
                self.accepted_quality_episode = int(completed_episodes)
            return "accepted"
        self.rejected_quality_updates += 1
        self._record_constrained_promotion_diagnostics(
            completion_rate=rate,
            score=score,
            normalized_quality_score=normalized_quality_score,
            event="rejected",
            reason="completion_below_floor",
            anchor_score=self.accepted_quality_score,
            anchor_normalized_quality_score=(
                self.accepted_normalized_quality_score
            ),
            anchor_episode=self.accepted_quality_episode,
        )
        return "rejected"

    def _observe_single_objective_candidate(
        self,
        *,
        completion_rate: float,
        completed_episodes: int,
        score: tuple[float, float, float, float] | None,
        truncated_count: int,
        schedule_violation_count: int,
        physical_safety_pass: bool,
    ) -> str:
        finite = bool(
            score is not None
            and len(score) == 4
            and math.isfinite(float(score[1]))
        )
        candidate = float(score[1]) if finite and score is not None else None
        completion_pass = completion_rate >= self.quality_completion_floor
        truncation_pass = int(truncated_count) == 0
        violation_pass = int(schedule_violation_count) == 0
        physical_pass = bool(physical_safety_pass)
        previous_candidate_anchor = self.single_objective_candidate_anchor_value
        exploration_gate = bool(
            completion_pass and violation_pass and physical_pass and finite
        )
        window_stat = None
        audit_required = False
        if not exploration_gate:
            self.single_objective_window_values.clear()
            self.single_objective_window_episodes.clear()
            self.rejected_quality_updates += 1
            reason = (
                "completion_below_floor" if not completion_pass else
                "schedule_violation_nonzero" if not violation_pass else
                "physical_safety_failed" if not physical_pass else
                "missing_or_non_finite_objective"
            )
            event = "rejected"
        else:
            self.single_objective_window_values.append(float(candidate))
            self.single_objective_window_episodes.append(int(completed_episodes))
            self.single_objective_window_values = self.single_objective_window_values[-self.single_objective_window_size:]
            self.single_objective_window_episodes = self.single_objective_window_episodes[-self.single_objective_window_size:]
            if len(self.single_objective_window_values) < self.single_objective_window_size:
                event, reason = "window_warmup", "window_warmup"
            else:
                window_stat = float(statistics.median(self.single_objective_window_values))
                audit_required = (
                    self.single_objective_candidate_anchor_value is None
                    or window_stat
                    < float(self.single_objective_candidate_anchor_value)
                    - self.single_objective_candidate_improvement_epsilon
                )
                if audit_required:
                    self.single_objective_candidate_anchor_value = window_stat
                    self.single_objective_candidate_episode = int(
                        completed_episodes
                    )
                    event, reason = (
                        "audit_required",
                        "candidate_window_improved",
                    )
                else:
                    event, reason = "not_promoted", "window_not_improved"
                    self.not_promoted_quality_updates += 1

        self.last_promotion_diagnostics = {
            "promotion_mode": SINGLE_OBJECTIVE_PROMOTION_MODE,
            "promotion_event": event,
            "promotion_decision_reason": reason,
            "promotion_target_objective": self.single_objective_name,
            "promotion_statistic": "rolling_median",
            "window_size": self.single_objective_window_size,
            "window_statistic": self.single_objective_window_statistic,
            "window_count": len(self.single_objective_window_values),
            "window_objective_values": list(self.single_objective_window_values),
            "window_objective_episodes": list(self.single_objective_window_episodes),
            "window_objective_statistic": window_stat,
            "promotion_candidate_objective_value": candidate,
            "candidate_anchor_value": self.single_objective_candidate_anchor_value,
            "previous_candidate_anchor_value": previous_candidate_anchor,
            "accepted_window_median": self.accepted_single_objective_window_value,
            "accepted_failed_instance_count": self.accepted_single_objective_failed_instances,
            "promotion_anchor_episode": self.accepted_quality_episode,
            "promotion_completion_constraint_pass": completion_pass,
            "promotion_truncation_constraint_pass": truncation_pass,
            "promotion_violation_constraint_pass": violation_pass,
            "promotion_physical_safety_constraint_pass": physical_pass,
            "audit_required": audit_required,
        }
        return event

    def observe_single_objective_audit(
        self,
        audit: dict[str, object],
        *,
        completed_episodes: int,
        window_median: float,
    ) -> str:
        if self.quality_checkpoint_promotion != SINGLE_OBJECTIVE_PROMOTION_MODE:
            raise RuntimeError("single-objective audit requires its promotion mode")
        instance_count = int(audit.get("instance_count", 0))
        completed_count = int(audit.get("completed_count", 0))
        failed_count = instance_count - completed_count
        completion_rate = float(audit.get("completion_rate", math.nan))
        truncated_count = int(audit.get("truncated_count", 0))
        violation_count = int(audit.get("schedule_violation_count", 0))
        physical_pass = bool(audit.get("physical_safety_pass", False))
        objective_value = audit.get("single_objective_value")
        objective_pass = bool(
            objective_value is not None
            and math.isfinite(float(objective_value))
        )
        completion_pass = bool(
            instance_count == self.single_objective_audit_instance_limit
            and math.isfinite(completion_rate)
            and completion_rate
            >= self.single_objective_audit_completion_target - 1e-12
            and failed_count
            <= self.single_objective_audit_max_failed_instances
        )
        violation_pass = bool(
            violation_count
            == self.single_objective_audit_schedule_violation_target
        )
        safety_pass = bool(
            physical_pass
            if self.single_objective_audit_physical_safety_required
            else True
        )
        audit_pass = bool(
            completion_pass and violation_pass and safety_pass and objective_pass
        )
        previous_rank = (
            None
            if self.accepted_single_objective_failed_instances is None
            or self.accepted_single_objective_window_value is None
            else (
                int(self.accepted_single_objective_failed_instances),
                float(self.accepted_single_objective_window_value),
            )
        )
        candidate_rank = (int(failed_count), float(window_median))
        accepted = bool(
            audit_pass
            and (previous_rank is None or candidate_rank < previous_rank)
        )
        self.single_objective_audit_count += 1
        if accepted:
            self.accepted_single_objective_failed_instances = failed_count
            self.accepted_single_objective_window_value = float(window_median)
            audit_value = audit.get("single_objective_value")
            self.accepted_single_objective_audit_value = (
                None if audit_value is None else float(audit_value)
            )
            self.accepted_single_objective_value = float(window_median)
            self.accepted_quality_episode = int(completed_episodes)
            self.accepted_quality_updates += 1
            event = "accepted"
            reason = "first_audit_pass" if previous_rank is None else "audit_rank_improved"
        elif not audit_pass:
            self.rejected_quality_updates += 1
            event = "audit_rejected"
            reason = (
                "audit_completion_below_98"
                if not completion_pass
                else "audit_schedule_violation_nonzero"
                if not violation_pass
                else "audit_physical_safety_failed"
                if not safety_pass
                else "audit_objective_non_finite"
            )
        else:
            self.not_promoted_quality_updates += 1
            event = "audit_passed_not_accepted"
            reason = "audit_rank_not_improved"
        self.last_single_objective_audit_diagnostics = {
            "audit_event": event,
            "audit_decision_reason": reason,
            "audit_instance_count": instance_count,
            "audit_completed_count": completed_count,
            "audit_failed_instance_count": failed_count,
            "audit_completion_rate": completion_rate,
            "audit_truncated_count": truncated_count,
            "audit_schedule_violation_count": violation_count,
            "audit_physical_safety_pass": physical_pass,
            "audit_completion_pass": completion_pass,
            "audit_violation_pass": violation_pass,
            "audit_safety_pass": safety_pass,
            "audit_objective_pass": objective_pass,
            "audit_single_objective_value": (
                None if objective_value is None else float(objective_value)
            ),
            "audit_pass": audit_pass,
            "audit_window_median": float(window_median),
            "audit_candidate_rank": list(candidate_rank),
            "audit_previous_accepted_rank": (
                None if previous_rank is None else list(previous_rank)
            ),
            "audit_accepted_rank": (
                None
                if self.accepted_single_objective_failed_instances is None
                or self.accepted_single_objective_window_value is None
                else [
                    self.accepted_single_objective_failed_instances,
                    self.accepted_single_objective_window_value,
                ]
            ),
            "accepted_checkpoint_episode": self.accepted_quality_episode,
        }
        return event

    def reset_single_objective_window(self) -> None:
        """Clear the rolling daily window while retaining candidate/accepted anchors."""
        self.single_objective_window_values.clear()
        self.single_objective_window_episodes.clear()

    def observe_pareto_snapshot(
        self,
        snapshot: dict[str, object],
        *,
        completed_episodes: int,
    ) -> str:
        if self.quality_checkpoint_promotion not in PARETO_PROMOTION_MODES:
            raise RuntimeError("Pareto snapshots require a Pareto guard")
        constraints = self.quality_promotion_constraints
        candidate_hv = float(snapshot["mean_hypervolume"])
        candidate_canonical = float(snapshot["canonical_quality"])
        safety_pass = bool(snapshot["all_safe"])
        coverage_pass = bool(snapshot.get("coverage_pass", True))
        controllability_pass = bool(
            snapshot.get("controllability_pass", True)
        )
        worker_direct_preference_pass = bool(
            snapshot.get("worker_direct_preference_pass", True)
        )
        e2_3_mode = (
            self.quality_checkpoint_promotion == "pareto_guarded_e2_3_v1"
        )
        e2_4_mode = (
            self.quality_checkpoint_promotion == "pareto_guarded_e2_4_v1"
        )
        e2_5_mode = (
            self.quality_checkpoint_promotion == "pareto_guarded_e2_5_v1"
        )
        e2_6_mode = (
            self.quality_checkpoint_promotion == "pareto_guarded_e2_6_v1"
        )
        e2_7_mode = (
            self.quality_checkpoint_promotion
            == "pareto_guarded_e2_7_development_v1"
        )
        strict_pareto_mode = (
            e2_3_mode or e2_4_mode or e2_5_mode or e2_6_mode or e2_7_mode
        )
        preference_response_pass = bool(
            snapshot.get("preference_response_pass", True)
        )
        if strict_pareto_mode:
            coverage_pass = bool(
                coverage_pass and snapshot.get("scope") == "full_grid_22"
            )
        finite = math.isfinite(candidate_hv) and math.isfinite(
            candidate_canonical
        )
        anchor_hv = (
            self.accepted_pareto_hv
            if self.accepted_pareto_hv is not None
            else self.reference_e1_pareto_hv
        )
        anchor_canonical = (
            self.accepted_pareto_canonical_quality
            if self.accepted_pareto_canonical_quality is not None
            else self.reference_e1_canonical_quality
        )
        baseline = anchor_hv is None or anchor_canonical is None
        hv_pass = bool(
            finite
            and (
                baseline
                or candidate_hv
                >= float(anchor_hv)
                + constraints["minimum_hv_improvement"]
                - 1e-12
            )
        )
        canonical_pass = bool(
            finite
            and (
                baseline
                or candidate_canonical
                <= float(anchor_canonical)
                * (1.0 + constraints["canonical_relative_tolerance"])
                + constraints["canonical_absolute_tolerance"]
                + 1e-12
            )
        )
        promoted = bool(
            safety_pass
            and (coverage_pass if strict_pareto_mode else True)
            and (controllability_pass if strict_pareto_mode else True)
            and (worker_direct_preference_pass if strict_pareto_mode else True)
            and (preference_response_pass if (e2_4_mode or e2_5_mode or e2_6_mode or e2_7_mode) else True)
            and (bool(snapshot.get("low_flow_safety_pass", True)) if (e2_5_mode or e2_6_mode or e2_7_mode) else True)
            and (bool(snapshot.get("counterfactual_gate_pass", True)) if e2_6_mode else True)
            and (bool(snapshot.get("centered_gate_pass", True)) if e2_7_mode else True)
            and (
                bool(snapshot.get("e2_3_failure_replay_pass", False))
                if e2_7_mode
                else True
            )
            and (
                bool(snapshot.get("canonical_development_quality_pass", True))
                if e2_7_mode
                else True
            )
            and (
                bool(snapshot.get("heldout_hv_pass", False))
                if e2_7_mode
                else True
            )
            and hv_pass
            and canonical_pass
        )
        if e2_7_mode:
            if promoted:
                self.development_consecutive_full_grid_passes += 1
                promoted = self.development_consecutive_full_grid_passes >= 2
            else:
                self.development_consecutive_full_grid_passes = 0
        if promoted:
            event = "accepted" if baseline else "promoted"
            reason = "pareto_baseline" if baseline else "pareto_hv_improved"
            self.accepted_pareto_hv = candidate_hv
            self.accepted_pareto_canonical_quality = candidate_canonical
            self.accepted_quality_episode = int(completed_episodes)
            self.accepted_quality_updates += 1
        else:
            hard_rejection = bool(
                not safety_pass
                or (strict_pareto_mode and not coverage_pass)
                or (strict_pareto_mode and not controllability_pass)
                or (strict_pareto_mode and not worker_direct_preference_pass)
                or ((e2_4_mode or e2_5_mode or e2_6_mode or e2_7_mode) and not preference_response_pass)
                or ((e2_5_mode or e2_6_mode or e2_7_mode) and not bool(snapshot.get("low_flow_safety_pass", False)))
                or (e2_6_mode and not bool(snapshot.get("counterfactual_gate_pass", False)))
                or (e2_7_mode and not bool(snapshot.get("centered_gate_pass", False)))
                or (
                    e2_7_mode
                    and not bool(snapshot.get("e2_3_failure_replay_pass", False))
                )
                or (
                    e2_7_mode
                    and not bool(
                        snapshot.get("canonical_development_quality_pass", False)
                    )
                )
                or (
                    e2_7_mode
                    and not bool(snapshot.get("heldout_hv_pass", False))
                )
            )
            event = "rejected" if hard_rejection else "not_promoted"
            reason = (
                "safety_failed"
                if not safety_pass
                else "coverage_failed"
                if strict_pareto_mode and not coverage_pass
                else "controllability_failed"
                if strict_pareto_mode and not controllability_pass
                else "worker_direct_preference_not_zero"
                if strict_pareto_mode and not worker_direct_preference_pass
                else "preference_response_direction_failed"
                if (e2_4_mode or e2_5_mode or e2_6_mode or e2_7_mode) and not preference_response_pass
                else "low_flow_safety_failed"
                if (e2_5_mode or e2_6_mode or e2_7_mode) and not bool(snapshot.get("low_flow_safety_pass", False))
                else "counterfactual_gate_failed"
                if e2_6_mode and not bool(snapshot.get("counterfactual_gate_pass", False))
                else "centered_gate_failed"
                if e2_7_mode and not bool(snapshot.get("centered_gate_pass", False))
                else "e2_3_failure_replay_failed"
                if e2_7_mode
                and not bool(snapshot.get("e2_3_failure_replay_pass", False))
                else "canonical_heuristic_guard_failed"
                if e2_7_mode
                and not bool(snapshot.get("canonical_development_quality_pass", False))
                else "heldout_hv_guard_failed"
                if e2_7_mode
                and not bool(snapshot.get("heldout_hv_pass", False))
                else "awaiting_second_consecutive_full_grid"
                if e2_7_mode
                and self.development_consecutive_full_grid_passes == 1
                else "hypervolume_not_improved"
                if not hv_pass
                else "canonical_quality_guard_failed"
            )
            if event == "rejected":
                self.rejected_quality_updates += 1
            else:
                self.not_promoted_quality_updates += 1
        self.last_promotion_diagnostics = {
            "promotion_mode": self.quality_checkpoint_promotion,
            "promotion_event": event,
            "promotion_decision_reason": reason,
            "promotion_safety_pass": safety_pass,
            "promotion_coverage_pass": coverage_pass,
            "promotion_controllability_pass": controllability_pass,
            "promotion_worker_direct_preference_pass": (
                worker_direct_preference_pass
            ),
            "promotion_preference_response_pass": preference_response_pass,
            "promotion_low_flow_safety_pass": bool(snapshot.get("low_flow_safety_pass", True)),
            "promotion_counterfactual_gate_pass": bool(
                snapshot.get("counterfactual_gate_pass", True)
            ),
            "promotion_centered_gate_pass": bool(
                snapshot.get("centered_gate_pass", True)
            ),
            "promotion_e2_3_failure_replay_pass": bool(
                snapshot.get("e2_3_failure_replay_pass", True)
            ),
            "promotion_canonical_development_quality_pass": bool(
                snapshot.get("canonical_development_quality_pass", True)
            ),
            "promotion_heldout_hv_pass": bool(
                snapshot.get("heldout_hv_pass", True)
            ),
            "development_consecutive_full_grid_passes": (
                self.development_consecutive_full_grid_passes
            ),
            "promotion_counterfactual_instance_coverage": int(
                snapshot.get("counterfactual_instance_coverage", 0)
            ),
            "promotion_counterfactual_high_flow_flip_rate": float(
                snapshot.get("counterfactual_high_flow_commit_flip_rate", 0.0)
            ),
            "promotion_unique_action_trace_pass": bool(
                snapshot.get("unique_action_trace_pass", True)
            ),
            "promotion_unique_objective_pass": bool(
                snapshot.get("unique_objective_pass", True)
            ),
            "promotion_nondominated_pass": bool(
                snapshot.get("nondominated_pass", True)
            ),
            "promotion_missing_candidate_count": int(
                snapshot.get("missing_candidate_count", 0)
            ),
            "promotion_unexpected_candidate_count": int(
                snapshot.get("unexpected_candidate_count", 0)
            ),
            "promotion_duplicate_candidate_count": int(
                snapshot.get("duplicate_candidate_count", 0)
            ),
            "promotion_mean_unique_action_trace_count": float(
                snapshot.get("mean_unique_action_trace_count", 0.0)
            ),
            "promotion_mean_unique_objective_count": float(
                snapshot.get("mean_unique_objective_count", 0.0)
            ),
            "promotion_mean_nondominated_count": float(
                snapshot.get("mean_nondominated_count", 0.0)
            ),
            "promotion_hv_pass": hv_pass,
            "promotion_canonical_guard_pass": canonical_pass,
            "promotion_candidate_hv": candidate_hv,
            "promotion_anchor_hv": anchor_hv,
            "promotion_candidate_canonical_quality": candidate_canonical,
            "promotion_anchor_canonical_quality": anchor_canonical,
            "promotion_anchor_episode": self.accepted_quality_episode,
        }
        return event

    def evaluate_final_pareto_snapshot(
        self, snapshot: dict[str, object]
    ) -> dict[str, object]:
        """Pure final acceptance check for the tiered E2.4--E2.7 protocol."""

        mode = self.quality_checkpoint_promotion
        e2_4_mode = mode == "pareto_guarded_e2_4_v1"
        e2_5_mode = mode == "pareto_guarded_e2_5_v1"
        e2_6_mode = mode == "pareto_guarded_e2_6_v1"
        e2_7_mode = mode == "pareto_guarded_e2_7_development_v1"
        strict = e2_4_mode or e2_5_mode or e2_6_mode or e2_7_mode
        candidate_hv = float(snapshot.get("mean_hypervolume", math.nan))
        candidate_canonical = float(snapshot.get("canonical_quality", math.nan))
        reference_hv = self.reference_e1_pareto_hv
        reference_canonical = self.reference_e1_canonical_quality
        baseline = reference_hv is None or reference_canonical is None
        finite = math.isfinite(candidate_hv) and math.isfinite(candidate_canonical)
        hv_pass = bool(
            finite
            and (
                baseline
                or candidate_hv
                >= float(reference_hv)
                + self.quality_promotion_constraints["minimum_hv_improvement"]
                - 1e-12
            )
        )
        canonical_quality_pass = bool(
            finite
            and (
                baseline
                or candidate_canonical
                <= float(reference_canonical)
                * (
                    1.0
                    + self.quality_promotion_constraints[
                        "canonical_relative_tolerance"
                    ]
                )
                + self.quality_promotion_constraints[
                    "canonical_absolute_tolerance"
                ]
                + 1e-12
            )
        )
        checks = {
            "evaluation_integrity_pass": bool(
                snapshot.get("evaluation_integrity_pass", False)
            ),
            "physical_safety_pass": bool(snapshot.get("physical_safety_pass", False)),
            "completion_pass": bool(snapshot.get("completion_pass", False)),
            "coverage_pass": bool(snapshot.get("coverage_pass", False)),
            "controllability_pass": bool(snapshot.get("controllability_pass", False)),
            "worker_direct_preference_pass": bool(
                snapshot.get("worker_direct_preference_pass", False)
            ),
            "preference_response_pass": bool(
                snapshot.get("preference_response_pass", not strict)
            ),
            "low_flow_safety_pass": bool(
                snapshot.get("low_flow_safety_pass", not (e2_5_mode or e2_6_mode or e2_7_mode))
            ),
            "counterfactual_gate_pass": bool(
                snapshot.get("counterfactual_gate_pass", not e2_6_mode)
            ),
            "centered_gate_pass": bool(
                snapshot.get("centered_gate_pass", not e2_7_mode)
            ),
            "e2_3_failure_replay_pass": bool(
                snapshot.get("e2_3_failure_replay_pass", not e2_7_mode)
            ),
            "canonical_development_quality_pass": bool(
                snapshot.get("canonical_development_quality_pass", not e2_7_mode)
            ),
            "heldout_hv_pass": bool(
                snapshot.get("heldout_hv_pass", not e2_7_mode)
            ),
            "hypervolume_pass": hv_pass,
            "canonical_quality_pass": canonical_quality_pass,
        }
        required = (
            checks["evaluation_integrity_pass"]
            and checks["physical_safety_pass"]
            and checks["completion_pass"]
            and (
                not strict
                or (
                    checks["coverage_pass"]
                    and checks["controllability_pass"]
                    and checks["worker_direct_preference_pass"]
                )
            )
            and (
                not (e2_4_mode or e2_5_mode or e2_6_mode or e2_7_mode)
                or checks["preference_response_pass"]
            )
            and (
                not (e2_5_mode or e2_6_mode or e2_7_mode)
                or checks["low_flow_safety_pass"]
            )
            and (not e2_6_mode or checks["counterfactual_gate_pass"])
            and (not e2_7_mode or checks["centered_gate_pass"])
            and (not e2_7_mode or checks["e2_3_failure_replay_pass"])
            and (
                not e2_7_mode
                or checks["canonical_development_quality_pass"]
            )
            and (not e2_7_mode or checks["heldout_hv_pass"])
            and checks["hypervolume_pass"]
            and checks["canonical_quality_pass"]
        )
        failed_checks = [name for name, passed in checks.items() if not passed]
        return {
            "pass": bool(required),
            "checks": checks,
            "failed_checks": failed_checks,
            "reference_hypervolume": reference_hv,
            "reference_canonical_quality": reference_canonical,
        }

    def _observe_balanced_greedy_candidate(
        self,
        *,
        completion_rate: float,
        completed_episodes: int,
        score: tuple[float, float, float, float] | None,
        normalized_quality_score: float | None,
    ) -> str:
        anchor = self.accepted_normalized_quality_score
        anchor_score = self.accepted_quality_score
        finite = bool(
            score is not None
            and len(score) == 4
            and normalized_quality_score is not None
            and math.isfinite(float(normalized_quality_score))
            and all(math.isfinite(float(value)) for value in score)
            and anchor is not None
            and math.isfinite(float(anchor))
            and anchor_score is not None
            and len(anchor_score) == 4
        )
        constraints = self.quality_promotion_constraints
        completion_pass = completion_rate >= 1.0 - 1e-12
        quality_pass = bool(
            finite
            and float(normalized_quality_score)
            <= float(anchor)
            - constraints["minimum_normalized_quality_improvement"]
            + 1e-12
        )
        variance_pass = bool(
            finite
            and float(score[3])
            <= float(anchor_score[3])
            * (1.0 + constraints["variance_relative_tolerance"])
            + 1e-12
        )
        checks = (
            (completion_pass, "greedy_completion_below_one"),
            (finite, "missing_or_non_finite_promotion_metric"),
            (quality_pass, "normalized_quality_not_improved"),
            (variance_pass, "variance_regressed"),
        )
        reason = next(
            (failure for passed, failure in checks if not passed),
            "sampled_guard_pending",
        )
        event = (
            "sampled_guard_pending"
            if reason == "sampled_guard_pending"
            else "rejected"
        )
        if event == "sampled_guard_pending":
            self.pending_quality_score = score
            self.pending_normalized_quality_score = float(
                normalized_quality_score
            )
            self.pending_quality_episode = int(completed_episodes)
        else:
            self.rejected_quality_updates += 1
        self.last_promotion_diagnostics = {
            "promotion_mode": "balanced_guarded_v7",
            "promotion_event": event,
            "promotion_decision_reason": reason,
            "promotion_candidate_normalized_quality_score": (
                float(normalized_quality_score)
                if normalized_quality_score is not None
                and math.isfinite(float(normalized_quality_score))
                else None
            ),
            "promotion_anchor_normalized_quality_score": anchor,
            "promotion_anchor_episode": self.accepted_quality_episode,
            "promotion_completion_constraint_pass": completion_pass,
            "promotion_variance_constraint_pass": variance_pass,
            "promotion_normalized_quality_constraint_pass": quality_pass,
            "promotion_sampled_guard_executed": False,
        }
        return event

    def observe_sampled_guard(
        self,
        sampled: dict,
        *,
        completed_episodes: int,
        transition_anchor: bool = False,
    ) -> str:
        """Resolve the delayed sampled guard after greedy eligibility."""
        if self.quality_checkpoint_promotion != "balanced_guarded_v7":
            raise ValueError("sampled guard is only valid for balanced_guarded_v7")
        completion = float(
            sampled.get(
                "minimum_repeat_completion_rate",
                sampled.get("completion_rate", float("nan")),
            )
        )
        cvar = float(sampled.get("fatigue_cvar90", float("nan")))
        safe_pass = bool(sampled.get("fatigue_safe_line_pass", False))
        finite = math.isfinite(completion) and math.isfinite(cvar)
        if transition_anchor:
            passed = finite and safe_pass
            reason = "transition_anchor" if passed else "unsafe_transition_anchor"
            event = "transition" if passed else "rejected"
            if not passed:
                self.phase = "feasibility"
                self.phase_transition_episode = None
                self.consecutive_successes = 0
                self.accepted_quality_score = None
                self.accepted_normalized_quality_score = None
                self.accepted_quality_episode = None
        else:
            anchor_completion = self.accepted_sampled_completion_rate
            anchor_cvar = self.accepted_sampled_fatigue_cvar90
            constraints = self.quality_promotion_constraints
            completion_pass = bool(
                finite
                and anchor_completion is not None
                and completion
                >= anchor_completion
                - constraints["sampled_completion_drop_tolerance"]
                - 1e-12
            )
            fatigue_limit = (
                float(anchor_cvar)
                * (1.0 + constraints["fatigue_cvar_relative_tolerance"])
                if anchor_cvar is not None
                else float("nan")
            )
            fatigue_pass = bool(
                finite
                and anchor_cvar is not None
                and cvar <= fatigue_limit + 1e-12
            )
            checks = (
                (completion_pass, "sampled_completion_regressed"),
                (fatigue_pass, "sampled_fatigue_cvar_regressed"),
                (safe_pass, "sampled_fatigue_safety_violation"),
            )
            reason = next(
                (failure for passed, failure in checks if not passed),
                "accepted",
            )
            event = "accepted" if reason == "accepted" else "rejected"
            self.last_promotion_diagnostics.update(
                {
                    "promotion_sampled_completion_constraint_pass": completion_pass,
                    "promotion_sampled_fatigue_cvar_constraint_pass": fatigue_pass,
                }
            )
        self.last_promotion_diagnostics.update(
            {
                "promotion_event": event,
                "promotion_decision_reason": reason,
                "promotion_sampled_guard_executed": True,
                "promotion_candidate_sampled_completion_rate": completion,
                "promotion_anchor_sampled_completion_rate": self.accepted_sampled_completion_rate,
                "promotion_candidate_sampled_fatigue_cvar90": cvar,
                "promotion_anchor_sampled_fatigue_cvar90": self.accepted_sampled_fatigue_cvar90,
                "promotion_sampled_fatigue_safe_line_pass": safe_pass,
            }
        )
        if event in {"transition", "accepted"}:
            self.accepted_sampled_completion_rate = completion
            self.accepted_sampled_fatigue_cvar90 = cvar
            if event == "accepted":
                self.accepted_quality_score = self.pending_quality_score
                self.accepted_normalized_quality_score = (
                    self.pending_normalized_quality_score
                )
                self.accepted_quality_episode = int(completed_episodes)
                self.accepted_quality_updates += 1
        else:
            self.rejected_quality_updates += 1
        self.pending_quality_score = None
        self.pending_normalized_quality_score = None
        self.pending_quality_episode = None
        return event

    def _observe_constrained_candidate(
        self,
        *,
        completion_rate: float,
        completed_episodes: int,
        score: tuple[float, float, float, float] | None,
        normalized_quality_score: float | None,
    ) -> tuple[str, str]:
        anchor_score = self.accepted_quality_score
        anchor_normalized = self.accepted_normalized_quality_score
        anchor_episode = self.accepted_quality_episode
        values = (
            *(() if score is None else score),
            normalized_quality_score,
        )
        if (
            score is None
            or len(score) != 4
            or normalized_quality_score is None
            or anchor_score is None
            or len(anchor_score) != 4
            or anchor_normalized is None
            or not all(
                value is not None and math.isfinite(float(value))
                for value in values
            )
            or not all(math.isfinite(float(value)) for value in anchor_score)
            or not math.isfinite(float(anchor_normalized))
        ):
            reason = "missing_or_non_finite_promotion_metric"
            self._record_constrained_promotion_diagnostics(
                completion_rate=completion_rate,
                score=score,
                normalized_quality_score=normalized_quality_score,
                event="rejected",
                reason=reason,
                anchor_score=anchor_score,
                anchor_normalized_quality_score=anchor_normalized,
                anchor_episode=anchor_episode,
            )
            return "rejected", reason

        candidate_flow = float(score[1])
        candidate_cost = float(score[2])
        candidate_variance = float(score[3])
        anchor_flow = float(anchor_score[1])
        anchor_cost = float(anchor_score[2])
        anchor_variance = float(anchor_score[3])
        constraints = self.quality_promotion_constraints
        flow_limit = anchor_flow * (
            1.0 + constraints["flow_relative_tolerance"]
        )
        cost_limit = anchor_cost * (
            1.0 + constraints["cost_relative_tolerance"]
        )
        variance_limit = anchor_variance * (
            1.0 + constraints["variance_relative_tolerance"]
        )
        flow_pass = candidate_flow <= flow_limit or math.isclose(
            candidate_flow, flow_limit, rel_tol=1e-12, abs_tol=1e-12
        )
        cost_pass = candidate_cost <= cost_limit or math.isclose(
            candidate_cost, cost_limit, rel_tol=1e-12, abs_tol=1e-12
        )
        variance_pass = candidate_variance <= variance_limit or math.isclose(
            candidate_variance,
            variance_limit,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        normalized_pass = float(normalized_quality_score) < (
            float(anchor_normalized)
            - constraints["minimum_normalized_score_improvement"]
        )
        checks = (
            (flow_pass, "flow_tolerance_exceeded"),
            (cost_pass, "cost_regressed"),
            (variance_pass, "variance_regressed"),
            (normalized_pass, "normalized_quality_not_improved"),
        )
        reason = next(
            (failure_reason for passed, failure_reason in checks if not passed),
            "accepted",
        )
        event = "accepted" if reason == "accepted" else "rejected"
        self._record_constrained_promotion_diagnostics(
            completion_rate=completion_rate,
            score=score,
            normalized_quality_score=normalized_quality_score,
            event=event,
            reason=reason,
            anchor_score=anchor_score,
            anchor_normalized_quality_score=anchor_normalized,
            anchor_episode=anchor_episode,
            flow_pass=flow_pass,
            cost_pass=cost_pass,
            variance_pass=variance_pass,
            normalized_pass=normalized_pass,
        )
        if event == "accepted":
            self.accepted_quality_score = score
            self.accepted_normalized_quality_score = float(
                normalized_quality_score
            )
            self.accepted_quality_episode = int(completed_episodes)
        return event, reason

    def _record_constrained_promotion_diagnostics(
        self,
        *,
        completion_rate: float,
        score: tuple[float, float, float, float] | None,
        normalized_quality_score: float | None,
        event: str,
        reason: str,
        anchor_score: tuple[float, float, float, float] | None,
        anchor_normalized_quality_score: float | None,
        anchor_episode: int | None,
        flow_pass: bool | None = None,
        cost_pass: bool | None = None,
        variance_pass: bool | None = None,
        normalized_pass: bool | None = None,
    ) -> None:
        if self.quality_checkpoint_promotion != "constrained_weighted":
            return

        def objective(
            value: tuple[float, float, float, float] | None,
            index: int,
        ) -> float | None:
            if value is None or len(value) != 4:
                return None
            result = float(value[index])
            return result if math.isfinite(result) else None

        completion_pass = (
            math.isfinite(float(completion_rate))
            and float(completion_rate) >= self.quality_completion_floor
        )
        self.last_promotion_diagnostics = {
            "promotion_mode": self.quality_checkpoint_promotion,
            "promotion_event": event,
            "promotion_decision_reason": reason,
            "promotion_candidate_normalized_quality_score": (
                float(normalized_quality_score)
                if normalized_quality_score is not None
                and math.isfinite(float(normalized_quality_score))
                else None
            ),
            "promotion_anchor_normalized_quality_score": (
                float(anchor_normalized_quality_score)
                if anchor_normalized_quality_score is not None
                and math.isfinite(float(anchor_normalized_quality_score))
                else None
            ),
            "promotion_anchor_episode": anchor_episode,
            "promotion_candidate_flow_time_objective": objective(score, 1),
            "promotion_candidate_reconfiguration_cost": objective(score, 2),
            "promotion_candidate_worker_load_variance": objective(score, 3),
            "promotion_anchor_flow_time_objective": objective(anchor_score, 1),
            "promotion_anchor_reconfiguration_cost": objective(anchor_score, 2),
            "promotion_anchor_worker_load_variance": objective(anchor_score, 3),
            "promotion_completion_constraint_pass": completion_pass,
            "promotion_flow_constraint_pass": flow_pass,
            "promotion_cost_constraint_pass": cost_pass,
            "promotion_variance_constraint_pass": variance_pass,
            "promotion_normalized_quality_constraint_pass": normalized_pass,
        }

    @property
    def formal_training_status(self) -> str:
        if not self.enabled:
            return "legacy_weighted_sum"
        if self.phase_transition_episode is None:
            return "feasibility_not_reached"
        if (
            self.quality_checkpoint_promotion in PARETO_PROMOTION_MODES
            and self.accepted_pareto_hv is None
        ):
            return "pareto_baseline_not_reached"
        if (
            self.quality_checkpoint_promotion
            == SINGLE_OBJECTIVE_PROMOTION_MODE
            and self.accepted_single_objective_value is None
        ):
            return "single_objective_98_candidate_not_reached"
        if (
            self.quality_checkpoint_promotion
            == SINGLE_OBJECTIVE_PROMOTION_MODE
        ):
            return "accepted_98_experiment_candidate"
        return "quality_constrained"

    def as_dict(self) -> dict:
        result = {
            "enabled": self.enabled,
            "phase": self.phase,
            "completion_target": self.completion_target,
            "consecutive_validations_required": self.consecutive_required,
            "consecutive_validation_successes": self.consecutive_successes,
            "quality_completion_floor": self.quality_completion_floor,
            "quality_checkpoint_promotion": self.quality_checkpoint_promotion,
            "accepted_quality_score": self.accepted_quality_score,
            "phase_transition_episode": self.phase_transition_episode,
            "accepted_quality_updates": self.accepted_quality_updates,
            "rejected_quality_updates": self.rejected_quality_updates,
            "not_promoted_quality_updates": (
                self.not_promoted_quality_updates
            ),
            "formal_training_status": self.formal_training_status,
        }
        if self.quality_checkpoint_promotion in {
            "constrained_weighted",
            "aligned_quality",
            "balanced_guarded_v7",
            SINGLE_OBJECTIVE_PROMOTION_MODE,
            "pareto_guarded_e2_v1",
            "pareto_guarded_e2_3_v1",
            "pareto_guarded_e2_4_v1",
            "pareto_guarded_e2_5_v1",
            "pareto_guarded_e2_6_v1",
            "pareto_guarded_e2_7_development_v1",
        }:
            result.update(
                {
                    "accepted_normalized_quality_score": (
                        self.accepted_normalized_quality_score
                    ),
                    "accepted_quality_episode": self.accepted_quality_episode,
                    "quality_promotion_constraints": dict(
                        self.quality_promotion_constraints
                    ),
                    "last_promotion_diagnostics": dict(
                        self.last_promotion_diagnostics
                    ),
                    "accepted_sampled_completion_rate": (
                        self.accepted_sampled_completion_rate
                    ),
                    "accepted_sampled_fatigue_cvar90": (
                        self.accepted_sampled_fatigue_cvar90
                    ),
                    "accepted_pareto_hv": self.accepted_pareto_hv,
                    "accepted_pareto_canonical_quality": (
                        self.accepted_pareto_canonical_quality
                    ),
                    "reference_e1_pareto_hv": self.reference_e1_pareto_hv,
                    "reference_e1_canonical_quality": (
                        self.reference_e1_canonical_quality
                    ),
                    "development_consecutive_full_grid_passes": (
                        self.development_consecutive_full_grid_passes
                    ),
                    "single_objective_name": self.single_objective_name,
                    "accepted_single_objective_value": (
                        self.accepted_single_objective_value
                    ),
                    "single_objective_window_size": self.single_objective_window_size,
                    "single_objective_window_statistic": self.single_objective_window_statistic,
                    "single_objective_rollback_below_floor_consecutive": self.single_objective_rollback_below_floor_consecutive,
                    "single_objective_candidate_improvement_epsilon": self.single_objective_candidate_improvement_epsilon,
                    "single_objective_audit_instance_limit": self.single_objective_audit_instance_limit,
                    "single_objective_audit_completion_target": self.single_objective_audit_completion_target,
                    "single_objective_audit_max_failed_instances": self.single_objective_audit_max_failed_instances,
                    "single_objective_window_values": list(self.single_objective_window_values),
                    "single_objective_window_episodes": list(self.single_objective_window_episodes),
                    "single_objective_candidate_anchor_value": self.single_objective_candidate_anchor_value,
                    "single_objective_candidate_episode": self.single_objective_candidate_episode,
                    "single_objective_audit_count": self.single_objective_audit_count,
                    "accepted_single_objective_failed_instances": self.accepted_single_objective_failed_instances,
                    "accepted_single_objective_window_value": self.accepted_single_objective_window_value,
                    "accepted_single_objective_audit_value": self.accepted_single_objective_audit_value,
                    "last_single_objective_audit_diagnostics": dict(
                        self.last_single_objective_audit_diagnostics
                    ),
                }
            )
        return result


@dataclass
class ParetoSafetyGuard:
    """Track E2.4 multi-preference safety failures independently of canonical validation."""

    consecutive_failures_required: int
    learning_rate_decay_factor: float
    minimum_learning_rate: float
    consecutive_failures: int = 0
    warning_count: int = 0
    rollback_count: int = 0
    last_event: str = "not_started"
    last_scope: str | None = None
    last_failure_reason: str | None = None
    last_rollback_source: str | None = None
    tiered_protocol: bool = False

    @classmethod
    def from_config(cls, config: dict) -> "ParetoSafetyGuard | None":
        promotion = str(
            config["training"]["two_stage"].get(
                "quality_checkpoint_promotion", ""
            )
        )
        if promotion not in {
            "pareto_guarded_e2_4_v1",
            "pareto_guarded_e2_5_v1",
            "pareto_guarded_e2_6_v1",
            "pareto_guarded_e2_7_development_v1",
        }:
            return None
        settings = _pareto_promotion_settings(config)
        tiered = _uses_tiered_training_gates(config)
        return cls(
            consecutive_failures_required=int(
                1 if tiered else settings["safety_guard_consecutive_failures"]
            ),
            learning_rate_decay_factor=float(
                settings["safety_guard_learning_rate_decay_factor"]
            ),
            minimum_learning_rate=float(
                settings["safety_guard_minimum_learning_rate"]
            ),
            tiered_protocol=tiered,
        )

    def observe(self, snapshot: dict[str, object]) -> str:
        scope = str(snapshot.get("scope", ""))
        safety_pass = bool(
            snapshot.get("physical_safety_pass", False)
            if self.tiered_protocol
            else snapshot.get("coverage_pass", False)
            and snapshot.get("all_safe", False)
        )
        self.last_scope = scope
        self.last_failure_reason = None
        if safety_pass:
            self.consecutive_failures = 0
            self.last_event = "safe"
            return self.last_event
        self.consecutive_failures += 1
        if self.tiered_protocol and not bool(
            snapshot.get("physical_safety_pass", False)
        ):
            self.last_failure_reason = "physical_safety_failed"
        elif not bool(snapshot.get("coverage_pass", False)):
            self.last_failure_reason = "coverage_failed"
        elif not bool(snapshot.get("all_safe", False)):
            self.last_failure_reason = "safety_failed"
        else:
            self.last_failure_reason = "unknown_safety_guard_failure"
        if self.consecutive_failures >= self.consecutive_failures_required:
            self.rollback_count += 1
            self.last_event = "rollback"
            return self.last_event
        self.warning_count += 1
        self.last_event = "warning"
        return self.last_event

    def record_rollback(self, source: str) -> None:
        self.last_rollback_source = str(source)
        self.consecutive_failures = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "consecutive_failures_required": (
                self.consecutive_failures_required
            ),
            "learning_rate_decay_factor": self.learning_rate_decay_factor,
            "minimum_learning_rate": self.minimum_learning_rate,
            "consecutive_failures": self.consecutive_failures,
            "warning_count": self.warning_count,
            "rollback_count": self.rollback_count,
            "last_event": self.last_event,
            "last_scope": self.last_scope,
            "last_failure_reason": self.last_failure_reason,
            "last_rollback_source": self.last_rollback_source,
            "tiered_protocol": self.tiered_protocol,
        }


@dataclass
class ValidationStabilityController:
    rollback_completion_drop: float
    rollback_consecutive_required: int
    rollback_cooldown_validations: int
    plateau_patience: int
    decay_factor: float
    minimum_learning_rate: float
    sampled_every: int
    sampled_repeats: int
    sampled_seed_offset: int
    sampled_episode_milestones: tuple[int, ...] | None
    current_learning_rate: float
    rollback_completion_floor: float | None = None
    best_score: tuple[float, float, float, float] | None = None
    best_completion_rate: float | None = None
    best_episode: int | None = None
    validations_without_improvement: int = 0
    feasibility_rollbacks: int = 0
    learning_rate_decays: int = 0
    validation_count: int = 0
    sampled_validation_runs: int = 0
    consecutive_degraded_validations: int = 0
    rollback_cooldown_remaining: int = 0
    rollback_cooldown_validation_count: int = 0
    rollback_cooldown_blocked_count: int = 0

    @classmethod
    def from_config(cls, config: dict) -> "ValidationStabilityController":
        settings = config["training"]["validation_control"]
        rollback_drop = float(
            settings["feasibility_rollback"]["completion_drop"]
        )
        rollback_consecutive = int(
            settings["feasibility_rollback"].get(
                "consecutive_validations", 1
            )
        )
        rollback_cooldown = int(
            settings["feasibility_rollback"].get(
                "cooldown_validations", 0
            )
        )
        plateau = settings["learning_rate_plateau"]
        patience = int(plateau["patience_validations"])
        factor = float(plateau["factor"])
        minimum = float(plateau["minimum"])
        sampled = settings["sampled"]
        sampled_every = int(sampled["every_validations"])
        sampled_repeats = int(sampled["repeats"])
        seed_offset = int(sampled["seed_offset"])
        raw_milestones = sampled.get("episode_milestones")
        milestones = (
            None
            if raw_milestones is None
            else tuple(sorted({int(value) for value in raw_milestones}))
        )
        initial_learning_rate = float(config["ppo"]["learning_rate"])
        promotion_mode = str(
            config["training"]["two_stage"].get(
                "quality_checkpoint_promotion", ""
            )
        ).strip().lower()
        rollback_floor = None
        if promotion_mode == SINGLE_OBJECTIVE_PROMOTION_MODE:
            rollback_floor = float(
                config["training"]["two_stage"].get(
                    "quality_completion_floor", 0.95
                )
            )
            single_settings = config["training"]["two_stage"].get(
                "single_objective_promotion", {}
            )
            rollback_consecutive = int(
                single_settings.get(
                    "rollback_below_floor_consecutive", rollback_consecutive
                )
            )
        if not 0.0 < rollback_drop <= 1.0:
            raise ValueError(
                "feasibility rollback completion_drop must be in (0, 1]"
            )
        if rollback_consecutive < 1 or rollback_cooldown < 0:
            raise ValueError(
                "rollback consecutive validations must be positive and "
                "cooldown must be non-negative"
            )
        if patience < 1:
            raise ValueError(
                "learning-rate plateau patience must be positive"
            )
        if not 0.0 < factor < 1.0:
            raise ValueError(
                "learning-rate plateau factor must be in (0, 1)"
            )
        if minimum <= 0.0 or minimum > initial_learning_rate:
            raise ValueError(
                "minimum learning rate must be positive and no greater "
                "than the initial learning rate"
            )
        if sampled_every < 1 or sampled_repeats < 1:
            raise ValueError(
                "sampled validation cadence and repeats must be positive"
            )
        if milestones is not None and any(value < 1 for value in milestones):
            raise ValueError("sampled validation milestones must be positive")
        return cls(
            rollback_completion_drop=rollback_drop,
            rollback_consecutive_required=rollback_consecutive,
            rollback_cooldown_validations=rollback_cooldown,
            plateau_patience=patience,
            decay_factor=factor,
            minimum_learning_rate=minimum,
            sampled_every=sampled_every,
            sampled_repeats=sampled_repeats,
            sampled_seed_offset=seed_offset,
            sampled_episode_milestones=milestones,
            current_learning_rate=initial_learning_rate,
            rollback_completion_floor=rollback_floor,
        )

    def observe_greedy(
        self,
        score: tuple[float, float, float, float],
        completion_rate: float,
        *,
        completed_episodes: int,
        feasibility_phase: bool,
    ) -> dict[str, object]:
        self.validation_count += 1
        rate = float(completion_rate)
        cooldown_active = self.rollback_cooldown_remaining > 0
        if cooldown_active:
            self.rollback_cooldown_validation_count += 1
            self.rollback_cooldown_remaining -= 1
        improved = self.best_score is None or score < self.best_score
        if improved:
            self.best_score = score
            self.best_completion_rate = rate
            self.best_episode = int(completed_episodes)
            self.validations_without_improvement = 0
            self.consecutive_degraded_validations = 0
        else:
            self.validations_without_improvement += 1
        if self.rollback_completion_floor is not None:
            degraded = bool(
                not feasibility_phase
                and rate < self.rollback_completion_floor
            )
        else:
            degraded = bool(
                not improved
                and rate
                <= 1.0 - self.rollback_completion_drop + 1e-12
            )
        if degraded:
            self.consecutive_degraded_validations += 1
        elif self.rollback_completion_floor is not None or not improved:
            self.consecutive_degraded_validations = 0
        rollback_ready = bool(
            degraded
            and self.consecutive_degraded_validations
            >= self.rollback_consecutive_required
        )
        if rollback_ready and cooldown_active:
            self.rollback_cooldown_blocked_count += 1
        rollback = rollback_ready and not cooldown_active
        if rollback:
            self.feasibility_rollbacks += 1
            self.consecutive_degraded_validations = 0
            self.rollback_cooldown_remaining = (
                self.rollback_cooldown_validations
            )
            self.validations_without_improvement = 0
        previous_learning_rate = self.current_learning_rate
        decay_applied = False
        if (
            not improved
            and not rollback
            and self.validations_without_improvement
            >= self.plateau_patience
        ):
            next_learning_rate = max(
                self.minimum_learning_rate,
                self.current_learning_rate * self.decay_factor,
            )
            if next_learning_rate < self.current_learning_rate - 1e-15:
                self.current_learning_rate = next_learning_rate
                self.learning_rate_decays += 1
                decay_applied = True
            self.validations_without_improvement = 0
        return {
            "improved": improved,
            "rollback": rollback,
            "degraded": degraded,
            "consecutive_degraded_validations": (
                self.consecutive_degraded_validations
            ),
            "rollback_cooldown_remaining": (
                self.rollback_cooldown_remaining
            ),
            "rollback_cooldown_validation_count": (
                self.rollback_cooldown_validation_count
            ),
            "rollback_cooldown_blocked_count": (
                self.rollback_cooldown_blocked_count
            ),
            "best_completion_rate": self.best_completion_rate,
            "best_episode": self.best_episode,
            "validations_without_improvement": (
                self.validations_without_improvement
            ),
            "learning_rate_before_validation": previous_learning_rate,
            "learning_rate_after_validation": self.current_learning_rate,
            "learning_rate_decay_applied": decay_applied,
        }

    def reset_plateau(self) -> None:
        self.validations_without_improvement = 0
        self.consecutive_degraded_validations = 0
        self.rollback_cooldown_remaining = 0

    def should_run_sampled(
        self,
        *,
        final_validation: bool,
        completed_episodes: int,
    ) -> bool:
        if final_validation:
            return True
        if self.sampled_episode_milestones is not None:
            return int(completed_episodes) in self.sampled_episode_milestones
        return self.validation_count % self.sampled_every == 0

    def sampled_seeds(self, algorithm_seed: int) -> list[int]:
        return [
            int(algorithm_seed) + self.sampled_seed_offset + repeat
            for repeat in range(self.sampled_repeats)
        ]

    def as_dict(self) -> dict[str, object]:
        return {
            "rollback_completion_drop": self.rollback_completion_drop,
            "rollback_completion_floor": self.rollback_completion_floor,
            "rollback_consecutive_validations": (
                self.rollback_consecutive_required
            ),
            "rollback_cooldown_validations": (
                self.rollback_cooldown_validations
            ),
            "rollback_cooldown_remaining": (
                self.rollback_cooldown_remaining
            ),
            "rollback_cooldown_validation_count": (
                self.rollback_cooldown_validation_count
            ),
            "rollback_cooldown_blocked_count": (
                self.rollback_cooldown_blocked_count
            ),
            "consecutive_degraded_validations": (
                self.consecutive_degraded_validations
            ),
            "plateau_patience_validations": self.plateau_patience,
            "learning_rate_decay_factor": self.decay_factor,
            "minimum_learning_rate": self.minimum_learning_rate,
            "current_learning_rate": self.current_learning_rate,
            "best_completion_rate": self.best_completion_rate,
            "best_episode": self.best_episode,
            "validations_without_improvement": (
                self.validations_without_improvement
            ),
            "feasibility_rollbacks": self.feasibility_rollbacks,
            "learning_rate_decays": self.learning_rate_decays,
            "greedy_validation_runs": self.validation_count,
            "sampled_validation_runs": self.sampled_validation_runs,
            "sampled_every_validations": self.sampled_every,
            "sampled_repeats": self.sampled_repeats,
            "sampled_episode_milestones": self.sampled_episode_milestones,
        }


def _collect_serial_batch(
    *,
    config: dict,
    agent: PPOAgent,
    environment: AssemblySchedulingEnv,
    instance,
    record: GeneratedInstanceRecord | None,
    episode_index: int,
    sampling_start: float,
    generation_time_seconds: float,
    step_limit: int | None,
    reward_phase: str | None = None,
) -> TrainingRolloutBatch:
    forced_action_compression = bool(
        config["training"].get("forced_action_compression", False)
    )
    gamma = float(config["ppo"]["gamma"])
    if forced_action_compression and gamma != 1.0:
        raise ValueError(
            "forced action compression requires ppo.gamma = 1.0"
        )
    effective_reward_phase = (
        "feasibility"
        if reward_phase is None
        and str(
            config["reward"].get("mode", "legacy_weighted_sum")
        )
        == "hierarchical_constrained_v1"
        else "legacy"
        if reward_phase is None
        else str(reward_phase)
    )
    preference, preference_source = sample_episode_preference(
        config,
        algorithm_seed=int(config["seed"]),
        episode_index=int(episode_index),
    )
    observation = (
        environment.reset(instance, preference=preference)
        if preference_enabled(config)
        else environment.reset(instance)
    )
    buffer = RolloutBuffer(
        preserve_graph=agent.requires_graph_observation
    )
    action_generator = (
        torch.Generator(device=agent.device).manual_seed(
            derive_episode_action_seed(
                int(config["seed"]),
                int(episode_index),
            )
        )
        if preference_enabled(config)
        else None
    )
    step_count = 0
    reward_sum = 0.0
    reward_components = {
        "flow": 0.0,
        "cost": 0.0,
        "variance": 0.0,
        "completion_progress": 0.0,
        "completion_bonus": 0.0,
        "quality": 0.0,
        "truncation": 0.0,
        "unfinished": 0.0,
        "feasibility_shaping": 0.0,
        **(
            {"defer_risk_shaping": 0.0}
            if bool(
                config.get("environment", {})
                .get("production_defer", {})
                .get("shield", {})
                .get("enabled", False)
            )
            else {}
        ),
    }
    inference_time = 0.0
    environment_step_time = 0.0
    forced_action_count = 0
    policy_step_count = 0
    pending_transition: dict | None = None
    unattributed_forced_reward = 0.0

    def commit_pending(*, done: bool) -> None:
        nonlocal pending_transition
        if pending_transition is None:
            return
        buffer.add(
            pending_transition["observation"],
            pending_transition["action_mask"],
            pending_transition["action"],
            pending_transition["log_probability"],
            pending_transition["value"],
            pending_transition["reward"],
            done,
        )
        pending_transition = None

    while not (environment.terminated or environment.truncated):
        action_mask = environment.get_action_mask()
        forced_action = (
            forced_action_from_mask(action_mask)
            if forced_action_compression
            else None
        )
        sampled_policy_action = forced_action is None
        if sampled_policy_action:
            commit_pending(done=False)
            inference_start = time.perf_counter()
            if action_generator is None:
                action, log_probability, value = agent.act(
                    observation,
                    action_mask,
                )
            else:
                action, log_probability, value = agent.act(
                    observation,
                    action_mask,
                    generator=action_generator,
                )
            inference_time += time.perf_counter() - inference_start
            policy_step_count += 1
            pending_transition = {
                "observation": observation,
                "action_mask": action_mask,
                "action": action,
                "log_probability": log_probability,
                "value": value,
                "reward": unattributed_forced_reward,
            }
            unattributed_forced_reward = 0.0
        else:
            action = forced_action
            forced_action_count += 1
        step_start = time.perf_counter()
        next_observation, reward_vector, terminated, truncated, _ = (
            environment.step(action)
        )
        environment_step_time += time.perf_counter() - step_start
        scalar_reward = reward_vector.scalarize(
            config["reward"],
            effective_reward_phase,
        )
        if pending_transition is not None:
            pending_transition["reward"] += scalar_reward
        else:
            unattributed_forced_reward += scalar_reward
        observation = next_observation
        reward_sum += scalar_reward
        for name, value in reward_vector.as_dict().items():
            reward_components[name] += float(value)
        step_count += 1
        if terminated or truncated:
            commit_pending(done=True)
        if step_limit is not None and step_count >= step_limit:
            if not (terminated or truncated):
                commit_pending(done=False)
            break
    last_value = 0.0
    if (
        len(buffer) > 0
        and not (environment.terminated or environment.truncated)
    ):
        inference_start = time.perf_counter()
        last_value = agent.value(
            observation,
            environment.get_action_mask(),
        )
        inference_time += time.perf_counter() - inference_start
    buffer.compute_gae(
        last_value=last_value,
        gamma=gamma,
        gae_lambda=float(config["ppo"]["gae_lambda"]),
    )
    if pending_transition is not None:
        raise RuntimeError("episode ended with an uncommitted transition")
    if step_count != policy_step_count + forced_action_count:
        raise RuntimeError("compressed rollout step accounting diverged")
    if len(buffer) != policy_step_count:
        raise RuntimeError("compressed rollout buffer accounting diverged")
    attributed_reward = sum(
        transition.reward for transition in buffer.transitions
    )
    reward_attribution_error = (
        attributed_reward + unattributed_forced_reward - reward_sum
    )
    if abs(reward_attribution_error) > 1e-8:
        raise RuntimeError(
            "compressed rollout reward attribution diverged by "
            f"{reward_attribution_error}"
        )
    metrics = environment.metrics()
    metrics["schedule_violations"] = environment.validate_schedule()
    metadata = (
        {
            key: record.metadata.get(key)
            for key in ("seed", "pressure_type", "cost_profile")
        }
        if record is not None
        else {
            "seed": None,
            "pressure_type": "fixed",
            "cost_profile": "fixed",
        }
    )
    episode = EpisodeRollout(
        episode_index=episode_index,
        instance_id=instance.instance_id,
        metadata=metadata,
        buffer=buffer,
        reward_sum=reward_sum,
        step_count=step_count,
        metrics=metrics,
        generation_time_seconds=generation_time_seconds,
        environment_step_time_seconds=environment_step_time,
        preference=preference,
        preference_source=preference_source,
        reward_phase=effective_reward_phase,
        reward_components=reward_components,
        expected_reward=proxy_return_from_metrics(
            metrics,
            config["reward"],
            effective_reward_phase,
            preference=preference,
        ),
        unattributed_forced_reward=unattributed_forced_reward,
    )
    reward_identity_tolerance = float(
        config["training"]
        .get("ablation_gate", {})
        .get("reward_identity_tolerance", 1e-8)
    )
    reward_identity_error = episode.base_reward_sum - episode.expected_reward
    if abs(reward_identity_error) > reward_identity_tolerance:
        raise RuntimeError(
            "trajectory reward identity diverged by "
            f"{reward_identity_error}"
        )
    return TrainingRolloutBatch(
        episodes=[episode],
        buffer=buffer,
        sampling_wall_time_seconds=(
            time.perf_counter() - sampling_start
        ),
        policy_inference_time_seconds=inference_time,
    )


def _validation_log_row(
    validation: dict,
    *,
    completed_episodes: int,
) -> dict:
    completed_summary = validation["completed_metrics"]
    all_summary = validation["all_instance_metrics"]
    gap_summary = validation["gap_metrics"]

    def summary_value(
        summary: dict,
        name: str,
        statistic: str,
    ):
        metric = summary.get(name)
        return metric.get(statistic) if metric is not None else None

    return {
        "episode": completed_episodes,
        "dataset": validation["dataset"],
        "instance_count": validation["instance_count"],
        "completed_count": validation["completed_count"],
        "completion_rate": validation["completion_rate"],
        "truncated_count": validation["truncated_count"],
        "schedule_violation_count": validation.get(
            "schedule_violation_count", 0
        ),
        "physical_safety_pass": validation.get("physical_safety_pass", True),
        "window_size": None,
        "window_statistic": None,
        "window_count": None,
        "window_objective_values": None,
        "window_objective_episodes": None,
        "window_objective_statistic": None,
        "candidate_anchor_value": None,
        "previous_candidate_anchor_value": None,
        "accepted_window_median": None,
        "accepted_failed_instance_count": None,
        "audit_required": False,
        "audit_event": None,
        "audit_failed_instance_count": None,
        "audit_completion_rate": None,
        "accepted_checkpoint_episode": None,
        "ranker_top_decision_count": validation.get(
            "ranker_top_decision_count", 0
        ),
        "preference_override_count": validation.get(
            "preference_override_count", 0
        ),
        "preference_override_rate": validation.get(
            "preference_override_rate", 0.0
        ),
        "mean_preference_logit_std": validation.get(
            "mean_preference_logit_std", 0.0
        ),
        "production_ranker_top_decision_count": validation.get(
            "production_ranker_top_decision_count", 0
        ),
        "production_preference_override_count": validation.get(
            "production_preference_override_count", 0
        ),
        "production_preference_override_rate": validation.get(
            "production_preference_override_rate", 0.0
        ),
        "production_mean_preference_logit_std": validation.get(
            "production_mean_preference_logit_std", 0.0
        ),
        "worker_ranker_top_decision_count": validation.get(
            "worker_ranker_top_decision_count", 0
        ),
        "worker_preference_override_count": validation.get(
            "worker_preference_override_count", 0
        ),
        "worker_preference_override_rate": validation.get(
            "worker_preference_override_rate", 0.0
        ),
        "worker_mean_preference_logit_std": validation.get(
            "worker_mean_preference_logit_std", 0.0
        ),
        "current_worker_matching_deficit": validation.get(
            "current_worker_matching_deficit", 0
        ),
        "maximum_worker_matching_deficit": validation.get(
            "maximum_worker_matching_deficit", 0
        ),
        "deficit_reducing_worker_action_candidate_count": validation.get(
            "deficit_reducing_worker_action_candidate_count", 0
        ),
        "deficit_reducing_worker_action_count": validation.get(
            "deficit_reducing_worker_action_count", 0
        ),
        "matching_deficit_recovery_advance_count": validation.get(
            "matching_deficit_recovery_advance_count", 0
        ),
        "current_matching_admission_masked_action_count": validation.get(
            "current_matching_admission_masked_action_count", 0
        ),
        "future_installation_admission_candidate_count": validation.get(
            "future_installation_admission_candidate_count", 0
        ),
        "future_installation_admission_masked_action_count": validation.get(
            "future_installation_admission_masked_action_count", 0
        ),
        "future_installation_admission_masked_action_ratio": validation.get(
            "future_installation_admission_masked_action_ratio", 0.0
        ),
        "future_installation_matching_deficit_after_commit": validation.get(
            "future_installation_matching_deficit_after_commit", 0
        ),
        "maximum_projected_installation_deficit": validation.get(
            "maximum_projected_installation_deficit", 0
        ),
        **{
            name: validation.get(name, 0)
            for name in MATCHING_RECOVERY_DIAGNOSTIC_FIELDS
        },
        **{
            name: validation.get(name, 0)
            for name in PREFERENCE_POLICY_DIAGNOSTIC_FIELDS
        },
        "mean_makespan": completed_summary["makespan"]["mean"],
        "std_makespan": completed_summary["makespan"]["std"],
        "mean_total_flow_time": completed_summary[
            "total_flow_time"
        ]["mean"],
        "std_total_flow_time": completed_summary[
            "total_flow_time"
        ]["std"],
        "mean_flow_time_objective": all_summary[
            "flow_time_objective"
        ]["mean"],
        "std_flow_time_objective": all_summary[
            "flow_time_objective"
        ]["std"],
        "mean_reconfiguration_cost": all_summary[
            "reconfiguration_cost"
        ]["mean"],
        "std_reconfiguration_cost": all_summary[
            "reconfiguration_cost"
        ]["std"],
        "mean_worker_load_variance": all_summary[
            "worker_load_variance"
        ]["mean"],
        "std_worker_load_variance": all_summary[
            "worker_load_variance"
        ]["std"],
        "mean_quality_score": summary_value(
            all_summary, "quality_score", "mean"
        ),
        "std_quality_score": summary_value(
            all_summary, "quality_score", "std"
        ),
        "mean_heuristic_quality_score": summary_value(
            all_summary, "heuristic_quality_score", "mean"
        ),
        "std_heuristic_quality_score": summary_value(
            all_summary, "heuristic_quality_score", "std"
        ),
        "mean_relative_heuristic_gap_percent": gap_summary[
            "relative_heuristic_gap_percent"
        ]["mean"],
        "std_relative_heuristic_gap_percent": gap_summary[
            "relative_heuristic_gap_percent"
        ]["std"],
        "mean_makespan_heuristic_gap_percent": summary_value(
            gap_summary,
            "makespan_heuristic_gap_percent",
            "mean",
        ),
        "std_makespan_heuristic_gap_percent": summary_value(
            gap_summary,
            "makespan_heuristic_gap_percent",
            "std",
        ),
        "mean_reconfiguration_cost_heuristic_gap_percent": summary_value(
            gap_summary,
            "reconfiguration_cost_heuristic_gap_percent",
            "mean",
        ),
        "std_reconfiguration_cost_heuristic_gap_percent": summary_value(
            gap_summary,
            "reconfiguration_cost_heuristic_gap_percent",
            "std",
        ),
        "mean_worker_load_variance_heuristic_gap_percent": summary_value(
            gap_summary,
            "worker_load_variance_heuristic_gap_percent",
            "mean",
        ),
        "std_worker_load_variance_heuristic_gap_percent": summary_value(
            gap_summary,
            "worker_load_variance_heuristic_gap_percent",
            "std",
        ),
        **{
            f"{statistic}_{name}": summary_value(
                all_summary,
                name,
                statistic,
            )
            for name in (
                "maximum_worker_fatigue",
                "mean_peak_worker_fatigue",
                "safe_fatigue_limit",
                "fatigue_masked_action_ratio",
                "worker_competition_event_count",
                "worker_matching_deficit_event_count",
                "resource_admission_masked_action_count",
                "resource_admission_masked_action_ratio",
                "minimum_worker_alternatives",
                "matching_preserving_worker_action_count",
                "candidate_recovery_advance_count",
                "machine_waiting_for_worker_time",
                "completed_reconfigurations",
                "worker_switch_ratio",
                "unfinished_orders",
                "feasibility_proxy_return",
            )
            for statistic in ("mean", "std")
        },
        "total_inference_time_seconds": validation[
            "total_inference_time_seconds"
        ],
        "total_solve_time_seconds": validation[
            "total_solve_time_seconds"
        ],
        "parallel_envs": validation.get("parallel_envs", 1),
    }


def _evaluate_sampled_validation(
    config: dict,
    *,
    dataset_name: str,
    ppo_agent: PPOAgent,
    instance_limit: int | None,
    sampling_seeds: list[int],
    runner: ParallelEpisodeRunner | None = None,
    use_parallel: bool = False,
) -> dict:
    all_rows: list[dict] = []
    reference: dict | None = None
    repeat_completion_rates: list[float] = []
    for sampling_seed in sampling_seeds:
        if use_parallel:
            if runner is None:
                raise ValueError("parallel sampled validation requires a runner")
            rows, aggregate = evaluate_dataset_parallel(
                config,
                dataset_name=dataset_name,
                ppo_agent=ppo_agent,
                runner=runner,
                instance_limit=instance_limit,
                decode_mode="sampled",
                sampling_seed=sampling_seed,
            )
        else:
            rows, _, _, aggregate = evaluate_dataset(
                config,
                dataset_name=dataset_name,
                policy_name="ppo",
                ppo_agent=ppo_agent,
                instance_limit=instance_limit,
                decode_mode="sampled",
                sampling_seed=sampling_seed,
            )
        all_rows.extend(rows)
        reference = aggregate
        repeat_completion_rates.append(float(aggregate["completion_rate"]))
    if reference is None:
        raise ValueError("sampled validation requires at least one seed")
    combined = aggregate_evaluation_rows(
        all_rows,
        dataset=dataset_name,
        policy="ppo",
        manifest=str(reference["manifest"]),
        schema_version=result_schema_version(config),
    )
    combined["decode_mode"] = "sampled"
    combined["parallel_envs"] = reference.get("parallel_envs", 1)
    combined["repeat_count"] = len(sampling_seeds)
    combined["unique_instance_count"] = (
        combined["instance_count"] // len(sampling_seeds)
    )
    combined["repeat_completion_rates"] = repeat_completion_rates
    combined["minimum_repeat_completion_rate"] = min(
        repeat_completion_rates
    )
    fatigue_values = sorted(
        float(row["maximum_worker_fatigue"])
        for row in all_rows
        if row.get("maximum_worker_fatigue") is not None
        and math.isfinite(float(row["maximum_worker_fatigue"]))
    )
    raw_constraints = config["training"]["two_stage"].get(
        "quality_promotion_constraints"
    )
    tail_fraction = float(
        raw_constraints.get("tail_fraction", 0.10)
        if isinstance(raw_constraints, dict)
        else 0.10
    )
    tail_count = max(1, int(math.ceil(len(fatigue_values) * tail_fraction)))
    fatigue_tail = fatigue_values[-tail_count:] if fatigue_values else []
    combined["fatigue_cvar90"] = (
        sum(fatigue_tail) / len(fatigue_tail)
        if fatigue_tail
        else float("nan")
    )
    combined["maximum_observed_fatigue"] = (
        max(fatigue_values) if fatigue_values else float("nan")
    )
    combined["fatigue_safe_line_pass"] = all(
        float(row.get("maximum_worker_fatigue", float("inf")))
        <= float(row.get("safe_fatigue_limit", float("-inf"))) + 1e-12
        for row in all_rows
    )
    return combined


def _normalized_validation_quality_score(
    score: tuple[float, float, float, float],
    reward_config: dict,
) -> float | None:
    del reward_config
    if len(score) != 4:
        return None
    result = float(score[1])
    return result if math.isfinite(result) and result >= 0.0 else None


def _single_objective_guard_score(
    validation: dict,
    objective_name: str,
) -> tuple[float, float, float, float]:
    if objective_name not in SINGLE_OBJECTIVE_METRICS:
        raise ValueError(f"unsupported single objective {objective_name!r}")
    metric_name = SINGLE_OBJECTIVE_METRICS[objective_name]
    metric = validation["all_instance_metrics"].get(metric_name, {})
    raw_value = metric.get("mean") if isinstance(metric, dict) else None
    objective_value = math.inf if raw_value is None else float(raw_value)
    return (
        -float(validation["completion_rate"]),
        objective_value,
        0.0,
        0.0,
    )


def _balanced_guard_score(
    validation: dict,
) -> tuple[float, float, float, float]:
    base = evaluation_selection_key(validation)
    metrics = validation["all_instance_metrics"]

    def mean(name: str) -> float:
        value = metrics.get(name, {}).get("mean")
        return float(value) if value is not None else math.inf

    return (
        float(base[0]),
        float(base[1]),
        mean("reconfiguration_cost"),
        mean("worker_load_variance"),
    )


def _official_evaluation_sampling_seeds(config: dict) -> list[int]:
    seeds = [
        int(value)
        for value in config["training"].get(
            "final_evaluation_sampling_seeds",
            (100011, 100012, 100013),
        )
    ]
    if seeds != [100011, 100012, 100013]:
        raise ValueError(
            "M1 final evaluation sampling seeds must be "
            "100011/100012/100013"
        )
    return seeds


def _checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reevaluate_checkpoint_from_disk(
    config: dict,
    *,
    checkpoint: Path,
    bootstrap_observation,
    dataset_name: str,
    instance_limit: int | None,
    sampling_seeds: list[int],
    greedy_only: bool = False,
) -> dict[str, object]:
    """Load an isolated agent and produce the only final reported metrics."""
    evaluation_agent = PPOAgent(
        build_actor_critic(bootstrap_observation, config["network"]),
        config["ppo"],
        device=config["device"],
    )
    metadata = evaluation_agent.load(checkpoint, load_optimizer=False)
    greedy_rows, _, _, greedy = evaluate_dataset(
        config,
        dataset_name=dataset_name,
        policy_name="ppo",
        ppo_agent=evaluation_agent,
        instance_limit=instance_limit,
        decode_mode="greedy",
    )
    greedy["physical_safety_pass"] = _rows_are_physically_safe(
        greedy_rows, 1e-9
    )
    sampled = None
    if not greedy_only:
        sampled = _evaluate_sampled_validation(
            config,
            dataset_name=dataset_name,
            ppo_agent=evaluation_agent,
            instance_limit=instance_limit,
            sampling_seeds=sampling_seeds,
        )
    manifest_path = _validation_manifest_path(config)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "checkpoint_metadata": metadata,
        "provenance": build_provenance(
            config,
            dataset_manifest_path=(
                manifest_path if manifest_path.is_file() else None
            ),
            checkpoint_path=checkpoint,
            checkpoint_metadata=metadata,
        ),
        "evaluation_config": {
            "dataset": dataset_name,
            "instance_limit": instance_limit,
            "greedy": True,
            "sampling_seeds": list(sampling_seeds),
        },
        "greedy": greedy,
        "sampled": sampled,
    }


def _assert_single_objective_checkpoint_evaluation(
    phase_controller: TrainingPhaseController,
    evaluation: dict[str, object],
) -> None:
    if (
        phase_controller.quality_checkpoint_promotion
        != SINGLE_OBJECTIVE_PROMOTION_MODE
    ):
        return
    greedy = evaluation.get("greedy")
    if not isinstance(greedy, dict):
        raise RuntimeError("single-objective checkpoint is missing greedy evaluation")
    instance_count = int(greedy.get("instance_count", 0))
    completed_count = int(greedy.get("completed_count", 0))
    failed_count = instance_count - completed_count
    completion_pass = bool(
        instance_count == phase_controller.single_objective_audit_instance_limit
        and float(greedy.get("completion_rate", math.nan))
        >= phase_controller.single_objective_audit_completion_target - 1e-12
        and failed_count
        <= phase_controller.single_objective_audit_max_failed_instances
    )
    violation_pass = bool(
        int(greedy.get("schedule_violation_count", 0))
        == phase_controller.single_objective_audit_schedule_violation_target
    )
    physical_pass = bool(greedy.get("physical_safety_pass", False))
    if not (completion_pass and violation_pass and physical_pass):
        raise RuntimeError(
            "single-objective checkpoint failed final 500-instance audit: "
            f"instances={instance_count}, failed={failed_count}, "
            f"completion={completion_pass}, violation={violation_pass}, "
            f"physical_safety={physical_pass}"
        )
    objective_name = phase_controller.single_objective_name
    if objective_name is None:
        raise RuntimeError("single-objective checkpoint target is missing")
    score = _single_objective_guard_score(greedy, objective_name)
    expected = phase_controller.accepted_single_objective_audit_value
    if expected is None or not math.isclose(
        float(score[1]), float(expected), rel_tol=0.0, abs_tol=1e-8
    ):
        raise RuntimeError(
            "single-objective checkpoint objective changed after disk reload: "
            f"expected={expected}, observed={score[1]}"
        )
    metadata = evaluation.get("checkpoint_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("single-objective checkpoint metadata is missing")
    saved = metadata.get("single_objective_audit_value")
    if saved is None or not math.isclose(
        float(score[1]), float(saved), rel_tol=0.0, abs_tol=1e-8
    ):
        raise RuntimeError(
            "single-objective checkpoint metadata disagrees with disk audit: "
            f"saved={saved}, observed={score[1]}"
        )


def _single_objective_checkpoint_metadata(
    phase_controller: TrainingPhaseController,
    *,
    checkpoint_role: str | None = None,
) -> dict[str, object]:
    if (
        phase_controller.quality_checkpoint_promotion
        != SINGLE_OBJECTIVE_PROMOTION_MODE
    ):
        return {}
    diagnostics = phase_controller.last_promotion_diagnostics
    audit_diagnostics = phase_controller.last_single_objective_audit_diagnostics
    return {
        "single_objective_target": phase_controller.single_objective_name,
        "single_objective_statistic": phase_controller.single_objective_window_statistic,
        "single_objective_window_size": phase_controller.single_objective_window_size,
        "single_objective_window_count": len(phase_controller.single_objective_window_values),
        "single_objective_window_episodes": list(phase_controller.single_objective_window_episodes),
        "single_objective_window_values": list(phase_controller.single_objective_window_values),
        "single_objective_window_statistic_value": diagnostics.get(
            "window_objective_statistic"
        ),
        "single_objective_checkpoint_role": checkpoint_role,
        "single_objective_candidate_value": diagnostics.get(
            "promotion_candidate_objective_value"
        ),
        "single_objective_candidate_anchor_value": (
            phase_controller.single_objective_candidate_anchor_value
        ),
        "single_objective_candidate_improvement_epsilon": (
            phase_controller.single_objective_candidate_improvement_epsilon
        ),
        "single_objective_audit_instance_limit": (
            phase_controller.single_objective_audit_instance_limit
        ),
        "single_objective_audit_completion_target": (
            phase_controller.single_objective_audit_completion_target
        ),
        "single_objective_audit_max_failed_instances": (
            phase_controller.single_objective_audit_max_failed_instances
        ),
        "single_objective_accepted_failed_instances": (
            phase_controller.accepted_single_objective_failed_instances
        ),
        "single_objective_accepted_window_median": (
            phase_controller.accepted_single_objective_window_value
        ),
        "single_objective_audit_value": (
            phase_controller.accepted_single_objective_audit_value
        ),
        "single_objective_audit_diagnostics": dict(audit_diagnostics),
        "formal_eligible": False,
    }


def _single_objective_failure_rows(
    rows: list[dict], *, episode: int, fatigue_tolerance: float = 1e-9,
    audit_event: str | None = None,
) -> list[dict]:
    failures: list[dict] = []
    for row in rows:
        truncated = bool(row.get("truncated", False))
        violations = int(row.get("schedule_violation_count", 0))
        unfinished = row.get("unfinished_orders", 0)
        complete = bool(row.get("terminated", False)) and not truncated
        fatigue = row.get("maximum_worker_fatigue")
        safe_limit = row.get("safe_fatigue_limit")
        physical_bad = (
            fatigue is None
            or safe_limit is None
            or float(fatigue) > float(safe_limit) + fatigue_tolerance
        )
        reasons = []
        if not complete:
            reasons.append("incomplete")
        if truncated:
            reasons.append("truncated")
        if violations:
            reasons.append("schedule_violation")
        if physical_bad:
            reasons.append("physical_safety")
        if reasons:
            failures.append(
                {
                    "episode": int(episode),
                    "instance_id": row.get("instance_id"),
                    "truncated": truncated,
                    "schedule_violation_count": violations,
                    "unfinished_orders": unfinished,
                    "maximum_worker_fatigue": fatigue,
                    "safe_fatigue_limit": safe_limit,
                    "failure_reason": ";".join(reasons),
                }
            )
            if audit_event is not None:
                failures[-1]["audit_event"] = audit_event
    return failures


def _evaluate_single_objective_audit(
    config: dict,
    *,
    dataset_name: str,
    ppo_agent: PPOAgent,
    phase_controller: TrainingPhaseController,
    runner: ParallelEpisodeRunner | None = None,
    use_parallel: bool = False,
) -> tuple[list[dict], dict]:
    """Evaluate the improved daily candidate on the complete fixed manifest."""
    limit = phase_controller.single_objective_audit_instance_limit
    if use_parallel:
        if runner is None:
            raise ValueError("parallel single-objective audit requires a runner")
        rows, audit = evaluate_dataset_parallel(
            config,
            dataset_name=dataset_name,
            ppo_agent=ppo_agent,
            runner=runner,
            instance_limit=limit,
            decode_mode="greedy",
        )
    else:
        rows, _, _, audit = evaluate_dataset(
            config,
            dataset_name=dataset_name,
            policy_name="ppo",
            ppo_agent=ppo_agent,
            instance_limit=limit,
            decode_mode="greedy",
        )
    audit["physical_safety_pass"] = _rows_are_physically_safe(rows, 1e-9)
    objective_name = phase_controller.single_objective_name
    if objective_name is None:
        raise RuntimeError("single-objective audit target is missing")
    audit["single_objective_value"] = _single_objective_guard_score(
        audit, objective_name
    )[1]
    return rows, audit


def _single_objective_audit_log_row(
    config: dict,
    *,
    episode: int,
    phase_controller: TrainingPhaseController,
) -> dict[str, object]:
    diagnostics = phase_controller.last_single_objective_audit_diagnostics
    manifest_path = _validation_manifest_path(config)
    manifest_sha256 = (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if manifest_path.is_file()
        else None
    )
    return {
        "episode": int(episode),
        "single_objective_target": phase_controller.single_objective_name,
        "window_size": phase_controller.single_objective_window_size,
        "window_statistic": phase_controller.single_objective_window_statistic,
        "window_episodes": list(phase_controller.single_objective_window_episodes),
        "window_objective_values": list(phase_controller.single_objective_window_values),
        "window_objective_statistic": diagnostics.get("audit_window_median"),
        "candidate_anchor_value": phase_controller.single_objective_candidate_anchor_value,
        "validation_manifest_sha256": manifest_sha256,
        **diagnostics,
    }


def _can_reuse_final_sampled_validation(
    *,
    final_episode: int,
    sampled_episode: int | None,
    validation_event: str | None,
    sampled_validation: dict | None,
) -> bool:
    return bool(
        sampled_validation is not None
        and sampled_episode == final_episode
        and validation_event in {"transition", "accepted"}
    )


def _resolve_final_accepted_sampled_validation(
    config: dict,
    *,
    dataset_name: str,
    ppo_agent: PPOAgent,
    instance_limit: int | None,
    sampling_seeds: list[int],
    final_episode: int,
    sampled_episode: int | None,
    validation_event: str | None,
    sampled_validation: dict | None,
    runner: ParallelEpisodeRunner | None = None,
    use_parallel: bool = False,
) -> tuple[dict, str, bool]:
    if _can_reuse_final_sampled_validation(
        final_episode=final_episode,
        sampled_episode=sampled_episode,
        validation_event=validation_event,
        sampled_validation=sampled_validation,
    ):
        return (
            sampled_validation,
            "reused_final_accepted_candidate",
            False,
        )
    resolved = _evaluate_sampled_validation(
        config,
        dataset_name=dataset_name,
        ppo_agent=ppo_agent,
        instance_limit=instance_limit,
        sampling_seeds=sampling_seeds,
        runner=runner,
        use_parallel=use_parallel,
    )
    event_label = validation_event or "missing_final_validation_event"
    return resolved, f"rerun_after_{event_label}", True


def _attach_sampled_validation(
    validation_row: dict,
    sampled: dict,
    *,
    completed_episodes: int,
) -> None:
    sampled_row = _validation_log_row(
        sampled,
        completed_episodes=completed_episodes,
    )
    for key, value in sampled_row.items():
        if key not in {"episode", "dataset"}:
            validation_row[f"sampled_{key}"] = value
    validation_row["sampled_repeat_count"] = sampled["repeat_count"]
    validation_row["sampled_unique_instance_count"] = sampled[
        "unique_instance_count"
    ]
    for name in (
        "completion_rate",
        "mean_unfinished_orders",
        "mean_feasibility_proxy_return",
        "mean_relative_heuristic_gap_percent",
    ):
        greedy_value = validation_row.get(name)
        sampled_value = validation_row.get(f"sampled_{name}")
        validation_row[f"sampled_minus_greedy_{name}"] = (
            float(sampled_value) - float(greedy_value)
            if sampled_value is not None and greedy_value is not None
            else None
        )


ACTION_TYPE_COUNT_FIELDS = (
    "direct_process_action_count",
    "commit_reconfig_action_count",
    "defer_production_action_count",
    "worker_assign_action_count",
    "advance_event_action_count",
)


FORCED_ACTION_COUNT_FIELDS = (
    "forced_action_state_count",
    "forced_production_count",
    "forced_worker_count",
    "forced_pair_count",
    "forced_advance_count",
    "forced_production_pair_count",
    "forced_production_advance_count",
    "forced_worker_pair_count",
    "forced_worker_advance_count",
    "forced_pair_advance_blocked_non_delay_count",
    "forced_worker_pair_non_delay_count",
    "forced_pair_advance_physically_unavailable_count",
    "forced_advance_pair_physically_unavailable_count",
    "forced_wait_dis_count",
    "forced_wait_ins_count",
    "forced_mixed_wait_stage_count",
    "forced_phase_handoff_count",
    "forced_recovery_advance_count",
    "forced_future_event_advance_count",
    "forced_direct_process_count",
    "forced_commit_reconfig_count",
    "forced_defer_production_count",
    "forced_worker_assign_count",
    "forced_advance_event_count",
    "forced_action_chain_count",
)

FORCED_ACTION_VALUE_FIELDS = (
    "longest_forced_action_chain",
    "mean_forced_action_chain_length",
)


def _training_effect_fields(metrics: dict) -> dict:
    total_orders = int(metrics.get("total_orders", 0))
    total_operations = int(metrics.get("total_operations", 0))
    return {
        "terminated": bool(metrics.get("terminated", False)),
        "truncated": bool(metrics.get("truncated", False)),
        "terminal_reason": metrics.get("terminal_reason"),
        "completed_order_ratio": (
            float(metrics.get("completed_orders", 0)) / total_orders
            if total_orders
            else None
        ),
        "completed_operation_ratio": (
            float(metrics.get("completed_operations", 0))
            / total_operations
            if total_operations
            else None
        ),
        "total_flow_time": metrics.get("total_flow_time"),
        "flow_time_objective": metrics.get("flow_time_objective"),
        "reconfiguration_cost": metrics.get(
            "reconfiguration_cost"
        ),
        "worker_load_variance": metrics.get("worker_load_variance"),
        "quality_score": metrics.get("quality_score"),
        "preference_quality_score": metrics.get(
            "preference_quality_score"
        ),
        "maximum_worker_fatigue": metrics.get(
            "maximum_worker_fatigue"
        ),
        "mean_peak_worker_fatigue": metrics.get(
            "mean_peak_worker_fatigue"
        ),
        "safe_fatigue_limit": metrics.get("safe_fatigue_limit"),
        "fatigue_masked_action_count": metrics.get(
            "fatigue_masked_action_count"
        ),
        "fatigue_masked_action_ratio": metrics.get(
            "fatigue_masked_action_ratio"
        ),
        "worker_competition_event_count": metrics.get(
            "worker_competition_event_count"
        ),
        "worker_matching_deficit_event_count": metrics.get(
            "worker_matching_deficit_event_count"
        ),
        "resource_admission_masked_action_count": metrics.get(
            "resource_admission_masked_action_count"
        ),
        "resource_admission_masked_action_ratio": metrics.get(
            "resource_admission_masked_action_ratio"
        ),
        "minimum_worker_alternatives": metrics.get(
            "minimum_worker_alternatives"
        ),
        "matching_preserving_worker_action_count": metrics.get(
            "matching_preserving_worker_action_count"
        ),
        "candidate_recovery_advance_count": metrics.get(
            "candidate_recovery_advance_count"
        ),
        "production_defer_recovery_improvement_count": metrics.get(
            "production_defer_recovery_improvement_count"
        ),
        "production_defer_wait_ticks": metrics.get(
            "production_defer_wait_ticks"
        ),
        "production_defer_wait_time": metrics.get(
            "production_defer_wait_time"
        ),
        "production_defer_shield_candidate_count": metrics.get(
            "production_defer_shield_candidate_count", 0
        ),
        "production_defer_shield_masked_count": metrics.get(
            "production_defer_shield_masked_count", 0
        ),
        "production_defer_shield_max_risk": metrics.get(
            "production_defer_shield_max_risk", 0.0
        ),
        "production_defer_shield_max_wait_ticks": metrics.get(
            "production_defer_shield_max_wait_ticks", 0
        ),
        "production_defer_shield_max_work_lower_bound_ticks": metrics.get(
            "production_defer_shield_max_work_lower_bound_ticks", 0
        ),
        "production_defer_shield_min_deadline_slack_ticks": metrics.get(
            "production_defer_shield_min_deadline_slack_ticks"
        ),
        "production_defer_shield_reason_counts": json.dumps(
            metrics.get("production_defer_shield_reason_counts", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        **{
            name: metrics.get(name, 0)
            for name in (
                "conditional_worker_wait_opportunity_count",
                "conditional_worker_wait_selected_count",
                "conditional_worker_wait_total_ticks",
                "conditional_worker_wait_total_time",
                "conditional_worker_wait_pair_gain_sum",
                "conditional_worker_wait_fatigue_improvement_sum",
                "conditional_worker_wait_duration_improvement_ticks_sum",
                "conditional_worker_wait_max_consecutive_observed",
                "reconfiguration_reuse_count",
                "qualification_scarcity_regret",
                "qualification_scarcity_decision_count",
            )
        },
        "conditional_worker_wait_reason_counts": json.dumps(
            metrics.get("conditional_worker_wait_reason_counts", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "machine_waiting_for_worker_time": metrics.get(
            "machine_waiting_for_worker_time"
        ),
        "completed_reconfigurations": metrics.get(
            "completed_reconfigurations"
        ),
        "worker_switch_ratio": metrics.get("worker_switch_ratio"),
        **{
            name: metrics.get(name, 0)
            for name in (
                *ACTION_TYPE_COUNT_FIELDS,
                *FORCED_ACTION_COUNT_FIELDS,
                *FORCED_ACTION_VALUE_FIELDS,
            )
        },
        "schedule_violation_count": len(
            metrics.get("schedule_violations", [])
        ),
    }


def _forced_action_summary(rows: list[dict]) -> dict[str, object]:
    counts = {
        name: sum(int(row.get(name, 0) or 0) for row in rows)
        for name in FORCED_ACTION_COUNT_FIELDS
    }
    longest = [
        int(row.get("longest_forced_action_chain", 0) or 0)
        for row in rows
    ]
    total_chains = counts["forced_action_chain_count"]
    total_states = counts["forced_action_state_count"]
    return {
        **counts,
        "mean_forced_action_chain_length": (
            total_states / total_chains if total_chains else 0.0
        ),
        "mean_episode_longest_forced_action_chain": (
            float(np.mean(longest)) if longest else 0.0
        ),
        "p95_episode_longest_forced_action_chain": (
            float(np.percentile(longest, 95)) if longest else 0.0
        ),
        "maximum_forced_action_chain": max(longest, default=0),
    }


def _mean_finite(rows: list[dict], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None
        and math.isfinite(float(row[field]))
    ]
    return float(np.mean(values)) if values else None


def _late_training_diagnostics(
    rows: list[dict],
    *,
    window: int = 500,
) -> dict[str, object]:
    selected = rows[-min(len(rows), int(window)) :]
    fields = (
        "completed_order_ratio",
        "completed_operation_ratio",
        "machine_waiting_for_worker_time",
        "fatigue_masked_action_ratio",
        "resource_admission_masked_action_count",
        "resource_admission_masked_action_ratio",
        "worker_matching_deficit_event_count",
        "minimum_worker_alternatives",
        "matching_preserving_worker_action_count",
        "candidate_recovery_advance_count",
        "production_defer_recovery_improvement_count",
        "production_defer_wait_time",
        "reward_base",
        "reward_shaping",
        "reward_training",
    )
    pressure_profiles: dict[str, dict[str, object]] = {}
    for pressure in sorted(
        {
            str(row.get("pressure_type"))
            for row in selected
            if row.get("pressure_type") is not None
        }
    ):
        pressure_rows = [
            row
            for row in selected
            if str(row.get("pressure_type")) == pressure
        ]
        pressure_profiles[pressure] = {
            "sample_count": len(pressure_rows),
            "completion_rate": _mean_finite(
                [
                    {
                        "completed": float(
                            bool(row.get("terminated"))
                            and not bool(row.get("truncated"))
                        )
                    }
                    for row in pressure_rows
                ],
                "completed",
            ),
            "mean_completed_order_ratio": _mean_finite(
                pressure_rows,
                "completed_order_ratio",
            ),
        }
    return {
        "requested_window_episodes": int(window),
        "observed_episode_count": len(selected),
        "completion_rate": _mean_finite(
            [
                {
                    "completed": float(
                        bool(row.get("terminated"))
                        and not bool(row.get("truncated"))
                    )
                }
                for row in selected
            ],
            "completed",
        ),
        "means": {field: _mean_finite(selected, field) for field in fields},
        "by_pressure_type": pressure_profiles,
    }


def _ablation_gate_summary(
    config: dict,
    rows: list[dict],
    validation_rows: list[dict],
    stability_controller: ValidationStabilityController,
    best_feasibility_instance_rows: list[dict],
) -> dict[str, object] | None:
    variant = config["training"].get("ablation_variant")
    if variant is None:
        return None
    settings = config["training"].get("ablation_gate", {})
    reconfiguration_window = int(
        settings.get(
            "training_window_instances",
            settings.get("training_window_episodes", 200),
        )
    )
    all_reconfiguration_rows = [
        row
        for row in rows
        if row.get("pressure_type") == "reconfiguration_bottleneck"
    ]
    reconfiguration_rows = all_reconfiguration_rows[
        -min(len(all_reconfiguration_rows), reconfiguration_window) :
    ]
    reconfiguration_completion_rate = _mean_finite(
        [
            {
                "completed": float(
                    bool(row.get("terminated"))
                    and not bool(row.get("truncated"))
                )
            }
            for row in reconfiguration_rows
        ],
        "completed",
    )
    failure_instance_id = str(
        settings.get(
            "failure_instance_id",
            "validation_reconfiguration_bottleneck_2000009",
        )
    )
    failure_row = next(
        (
            row
            for row in best_feasibility_instance_rows
            if str(row.get("instance_id")) == failure_instance_id
        ),
        None,
    )
    recent_validation = validation_rows[-10:]
    recent_completion_rate = _mean_finite(
        recent_validation,
        "completion_rate",
    )
    rollback_rate = (
        stability_controller.feasibility_rollbacks
        / stability_controller.validation_count
        if stability_controller.validation_count
        else 0.0
    )
    identity_errors = [
        abs(float(row["reward_identity_error"]))
        for row in rows
        if row.get("reward_identity_error") is not None
        and math.isfinite(float(row["reward_identity_error"]))
    ]
    maximum_identity_error = max(identity_errors, default=0.0)
    violation_count = sum(
        int(row.get("schedule_violation_count", 0)) for row in rows
    ) + sum(
        int(row.get("schedule_violation_count", 0))
        for row in validation_rows
    )
    checks = {
        "reconfiguration_training_completion": bool(
            reconfiguration_completion_rate is not None
            and reconfiguration_completion_rate
            >= float(settings.get("reconfiguration_completion_rate", 0.60))
        ),
        "failure_instance_completed": bool(
            failure_row is not None
            and bool(failure_row.get("terminated"))
            and not bool(failure_row.get("truncated"))
        ),
        "failure_instance_worker_wait": bool(
            failure_row is not None
            and float(failure_row.get("machine_waiting_for_worker_time", math.inf))
            < float(settings.get("failure_instance_max_worker_wait", 100.0))
        ),
        "validation_reached_full_completion": bool(
            validation_rows
            and max(float(row["completion_rate"]) for row in validation_rows)
            >= 1.0 - 1e-12
        ),
        "recent_validation_completion": bool(
            recent_completion_rate is not None
            and recent_completion_rate
            >= float(settings.get("last_ten_validation_completion_rate", 0.95))
        ),
        "rollback_rate": rollback_rate
        < float(settings.get("maximum_rollback_rate", 0.20)),
        "learning_rate_floor": (
            stability_controller.current_learning_rate
            >= float(settings.get("minimum_learning_rate", 2.5e-5))
            - 1e-15
        ),
        "zero_constraint_violations": violation_count == 0,
        "base_reward_identity": maximum_identity_error
        <= float(settings.get("reward_identity_tolerance", 1e-8)),
    }
    return {
        "variant": str(variant),
        "passed": all(checks.values()),
        "checks": checks,
        "reconfiguration_training_requested_sample_count": (
            reconfiguration_window
        ),
        "reconfiguration_training_available_sample_count": len(
            all_reconfiguration_rows
        ),
        "reconfiguration_training_sample_count": len(reconfiguration_rows),
        "reconfiguration_training_completion_rate": (
            reconfiguration_completion_rate
        ),
        "failure_instance_id": failure_instance_id,
        "failure_instance": failure_row,
        "last_ten_validation_completion_rate": recent_completion_rate,
        "rollback_rate": rollback_rate,
        "current_learning_rate": stability_controller.current_learning_rate,
        "constraint_violation_count": violation_count,
        "maximum_base_reward_identity_error": maximum_identity_error,
    }


def _apply_ablation_variant(config: dict, variant: str | None) -> None:
    if variant is None:
        return
    normalized = str(variant).upper()
    if normalized == "E0":
        raise ValueError(
            "E0 reuses the existing seed-11 baseline and must not be retrained"
        )
    variants = {
        "E1", "E2", "E3", "R11", "S11", "L11", "Q11", "Q12", "Q13"
    }
    if normalized not in variants:
        raise ValueError(
            "ablation variant must be E1, E2, E3, R11, S11, L11, Q11, "
            "Q12, or Q13"
        )
    config["training"]["ablation_variant"] = normalized
    config["training"]["episodes"] = 600
    config["training"]["parallel_envs"] = 10
    config["training"]["validation_parallel_envs"] = 10
    config["training"]["validation_interval_episodes"] = 10
    config["seed"] = 11
    control = config["environment"]["worker_resource_control"]
    control["mode"] = "matching_admission_v1"
    config["reward"]["feasibility_shaping"]["enabled"] = normalized in {
        "E2", "E3", "S11", "L11", "Q11", "Q12", "Q13"
    }
    two_stage = config["training"]["two_stage"]
    if normalized == "Q13":
        two_stage["quality_checkpoint_promotion"] = "constrained_weighted"
        two_stage["quality_promotion_constraints"] = {
            "flow_relative_tolerance": 0.005,
            "cost_relative_tolerance": 0.0,
            "variance_relative_tolerance": 0.0,
            "minimum_normalized_score_improvement": 1e-12,
        }
    else:
        two_stage["quality_checkpoint_promotion"] = (
            "score_improving"
            if normalized in {"Q11", "Q12"}
            else "completion_only"
        )
        two_stage.pop("quality_promotion_constraints", None)
    sampled = config["training"]["validation_control"]["sampled"]
    if normalized in {"R11", "S11", "L11", "Q11", "Q12", "Q13"}:
        sampled["episode_milestones"] = [200, 400]
    else:
        sampled.pop("episode_milestones", None)
    if normalized == "Q12":
        reward = config["reward"]
        reward["flow_scale"] = 1200.0
        reward["cost_scale"] = 1000.0
        reward["variance_scale"] = 50.0
        reward["quality_weights"] = {
            "flow": 0.5,
            "cost": 0.3,
            "variance": 0.2,
        }
    rollback = config["training"]["validation_control"][
        "feasibility_rollback"
    ]
    plateau = config["training"]["validation_control"][
        "learning_rate_plateau"
    ]
    if normalized in {"E1", "E2"}:
        rollback["consecutive_validations"] = 1
        rollback["cooldown_validations"] = 0
        plateau["patience_validations"] = 10
        plateau["minimum"] = 1e-5
    elif normalized in {"R11", "S11"}:
        rollback["consecutive_validations"] = 2
        rollback["cooldown_validations"] = 3
        plateau["patience_validations"] = 10
        plateau["minimum"] = 1e-5
    else:
        rollback["consecutive_validations"] = 2
        rollback["cooldown_validations"] = 3
        plateau["patience_validations"] = 15
        plateau["minimum"] = 2.5e-5


def train(
    config: dict,
    *,
    smoke: bool = False,
    run_name: str | None = None,
    online_instances: bool | None = None,
    algorithm_seed: int | None = None,
    parallel_envs: int | None = None,
    visdom_enabled: bool | None = None,
    ablation_variant: str | None = None,
    initial_checkpoint: str | Path | None = None,
    warm_start_checkpoint: str | Path | None = None,
) -> Path:
    config = deepcopy(config)
    _apply_ablation_variant(config, ablation_variant)
    if (
        ablation_variant is not None
        and algorithm_seed is not None
        and int(algorithm_seed) != 11
    ):
        raise ValueError("screening ablations require algorithm seed 11")
    override_visdom_enabled(config, visdom_enabled)
    forced_action_compression = config["training"].get(
        "forced_action_compression", False
    )
    if not isinstance(forced_action_compression, bool):
        raise ValueError(
            "training.forced_action_compression must be boolean"
        )
    if forced_action_compression and float(config["ppo"]["gamma"]) != 1.0:
        raise ValueError(
            "forced action compression requires ppo.gamma = 1.0"
        )
    worker_local_physical_forced_actions = config["training"].get(
        "worker_local_physical_forced_actions",
        True,
    )
    if not isinstance(worker_local_physical_forced_actions, bool):
        raise ValueError(
            "training.worker_local_physical_forced_actions must be boolean"
        )
    effective_algorithm_seed = validate_algorithm_seed(
        config,
        int(config["seed"]) if algorithm_seed is None else algorithm_seed,
    )
    config["seed"] = effective_algorithm_seed
    configured_warm_start = config["training"].get("warm_start")
    if configured_warm_start is not None and not isinstance(
        configured_warm_start, dict
    ):
        raise TypeError("training.warm_start must be an object")
    configured_warm_path = (
        configured_warm_start.get("checkpoint")
        if isinstance(configured_warm_start, dict)
        else None
    )
    if initial_checkpoint is not None and warm_start_checkpoint is not None:
        raise ValueError(
            "--initial-checkpoint and --warm-start-checkpoint are mutually exclusive"
        )
    effective_warm_start = (
        None
        if initial_checkpoint is not None
        else warm_start_checkpoint or configured_warm_path
    )
    if initial_checkpoint is not None:
        config["training"]["initial_checkpoint"] = str(
            Path(initial_checkpoint).resolve()
        )
    if effective_warm_start is not None:
        warm_path = project_path(effective_warm_start).resolve()
        required_source = (
            configured_warm_start.get("required_source_checkpoint")
            if isinstance(configured_warm_start, dict)
            else None
        )
        if required_source is not None and warm_path != project_path(
            required_source
        ).resolve():
            raise ValueError(
                "warm-start source is not the configured E1 accepted checkpoint"
            )
        effective_warm_start = warm_path
        config["training"]["warm_start_checkpoint"] = str(warm_path)
    set_seed(effective_algorithm_seed)
    use_online_instances = (
        bool(config["training"]["online_instances"])
        if online_instances is None
        else bool(online_instances)
    )
    if not smoke and not use_online_instances:
        raise ValueError(
            "fixed-instance training is only available with --smoke"
        )
    episodes = int(
        config["training"]["smoke_episodes"]
        if smoke
        else config["training"]["episodes"]
    )
    configured_parallel_envs = int(
        config["training"]["smoke_parallel_envs"]
        if smoke
        else config["training"]["parallel_envs"]
    )
    effective_parallel_envs = (
        configured_parallel_envs
        if parallel_envs is None
        else int(parallel_envs)
    )
    if effective_parallel_envs < 1:
        raise ValueError("parallel_envs must be positive")
    if not smoke and config["training"].get("preference_stage_schedule"):
        expected_updates = math.ceil(episodes / effective_parallel_envs)
        configured_final_update = int(
            config["training"]["preference_stage_schedule"]["final_update"]
        )
        if expected_updates != configured_final_update:
            raise ValueError(
                "E2.7 trajectory/parallel budget must produce exactly 200 PPO updates"
            )
    preference_grouping = training_preference_group(config)
    training_base_instance_count(config, episodes)
    development_mode = _development_acceptance_enabled(config)
    if development_mode:
        if effective_algorithm_seed != 11:
            raise ValueError("E2.7 development acceptance is fixed to seed 11")
        if preference_grouping is not None:
            raise ValueError(
                "E2.7 requires one independently generated instance per trajectory"
            )
        if effective_warm_start is None and initial_checkpoint is None:
            raise ValueError(
                "E2.7 requires either the E1 warm-start or a strict E2.7 resume"
            )
    if preference_grouping is not None:
        group_size = int(preference_grouping["group_size"])
        if effective_parallel_envs == 1:
            raise ValueError(
                "grouped preference training requires parallel_envs >= group size"
            )
        if effective_parallel_envs % group_size:
            raise ValueError(
                "parallel_envs must be divisible by the preference group size"
            )
    if not use_online_instances:
        if parallel_envs is not None and effective_parallel_envs != 1:
            raise ValueError(
                "parallel sampling requires online training instances"
            )
        effective_parallel_envs = 1
    torch_threads = int(config["training"]["torch_num_threads"])
    if torch_threads < 1:
        raise ValueError("training.torch_num_threads must be positive")
    import torch

    torch.set_num_threads(torch_threads)
    validation_parallel_envs = (
        effective_parallel_envs
        if smoke or parallel_envs is not None
        else int(config["training"]["validation_parallel_envs"])
    )
    if validation_parallel_envs < 1:
        raise ValueError(
            "training.validation_parallel_envs must be positive"
        )
    validation_interval = int(
        config["training"]["validation_interval_episodes"]
    )
    if validation_interval < 1:
        raise ValueError(
            "training.validation_interval_episodes must be positive"
        )
    if (
        effective_parallel_envs > 1
        and not smoke
        and validation_interval % effective_parallel_envs != 0
    ):
        raise ValueError(
            "validation_interval_episodes must be divisible by "
            "parallel_envs"
        )
    if effective_parallel_envs > 1:
        parallel_key = (
            "smoke_parallel_envs" if smoke else "parallel_envs"
        )
        config["training"][parallel_key] = effective_parallel_envs
        config["training"]["validation_parallel_envs"] = (
            validation_parallel_envs
        )
        return _train_parallel(
            config,
            smoke=smoke,
            run_name=run_name,
            episodes=episodes,
            parallel_envs=min(effective_parallel_envs, episodes),
            validation_parallel_envs=validation_parallel_envs,
            initial_checkpoint=initial_checkpoint,
            warm_start_checkpoint=effective_warm_start,
        )
    if _uses_tiered_training_gates(config):
        # Keep a single source of truth for E2.4--E2.7 gate semantics.  A
        # requested one-environment run still uses the parallel runner with a
        # worker count of one, rather than the historical serial loop.
        serial_parallel_key = "smoke_parallel_envs" if smoke else "parallel_envs"
        config["training"][serial_parallel_key] = 1
        config["training"]["validation_parallel_envs"] = 1
        return _train_parallel(
            config,
            smoke=smoke,
            run_name=run_name,
            episodes=episodes,
            parallel_envs=1,
            validation_parallel_envs=1,
            initial_checkpoint=initial_checkpoint,
            warm_start_checkpoint=effective_warm_start,
        )
    serial_parallel_key = (
        "smoke_parallel_envs" if smoke else "parallel_envs"
    )
    config["training"][serial_parallel_key] = 1
    config["training"]["validation_parallel_envs"] = 1
    online_dataset = (
        OnlineInstanceDataset(
            config=config,
            template=load_instance_yaml(
                project_path(config["paths"]["fixed_instance"])
            ),
            episode_count=episodes,
        )
        if use_online_instances
        else None
    )
    fixed_instance = (
        None if use_online_instances else load_configured_instance(config)
    )
    first_record = online_dataset[0] if online_dataset is not None else None
    instance = (
        first_record.instance
        if first_record is not None
        else fixed_instance
    )
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    bootstrap_observation = observation.copy()
    network = build_actor_critic(observation, config["network"])
    agent = PPOAgent(network, config["ppo"], device=config["device"])
    warm_start_report = None
    initial_metadata: dict[str, Any] = {}
    if initial_checkpoint is not None:
        initial_metadata = agent.load(initial_checkpoint, load_optimizer=True)
        warm_start_report = _restore_e2_7_resume_provenance(
            agent, initial_metadata, config
        )
    elif effective_warm_start is not None:
        expected_count = int(
            (configured_warm_start or {}).get(
                "expected_shared_parameter_count", 116
            )
        )
        warm_start_report = agent.warm_start_from_e1(
            effective_warm_start,
            expected_shared_parameter_count=expected_count,
        )
        warm_start_report = agent.verify_warm_start_canonical_identity(
            bootstrap_observation,
            environment.get_action_mask(),
            source_checkpoint=effective_warm_start,
        )
    run_directory = create_run_directory(
        project_path(config["paths"]["result_root"]),
        label="train_smoke" if smoke else "train",
        run_name=run_name,
    )
    write_config(run_directory, config)
    if warm_start_report is not None:
        write_json(run_directory / "warm_start_mapping.json", warm_start_report)
    visdom_settings = resolve_visdom_settings(config)
    dashboard = create_training_dashboard(
        config=config,
        run_directory=run_directory,
        total_episodes=episodes,
    )
    smoke_limit = int(config["training"]["smoke_rollout_steps"])
    validation_split = str(
        config["training"]["validation_split"]
    )
    validation_interval = int(
        config["training"]["validation_interval_episodes"]
    )
    if validation_interval < 1:
        raise ValueError(
            "training.validation_interval_episodes must be positive"
        )
    validation_limit = (
        config["training"]["smoke_validation_instance_limit"]
        if smoke
        else config["training"]["validation_instance_limit"]
    )
    validation_limit = (
        None if validation_limit is None else int(validation_limit)
    )
    _validate_single_objective_validation_protocol(
        config, smoke=smoke, validation_limit=validation_limit
    )
    rows: list[dict] = []
    update_rows: list[dict] = []
    validation_rows: list[dict] = []
    single_objective_audit_rows: list[dict] = []
    single_objective_audit_failure_rows: list[dict] = []
    pareto_validation_rows: list[dict] = []
    pareto_candidate_rows: list[dict] = []
    instance_ids: list[str] = []
    best_checkpoint = run_directory / "best_checkpoint.pt"
    best_feasibility_checkpoint = (
        run_directory / "best_feasibility_checkpoint.pt"
    )
    phase1_checkpoint = run_directory / "phase1_checkpoint.pt"
    accepted_checkpoint = _accepted_checkpoint_path(run_directory, config)
    candidate_checkpoint = run_directory / "single_objective_candidate_checkpoint.pt"
    safe_checkpoint = run_directory / "safe_checkpoint.pt"
    anchor_safe_checkpoint = run_directory / "anchor_safe_checkpoint.pt"
    full_grid_safe_checkpoint = (
        run_directory / "full_grid_safe_checkpoint.pt"
    )
    last_checkpoint = run_directory / "last_checkpoint.pt"
    best_validation: dict | None = None
    best_feasibility_validation: dict | None = None
    best_feasibility_instance_rows: list[dict] = []
    last_sampled_validation: dict | None = None
    last_sampled_validation_episode: int | None = None
    last_validation_event: str | None = None
    final_accepted_sampled_validation: dict | None = None
    final_accepted_sampled_validation_source: str | None = None
    best_score: tuple[float, float, float, float] | None = None
    phase_controller = TrainingPhaseController.from_config(config)
    stability_controller = ValidationStabilityController.from_config(config)
    preference_stage_controller = E2_7PreferenceStageController.from_config(config)
    if preference_stage_controller is not None and initial_checkpoint is not None:
        preference_stage_controller.restore(
            initial_metadata.get("preference_stage_controller")
        )
    for episode in range(episodes):
        reward_phase = phase_controller.phase
        sampling_start = time.perf_counter()
        generation_start = time.perf_counter()
        record = (
            first_record
            if episode == 0
            else online_dataset[episode] if online_dataset is not None else None
        )
        generation_time = (
            time.perf_counter() - generation_start
            if online_dataset is not None and episode > 0
            else 0.0
        )
        instance = (
            record.instance
            if record is not None
            else fixed_instance
        )
        instance_ids.append(instance.instance_id)
        rollout = _collect_serial_batch(
            config=config,
            agent=agent,
            environment=environment,
            instance=instance,
            record=record,
            episode_index=episode,
            sampling_start=sampling_start,
            generation_time_seconds=generation_time,
            step_limit=smoke_limit if smoke else None,
            reward_phase=reward_phase,
        )
        episode_rollout = rollout.episodes[0]
        buffer = rollout.buffer
        step_count = episode_rollout.step_count
        reward_sum = episode_rollout.reward_sum
        reward_components = episode_rollout.reward_components
        inference_time = rollout.policy_inference_time_seconds
        environment_step_time = (
            episode_rollout.environment_step_time_seconds
        )
        sampling_time = rollout.sampling_wall_time_seconds
        update_start = time.perf_counter()
        if len(buffer) == 0:
            raise RuntimeError(
                "training batch contains no policy transitions after "
                "forced action compression"
            )
        preference_stage = (
            preference_stage_controller.apply(agent)
            if preference_stage_controller is not None
            else None
        )
        losses = agent.update(buffer, reward_phase=reward_phase)
        if preference_stage_controller is not None:
            preference_stage_controller.observe(losses, update_id=episode + 1)
        update_time = time.perf_counter() - update_start
        metrics = episode_rollout.metrics
        expected_reward = episode_rollout.expected_reward
        shaping_reward = (
            reward_components["feasibility_shaping"]
            + reward_components.get("defer_risk_shaping", 0.0)
        )
        base_reward = reward_sum - shaping_reward
        row = {
            "episode": episode,
            "update_id": episode + 1,
            "instance_id": instance.instance_id,
            "instance_seed": (
                record.metadata["seed"] if record is not None else None
            ),
            "pressure_type": (
                record.metadata["pressure_type"] if record is not None else "fixed"
            ),
            "cost_profile": (
                record.metadata["cost_profile"] if record is not None else "fixed"
            ),
            "w_flow": episode_rollout.preference.flow,
            "w_cost": episode_rollout.preference.cost,
            "w_variance": episode_rollout.preference.variance,
            "preference_source": episode_rollout.preference_source,
            "steps": step_count,
            "policy_steps": len(buffer),
            "forced_actions": step_count - len(buffer),
            "forced_action_ratio": (
                (step_count - len(buffer)) / step_count
                if step_count > 0
                else 0.0
            ),
            "unattributed_forced_reward": (
                episode_rollout.unattributed_forced_reward
            ),
            "reward": reward_sum,
            "reward_base": base_reward,
            "reward_shaping": shaping_reward,
            "reward_training": reward_sum,
            "expected_reward": expected_reward,
            "reward_identity_error": base_reward - expected_reward,
            "reward_phase": reward_phase,
            "preference_stage": preference_stage,
            **{
                f"reward_{name}": value
                for name, value in reward_components.items()
            },
            "completed_operations": metrics["completed_operations"],
            "time": metrics["time"],
            **_training_effect_fields(metrics),
            "parallel_envs": 1,
            "batch_transition_count": len(buffer),
            "sampling_wall_time_seconds": sampling_time,
            "policy_inference_time_seconds": inference_time,
            "generation_time_seconds": generation_time,
            "environment_step_time_seconds": environment_step_time,
            "ppo_update_time_seconds": update_time,
            "loss_scope": "single_episode",
            "candidate_status": (
                "pending" if reward_phase == "quality" else "not_applicable"
            ),
            **losses,
        }
        rows.append(row)
        training_time = sampling_time + update_time
        update_rows.append(
            {
                "update_id": episode + 1,
                "episode_start": episode,
                "episode_end": episode,
                "episode_count": 1,
                "parallel_envs": 1,
                "transition_count": len(buffer),
                "environment_step_count": step_count,
                "forced_action_count": step_count - len(buffer),
                "forced_action_ratio": (
                    (step_count - len(buffer)) / step_count
                    if step_count > 0
                    else 0.0
                ),
                "sampling_wall_time_seconds": sampling_time,
                "policy_inference_time_seconds": inference_time,
                "generation_time_seconds": generation_time,
                "environment_step_time_seconds": (
                    environment_step_time
                ),
                "ppo_update_time_seconds": update_time,
                "transitions_per_second": (
                    len(buffer) / training_time
                    if training_time > 0
                    else 0.0
                ),
                "reward_phase": reward_phase,
                "preference_stage": preference_stage,
                "candidate_status": (
                    "pending"
                    if reward_phase == "quality"
                    else "not_applicable"
                ),
                **losses,
            }
        )
        completed_episodes = episode + 1
        regular_validation_due = (
            completed_episodes % validation_interval == 0
            or completed_episodes == episodes
        )
        should_validate = phase_controller.should_validate(
            regular_validation_due
        )
        if should_validate:
            validation_instance_rows, _, _, validation = evaluate_dataset(
                config,
                dataset_name=validation_split,
                policy_name="ppo",
                ppo_agent=agent,
                instance_limit=validation_limit,
            )
            if phase_controller.quality_checkpoint_promotion == SINGLE_OBJECTIVE_PROMOTION_MODE:
                validation["physical_safety_pass"] = _rows_are_physically_safe(
                    validation_instance_rows, 1e-9
                )
            validation_row = _validation_log_row(
                validation,
                completed_episodes=completed_episodes,
            )
            score = evaluation_selection_key(validation)
            if (
                phase_controller.quality_checkpoint_promotion
                == "balanced_guarded_v7"
            ):
                score = _balanced_guard_score(validation)
            elif (
                phase_controller.quality_checkpoint_promotion
                == SINGLE_OBJECTIVE_PROMOTION_MODE
            ):
                if phase_controller.single_objective_name is None:
                    raise RuntimeError("single-objective target is not configured")
                score = _single_objective_guard_score(
                    validation,
                    phase_controller.single_objective_name,
                )
            normalized_quality_score = (
                _normalized_validation_quality_score(
                    score,
                    config["reward"],
                )
                if phase_controller.quality_checkpoint_promotion
                in {
                    "constrained_weighted",
                    "aligned_quality",
                    "balanced_guarded_v7",
                }
                else None
            )
            stability = stability_controller.observe_greedy(
                score,
                validation["completion_rate"],
                completed_episodes=completed_episodes,
                feasibility_phase=reward_phase == "feasibility",
            )
            if stability_controller.should_run_sampled(
                final_validation=completed_episodes == episodes,
                completed_episodes=completed_episodes,
            ) and (
                phase_controller.quality_checkpoint_promotion
                not in ({"balanced_guarded_v7"} | PARETO_PROMOTION_MODES)
            ):
                sampled_validation = _evaluate_sampled_validation(
                    config,
                    dataset_name=validation_split,
                    ppo_agent=agent,
                    instance_limit=validation_limit,
                    sampling_seeds=stability_controller.sampled_seeds(
                        int(config["seed"])
                    ),
                )
                stability_controller.sampled_validation_runs += 1
                last_sampled_validation = sampled_validation
                last_sampled_validation_episode = completed_episodes
                _attach_sampled_validation(
                    validation_row,
                    sampled_validation,
                    completed_episodes=completed_episodes,
                )
            validation_event = phase_controller.observe_validation(
                validation["completion_rate"],
                completed_episodes=completed_episodes,
                score=score,
                normalized_quality_score=normalized_quality_score,
                truncated_count=int(validation.get("truncated_count", 0)),
                schedule_violation_count=int(
                    validation.get("schedule_violation_count", 0)
                ),
                physical_safety_pass=bool(
                    validation.get("physical_safety_pass", True)
                ),
            )
            if (
                phase_controller.quality_checkpoint_promotion
                == "balanced_guarded_v7"
                and validation_event
                in {"transition", "sampled_guard_pending"}
            ):
                sampled_validation = _evaluate_sampled_validation(
                    config,
                    dataset_name=validation_split,
                    ppo_agent=agent,
                    instance_limit=validation_limit,
                    sampling_seeds=_official_evaluation_sampling_seeds(
                        config
                    ),
                )
                stability_controller.sampled_validation_runs += 1
                last_sampled_validation = sampled_validation
                last_sampled_validation_episode = completed_episodes
                _attach_sampled_validation(
                    validation_row,
                    sampled_validation,
                    completed_episodes=completed_episodes,
                )
                validation_event = phase_controller.observe_sampled_guard(
                    sampled_validation,
                    completed_episodes=completed_episodes,
                    transition_anchor=validation_event == "transition",
                )
            daily_validation_event = validation_event
            audit_event = None
            validation_row["candidate_phase"] = reward_phase
            validation_row["validation_event"] = daily_validation_event
            validation_row.update(
                phase_controller.last_promotion_diagnostics
            )
            validation_row["phase_after_validation"] = (
                phase_controller.phase
            )
            validation_row["consecutive_completion_successes"] = (
                phase_controller.consecutive_successes
            )
            validation_row.update(stability)
            validation_row["feasibility_rollback_applied"] = bool(
                stability["rollback"]
            )
            if (
                phase_controller.quality_checkpoint_promotion
                == SINGLE_OBJECTIVE_PROMOTION_MODE
                and daily_validation_event == "audit_required"
            ):
                window_median = phase_controller.last_promotion_diagnostics.get(
                    "window_objective_statistic"
                )
                if window_median is None:
                    raise RuntimeError("single-objective audit is missing window median")
                agent.save(
                    candidate_checkpoint,
                    metadata={
                        **_checkpoint_protocol_metadata(config),
                        **_single_objective_checkpoint_metadata(
                            phase_controller,
                            checkpoint_role="audit_pending_candidate",
                        ),
                        "accepted_episode": None,
                        "daily_validation": validation_row,
                    },
                )
                audit_instance_rows, audit = _evaluate_single_objective_audit(
                    config,
                    dataset_name=validation_split,
                    ppo_agent=agent,
                    phase_controller=phase_controller,
                )
                audit_event = phase_controller.observe_single_objective_audit(
                    audit,
                    completed_episodes=completed_episodes,
                    window_median=float(window_median),
                )
                audit_log_row = _single_objective_audit_log_row(
                    config,
                    episode=completed_episodes,
                    phase_controller=phase_controller,
                )
                single_objective_audit_rows.append(audit_log_row)
                single_objective_audit_failure_rows.extend(
                    _single_objective_failure_rows(
                        audit_instance_rows,
                        episode=completed_episodes,
                        audit_event=audit_event,
                    )
                )
                validation_row.update(
                    {
                        "audit_event": audit_event,
                        "audit_failed_instance_count": audit_log_row.get(
                            "audit_failed_instance_count"
                        ),
                        "audit_completion_rate": audit_log_row.get(
                            "audit_completion_rate"
                        ),
                        "accepted_checkpoint_episode": (
                            phase_controller.accepted_quality_episode
                        ),
                    }
                )
                if audit_event == "accepted":
                    agent.save(
                        candidate_checkpoint,
                        metadata={
                            **_checkpoint_protocol_metadata(config),
                            **_single_objective_checkpoint_metadata(
                                phase_controller,
                                checkpoint_role=(
                                    "accepted_98_experiment_candidate"
                                ),
                            ),
                            "formal_eligible": False,
                            "accepted_episode": completed_episodes,
                            "daily_validation": validation_row,
                            "audit": audit_log_row,
                        },
                    )
                    candidate_checkpoint.replace(accepted_checkpoint)
                    best_score = score
                    best_validation = validation_row
                    row["candidate_status"] = "accepted_98"
                    update_rows[-1]["candidate_status"] = "accepted_98"
                elif candidate_checkpoint.exists():
                    candidate_checkpoint.unlink()
            validation_event = audit_event or daily_validation_event
            last_validation_event = validation_event
            validation_rows.append(validation_row)
            validation_is_safe = (
                _single_objective_hard_gate(validation)["all"]
                if phase_controller.quality_checkpoint_promotion
                == SINGLE_OBJECTIVE_PROMOTION_MODE
                else validation["completion_rate"] >= 1.0 - 1e-12
            )
            if validation_is_safe:
                agent.save(
                    safe_checkpoint,
                    metadata={
                        "checkpoint_role": "latest_safe",
                        "safe_episode": completed_episodes,
                        "validation": validation_row,
                    },
                )
            if bool(stability["improved"]) and reward_phase == "feasibility":
                best_feasibility_validation = validation_row
                best_feasibility_instance_rows = [
                    dict(value) for value in validation_instance_rows
                ]
                agent.save(
                    best_feasibility_checkpoint,
                    metadata={
                        "feature_dimensions": observation.feature_dimensions,
                        "edge_feature_dimensions": (
                            observation.edge_feature_dimensions
                        ),
                        "seed": config["seed"],
                        "smoke": smoke,
                        "online_instances": use_online_instances,
                        "generator_version": (
                            config["generator"]["version"]
                            if use_online_instances
                            else None
                        ),
                        "best_feasibility_episode": completed_episodes,
                        "learning_rate": (
                            stability_controller.current_learning_rate
                        ),
                        "validation": validation_row,
                    },
                )
                row["candidate_status"] = "feasibility_best"
                update_rows[-1]["candidate_status"] = "feasibility_best"
            is_new_best = False
            if validation_event == "transition":
                if (
                    getattr(agent.network, "production_gate_version", "none")
                    in POST_FEASIBILITY_RESIDUAL_GATE_VERSIONS
                ):
                    agent.network.set_production_state_gate_frozen(True)
                    agent.network.set_production_flow_commit_residual_enabled(True)
                transition_metadata = {
                    **_single_objective_checkpoint_metadata(
                        phase_controller
                    ),
                    "feature_dimensions": observation.feature_dimensions,
                    "edge_feature_dimensions": (
                        observation.edge_feature_dimensions
                    ),
                    "seed": config["seed"],
                    "phase_transition_episode": completed_episodes,
                    "validation": validation_row,
                }
                agent.save(phase1_checkpoint, metadata=transition_metadata)
                row["candidate_status"] = "phase_transition"
                update_rows[-1]["candidate_status"] = "phase_transition"
            if _checkpoint_eligible_validation_event(
                validation_event,
                phase_controller.quality_checkpoint_promotion,
            ) and (
                phase_controller.quality_checkpoint_promotion
                != SINGLE_OBJECTIVE_PROMOTION_MODE
            ):
                accepted_metadata = {
                    **_checkpoint_protocol_metadata(config),
                    **_single_objective_checkpoint_metadata(
                        phase_controller
                    ),
                    "checkpoint_role": (
                        "single_seed_development_pareto"
                        if _development_acceptance_enabled(config)
                        else "shadow_best"
                    ),
                    "development_scope": (
                        "single_seed_development"
                        if _development_acceptance_enabled(config)
                        else None
                    ),
                    "formal_eligible": False
                    if _development_acceptance_enabled(config)
                    else True,
                    "warm_start": warm_start_report,
                    "seed": config["seed"],
                    "accepted_episode": completed_episodes,
                    "quality_score": normalized_quality_score,
                    "single_objective_name": (
                        phase_controller.single_objective_name
                    ),
                    "single_objective_statistic": (
                        phase_controller.single_objective_window_statistic
                        if phase_controller.quality_checkpoint_promotion
                        == SINGLE_OBJECTIVE_PROMOTION_MODE
                        else None
                    ),
                    "single_objective_value": (
                        phase_controller.accepted_single_objective_value
                    ),
                    "validation": validation_row,
                }
                agent.save(
                    accepted_checkpoint,
                    metadata=accepted_metadata,
                )
                if not _development_acceptance_enabled(config):
                    shutil.copyfile(accepted_checkpoint, best_checkpoint)
                best_score = score
                best_validation = validation_row
                is_new_best = True
                if validation_event != "transition":
                    row["candidate_status"] = "promoted"
                    update_rows[-1]["candidate_status"] = "promoted"
            elif validation_event in {
                "not_promoted",
                "rejected",
                "audit_rejected",
                "audit_passed_not_accepted",
            }:
                row["candidate_status"] = "not_promoted"
                update_rows[-1]["candidate_status"] = "not_promoted"
            if bool(stability["rollback"]):
                if not safe_checkpoint.exists():
                    raise RuntimeError(
                        "catastrophic rollback requested before a safe "
                        "checkpoint was established"
                    )
                agent.load(
                    safe_checkpoint,
                    load_optimizer=True,
                )
                if phase_controller.quality_checkpoint_promotion == SINGLE_OBJECTIVE_PROMOTION_MODE:
                    phase_controller.reset_single_objective_window()
                if (
                    getattr(agent.network, "production_gate_version", "none")
                    in POST_FEASIBILITY_RESIDUAL_GATE_VERSIONS
                ):
                    agent.network.set_production_state_gate_frozen(True)
                    agent.network.set_production_flow_commit_residual_enabled(True)
                row["candidate_status"] = "catastrophic_rolled_back"
                update_rows[-1]["candidate_status"] = (
                    "catastrophic_rolled_back"
                )
            agent.set_learning_rate(
                stability_controller.current_learning_rate
            )
            if validation_event == "transition":
                stability_controller.reset_plateau()
            if not phase_controller.enabled and (
                best_score is None or score < best_score
            ):
                best_score = score
                best_validation = validation_row
                agent.save(
                    accepted_checkpoint,
                    metadata={
                        **_checkpoint_protocol_metadata(config),
                        "feature_dimensions": (
                            observation.feature_dimensions
                        ),
                        "edge_feature_dimensions": (
                            observation.edge_feature_dimensions
                        ),
                        "seed": config["seed"],
                        "smoke": smoke,
                        "online_instances": use_online_instances,
                        "generator_version": (
                            config["generator"]["version"]
                            if use_online_instances
                            else None
                        ),
                        "best_episode": completed_episodes,
                        "validation": validation_row,
                    },
                )
                shutil.copyfile(accepted_checkpoint, best_checkpoint)
                is_new_best = True
            dashboard.log_validation(
                validation_row,
                best_validation=best_validation,
                phase_state=phase_controller.as_dict(),
            )
            if validation_event in {
                "transition",
                "promoted",
                "audit_required",
                "audit_rejected",
                "audit_passed_not_accepted",
                "not_promoted",
                "accepted",
                "rejected",
            }:
                dashboard.log_event(
                    f"episode {completed_episodes}: "
                    f"validation event={validation_event}"
                )
            if is_new_best:
                dashboard.log_event(
                    f"episode {completed_episodes}: new best checkpoint"
                )
            if dashboard.should_capture_diagnostic(
                validation_event=validation_event,
                is_new_best=is_new_best,
            ):
                try:
                    trace = evaluate_representative_diagnostic(
                        config,
                        dataset_name=validation_split,
                        ppo_agent=agent,
                        instance_index=int(
                            visdom_settings[
                                "representative_instance_index"
                            ]
                        ),
                    )
                    dashboard.log_diagnostic(
                        trace,
                        completed_episodes=completed_episodes,
                    )
                except Exception as error:
                    dashboard.log_event(
                        "representative diagnostic failed at episode "
                        f"{completed_episodes}: {error}"
                    )
                    if bool(visdom_settings["fail_fast"]):
                        raise
            print(
                json.dumps(
                    {"validation": validation_row},
                    ensure_ascii=False,
                )
            )
        dashboard.log_update(
            update_rows[-1],
            [row],
            phase_controller.as_dict(),
        )
        print(json.dumps(row, ensure_ascii=False))
    if (
        config["training"].get("ablation_variant") == "Q13"
        and phase_controller.phase_transition_episode is not None
    ):
        (
            final_accepted_sampled_validation,
            final_accepted_sampled_validation_source,
            reran_final_sampled_validation,
        ) = _resolve_final_accepted_sampled_validation(
            config,
            dataset_name=validation_split,
            ppo_agent=agent,
            instance_limit=validation_limit,
            sampling_seeds=_official_evaluation_sampling_seeds(config),
            final_episode=episodes,
            sampled_episode=last_sampled_validation_episode,
            validation_event=last_validation_event,
            sampled_validation=last_sampled_validation,
        )
        if reran_final_sampled_validation:
            stability_controller.sampled_validation_runs += 1
    single_objective_mode = (
        phase_controller.quality_checkpoint_promotion
        == SINGLE_OBJECTIVE_PROMOTION_MODE
    )
    formal_eligible = (not single_objective_mode) and (
        not phase_controller.enabled
        or (
            phase_controller.phase_transition_episode is not None
            and (
                phase_controller.quality_checkpoint_promotion
                not in PARETO_PROMOTION_MODES
                or phase_controller.accepted_pareto_hv is not None
            )
        )
    ) and not _development_acceptance_enabled(config)
    experiment_candidate_accepted = bool(
        single_objective_mode
        and accepted_checkpoint.exists()
        and phase_controller.accepted_single_objective_value is not None
    )
    development_accepted = bool(
        _development_acceptance_enabled(config)
        and accepted_checkpoint.exists()
        and phase_controller.accepted_pareto_hv is not None
    )
    if (
        not formal_eligible
        and not _development_acceptance_enabled(config)
        and not single_objective_mode
        and phase_controller.formal_training_status
        not in {
            "feasibility_not_reached",
            "pareto_baseline_not_reached",
            "single_objective_98_candidate_not_reached",
        }
    ):
        raise RuntimeError("invalid hierarchical training state")
    if formal_eligible and (best_validation is None or best_score is None):
        raise RuntimeError("training completed without validation")
    q13_final_sampled_metadata = (
        {
            "final_accepted_sampled_validation": (
                final_accepted_sampled_validation
            ),
            "final_accepted_sampled_validation_source": (
                final_accepted_sampled_validation_source
            ),
            "final_accepted_checkpoint_episode": (
                phase_controller.accepted_quality_episode
            ),
        }
        if config["training"].get("ablation_variant") == "Q13"
        else {}
    )
    final_metadata = {
            **_checkpoint_protocol_metadata(config),
            "feature_dimensions": observation.feature_dimensions,
            "edge_feature_dimensions": (
                observation.edge_feature_dimensions
            ),
            "seed": config["seed"],
            "smoke": smoke,
            "online_instances": use_online_instances,
            "generator_version": (
                config["generator"]["version"]
                if use_online_instances
                else None
            ),
            "best_checkpoint": (
                str(best_checkpoint) if best_checkpoint.exists() else None
            ),
            "best_validation": best_validation,
            "best_feasibility_checkpoint": (
                str(best_feasibility_checkpoint)
                if best_feasibility_checkpoint.exists()
                else None
            ),
            "best_feasibility_validation": (
                best_feasibility_validation
            ),
            "validation_stability": stability_controller.as_dict(),
            "preference_stage_controller": (
                preference_stage_controller.as_dict()
                if preference_stage_controller is not None
                else None
            ),
            "last_sampled_validation": last_sampled_validation,
            "formal_training_status": (
                phase_controller.formal_training_status
            ),
            "training_phase": phase_controller.as_dict(),
            "formal_eligible": formal_eligible,
            "experiment_candidate_accepted": experiment_candidate_accepted,
            "development_accepted": development_accepted,
            "development_scope": (
                "single_seed_development"
                if _development_acceptance_enabled(config)
                else None
            ),
            "warm_start": warm_start_report,
            **q13_final_sampled_metadata,
        }
    checkpoint: Path | None
    last_candidate_checkpoint: Path | None
    agent.save(
        last_checkpoint,
        metadata={**final_metadata, "checkpoint_role": "last_online"},
    )
    if single_objective_mode:
        checkpoint = None
        last_candidate_checkpoint = None
    elif formal_eligible:
        if not accepted_checkpoint.exists():
            raise RuntimeError(
                "formal training completed without a shadow-best checkpoint"
            )
        checkpoint = run_directory / "checkpoint.pt"
        last_candidate_checkpoint = None
        shutil.copyfile(accepted_checkpoint, checkpoint)
        shutil.copyfile(accepted_checkpoint, best_checkpoint)
    else:
        checkpoint = None
        last_candidate_checkpoint = (
            run_directory / "last_candidate_checkpoint.pt"
        )
        shutil.copyfile(last_checkpoint, last_candidate_checkpoint)
    final_checkpoint_evaluation = None
    checkpoint_sha256 = None
    if single_objective_mode and accepted_checkpoint.exists():
        final_checkpoint_evaluation = _reevaluate_checkpoint_from_disk(
            config,
            checkpoint=accepted_checkpoint,
            bootstrap_observation=bootstrap_observation,
            dataset_name=validation_split,
            instance_limit=phase_controller.single_objective_audit_instance_limit,
            sampling_seeds=[],
            greedy_only=True,
        )
        try:
            _assert_single_objective_checkpoint_evaluation(
                phase_controller,
                final_checkpoint_evaluation,
            )
        except Exception as error:
            invalidated_checkpoint = (
                run_directory / "invalidated_accepted_checkpoint.pt"
            )
            accepted_checkpoint.replace(invalidated_checkpoint)
            write_csv(run_directory / "train_log.csv", rows)
            write_csv(run_directory / "update_log.csv", update_rows)
            write_csv(run_directory / "validation_log.csv", validation_rows)
            write_csv(
                run_directory / "single_objective_audit_log.csv",
                single_objective_audit_rows,
            )
            write_csv(
                run_directory / "single_objective_audit_failures.csv",
                single_objective_audit_failure_rows,
            )
            write_json(
                run_directory / "failure.json",
                {
                    "status": "accepted_checkpoint_invalidated",
                    "error": str(error),
                    "invalidated_checkpoint": str(invalidated_checkpoint),
                    "formal_eligible": False,
                },
            )
            raise RuntimeError(
                "single-objective accepted checkpoint failed final audit"
            ) from error
    elif checkpoint is not None:
        checkpoint_sha256 = _checkpoint_sha256(checkpoint)
        accepted_sha256 = _checkpoint_sha256(accepted_checkpoint)
        best_sha256 = _checkpoint_sha256(best_checkpoint)
        if len({checkpoint_sha256, accepted_sha256, best_sha256}) != 1:
            raise RuntimeError(
                "official, accepted, and best checkpoint hashes diverged"
            )
        final_checkpoint_evaluation = _reevaluate_checkpoint_from_disk(
            config,
            checkpoint=checkpoint,
            bootstrap_observation=bootstrap_observation,
            dataset_name=validation_split,
            instance_limit=validation_limit,
            sampling_seeds=_official_evaluation_sampling_seeds(config),
        )
    summary_checkpoint = (
        accepted_checkpoint
        if single_objective_mode and accepted_checkpoint.exists()
        else checkpoint or last_checkpoint
    )
    summary_provenance = build_provenance(
        config,
        dataset_manifest_path=_validation_manifest_path(config),
        checkpoint_path=summary_checkpoint,
        checkpoint_metadata=_checkpoint_protocol_metadata(config),
    )
    write_csv(run_directory / "train_log.csv", rows)
    write_csv(run_directory / "update_log.csv", update_rows)
    write_csv(run_directory / "validation_log.csv", validation_rows)
    if single_objective_mode:
        write_csv(
            run_directory / "single_objective_audit_log.csv",
            single_objective_audit_rows,
        )
        write_csv(
            run_directory / "single_objective_audit_failures.csv",
            single_objective_audit_failure_rows,
        )
    write_json(
        run_directory / "summary.json",
        {
            "episodes": episodes,
            "online_instances": use_online_instances,
            "parallel_envs": 1,
            "updates": len(update_rows),
            "transitions": sum(
                int(row["transition_count"]) for row in update_rows
            ),
            "environment_steps": sum(
                int(row["steps"]) for row in rows
            ),
            "forced_actions": sum(
                int(row["forced_actions"]) for row in rows
            ),
            "forced_action_ratio": (
                sum(int(row["forced_actions"]) for row in rows)
                / sum(int(row["steps"]) for row in rows)
                if sum(int(row["steps"]) for row in rows) > 0
                else 0.0
            ),
            "forced_action_diagnostics": _forced_action_summary(rows),
            "mean_policy_steps_per_episode": (
                float(np.mean([row["policy_steps"] for row in rows]))
                if rows
                else 0.0
            ),
            "unique_instance_count": len(set(instance_ids)),
            "checkpoint": _run_relative_checkpoint(checkpoint, run_directory),
            "checkpoint_sha256": checkpoint_sha256,
            "provenance": summary_provenance,
            "final_checkpoint_evaluation": final_checkpoint_evaluation,
            "accepted_checkpoint": (
                _run_relative_checkpoint(accepted_checkpoint, run_directory)
                if accepted_checkpoint.exists()
                and not _development_acceptance_enabled(config)
                else None
            ),
            "development_accepted_pareto_checkpoint": (
                _run_relative_checkpoint(accepted_checkpoint, run_directory)
                if development_accepted
                else None
            ),
            "development_accepted": development_accepted,
            "development_failure_reason": (
                None
                if development_accepted
                else phase_controller.last_promotion_diagnostics.get(
                    "promotion_decision_reason",
                    phase_controller.formal_training_status,
                )
            ),
            "development_scope": (
                "single_seed_development"
                if _development_acceptance_enabled(config)
                else None
            ),
            "formal_eligible": formal_eligible,
            "experiment_candidate_accepted": experiment_candidate_accepted,
            "warm_start": warm_start_report,
            "last_checkpoint": _run_relative_checkpoint(last_checkpoint, run_directory),
            "safe_checkpoint": (
                _run_relative_checkpoint(safe_checkpoint, run_directory) if safe_checkpoint.exists() else None
            ),
            "last_candidate_checkpoint": (
                str(last_candidate_checkpoint)
                if last_candidate_checkpoint is not None
                else None
            ),
            "best_checkpoint": (
                str(best_checkpoint) if best_checkpoint.exists() else None
            ),
            "best_validation": best_validation,
            "best_feasibility_checkpoint": (
                str(best_feasibility_checkpoint)
                if best_feasibility_checkpoint.exists()
                else None
            ),
            "best_feasibility_validation": (
                best_feasibility_validation
            ),
            "best_feasibility_episode": (
                best_feasibility_validation["episode"]
                if best_feasibility_validation is not None
                else None
            ),
            "feasibility_rollbacks": (
                stability_controller.feasibility_rollbacks
            ),
            "learning_rate_decays": (
                stability_controller.learning_rate_decays
            ),
            "phase1_checkpoint": (
                str(phase1_checkpoint)
                if phase1_checkpoint.exists()
                else None
            ),
            "formal_training_status": (
                phase_controller.formal_training_status
            ),
            "training_phase": phase_controller.as_dict(),
            "single_objective_audit": (
                {
                    "daily_validation_instance_limit": validation_limit,
                    "audit_instance_limit": (
                        phase_controller.single_objective_audit_instance_limit
                    ),
                    "audit_count": len(single_objective_audit_rows),
                    "audit_failure_row_count": len(
                        single_objective_audit_failure_rows
                    ),
                    "accepted_status": phase_controller.formal_training_status,
                    "project_formal_completion_target": 1.0,
                }
                if single_objective_mode
                else None
            ),
            "validation_stability": stability_controller.as_dict(),
            "validation_runs": len(validation_rows),
            "sampled_validation_runs": (
                stability_controller.sampled_validation_runs
            ),
            "last_sampled_validation": last_sampled_validation,
            **q13_final_sampled_metadata,
            "late_500_episode_diagnostics": (
                _late_training_diagnostics(rows)
            ),
            "ablation_gate": _ablation_gate_summary(
                config,
                rows,
                validation_rows,
                stability_controller,
                best_feasibility_instance_rows,
            ),
            "visdom": {
                "enabled": bool(dashboard.enabled),
                "connected": bool(dashboard.connected),
                "environment": dashboard.environment,
                "event_log": (
                    str(run_directory / "visdom_events.log")
                    if dashboard.enabled
                    else None
                ),
            },
            "last_episode": rows[-1],
            "last_update": update_rows[-1],
            "policy_head_diagnostics": agent.policy_head_diagnostics(),
        },
    )
    dashboard.log_event(
        "training completed with status="
        f"{phase_controller.formal_training_status}"
    )
    dashboard.close()
    return run_directory


def _train_parallel(
    config: dict,
    *,
    smoke: bool,
    run_name: str | None,
    episodes: int,
    parallel_envs: int,
    validation_parallel_envs: int,
    initial_checkpoint: str | Path | None = None,
    warm_start_checkpoint: str | Path | None = None,
) -> Path:
    template = load_instance_yaml(
        project_path(config["paths"]["fixed_instance"])
    )
    bootstrap_environment = AssemblySchedulingEnv(config)
    bootstrap_observation = bootstrap_environment.reset(template)
    network = build_actor_critic(
        bootstrap_observation,
        config["network"],
    )
    agent = PPOAgent(network, config["ppo"], device=config["device"])
    warm_start_report = None
    initial_metadata: dict[str, Any] = {}
    if initial_checkpoint is not None:
        initial_metadata = agent.load(initial_checkpoint, load_optimizer=True)
        warm_start_report = _restore_e2_7_resume_provenance(
            agent, initial_metadata, config
        )
    elif warm_start_checkpoint is not None:
        warm_start_settings = config["training"].get("warm_start", {})
        warm_start_report = agent.warm_start_from_e1(
            warm_start_checkpoint,
            expected_shared_parameter_count=int(
                warm_start_settings.get("expected_shared_parameter_count", 116)
            ),
        )
        warm_start_report = agent.verify_warm_start_canonical_identity(
            bootstrap_observation,
            bootstrap_environment.get_action_mask(),
            source_checkpoint=warm_start_checkpoint,
        )
    run_directory = create_run_directory(
        project_path(config["paths"]["result_root"]),
        label="train_smoke_parallel" if smoke else "train_parallel",
        run_name=run_name,
    )
    write_config(run_directory, config)
    if warm_start_report is not None:
        write_json(run_directory / "warm_start_mapping.json", warm_start_report)
    visdom_settings = resolve_visdom_settings(config)
    dashboard = create_training_dashboard(
        config=config,
        run_directory=run_directory,
        total_episodes=episodes,
    )
    validation_split = str(
        config["training"]["validation_split"]
    )
    validation_interval = int(
        config["training"]["validation_interval_episodes"]
    )
    validation_limit = (
        config["training"]["smoke_validation_instance_limit"]
        if smoke
        else config["training"]["validation_instance_limit"]
    )
    validation_limit = (
        None if validation_limit is None else int(validation_limit)
    )
    _validate_single_objective_validation_protocol(
        config, smoke=smoke, validation_limit=validation_limit
    )
    step_limit = (
        int(config["training"]["smoke_rollout_steps"])
        if smoke
        else None
    )
    runner_worker_count = max(
        parallel_envs,
        validation_parallel_envs,
    )
    rows: list[dict] = []
    update_rows: list[dict] = []
    validation_rows: list[dict] = []
    pareto_validation_rows: list[dict] = []
    single_objective_audit_rows: list[dict] = []
    single_objective_audit_failure_rows: list[dict] = []
    pareto_candidate_rows: list[dict] = []
    e2_3_failure_replay_rows: list[dict] = []
    latest_e2_3_failure_replay: dict[str, object] | None = None
    e2_7_safety_replay_rows: list[dict[str, Any]] = []
    latest_e2_7_safety_replay: dict[str, Any] | None = None
    latest_rejected_candidate: dict[str, Any] | None = None
    e2_7_heldout_candidate_rows: list[dict] = []
    latest_e2_7_heldout_report: dict[str, object] | None = None
    tiered_gate_policy = _tiered_training_gates(config)
    tiered_protocol = tiered_gate_policy is not None
    tiered_monitoring_warnings: list[dict[str, Any]] = []
    best_safe_candidate_rank: tuple[float, float, int] | None = None
    final_acceptance: dict[str, Any] | None = None
    instance_ids: list[str] = []
    best_checkpoint = run_directory / "best_checkpoint.pt"
    best_feasibility_checkpoint = (
        run_directory / "best_feasibility_checkpoint.pt"
    )
    phase1_checkpoint = run_directory / "phase1_checkpoint.pt"
    accepted_checkpoint = _accepted_checkpoint_path(run_directory, config)
    candidate_checkpoint = run_directory / "single_objective_candidate_checkpoint.pt"
    safe_checkpoint = run_directory / "safe_checkpoint.pt"
    anchor_safe_checkpoint = run_directory / "anchor_safe_checkpoint.pt"
    full_grid_safe_checkpoint = (
        run_directory / "full_grid_safe_checkpoint.pt"
    )
    best_safe_candidate_checkpoint = (
        run_directory / "best_safe_candidate_checkpoint.pt"
    )
    last_safe_checkpoint = run_directory / "last_safe_checkpoint.pt"
    last_checkpoint = run_directory / "last_checkpoint.pt"
    best_validation: dict | None = None
    best_feasibility_validation: dict | None = None
    best_feasibility_instance_rows: list[dict] = []
    last_sampled_validation: dict | None = None
    last_sampled_validation_episode: int | None = None
    last_validation_event: str | None = None
    final_accepted_sampled_validation: dict | None = None
    final_accepted_sampled_validation_source: str | None = None
    best_score: tuple[float, float, float, float] | None = None
    phase_controller = TrainingPhaseController.from_config(config)
    stability_controller = ValidationStabilityController.from_config(config)
    preference_stage_controller = E2_7PreferenceStageController.from_config(config)
    if preference_stage_controller is not None and initial_checkpoint is not None:
        preference_stage_controller.restore(
            initial_metadata.get("preference_stage_controller")
        )
    total_transitions = 0
    total_environment_steps = 0
    total_forced_actions = 0
    total_worker_step_commands = 0
    total_worker_local_physical_forced_actions = 0
    total_sampling_time = 0.0
    total_inference_time = 0.0
    total_update_time = 0.0
    update_id = 0
    quality_update_id = 0
    pareto_mode = (
        phase_controller.quality_checkpoint_promotion
        in PARETO_PROMOTION_MODES
    )
    pareto_settings = (
        _pareto_promotion_settings(config) if pareto_mode else None
    )
    pareto_safety_guard = ParetoSafetyGuard.from_config(config)
    with ParallelEpisodeRunner(
        config=config,
        template=template,
        episode_count=training_base_instance_count(config, episodes),
        worker_count=runner_worker_count,
    ) as runner:
        safe_state_pool_report = _build_e2_7_safe_state_pool(
            config,
            agent=agent,
            runner=runner,
        )
        if safe_state_pool_report is not None:
            write_json(
                run_directory / "safe_dual_legal_state_pool.json",
                safe_state_pool_report["gate"],
            )
            write_json(
                run_directory / "e2_7_preference_state_pools.json",
                safe_state_pool_report,
            )
        for batch_start in range(0, episodes, parallel_envs):
            reward_phase = phase_controller.phase
            episode_indices = list(
                range(
                    batch_start,
                    min(batch_start + parallel_envs, episodes),
                )
            )
            try:
                rollout = runner.collect_training_batch(
                    agent,
                    episode_indices,
                    gamma=float(config["ppo"]["gamma"]),
                    gae_lambda=float(config["ppo"]["gae_lambda"]),
                    step_limit=step_limit,
                    reward_phase=reward_phase,
                )
            except RuntimeError as error:
                text_error = str(error).lower()
                if tiered_protocol and (
                    "illegal action" in text_error or "non-finite" in text_error
                ):
                    latest_rejected_candidate = _save_rejected_candidate(
                        config,
                        run_directory=run_directory,
                        agent=agent,
                        update_id=update_id,
                        stage=(
                            preference_stage_controller.stage
                            if preference_stage_controller is not None
                            else None
                        ),
                        failure_source="rollout_hard_failure",
                        failure_cell={"error": str(error)},
                        preference_stage_controller=(
                            preference_stage_controller
                        ),
                    )
                    _write_tiered_hard_failure_summary(
                        run_directory,
                        config,
                        reason="rollout_hard_failure",
                        update_id=update_id,
                        rejected_candidate=latest_rejected_candidate,
                    )
                raise
            update_start = time.perf_counter()
            if rollout.transition_count == 0:
                raise RuntimeError(
                    "training batch contains no policy transitions after "
                    "forced action compression"
                )
            preference_stage = (
                preference_stage_controller.apply(agent)
                if preference_stage_controller is not None
                else None
            )
            try:
                losses = agent.update(rollout.buffer, reward_phase=reward_phase)
            except (FloatingPointError, RuntimeError) as error:
                text_error = str(error).lower()
                hard_error = (
                    tiered_protocol
                    and (
                        "non-finite" in text_error
                        or "canonical identity" in text_error
                        or "canonical policy" in text_error
                        or "illegal action" in text_error
                    )
                )
                if hard_error:
                    latest_rejected_candidate = _save_rejected_candidate(
                        config,
                        run_directory=run_directory,
                        agent=agent,
                        update_id=update_id + 1,
                        stage=(
                            preference_stage_controller.stage
                            if preference_stage_controller is not None
                            else None
                        ),
                        failure_source="optimizer_or_canonical_hard_failure",
                        failure_cell={"error": str(error)},
                        preference_stage_controller=(
                            preference_stage_controller
                        ),
                    )
                    _write_tiered_hard_failure_summary(
                        run_directory,
                        config,
                        reason="optimizer_or_canonical_hard_failure",
                        update_id=update_id + 1,
                        rejected_candidate=latest_rejected_candidate,
                    )
                raise
            update_time = time.perf_counter() - update_start
            update_id += 1
            hard_failure = (
                _tiered_hard_failure_reason(losses, config)
                if tiered_protocol
                else None
            )
            if hard_failure is not None:
                latest_rejected_candidate = _save_rejected_candidate(
                    config,
                    run_directory=run_directory,
                    agent=agent,
                    update_id=update_id,
                    stage=(
                        preference_stage_controller.stage
                        if preference_stage_controller is not None
                        else None
                    ),
                    failure_source=hard_failure,
                    failure_cell={"losses": losses, "hard_failure": hard_failure},
                    preference_stage_controller=preference_stage_controller,
                )
                _write_tiered_hard_failure_summary(
                    run_directory,
                    config,
                    reason=hard_failure,
                    update_id=update_id,
                    rejected_candidate=latest_rejected_candidate,
                )
                raise RuntimeError(f"hard training gate failed: {hard_failure}")
            stage_failure: dict[str, Any] | None = None
            if preference_stage_controller is not None:
                stage_proposal = preference_stage_controller.propose(
                    losses, update_id=update_id
                )
            else:
                stage_proposal = None
            if stage_proposal is not None and bool(
                stage_proposal["transition_requested"]
            ):
                if pareto_settings is None:
                    raise RuntimeError(
                        "E2.7 stage entry requires Pareto safety settings"
                    )
                entry_rows, entry_snapshot = _evaluate_pareto_preferences(
                    config,
                    preferences=tuple(
                        simplex_lattice(5, include=(CANONICAL_PREFERENCE,))
                    ),
                    scope="full_grid_22",
                    ppo_agent=agent,
                    runner=runner,
                    dataset_name=validation_split,
                    instance_limit=validation_limit,
                    validation_parallel_envs=validation_parallel_envs,
                    update_id=update_id,
                    completed_episodes=episode_indices[-1] + 1,
                    fatigue_tolerance=float(
                        pareto_settings["fatigue_absolute_tolerance"]
                    ),
                    canonical_rows=None,
                )
                if tiered_protocol and not bool(
                    entry_snapshot["evaluation_integrity_pass"]
                ):
                    latest_rejected_candidate = _save_rejected_candidate(
                        config,
                        run_directory=run_directory,
                        agent=agent,
                        update_id=update_id,
                        stage=str(stage_proposal["stage"]),
                        failure_source="evaluation_integrity_failed",
                        failure_cell=entry_snapshot,
                        preference_stage_controller=preference_stage_controller,
                    )
                    _write_tiered_hard_failure_summary(
                        run_directory,
                        config,
                        reason="evaluation_integrity_failed",
                        update_id=update_id,
                        rejected_candidate=latest_rejected_candidate,
                    )
                    raise RuntimeError(
                        "hard training gate failed: evaluation_integrity_failed"
                    )
                entry_safe = bool(
                    entry_snapshot["physical_safety_pass"]
                    if tiered_protocol
                    else entry_snapshot["coverage_pass"]
                    and entry_snapshot["all_safe"]
                )
                if stage_proposal["stage"] == "gate":
                    entry_response_pass = bool(
                        entry_snapshot["centered_gate_pass"]
                    )
                    pair_response = None
                else:
                    pair_response = _evaluate_e2_7_full_grid_pair_response(
                        config,
                        agent=agent,
                        runner=runner,
                        validation_parallel_envs=validation_parallel_envs,
                        minimum_rate=(
                            preference_stage_controller
                            .minimum_production_pair_correct_rate
                        ),
                    )
                    entry_response_pass = bool(
                        pair_response["pass"]
                        and float(
                            losses.get(
                                "canonical_identity_max_abs_error", math.inf
                            )
                        )
                        <= 1e-8
                    )
                entry_snapshot.update(
                    {
                        "stage_entry_full_grid": True,
                        "stage_entry_safe": entry_safe,
                        "stage_entry_response_pass": entry_response_pass,
                        "stage_entry_from": stage_proposal["stage"],
                        "stage_entry_to": stage_proposal["next_stage"],
                        "stage_entry_pair_response": pair_response,
                    }
                )
                pareto_validation_rows.append(entry_snapshot)
                pareto_candidate_rows.extend(
                    {
                        **row,
                        "update_id": update_id,
                        "completed_episodes": episode_indices[-1] + 1,
                        "stage_entry_full_grid": True,
                    }
                    for row in entry_rows
                )
                if tiered_protocol:
                    (
                        best_safe_candidate_rank,
                        _entry_promoted_best_safe_candidate,
                    ) = _save_tiered_safe_candidate(
                        config,
                        agent=agent,
                        snapshot=entry_snapshot,
                        completed_episodes=episode_indices[-1] + 1,
                        last_safe_checkpoint=last_safe_checkpoint,
                        best_safe_candidate_checkpoint=(
                            best_safe_candidate_checkpoint
                        ),
                        best_rank=best_safe_candidate_rank,
                        preference_stage_controller=preference_stage_controller,
                    )
                    entry_snapshot[
                        "tiered_best_safe_candidate_promoted"
                    ] = _entry_promoted_best_safe_candidate
                if not entry_safe:
                    latest_rejected_candidate = _save_rejected_candidate(
                        config,
                        run_directory=run_directory,
                        agent=agent,
                        update_id=update_id,
                        stage=str(stage_proposal["stage"]),
                        failure_source="e2_7_stage_entry_safety",
                        failure_cell=entry_snapshot,
                        preference_stage_controller=(
                            preference_stage_controller
                        ),
                    )
                    rollback_source = next(
                        (
                            path
                            for path in (
                                last_safe_checkpoint,
                                full_grid_safe_checkpoint,
                                anchor_safe_checkpoint,
                                safe_checkpoint,
                            )
                            if path.exists()
                        ),
                        None,
                    )
                    if rollback_source is None:
                        raise RuntimeError(
                            "E2.7 stage-entry safety failed before a safe "
                            "rollback checkpoint was established"
                        )
                    failure_learning_rate = agent.learning_rate
                    _restore_e2_7_rollback_checkpoint(
                        agent,
                        rollback_source,
                        preference_stage_controller,
                        restore_stage_controller=not tiered_protocol,
                    )
                    if tiered_protocol:
                        preference_stage_controller.apply(agent)
                        guarded_learning_rate = max(
                            float(pareto_settings["safety_guard_minimum_learning_rate"]),
                            failure_learning_rate
                            * float(
                                pareto_settings[
                                    "safety_guard_learning_rate_decay_factor"
                                ]
                            ),
                        )
                        agent.set_learning_rate(guarded_learning_rate)
                        stability_controller.current_learning_rate = guarded_learning_rate
                    stage_outcome = (
                        preference_stage_controller.commit(
                            stage_proposal, transition_confirmed=False
                        )
                        if tiered_protocol
                        else {"transitioned": False, "failure": None}
                    )
                else:
                    stage_outcome = preference_stage_controller.commit(
                        stage_proposal,
                        transition_confirmed=(
                            entry_safe if tiered_protocol else entry_response_pass
                        ),
                    )
                    if stage_outcome.get("warning"):
                        tiered_monitoring_warnings.append(
                            {
                                "update_id": update_id,
                                "stage": stage_proposal["stage"],
                                "warning": stage_outcome["warning"],
                            }
                        )
                if stage_outcome["transitioned"]:
                    preference_stage_controller.apply(agent)
                    agent.save(
                        full_grid_safe_checkpoint,
                        metadata={
                            **_checkpoint_protocol_metadata(config),
                            "checkpoint_role": "full_grid_safe",
                            "safe_episode": episode_indices[-1] + 1,
                            "pareto_snapshot": entry_snapshot,
                            "preference_stage_controller": (
                                preference_stage_controller.as_dict()
                            ),
                        },
                    )
                if stage_outcome["failure"]:
                    stage_failure = {
                        "reason": stage_outcome["failure"],
                        "stage": stage_proposal["stage"],
                        "entry_snapshot": entry_snapshot,
                    }
            elif stage_proposal is not None:
                stage_outcome = preference_stage_controller.commit(stage_proposal)
                if stage_outcome["failure"]:
                    stage_failure = {
                        "reason": stage_outcome["failure"],
                        "stage": stage_proposal["stage"],
                        "entry_snapshot": None,
                    }
            if reward_phase == "quality":
                quality_update_id += 1
            if (
                _e2_3_failure_replay_cells(config)
                and update_id % 5 == 0
            ):
                replay_rows, latest_e2_3_failure_replay = (
                    _evaluate_e2_3_failure_replay(
                        config,
                        agent=agent,
                        runner=runner,
                        update_id=update_id,
                    )
                )
                e2_3_failure_replay_rows.extend(replay_rows)
            safety_replay_rolled_back = False
            if (
                preference_stage in {"production_pair", "worker_variance"}
                and _e2_7_safety_replay_cell(config) is not None
            ):
                safety_rows, latest_e2_7_safety_replay = (
                    _evaluate_e2_7_safety_replay(
                        config,
                        agent=agent,
                        runner=runner,
                        update_id=update_id,
                        stage=str(preference_stage),
                    )
                )
                e2_7_safety_replay_rows.extend(safety_rows)
                replay_physical_failure = not bool(
                    latest_e2_7_safety_replay[
                        "physical_safety_pass" if tiered_protocol else "pass"
                    ]
                )
                if replay_physical_failure:
                    failure_cell = dict(safety_rows[0]) if safety_rows else None
                    latest_rejected_candidate = _save_rejected_candidate(
                        config,
                        run_directory=run_directory,
                        agent=agent,
                        update_id=update_id,
                        stage=preference_stage,
                        failure_source="e2_7_safety_replay",
                        failure_cell=failure_cell,
                        preference_stage_controller=(
                            preference_stage_controller
                        ),
                    )
                    rollback_checkpoint = (
                        last_safe_checkpoint
                        if tiered_protocol and last_safe_checkpoint.exists()
                        else full_grid_safe_checkpoint
                    )
                    if not rollback_checkpoint.exists():
                        raise RuntimeError(
                            "E2.7 safety replay failed before a full-grid-safe "
                            "checkpoint was established"
                        )
                    failure_learning_rate = agent.learning_rate
                    _restore_e2_7_rollback_checkpoint(
                        agent,
                        rollback_checkpoint,
                        preference_stage_controller,
                        restore_stage_controller=not tiered_protocol,
                    )
                    if tiered_protocol and preference_stage_controller is not None:
                        preference_stage_controller.apply(agent)
                        guarded_learning_rate = max(
                            float(pareto_settings["safety_guard_minimum_learning_rate"]),
                            failure_learning_rate
                            * float(
                                pareto_settings[
                                    "safety_guard_learning_rate_decay_factor"
                                ]
                            ),
                        )
                        agent.set_learning_rate(guarded_learning_rate)
                        stability_controller.current_learning_rate = guarded_learning_rate
                    safety_replay_rolled_back = True
            transition_count = rollout.transition_count
            total_transitions += transition_count
            total_environment_steps += rollout.environment_step_count
            total_forced_actions += rollout.forced_action_count
            total_worker_step_commands += (
                rollout.worker_step_command_count
            )
            total_worker_local_physical_forced_actions += (
                rollout.worker_local_physical_forced_action_count
            )
            total_sampling_time += rollout.sampling_wall_time_seconds
            total_inference_time += (
                rollout.policy_inference_time_seconds
            )
            total_update_time += update_time
            training_time = (
                rollout.sampling_wall_time_seconds + update_time
            )
            update_row = {
                "update_id": update_id,
                "episode_start": episode_indices[0],
                "episode_end": episode_indices[-1],
                "episode_count": len(episode_indices),
                "parallel_envs": len(episode_indices),
                "transition_count": transition_count,
                "environment_step_count": (
                    rollout.environment_step_count
                ),
                "forced_action_count": rollout.forced_action_count,
                "forced_action_ratio": rollout.forced_action_ratio,
                "worker_step_command_count": (
                    rollout.worker_step_command_count
                ),
                "worker_local_physical_forced_action_count": (
                    rollout.worker_local_physical_forced_action_count
                ),
                "worker_local_physical_forced_share": (
                    rollout.worker_local_physical_forced_share
                ),
                "estimated_worker_step_round_trips_avoided": (
                    rollout.worker_local_physical_forced_action_count
                ),
                "sampling_wall_time_seconds": (
                    rollout.sampling_wall_time_seconds
                ),
                "policy_inference_time_seconds": (
                    rollout.policy_inference_time_seconds
                ),
                "generation_time_seconds": sum(
                    episode.generation_time_seconds
                    for episode in rollout.episodes
                ),
                "environment_step_time_seconds": sum(
                    episode.environment_step_time_seconds
                    for episode in rollout.episodes
                ),
                "ppo_update_time_seconds": update_time,
                "transitions_per_second": (
                    transition_count / training_time
                    if training_time > 0
                    else 0.0
                ),
                "reward_phase": reward_phase,
                "preference_stage": preference_stage,
                "e2_7_safety_replay_pass": (
                    latest_e2_7_safety_replay.get("pass")
                    if latest_e2_7_safety_replay is not None
                    and preference_stage in {"production_pair", "worker_variance"}
                    else None
                ),
                "e2_7_safety_replay_rollback": safety_replay_rolled_back,
                "failure_reason": (
                    stage_failure["reason"] if stage_failure is not None else None
                ),
                "candidate_status": (
                    "pending"
                    if reward_phase == "quality"
                    else "not_applicable"
                ),
                **losses,
            }
            update_rows.append(update_row)
            for episode in rollout.episodes:
                instance_ids.append(episode.instance_id)
                row = {
                    "episode": episode.episode_index,
                    "trajectory_index": episode.episode_index,
                    "base_instance_index": episode.base_instance_index,
                    "preference_slot": episode.preference_slot,
                    "preference_group_id": episode.preference_group_id,
                    "update_id": update_id,
                    "instance_id": episode.instance_id,
                    "instance_seed": episode.metadata["seed"],
                    "pressure_type": episode.metadata[
                        "pressure_type"
                    ],
                    "cost_profile": episode.metadata["cost_profile"],
                    "w_flow": episode.preference.flow,
                    "w_cost": episode.preference.cost,
                    "w_variance": episode.preference.variance,
                    "preference_source": episode.preference_source,
                    "steps": episode.step_count,
                    "policy_steps": episode.policy_step_count,
                    "forced_actions": episode.forced_action_count,
                    "forced_action_ratio": episode.forced_action_ratio,
                    "worker_step_command_count": (
                        episode.worker_step_command_count
                    ),
                    "worker_local_physical_forced_action_count": (
                        episode.worker_local_physical_forced_action_count
                    ),
                    "worker_local_physical_forced_share": (
                        episode.worker_local_physical_forced_share
                    ),
                    "estimated_worker_step_round_trips_avoided": (
                        episode.worker_local_physical_forced_action_count
                    ),
                    "unattributed_forced_reward": (
                        episode.unattributed_forced_reward
                    ),
                    "reward": episode.reward_sum,
                    "reward_base": episode.base_reward_sum,
                    "reward_shaping": episode.reward_components.get(
                        "feasibility_shaping", 0.0
                    )
                    + episode.reward_components.get(
                        "defer_risk_shaping", 0.0
                    ),
                    "reward_training": episode.reward_sum,
                    "expected_reward": episode.expected_reward,
                    "reward_identity_error": (
                        episode.base_reward_sum - episode.expected_reward
                    ),
                    "reward_phase": episode.reward_phase,
                    **{
                        f"reward_{name}": value
                        for name, value in episode.reward_components.items()
                    },
                    "completed_operations": episode.metrics[
                        "completed_operations"
                    ],
                    "time": episode.metrics["time"],
                    **_training_effect_fields(episode.metrics),
                    "parallel_envs": len(episode_indices),
                    "batch_transition_count": transition_count,
                    "sampling_wall_time_seconds": (
                        rollout.sampling_wall_time_seconds
                    ),
                    "policy_inference_time_seconds": (
                        rollout.policy_inference_time_seconds
                    ),
                    "generation_time_seconds": (
                        episode.generation_time_seconds
                    ),
                    "environment_step_time_seconds": (
                        episode.environment_step_time_seconds
                    ),
                    "ppo_update_time_seconds": update_time,
                    "loss_scope": "parallel_episode_batch",
                    "candidate_status": (
                        "pending"
                        if reward_phase == "quality"
                        else "not_applicable"
                    ),
                    **losses,
                }
                rows.append(row)
            if stage_failure is not None:
                latest_rejected_candidate = _save_rejected_candidate(
                    config,
                    run_directory=run_directory,
                    agent=agent,
                    update_id=update_id,
                    stage=str(stage_failure["stage"]),
                    failure_source=str(stage_failure["reason"]),
                    failure_cell=(
                        stage_failure["entry_snapshot"]
                        if isinstance(stage_failure["entry_snapshot"], dict)
                        else update_row
                    ),
                    preference_stage_controller=preference_stage_controller,
                )
                rollback_source = next(
                    (
                        path
                        for path in (
                            full_grid_safe_checkpoint,
                            anchor_safe_checkpoint,
                            safe_checkpoint,
                        )
                        if path.exists()
                    ),
                    None,
                )
                if rollback_source is not None:
                    _restore_e2_7_rollback_checkpoint(
                        agent,
                        rollback_source,
                        preference_stage_controller,
                    )
                agent.save(
                    last_checkpoint,
                    metadata={
                        **_checkpoint_protocol_metadata(config),
                        "checkpoint_role": "last_online",
                        "failure_reason": stage_failure["reason"],
                        "failed_update_id": update_id,
                        "preference_stage_controller": (
                            preference_stage_controller.as_dict()
                            if preference_stage_controller is not None
                            else None
                        ),
                    },
                )
                write_csv(run_directory / "train_log.csv", rows)
                write_csv(run_directory / "update_log.csv", update_rows)
                write_csv(run_directory / "validation_log.csv", validation_rows)
                if pareto_validation_rows:
                    write_csv(
                        run_directory / "pareto_validation_log.csv",
                        pareto_validation_rows,
                    )
                if pareto_candidate_rows:
                    write_csv(
                        run_directory / "pareto_validation_candidates.csv",
                        pareto_candidate_rows,
                    )
                write_json(
                    run_directory / "summary.json",
                    {
                        "training_status": "preference_stage_failed",
                        "failure_reason": stage_failure["reason"],
                        "failed_update_id": update_id,
                        "last_update": update_row,
                        "latest_rejected_candidate": latest_rejected_candidate,
                        "preference_stage_controller": (
                            preference_stage_controller.as_dict()
                            if preference_stage_controller is not None
                            else None
                        ),
                        "last_checkpoint": _run_relative_checkpoint(
                            last_checkpoint, run_directory
                        ),
                    },
                )
                raise RuntimeError(
                    "preference_stage_failed: "
                    f"stage={stage_failure['stage']}, update={update_id}"
                )
            completed_episodes = episode_indices[-1] + 1
            regular_validation_due = (
                completed_episodes % validation_interval == 0
                or completed_episodes == episodes
            )
            should_validate = phase_controller.should_validate(
                regular_validation_due
            )
            if should_validate:
                if validation_parallel_envs > 1:
                    validation_instance_rows, validation = (
                        evaluate_dataset_parallel(
                        config,
                        dataset_name=validation_split,
                        ppo_agent=agent,
                        runner=runner,
                        instance_limit=validation_limit,
                        )
                    )
                else:
                    validation_instance_rows, _, _, validation = (
                        evaluate_dataset(
                        config,
                        dataset_name=validation_split,
                        policy_name="ppo",
                        ppo_agent=agent,
                        instance_limit=validation_limit,
                        )
                    )
                if phase_controller.quality_checkpoint_promotion == SINGLE_OBJECTIVE_PROMOTION_MODE:
                    validation["physical_safety_pass"] = _rows_are_physically_safe(
                        validation_instance_rows, 1e-9
                    )
                validation_row = _validation_log_row(
                    validation,
                    completed_episodes=completed_episodes,
                )
                score = evaluation_selection_key(validation)
                if (
                    phase_controller.quality_checkpoint_promotion
                    == "balanced_guarded_v7"
                ):
                    score = _balanced_guard_score(validation)
                elif (
                    phase_controller.quality_checkpoint_promotion
                    == SINGLE_OBJECTIVE_PROMOTION_MODE
                ):
                    if phase_controller.single_objective_name is None:
                        raise RuntimeError(
                            "single-objective target is not configured"
                        )
                    score = _single_objective_guard_score(
                        validation,
                        phase_controller.single_objective_name,
                    )
                normalized_quality_score = (
                    _normalized_validation_quality_score(
                        score,
                        config["reward"],
                    )
                    if phase_controller.quality_checkpoint_promotion
                    in {
                        "constrained_weighted",
                        "aligned_quality",
                        "balanced_guarded_v7",
                    }
                    else None
                )
                stability = stability_controller.observe_greedy(
                    score,
                    validation["completion_rate"],
                    completed_episodes=completed_episodes,
                    feasibility_phase=reward_phase == "feasibility",
                )
                if tiered_protocol and bool(stability["rollback"]):
                    # Completion volatility is a monitored training signal in
                    # the tiered protocol.  Keep plateau LR control, but never
                    # restore a policy solely because completion dipped.
                    stability["rollback"] = False
                    stability["rollback_suppressed_by_tiered_policy"] = True
                    tiered_monitoring_warnings.append(
                        {
                            "update_id": update_id,
                            "warning": "completion_rollback_suppressed",
                        }
                    )
                if stability_controller.should_run_sampled(
                    final_validation=completed_episodes == episodes,
                    completed_episodes=completed_episodes,
                ) and (
                    phase_controller.quality_checkpoint_promotion
                    not in ({"balanced_guarded_v7"} | PARETO_PROMOTION_MODES)
                ):
                    sampled_validation = _evaluate_sampled_validation(
                        config,
                        dataset_name=validation_split,
                        ppo_agent=agent,
                        instance_limit=validation_limit,
                        sampling_seeds=(
                            stability_controller.sampled_seeds(
                                int(config["seed"])
                            )
                        ),
                        runner=runner,
                        use_parallel=validation_parallel_envs > 1,
                    )
                    stability_controller.sampled_validation_runs += 1
                    last_sampled_validation = sampled_validation
                    last_sampled_validation_episode = completed_episodes
                    _attach_sampled_validation(
                        validation_row,
                        sampled_validation,
                        completed_episodes=completed_episodes,
                    )
                validation_event = phase_controller.observe_validation(
                    validation["completion_rate"],
                    completed_episodes=completed_episodes,
                    score=score,
                    normalized_quality_score=normalized_quality_score,
                    truncated_count=int(validation.get("truncated_count", 0)),
                    schedule_violation_count=int(
                        validation.get("schedule_violation_count", 0)
                    ),
                    physical_safety_pass=bool(
                        validation.get("physical_safety_pass", True)
                    ),
                )
                transitioned_now = validation_event == "transition"
                pareto_guard_rollback_requested = False
                rollback_guard_snapshot: dict[str, Any] | None = None
                if pareto_mode:
                    if pareto_settings is None:
                        raise RuntimeError("Pareto promotion settings are missing")
                    guard_snapshot: dict[str, object] | None = None
                    anchor_due = bool(
                        phase_controller.phase == "quality"
                        and (
                            validation_event == "transition"
                            or completed_episodes == episodes
                            or (
                                reward_phase == "quality"
                                and quality_update_id
                                % int(
                                    pareto_settings[
                                        "anchor_validate_every_updates"
                                    ]
                                )
                                == 0
                            )
                        )
                    )
                    if anchor_due:
                        anchor_rows, anchor_snapshot = (
                            _evaluate_pareto_preferences(
                                config,
                                preferences=_pareto_anchor_preferences(config),
                                scope="anchors_5",
                                ppo_agent=agent,
                                runner=runner,
                                dataset_name=validation_split,
                                instance_limit=validation_limit,
                                validation_parallel_envs=(
                                    validation_parallel_envs
                                ),
                                update_id=update_id,
                                completed_episodes=completed_episodes,
                                fatigue_tolerance=float(
                                    pareto_settings[
                                        "fatigue_absolute_tolerance"
                                    ]
                                ),
                                canonical_rows=validation_instance_rows,
                            )
                        )
                        if (
                            phase_controller.quality_checkpoint_promotion
                            in {
                                "pareto_guarded_e2_3_v1",
                                "pareto_guarded_e2_4_v1",
                                "pareto_guarded_e2_5_v1",
                                "pareto_guarded_e2_6_v1",
                                "pareto_guarded_e2_7_development_v1",
                            }
                        ):
                            anchor_event = "audit_only"
                            anchor_diagnostics: dict[str, object] = {}
                        else:
                            validation_event = (
                                phase_controller.observe_pareto_snapshot(
                                    anchor_snapshot,
                                    completed_episodes=completed_episodes,
                                )
                            )
                            anchor_event = validation_event
                            anchor_diagnostics = dict(
                                phase_controller.last_promotion_diagnostics
                            )
                        anchor_snapshot.update(
                            {
                                "quality_update_id": quality_update_id,
                                "promotion_event": anchor_event,
                                **anchor_diagnostics,
                            }
                        )
                        pareto_validation_rows.append(anchor_snapshot)
                        pareto_candidate_rows.extend(
                            {
                                **row,
                                "update_id": update_id,
                                "completed_episodes": completed_episodes,
                            }
                            for row in anchor_rows
                        )
                        validation_row.update(
                            {
                                "pareto_anchor_mean_hypervolume": (
                                    anchor_snapshot["mean_hypervolume"]
                                ),
                                "pareto_anchor_canonical_quality": (
                                    anchor_snapshot["canonical_quality"]
                                ),
                                "pareto_anchor_all_safe": (
                                    anchor_snapshot["all_safe"]
                                ),
                            }
                        )
                        if pareto_safety_guard is not None:
                            guard_snapshot = anchor_snapshot
                    full_grid_due = bool(
                        not (
                            smoke
                            and _development_acceptance_enabled(config)
                        )
                        and (
                            completed_episodes == episodes
                            or (
                            reward_phase == "quality"
                            and quality_update_id
                            % int(
                                pareto_settings[
                                    "full_grid_validate_every_updates"
                                ]
                            )
                                == 0
                            )
                        )
                    )
                    if full_grid_due:
                        full_rows, full_snapshot = _evaluate_pareto_preferences(
                            config,
                            preferences=tuple(
                                simplex_lattice(
                                    5, include=(CANONICAL_PREFERENCE,)
                                )
                            ),
                            scope="full_grid_22",
                            ppo_agent=agent,
                            runner=runner,
                            dataset_name=validation_split,
                            instance_limit=validation_limit,
                            validation_parallel_envs=validation_parallel_envs,
                            update_id=update_id,
                            completed_episodes=completed_episodes,
                            fatigue_tolerance=float(
                                pareto_settings["fatigue_absolute_tolerance"]
                            ),
                            canonical_rows=validation_instance_rows,
                        )
                        full_snapshot["e2_3_failure_replay_pass"] = bool(
                            latest_e2_3_failure_replay
                            and latest_e2_3_failure_replay.get("pass", False)
                        ) if _e2_3_failure_replay_cells(config) else True
                        full_snapshot["e2_3_failure_replay"] = (
                            latest_e2_3_failure_replay
                        )
                        if (
                            phase_controller.quality_checkpoint_promotion
                            == "pareto_guarded_e2_7_development_v1"
                            and not tiered_protocol
                            and phase_controller.phase == "quality"
                            and _e2_7_local_development_gate_pass(
                                config, full_snapshot, phase_controller
                            )
                        ):
                            (
                                latest_e2_7_heldout_report,
                                heldout_rows,
                            ) = _evaluate_e2_7_heldout_hv(
                                config,
                                validation_snapshot=full_snapshot,
                                ppo_agent=agent,
                                runner=runner,
                                validation_parallel_envs=(
                                    validation_parallel_envs
                                ),
                                update_id=update_id,
                                completed_episodes=completed_episodes,
                                fatigue_tolerance=float(
                                    pareto_settings[
                                        "fatigue_absolute_tolerance"
                                    ]
                                ),
                            )
                            e2_7_heldout_candidate_rows.extend(heldout_rows)
                            full_snapshot["heldout_hv_pass"] = bool(
                                latest_e2_7_heldout_report["pass"]
                            )
                            full_snapshot["heldout_hv_report"] = (
                                latest_e2_7_heldout_report
                            )
                            write_json(
                                run_directory
                                / "e2_7_heldout_comparison.json",
                                latest_e2_7_heldout_report,
                            )
                        elif (
                            phase_controller.quality_checkpoint_promotion
                            == "pareto_guarded_e2_7_development_v1"
                        ):
                            full_snapshot["heldout_hv_pass"] = False
                            full_snapshot["heldout_hv_report"] = {
                                "version": "e2_7_equal_budget_heldout_hv_v1",
                                "pass": False,
                                "reason": (
                                    "deferred_until_final_acceptance"
                                    if tiered_protocol
                                    else "local_development_gate_not_ready"
                                ),
                            }
                            latest_e2_7_heldout_report = dict(
                                full_snapshot["heldout_hv_report"]
                            )
                            write_json(
                                run_directory
                                / "e2_7_heldout_comparison.json",
                                latest_e2_7_heldout_report,
                            )
                        if (
                            phase_controller.quality_checkpoint_promotion
                            == "pareto_guarded_e2_7_development_v1"
                        ):
                            write_json(
                                run_directory / "e2_7_full_grid_report.json",
                                full_snapshot,
                            )
                            write_json(
                                run_directory / "e2_7_gate_flip_report.json",
                                {
                                    "version": "centered_gate_flip_report_v1",
                                    "update_id": update_id,
                                    "completed_episodes": completed_episodes,
                                    "dual_legal_state_count": full_snapshot.get(
                                        "centered_gate_dual_legal_state_count", 0
                                    ),
                                    "flow_cost_flip_count": full_snapshot.get(
                                        "centered_gate_flow_cost_flip_count", 0
                                    ),
                                    "flow_variance_flip_count": full_snapshot.get(
                                        "centered_gate_flow_variance_flip_count", 0
                                    ),
                                    "flow_cost_flip_rate": full_snapshot.get(
                                        "centered_gate_flow_cost_flip_rate", 0.0
                                    ),
                                    "flow_variance_flip_rate": full_snapshot.get(
                                        "centered_gate_flow_variance_flip_rate", 0.0
                                    ),
                                    "extreme_flip_rate": full_snapshot.get(
                                        "centered_gate_extreme_flip_rate", 0.0
                                    ),
                                    "monotonicity_violation_count": (
                                        full_snapshot.get(
                                            "centered_gate_monotonicity_violation_count",
                                            0,
                                        )
                                    ),
                                    "pass": full_snapshot.get(
                                        "centered_gate_pass", False
                                    ),
                                },
                            )
                        if (
                            phase_controller.quality_checkpoint_promotion
                            in {
                                "pareto_guarded_e2_3_v1",
                                "pareto_guarded_e2_4_v1",
                                "pareto_guarded_e2_5_v1",
                                "pareto_guarded_e2_6_v1",
                                "pareto_guarded_e2_7_development_v1",
                            }
                            and not tiered_protocol
                            and phase_controller.phase == "quality"
                        ):
                            validation_event = (
                                phase_controller.observe_pareto_snapshot(
                                    full_snapshot,
                                    completed_episodes=completed_episodes,
                                )
                            )
                            full_event = validation_event
                            full_diagnostics = dict(
                                phase_controller.last_promotion_diagnostics
                            )
                        else:
                            full_event = "audit_only"
                            full_diagnostics = {}
                        full_snapshot.update(
                            {
                                "quality_update_id": quality_update_id,
                                "promotion_event": full_event,
                                **full_diagnostics,
                            }
                        )
                        pareto_validation_rows.append(full_snapshot)
                        pareto_candidate_rows.extend(
                            {
                                **row,
                                "update_id": update_id,
                                "completed_episodes": completed_episodes,
                            }
                            for row in full_rows
                        )
                        validation_row.update(
                            {
                                "pareto_full_mean_hypervolume": (
                                    full_snapshot["mean_hypervolume"]
                                ),
                                "pareto_full_mean_unique_action_trace_count": (
                                    full_snapshot[
                                        "mean_unique_action_trace_count"
                                    ]
                                ),
                                "pareto_full_mean_unique_objective_count": (
                                    full_snapshot[
                                        "mean_unique_objective_count"
                                    ]
                                ),
                                "pareto_full_mean_nondominated_count": (
                                    full_snapshot[
                                        "mean_nondominated_count"
                                    ]
                                ),
                                "pareto_full_coverage_pass": full_snapshot[
                                    "coverage_pass"
                                ],
                                "pareto_full_controllability_pass": (
                                    full_snapshot["controllability_pass"]
                                ),
                            }
                        )
                        if pareto_safety_guard is not None:
                            # A full-grid audit supersedes the anchor outcome
                            # at the same update and therefore counts once.
                            guard_snapshot = full_snapshot
                    if (
                        pareto_safety_guard is not None
                        and guard_snapshot is not None
                    ):
                        if tiered_protocol and not bool(
                            guard_snapshot.get("evaluation_integrity_pass", False)
                        ):
                            latest_rejected_candidate = _save_rejected_candidate(
                                config,
                                run_directory=run_directory,
                                agent=agent,
                                update_id=update_id,
                                stage=preference_stage,
                                failure_source="evaluation_integrity_failed",
                                failure_cell=guard_snapshot,
                                preference_stage_controller=(
                                    preference_stage_controller
                                ),
                            )
                            _write_tiered_hard_failure_summary(
                                run_directory,
                                config,
                                reason="evaluation_integrity_failed",
                                update_id=update_id,
                                rejected_candidate=latest_rejected_candidate,
                            )
                            raise RuntimeError(
                                "hard training gate failed: evaluation_integrity_failed"
                            )
                        guard_event = pareto_safety_guard.observe(guard_snapshot)
                        guard_scope = str(guard_snapshot["scope"])
                        guard_safe = bool(
                            guard_snapshot["physical_safety_pass"]
                            if tiered_protocol
                            else guard_snapshot["coverage_pass"]
                            and guard_snapshot["all_safe"]
                        )
                        if tiered_protocol:
                            (
                                best_safe_candidate_rank,
                                promoted_best_safe_candidate,
                            ) = _save_tiered_safe_candidate(
                                config,
                                agent=agent,
                                snapshot=guard_snapshot,
                                completed_episodes=completed_episodes,
                                last_safe_checkpoint=last_safe_checkpoint,
                                best_safe_candidate_checkpoint=(
                                    best_safe_candidate_checkpoint
                                ),
                                best_rank=best_safe_candidate_rank,
                                preference_stage_controller=(
                                    preference_stage_controller
                                ),
                            )
                            guard_snapshot[
                                "tiered_best_safe_candidate_promoted"
                            ] = promoted_best_safe_candidate
                        if guard_safe:
                            safe_target = (
                                full_grid_safe_checkpoint
                                if guard_scope == "full_grid_22"
                                else anchor_safe_checkpoint
                            )
                            agent.save(
                                safe_target,
                                metadata={
                                    **_checkpoint_protocol_metadata(config),
                                    "checkpoint_role": (
                                        "full_grid_safe"
                                        if guard_scope == "full_grid_22"
                                        else "anchor_safe"
                                    ),
                                    "safe_episode": completed_episodes,
                                    "pareto_snapshot": guard_snapshot,
                                    "preference_stage_controller": (
                                        preference_stage_controller.as_dict()
                                        if preference_stage_controller is not None
                                        else None
                                    ),
                                },
                            )
                        guard_snapshot.update(
                            {
                                "safety_guard_event": guard_event,
                                "safety_guard_consecutive_failures": (
                                    pareto_safety_guard.consecutive_failures
                                ),
                                "safety_guard_warning_count": (
                                    pareto_safety_guard.warning_count
                                ),
                                "safety_guard_rollback_count": (
                                    pareto_safety_guard.rollback_count
                                ),
                            }
                        )
                        validation_row.update(
                            {
                                "pareto_safety_guard_event": guard_event,
                                "pareto_safety_guard_scope": guard_scope,
                                "pareto_safety_guard_consecutive_failures": (
                                    pareto_safety_guard.consecutive_failures
                                ),
                                "pareto_safety_guard_warning_count": (
                                    pareto_safety_guard.warning_count
                                ),
                                "pareto_safety_guard_rollback_count": (
                                    pareto_safety_guard.rollback_count
                                ),
                            }
                        )
                        rollback_anchor_available = any(
                            path.exists()
                            for path in (
                                full_grid_safe_checkpoint,
                                anchor_safe_checkpoint,
                                phase1_checkpoint,
                            )
                        )
                        pareto_guard_rollback_requested = bool(
                            guard_event == "rollback"
                            and rollback_anchor_available
                        )
                        if pareto_guard_rollback_requested:
                            rollback_guard_snapshot = dict(guard_snapshot)
                        if (
                            guard_event == "rollback"
                            and not rollback_anchor_available
                            and _development_acceptance_enabled(config)
                        ):
                            guard_snapshot[
                                "safety_guard_rollback_skipped_reason"
                            ] = "no_safe_or_phase_checkpoint"
                            validation_row[
                                "pareto_safety_guard_rollback_skipped_reason"
                            ] = "no_safe_or_phase_checkpoint"
                if (
                    phase_controller.quality_checkpoint_promotion
                    == "balanced_guarded_v7"
                    and validation_event
                    in {"transition", "sampled_guard_pending"}
                ):
                    sampled_validation = _evaluate_sampled_validation(
                        config,
                        dataset_name=validation_split,
                        ppo_agent=agent,
                        instance_limit=validation_limit,
                        sampling_seeds=(
                            _official_evaluation_sampling_seeds(config)
                        ),
                        runner=runner,
                        use_parallel=validation_parallel_envs > 1,
                    )
                    stability_controller.sampled_validation_runs += 1
                    last_sampled_validation = sampled_validation
                    last_sampled_validation_episode = completed_episodes
                    _attach_sampled_validation(
                        validation_row,
                        sampled_validation,
                        completed_episodes=completed_episodes,
                    )
                    validation_event = phase_controller.observe_sampled_guard(
                        sampled_validation,
                        completed_episodes=completed_episodes,
                        transition_anchor=validation_event == "transition",
                    )
                daily_validation_event = validation_event
                audit_event = None
                validation_row["candidate_phase"] = reward_phase
                validation_row["validation_event"] = daily_validation_event
                validation_row.update(
                    phase_controller.last_promotion_diagnostics
                )
                validation_row["phase_after_validation"] = (
                    phase_controller.phase
                )
                validation_row["consecutive_completion_successes"] = (
                    phase_controller.consecutive_successes
                )
                validation_row.update(stability)
                validation_row["feasibility_rollback_applied"] = bool(
                    stability["rollback"]
                    and not (
                        pareto_safety_guard is not None
                        and reward_phase == "quality"
                    )
                )
                if (
                    phase_controller.quality_checkpoint_promotion
                    == SINGLE_OBJECTIVE_PROMOTION_MODE
                    and daily_validation_event == "audit_required"
                ):
                    window_median = (
                        phase_controller.last_promotion_diagnostics.get(
                            "window_objective_statistic"
                        )
                    )
                    if window_median is None:
                        raise RuntimeError(
                            "single-objective audit is missing window median"
                        )
                    agent.save(
                        candidate_checkpoint,
                        metadata={
                            **_checkpoint_protocol_metadata(config),
                            **_single_objective_checkpoint_metadata(
                                phase_controller,
                                checkpoint_role="audit_pending_candidate",
                            ),
                            "accepted_episode": None,
                            "daily_validation": validation_row,
                        },
                    )
                    audit_instance_rows, audit = (
                        _evaluate_single_objective_audit(
                            config,
                            dataset_name=validation_split,
                            ppo_agent=agent,
                            phase_controller=phase_controller,
                            runner=runner,
                            use_parallel=validation_parallel_envs > 1,
                        )
                    )
                    audit_event = phase_controller.observe_single_objective_audit(
                        audit,
                        completed_episodes=completed_episodes,
                        window_median=float(window_median),
                    )
                    audit_log_row = _single_objective_audit_log_row(
                        config,
                        episode=completed_episodes,
                        phase_controller=phase_controller,
                    )
                    single_objective_audit_rows.append(audit_log_row)
                    single_objective_audit_failure_rows.extend(
                        _single_objective_failure_rows(
                            audit_instance_rows,
                            episode=completed_episodes,
                            audit_event=audit_event,
                        )
                    )
                    validation_row.update(
                        {
                            "audit_event": audit_event,
                            "audit_failed_instance_count": audit_log_row.get(
                                "audit_failed_instance_count"
                            ),
                            "audit_completion_rate": audit_log_row.get(
                                "audit_completion_rate"
                            ),
                            "accepted_checkpoint_episode": (
                                phase_controller.accepted_quality_episode
                            ),
                        }
                    )
                    if audit_event == "accepted":
                        agent.save(
                            candidate_checkpoint,
                            metadata={
                                **_checkpoint_protocol_metadata(config),
                                **_single_objective_checkpoint_metadata(
                                    phase_controller,
                                    checkpoint_role=(
                                        "accepted_98_experiment_candidate"
                                    ),
                                ),
                                "formal_eligible": False,
                                "accepted_episode": completed_episodes,
                                "daily_validation": validation_row,
                                "audit": audit_log_row,
                            },
                        )
                        candidate_checkpoint.replace(accepted_checkpoint)
                        best_score = score
                        best_validation = validation_row
                        update_row["candidate_status"] = "accepted_98"
                        for recent_row in rows[-len(rollout.episodes) :]:
                            recent_row["candidate_status"] = "accepted_98"
                    elif candidate_checkpoint.exists():
                        candidate_checkpoint.unlink()
                validation_event = audit_event or daily_validation_event
                last_validation_event = validation_event
                validation_rows.append(validation_row)
                canonical_safe = (
                    _rows_are_safe(
                        validation_instance_rows,
                        float(
                            pareto_settings["fatigue_absolute_tolerance"]
                        ),
                    )
                    if pareto_mode and pareto_settings is not None
                    else (
                        _single_objective_hard_gate(validation)["all"]
                        if phase_controller.quality_checkpoint_promotion
                        == SINGLE_OBJECTIVE_PROMOTION_MODE
                        else validation["completion_rate"] >= 1.0 - 1e-12
                    )
                )
                if canonical_safe:
                    agent.save(
                        safe_checkpoint,
                        metadata={
                            "checkpoint_role": "latest_safe",
                            "safe_episode": completed_episodes,
                            "validation": validation_row,
                            "preference_stage_controller": (
                                preference_stage_controller.as_dict()
                                if preference_stage_controller is not None
                                else None
                            ),
                        },
                    )
                if (
                    bool(stability["improved"])
                    and reward_phase == "feasibility"
                ):
                    best_feasibility_validation = validation_row
                    best_feasibility_instance_rows = [
                        dict(value) for value in validation_instance_rows
                    ]
                    agent.save(
                        best_feasibility_checkpoint,
                        metadata={
                            "feature_dimensions": (
                                bootstrap_observation.feature_dimensions
                            ),
                            "edge_feature_dimensions": (
                                bootstrap_observation.edge_feature_dimensions
                            ),
                            "seed": config["seed"],
                            "smoke": smoke,
                            "online_instances": True,
                            "generator_version": config["generator"][
                                "version"
                            ],
                            "parallel_envs": parallel_envs,
                            "update_id": update_id,
                            "best_feasibility_episode": (
                                completed_episodes
                            ),
                            "learning_rate": (
                                stability_controller.current_learning_rate
                            ),
                            "validation": validation_row,
                        },
                    )
                    update_row["candidate_status"] = "feasibility_best"
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = "feasibility_best"
                is_new_best = False
                if transitioned_now:
                    gate = getattr(agent.network, "production_gate_version", "none")
                    if gate in POST_FEASIBILITY_RESIDUAL_GATE_VERSIONS:
                        agent.network.set_production_state_gate_frozen(True)
                        agent.network.set_production_flow_commit_residual_enabled(True)
                    transition_metadata = {
                        **_single_objective_checkpoint_metadata(
                            phase_controller
                        ),
                        "feature_dimensions": (
                            bootstrap_observation.feature_dimensions
                        ),
                        "edge_feature_dimensions": (
                            bootstrap_observation.edge_feature_dimensions
                        ),
                        "seed": config["seed"],
                        "parallel_envs": parallel_envs,
                        "phase_transition_episode": completed_episodes,
                        "validation": validation_row,
                    }
                    agent.save(
                        phase1_checkpoint,
                        metadata=transition_metadata,
                    )
                    update_row["candidate_status"] = "phase_transition"
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = "phase_transition"
                checkpoint_eligible_event = (
                    _checkpoint_eligible_validation_event(
                        validation_event,
                        phase_controller.quality_checkpoint_promotion,
                    )
                )
                if (
                    checkpoint_eligible_event
                    and not tiered_protocol
                    and phase_controller.quality_checkpoint_promotion
                    != SINGLE_OBJECTIVE_PROMOTION_MODE
                ):
                    agent.save(
                        accepted_checkpoint,
                        metadata={
                            **_checkpoint_protocol_metadata(config),
                            **_single_objective_checkpoint_metadata(
                                phase_controller
                            ),
                            "checkpoint_role": (
                                "single_seed_development_pareto"
                                if _development_acceptance_enabled(config)
                                else "shadow_best"
                            ),
                            "development_scope": (
                                "single_seed_development"
                                if _development_acceptance_enabled(config)
                                else None
                            ),
                            "formal_eligible": False
                            if _development_acceptance_enabled(config)
                            else True,
                            "warm_start": warm_start_report,
                            "heldout_comparison": (
                                latest_e2_7_heldout_report
                            ),
                            "seed": config["seed"],
                            "parallel_envs": parallel_envs,
                            "accepted_episode": completed_episodes,
                            "quality_score": normalized_quality_score,
                            "single_objective_name": (
                                phase_controller.single_objective_name
                            ),
                            "single_objective_statistic": (
                                phase_controller.single_objective_window_statistic
                                if phase_controller.quality_checkpoint_promotion
                                == SINGLE_OBJECTIVE_PROMOTION_MODE
                                else None
                            ),
                            "single_objective_value": (
                                phase_controller.accepted_single_objective_value
                            ),
                            "validation": validation_row,
                        },
                    )
                    if not _development_acceptance_enabled(config):
                        shutil.copyfile(accepted_checkpoint, best_checkpoint)
                    best_score = score
                    best_validation = validation_row
                    is_new_best = True
                    if validation_event != "transition":
                        update_row["candidate_status"] = "promoted"
                        for row in rows[-len(rollout.episodes) :]:
                            row["candidate_status"] = "promoted"
                elif validation_event in {
                    "not_promoted",
                    "rejected",
                    "audit_rejected",
                    "audit_passed_not_accepted",
                }:
                    update_row["candidate_status"] = "not_promoted"
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = "not_promoted"
                if (
                    pareto_guard_rollback_requested
                    and pareto_safety_guard is not None
                ):
                    latest_rejected_candidate = _save_rejected_candidate(
                        config,
                        run_directory=run_directory,
                        agent=agent,
                        update_id=update_id,
                        stage=preference_stage,
                        failure_source="pareto_safety_guard",
                        failure_cell=rollback_guard_snapshot,
                        preference_stage_controller=(
                            preference_stage_controller
                        ),
                    )
                    rollback_source: tuple[str, Path] | None = next(
                        (
                            (name, path)
                            for name, path in (
                                ("last_safe", last_safe_checkpoint),
                                (
                                    "full_grid_safe",
                                    full_grid_safe_checkpoint,
                                ),
                                ("anchor_safe", anchor_safe_checkpoint),
                                ("phase1", phase1_checkpoint),
                            )
                            if path.exists()
                        ),
                        None,
                    )
                    if rollback_source is None:
                        raise RuntimeError(
                            "Pareto safety rollback requested before a "
                            "phase1 checkpoint was established"
                        )
                    failure_learning_rate = agent.learning_rate
                    _restore_e2_7_rollback_checkpoint(
                        agent,
                        rollback_source[1],
                        preference_stage_controller,
                        restore_stage_controller=not tiered_protocol,
                    )
                    if tiered_protocol and preference_stage_controller is not None:
                        # The optimizer/model rewind must not consume or erase
                        # monitored-v4 curriculum budget already spent.
                        preference_stage_controller.apply(agent)
                    if (
                        getattr(agent.network, "production_gate_version", "none")
                        in POST_FEASIBILITY_RESIDUAL_GATE_VERSIONS
                    ):
                        agent.network.set_production_state_gate_frozen(True)
                        agent.network.set_production_flow_commit_residual_enabled(True)
                    restored_learning_rate = agent.learning_rate
                    guarded_learning_rate = max(
                        pareto_safety_guard.minimum_learning_rate,
                        pareto_safety_guard.learning_rate_decay_factor
                        * min(failure_learning_rate, restored_learning_rate),
                    )
                    agent.set_learning_rate(guarded_learning_rate)
                    stability_controller.current_learning_rate = (
                        guarded_learning_rate
                    )
                    stability_controller.reset_plateau()
                    pareto_safety_guard.record_rollback(rollback_source[0])
                    validation_row.update(
                        {
                            "pareto_safety_guard_rollback_applied": True,
                            "pareto_safety_guard_rollback_source": (
                                rollback_source[0]
                            ),
                            "pareto_safety_guard_rollback_learning_rate": (
                                guarded_learning_rate
                            ),
                            "pareto_safety_guard_consecutive_failures": (
                                pareto_safety_guard.consecutive_failures
                            ),
                        }
                    )
                    update_row["candidate_status"] = "pareto_guard_rolled_back"
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = "pareto_guard_rolled_back"
                if (
                    bool(stability["rollback"])
                    and not (
                        pareto_safety_guard is not None
                        and reward_phase == "quality"
                    )
                ):
                    if not safe_checkpoint.exists():
                        raise RuntimeError(
                            "catastrophic rollback requested before a safe "
                            "checkpoint was established"
                        )
                    _restore_e2_7_rollback_checkpoint(
                        agent,
                        safe_checkpoint,
                        preference_stage_controller,
                    )
                    if phase_controller.quality_checkpoint_promotion == SINGLE_OBJECTIVE_PROMOTION_MODE:
                        phase_controller.reset_single_objective_window()
                    update_row["candidate_status"] = (
                        "catastrophic_rolled_back"
                    )
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = (
                            "catastrophic_rolled_back"
                        )
                agent.set_learning_rate(
                    stability_controller.current_learning_rate
                )
                if transitioned_now:
                    stability_controller.reset_plateau()
                if not phase_controller.enabled and (
                    best_score is None or score < best_score
                ):
                    best_score = score
                    best_validation = validation_row
                    agent.save(
                        accepted_checkpoint,
                        metadata={
                            **_checkpoint_protocol_metadata(config),
                            "feature_dimensions": (
                                bootstrap_observation.feature_dimensions
                            ),
                            "edge_feature_dimensions": (
                                bootstrap_observation.edge_feature_dimensions
                            ),
                            "seed": config["seed"],
                            "smoke": smoke,
                            "online_instances": True,
                            "generator_version": config["generator"][
                                "version"
                            ],
                            "parallel_envs": parallel_envs,
                            "update_id": update_id,
                            "best_episode": completed_episodes,
                            "validation": validation_row,
                        },
                    )
                    shutil.copyfile(accepted_checkpoint, best_checkpoint)
                    is_new_best = True
                dashboard.log_validation(
                    validation_row,
                    best_validation=best_validation,
                    phase_state=phase_controller.as_dict(),
                )
                if validation_event in {
                    "transition",
                    "promoted",
                    "audit_required",
                    "audit_rejected",
                    "audit_passed_not_accepted",
                    "not_promoted",
                    "accepted",
                    "rejected",
                }:
                    dashboard.log_event(
                        f"episode {completed_episodes}: "
                        f"validation event={validation_event}"
                    )
                if is_new_best:
                    dashboard.log_event(
                        f"episode {completed_episodes}: "
                        "new best checkpoint"
                    )
                if dashboard.should_capture_diagnostic(
                    validation_event=validation_event,
                    is_new_best=is_new_best,
                ):
                    try:
                        trace = evaluate_representative_diagnostic(
                            config,
                            dataset_name=validation_split,
                            ppo_agent=agent,
                            instance_index=int(
                                visdom_settings[
                                    "representative_instance_index"
                                ]
                            ),
                        )
                        dashboard.log_diagnostic(
                            trace,
                            completed_episodes=completed_episodes,
                        )
                    except Exception as error:
                        dashboard.log_event(
                            "representative diagnostic failed at episode "
                            f"{completed_episodes}: {error}"
                        )
                        if bool(visdom_settings["fail_fast"]):
                            raise
                print(
                    json.dumps(
                        {"validation": validation_row},
                        ensure_ascii=False,
                    )
                )
            batch_rows = rows[-len(rollout.episodes) :]
            dashboard.log_update(
                update_row,
                batch_rows,
                phase_controller.as_dict(),
            )
            for row in batch_rows:
                print(json.dumps(row, ensure_ascii=False))
            if preference_stage_controller is not None:
                agent.save(
                    last_checkpoint,
                    metadata={
                        **_checkpoint_protocol_metadata(config),
                        "checkpoint_role": "last_online",
                        "update_id": update_id,
                        "safe_dual_legal_state_pool": safe_state_pool_report,
                        "preference_stage_controller": (
                            preference_stage_controller.as_dict()
                        ),
                    },
                )
                write_csv(run_directory / "train_log.csv", rows)
                write_csv(run_directory / "update_log.csv", update_rows)
                write_csv(run_directory / "validation_log.csv", validation_rows)
                if pareto_validation_rows:
                    write_csv(
                        run_directory / "pareto_validation_log.csv",
                        pareto_validation_rows,
                    )
                if pareto_candidate_rows:
                    write_csv(
                        run_directory / "pareto_validation_candidates.csv",
                        pareto_candidate_rows,
                    )
                if e2_3_failure_replay_rows:
                    write_csv(
                        run_directory / "e2_3_failure_replay.csv",
                        e2_3_failure_replay_rows,
                    )
                if e2_7_safety_replay_rows:
                    write_csv(
                        run_directory / "e2_7_safety_replay.csv",
                        e2_7_safety_replay_rows,
                    )
                write_json(
                    run_directory / "summary.json",
                    {
                        "training_status": "running",
                        "updates": update_id,
                        "last_update": update_row,
                        "last_checkpoint": _run_relative_checkpoint(
                            last_checkpoint, run_directory
                        ),
                        "preference_stage_controller": (
                            preference_stage_controller.as_dict()
                        ),
                        "latest_rejected_candidate": latest_rejected_candidate,
                    },
                )
        if (
            config["training"].get("ablation_variant") == "Q13"
            and phase_controller.phase_transition_episode is not None
        ):
            (
                final_accepted_sampled_validation,
                final_accepted_sampled_validation_source,
                reran_final_sampled_validation,
            ) = _resolve_final_accepted_sampled_validation(
                config,
                dataset_name=validation_split,
                ppo_agent=agent,
                instance_limit=validation_limit,
                sampling_seeds=stability_controller.sampled_seeds(
                    int(config["seed"])
                ),
                final_episode=episodes,
                sampled_episode=last_sampled_validation_episode,
                validation_event=last_validation_event,
                sampled_validation=last_sampled_validation,
                runner=runner,
                use_parallel=validation_parallel_envs > 1,
            )
            if reran_final_sampled_validation:
                stability_controller.sampled_validation_runs += 1
        if tiered_protocol:
            candidate_reports: dict[str, dict[str, Any]] = {}
            final_artifact_rows: list[dict[str, Any]] = []
            evaluated_hashes: dict[str, dict[str, Any]] = {}
            role_paths = (
                ("best_safe", best_safe_candidate_checkpoint),
                ("last_safe", last_safe_checkpoint),
            )
            for role, candidate_path in role_paths:
                if not candidate_path.exists():
                    report = {
                        "version": "tiered_final_candidate_acceptance_v1",
                        "role": role,
                        "checkpoint": None,
                        "acceptance_status": "failed",
                        "reason": "safe_candidate_not_available",
                    }
                else:
                    checkpoint_hash = _checkpoint_sha256(candidate_path)
                    if checkpoint_hash in evaluated_hashes:
                        source_report = evaluated_hashes[checkpoint_hash]
                        report = {
                            **source_report,
                            "role": role,
                            "deduplicated_from_role": source_report["role"],
                        }
                    else:
                        report, candidate_rows = _evaluate_tiered_final_candidate(
                            config,
                            role=role,
                            checkpoint=candidate_path,
                            agent=agent,
                            runner=runner,
                            phase_controller=phase_controller,
                            validation_split=validation_split,
                            validation_limit=validation_limit,
                            validation_parallel_envs=validation_parallel_envs,
                            update_id=update_id,
                            completed_episodes=episodes,
                            fatigue_tolerance=float(
                                pareto_settings["fatigue_absolute_tolerance"]
                            ),
                        )
                        final_artifact_rows.extend(candidate_rows)
                        evaluated_hashes[checkpoint_hash] = report
                candidate_reports[role] = report
                write_json(
                    run_directory / f"final_acceptance_{role}.json", report
                )
            primary_role = _select_tiered_primary_role(candidate_reports)
            final_acceptance = {
                "version": "tiered_final_acceptance_v1",
                "rule": "any_candidate_passes",
                "acceptance_status": (
                    "passed" if primary_role is not None else "failed"
                ),
                "primary_role": primary_role,
                "candidate_reports": candidate_reports,
                "monitoring_warnings": list(tiered_monitoring_warnings),
                "rollback_count": (
                    pareto_safety_guard.rollback_count
                    if pareto_safety_guard is not None
                    else 0
                ),
            }
            write_json(run_directory / "final_acceptance.json", final_acceptance)
            if final_artifact_rows:
                write_csv(
                    run_directory / "final_acceptance_candidates.csv",
                    final_artifact_rows,
                )
            if primary_role is not None:
                primary_path = dict(role_paths)[primary_role]
                shutil.copyfile(primary_path, accepted_checkpoint)
                shutil.copyfile(primary_path, best_checkpoint)
                latest_e2_7_heldout_report = candidate_reports[primary_role].get(
                    "heldout"
                )
    single_objective_mode = (
        phase_controller.quality_checkpoint_promotion
        == SINGLE_OBJECTIVE_PROMOTION_MODE
    )
    formal_eligible = bool(tiered_protocol) or ((not single_objective_mode) and (
        (
            not phase_controller.enabled
            or (
                phase_controller.phase_transition_episode is not None
                and (
                    phase_controller.quality_checkpoint_promotion
                    not in PARETO_PROMOTION_MODES
                    or phase_controller.accepted_pareto_hv is not None
                )
            )
        )
        and not _development_acceptance_enabled(config)
    ))
    experiment_candidate_accepted = bool(
        single_objective_mode
        and accepted_checkpoint.exists()
        and phase_controller.accepted_single_objective_value is not None
    )
    development_accepted = (
        bool(
            final_acceptance is not None
            and final_acceptance["acceptance_status"] == "passed"
        )
        if tiered_protocol
        else bool(
            _development_acceptance_enabled(config)
            and accepted_checkpoint.exists()
            and phase_controller.accepted_pareto_hv is not None
        )
    )
    if (
        not tiered_protocol
        and not formal_eligible
        and not single_objective_mode
        and not _development_acceptance_enabled(config)
        and phase_controller.formal_training_status
        not in {
            "feasibility_not_reached",
            "pareto_baseline_not_reached",
            "single_objective_98_candidate_not_reached",
        }
    ):
        raise RuntimeError("invalid hierarchical training state")
    if (
        not tiered_protocol
        and formal_eligible
        and (best_validation is None or best_score is None)
    ):
        raise RuntimeError("training completed without validation")
    q13_final_sampled_metadata = (
        {
            "final_accepted_sampled_validation": (
                final_accepted_sampled_validation
            ),
            "final_accepted_sampled_validation_source": (
                final_accepted_sampled_validation_source
            ),
            "final_accepted_checkpoint_episode": (
                phase_controller.accepted_quality_episode
            ),
        }
        if config["training"].get("ablation_variant") == "Q13"
        else {}
    )
    final_metadata = {
            **_checkpoint_protocol_metadata(config),
            "feature_dimensions": (
                bootstrap_observation.feature_dimensions
            ),
            "edge_feature_dimensions": (
                bootstrap_observation.edge_feature_dimensions
            ),
            "seed": config["seed"],
            "smoke": smoke,
            "online_instances": True,
            "generator_version": config["generator"]["version"],
            "parallel_envs": parallel_envs,
            "safe_dual_legal_state_pool": safe_state_pool_report,
            "updates": update_id,
            "last_update": update_rows[-1] if update_rows else None,
            "transitions": total_transitions,
            "environment_steps": total_environment_steps,
            "forced_actions": total_forced_actions,
            "forced_action_ratio": (
                total_forced_actions / total_environment_steps
                if total_environment_steps > 0
                else 0.0
            ),
            "worker_step_command_count": total_worker_step_commands,
            "worker_local_physical_forced_action_count": (
                total_worker_local_physical_forced_actions
            ),
            "worker_local_physical_forced_share": (
                total_worker_local_physical_forced_actions
                / total_forced_actions
                if total_forced_actions > 0
                else 0.0
            ),
            "estimated_worker_step_round_trips_avoided": (
                total_worker_local_physical_forced_actions
            ),
            "forced_action_diagnostics": _forced_action_summary(rows),
            "mean_policy_steps_per_episode": (
                total_transitions / episodes if episodes > 0 else 0.0
            ),
            "best_checkpoint": (
                str(best_checkpoint) if best_checkpoint.exists() else None
            ),
            "best_validation": best_validation,
            "best_feasibility_checkpoint": (
                str(best_feasibility_checkpoint)
                if best_feasibility_checkpoint.exists()
                else None
            ),
            "best_feasibility_validation": (
                best_feasibility_validation
            ),
            "validation_stability": stability_controller.as_dict(),
            "preference_stage_controller": (
                preference_stage_controller.as_dict()
                if preference_stage_controller is not None
                else None
            ),
            "last_sampled_validation": last_sampled_validation,
            "formal_training_status": (
                phase_controller.formal_training_status
            ),
            "training_phase": phase_controller.as_dict(),
            "formal_eligible": formal_eligible,
            "experiment_candidate_accepted": experiment_candidate_accepted,
            "development_accepted": development_accepted,
            "development_scope": (
                "single_seed_development"
                if _development_acceptance_enabled(config)
                else None
            ),
            "warm_start": warm_start_report,
            "heldout_comparison": latest_e2_7_heldout_report,
            **q13_final_sampled_metadata,
        }
    checkpoint: Path | None
    last_candidate_checkpoint: Path | None
    agent.save(
        last_checkpoint,
        metadata={**final_metadata, "checkpoint_role": "last_online"},
    )
    if single_objective_mode:
        checkpoint = None
        last_candidate_checkpoint = None
    elif formal_eligible and accepted_checkpoint.exists():
        checkpoint = run_directory / "checkpoint.pt"
        last_candidate_checkpoint = None
        shutil.copyfile(accepted_checkpoint, checkpoint)
        shutil.copyfile(accepted_checkpoint, best_checkpoint)
    else:
        if formal_eligible and not tiered_protocol:
            raise RuntimeError(
                "formal training completed without a shadow-best checkpoint"
            )
        checkpoint = None
        last_candidate_checkpoint = (
            run_directory / "last_candidate_checkpoint.pt"
        )
        shutil.copyfile(last_checkpoint, last_candidate_checkpoint)
    final_checkpoint_evaluation = None
    checkpoint_sha256 = None
    if single_objective_mode and accepted_checkpoint.exists():
        final_checkpoint_evaluation = _reevaluate_checkpoint_from_disk(
            config,
            checkpoint=accepted_checkpoint,
            bootstrap_observation=bootstrap_observation,
            dataset_name=validation_split,
            instance_limit=phase_controller.single_objective_audit_instance_limit,
            sampling_seeds=[],
            greedy_only=True,
        )
        try:
            _assert_single_objective_checkpoint_evaluation(
                phase_controller,
                final_checkpoint_evaluation,
            )
        except Exception as error:
            invalidated_checkpoint = (
                run_directory / "invalidated_accepted_checkpoint.pt"
            )
            accepted_checkpoint.replace(invalidated_checkpoint)
            write_csv(run_directory / "train_log.csv", rows)
            write_csv(run_directory / "update_log.csv", update_rows)
            write_csv(run_directory / "validation_log.csv", validation_rows)
            write_csv(
                run_directory / "single_objective_audit_log.csv",
                single_objective_audit_rows,
            )
            write_csv(
                run_directory / "single_objective_audit_failures.csv",
                single_objective_audit_failure_rows,
            )
            write_json(
                run_directory / "failure.json",
                {
                    "status": "accepted_checkpoint_invalidated",
                    "error": str(error),
                    "invalidated_checkpoint": str(invalidated_checkpoint),
                    "formal_eligible": False,
                },
            )
            raise RuntimeError(
                "single-objective accepted checkpoint failed final audit"
            ) from error
    elif checkpoint is not None:
        checkpoint_sha256 = _checkpoint_sha256(checkpoint)
        accepted_sha256 = _checkpoint_sha256(accepted_checkpoint)
        best_sha256 = _checkpoint_sha256(best_checkpoint)
        if len({checkpoint_sha256, accepted_sha256, best_sha256}) != 1:
            raise RuntimeError(
                "official, accepted, and best checkpoint hashes diverged"
            )
        final_checkpoint_evaluation = _reevaluate_checkpoint_from_disk(
            config,
            checkpoint=checkpoint,
            bootstrap_observation=bootstrap_observation,
            dataset_name=validation_split,
            instance_limit=validation_limit,
            sampling_seeds=_official_evaluation_sampling_seeds(config),
        )
    summary_checkpoint = (
        accepted_checkpoint
        if single_objective_mode and accepted_checkpoint.exists()
        else checkpoint or last_checkpoint
    )
    summary_provenance = build_provenance(
        config,
        dataset_manifest_path=_validation_manifest_path(config),
        checkpoint_path=summary_checkpoint,
        checkpoint_metadata=_checkpoint_protocol_metadata(config),
    )
    write_csv(run_directory / "train_log.csv", rows)
    write_csv(run_directory / "update_log.csv", update_rows)
    write_csv(run_directory / "validation_log.csv", validation_rows)
    if single_objective_mode:
        write_csv(
            run_directory / "single_objective_audit_log.csv",
            single_objective_audit_rows,
        )
        write_csv(
            run_directory / "single_objective_audit_failures.csv",
            single_objective_audit_failure_rows,
        )
    if pareto_validation_rows:
        write_csv(
            run_directory / "pareto_validation_log.csv",
            pareto_validation_rows,
        )
    if pareto_candidate_rows:
        write_csv(
            run_directory / "pareto_validation_candidates.csv",
            pareto_candidate_rows,
        )
    if e2_3_failure_replay_rows:
        write_csv(
            run_directory / "e2_3_failure_replay.csv",
            e2_3_failure_replay_rows,
        )
    if e2_7_safety_replay_rows:
        write_csv(
            run_directory / "e2_7_safety_replay.csv",
            e2_7_safety_replay_rows,
        )
    if e2_7_heldout_candidate_rows:
        write_csv(
            run_directory / "e2_7_heldout_candidates.csv",
            e2_7_heldout_candidate_rows,
        )
    if _development_acceptance_enabled(config):
        shield_reasons: Counter[str] = Counter()
        for row in rows:
            raw_reasons = row.get("production_defer_shield_reason_counts", "{}")
            try:
                reasons = (
                    json.loads(raw_reasons)
                    if isinstance(raw_reasons, str)
                    else dict(raw_reasons or {})
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                reasons = {"invalid_serialized_reason_counts": 1}
            shield_reasons.update(
                {str(key): int(value) for key, value in reasons.items()}
            )
        write_json(
            run_directory / "e2_7_defer_shield_report.json",
            {
                "version": "deadline_progress_shield_report_v1",
                "trajectory_count": len(rows),
                "candidate_count": sum(
                    int(row.get("production_defer_shield_candidate_count", 0) or 0)
                    for row in rows
                ),
                "masked_count": sum(
                    int(row.get("production_defer_shield_masked_count", 0) or 0)
                    for row in rows
                ),
                "maximum_risk": max(
                    (
                        float(row.get("production_defer_shield_max_risk", 0.0) or 0.0)
                        for row in rows
                    ),
                    default=0.0,
                ),
                "maximum_wait_ticks": max(
                    (
                        int(row.get("production_defer_shield_max_wait_ticks", 0) or 0)
                        for row in rows
                    ),
                    default=0,
                ),
                "maximum_remaining_work_lower_bound_ticks": max(
                    (
                        int(
                            row.get(
                                "production_defer_shield_max_work_lower_bound_ticks",
                                0,
                            )
                            or 0
                        )
                        for row in rows
                    ),
                    default=0,
                ),
                "minimum_deadline_slack_ticks": min(
                    (
                        int(row["production_defer_shield_min_deadline_slack_ticks"])
                        for row in rows
                        if row.get(
                            "production_defer_shield_min_deadline_slack_ticks"
                        )
                        is not None
                    ),
                    default=None,
                ),
                "reason_counts": dict(shield_reasons),
            },
        )
    total_training_time = total_sampling_time + total_update_time
    write_json(
        run_directory / "summary.json",
        {
            "result_schema_version": result_schema_version(config),
            "training_status": (
                "completed" if tiered_protocol else phase_controller.formal_training_status
            ),
            "acceptance_status": (
                final_acceptance["acceptance_status"]
                if final_acceptance is not None
                else "not_configured"
            ),
            "final_acceptance": final_acceptance,
            "episodes": episodes,
            "trajectory_count": episodes,
            "base_instance_count": training_base_instance_count(
                config, episodes
            ),
            "safe_dual_legal_state_pool": safe_state_pool_report,
            "online_instances": True,
            "parallel_envs": parallel_envs,
            "validation_parallel_envs": validation_parallel_envs,
            "pareto_safety_guard": (
                pareto_safety_guard.as_dict()
                if pareto_safety_guard is not None
                else None
            ),
            "pareto_validation_events": len(pareto_validation_rows),
            "latest_pareto_validation": (
                pareto_validation_rows[-1]
                if pareto_validation_rows
                else None
            ),
            "updates": update_id,
            "last_update": update_rows[-1] if update_rows else None,
            "transitions": total_transitions,
            "environment_steps": total_environment_steps,
            "forced_actions": total_forced_actions,
            "forced_action_ratio": (
                total_forced_actions / total_environment_steps
                if total_environment_steps > 0
                else 0.0
            ),
            "worker_step_command_count": total_worker_step_commands,
            "worker_local_physical_forced_action_count": (
                total_worker_local_physical_forced_actions
            ),
            "worker_local_physical_forced_share": (
                total_worker_local_physical_forced_actions
                / total_forced_actions
                if total_forced_actions > 0
                else 0.0
            ),
            "estimated_worker_step_round_trips_avoided": (
                total_worker_local_physical_forced_actions
            ),
            "forced_action_diagnostics": _forced_action_summary(rows),
            "mean_policy_steps_per_episode": (
                total_transitions / episodes if episodes > 0 else 0.0
            ),
            "unique_instance_count": len(set(instance_ids)),
            "total_sampling_time_seconds": total_sampling_time,
            "total_policy_inference_time_seconds": (
                total_inference_time
            ),
            "total_ppo_update_time_seconds": total_update_time,
            "mean_transitions_per_second": (
                total_transitions / total_training_time
                if total_training_time > 0
                else 0.0
            ),
            "checkpoint": _run_relative_checkpoint(checkpoint, run_directory),
            "checkpoint_sha256": checkpoint_sha256,
            "provenance": summary_provenance,
            "final_checkpoint_evaluation": final_checkpoint_evaluation,
            "accepted_checkpoint": (
                _run_relative_checkpoint(accepted_checkpoint, run_directory)
                if accepted_checkpoint.exists()
                and (tiered_protocol or not _development_acceptance_enabled(config))
                else None
            ),
            "latest_e2_3_failure_replay": latest_e2_3_failure_replay,
            "latest_e2_7_safety_replay": latest_e2_7_safety_replay,
            "latest_rejected_candidate": latest_rejected_candidate,
            "preference_stage_controller": (
                preference_stage_controller.as_dict()
                if preference_stage_controller is not None
                else None
            ),
            "latest_e2_7_heldout_comparison": latest_e2_7_heldout_report,
            "e2_3_failure_replay_event_count": (
                len(e2_3_failure_replay_rows) // 10
                if e2_3_failure_replay_rows
                else 0
            ),
            "development_accepted_pareto_checkpoint": (
                _run_relative_checkpoint(accepted_checkpoint, run_directory)
                if development_accepted
                else None
            ),
            "development_accepted": development_accepted,
            "development_failure_reason": (
                None
                if development_accepted
                else phase_controller.last_promotion_diagnostics.get(
                    "promotion_decision_reason",
                    phase_controller.formal_training_status,
                )
            ),
            "development_scope": (
                "single_seed_development"
                if _development_acceptance_enabled(config)
                else None
            ),
            "formal_eligible": formal_eligible,
            "experiment_candidate_accepted": experiment_candidate_accepted,
            "warm_start": warm_start_report,
            "last_checkpoint": _run_relative_checkpoint(last_checkpoint, run_directory),
            "last_safe_checkpoint": (
                _run_relative_checkpoint(last_safe_checkpoint, run_directory)
                if last_safe_checkpoint.exists()
                else None
            ),
            "best_safe_candidate_checkpoint": (
                _run_relative_checkpoint(
                    best_safe_candidate_checkpoint, run_directory
                )
                if best_safe_candidate_checkpoint.exists()
                else None
            ),
            "tiered_gate_policy": tiered_gate_policy,
            "tiered_monitoring_warnings": tiered_monitoring_warnings,
            "safe_checkpoint": (
                str(safe_checkpoint) if safe_checkpoint.exists() else None
            ),
            "anchor_safe_checkpoint": (
                str(anchor_safe_checkpoint)
                if anchor_safe_checkpoint.exists()
                else None
            ),
            "full_grid_safe_checkpoint": (
                str(full_grid_safe_checkpoint)
                if full_grid_safe_checkpoint.exists()
                else None
            ),
            "last_candidate_checkpoint": (
                str(last_candidate_checkpoint)
                if last_candidate_checkpoint is not None
                else None
            ),
            "best_checkpoint": (
                str(best_checkpoint) if best_checkpoint.exists() else None
            ),
            "best_validation": best_validation,
            "best_feasibility_checkpoint": (
                str(best_feasibility_checkpoint)
                if best_feasibility_checkpoint.exists()
                else None
            ),
            "best_feasibility_validation": (
                best_feasibility_validation
            ),
            "best_feasibility_episode": (
                best_feasibility_validation["episode"]
                if best_feasibility_validation is not None
                else None
            ),
            "feasibility_rollbacks": (
                stability_controller.feasibility_rollbacks
            ),
            "learning_rate_decays": (
                stability_controller.learning_rate_decays
            ),
            "phase1_checkpoint": (
                str(phase1_checkpoint)
                if phase1_checkpoint.exists()
                else None
            ),
            "formal_training_status": (
                phase_controller.formal_training_status
            ),
            "training_phase": phase_controller.as_dict(),
            "single_objective_audit": (
                {
                    "daily_validation_instance_limit": validation_limit,
                    "audit_instance_limit": (
                        phase_controller.single_objective_audit_instance_limit
                    ),
                    "audit_count": len(single_objective_audit_rows),
                    "audit_failure_row_count": len(
                        single_objective_audit_failure_rows
                    ),
                    "accepted_status": phase_controller.formal_training_status,
                    "project_formal_completion_target": 1.0,
                }
                if single_objective_mode
                else None
            ),
            "validation_stability": stability_controller.as_dict(),
            "validation_runs": len(validation_rows),
            "sampled_validation_runs": (
                stability_controller.sampled_validation_runs
            ),
            "last_sampled_validation": last_sampled_validation,
            **q13_final_sampled_metadata,
            "late_500_episode_diagnostics": (
                _late_training_diagnostics(rows)
            ),
            "ablation_gate": _ablation_gate_summary(
                config,
                rows,
                validation_rows,
                stability_controller,
                best_feasibility_instance_rows,
            ),
            "visdom": {
                "enabled": bool(dashboard.enabled),
                "connected": bool(dashboard.connected),
                "environment": dashboard.environment,
                "event_log": (
                    str(run_directory / "visdom_events.log")
                    if dashboard.enabled
                    else None
                ),
            },
            "last_episode": rows[-1],
            "last_update": update_rows[-1],
            "policy_head_diagnostics": agent.policy_head_diagnostics(),
        },
    )
    dashboard.log_event(
        "training completed with status="
        f"{phase_controller.formal_training_status}"
    )
    dashboard.close()
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the lightweight PPO policy")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--episodes",
        type=int,
        help="override training.episodes for this run",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--initial-checkpoint",
        help=(
            "initialize policy and optimizer from a compatible checkpoint"
        ),
    )
    checkpoint_group.add_argument(
        "--warm-start-checkpoint",
        help=(
            "load every E1 shared network tensor strictly, initialize only new "
            "adapter parameters, and create a fresh optimizer"
        ),
    )
    instance_group = parser.add_mutually_exclusive_group()
    instance_group.add_argument(
        "--online-instances",
        dest="online_instances",
        action="store_true",
    )
    instance_group.add_argument(
        "--fixed-instance",
        dest="online_instances",
        action="store_false",
    )
    parser.set_defaults(online_instances=None)
    parser.add_argument("--algorithm-seed", type=int)
    parser.add_argument("--parallel-envs", type=int)
    parser.add_argument(
        "--ablation",
        choices=(
            "E1", "E2", "E3", "R11", "S11", "L11", "Q11", "Q12",
            "Q13", "e1", "e2", "e3", "r11", "s11", "l11", "q11",
            "q12", "q13",
        ),
        help=(
            "run a 600-episode seed-11 screening configuration"
        ),
    )
    visdom_group = parser.add_mutually_exclusive_group()
    visdom_group.add_argument(
        "--visdom",
        dest="visdom_enabled",
        action="store_true",
    )
    visdom_group.add_argument(
        "--no-visdom",
        dest="visdom_enabled",
        action="store_false",
    )
    parser.set_defaults(visdom_enabled=None)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.episodes is not None:
        if args.episodes <= 0:
            parser.error("--episodes must be positive")
        config["training"]["episodes"] = args.episodes
    run_directory = train(
        config,
        smoke=args.smoke,
        run_name=args.run_name,
        online_instances=args.online_instances,
        algorithm_seed=args.algorithm_seed,
        parallel_envs=args.parallel_envs,
        visdom_enabled=args.visdom_enabled,
        ablation_variant=args.ablation,
        initial_checkpoint=args.initial_checkpoint,
        warm_start_checkpoint=args.warm_start_checkpoint,
    )
    print(f"training artifacts: {run_directory}")


if __name__ == "__main__":
    main()
