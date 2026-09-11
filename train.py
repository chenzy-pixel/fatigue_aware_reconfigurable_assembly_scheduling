from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from agent.ppo import PPOAgent, RolloutBuffer, build_actor_critic
from agent.ppo.parallel import (
    EpisodeRollout,
    ParallelEpisodeRunner,
    TrainingRolloutBatch,
    forced_action_from_mask,
)
from configs import load_config, project_path
from configs.config import public_config
from data.dataset import (
    GeneratedInstanceRecord,
    OnlineInstanceDataset,
    validate_algorithm_seed,
)
from data.models import load_instance_yaml
from environment import (
    AssemblySchedulingEnv,
    proxy_return_from_metrics,
)
from eval import (
    evaluate_dataset,
    evaluate_dataset_parallel,
    evaluate_preference_grid_parallel,
    evaluate_representative_diagnostic,
    load_configured_instance,
)
from result import (
    EVALUATION_SCHEMA_VERSION,
    CURRENT_RUNTIME_DIAGNOSTIC_FIELDS,
    aggregate_evaluation_rows,
    build_provenance,
    create_run_directory,
    evaluation_selection_key,
    capture_terminal_output,
    compare_preference_conditioned_checkpoints,
)
from result.io import write_config, write_csv, write_json
from result.visdom_dashboard import (
    create_training_dashboard,
    override_visdom_enabled,
    resolve_visdom_settings,
)
from utils import set_seed
from training import (
    PARETO_PROMOTION_MODE,
    SINGLE_OBJECTIVE_PROMOTION_MODE,
    TrainingPhaseController,
)

SINGLE_OBJECTIVE_METRICS = {
    "flow": "flow_time_objective",
    "cost": "reconfiguration_cost",
    "variance": "worker_load_variance",
}


def _row_is_physically_safe(row: dict, fatigue_tolerance: float) -> bool:
    return bool(
        int(row.get("schedule_violation_count", 0)) == 0
        and float(row.get("maximum_worker_fatigue", math.inf))
        <= float(row.get("safe_fatigue_limit", -math.inf))
        + float(fatigue_tolerance)
    )


def _rows_are_physically_safe(rows: list[dict], fatigue_tolerance: float) -> bool:
    return bool(rows) and all(
        _row_is_physically_safe(row, fatigue_tolerance) for row in rows
    )


def _checkpoint_eligible_validation_event(event: str, promotion: str) -> bool:
    """Only an independently audited candidate may become accepted."""
    if promotion != SINGLE_OBJECTIVE_PROMOTION_MODE:
        raise ValueError("unsupported checkpoint promotion protocol")
    return event == "accepted"


def _promote_accepted_checkpoint(
    *,
    event: str,
    config: dict,
    phase_controller: TrainingPhaseController,
    agent: PPOAgent,
    accepted_checkpoint: Path,
    best_checkpoint: Path,
    completed_episodes: int,
    parallel_envs: int,
    validation_row: dict,
) -> bool:
    """Persist an audited accepted model and its exact best-checkpoint copy."""

    if not _checkpoint_eligible_validation_event(
        event,
        phase_controller.quality_checkpoint_promotion,
    ):
        return False
    if not phase_controller.is_formally_accepted:
        raise RuntimeError("accepted event lacks complete formal audit state")
    agent.save(
        accepted_checkpoint,
        metadata={
            **_checkpoint_protocol_metadata(config),
            **_single_objective_checkpoint_metadata(
                phase_controller,
                checkpoint_role="accepted",
            ),
            "checkpoint_role": "accepted",
            "seed": config["seed"],
            "parallel_envs": parallel_envs,
            "accepted_episode": completed_episodes,
            "validation": validation_row,
        },
    )
    shutil.copyfile(accepted_checkpoint, best_checkpoint)
    return True


def _validation_manifest_path(config: dict) -> Path:
    split = str(config["training"]["validation_split"])
    return project_path(config["paths"]["manifests_root"]) / split / "manifest.json"


def _validate_single_objective_validation_protocol(
    config: dict, *, smoke: bool, validation_limit: int | None
) -> None:
    """Require a deterministic manifest large enough for the formal audit."""
    if smoke or str(
        config["training"]["two_stage"].get("quality_checkpoint_promotion", "")
    ).strip().lower() != SINGLE_OBJECTIVE_PROMOTION_MODE:
        return
    settings = config["training"]["two_stage"][
        "single_objective_promotion"
    ]
    audit_limit = int(settings["audit_instance_limit"])
    audit_offset = int(settings.get("audit_instance_offset", 0))
    if validation_limit != 50 or audit_limit != 200:
        raise ValueError(
            "single-objective validation must use 50 daily instances and "
            "a 200-instance audit"
        )
    manifest_path = _validation_manifest_path(config)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"single-objective validation manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_generator = str(config["generator"]["version"])
    if manifest.get("generator_version") != expected_generator:
        raise ValueError(
            "single-objective validation manifest has a stale generator "
            f"fingerprint: expected {expected_generator}, got "
            f"{manifest.get('generator_version')}"
        )
    files = manifest.get("files", [])
    manifest_count = int(manifest.get("instance_count", len(files)))
    if (
        not isinstance(files, list)
        or manifest_count != len(files)
        or manifest_count < audit_offset + audit_limit
    ):
        raise ValueError(
            "single-objective validation requires a manifest with at least "
            f"{audit_offset + audit_limit} instances/records; found {manifest_count}"
        )


def _validate_pareto_validation_protocol(
    config: dict, *, smoke: bool, validation_limit: int | None
) -> None:
    """Require the frozen V8 universal protocol before a formal run starts."""

    two_stage = config["training"]["two_stage"]
    if str(two_stage.get("quality_checkpoint_promotion", "")) != PARETO_PROMOTION_MODE:
        return
    settings = two_stage.get("pareto_promotion")
    if not isinstance(settings, dict):
        raise ValueError("V8 universal training requires pareto_promotion settings")
    required = {
        "validation_instance_limit": 50,
        "audit_instance_limit": 200,
        "audit_instance_offset": 50,
        "preference_count": 66,
        "bootstrap_replicates": 10_000,
    }
    for name, expected in required.items():
        if int(settings.get(name, -1)) != expected:
            raise ValueError(f"V8 universal protocol requires {name}={expected}")
    if not math.isclose(float(settings.get("simplex_step", math.nan)), 0.1):
        raise ValueError("V8 universal protocol requires simplex_step=0.1")
    if not math.isclose(
        float(settings.get("validation_completion_floor", math.nan)), 0.95
    ) or not math.isclose(
        float(settings.get("audit_completion_floor", math.nan)), 0.98
    ):
        raise ValueError("V8 universal completion floors must be 0.95/0.98")
    if int(settings.get("audit_max_failed_instances", -1)) != 4:
        raise ValueError("V8 universal audit permits at most four failed instances")
    preference = config.get("preference", {}).get("quality", {})
    if (
        preference.get("mode") != "universal_sobol_v1"
        or int(preference.get("block_size", -1)) != 20
        or int(preference.get("endpoint_repeats", -1)) != 2
        or int(preference.get("sobol_count", -1)) != 14
    ):
        raise ValueError("V8 universal preference sampling must use 20/2/14 quotas")
    if smoke:
        return
    if validation_limit != 50 or int(config["training"]["validation_interval_episodes"]) != 100:
        raise ValueError("V8 universal validation requires 50 instances every 100 quality episodes")
    scalarizer = config.get("objective_scalarizer", {})
    manifest_sha = str(scalarizer.get("normalization_manifest_sha256") or "")
    content_sha = str(scalarizer.get("normalization_manifest_content_sha256") or "")
    if (
        scalarizer.get("scale_source") != "frozen_manifest"
        or scalarizer.get("type") != "normalized_augmented_tchebycheff_v1"
        or not math.isclose(float(scalarizer.get("rho", math.nan)), 0.05)
        or len(manifest_sha) != 64
        or len(content_sha) != 64
        or config.get("network", {}).get("normalization_manifest_sha256")
        != manifest_sha
    ):
        raise ValueError(
            "formal V8 universal training requires one verified frozen normalization manifest"
        )
    bounds = settings.get("endpoint_prediction_upper_bounds")
    if not isinstance(bounds, dict) or set(bounds) != set(SINGLE_OBJECTIVE_METRICS):
        raise ValueError("V8 universal training requires all specialist endpoint bounds")
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in bounds.values()):
        raise ValueError("V8 specialist endpoint bounds must be finite and positive")
    manifest_path = _validation_manifest_path(config)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"V8 validation manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    if (
        manifest.get("generator_version") != str(config["generator"]["version"])
        or not isinstance(files, list)
        or int(manifest.get("instance_count", len(files))) != len(files)
        or len(files) < 250
    ):
        raise ValueError("V8 validation manifest must contain at least 200 current instances")


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
        "result_schema_version": EVALUATION_SCHEMA_VERSION,
        "normalization_manifest_sha256": config.get(
            "objective_scalarizer", {}
        ).get("normalization_manifest_sha256"),
        "provenance": provenance,
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
                "minimum_worker_alternatives",
                "wait_total_time",
                "production_wait_time",
                "worker_wait_time",
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
    instance_offset: int = 0,
    sampling_seeds: list[int],
    greedy_only: bool = False,
    runner: ParallelEpisodeRunner | None = None,
) -> dict[str, object]:
    """Load an isolated agent and produce the only final reported metrics."""
    if instance_offset and not greedy_only:
        raise ValueError("offset checkpoint re-evaluation currently requires greedy_only")
    evaluation_agent = PPOAgent(
        build_actor_critic(bootstrap_observation, config["network"]),
        config["ppo"],
        device=config["device"],
    )
    metadata = evaluation_agent.load(checkpoint, load_optimizer=False)
    if runner is None:
        greedy_rows, _, _, greedy = evaluate_dataset(
            config,
            dataset_name=dataset_name,
            policy_name="ppo",
            ppo_agent=evaluation_agent,
            instance_limit=instance_limit,
            instance_offset=instance_offset,
            decode_mode="greedy",
        )
    else:
        greedy_rows, greedy = evaluate_dataset_parallel(
            config,
            dataset_name=dataset_name,
            ppo_agent=evaluation_agent,
            runner=runner,
            instance_limit=instance_limit,
            instance_offset=instance_offset,
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
            runner=runner,
            use_parallel=runner is not None,
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
            "instance_offset": int(instance_offset),
            "greedy": True,
            "sampling_seeds": list(sampling_seeds),
            "execution_mode": "parallel" if runner is not None else "serial",
            "parallel_envs": int(greedy.get("parallel_envs", 1)),
        },
        "greedy": greedy,
        "sampled": sampled,
    }


def _reevaluate_checkpoint_with_parallel_runner(
    config: dict,
    *,
    checkpoint: Path,
    bootstrap_observation,
    dataset_name: str,
    instance_limit: int | None,
    instance_offset: int = 0,
    sampling_seeds: list[int],
    greedy_only: bool,
    template,
    episode_count: int,
    parallel_worker_count: int,
) -> dict[str, object]:
    """Run final checkpoint verification on an isolated worker pool."""
    with ParallelEpisodeRunner(
        config=config,
        template=template,
        episode_count=episode_count,
        worker_count=parallel_worker_count,
    ) as runner:
        return _reevaluate_checkpoint_from_disk(
            config,
            checkpoint=checkpoint,
            bootstrap_observation=bootstrap_observation,
            dataset_name=dataset_name,
            instance_limit=instance_limit,
            instance_offset=instance_offset,
            sampling_seeds=sampling_seeds,
            greedy_only=greedy_only,
            runner=runner,
        )


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
            "single-objective checkpoint failed final "
            f"{phase_controller.single_objective_audit_instance_limit}-instance audit: "
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
    saved = metadata.get("accepted_single_objective_audit_value")
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
        "single_objective_name": phase_controller.single_objective_name,
        "single_objective_window_statistic": (
            phase_controller.single_objective_window_statistic
        ),
        "accepted_single_objective_value": (
            phase_controller.accepted_single_objective_value
        ),
        "accepted_single_objective_audit_value": (
            phase_controller.accepted_single_objective_audit_value
        ),
        "accepted_single_objective_failed_instances": (
            phase_controller.accepted_single_objective_failed_instances
        ),
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
        "single_objective_audit_instance_offset": (
            phase_controller.single_objective_audit_instance_offset
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
        "formal_eligible": phase_controller.is_formally_accepted,
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
            instance_offset=phase_controller.single_objective_audit_instance_offset,
            decode_mode="greedy",
        )
    else:
        rows, _, _, audit = evaluate_dataset(
            config,
            dataset_name=dataset_name,
            policy_name="ppo",
            ppo_agent=ppo_agent,
            instance_limit=limit,
            instance_offset=phase_controller.single_objective_audit_instance_offset,
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
    "worker_assign_action_count",
    "wait_action_count",
    "production_wait_action_count",
    "worker_wait_action_count",
)


FORCED_ACTION_COUNT_FIELDS = (
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
        **{
            name: metrics.get(name, 0)
            for name in CURRENT_RUNTIME_DIAGNOSTIC_FIELDS
        },
        "wait_reason_counts": json.dumps(
            metrics.get("wait_reason_counts", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "wait_mask_reason_counts": json.dumps(
            metrics.get("wait_mask_reason_counts", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "minimum_worker_alternatives": metrics.get(
            "minimum_worker_alternatives"
        ),
        "wait_total_ticks": metrics.get("wait_total_ticks"),
        "wait_total_time": metrics.get("wait_total_time"),
        "production_wait_ticks": metrics.get("production_wait_ticks"),
        "production_wait_time": metrics.get("production_wait_time"),
        "worker_wait_ticks": metrics.get("worker_wait_ticks"),
        "worker_wait_time": metrics.get("worker_wait_time"),
        "wait_min_estimated_deadline_slack_ticks": metrics.get(
            "wait_min_estimated_deadline_slack_ticks"
        ),
        **{
            name: metrics.get(name, 0)
            for name in (
                "reconfiguration_reuse_count",
                "qualification_scarcity_regret",
                "qualification_scarcity_decision_count",
            )
        },
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
        "worker_matching_deficit_event_count",
        "minimum_worker_alternatives",
        "wait_total_time",
        "production_wait_time",
        "worker_wait_time",
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


class TrainingEngine:
    """One collector/checkpoint pipeline for both serial and parallel training."""

    def __init__(
        self,
        config: dict,
        *,
        smoke: bool = False,
        run_name: str | None = None,
        algorithm_seed: int | None = None,
        worker_count: int | None = None,
        visdom_enabled: bool | None = None,
        initial_checkpoint: str | Path | None = None,
    ) -> None:
        self.config = deepcopy(config)
        self.smoke = bool(smoke)
        self.run_name = run_name
        self.algorithm_seed = algorithm_seed
        self.worker_count = worker_count
        self.visdom_enabled = visdom_enabled
        self.initial_checkpoint = initial_checkpoint

    def run(self) -> Path:
        config = self.config
        override_visdom_enabled(config, self.visdom_enabled)
        compression = config["training"].get("forced_action_compression", False)
        if not isinstance(compression, bool):
            raise ValueError("training.forced_action_compression must be boolean")
        if compression and float(config["ppo"]["gamma"]) != 1.0:
            raise ValueError("forced action compression requires ppo.gamma = 1.0")
        local_forced = config["training"].get(
            "worker_local_physical_forced_actions", True
        )
        if not isinstance(local_forced, bool):
            raise ValueError(
                "training.worker_local_physical_forced_actions must be boolean"
            )
        seed = validate_algorithm_seed(
            config,
            int(config["seed"])
            if self.algorithm_seed is None
            else self.algorithm_seed,
        )
        config["seed"] = seed
        set_seed(seed)
        if self.initial_checkpoint is not None:
            config["training"]["initial_checkpoint"] = str(
                Path(self.initial_checkpoint).resolve()
            )
        episodes = int(
            config["training"]["smoke_episodes"]
            if self.smoke
            else config["training"]["episodes"]
        )
        configured_workers = int(
            config["training"]["smoke_parallel_envs"]
            if self.smoke
            else config["training"]["parallel_envs"]
        )
        workers = configured_workers if self.worker_count is None else int(self.worker_count)
        if workers < 1:
            raise ValueError("worker_count must be positive")
        validation_workers = (
            min(workers, episodes)
            if self.smoke or self.worker_count is not None
            else int(config["training"]["validation_parallel_envs"])
        )
        if validation_workers < 1:
            raise ValueError("training.validation_parallel_envs must be positive")
        interval = int(config["training"]["validation_interval_episodes"])
        if interval < 1:
            raise ValueError("training.validation_interval_episodes must be positive")
        if workers > 1 and not self.smoke and interval % workers != 0:
            raise ValueError(
                "validation_interval_episodes must be divisible by worker_count"
            )
        threads = int(config["training"]["torch_num_threads"])
        if threads < 1:
            raise ValueError("training.torch_num_threads must be positive")
        torch.set_num_threads(threads)
        key = "smoke_parallel_envs" if self.smoke else "parallel_envs"
        config["training"][key] = workers
        config["training"]["validation_parallel_envs"] = validation_workers
        return _train_parallel(
            config,
            smoke=self.smoke,
            run_name=self.run_name,
            episodes=episodes,
            parallel_envs=min(workers, episodes),
            validation_parallel_envs=validation_workers,
            initial_checkpoint=self.initial_checkpoint,
        )


def train(
    config: dict,
    *,
    smoke: bool = False,
    run_name: str | None = None,
    online_instances: bool | None = None,
    algorithm_seed: int | None = None,
    parallel_envs: int | None = None,
    visdom_enabled: bool | None = None,
    initial_checkpoint: str | Path | None = None,
) -> Path:
    if online_instances is False:
        raise ValueError("latest-only training uses the online instance collector")
    return TrainingEngine(
        config,
        smoke=smoke,
        run_name=run_name,
        algorithm_seed=algorithm_seed,
        worker_count=parallel_envs,
        visdom_enabled=visdom_enabled,
        initial_checkpoint=initial_checkpoint,
    ).run()


def _train_parallel(
    config: dict,
    *,
    smoke: bool,
    run_name: str | None,
    episodes: int,
    parallel_envs: int,
    validation_parallel_envs: int,
    initial_checkpoint: str | Path | None = None,
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
    if initial_checkpoint is not None:
        agent.load(initial_checkpoint, load_optimizer=True)
    run_directory = create_run_directory(
        project_path(config["paths"]["result_root"]),
        label="train_smoke_parallel" if smoke else "train_parallel",
        run_name=run_name,
    )
    write_config(run_directory, config)
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
    _validate_pareto_validation_protocol(
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
    pareto_validation_instance_rows: list[dict] = []
    pareto_audit_instance_rows: list[dict] = []
    single_objective_audit_rows: list[dict] = []
    single_objective_audit_failure_rows: list[dict] = []
    instance_ids: list[str] = []
    best_checkpoint = run_directory / "best_checkpoint.pt"
    best_feasibility_checkpoint = (
        run_directory / "best_feasibility_checkpoint.pt"
    )
    phase1_checkpoint = run_directory / "phase1_checkpoint.pt"
    accepted_checkpoint = run_directory / "accepted_checkpoint.pt"
    candidate_checkpoint = run_directory / "audit_candidate_checkpoint.pt"
    safe_checkpoint = run_directory / "safe_checkpoint.pt"
    last_checkpoint = run_directory / "last_checkpoint.pt"
    best_validation: dict | None = None
    best_feasibility_validation: dict | None = None
    best_feasibility_instance_rows: list[dict] = []
    last_sampled_validation: dict | None = None
    last_sampled_validation_episode: int | None = None
    best_score: tuple[float, float, float, float] | None = None
    pareto_incumbent_rows: list[dict] | None = None
    pareto_incumbent_checkpoint: Path | None = None
    phase_controller = TrainingPhaseController.from_config(config)
    stability_controller = ValidationStabilityController.from_config(config)
    total_transitions = 0
    total_environment_steps = 0
    total_forced_actions = 0
    total_worker_step_commands = 0
    total_worker_local_physical_forced_actions = 0
    total_sampling_time = 0.0
    total_inference_time = 0.0
    total_update_time = 0.0
    update_id = 0
    quality_episode_count = 0
    with ParallelEpisodeRunner(
        config=config,
        template=template,
        episode_count=episodes,
        worker_count=runner_worker_count,
    ) as runner:
        for batch_start in range(0, episodes, parallel_envs):
            reward_phase = phase_controller.phase
            episode_indices = list(
                range(
                    batch_start,
                    min(batch_start + parallel_envs, episodes),
                )
            )
            quality_episode_indices = (
                list(
                    range(
                        quality_episode_count,
                        quality_episode_count + len(episode_indices),
                    )
                )
                if reward_phase == "quality"
                else None
            )
            rollout = runner.collect_training_batch(
                agent,
                episode_indices,
                gamma=float(config["ppo"]["gamma"]),
                gae_lambda=float(config["ppo"]["gae_lambda"]),
                step_limit=step_limit,
                reward_phase=reward_phase,
                quality_episode_indices=quality_episode_indices,
            )
            if reward_phase == "quality":
                quality_episode_count += len(episode_indices)
            update_start = time.perf_counter()
            if rollout.transition_count == 0:
                raise RuntimeError(
                    "training batch contains no policy transitions after "
                    "forced action compression"
                )
            losses = agent.update(rollout.buffer)
            update_time = time.perf_counter() - update_start
            update_id += 1
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
                    "update_id": update_id,
                    "instance_id": episode.instance_id,
                    "instance_seed": episode.metadata["seed"],
                    "pressure_type": episode.metadata[
                        "pressure_type"
                    ],
                    "cost_profile": episode.metadata["cost_profile"],
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
                    ),
                    "reward_training": episode.reward_sum,
                    "expected_reward": episode.expected_reward,
                    "reward_identity_error": (
                        episode.base_reward_sum - episode.expected_reward
                    ),
                    "reward_phase": episode.reward_phase,
                    "preference": episode.metadata.get("preference"),
                    "preference_key": episode.metadata.get("preference_key"),
                    **episode.policy_diagnostics,
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
            completed_episodes = episode_indices[-1] + 1
            regular_validation_due = (
                (
                    quality_episode_count > 0
                    and quality_episode_count % 100 == 0
                )
                if (
                    reward_phase == "quality"
                    and phase_controller.quality_checkpoint_promotion
                    == PARETO_PROMOTION_MODE
                )
                else completed_episodes % validation_interval == 0
            ) or completed_episodes == episodes
            should_validate = phase_controller.should_validate(
                regular_validation_due
            )
            if should_validate:
                pareto_quality_validation = bool(
                    reward_phase == "quality"
                    and phase_controller.quality_checkpoint_promotion
                    == PARETO_PROMOTION_MODE
                )
                if pareto_quality_validation:
                    validation_instance_rows, validation = (
                        evaluate_preference_grid_parallel(
                            config,
                            dataset_name=validation_split,
                            ppo_agent=agent,
                            runner=runner,
                            instance_limit=50,
                        )
                    )
                    pareto_validation_instance_rows.extend(
                        {
                            "validation_episode": completed_episodes,
                            **row,
                        }
                        for row in validation_instance_rows
                    )
                else:
                    validation_instance_rows, validation = evaluate_dataset_parallel(
                        config,
                        dataset_name=validation_split,
                        ppo_agent=agent,
                        runner=runner,
                        instance_limit=validation_limit,
                    )
                validation["physical_safety_pass"] = _rows_are_physically_safe(
                    validation_instance_rows, 1e-9
                )
                validation_row = _validation_log_row(
                    validation,
                    completed_episodes=completed_episodes,
                )
                score = (
                    _single_objective_guard_score(
                        validation, phase_controller.single_objective_name
                    )
                    if phase_controller.single_objective_name is not None
                    else evaluation_selection_key(validation)
                )
                stability = stability_controller.observe_greedy(
                    score,
                    validation["completion_rate"],
                    completed_episodes=completed_episodes,
                    feasibility_phase=reward_phase == "feasibility",
                )
                if (
                    not pareto_quality_validation
                    and stability_controller.should_run_sampled(
                    final_validation=completed_episodes == episodes,
                    completed_episodes=completed_episodes,
                    )
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
                        use_parallel=True,
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
                    truncated_count=int(validation.get("truncated_count", 0)),
                    schedule_violation_count=int(
                        validation.get("schedule_violation_count", 0)
                    ),
                    physical_safety_pass=bool(validation["physical_safety_pass"]),
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
                if pareto_quality_validation:
                    if pareto_incumbent_rows is None or pareto_incumbent_checkpoint is None:
                        raise RuntimeError("V8 quality validation has no phase-1 incumbent")
                    settings = config["training"]["two_stage"]["pareto_promotion"]
                    endpoint_bounds = settings.get(
                        "endpoint_prediction_upper_bounds"
                    )
                    if not isinstance(endpoint_bounds, dict):
                        raise RuntimeError(
                            "V8 universal promotion requires specialist endpoint prediction bounds"
                        )
                    scales_mapping = config["objective_scalarizer"]["scales"]
                    scales = tuple(
                        float(scales_mapping[name])
                        for name in ("flow", "cost", "variance")
                    )
                    pareto_result = compare_preference_conditioned_checkpoints(
                        validation_instance_rows,
                        pareto_incumbent_rows,
                        scales=scales,
                        endpoint_prediction_upper_bounds=endpoint_bounds,
                        bootstrap_seed=int(settings.get("bootstrap_seed", 20260811)),
                    )
                    daily_validation_event = phase_controller.observe_pareto_promotion(
                        pareto_result,
                        completed_episodes=completed_episodes,
                        audited=False,
                    )
                    validation_event = daily_validation_event
                    validation_row["validation_event"] = daily_validation_event
                    validation_row["pareto_validation_result"] = json.dumps(
                        pareto_result, ensure_ascii=False, sort_keys=True
                    )
                    if daily_validation_event == "audit_required":
                        agent.save(
                            candidate_checkpoint,
                            metadata={
                                **_checkpoint_protocol_metadata(config),
                                "checkpoint_role": "pareto_audit_pending_candidate",
                                "pareto_validation_result": pareto_result,
                            },
                        )
                        candidate_network = build_actor_critic(
                            bootstrap_observation, config["network"]
                        )
                        candidate_agent = PPOAgent(
                            candidate_network, config["ppo"], device=config["device"]
                        )
                        candidate_metadata = candidate_agent.load(
                            candidate_checkpoint, load_optimizer=False
                        )
                        candidate_audit_rows, _ = evaluate_preference_grid_parallel(
                            config,
                            dataset_name=validation_split,
                            ppo_agent=candidate_agent,
                            runner=runner,
                            instance_limit=200,
                            instance_offset=int(settings["audit_instance_offset"]),
                        )
                        incumbent_network = build_actor_critic(
                            bootstrap_observation, config["network"]
                        )
                        incumbent_agent = PPOAgent(
                            incumbent_network, config["ppo"], device=config["device"]
                        )
                        incumbent_agent.load(
                            pareto_incumbent_checkpoint, load_optimizer=False
                        )
                        incumbent_audit_rows, _ = evaluate_preference_grid_parallel(
                            config,
                            dataset_name=validation_split,
                            ppo_agent=incumbent_agent,
                            runner=runner,
                            instance_limit=200,
                            instance_offset=int(settings["audit_instance_offset"]),
                        )
                        pareto_audit_instance_rows.extend(
                            {
                                "audit_episode": completed_episodes,
                                "arm": arm,
                                **row,
                            }
                            for arm, audit_rows in (
                                ("candidate", candidate_audit_rows),
                                ("incumbent", incumbent_audit_rows),
                            )
                            for row in audit_rows
                        )
                        pareto_audit_result = compare_preference_conditioned_checkpoints(
                            candidate_audit_rows,
                            incumbent_audit_rows,
                            scales=scales,
                            endpoint_prediction_upper_bounds=endpoint_bounds,
                            audit=True,
                            bootstrap_seed=int(settings.get("bootstrap_seed", 20260811)),
                        )
                        audit_event = phase_controller.observe_pareto_promotion(
                            pareto_audit_result,
                            completed_episodes=completed_episodes,
                            audited=True,
                        )
                        validation_event = audit_event
                        validation_row["validation_event"] = audit_event
                        validation_row["pareto_audit_result"] = json.dumps(
                            pareto_audit_result,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        if audit_event == "accepted":
                            agent.save(
                                accepted_checkpoint,
                                metadata={
                                    **_checkpoint_protocol_metadata(config),
                                    "checkpoint_role": "accepted",
                                    "accepted_episode": completed_episodes,
                                    "pareto_validation_result": pareto_result,
                                    "pareto_audit_result": pareto_audit_result,
                                },
                            )
                            accepted_payload = torch.load(
                                accepted_checkpoint,
                                map_location="cpu",
                                weights_only=False,
                            )
                            accepted_weights_sha = accepted_payload.get(
                                "metadata", {}
                            ).get("network_weights_sha256")
                            if accepted_weights_sha != candidate_metadata.get(
                                "network_weights_sha256"
                            ):
                                raise RuntimeError(
                                    "audited candidate and accepted checkpoint weights diverged"
                                )
                            shutil.copyfile(accepted_checkpoint, best_checkpoint)
                            pareto_incumbent_rows = [
                                dict(row) for row in validation_instance_rows
                            ]
                            pareto_incumbent_checkpoint = accepted_checkpoint
                            best_score = score
                            best_validation = validation_row
                        if candidate_checkpoint.exists():
                            candidate_checkpoint.unlink()
                if (
                    daily_validation_event == "audit_required"
                    and phase_controller.quality_checkpoint_promotion
                    == SINGLE_OBJECTIVE_PROMOTION_MODE
                ):
                    window_median = phase_controller.last_promotion_diagnostics.get(
                        "window_objective_statistic"
                    )
                    if window_median is None:
                        raise RuntimeError(
                            "single-objective audit is missing its window median"
                        )
                    agent.save(
                        candidate_checkpoint,
                        metadata={
                            **_checkpoint_protocol_metadata(config),
                            **_single_objective_checkpoint_metadata(
                                phase_controller,
                                checkpoint_role="audit_pending_candidate",
                            ),
                            "daily_validation": validation_row,
                        },
                    )
                    audit_instance_rows, audit = _evaluate_single_objective_audit(
                        config,
                        dataset_name=validation_split,
                        ppo_agent=agent,
                        phase_controller=phase_controller,
                        runner=runner,
                        use_parallel=True,
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
                    validation_event = audit_event
                    validation_row.update(
                        {
                            "validation_event": audit_event,
                            "audit_event": audit_event,
                            "audit_failed_instance_count": audit_log_row.get(
                                "audit_failed_instance_count"
                            ),
                            "audit_completion_rate": audit_log_row.get(
                                "audit_completion_rate"
                            ),
                        }
                    )
                    if candidate_checkpoint.exists():
                        candidate_checkpoint.unlink()
                validation_rows.append(validation_row)
                if validation["completion_rate"] >= 1.0 - 1e-12:
                    agent.save(
                        safe_checkpoint,
                        metadata={
                            **_checkpoint_protocol_metadata(config),
                            "checkpoint_role": "latest_safe",
                            "safe_episode": completed_episodes,
                            "validation": validation_row,
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
                is_new_best = bool(
                    validation_event == "accepted"
                    and phase_controller.quality_checkpoint_promotion
                    == PARETO_PROMOTION_MODE
                )
                if is_new_best:
                    update_row["candidate_status"] = "accepted"
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = "accepted"
                if validation_event == "transition":
                    transition_metadata = {
                        **_checkpoint_protocol_metadata(config),
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
                    if (
                        phase_controller.quality_checkpoint_promotion
                        == PARETO_PROMOTION_MODE
                    ):
                        pareto_incumbent_rows, _ = (
                            evaluate_preference_grid_parallel(
                                config,
                                dataset_name=validation_split,
                                ppo_agent=agent,
                                runner=runner,
                                instance_limit=50,
                            )
                        )
                        pareto_incumbent_checkpoint = phase1_checkpoint
                        pareto_validation_instance_rows.extend(
                            {
                                "validation_episode": completed_episodes,
                                "arm": "phase1_incumbent",
                                **row,
                            }
                            for row in pareto_incumbent_rows
                        )
                    update_row["candidate_status"] = "phase_transition"
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = "phase_transition"
                if (
                    phase_controller.quality_checkpoint_promotion
                    == SINGLE_OBJECTIVE_PROMOTION_MODE
                    and _promote_accepted_checkpoint(
                    event=validation_event,
                    config=config,
                    phase_controller=phase_controller,
                    agent=agent,
                    accepted_checkpoint=accepted_checkpoint,
                    best_checkpoint=best_checkpoint,
                    completed_episodes=completed_episodes,
                    parallel_envs=parallel_envs,
                    validation_row=validation_row,
                    )
                ):
                    best_score = score
                    best_validation = validation_row
                    is_new_best = True
                    update_row["candidate_status"] = "accepted"
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = "accepted"
                elif validation_event in {
                    "not_promoted",
                    "rejected",
                    "audit_rejected",
                    "audit_passed_not_accepted",
                }:
                    update_row["candidate_status"] = "not_promoted"
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = "not_promoted"
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
    formal_eligible = phase_controller.is_formally_accepted
    accepted_checkpoint_exists = accepted_checkpoint.exists()
    if formal_eligible != accepted_checkpoint_exists:
        raise RuntimeError(
            "formal acceptance state and accepted checkpoint presence diverged"
        )
    if formal_eligible and not best_checkpoint.exists():
        raise RuntimeError("formally accepted training lacks best_checkpoint.pt")
    if formal_eligible and (best_validation is None or best_score is None):
        raise RuntimeError("training completed without validation")
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
            "updates": update_id,
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
            "last_sampled_validation": last_sampled_validation,
            "formal_training_status": (
                phase_controller.formal_training_status
            ),
            "training_phase": phase_controller.as_dict(),
            "run_formal_eligible": formal_eligible,
        }
    agent.save(
        last_checkpoint,
        metadata={
            **final_metadata,
            "checkpoint_role": "last_online",
            "formal_eligible": False,
        },
    )
    checkpoint: Path | None = accepted_checkpoint if formal_eligible else None
    last_candidate_checkpoint: Path | None = None
    final_checkpoint_evaluation = None
    checkpoint_sha256 = None
    if formal_eligible:
        checkpoint_sha256 = _checkpoint_sha256(accepted_checkpoint)
        best_sha256 = _checkpoint_sha256(best_checkpoint)
        if checkpoint_sha256 != best_sha256:
            raise RuntimeError(
                "accepted and best checkpoint hashes diverged"
            )
        if phase_controller.quality_checkpoint_promotion == PARETO_PROMOTION_MODE:
            verification_agent = PPOAgent(
                build_actor_critic(bootstrap_observation, config["network"]),
                config["ppo"],
                device=config["device"],
            )
            verified_metadata = verification_agent.load(
                accepted_checkpoint, load_optimizer=False
            )
            audit_result = verified_metadata.get("pareto_audit_result")
            if not isinstance(audit_result, dict) or not bool(
                audit_result.get("accepted", False)
            ):
                raise RuntimeError(
                    "accepted V8 checkpoint does not contain a successful 200x66 audit"
                )
            final_checkpoint_evaluation = {
                "checkpoint": str(accepted_checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_metadata": verified_metadata,
                "evaluation_config": {
                    "dataset": validation_split,
                    "instance_offset": int(
                        config["training"]["two_stage"]["pareto_promotion"][
                            "audit_instance_offset"
                        ]
                    ),
                    "instance_limit": 200,
                    "preference_count": 66,
                    "execution_mode": "checkpoint_reloaded_audit",
                },
                "pareto_audit_result": audit_result,
            }
        else:
            # Specialists retain the independent single-objective disk audit.
            final_checkpoint_evaluation = _reevaluate_checkpoint_with_parallel_runner(
                config,
                checkpoint=accepted_checkpoint,
                bootstrap_observation=bootstrap_observation,
                dataset_name=validation_split,
                instance_limit=phase_controller.single_objective_audit_instance_limit,
                instance_offset=phase_controller.single_objective_audit_instance_offset,
                sampling_seeds=[],
                greedy_only=True,
                template=template,
                episode_count=episodes,
                parallel_worker_count=validation_parallel_envs,
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
            invalidated_best_checkpoint = (
                run_directory / "invalidated_best_checkpoint.pt"
            )
            accepted_checkpoint.replace(invalidated_checkpoint)
            best_checkpoint.replace(invalidated_best_checkpoint)
            write_csv(run_directory / "train_log.csv", rows)
            write_csv(run_directory / "update_log.csv", update_rows)
            write_csv(run_directory / "validation_log.csv", validation_rows)
            write_csv(
                run_directory / "pareto_validation_instance_metrics.csv",
                pareto_validation_instance_rows,
            )
            write_csv(
                run_directory / "pareto_audit_instance_metrics.csv",
                pareto_audit_instance_rows,
            )
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
                    "invalidated_best_checkpoint": str(
                        invalidated_best_checkpoint
                    ),
                    "formal_eligible": False,
                },
            )
            raise RuntimeError(
                "single-objective accepted checkpoint failed final audit"
            ) from error
    summary_checkpoint = checkpoint or last_checkpoint
    summary_provenance = build_provenance(
        config,
        dataset_manifest_path=_validation_manifest_path(config),
        checkpoint_path=summary_checkpoint,
        checkpoint_metadata=_checkpoint_protocol_metadata(config),
    )
    write_csv(run_directory / "train_log.csv", rows)
    write_csv(run_directory / "update_log.csv", update_rows)
    write_csv(run_directory / "validation_log.csv", validation_rows)
    if pareto_validation_instance_rows:
        write_csv(
            run_directory / "pareto_validation_instance_metrics.csv",
            pareto_validation_instance_rows,
        )
    if pareto_audit_instance_rows:
        write_csv(
            run_directory / "pareto_audit_instance_metrics.csv",
            pareto_audit_instance_rows,
        )
    if single_objective_audit_rows:
        write_csv(
            run_directory / "single_objective_audit_log.csv",
            single_objective_audit_rows,
        )
    if single_objective_audit_failure_rows:
        write_csv(
            run_directory / "single_objective_audit_failures.csv",
            single_objective_audit_failure_rows,
        )
    total_training_time = total_sampling_time + total_update_time
    write_json(
        run_directory / "summary.json",
        {
            "episodes": episodes,
            "online_instances": True,
            "parallel_envs": parallel_envs,
            "validation_parallel_envs": validation_parallel_envs,
            "updates": update_id,
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
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "checkpoint_sha256": checkpoint_sha256,
            "provenance": summary_provenance,
            "final_checkpoint_evaluation": final_checkpoint_evaluation,
            "accepted_checkpoint": (
                str(accepted_checkpoint)
                if accepted_checkpoint.exists()
                else None
            ),
            "last_checkpoint": str(last_checkpoint),
            "safe_checkpoint": (
                str(safe_checkpoint) if safe_checkpoint.exists() else None
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
            "formal_eligible": formal_eligible,
            "training_phase": phase_controller.as_dict(),
            "single_objective_audit": {
                "daily_validation_instance_limit": validation_limit,
                "audit_instance_limit": (
                    phase_controller.single_objective_audit_instance_limit
                ),
                "audit_completion_target": (
                    phase_controller.single_objective_audit_completion_target
                ),
                "audit_max_failed_instances": (
                    phase_controller.single_objective_audit_max_failed_instances
                ),
                "audit_count": len(single_objective_audit_rows),
                "audit_failure_row_count": len(
                    single_objective_audit_failure_rows
                ),
                "accepted_status": phase_controller.formal_training_status,
                "project_formal_completion_target": 1.0,
            },
            "pareto_promotion": {
                "validation_instance_row_count": len(
                    pareto_validation_instance_rows
                ),
                "audit_instance_row_count": len(pareto_audit_instance_rows),
                "last_result": phase_controller.last_pareto_promotion,
                "normalization_manifest_sha256": config[
                    "objective_scalarizer"
                ].get("normalization_manifest_sha256"),
            },
            "validation_stability": stability_controller.as_dict(),
            "validation_runs": len(validation_rows),
            "sampled_validation_runs": (
                stability_controller.sampled_validation_runs
            ),
            "last_sampled_validation": last_sampled_validation,
            "late_500_episode_diagnostics": (
                _late_training_diagnostics(rows)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the lightweight PPO policy")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--episodes",
        type=int,
        help="override training.episodes for this run",
    )
    parser.add_argument(
        "--initial-checkpoint",
        help=(
            "initialize policy and optimizer from a compatible checkpoint"
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
    result_root = project_path(config["paths"]["result_root"])
    result_root.mkdir(parents=True, exist_ok=True)
    run_key = Path(args.run_name).name if args.run_name else "unnamed_train"
    staging_log = result_root / f".{run_key}.{os.getpid()}.terminal.log.tmp"
    expected_run = result_root / args.run_name if args.run_name else None
    run_directory: Path | None = None
    exit_code = 0
    try:
        with capture_terminal_output(staging_log):
            print(f"[terminal-log] started_at={datetime.now(timezone.utc).isoformat()}")
            print(
                "[terminal-log] command="
                + shlex.join([Path(sys.executable).name, *sys.argv])
            )
            try:
                run_directory = train(
                    config,
                    smoke=args.smoke,
                    run_name=args.run_name,
                    online_instances=args.online_instances,
                    algorithm_seed=args.algorithm_seed,
                    parallel_envs=args.parallel_envs,
                    visdom_enabled=args.visdom_enabled,
                    initial_checkpoint=args.initial_checkpoint,
                )
                print(f"training artifacts: {run_directory}")
            except KeyboardInterrupt:
                exit_code = 130
                traceback.print_exc()
            except Exception as error:
                exit_code = 1
                traceback.print_exc()
                failure_directory = (
                    expected_run
                    if expected_run is not None and expected_run.is_dir()
                    else run_directory
                )
                if failure_directory is not None:
                    write_json(
                        failure_directory / "failure.json",
                        {
                            "version": "training_failure_v1",
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                            "exception_type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    write_csv(
                        failure_directory / "failure_partial.csv",
                        [{"exception_type": type(error).__name__, "message": str(error)}],
                    )
            finally:
                print(f"[terminal-log] finished_at={datetime.now(timezone.utc).isoformat()}")
                print(f"[terminal-log] exit_code={exit_code}")
    finally:
        if staging_log.exists():
            destination_directory = run_directory or expected_run
            if destination_directory is None:
                destination_directory = result_root
            destination_directory.mkdir(parents=True, exist_ok=True)
            os.replace(staging_log, destination_directory / "terminal.log")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
