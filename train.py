from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agent.ppo import PPOAgent, RolloutBuffer, build_actor_critic
from agent.ppo.parallel import (
    EpisodeRollout,
    ParallelEpisodeRunner,
    TrainingRolloutBatch,
)
from configs import load_config, project_path
from data.dataset import (
    GeneratedInstanceRecord,
    OnlineInstanceDataset,
    validate_algorithm_seed,
)
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv, proxy_return_from_metrics
from eval import (
    evaluate_dataset,
    evaluate_dataset_parallel,
    evaluate_representative_diagnostic,
    load_configured_instance,
)
from result import (
    aggregate_evaluation_rows,
    create_run_directory,
    evaluation_selection_key,
)
from result.io import write_config, write_csv, write_json
from result.visdom_dashboard import (
    create_training_dashboard,
    override_visdom_enabled,
    resolve_visdom_settings,
)
from utils import set_seed


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
    accepted_quality_score: tuple[float, float, float, float] | None = None

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
        if promotion not in {"completion_only", "score_improving"}:
            raise ValueError(
                "two_stage.quality_checkpoint_promotion must be "
                "'completion_only' or 'score_improving'"
            )
        if not bool(settings["quality_validate_every_update"]):
            raise ValueError(
                "hierarchical constrained training requires validation "
                "after every quality-phase update"
            )
        return cls(
            enabled=True,
            completion_target=target,
            consecutive_required=required,
            quality_completion_floor=floor,
            quality_checkpoint_promotion=promotion,
            phase="feasibility",
        )

    def should_validate(self, regular_due: bool) -> bool:
        return self.phase == "quality" or regular_due

    def observe_validation(
        self,
        completion_rate: float,
        *,
        completed_episodes: int,
        score: tuple[float, float, float, float] | None = None,
    ) -> str:
        rate = float(completion_rate)
        if not self.enabled:
            return "legacy"
        if self.phase == "feasibility":
            if rate >= self.completion_target:
                self.consecutive_successes += 1
            else:
                self.consecutive_successes = 0
            if self.consecutive_successes >= self.consecutive_required:
                self.phase = "quality"
                self.phase_transition_episode = int(completed_episodes)
                self.accepted_quality_score = score
                return "transition"
            return "feasibility"
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
            self.accepted_quality_updates += 1
            if score is not None:
                self.accepted_quality_score = score
            return "accepted"
        self.rejected_quality_updates += 1
        return "rejected"

    @property
    def formal_training_status(self) -> str:
        if not self.enabled:
            return "legacy_weighted_sum"
        if self.phase_transition_episode is None:
            return "feasibility_not_reached"
        return "quality_constrained"

    def as_dict(self) -> dict:
        return {
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
            "formal_training_status": self.formal_training_status,
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
        previous_best_rate = self.best_completion_rate
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
        degraded = bool(
            feasibility_phase
            and not improved
            and previous_best_rate is not None
            and previous_best_rate - rate
            >= self.rollback_completion_drop - 1e-12
        )
        if degraded:
            self.consecutive_degraded_validations += 1
        elif not improved:
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
    observation = environment.reset(instance)
    buffer = RolloutBuffer(
        preserve_graph=agent.requires_graph_observation
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
    }
    inference_time = 0.0
    environment_step_time = 0.0
    while not (environment.terminated or environment.truncated):
        action_mask = environment.get_action_mask()
        inference_start = time.perf_counter()
        action, log_probability, value = agent.act(
            observation,
            action_mask,
        )
        inference_time += time.perf_counter() - inference_start
        step_start = time.perf_counter()
        next_observation, reward_vector, terminated, truncated, _ = (
            environment.step(action)
        )
        environment_step_time += time.perf_counter() - step_start
        scalar_reward = reward_vector.scalarize(
            config["reward"],
            effective_reward_phase,
        )
        buffer.add(
            observation,
            action_mask,
            action,
            log_probability,
            value,
            scalar_reward,
            terminated or truncated,
        )
        observation = next_observation
        reward_sum += scalar_reward
        for name, value in reward_vector.as_dict().items():
            reward_components[name] += float(value)
        step_count += 1
        if step_limit is not None and step_count >= step_limit:
            break
    last_value = 0.0
    if not (environment.terminated or environment.truncated):
        inference_start = time.perf_counter()
        last_value = agent.value(
            observation,
            environment.get_action_mask(),
        )
        inference_time += time.perf_counter() - inference_start
    buffer.compute_gae(
        last_value=last_value,
        gamma=float(config["ppo"]["gamma"]),
        gae_lambda=float(config["ppo"]["gae_lambda"]),
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
        reward_phase=effective_reward_phase,
        reward_components=reward_components,
        expected_reward=proxy_return_from_metrics(
            metrics,
            config["reward"],
            effective_reward_phase,
        ),
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
    return combined


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
        "machine_waiting_for_worker_time": metrics.get(
            "machine_waiting_for_worker_time"
        ),
        "completed_reconfigurations": metrics.get(
            "completed_reconfigurations"
        ),
        "worker_switch_ratio": metrics.get("worker_switch_ratio"),
        "schedule_violation_count": len(
            metrics.get("schedule_violations", [])
        ),
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
    variants = {"E1", "E2", "E3", "R11", "S11", "L11", "Q11"}
    if normalized not in variants:
        raise ValueError(
            "ablation variant must be E1, E2, E3, R11, S11, L11, or Q11"
        )
    config["training"]["ablation_variant"] = normalized
    config["training"]["episodes"] = 600
    config["training"]["validation_interval_episodes"] = 10
    config["seed"] = 11
    control = config["environment"]["worker_resource_control"]
    control["mode"] = "matching_admission_v1"
    config["reward"]["feasibility_shaping"]["enabled"] = normalized in {
        "E2", "E3", "S11", "L11", "Q11"
    }
    config["training"]["two_stage"]["quality_checkpoint_promotion"] = (
        "score_improving" if normalized == "Q11" else "completion_only"
    )
    sampled = config["training"]["validation_control"]["sampled"]
    if normalized in {"R11", "S11", "L11", "Q11"}:
        sampled["episode_milestones"] = [200, 400]
    else:
        sampled.pop("episode_milestones", None)
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
    effective_algorithm_seed = validate_algorithm_seed(
        config,
        int(config["seed"]) if algorithm_seed is None else algorithm_seed,
    )
    config["seed"] = effective_algorithm_seed
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
    network = build_actor_critic(observation, config["network"])
    agent = PPOAgent(network, config["ppo"], device=config["device"])
    run_directory = create_run_directory(
        project_path(config["paths"]["result_root"]),
        label="train_smoke" if smoke else "train",
        run_name=run_name,
    )
    write_config(run_directory, config)
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
    rows: list[dict] = []
    update_rows: list[dict] = []
    validation_rows: list[dict] = []
    instance_ids: list[str] = []
    best_checkpoint = run_directory / "best_checkpoint.pt"
    best_feasibility_checkpoint = (
        run_directory / "best_feasibility_checkpoint.pt"
    )
    phase1_checkpoint = run_directory / "phase1_checkpoint.pt"
    accepted_checkpoint = run_directory / "accepted_checkpoint.pt"
    best_validation: dict | None = None
    best_feasibility_validation: dict | None = None
    best_feasibility_instance_rows: list[dict] = []
    last_sampled_validation: dict | None = None
    best_score: tuple[float, float, float, float] | None = None
    phase_controller = TrainingPhaseController.from_config(config)
    stability_controller = ValidationStabilityController.from_config(config)
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
        observation = environment.reset(instance)
        buffer = RolloutBuffer(
            preserve_graph=agent.requires_graph_observation
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
        }
        inference_time = 0.0
        environment_step_time = 0.0
        while not (environment.terminated or environment.truncated):
            action_mask = environment.get_action_mask()
            inference_start = time.perf_counter()
            action, log_probability, value = agent.act(
                observation, action_mask
            )
            inference_time += time.perf_counter() - inference_start
            environment_step_start = time.perf_counter()
            next_observation, reward_vector, terminated, truncated, _ = (
                environment.step(action)
            )
            environment_step_time += (
                time.perf_counter() - environment_step_start
            )
            scalar_reward = reward_vector.scalarize(
                config["reward"],
                reward_phase,
            )
            buffer.add(
                observation,
                action_mask,
                action,
                log_probability,
                value,
                scalar_reward,
                terminated or truncated,
            )
            observation = next_observation
            reward_sum += scalar_reward
            for name, value in reward_vector.as_dict().items():
                reward_components[name] += float(value)
            step_count += 1
            if smoke and step_count >= smoke_limit:
                break
        last_value = 0.0
        if not (environment.terminated or environment.truncated):
            inference_start = time.perf_counter()
            last_value = agent.value(
                observation, environment.get_action_mask()
            )
            inference_time += time.perf_counter() - inference_start
        buffer.compute_gae(
            last_value=last_value,
            gamma=float(config["ppo"]["gamma"]),
            gae_lambda=float(config["ppo"]["gae_lambda"]),
        )
        sampling_time = time.perf_counter() - sampling_start
        update_start = time.perf_counter()
        losses = agent.update(buffer)
        update_time = time.perf_counter() - update_start
        metrics = environment.metrics()
        metrics["schedule_violations"] = (
            environment.validate_schedule()
        )
        expected_reward = proxy_return_from_metrics(
            metrics,
            config["reward"],
            reward_phase,
        )
        shaping_reward = reward_components["feasibility_shaping"]
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
            "steps": step_count,
            "reward": reward_sum,
            "reward_base": base_reward,
            "reward_shaping": shaping_reward,
            "reward_training": reward_sum,
            "expected_reward": expected_reward,
            "reward_identity_error": base_reward - expected_reward,
            "reward_phase": reward_phase,
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
            validation_row = _validation_log_row(
                validation,
                completed_episodes=completed_episodes,
            )
            score = evaluation_selection_key(validation)
            stability = stability_controller.observe_greedy(
                score,
                validation["completion_rate"],
                completed_episodes=completed_episodes,
                feasibility_phase=reward_phase == "feasibility",
            )
            if stability_controller.should_run_sampled(
                final_validation=completed_episodes == episodes,
                completed_episodes=completed_episodes,
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
                _attach_sampled_validation(
                    validation_row,
                    sampled_validation,
                    completed_episodes=completed_episodes,
                )
            validation_event = phase_controller.observe_validation(
                validation["completion_rate"],
                completed_episodes=completed_episodes,
                score=score,
            )
            validation_row["candidate_phase"] = reward_phase
            validation_row["validation_event"] = validation_event
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
            validation_rows.append(validation_row)
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
            if validation_event == "transition":
                transition_metadata = {
                    "feature_dimensions": observation.feature_dimensions,
                    "edge_feature_dimensions": (
                        observation.edge_feature_dimensions
                    ),
                    "seed": config["seed"],
                    "phase_transition_episode": completed_episodes,
                    "validation": validation_row,
                }
                agent.save(phase1_checkpoint, metadata=transition_metadata)
                agent.save(accepted_checkpoint, metadata=transition_metadata)
                row["candidate_status"] = "phase_transition"
                update_rows[-1]["candidate_status"] = "phase_transition"
            elif validation_event == "accepted":
                agent.save(
                    accepted_checkpoint,
                    metadata={
                        "seed": config["seed"],
                        "accepted_episode": completed_episodes,
                        "validation": validation_row,
                    },
                )
                row["candidate_status"] = "accepted"
                update_rows[-1]["candidate_status"] = "accepted"
            elif validation_event == "rejected":
                if not accepted_checkpoint.exists():
                    raise RuntimeError(
                        "quality candidate rejected before an accepted "
                        "checkpoint was established"
                    )
                agent.load(accepted_checkpoint, load_optimizer=True)
                row["candidate_status"] = "rejected_rolled_back"
                update_rows[-1]["candidate_status"] = (
                    "rejected_rolled_back"
                )
            elif bool(stability["rollback"]):
                if not best_feasibility_checkpoint.exists():
                    raise RuntimeError(
                        "feasibility rollback requested before a best "
                        "checkpoint was established"
                    )
                agent.load(
                    best_feasibility_checkpoint,
                    load_optimizer=True,
                )
                row["candidate_status"] = "feasibility_rolled_back"
                update_rows[-1]["candidate_status"] = (
                    "feasibility_rolled_back"
                )
            agent.set_learning_rate(
                stability_controller.current_learning_rate
            )
            if validation_event == "transition":
                stability_controller.reset_plateau()
            checkpoint_eligible = (
                not phase_controller.enabled
                or validation_event in {"transition", "accepted"}
            )
            is_new_best = checkpoint_eligible and (
                best_score is None or score < best_score
            )
            if is_new_best:
                best_score = score
                best_validation = validation_row
                agent.save(
                    best_checkpoint,
                    metadata={
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
            dashboard.log_validation(
                validation_row,
                best_validation=best_validation,
                phase_state=phase_controller.as_dict(),
            )
            if validation_event in {
                "transition",
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
    formal_eligible = (
        not phase_controller.enabled
        or phase_controller.phase_transition_episode is not None
    )
    if (
        not formal_eligible
        and phase_controller.formal_training_status
        != "feasibility_not_reached"
    ):
        raise RuntimeError("invalid hierarchical training state")
    if formal_eligible and (best_validation is None or best_score is None):
        raise RuntimeError("training completed without validation")
    final_metadata = {
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
            "last_sampled_validation": last_sampled_validation,
            "formal_training_status": (
                phase_controller.formal_training_status
            ),
            "training_phase": phase_controller.as_dict(),
            "formal_eligible": formal_eligible,
        }
    checkpoint: Path | None
    last_candidate_checkpoint: Path | None
    if formal_eligible:
        checkpoint = run_directory / "checkpoint.pt"
        last_candidate_checkpoint = None
        agent.save(checkpoint, metadata=final_metadata)
    else:
        checkpoint = None
        last_candidate_checkpoint = (
            run_directory / "last_candidate_checkpoint.pt"
        )
        agent.save(last_candidate_checkpoint, metadata=final_metadata)
    write_csv(run_directory / "train_log.csv", rows)
    write_csv(run_directory / "update_log.csv", update_rows)
    write_csv(run_directory / "validation_log.csv", validation_rows)
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
            "unique_instance_count": len(set(instance_ids)),
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
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
            "validation_stability": stability_controller.as_dict(),
            "validation_runs": len(validation_rows),
            "sampled_validation_runs": (
                stability_controller.sampled_validation_runs
            ),
            "last_sampled_validation": last_sampled_validation,
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
    instance_ids: list[str] = []
    best_checkpoint = run_directory / "best_checkpoint.pt"
    best_feasibility_checkpoint = (
        run_directory / "best_feasibility_checkpoint.pt"
    )
    phase1_checkpoint = run_directory / "phase1_checkpoint.pt"
    accepted_checkpoint = run_directory / "accepted_checkpoint.pt"
    best_validation: dict | None = None
    best_feasibility_validation: dict | None = None
    best_feasibility_instance_rows: list[dict] = []
    last_sampled_validation: dict | None = None
    best_score: tuple[float, float, float, float] | None = None
    phase_controller = TrainingPhaseController.from_config(config)
    stability_controller = ValidationStabilityController.from_config(config)
    total_transitions = 0
    total_sampling_time = 0.0
    total_inference_time = 0.0
    total_update_time = 0.0
    update_id = 0
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
            rollout = runner.collect_training_batch(
                agent,
                episode_indices,
                gamma=float(config["ppo"]["gamma"]),
                gae_lambda=float(config["ppo"]["gae_lambda"]),
                step_limit=step_limit,
                reward_phase=reward_phase,
            )
            update_start = time.perf_counter()
            losses = agent.update(rollout.buffer)
            update_time = time.perf_counter() - update_start
            update_id += 1
            transition_count = rollout.transition_count
            total_transitions += transition_count
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
                validation_row = _validation_log_row(
                    validation,
                    completed_episodes=completed_episodes,
                )
                score = evaluation_selection_key(validation)
                stability = stability_controller.observe_greedy(
                    score,
                    validation["completion_rate"],
                    completed_episodes=completed_episodes,
                    feasibility_phase=reward_phase == "feasibility",
                )
                if stability_controller.should_run_sampled(
                    final_validation=completed_episodes == episodes,
                    completed_episodes=completed_episodes,
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
                    _attach_sampled_validation(
                        validation_row,
                        sampled_validation,
                        completed_episodes=completed_episodes,
                    )
                validation_event = phase_controller.observe_validation(
                    validation["completion_rate"],
                    completed_episodes=completed_episodes,
                    score=score,
                )
                validation_row["candidate_phase"] = reward_phase
                validation_row["validation_event"] = validation_event
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
                validation_rows.append(validation_row)
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
                if validation_event == "transition":
                    transition_metadata = {
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
                    agent.save(
                        accepted_checkpoint,
                        metadata=transition_metadata,
                    )
                    update_row["candidate_status"] = "phase_transition"
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = "phase_transition"
                elif validation_event == "accepted":
                    agent.save(
                        accepted_checkpoint,
                        metadata={
                            "seed": config["seed"],
                            "parallel_envs": parallel_envs,
                            "accepted_episode": completed_episodes,
                            "validation": validation_row,
                        },
                    )
                    update_row["candidate_status"] = "accepted"
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = "accepted"
                elif validation_event == "rejected":
                    if not accepted_checkpoint.exists():
                        raise RuntimeError(
                            "quality candidate rejected before an accepted "
                            "checkpoint was established"
                        )
                    agent.load(accepted_checkpoint, load_optimizer=True)
                    update_row["candidate_status"] = (
                        "rejected_rolled_back"
                    )
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = (
                            "rejected_rolled_back"
                        )
                elif bool(stability["rollback"]):
                    if not best_feasibility_checkpoint.exists():
                        raise RuntimeError(
                            "feasibility rollback requested before a best "
                            "checkpoint was established"
                        )
                    agent.load(
                        best_feasibility_checkpoint,
                        load_optimizer=True,
                    )
                    update_row["candidate_status"] = (
                        "feasibility_rolled_back"
                    )
                    for row in rows[-len(rollout.episodes) :]:
                        row["candidate_status"] = (
                            "feasibility_rolled_back"
                        )
                agent.set_learning_rate(
                    stability_controller.current_learning_rate
                )
                if validation_event == "transition":
                    stability_controller.reset_plateau()
                checkpoint_eligible = (
                    not phase_controller.enabled
                    or validation_event in {"transition", "accepted"}
                )
                is_new_best = checkpoint_eligible and (
                    best_score is None or score < best_score
                )
                if is_new_best:
                    best_score = score
                    best_validation = validation_row
                    agent.save(
                        best_checkpoint,
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
                            "best_episode": completed_episodes,
                            "validation": validation_row,
                        },
                    )
                dashboard.log_validation(
                    validation_row,
                    best_validation=best_validation,
                    phase_state=phase_controller.as_dict(),
                )
                if validation_event in {
                    "transition",
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
    formal_eligible = (
        not phase_controller.enabled
        or phase_controller.phase_transition_episode is not None
    )
    if (
        not formal_eligible
        and phase_controller.formal_training_status
        != "feasibility_not_reached"
    ):
        raise RuntimeError("invalid hierarchical training state")
    if formal_eligible and (best_validation is None or best_score is None):
        raise RuntimeError("training completed without validation")
    final_metadata = {
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
            "formal_eligible": formal_eligible,
        }
    checkpoint: Path | None
    last_candidate_checkpoint: Path | None
    if formal_eligible:
        checkpoint = run_directory / "checkpoint.pt"
        last_candidate_checkpoint = None
        agent.save(checkpoint, metadata=final_metadata)
    else:
        checkpoint = None
        last_candidate_checkpoint = (
            run_directory / "last_candidate_checkpoint.pt"
        )
        agent.save(last_candidate_checkpoint, metadata=final_metadata)
    write_csv(run_directory / "train_log.csv", rows)
    write_csv(run_directory / "update_log.csv", update_rows)
    write_csv(run_directory / "validation_log.csv", validation_rows)
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
            "validation_stability": stability_controller.as_dict(),
            "validation_runs": len(validation_rows),
            "sampled_validation_runs": (
                stability_controller.sampled_validation_runs
            ),
            "last_sampled_validation": last_sampled_validation,
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
            "E1", "E2", "E3", "R11", "S11", "L11", "Q11",
            "e1", "e2", "e3", "r11", "s11", "l11", "q11",
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
    run_directory = train(
        load_config(args.config),
        smoke=args.smoke,
        run_name=args.run_name,
        online_instances=args.online_instances,
        algorithm_seed=args.algorithm_seed,
        parallel_envs=args.parallel_envs,
        visdom_enabled=args.visdom_enabled,
        ablation_variant=args.ablation,
    )
    print(f"training artifacts: {run_directory}")


if __name__ == "__main__":
    main()
