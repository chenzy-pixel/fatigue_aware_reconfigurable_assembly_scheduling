from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.parallel import ParallelEpisodeRunner
from configs import load_config, project_path
from configs.config import public_config
from data import load_dataset_split
from data.dataset import validate_algorithm_seed
from data.models import load_instance_yaml
from eval import evaluate_dataset_parallel, evaluate_preference_grid_parallel
from environment import PreferenceContext, simplex_lattice
from result import (
    EVALUATION_SCHEMA_VERSION,
    aggregate_evaluation_rows,
    build_provenance,
    create_run_directory,
    dataset_manifest_snapshot,
    evaluation_quality_metric,
    report_training_failure,
    TrainingConsoleReporter,
)
from result.io import write_config, write_csv, write_json
from result.terminal_log import capture_terminal_output
from result.visdom_dashboard import (
    create_training_dashboard,
    override_visdom_enabled,
)
from training import LexicographicCheckpointSelector
from utils import (
    SAMPLED_EVALUATION_RNG_VERSION,
    configured_formal_evaluation_sampling_seeds,
    set_seed,
)


SINGLE_OBJECTIVE_WEIGHTS = {
    (1.0, 0.0, 0.0): "flow",
    (0.0, 1.0, 0.0): "cost",
    (0.0, 0.0, 1.0): "variance",
}


def _objective_name(config: dict) -> str | None:
    quality = config.get("preference", {}).get("quality", {})
    if not isinstance(quality, dict) or str(quality.get("mode")) != "fixed":
        return None
    fixed = quality.get("fixed")
    if isinstance(fixed, dict):
        weights = tuple(float(fixed.get(name, 0.0)) for name in ("flow", "cost", "variance"))
    elif isinstance(fixed, (list, tuple)) and len(fixed) == 3:
        weights = tuple(float(value) for value in fixed)
    else:
        return None
    return SINGLE_OBJECTIVE_WEIGHTS.get(weights)


def _is_universal(config: dict) -> bool:
    quality = config.get("preference", {}).get("quality", {})
    return isinstance(quality, dict) and str(quality.get("mode")) == "universal_sobol_v1"


def _rows_are_physically_safe(rows: list[dict], tolerance: float = 1e-9) -> bool:
    return bool(rows) and all(
        int(row.get("schedule_violation_count", 0)) == 0
        and float(row.get("maximum_worker_fatigue", math.inf))
        <= float(row.get("safe_fatigue_limit", -math.inf)) + tolerance
        for row in rows
    )


def _validation_manifest_path(config: dict, split: str) -> Path:
    return project_path(config["paths"]["manifests_root"]) / split / "manifest.json"


def _checkpoint_metadata(
    config: dict,
    *,
    role: str,
    episode: int,
    validation_split: str,
    validation_instance_limit: int,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    formal = config["training"]["formal_evaluation"]
    manifest_path = _validation_manifest_path(config, validation_split)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    files = list(manifest["files"][: int(validation_instance_limit)])
    if _is_universal(config):
        preferences = [point.as_dict() for point in simplex_lattice(10, include=())]
    else:
        fixed = config["preference"]["quality"]["fixed"]
        preferences = [PreferenceContext.from_input(fixed).preference.as_dict()]
    return {
        "checkpoint_role": role,
        "checkpoint_episode": int(episode),
        "objective_name": _objective_name(config),
        "experiment_suite_version": config["experiment_suite_version"],
        "runtime_manifest": dict(config["runtime_manifest"]),
        "effective_config": public_config(config),
        "algorithm_seed": int(config["seed"]),
        "result_schema_version": EVALUATION_SCHEMA_VERSION,
        "selection_decode_mode": "sampled",
        "selection_temperature": float(formal["temperature"]),
        "validation_split": validation_split,
        "validation_instance_limit": int(validation_instance_limit),
        "validation_dataset_manifest": dataset_manifest_snapshot(manifest_path),
        "validation_instance_order": [str(item["path"]) for item in files],
        "validation_instance_seeds": [int(item["seed"]) for item in files],
        "validation_repeat_count": int(formal["validation_repeats"]),
        "fixed_preference_set": preferences,
        "preference_count": len(preferences),
        "validation_sampling_seeds": configured_formal_evaluation_sampling_seeds(
            config, "validation"
        ),
        "sampling_rng_version": SAMPLED_EVALUATION_RNG_VERSION,
        "derived_sampling_seed_rule": (
            "sha256(rng_version, sampling_seed, instance_id, preference_key)[:8]"
        ),
        "final_test_sampling_seeds": configured_formal_evaluation_sampling_seeds(
            config, "final_test"
        ),
        "validation": validation,
    }


def _aggregate_formal_rows(
    config: dict,
    *,
    rows: list[dict[str, Any]],
    dataset_name: str,
    manifest: str,
    unique_instance_count: int,
    repeat_count: int,
    universal: bool,
) -> dict[str, Any]:
    aggregate = aggregate_evaluation_rows(
        rows,
        dataset=dataset_name,
        policy="ppo",
        manifest=manifest,
        quality_metric=evaluation_quality_metric(config),
    )
    aggregate.update(
        {
            "decode_mode": "sampled",
            "result_role": "formal_sampled",
            "repeat_count": int(repeat_count),
            "unique_instance_count": int(unique_instance_count),
        }
    )
    if universal:
        keys = sorted(
            PreferenceContext.from_input(point).key
            for point in simplex_lattice(10, include=())
        )
        observed_keys = {str(row["preference_key"]) for row in rows}
        if not observed_keys.issubset(keys):
            raise ValueError("evaluation rows contain an unknown preference key")
        denominator = unique_instance_count * repeat_count
        completion_by_preference = {
            key: sum(
                bool(row["terminated"]) and not bool(row["truncated"])
                for row in rows
                if str(row["preference_key"]) == key
            )
            / denominator
            for key in keys
        }
        aggregate["cell_count"] = int(aggregate["instance_count"])
        aggregate["completed_cell_count"] = int(aggregate["completed_count"])
        aggregate["instance_count"] = int(unique_instance_count)
        aggregate["preference_count"] = len(keys)
        aggregate["completion_rate_by_preference"] = completion_by_preference
        quality_by_preference = {
            key: float(
                aggregate["preference_quality_by_key"].get(key, math.inf)
            )
            for key in keys
        }
        aggregate["preference_quality_by_key"] = quality_by_preference
        aggregate["preference_balanced_quality_score"] = (
            sum(quality_by_preference.values()) / len(keys)
            if all(math.isfinite(value) for value in quality_by_preference.values())
            else math.inf
        )
        aggregate["minimum_preference_completion_rate"] = min(
            completion_by_preference.values()
        )
        aggregate["completion_rate"] = aggregate[
            "minimum_preference_completion_rate"
        ]
    return aggregate


def _evaluate_policy(
    config: dict,
    *,
    dataset_name: str,
    ppo_agent: PPOAgent,
    runner: ParallelEpisodeRunner,
    instance_limit: int,
    decode_mode: str,
    sampling_seeds: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    universal = _is_universal(config)
    seeds: list[int | None] = (
        [None] if decode_mode == "greedy" else list(sampling_seeds or [])
    )
    if not seeds:
        raise ValueError("sampled evaluation requires at least one seed")
    rows: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    for repeat_index, seed in enumerate(seeds):
        if universal:
            current_rows, current = evaluate_preference_grid_parallel(
                config,
                dataset_name=dataset_name,
                ppo_agent=ppo_agent,
                runner=runner,
                instance_limit=instance_limit,
                decode_mode=decode_mode,
                sampling_seed=seed,
            )
        else:
            current_rows, current = evaluate_dataset_parallel(
                config,
                dataset_name=dataset_name,
                ppo_agent=ppo_agent,
                runner=runner,
                instance_limit=instance_limit,
                decode_mode=decode_mode,
                sampling_seed=seed,
            )
        for row in current_rows:
            row["sampling_repeat"] = repeat_index
        rows.extend(current_rows)
        reference = current
    if reference is None:
        raise RuntimeError("evaluation produced no aggregate")
    if decode_mode == "sampled":
        aggregate = _aggregate_formal_rows(
            config,
            rows=rows,
            dataset_name=dataset_name,
            manifest=str(reference["manifest"]),
            unique_instance_count=instance_limit,
            repeat_count=len(seeds),
            universal=universal,
        )
        aggregate["sampling_seeds"] = [int(seed) for seed in seeds if seed is not None]
    else:
        aggregate = reference
        aggregate["repeat_count"] = 1
        aggregate["unique_instance_count"] = instance_limit
    aggregate["physical_safety_pass"] = _rows_are_physically_safe(rows)
    return rows, aggregate


def _summary_value(summary: dict, name: str) -> float | None:
    value = summary.get(name, {}).get("mean")
    return None if value is None else float(value)


def _validation_log_row(aggregate: dict, *, episode: int) -> dict[str, Any]:
    completed = aggregate["completed_metrics"]
    all_metrics = aggregate["all_instance_metrics"]
    return {
        "episode": int(episode),
        "instance_count": int(aggregate["instance_count"]),
        "cell_count": aggregate.get("cell_count", aggregate["instance_count"]),
        "repeat_count": int(aggregate.get("repeat_count", 1)),
        "completion_rate": float(aggregate["completion_rate"]),
        "truncated_count": int(aggregate["truncated_count"]),
        "schedule_violation_count": int(aggregate["schedule_violation_count"]),
        "physical_safety_pass": bool(aggregate["physical_safety_pass"]),
        "preference_balanced_quality_score": float(
            aggregate["preference_balanced_quality_score"]
        ),
        "mean_quality_score": _summary_value(completed, "quality_score"),
        "mean_preference_quality_score": _summary_value(
            completed, "preference_quality_score"
        ),
        "mean_flow_time_objective": _summary_value(
            completed, "flow_time_objective"
        ),
        "mean_reconfiguration_cost": _summary_value(
            completed, "reconfiguration_cost"
        ),
        "mean_worker_load_variance": _summary_value(
            completed, "worker_load_variance"
        ),
        "mean_operation_progress": _summary_value(
            all_metrics, "operation_progress"
        ),
        "mean_single_stage_proxy_return": _summary_value(
            all_metrics, "single_stage_proxy_return"
        ),
        "mean_unfinished_orders": _summary_value(
            all_metrics, "unfinished_orders"
        ),
    }


def _attach_greedy_diagnostic(
    row: dict[str, Any], greedy: dict[str, Any]
) -> None:
    completed = greedy["completed_metrics"]
    all_metrics = greedy["all_instance_metrics"]
    row.update(
        {
            "greedy_completion_rate": float(greedy["completion_rate"]),
            "greedy_truncated_count": int(greedy["truncated_count"]),
            "greedy_schedule_violation_count": int(
                greedy["schedule_violation_count"]
            ),
            "greedy_physical_safety_pass": bool(
                greedy["physical_safety_pass"]
            ),
            "greedy_preference_balanced_quality_score": float(
                greedy["preference_balanced_quality_score"]
            ),
            "greedy_mean_flow_time_objective": _summary_value(
                completed, "flow_time_objective"
            ),
            "greedy_mean_reconfiguration_cost": _summary_value(
                completed, "reconfiguration_cost"
            ),
            "greedy_mean_worker_load_variance": _summary_value(
                completed, "worker_load_variance"
            ),
            "greedy_mean_operation_progress": _summary_value(
                all_metrics, "operation_progress"
            ),
            "greedy_mean_single_stage_proxy_return": _summary_value(
                all_metrics, "single_stage_proxy_return"
            ),
        }
    )


@dataclass
class LearningRatePlateauController:
    learning_rate: float
    minimum: float
    factor: float
    patience: int
    stale_validations: int = 0
    decay_count: int = 0

    @classmethod
    def from_config(cls, config: dict) -> "LearningRatePlateauController":
        raw = config["training"]["validation_control"]["learning_rate_plateau"]
        return cls(
            learning_rate=float(config["ppo"]["learning_rate"]),
            minimum=float(raw["minimum"]),
            factor=float(raw["factor"]),
            patience=int(raw["patience_validations"]),
        )

    def observe(self, improved: bool) -> bool:
        self.stale_validations = 0 if improved else self.stale_validations + 1
        if improved or self.stale_validations < self.patience:
            return False
        next_rate = max(self.minimum, self.learning_rate * self.factor)
        self.stale_validations = 0
        if math.isclose(next_rate, self.learning_rate, rel_tol=0.0, abs_tol=0.0):
            return False
        self.learning_rate = next_rate
        self.decay_count += 1
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "minimum_learning_rate": self.minimum,
            "plateau_factor": self.factor,
            "plateau_patience_validations": self.patience,
            "stale_validations": self.stale_validations,
            "learning_rate_decay_count": self.decay_count,
        }


def _episode_log_row(episode) -> dict[str, Any]:
    metrics = episode.metrics
    total_orders = max(1, int(metrics["total_orders"]))
    total_operations = max(1, int(metrics["total_operations"]))
    row = {
        "episode": int(episode.episode_index) + 1,
        "instance_id": episode.instance_id,
        "reward": float(episode.reward_sum),
        "expected_reward": float(episode.expected_reward),
        "reward_identity_error": float(
            episode.base_reward_sum - episode.expected_reward
        ),
        "terminated": bool(metrics["terminated"]),
        "truncated": bool(metrics["truncated"]),
        "task_succeeded": bool(metrics.get("task_succeeded", False)),
        "task_failed": bool(metrics.get("task_failed", False)),
        "terminal_reason": metrics["terminal_reason"],
        "completed_order_ratio": float(metrics["completed_orders"]) / total_orders,
        "completed_operation_ratio": float(metrics["completed_operations"])
        / total_operations,
        "initial_progress": float(metrics["initial_progress"]),
        "operation_progress": float(metrics["operation_progress"]),
        "initial_preference_quality_score": float(
            metrics["initial_preference_quality_score"]
        ),
        "preference_quality_score": float(metrics["preference_quality_score"]),
        "flow_time_objective": float(metrics["flow_time_objective"]),
        "reconfiguration_cost": float(metrics["reconfiguration_cost"]),
        "worker_load_variance": float(metrics["worker_load_variance"]),
        "preference_key": metrics.get("preference_key"),
        "step_count": int(episode.step_count),
        "policy_step_count": int(episode.policy_step_count),
        "forced_action_count": int(episode.forced_action_count),
        "forced_action_ratio": float(episode.forced_action_ratio),
    }
    row.update(
        {
            f"reward_{name}": float(value)
            for name, value in episode.reward_components.items()
        }
    )
    return row


def _failure_progress_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(
        float(row["operation_progress"])
        for row in rows
        if bool(row.get("task_failed", row.get("truncated", False)))
    )
    if not values:
        return {"count": 0, "mean": None, "median": None, "bins": {}}
    midpoint = len(values) // 2
    median = (
        values[midpoint]
        if len(values) % 2
        else 0.5 * (values[midpoint - 1] + values[midpoint])
    )
    edges = (0.25, 0.50, 0.75, 0.90, 0.99, 1.0)
    bins: dict[str, int] = {}
    lower = 0.0
    for upper in edges:
        label = f"[{lower:.2f},{upper:.2f}{']' if upper == 1.0 else ')'}"
        bins[label] = sum(
            lower <= value <= upper if upper == 1.0 else lower <= value < upper
            for value in values
        )
        lower = upper
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": median,
        "minimum": values[0],
        "maximum": values[-1],
        "bins": bins,
    }


class TrainingEngine:
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
        seed = validate_algorithm_seed(
            config,
            int(config["seed"])
            if self.algorithm_seed is None
            else int(self.algorithm_seed),
        )
        config["seed"] = seed
        set_seed(seed)
        if not math.isclose(float(config["ppo"]["gamma"]), 1.0):
            raise ValueError("single-stage training requires ppo.gamma = 1.0")
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
        config["training"][
            "smoke_parallel_envs" if self.smoke else "parallel_envs"
        ] = workers
        if self.initial_checkpoint is not None:
            config["training"]["initial_checkpoint"] = str(
                Path(self.initial_checkpoint).resolve()
            )
        return _train_single_stage(
            config,
            smoke=self.smoke,
            run_name=self.run_name,
            episodes=episodes,
            parallel_envs=min(workers, episodes),
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


def _train_single_stage(
    config: dict,
    *,
    smoke: bool,
    run_name: str | None,
    episodes: int,
    parallel_envs: int,
    initial_checkpoint: str | Path | None,
) -> Path:
    started_at = time.perf_counter()
    template = load_instance_yaml(project_path(config["paths"]["fixed_instance"]))
    from environment import AssemblySchedulingEnv

    bootstrap_environment = AssemblySchedulingEnv(config)
    bootstrap_observation = bootstrap_environment.reset(template)
    network = build_actor_critic(bootstrap_observation, config["network"])
    agent = PPOAgent(network, config["ppo"], device=config["device"])
    if initial_checkpoint is not None:
        agent.load(initial_checkpoint, load_optimizer=False)

    run_directory = create_run_directory(
        project_path(config["paths"]["result_root"]),
        label="single_stage_smoke" if smoke else "single_stage_train",
        run_name=run_name,
    )
    write_config(run_directory, config)
    dashboard = create_training_dashboard(
        config=config,
        run_directory=run_directory,
        total_episodes=episodes,
    )
    reporter = TrainingConsoleReporter(
        config=config,
        total_episodes=episodes,
        parallel_envs=parallel_envs,
        validation_instance_limit=int(
            config["training"]["smoke_validation_instance_limit"]
            if smoke
            else config["training"]["validation_instance_limit"]
        ),
        objective_name=_objective_name(config),
        started_at=started_at,
    )
    reporter.start_run()
    selector = LexicographicCheckpointSelector.from_config(config)
    plateau = LearningRatePlateauController.from_config(config)
    validation_split = str(config["training"]["validation_split"])
    validation_limit = int(
        config["training"]["smoke_validation_instance_limit"]
        if smoke
        else config["training"]["validation_instance_limit"]
    )
    validation_interval = int(config["training"]["validation_interval_episodes"])
    validation_seeds = configured_formal_evaluation_sampling_seeds(
        config, "validation"
    )
    step_limit = int(config["training"]["smoke_rollout_steps"]) if smoke else None
    validation_workers = int(config["training"]["validation_parallel_envs"])
    worker_count = max(parallel_envs, validation_workers)

    best_checkpoint = run_directory / "best_checkpoint.pt"
    last_checkpoint = run_directory / "last_checkpoint.pt"
    episode_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    sampled_validation_rows: list[dict[str, Any]] = []
    greedy_validation_rows: list[dict[str, Any]] = []
    best_checkpoint_metadata: dict[str, Any] | None = None
    best_validation_row: dict[str, Any] | None = None

    with ParallelEpisodeRunner(
        config=config,
        template=template,
        episode_count=episodes,
        worker_count=worker_count,
    ) as runner:
        for update_id, batch_start in enumerate(
            range(0, episodes, parallel_envs), start=1
        ):
            indices = list(range(batch_start, min(batch_start + parallel_envs, episodes)))
            rollout = runner.collect_training_batch(
                agent,
                indices,
                gamma=float(config["ppo"]["gamma"]),
                gae_lambda=float(config["ppo"]["gae_lambda"]),
                step_limit=step_limit,
            )
            if rollout.transition_count == 0:
                raise RuntimeError("training batch contains no policy transitions")
            losses = agent.update(rollout.buffer)
            batch_rows = [_episode_log_row(episode) for episode in rollout.episodes]
            episode_rows.extend(batch_rows)
            completed_episodes = indices[-1] + 1
            update_row = {
                "update_id": update_id,
                "episode_start": indices[0],
                "episode_end": indices[-1],
                "episode_count": len(indices),
                "transition_count": rollout.transition_count,
                "environment_step_count": rollout.environment_step_count,
                "forced_action_count": rollout.forced_action_count,
                "forced_action_ratio": rollout.forced_action_ratio,
                "learning_rate": plateau.learning_rate,
                **losses,
            }
            update_rows.append(update_row)
            dashboard.log_update(update_row, batch_rows, selector.as_dict())
            reporter.training_update(update_row, batch_rows)

            validation_due = (
                completed_episodes % validation_interval == 0
                or completed_episodes == episodes
            )
            if not validation_due:
                continue
            formal_rows, formal = _evaluate_policy(
                config,
                dataset_name=validation_split,
                ppo_agent=agent,
                runner=runner,
                instance_limit=validation_limit,
                decode_mode="sampled",
                sampling_seeds=validation_seeds,
            )
            greedy_rows, greedy = _evaluate_policy(
                config,
                dataset_name=validation_split,
                ppo_agent=agent,
                runner=runner,
                instance_limit=validation_limit,
                decode_mode="greedy",
            )
            sampled_validation_rows.extend(
                {"validation_episode": completed_episodes, **row}
                for row in formal_rows
            )
            greedy_validation_rows.extend(
                {"validation_episode": completed_episodes, **row}
                for row in greedy_rows
            )
            validation_row = _validation_log_row(
                formal, episode=completed_episodes
            )
            _attach_greedy_diagnostic(validation_row, greedy)
            event = selector.observe(
                formal,
                completed_episodes=completed_episodes,
                physical_safety_pass=bool(formal["physical_safety_pass"]),
            )
            validation_row.update(selector.last_decision)
            improved = event in {"best_initialized", "best_improved"}
            if improved:
                best_validation_row = dict(validation_row)
                best_checkpoint_metadata = _checkpoint_metadata(
                    config,
                    role="best",
                    episode=completed_episodes,
                    validation_split=validation_split,
                    validation_instance_limit=validation_limit,
                    validation=best_validation_row,
                )
                agent.save(
                    best_checkpoint,
                    metadata=best_checkpoint_metadata,
                )
            if plateau.observe(improved):
                agent.set_learning_rate(plateau.learning_rate)
            validation_row.update(plateau.as_dict())
            validation_rows.append(validation_row)
            dashboard.log_validation(
                validation_row,
                best_validation=best_validation_row,
                phase_state=selector.as_dict(),
            )
            reporter.validation(
                validation_row,
                selector_state=selector.as_dict(),
            )

        agent.save(
            last_checkpoint,
            metadata=_checkpoint_metadata(
                config,
                role="last",
                episode=episodes,
                validation_split=validation_split,
                validation_instance_limit=validation_limit,
                validation=(validation_rows[-1] if validation_rows else None),
            ),
        )

        final_sampled: dict[str, Any] | None = None
        final_greedy: dict[str, Any] | None = None
        final_sampled_rows: list[dict[str, Any]] = []
        final_greedy_rows: list[dict[str, Any]] = []
        if selector.has_best:
            agent.load(best_checkpoint, load_optimizer=False)
            final_split = validation_split if smoke else "test"
            final_limit = (
                validation_limit
                if smoke
                else len(load_dataset_split(config, final_split))
            )
            final_sampled_rows, final_sampled = _evaluate_policy(
                config,
                dataset_name=final_split,
                ppo_agent=agent,
                runner=runner,
                instance_limit=final_limit,
                decode_mode="sampled",
                sampling_seeds=configured_formal_evaluation_sampling_seeds(
                    config, "final_test"
                ),
            )
            final_greedy_rows, final_greedy = _evaluate_policy(
                config,
                dataset_name=final_split,
                ppo_agent=agent,
                runner=runner,
                instance_limit=final_limit,
                decode_mode="greedy",
            )

    write_csv(run_directory / "train_log.csv", episode_rows)
    write_csv(run_directory / "update_log.csv", update_rows)
    write_csv(run_directory / "validation_log.csv", validation_rows)
    write_csv(
        run_directory / "sampled_validation_instance_metrics.csv",
        sampled_validation_rows,
    )
    write_csv(
        run_directory / "greedy_validation_instance_metrics.csv",
        greedy_validation_rows,
    )
    if final_sampled_rows:
        write_csv(
            run_directory / "final_sampled_instance_metrics.csv",
            final_sampled_rows,
        )
    if final_greedy_rows:
        write_csv(
            run_directory / "final_greedy_instance_metrics.csv",
            final_greedy_rows,
        )
    checkpoint = best_checkpoint if selector.has_best else None
    provenance = build_provenance(
        config,
        dataset_manifest_path=_validation_manifest_path(
            config, validation_split
        ),
        checkpoint_path=checkpoint,
        checkpoint_metadata=best_checkpoint_metadata,
    )
    summary = {
        "experiment_name": config["experiment_name"],
        "seed": int(config["seed"]),
        "episodes": episodes,
        "objective_name": _objective_name(config),
        "reward_mode": config["reward"]["mode"],
        "checkpoint_selection": selector.as_dict(),
        "learning_rate_control": plateau.as_dict(),
        "best_checkpoint": str(best_checkpoint) if selector.has_best else None,
        "last_checkpoint": str(last_checkpoint),
        "final_sampled": final_sampled,
        "final_greedy": final_greedy,
        "final_sampled_failure_progress": _failure_progress_summary(
            final_sampled_rows
        ),
        "final_greedy_failure_progress": _failure_progress_summary(
            final_greedy_rows
        ),
        "elapsed_seconds": time.perf_counter() - started_at,
        "provenance": provenance,
    }
    write_json(run_directory / "summary.json", summary)
    dashboard.close()
    reporter.done(
        selector_state=selector.as_dict(),
        run_directory=run_directory,
        best_checkpoint=(best_checkpoint if selector.has_best else None),
        last_checkpoint=last_checkpoint,
        elapsed_seconds=float(summary["elapsed_seconds"]),
    )
    return run_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the single-stage PPO policy")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--run-name")
    parser.add_argument("--algorithm-seed", type=int)
    parser.add_argument("--parallel-envs", type=int)
    parser.add_argument(
        "--visdom-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--online-instances",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.episodes is not None:
        if args.episodes <= 0:
            parser.error("--episodes must be positive")
        config["training"]["episodes"] = int(args.episodes)
    result_root = project_path(config["paths"]["result_root"])
    result_root.mkdir(parents=True, exist_ok=True)
    run_key = Path(args.run_name).name if args.run_name else "unnamed_train"
    staging_log = result_root / f".{run_key}.{os.getpid()}.terminal.log.tmp"
    expected_run = result_root / args.run_name if args.run_name else None
    run_directory: Path | None = None
    exit_code = 0
    try:
        with capture_terminal_output(staging_log):
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
            except KeyboardInterrupt as error:
                exit_code = 130
                report_training_failure(
                    error,
                    run_directory=(run_directory or expected_run),
                    exit_code=exit_code,
                )
                traceback.print_exc()
            except Exception as error:
                exit_code = 1
                failure_directory = (
                    expected_run
                    if expected_run is not None and expected_run.is_dir()
                    else run_directory
                )
                report_training_failure(
                    error,
                    run_directory=failure_directory,
                    exit_code=exit_code,
                )
                traceback.print_exc()
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
                        [
                            {
                                "exception_type": type(error).__name__,
                                "message": str(error),
                            }
                        ],
                    )
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
