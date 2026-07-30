from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

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
from result import create_run_directory, evaluation_selection_key
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
    phase: str = "legacy"
    consecutive_successes: int = 0
    phase_transition_episode: int | None = None
    accepted_quality_updates: int = 0
    rejected_quality_updates: int = 0

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
            phase="feasibility",
        )

    def should_validate(self, regular_due: bool) -> bool:
        return self.phase == "quality" or regular_due

    def observe_validation(
        self,
        completion_rate: float,
        *,
        completed_episodes: int,
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
                return "transition"
            return "feasibility"
        if rate >= self.quality_completion_floor:
            self.accepted_quality_updates += 1
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
            "phase_transition_episode": self.phase_transition_episode,
            "accepted_quality_updates": self.accepted_quality_updates,
            "rejected_quality_updates": self.rejected_quality_updates,
            "formal_training_status": self.formal_training_status,
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
                "machine_waiting_for_worker_time",
                "completed_reconfigurations",
                "worker_switch_ratio",
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


def train(
    config: dict,
    *,
    smoke: bool = False,
    run_name: str | None = None,
    online_instances: bool | None = None,
    algorithm_seed: int | None = None,
    parallel_envs: int | None = None,
    visdom_enabled: bool | None = None,
) -> Path:
    config = deepcopy(config)
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
    phase1_checkpoint = run_directory / "phase1_checkpoint.pt"
    accepted_checkpoint = run_directory / "accepted_checkpoint.pt"
    best_validation: dict | None = None
    best_score: tuple[float, float, float, float] | None = None
    phase_controller = TrainingPhaseController.from_config(config)
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
            "expected_reward": expected_reward,
            "reward_identity_error": reward_sum - expected_reward,
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
            _, _, _, validation = evaluate_dataset(
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
            validation_event = phase_controller.observe_validation(
                validation["completion_rate"],
                completed_episodes=completed_episodes,
            )
            validation_row["candidate_phase"] = reward_phase
            validation_row["validation_event"] = validation_event
            validation_row["phase_after_validation"] = (
                phase_controller.phase
            )
            validation_row["consecutive_completion_successes"] = (
                phase_controller.consecutive_successes
            )
            validation_rows.append(validation_row)
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
            score = evaluation_selection_key(validation)
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
            "phase1_checkpoint": (
                str(phase1_checkpoint)
                if phase1_checkpoint.exists()
                else None
            ),
            "formal_training_status": (
                phase_controller.formal_training_status
            ),
            "training_phase": phase_controller.as_dict(),
            "validation_runs": len(validation_rows),
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
    phase1_checkpoint = run_directory / "phase1_checkpoint.pt"
    accepted_checkpoint = run_directory / "accepted_checkpoint.pt"
    best_validation: dict | None = None
    best_score: tuple[float, float, float, float] | None = None
    phase_controller = TrainingPhaseController.from_config(config)
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
                    "expected_reward": episode.expected_reward,
                    "reward_identity_error": (
                        episode.reward_sum - episode.expected_reward
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
                    _, validation = evaluate_dataset_parallel(
                        config,
                        dataset_name=validation_split,
                        ppo_agent=agent,
                        runner=runner,
                        instance_limit=validation_limit,
                    )
                else:
                    _, _, _, validation = evaluate_dataset(
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
                validation_event = phase_controller.observe_validation(
                    validation["completion_rate"],
                    completed_episodes=completed_episodes,
                )
                validation_row["candidate_phase"] = reward_phase
                validation_row["validation_event"] = validation_event
                validation_row["phase_after_validation"] = (
                    phase_controller.phase
                )
                validation_row["consecutive_completion_successes"] = (
                    phase_controller.consecutive_successes
                )
                validation_rows.append(validation_row)
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
                score = evaluation_selection_key(validation)
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
            "phase1_checkpoint": (
                str(phase1_checkpoint)
                if phase1_checkpoint.exists()
                else None
            ),
            "formal_training_status": (
                phase_controller.formal_training_status
            ),
            "training_phase": phase_controller.as_dict(),
            "validation_runs": len(validation_rows),
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
    )
    print(f"training artifacts: {run_directory}")


if __name__ == "__main__":
    main()
