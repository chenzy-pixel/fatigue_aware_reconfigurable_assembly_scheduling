from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from functools import wraps
from typing import Any, Sequence

import torch

from agent.ppo import (
    PPOAgent,
    build_actor_critic,
    summarize_policy_decision_diagnostics,
)
from agent.ppo.parallel import ParallelEpisodeRunner
from agent.baselines import HeuristicPolicy, RandomPolicy
from configs import load_config, project_path
from configs.formal_preferences import formal_preferences
from data import (
    AssemblyInstance,
    load_dataset_split,
    load_instance_pickle,
    load_instance_yaml,
    save_instance_pickle,
)
from data.dataset import PERSISTED_SPLITS, validate_algorithm_seed
from data.selection import resolve_instance_indices, subset_snapshot, select_validation_subsets
from result.metrics import successful_row
from environment import (
    AssemblySchedulingEnv,
    PreferenceContext,
    PreferenceContextInput,
    bounded_quality_score,
    proxy_return_from_metrics,
    terminal_quality_score,
)
from result import (
    aggregate_evaluation_rows,
    build_provenance,
    create_run_directory,
    dataset_manifest_snapshot,
    evaluation_quality_metric,
    quality_metric_sha256,
    relative_gap_percent,
)
from result.io import write_config, write_csv, write_json
from result.metrics import EVALUATION_SCHEMA_VERSION
from utils import (
    SAMPLED_EVALUATION_RNG_VERSION,
    action_trace_sha256,
    capture_global_rng_state,
    derive_evaluation_sampling_seed,
    configured_formal_evaluation_sampling_seeds,
    restore_global_rng_state,
    set_seed,
)


def _resolve_decode_mode(policy_name: str, decode_mode: str | None) -> str:
    if decode_mode is None:
        return "sampled" if policy_name == "ppo" else "greedy"
    return str(decode_mode)


def _resolve_sampling_seed(
    config: dict[str, Any],
    *,
    decode_mode: str,
    sampling_seed: int | None,
) -> int | None:
    if decode_mode != "sampled" or sampling_seed is not None:
        return sampling_seed
    return configured_formal_evaluation_sampling_seeds(
        config, "final_test"
    )[0]


def _preserve_rng_for_sampled(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        decode_mode = kwargs.get("decode_mode")
        policy_name = kwargs.get("policy_name")
        formal_sampled_default = bool(
            decode_mode is None and policy_name == "ppo"
        )
        if decode_mode != "sampled" and not formal_sampled_default:
            return function(*args, **kwargs)
        state = capture_global_rng_state()
        try:
            return function(*args, **kwargs)
        finally:
            restore_global_rng_state(state)

    return wrapped


def load_configured_instance(config: dict[str, Any]):
    cache = project_path(config["paths"]["instance_cache"])
    source = project_path(config["paths"]["fixed_instance"])
    if cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime:
        try:
            return load_instance_pickle(cache)
        except (OSError, TypeError, ValueError):
            pass
    instance = load_instance_yaml(source)
    save_instance_pickle(instance, cache)
    return instance


class EvaluationPolicy:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        policy_name: str,
        bootstrap_observation: Any,
        checkpoint: str | None = None,
        ppo_agent: PPOAgent | None = None,
        decode_mode: str | None = None,
        sampling_seed: int | None = None,
    ):
        self.policy_name = policy_name
        self.device = torch.device(config["device"])
        self.ppo_agent: PPOAgent | None = None
        self.policy: HeuristicPolicy | RandomPolicy | None = None
        decode_mode = _resolve_decode_mode(policy_name, decode_mode)
        sampling_seed = _resolve_sampling_seed(
            config,
            decode_mode=decode_mode,
            sampling_seed=sampling_seed,
        )
        if decode_mode not in {"greedy", "sampled"}:
            raise ValueError("decode_mode must be 'greedy' or 'sampled'")
        if policy_name != "ppo" and decode_mode != "greedy":
            raise ValueError("sampled decode_mode is only available for PPO")
        self.decode_mode = decode_mode
        self.generator: torch.Generator | None = None
        self.sampling_seed = sampling_seed
        self.derived_sampling_seed: int | None = None
        self.sampling_evaluation_key: str | None = None
        self._episode_actions: list[int] = []
        self._episode_policy_diagnostics: list[dict[str, Any]] = []
        if policy_name == "heuristic":
            if ppo_agent is not None or checkpoint is not None:
                raise ValueError(
                    "heuristic evaluation does not accept a PPO agent"
                )
            self.policy = HeuristicPolicy()
        elif policy_name == "random":
            if ppo_agent is not None or checkpoint is not None:
                raise ValueError(
                    "random evaluation does not accept a PPO agent"
                )
            self.policy = RandomPolicy(int(config["seed"]))
        elif policy_name == "ppo":
            if ppo_agent is not None and checkpoint is not None:
                raise ValueError(
                    "provide either ppo_agent or checkpoint, not both"
                )
            if ppo_agent is None:
                if checkpoint is None:
                    raise ValueError(
                        "--checkpoint is required for PPO evaluation"
                    )
                checkpoint_path = project_path(checkpoint)
                network = build_actor_critic(
                    bootstrap_observation,
                    config["network"],
                )
                ppo_agent = PPOAgent(
                    network,
                    config["ppo"],
                    device=config["device"],
                )
                from configs.runtime import assert_checkpoint_fatigue_mode
                assert_checkpoint_fatigue_mode(ppo_agent.load(checkpoint_path), config)
            self.ppo_agent = ppo_agent
            self.device = ppo_agent.device
            if self.decode_mode == "sampled":
                if sampling_seed is None:
                    raise ValueError(
                        "sampling_seed is required for sampled PPO evaluation"
                    )
        else:
            raise ValueError(f"unknown policy {policy_name}")

    def select_action(
        self,
        observation: Any,
        environment: AssemblySchedulingEnv,
    ) -> int:
        if self.ppo_agent is not None:
            action, _, _ = self.ppo_agent.act(
                observation,
                environment.get_action_mask(),
                deterministic=self.decode_mode == "greedy",
                generator=self.generator,
            )
            for diagnostic in (
                self.ppo_agent.consume_policy_decision_diagnostics()
            ):
                diagnostic["selected_action"] = int(action)
                diagnostic["ranker_top_selected"] = bool(
                    int(action)
                    == int(diagnostic.get("relative_top_action", -1))
                )
                self._episode_policy_diagnostics.append(diagnostic)
            self._episode_actions.append(int(action))
            return action
        if self.policy is None:
            raise RuntimeError("evaluation policy is not initialized")
        action = self.policy.select_action(environment)
        self._episode_actions.append(int(action))
        return action

    def begin_episode(
        self,
        instance_id: str,
        evaluation_key: str | None = None,
    ) -> None:
        self._episode_policy_diagnostics.clear()
        self._episode_actions.clear()
        if self.decode_mode == "sampled":
            if self.sampling_seed is None:
                raise RuntimeError("sampled evaluation has no sampling seed")
            self.sampling_evaluation_key = evaluation_key
            self.derived_sampling_seed = derive_evaluation_sampling_seed(
                self.sampling_seed,
                instance_id,
                evaluation_key,
            )
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.derived_sampling_seed
            )

    def episode_policy_metrics(self) -> dict[str, float | int]:
        return {
            "action_trace_sha256": action_trace_sha256(
                self._episode_actions
            ),
            **summarize_policy_decision_diagnostics(
                self._episode_policy_diagnostics
            ),
        }

    def synchronize(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def enter_evaluation_mode(self) -> bool | None:
        if self.ppo_agent is None:
            return None
        was_training = self.ppo_agent.network.training
        self.ppo_agent.network.eval()
        return was_training

    def restore_mode(self, was_training: bool | None) -> None:
        if self.ppo_agent is not None and was_training is not None:
            self.ppo_agent.network.train(was_training)


@_preserve_rng_for_sampled
def evaluate(
    config: dict[str, Any],
    *,
    policy_name: str,
    checkpoint: str | None = None,
    decode_mode: str | None = None,
    sampling_seed: int | None = None,
) -> tuple[AssemblySchedulingEnv, dict[str, Any]]:
    decode_mode = _resolve_decode_mode(policy_name, decode_mode)
    sampling_seed = _resolve_sampling_seed(
        config,
        decode_mode=decode_mode,
        sampling_seed=sampling_seed,
    )
    set_seed(validate_algorithm_seed(config, int(config["seed"])))
    instance = load_configured_instance(config)
    return evaluate_instance(
        config,
        instance=instance,
        policy_name=policy_name,
        checkpoint=checkpoint,
        decode_mode=decode_mode,
        sampling_seed=sampling_seed,
    )


@_preserve_rng_for_sampled
def evaluate_instance(
    config: dict[str, Any],
    *,
    instance: AssemblyInstance,
    policy_name: str,
    checkpoint: str | None = None,
    prepared_policy: EvaluationPolicy | None = None,
    decode_mode: str | None = None,
    sampling_seed: int | None = None,
    preference: PreferenceContextInput | None = None,
) -> tuple[AssemblySchedulingEnv, dict[str, Any]]:
    if decode_mode is None and prepared_policy is not None:
        decode_mode = prepared_policy.decode_mode
        if sampling_seed is None:
            sampling_seed = prepared_policy.sampling_seed
    decode_mode = _resolve_decode_mode(policy_name, decode_mode)
    sampling_seed = _resolve_sampling_seed(
        config,
        decode_mode=decode_mode,
        sampling_seed=sampling_seed,
    )
    if prepared_policy is None:
        set_seed(validate_algorithm_seed(config, int(config["seed"])))
    runner = prepared_policy
    if runner is None:
        bootstrap_environment = AssemblySchedulingEnv(config)
        bootstrap_observation = bootstrap_environment.reset(instance)
        runner = EvaluationPolicy(
            config,
            policy_name=policy_name,
            bootstrap_observation=bootstrap_observation,
            checkpoint=checkpoint,
            decode_mode=decode_mode,
            sampling_seed=sampling_seed,
        )
    if runner.policy_name != policy_name:
        raise ValueError(
            f"prepared policy is {runner.policy_name}, expected {policy_name}"
        )
    if runner.decode_mode != decode_mode:
        raise ValueError(
            f"prepared decode mode is {runner.decode_mode}, "
            f"expected {decode_mode}"
        )
    solve_start = time.perf_counter()
    env = AssemblySchedulingEnv(config)
    observation = env.reset(instance, preference=preference)
    evaluation_key = (
        None
        if preference is None
        else PreferenceContext.from_input(preference).key
    )
    runner.begin_episode(instance.instance_id, evaluation_key)
    inference_time = 0.0
    decisions = 0
    while not (env.terminated or env.truncated):
        runner.synchronize()
        inference_start = time.perf_counter()
        action = runner.select_action(observation, env)
        runner.synchronize()
        inference_time += time.perf_counter() - inference_start
        observation, _, _, _, _ = env.step(action)
        decisions += 1
    solve_time = time.perf_counter() - solve_start
    metrics = env.metrics()
    metrics["policy"] = policy_name
    metrics["arm"] = policy_name
    metrics["policy_execution_version"] = (
        runner.ppo_agent.network.execution_mode
        if runner.ppo_agent is not None else policy_name
    )
    metrics["decode_mode"] = decode_mode
    metrics["result_role"] = (
        "formal_sampled"
        if policy_name == "ppo" and decode_mode == "sampled"
        else "greedy_diagnostic"
        if policy_name == "ppo"
        else "baseline"
    )
    metrics["sampling_seed"] = runner.sampling_seed
    metrics["derived_sampling_seed"] = runner.derived_sampling_seed
    metrics["sampling_evaluation_key"] = runner.sampling_evaluation_key
    metrics["sampling_rng_version"] = (
        SAMPLED_EVALUATION_RNG_VERSION
        if decode_mode == "sampled"
        else None
    )
    metrics["single_stage_proxy_return"] = proxy_return_from_metrics(
        metrics,
        config,
        preference=metrics.get("preference"),
    )
    metrics["decisions"] = decisions
    metrics["inference_time_seconds"] = inference_time
    metrics["solve_time_seconds"] = solve_time
    metrics["inference_time_per_decision_ms"] = (
        1000.0 * inference_time / decisions if decisions else 0.0
    )
    metrics["schedule_violations"] = env.validate_schedule()
    metrics.update(runner.episode_policy_metrics())
    return env, metrics


def evaluate_representative_diagnostic(
    config: dict[str, Any],
    *,
    dataset_name: str,
    ppo_agent: PPOAgent,
    instance_index: int = 0,
) -> dict[str, Any]:
    """Run one fixed deterministic instance and capture display-only traces."""
    dataset = load_dataset_split(config, dataset_name)
    index = int(instance_index)
    if index < 0 or index >= len(dataset):
        raise ValueError(
            f"instance_index must be in [0, {len(dataset) - 1}]"
        )
    record = dataset[index]
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(record.instance)
    runner = EvaluationPolicy(
        config,
        policy_name="ppo",
        bootstrap_observation=observation,
        ppo_agent=ppo_agent,
        decode_mode="greedy",
    )
    was_training = runner.enter_evaluation_mode()
    worker_ids = [worker.spec.id for worker in environment.workers]
    fatigue_trace: list[dict[str, Any]] = []

    def capture_fatigue() -> None:
        snapshot = {
            "time": float(environment.current_time),
            "workers": {
                worker.spec.id: float(worker.fatigue)
                for worker in environment.workers
            },
        }
        if (
            fatigue_trace
            and fatigue_trace[-1]["time"] == snapshot["time"]
        ):
            fatigue_trace[-1] = snapshot
        else:
            fatigue_trace.append(snapshot)

    capture_fatigue()
    try:
        while not (environment.terminated or environment.truncated):
            action = runner.select_action(observation, environment)
            observation, _, _, _, _ = environment.step(action)
            capture_fatigue()
    finally:
        runner.restore_mode(was_training)
    metrics = environment.metrics()
    metrics["schedule_violations"] = environment.validate_schedule()
    return {
        "instance_id": record.instance.instance_id,
        "dataset": dataset_name,
        "instance_index": index,
        "metrics": metrics,
        "schedule": list(environment.schedule_log),
        "reconfigurations": list(environment.reconfiguration_log),
        "worker_ids": worker_ids,
        "worker_peak_fatigue": metrics["worker_peak_fatigue"],
        "safe_fatigue_limit": float(
            record.instance.fatigue.maximum_safe_fatigue
        ),
        "fatigue_trace": fatigue_trace,
    }


def _evaluation_row(
    record,
    metrics: dict[str, Any],
    config: dict[str, Any],
    quality_metric: dict[str, Any],
) -> dict[str, Any]:
    heuristic = record.metadata.get("heuristic_metrics") or {}
    pressure = record.metadata["pressure_metrics"]
    heuristic_flow_time = heuristic.get("heuristic_flow_time")
    heuristic_makespan = heuristic.get("heuristic_makespan")
    heuristic_cost = heuristic.get(
        "heuristic_reconfiguration_cost"
    )
    heuristic_variance = heuristic.get(
        "worker_workload_variance"
    )
    reference_valid = bool(heuristic.get("heuristic_completed")) and not bool(heuristic.get("heuristic_truncated", True)) and not heuristic.get("schedule_violations", [])
    reference_valid = reference_valid and float(heuristic.get("heuristic_makespan", math.inf)) <= record.instance.horizon + 1e-9
    reference_valid = reference_valid and float(heuristic.get("maximum_worker_fatigue", math.inf)) <= record.instance.fatigue.maximum_safe_fatigue + 1e-9
    comparison_valid = (reference_valid and bool(metrics["terminated"]) and not bool(metrics["truncated"])
                        and not metrics.get("schedule_violation_count", 0)
                        and not metrics.get("schedule_violations", [])
                        and metrics["time"] <= record.instance.horizon + 1e-9
                        and metrics.get("maximum_worker_fatigue", math.inf) <= record.instance.fatigue.maximum_safe_fatigue + 1e-9)
    def comparison_gap(value, reference):
        return relative_gap_percent(value, reference) if comparison_valid else None
    metric_hash = quality_metric_sha256(quality_metric)
    preference = metrics.get("preference") or {}
    return {
        "experiment_name": config.get("experiment_name"),
        "encoder_variant": config.get("network", {}).get("encoder_variant", "hetero_gnn"),
        "actor_head_variant": config.get("network", {}).get("actor_head_variant", "objective_experts"),
        "instance_id": record.instance.instance_id,
        "seed": record.metadata["seed"],
        "generator_version": record.metadata.get("generator_version"),
        "severity": record.metadata.get("severity", 1.0),
        "feasibility_status": record.metadata.get("feasibility_status", "unknown"),
        "diagnostic_status": record.metadata.get("diagnostic_status", "not_run"),
        "diagnostic_terminal_reason": record.metadata.get("diagnostic_terminal_reason"),
        "heuristic_comparison_valid": comparison_valid,
        **{name: record.metadata.get(name) for name in ("generator_config_sha256", "environment_config_sha256", "distribution_contract_sha256")},
        "algorithm_seed": int(config["seed"]),
        "arm": metrics.get("arm", "ppo"),
        "dataset": record.metadata.get("split", ""),
        "result_schema_version": EVALUATION_SCHEMA_VERSION,
        "experiment_suite_version": config["experiment_suite_version"],
        "normalization_manifest_sha256": config["objective_scalarizer"]["normalization_manifest_sha256"],
        "objective_scalarizer_type": config["objective_scalarizer"]["type"],
        "objective_scalarizer_rho": config["objective_scalarizer"]["rho"],
        **{f"objective_scale_{name}": config["objective_scalarizer"]["scales"][name] for name in ("flow", "cost", "variance")},
        "candidate_id": f"{metrics.get('preference_key', '')}:{metrics.get('sampling_repeat', metrics.get('sampling_seed', 0))}",
        "candidate_source": metrics.get("decode_mode", "sampled"),
        "pressure_type": record.metadata["pressure_type"],
        "cost_profile": record.metadata["cost_profile"],
        "ood_factor": record.metadata.get("ood_factor"),
        "terminated": metrics["terminated"],
        "truncated": metrics["truncated"],
        "task_succeeded": metrics.get("task_succeeded"),
        "task_failed": metrics.get("task_failed"),
        "termination_reason": metrics["terminal_reason"],
        "decisions": metrics["decisions"],
        "makespan": metrics["time"],
        "completed_orders": metrics["completed_orders"],
        "unfinished_orders": metrics["unfinished_orders"],
        "initial_progress": metrics.get("initial_progress"),
        "operation_progress": metrics.get("operation_progress"),
        "reward_version": metrics.get("reward_version"),
        "initial_preference_quality_score": metrics.get(
            "initial_preference_quality_score"
        ),
        "single_stage_proxy_return": metrics[
            "single_stage_proxy_return"
        ],
        "total_flow_time": metrics["total_flow_time"],
        "flow_time_objective": metrics["flow_time_objective"],
        "reconfiguration_cost": metrics["reconfiguration_cost"],
        "worker_load_variance": metrics["worker_load_variance"],
        "preference": metrics.get("preference"),
        "preference_key": metrics.get("preference_key"),
        "preference_quality_score": metrics.get("preference_quality_score"),
        "actual_preference_quality_score": metrics.get(
            "actual_preference_quality_score"
        ),
        "terminal_failure_penalty_configured": metrics.get(
            "terminal_failure_penalty_configured"
        ),
        "terminal_failure_penalty_applied": metrics.get(
            "terminal_failure_penalty_applied"
        ),
        "base_cumulative_reward": metrics.get("base_cumulative_reward"),
        "training_cumulative_reward": metrics.get(
            "training_cumulative_reward"
        ),
        "preference_flow": preference.get("flow"),
        "preference_cost": preference.get("cost"),
        "preference_variance": preference.get("variance"),
        **{
            name: metrics.get(name, 0)
            for name in (
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
                "longest_forced_action_chain",
                "mean_forced_action_chain_length",
            )
        },
        "quality_score": bounded_quality_score(
            metrics["flow_time_objective"],
            metrics["reconfiguration_cost"],
            metrics["worker_load_variance"],
            quality_metric,
        ),
        "heuristic_quality_score": bounded_quality_score(
            heuristic_flow_time,
            heuristic_cost,
            heuristic_variance,
            quality_metric,
        ) if reference_valid else None,
        "reward_quality_score": terminal_quality_score(
            metrics["flow_time_objective"],
            metrics["reconfiguration_cost"],
            metrics["worker_load_variance"],
            config,
            preference=preference,
            terminal_failure=bool(metrics["truncated"]),
        ),
        "heuristic_reward_quality_score": bounded_quality_score(
            heuristic_flow_time,
            heuristic_cost,
            heuristic_variance,
            config,
            preference=preference,
        ) if reference_valid else None,
        "quality_metric_version": quality_metric["version"],
        "quality_metric_sha256": metric_hash,
        "decode_mode": metrics.get("decode_mode"),
        "result_role": metrics.get("result_role"),
        "action_trace_sha256": metrics.get("action_trace_sha256"),
        "sampling_seed": metrics.get("sampling_seed"),
        "sampling_repeat": metrics.get("sampling_repeat"),
        "derived_sampling_seed": metrics.get("derived_sampling_seed"),
        "sampling_evaluation_key": metrics.get(
            "sampling_evaluation_key"
        ),
        "sampling_rng_version": metrics.get("sampling_rng_version"),
        "policy_execution_version": metrics.get("policy_execution_version", config["training"].get("policy_execution_version", "phase_batched_v1")),
        "policy_precision": config["training"].get("policy_precision", "float32"),
        "validation_parallel_envs": int(config["training"].get("validation_parallel_envs", 1)),
        "inference_time_seconds": metrics[
            "inference_time_seconds"
        ],
        "solve_time_seconds": metrics["solve_time_seconds"],
        "inference_time_per_decision_ms": metrics[
            "inference_time_per_decision_ms"
        ],
        "heuristic_completed": heuristic.get(
            "heuristic_completed"
        ),
        "heuristic_makespan": heuristic_makespan,
        "heuristic_flow_time": heuristic_flow_time,
        "heuristic_reconfiguration_cost": heuristic_cost,
        "heuristic_worker_load_variance": heuristic_variance,
        "relative_heuristic_gap_percent": comparison_gap(
            metrics["flow_time_objective"],
            heuristic_flow_time,
        ),
        "makespan_heuristic_gap_percent": comparison_gap(
            metrics["time"],
            heuristic_makespan,
        ),
        "reconfiguration_cost_heuristic_gap_percent": (
            comparison_gap(
                metrics["reconfiguration_cost"],
                heuristic_cost,
            )
        ),
        "worker_load_variance_heuristic_gap_percent": (
            comparison_gap(
                metrics["worker_load_variance"],
                heuristic_variance,
            )
        ),
        "maximum_worker_fatigue": metrics[
            "maximum_worker_fatigue"
        ],
        "mean_peak_worker_fatigue": metrics[
            "mean_peak_worker_fatigue"
        ],
        "safe_fatigue_limit": metrics["safe_fatigue_limit"],
        "fatigue_masked_action_count": metrics[
            "fatigue_masked_action_count"
        ],
        "fatigue_masked_action_ratio": metrics[
            "fatigue_masked_action_ratio"
        ],
        "worker_competition_event_count": metrics[
            "worker_competition_event_count"
        ],
        "worker_matching_deficit_event_count": metrics[
            "worker_matching_deficit_event_count"
        ],
        "minimum_worker_alternatives": metrics[
            "minimum_worker_alternatives"
        ],
        "wait_total_ticks": metrics["wait_total_ticks"],
        "wait_total_time": metrics["wait_total_time"],
        "production_wait_ticks": metrics["production_wait_ticks"],
        "production_wait_time": metrics["production_wait_time"],
        "worker_wait_ticks": metrics["worker_wait_ticks"],
        "worker_wait_time": metrics["worker_wait_time"],
        "wait_min_estimated_deadline_slack_ticks": metrics[
            "wait_min_estimated_deadline_slack_ticks"
        ],
        **{
            name: metrics.get(name, 0)
            for name in (
                "ranker_top_decision_count",
                "ranker_top_selected_count",
                "ranker_top_selection_rate",
                "context_override_count",
                "context_override_rate",
                "production_pair_plus_wait_state_count",
                "production_decision_state_count",
                "production_pair_plus_wait_ratio",
                "worker_pair_plus_wait_state_count",
                "worker_decision_state_count",
                "worker_pair_plus_wait_ratio",
                "mean_commit_set_logit",
                "reconfiguration_reuse_count",
                "qualification_scarcity_regret",
                "qualification_scarcity_decision_count",
            )
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
        **{
            name: metrics[name]
            for name in (
                "direct_process_action_count",
                "commit_reconfig_action_count",
                "worker_assign_action_count",
                "wait_action_count",
                "production_wait_action_count",
                "worker_wait_action_count",
            )
        },
        "machine_waiting_for_worker_time": metrics[
            "machine_waiting_for_worker_time"
        ],
        "completed_reconfigurations": metrics[
            "completed_reconfigurations"
        ],
        "fatigue_mode": metrics.get("fatigue_mode", "full"),
        "active_constraint_pass": (len(metrics["schedule_violations"]) == 0
            and (metrics.get("fatigue_mode", "full") == "neutral"
                 or float(metrics.get("fatigue_monitor_peak", metrics["maximum_worker_fatigue"]))
                 <= float(metrics["safe_fatigue_limit"]) + 1e-9)),
        "physical_safety_pass": (len(metrics["schedule_violations"]) == 0
            and float(metrics.get("fatigue_monitor_peak", metrics["maximum_worker_fatigue"]))
            <= float(metrics["safe_fatigue_limit"]) + 1e-9),
        **{name: value for name, value in metrics.items() if name.startswith("fatigue_monitor_")},
        **{name: metrics.get(name) for name in (
            "worker_reconfiguration_busy_minutes", "mean_interstage_idle_minutes",
            "max_consecutive_worker_stages", "completed_reconfigurations_per_minute",
            "completed_reconfigurations_per_operation")},
        "worker_switch_ratio": metrics["worker_switch_ratio"],
        "schedule_violation_count": len(
            metrics["schedule_violations"]
        ),
        "total_effective_load": pressure[
            "total_effective_load"
        ],
        "max_module_load": pressure["max_module_load"],
        "ready_configuration_gap_ratio": heuristic.get("ready_configuration_gap_ratio"),
        "heuristic_reconfiguration_ratio": heuristic.get("heuristic_reconfiguration_ratio"),
        "mean_wave_overlap_ratio": heuristic.get("mean_wave_overlap_ratio"),
    }


build_evaluation_row = _evaluation_row


def _stamp_selection(rows, dataset, indices):
    selection = subset_snapshot(dataset, indices, role="evaluation")
    for row in rows:
        row["dataset_manifest_sha256"] = selection["dataset_manifest_sha256"]
        row["subset_sha256"] = selection["subset_sha256"]
    return selection


@_preserve_rng_for_sampled
def evaluate_dataset(
    config: dict[str, Any],
    *,
    dataset_name: str,
    policy_name: str,
    checkpoint: str | None = None,
    ppo_agent: PPOAgent | None = None,
    instance_limit: int | None = None,
    instance_offset: int | None = None,
    instance_indices: Sequence[int] | None = None,
    decode_mode: str | None = None,
    sampling_seed: int | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    decode_mode = _resolve_decode_mode(policy_name, decode_mode)
    sampling_seed = _resolve_sampling_seed(
        config,
        decode_mode=decode_mode,
        sampling_seed=sampling_seed,
    )
    dataset = load_dataset_split(config, dataset_name)
    indices = resolve_instance_indices(dataset, instance_indices=instance_indices, instance_limit=instance_limit, instance_offset=instance_offset)
    offset = 0 if instance_offset is None else instance_offset
    effective_count = len(indices)
    records = [dataset[index] for index in indices]
    bootstrap_environment = AssemblySchedulingEnv(config)
    bootstrap_observation = bootstrap_environment.reset(
        records[0].instance
    )
    if ppo_agent is None:
        set_seed(validate_algorithm_seed(config, int(config["seed"])))
    runner = EvaluationPolicy(
        config,
        policy_name=policy_name,
        bootstrap_observation=bootstrap_observation,
        checkpoint=checkpoint,
        ppo_agent=ppo_agent,
        decode_mode=decode_mode,
        sampling_seed=sampling_seed,
    )
    was_training = runner.enter_evaluation_mode()
    quality_metric = evaluation_quality_metric(config)
    rows: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    reconfigurations: list[dict[str, Any]] = []
    try:
        for record in records:
            environment, metrics = evaluate_instance(
                config,
                instance=record.instance,
                policy_name=policy_name,
                prepared_policy=runner,
                decode_mode=decode_mode,
                sampling_seed=sampling_seed,
            )
            row = _evaluation_row(
                record,
                metrics,
                config,
                quality_metric,
            )
            rows.append(row)
            schedules.extend(
                {"instance_id": record.instance.instance_id, **value}
                for value in environment.schedule_log
            )
            reconfigurations.extend(
                {"instance_id": record.instance.instance_id, **value}
                for value in environment.reconfiguration_log
            )
    finally:
        runner.restore_mode(was_training)
    selection = _stamp_selection(rows, dataset, indices)
    aggregate = aggregate_evaluation_rows(
        rows,
        dataset=dataset_name,
        policy=policy_name,
        manifest=str(dataset.manifest_path),
        quality_metric=quality_metric,
    )
    aggregate["decode_mode"] = decode_mode
    aggregate["result_role"] = (
        "formal_sampled"
        if policy_name == "ppo" and decode_mode == "sampled"
        else "greedy_diagnostic"
        if policy_name == "ppo"
        else "baseline"
    )
    aggregate["sampling_seed"] = sampling_seed
    aggregate["sampling_rng_version"] = (
        SAMPLED_EVALUATION_RNG_VERSION
        if decode_mode == "sampled"
        else None
    )
    aggregate["instance_offset"] = offset
    aggregate["instance_indices"] = indices
    aggregate["subset_sha256"] = selection["subset_sha256"]
    aggregate["dataset_manifest_sha256"] = dataset_manifest_snapshot(
        dataset.manifest_path
    )["sha256"]
    return rows, schedules, reconfigurations, aggregate


@_preserve_rng_for_sampled
def evaluate_dataset_parallel(
    config: dict[str, Any],
    *,
    dataset_name: str,
    ppo_agent: PPOAgent,
    runner: ParallelEpisodeRunner,
    instance_limit: int | None = None,
    instance_offset: int | None = None,
    instance_indices: Sequence[int] | None = None,
    decode_mode: str = "sampled",
    sampling_seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate fixed records in parallel for periodic training validation."""
    dataset = load_dataset_split(config, dataset_name)
    indices = resolve_instance_indices(dataset, instance_indices=instance_indices, instance_limit=instance_limit, instance_offset=instance_offset)
    offset = 0 if instance_offset is None else instance_offset
    effective_count = len(indices)
    records = [dataset[index] for index in indices]
    was_training = ppo_agent.network.training
    ppo_agent.network.eval()
    parallelism = min(
        int(config["training"]["validation_parallel_envs"]),
        runner.worker_count,
        effective_count,
    )
    sampling_seed = _resolve_sampling_seed(
        config,
        decode_mode=decode_mode,
        sampling_seed=sampling_seed,
    )
    if decode_mode not in {"greedy", "sampled"}:
        raise ValueError("decode_mode must be 'greedy' or 'sampled'")
    if decode_mode == "sampled" and sampling_seed is None:
        raise ValueError(
            "sampling_seed is required for sampled PPO evaluation"
        )
    try:
        rollouts = runner.evaluate_records(
            ppo_agent,
            records,
            max_parallelism=parallelism,
            deterministic=decode_mode == "greedy",
            sampling_seed=sampling_seed,
        )
    finally:
        ppo_agent.network.train(was_training)
    rows = []
    quality_metric = evaluation_quality_metric(config)
    for rollout in rollouts:
        metrics = dict(rollout.metrics)
        metrics["policy_execution_version"] = ppo_agent.network.execution_mode
        metrics["decisions"] = rollout.decisions
        metrics["inference_time_seconds"] = (
            rollout.inference_time_seconds
        )
        metrics["solve_time_seconds"] = rollout.solve_time_seconds
        metrics["inference_time_per_decision_ms"] = (
            1000.0
            * rollout.inference_time_seconds
            / rollout.decisions
            if rollout.decisions
            else 0.0
        )
        metrics["single_stage_proxy_return"] = proxy_return_from_metrics(
            metrics,
            config,
            preference=metrics.get("preference"),
        )
        metrics["action_trace_sha256"] = rollout.action_trace_sha256
        metrics["decode_mode"] = decode_mode
        metrics["result_role"] = (
            "formal_sampled"
            if decode_mode == "sampled"
            else "greedy_diagnostic"
        )
        metrics["sampling_seed"] = rollout.sampling_seed
        metrics["derived_sampling_seed"] = rollout.derived_sampling_seed
        metrics["sampling_evaluation_key"] = (
            rollout.sampling_evaluation_key
        )
        metrics["sampling_rng_version"] = (
            SAMPLED_EVALUATION_RNG_VERSION
            if decode_mode == "sampled"
            else None
        )
        rows.append(
            _evaluation_row(
                records[rollout.record_index],
                metrics,
                config,
                quality_metric,
            )
        )
    selection = _stamp_selection(rows, dataset, indices)
    aggregate = aggregate_evaluation_rows(
        rows,
        dataset=dataset_name,
        policy="ppo",
        manifest=str(dataset.manifest_path),
        quality_metric=quality_metric,
    )
    aggregate["decode_mode"] = decode_mode
    aggregate["result_role"] = (
        "formal_sampled"
        if decode_mode == "sampled"
        else "greedy_diagnostic"
    )
    aggregate["sampling_seed"] = sampling_seed
    aggregate["sampling_rng_version"] = (
        SAMPLED_EVALUATION_RNG_VERSION
        if decode_mode == "sampled"
        else None
    )
    aggregate["instance_offset"] = offset
    aggregate["instance_indices"] = indices
    aggregate["subset_sha256"] = selection["subset_sha256"]
    aggregate["parallel_envs"] = parallelism
    aggregate["dataset_manifest_sha256"] = dataset_manifest_snapshot(
        dataset.manifest_path
    )["sha256"]
    return rows, aggregate


@_preserve_rng_for_sampled
def evaluate_preference_grid_parallel(
    config: dict[str, Any],
    *,
    dataset_name: str,
    ppo_agent: PPOAgent,
    runner: ParallelEpisodeRunner,
    instance_limit: int | None = None,
    instance_offset: int | None = None,
    instance_indices: Sequence[int] | None = None,
    preferences: tuple[PreferenceContextInput, ...] | None = None,
    decode_mode: str = "sampled",
    sampling_seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate each fixed instance at an ordered formal preference set."""

    dataset = load_dataset_split(config, dataset_name)
    indices = resolve_instance_indices(dataset, instance_indices=instance_indices, instance_limit=instance_limit, instance_offset=instance_offset)
    offset = 0 if instance_offset is None else instance_offset
    instance_limit = len(indices)
    grid = formal_preferences(config, "validation") if preferences is None else tuple(preferences)
    if not grid or len({PreferenceContext.from_input(point).key for point in grid}) != len(grid):
        raise ValueError("preference grid must be nonempty and contain unique points")
    if decode_mode not in {"greedy", "sampled"}:
        raise ValueError("decode_mode must be 'greedy' or 'sampled'")
    if decode_mode == "sampled" and sampling_seed is None:
        raise ValueError("sampled V8 evaluation requires a sampling_seed")
    source_records = [
        dataset[index] for index in indices
    ]
    records = [record for record in source_records for _ in grid]
    repeated_preferences = [preference for _ in source_records for preference in grid]
    was_training = ppo_agent.network.training
    ppo_agent.network.eval()
    try:
        rollouts = runner.evaluate_records(
            ppo_agent,
            records,
            max_parallelism=min(
                int(config["training"]["validation_parallel_envs"]),
                runner.worker_count,
                len(records),
            ),
            deterministic=decode_mode == "greedy",
            sampling_seed=sampling_seed,
            preferences=repeated_preferences,
        )
    finally:
        ppo_agent.network.train(was_training)
    quality_metric = evaluation_quality_metric(config)
    rows: list[dict[str, Any]] = []
    for rollout in rollouts:
        metrics = dict(rollout.metrics)
        metrics["policy_execution_version"] = ppo_agent.network.execution_mode
        metrics.update(
            {
                "decisions": rollout.decisions,
                "inference_time_seconds": rollout.inference_time_seconds,
                "solve_time_seconds": rollout.solve_time_seconds,
                "inference_time_per_decision_ms": (
                    1000.0 * rollout.inference_time_seconds / rollout.decisions
                    if rollout.decisions
                    else 0.0
                ),
                "action_trace_sha256": rollout.action_trace_sha256,
                "decode_mode": decode_mode,
                "result_role": (
                    "formal_sampled"
                    if decode_mode == "sampled"
                    else "greedy_diagnostic"
                ),
                "sampling_seed": rollout.sampling_seed,
                "sampling_repeat": 0,
                "derived_sampling_seed": rollout.derived_sampling_seed,
                "sampling_evaluation_key": (
                    rollout.sampling_evaluation_key
                ),
                "sampling_rng_version": (
                    SAMPLED_EVALUATION_RNG_VERSION
                    if decode_mode == "sampled"
                    else None
                ),
            }
        )
        metrics["single_stage_proxy_return"] = proxy_return_from_metrics(
            metrics,
            config,
            preference=metrics.get("preference"),
        )
        rows.append(
            _evaluation_row(
                records[rollout.record_index], metrics, config, quality_metric
            )
        )
    preference_keys = [PreferenceContext.from_input(point).key for point in grid]
    observed_keys = {str(row["preference_key"]) for row in rows}
    if observed_keys != set(preference_keys) or any(
        sum(str(row["preference_key"]) == key for row in rows) != instance_limit
        for key in preference_keys
    ):
        raise ValueError("preference grid evaluation produced incomplete cells")
    completion_by_preference = {
        key: sum(successful_row(row) for row in rows if row["preference_key"] == key)
        / instance_limit
        for key in preference_keys
    }
    selection = _stamp_selection(rows, dataset, indices)
    summary = aggregate_evaluation_rows(
        rows,
        dataset=dataset_name,
        policy="ppo",
        manifest=str(dataset.manifest_path),
        quality_metric=quality_metric,
    )
    cell_count = int(summary["instance_count"])
    summary.update({
        "dataset": dataset_name,
        "instance_offset": offset,
        "instance_indices": indices,
        "subset_sha256": selection["subset_sha256"],
        "instance_count": instance_limit,
        "preference_count": len(grid),
        "ordered_preference_set": [PreferenceContext.from_input(point).preference.as_dict() for point in grid],
        "cell_count": cell_count,
        "completed_cell_count": int(summary["completed_count"]),
        "cell_completion_rate": float(summary["completion_rate"]),
        "completed_count": sum(
            all(
                successful_row(row)
                for row in rows
                if row["instance_id"] == record.instance.instance_id
            )
            for record in source_records
        ),
        "completion_rate_by_preference": completion_by_preference,
        "minimum_preference_completion_rate": min(completion_by_preference.values()),
        "completion_rate": min(completion_by_preference.values()),
        "schedule_violation_count": sum(int(row["schedule_violation_count"]) for row in rows),
        "dataset_manifest_sha256": dataset_manifest_snapshot(dataset.manifest_path)["sha256"],
        "normalization_manifest_sha256": config["objective_scalarizer"].get(
            "normalization_manifest_sha256"
        ),
        "decode_mode": decode_mode,
        "result_role": (
            "formal_sampled"
            if decode_mode == "sampled"
            else "greedy_diagnostic"
        ),
        "sampling_seed": sampling_seed,
        "sampling_rng_version": (
            SAMPLED_EVALUATION_RNG_VERSION
            if decode_mode == "sampled"
            else None
        ),
    })
    return rows, summary


def _run_formal_grid_cli(config: dict[str, Any], args: Any, decode_mode: str) -> None:
    """Evaluate the configured Universal grid with the same parallel path as training."""
    if args.policy != "ppo" or not args.checkpoint:
        raise ValueError("--preference-set requires PPO and --checkpoint")
    if decode_mode != "sampled":
        raise ValueError("formal preference-set evaluation requires sampled decoding")
    if config["preference"]["quality"]["mode"] != "universal_sobol_v1":
        raise ValueError("--preference-set requires a Universal config")
    required_split = (
        str(config["training"]["validation_split"])
        if args.preference_set == "validation"
        else "test"
    )
    if args.dataset != required_split:
        raise ValueError(
            f"{args.preference_set} preference evaluation requires dataset {required_split}"
        )
    preferences = formal_preferences(config, args.preference_set)
    dataset = load_dataset_split(config, args.dataset)
    default_count = (
        int(config["training"]["validation_instance_limit"])
        if args.preference_set == "validation"
        else len(dataset)
    )
    if args.instance_indices is not None or args.instance_offset is not None or args.instance_limit is not None:
        indices = resolve_instance_indices(dataset, instance_indices=args.instance_indices,
            instance_limit=args.instance_limit, instance_offset=args.instance_offset)
    elif args.preference_set == "validation":
        indices = select_validation_subsets(dataset, config["generator"]["dataset_pressure_weights"],
            target_count=default_count, diagnostic_count=0)["target"]["instance_indices"]
    else:
        indices = list(range(default_count))
    instance_count = len(indices)
    bootstrap = AssemblySchedulingEnv(config).reset(dataset[0].instance)
    set_seed(int(config["seed"]))
    agent = PPOAgent(
        build_actor_critic(bootstrap, config["network"]),
        config["ppo"],
        device=config["device"],
    )
    checkpoint_path = project_path(args.checkpoint)
    metadata = agent.load(checkpoint_path, load_optimizer=False)
    from configs.runtime import assert_checkpoint_fatigue_mode
    assert_checkpoint_fatigue_mode(metadata, config)
    seeds = (
        [int(args.sampling_seed)]
        if args.sampling_seed is not None
        else configured_formal_evaluation_sampling_seeds(config, args.preference_set)
    )
    rows: list[dict[str, Any]] = []
    with ParallelEpisodeRunner(
        config=config,
        template=load_instance_yaml(project_path(config["paths"]["fixed_instance"])),
        episode_count=instance_count,
        worker_count=min(
            int(config["training"]["validation_parallel_envs"]),
            instance_count * len(preferences),
        ),
    ) as runner:
        for repeat_index, seed in enumerate(seeds):
            current, _ = evaluate_preference_grid_parallel(
                config,
                dataset_name=args.dataset,
                ppo_agent=agent,
                runner=runner,
                instance_indices=indices,
                preferences=preferences,
                decode_mode=decode_mode,
                sampling_seed=seed,
            )
            for row in current:
                row["sampling_repeat"] = repeat_index
            rows.extend(current)
    from train import _aggregate_formal_rows

    metrics = _aggregate_formal_rows(
        config,
        rows=rows,
        dataset_name=args.dataset,
        manifest=str(dataset.manifest_path),
        unique_instance_count=instance_count,
        repeat_count=len(seeds),
        universal=True,
        stage=args.preference_set,
        strict_counts=True,
    )
    metrics["sampling_seeds"] = [int(seed) for seed in seeds]
    selection = subset_snapshot(dataset, indices, role="evaluation")
    metrics.update(instance_indices=indices, subset_sha256=selection["subset_sha256"])
    metrics["dataset_manifest_sha256"] = dataset_manifest_snapshot(
        dataset.manifest_path
    )["sha256"]
    metrics["normalization_manifest_sha256"] = config["objective_scalarizer"].get(
        "normalization_manifest_sha256"
    )
    metrics["provenance"] = build_provenance(
        config,
        dataset_manifest_path=dataset.manifest_path,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=metadata,
        formal_evaluation_stage=args.preference_set,
        evaluation_subset_sha256=selection["subset_sha256"],
    )
    for row in rows:
        row["checkpoint_sha256"] = metrics["provenance"].get("checkpoint_sha256")
        row["effective_config_sha256"] = metrics["provenance"]["effective_config_sha256"]
    metrics["active_constraint_pass"] = bool(rows) and all(row["active_constraint_pass"] for row in rows)
    metrics["physical_safety_pass"] = bool(rows) and all(row["physical_safety_pass"] for row in rows)
    run_directory = create_run_directory(
        project_path(config["paths"]["result_root"]),
        label=f"eval_ppo_{decode_mode}_{args.dataset}_{args.preference_set}",
        run_name=args.run_name,
    )
    write_config(run_directory, config)
    write_json(run_directory / "metrics.json", metrics)
    write_csv(run_directory / "instance_metrics.csv", rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"results: {run_directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a scheduling policy")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--policy", choices=("heuristic", "random", "ppo"), default="heuristic"
    )
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--decode-mode",
        choices=("greedy", "sampled"),
        default=None,
    )
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--algorithm-seed", type=int)
    parser.add_argument(
        "--dataset",
        choices=PERSISTED_SPLITS,
        required=True,
    )
    parser.add_argument("--run-name")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--preference-set", choices=("validation", "final_test"))
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--instance-offset", type=int)
    parser.add_argument("--instance-indices", type=int, nargs="+")
    args = parser.parse_args()
    if args.instance_indices is not None and (args.instance_limit is not None or args.instance_offset is not None):
        parser.error("--instance-indices and --instance-offset/--instance-limit are mutually exclusive")

    config = deepcopy(load_config(args.config))
    if args.device is not None:
        config["device"] = args.device
    config["seed"] = validate_algorithm_seed(
        config,
        int(config["seed"])
        if args.algorithm_seed is None
        else args.algorithm_seed,
    )
    decode_mode = _resolve_decode_mode(args.policy, args.decode_mode)
    if args.preference_set is not None:
        _run_formal_grid_cli(config, args, decode_mode)
        return
    sampling_seeds: list[int | None]
    if decode_mode == "sampled":
        sampling_seeds = (
            [int(args.sampling_seed)]
            if args.sampling_seed is not None
            else list(
                configured_formal_evaluation_sampling_seeds(
                    config, "final_test"
                )
            )
        )
    else:
        sampling_seeds = [None]
    rows: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    reconfigurations: list[dict[str, Any]] = []
    per_repeat_metrics: list[dict[str, Any]] = []
    for repeat_index, sampling_seed in enumerate(sampling_seeds):
        repeat_rows, repeat_schedules, repeat_reconfigurations, repeat_metrics = (
            evaluate_dataset(
                config,
                dataset_name=args.dataset,
                policy_name=args.policy,
                checkpoint=args.checkpoint,
                instance_limit=args.instance_limit,
                instance_offset=args.instance_offset,
                instance_indices=args.instance_indices,
                decode_mode=decode_mode,
                sampling_seed=sampling_seed,
            )
        )
        for collection in (
            repeat_rows,
            repeat_schedules,
            repeat_reconfigurations,
        ):
            for row in collection:
                row["sampling_repeat"] = repeat_index
                row["sampling_seed"] = sampling_seed
        rows.extend(repeat_rows)
        schedules.extend(repeat_schedules)
        reconfigurations.extend(repeat_reconfigurations)
        per_repeat_metrics.append(repeat_metrics)
    if len(per_repeat_metrics) == 1:
        metrics = per_repeat_metrics[0]
    else:
        reference = per_repeat_metrics[0]
        metrics = aggregate_evaluation_rows(
            rows,
            dataset=args.dataset,
            policy=args.policy,
            manifest=str(reference["manifest"]),
            quality_metric=evaluation_quality_metric(config),
        )
        metrics.update(
            {
                "decode_mode": decode_mode,
                "sampling_seeds": [int(seed) for seed in sampling_seeds if seed is not None],
                "sampling_rng_version": SAMPLED_EVALUATION_RNG_VERSION,
                "repeat_count": len(sampling_seeds),
                "unique_instance_count": (
                    int(metrics["instance_count"]) // len(sampling_seeds)
                ),
                "per_repeat_metrics": per_repeat_metrics,
                "dataset_manifest_sha256": reference[
                    "dataset_manifest_sha256"
                ],
                "instance_indices": reference["instance_indices"],
                "subset_sha256": reference["subset_sha256"],
            }
        )
    metrics["result_role"] = (
        "formal_sampled" if decode_mode == "sampled" else "greedy_diagnostic"
    )
    checkpoint_path = (
        project_path(args.checkpoint) if args.checkpoint is not None else None
    )
    checkpoint_metadata = None
    if checkpoint_path is not None:
        checkpoint_payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_metadata = dict(checkpoint_payload.get("metadata", {}))
    metrics["provenance"] = build_provenance(
        config,
        dataset_manifest_path=metrics["manifest"],
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        evaluation_subset_sha256=metrics["subset_sha256"],
    )
    for row in rows:
        row["checkpoint_sha256"] = metrics["provenance"].get("checkpoint_sha256")
        row["effective_config_sha256"] = metrics["provenance"]["effective_config_sha256"]
    metrics["active_constraint_pass"] = bool(rows) and all(row["active_constraint_pass"] for row in rows)
    metrics["physical_safety_pass"] = bool(rows) and all(row["physical_safety_pass"] for row in rows)
    run_directory = create_run_directory(
        project_path(config["paths"]["result_root"]),
        label=(
            f"eval_{args.policy}_{decode_mode}_{args.dataset}"
        ),
        run_name=args.run_name,
    )
    write_config(run_directory, config)
    write_json(run_directory / "metrics.json", metrics)
    write_csv(run_directory / "instance_metrics.csv", rows)
    write_csv(run_directory / "schedule.csv", schedules)
    write_csv(
        run_directory / "reconfigurations.csv",
        reconfigurations,
    )
    from collections import defaultdict
    from environment.fatigue_monitor import audit_fatigue
    original_instances = {record.instance.instance_id: record.instance
                          for record in load_dataset_split(config, args.dataset)}
    grouped = defaultdict(list)
    for record in reconfigurations:
        grouped[(str(record["instance_id"]), int(record["sampling_repeat"]))].append(record)
    fatigue_segments = []
    for row in rows:
        key = (row["instance_id"], int(row["sampling_repeat"]))
        _, segments = audit_fatigue(original_instances[key[0]], grouped.get(key, []), float(row["makespan"]))
        fatigue_segments.extend({"instance_id": key[0], "sampling_repeat": key[1], **segment} for segment in segments)
    write_csv(run_directory / "fatigue_segments.csv", fatigue_segments)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"results: {run_directory}")


if __name__ == "__main__":
    main()
