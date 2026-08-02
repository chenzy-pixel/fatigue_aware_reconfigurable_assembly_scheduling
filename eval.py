from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from typing import Any

import torch

from agent.ppo import (
    PPOAgent,
    build_actor_critic,
)
from agent.ppo.parallel import ParallelEpisodeRunner
from agent.baselines import HeuristicPolicy, RandomPolicy
from configs import load_config, project_path
from data import (
    AssemblyInstance,
    load_dataset_split,
    load_instance_pickle,
    load_instance_yaml,
    save_instance_pickle,
)
from data.dataset import PERSISTED_SPLITS, validate_algorithm_seed
from environment import AssemblySchedulingEnv, proxy_return_from_metrics
from result import (
    aggregate_evaluation_rows,
    create_run_directory,
    relative_gap_percent,
)
from result.io import write_config, write_csv, write_json
from utils import set_seed


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
        decode_mode: str = "greedy",
        sampling_seed: int | None = None,
    ):
        self.policy_name = policy_name
        self.device = torch.device(config["device"])
        self.ppo_agent: PPOAgent | None = None
        self.policy: HeuristicPolicy | RandomPolicy | None = None
        if decode_mode not in {"greedy", "sampled"}:
            raise ValueError("decode_mode must be 'greedy' or 'sampled'")
        if policy_name != "ppo" and decode_mode != "greedy":
            raise ValueError("sampled decode_mode is only available for PPO")
        self.decode_mode = decode_mode
        self.generator: torch.Generator | None = None
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
                ppo_agent.load(checkpoint_path)
            self.ppo_agent = ppo_agent
            self.device = ppo_agent.device
            if self.decode_mode == "sampled":
                if sampling_seed is None:
                    raise ValueError(
                        "sampling_seed is required for sampled PPO evaluation"
                    )
                self.generator = torch.Generator(
                    device=self.device
                ).manual_seed(int(sampling_seed))
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
            return action
        if self.policy is None:
            raise RuntimeError("evaluation policy is not initialized")
        return self.policy.select_action(environment)

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


def evaluate(
    config: dict[str, Any],
    *,
    policy_name: str,
    checkpoint: str | None = None,
    decode_mode: str = "greedy",
    sampling_seed: int | None = None,
) -> tuple[AssemblySchedulingEnv, dict[str, Any]]:
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


def evaluate_instance(
    config: dict[str, Any],
    *,
    instance: AssemblyInstance,
    policy_name: str,
    checkpoint: str | None = None,
    prepared_policy: EvaluationPolicy | None = None,
    decode_mode: str = "greedy",
    sampling_seed: int | None = None,
) -> tuple[AssemblySchedulingEnv, dict[str, Any]]:
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
    observation = env.reset(instance)
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
    metrics["decode_mode"] = decode_mode
    metrics["feasibility_proxy_return"] = proxy_return_from_metrics(
        metrics,
        config["reward"],
        "feasibility",
    )
    metrics["decisions"] = decisions
    metrics["inference_time_seconds"] = inference_time
    metrics["solve_time_seconds"] = solve_time
    metrics["inference_time_per_decision_ms"] = (
        1000.0 * inference_time / decisions if decisions else 0.0
    )
    metrics["schedule_violations"] = env.validate_schedule()
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


def _evaluation_row(record, metrics: dict[str, Any]) -> dict[str, Any]:
    heuristic = record.metadata["heuristic_metrics"]
    pressure = record.metadata["pressure_metrics"]
    heuristic_flow_time = heuristic.get("heuristic_flow_time")
    heuristic_makespan = heuristic.get("heuristic_makespan")
    heuristic_cost = heuristic.get(
        "heuristic_reconfiguration_cost"
    )
    heuristic_variance = heuristic.get(
        "worker_workload_variance"
    )
    return {
        "instance_id": record.instance.instance_id,
        "seed": record.metadata["seed"],
        "pressure_type": record.metadata["pressure_type"],
        "cost_profile": record.metadata["cost_profile"],
        "ood_factor": record.metadata.get("ood_factor"),
        "terminated": metrics["terminated"],
        "truncated": metrics["truncated"],
        "termination_reason": metrics["terminal_reason"],
        "decisions": metrics["decisions"],
        "makespan": metrics["time"],
        "completed_orders": metrics["completed_orders"],
        "unfinished_orders": metrics["unfinished_orders"],
        "feasibility_proxy_return": metrics[
            "feasibility_proxy_return"
        ],
        "total_flow_time": metrics["total_flow_time"],
        "flow_time_objective": metrics["flow_time_objective"],
        "reconfiguration_cost": metrics["reconfiguration_cost"],
        "worker_load_variance": metrics["worker_load_variance"],
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
        "relative_heuristic_gap_percent": relative_gap_percent(
            metrics["flow_time_objective"],
            heuristic_flow_time,
        ),
        "makespan_heuristic_gap_percent": relative_gap_percent(
            metrics["time"],
            heuristic_makespan,
        ),
        "reconfiguration_cost_heuristic_gap_percent": (
            relative_gap_percent(
                metrics["reconfiguration_cost"],
                heuristic_cost,
            )
        ),
        "worker_load_variance_heuristic_gap_percent": (
            relative_gap_percent(
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
        "resource_admission_masked_action_count": metrics[
            "resource_admission_masked_action_count"
        ],
        "resource_admission_masked_action_ratio": metrics[
            "resource_admission_masked_action_ratio"
        ],
        "minimum_worker_alternatives": metrics[
            "minimum_worker_alternatives"
        ],
        "matching_preserving_worker_action_count": metrics[
            "matching_preserving_worker_action_count"
        ],
        "candidate_recovery_advance_count": metrics[
            "candidate_recovery_advance_count"
        ],
        "machine_waiting_for_worker_time": metrics[
            "machine_waiting_for_worker_time"
        ],
        "completed_reconfigurations": metrics[
            "completed_reconfigurations"
        ],
        "worker_switch_ratio": metrics["worker_switch_ratio"],
        "schedule_violation_count": len(
            metrics["schedule_violations"]
        ),
        "total_effective_load": pressure[
            "total_effective_load"
        ],
        "max_module_load": pressure["max_module_load"],
        "ready_configuration_gap_ratio": heuristic[
            "ready_configuration_gap_ratio"
        ],
        "heuristic_reconfiguration_ratio": heuristic[
            "heuristic_reconfiguration_ratio"
        ],
        "mean_wave_overlap_ratio": heuristic[
            "mean_wave_overlap_ratio"
        ],
    }


def evaluate_dataset(
    config: dict[str, Any],
    *,
    dataset_name: str,
    policy_name: str,
    checkpoint: str | None = None,
    ppo_agent: PPOAgent | None = None,
    instance_limit: int | None = None,
    decode_mode: str = "greedy",
    sampling_seed: int | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    dataset = load_dataset_split(config, dataset_name)
    effective_count = (
        len(dataset) if instance_limit is None else int(instance_limit)
    )
    if effective_count < 1 or effective_count > len(dataset):
        raise ValueError(
            f"instance_limit must be in [1, {len(dataset)}]"
        )
    records = [dataset[index] for index in range(effective_count)]
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
            )
            row = _evaluation_row(record, metrics)
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
    aggregate = aggregate_evaluation_rows(
        rows,
        dataset=dataset_name,
        policy=policy_name,
        manifest=str(dataset.manifest_path),
    )
    aggregate["decode_mode"] = decode_mode
    return rows, schedules, reconfigurations, aggregate


def evaluate_dataset_parallel(
    config: dict[str, Any],
    *,
    dataset_name: str,
    ppo_agent: PPOAgent,
    runner: ParallelEpisodeRunner,
    instance_limit: int | None = None,
    decode_mode: str = "greedy",
    sampling_seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate fixed records in parallel for periodic training validation."""
    dataset = load_dataset_split(config, dataset_name)
    effective_count = (
        len(dataset) if instance_limit is None else int(instance_limit)
    )
    if effective_count < 1 or effective_count > len(dataset):
        raise ValueError(
            f"instance_limit must be in [1, {len(dataset)}]"
        )
    records = [dataset[index] for index in range(effective_count)]
    was_training = ppo_agent.network.training
    ppo_agent.network.eval()
    parallelism = min(
        int(config["training"]["validation_parallel_envs"]),
        runner.worker_count,
        effective_count,
    )
    if decode_mode not in {"greedy", "sampled"}:
        raise ValueError("decode_mode must be 'greedy' or 'sampled'")
    if decode_mode == "sampled" and sampling_seed is None:
        raise ValueError(
            "sampling_seed is required for sampled PPO evaluation"
        )
    generator = (
        torch.Generator(device=ppo_agent.device).manual_seed(
            int(sampling_seed)
        )
        if decode_mode == "sampled"
        else None
    )
    try:
        rollouts = runner.evaluate_records(
            ppo_agent,
            records,
            max_parallelism=parallelism,
            deterministic=decode_mode == "greedy",
            generator=generator,
        )
    finally:
        ppo_agent.network.train(was_training)
    rows = []
    for rollout in rollouts:
        metrics = dict(rollout.metrics)
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
        metrics["feasibility_proxy_return"] = proxy_return_from_metrics(
            metrics,
            config["reward"],
            "feasibility",
        )
        rows.append(
            _evaluation_row(records[rollout.record_index], metrics)
        )
    aggregate = aggregate_evaluation_rows(
        rows,
        dataset=dataset_name,
        policy="ppo",
        manifest=str(dataset.manifest_path),
    )
    aggregate["decode_mode"] = decode_mode
    aggregate["parallel_envs"] = parallelism
    return rows, aggregate


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
        default="greedy",
    )
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--algorithm-seed", type=int)
    parser.add_argument(
        "--dataset",
        choices=PERSISTED_SPLITS,
        required=True,
    )
    parser.add_argument("--run-name")
    args = parser.parse_args()

    config = deepcopy(load_config(args.config))
    config["seed"] = validate_algorithm_seed(
        config,
        int(config["seed"])
        if args.algorithm_seed is None
        else args.algorithm_seed,
    )
    rows, schedules, reconfigurations, metrics = evaluate_dataset(
        config,
        dataset_name=args.dataset,
        policy_name=args.policy,
        checkpoint=args.checkpoint,
        decode_mode=args.decode_mode,
        sampling_seed=(
            args.sampling_seed
            if args.sampling_seed is not None
            else int(config["seed"]) + 100000
            if args.decode_mode == "sampled"
            else None
        ),
    )
    run_directory = create_run_directory(
        project_path(config["paths"]["result_root"]),
        label=(
            f"eval_{args.policy}_{args.decode_mode}_{args.dataset}"
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
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"results: {run_directory}")


if __name__ == "__main__":
    main()
