from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

import train as training_module
from agent.ppo import RolloutBuffer
from agent.ppo.parallel import (
    EpisodeRollout,
    TrainingRolloutBatch,
)
from data.dataset import GeneratedInstanceRecord
from environment import AssemblySchedulingEnv


def _validation_aggregate(flow_objective: float) -> dict:
    def metric(mean: float):
        return {"count": 2, "mean": mean, "std": 0.0}

    return {
        "evaluation_schema_version": "2.1.0",
        "dataset": "validation",
        "manifest": "fixed-manifest.json",
        "policy": "ppo",
        "instance_count": 2,
        "completed_count": 2,
        "completion_rate": 1.0,
        "truncated_count": 0,
        "schedule_violation_count": 0,
        "decision_count": 4,
        "total_inference_time_seconds": 0.1,
        "total_solve_time_seconds": 0.2,
        "completed_metrics": {
            "makespan": metric(100.0),
            "total_flow_time": metric(flow_objective),
        },
        "all_instance_metrics": {
            "flow_time_objective": metric(flow_objective),
            "reconfiguration_cost": metric(10.0),
            "worker_load_variance": metric(2.0),
            "inference_time_seconds": metric(0.05),
            "solve_time_seconds": metric(0.1),
            "inference_time_per_decision_ms": metric(1.0),
        },
        "gap_metrics": {
            "relative_heuristic_gap_percent": metric(5.0),
            "makespan_heuristic_gap_percent": metric(4.0),
            "reconfiguration_cost_heuristic_gap_percent": metric(3.0),
            "worker_load_variance_heuristic_gap_percent": metric(2.0),
        },
    }


def test_training_uses_unique_episode_instances_and_writes_validation_artifacts(
    config,
    fixed_instance,
    tmp_path,
    monkeypatch,
):
    effective_config = deepcopy(config)
    effective_config["paths"]["result_root"] = str(tmp_path)
    effective_config["training"]["smoke_episodes"] = 11
    effective_config["training"]["smoke_rollout_steps"] = 1
    effective_config["training"]["validation_interval_episodes"] = 10
    effective_config["training"]["smoke_validation_instance_limit"] = 2
    effective_config["training"]["two_stage"][
        "consecutive_validations"
    ] = 1

    class FakeOnlineDataset:
        def __init__(self, *, episode_count, **kwargs):
            self.episode_count = episode_count

        def __getitem__(self, index):
            instance = replace(
                fixed_instance,
                instance_id=f"train_balanced_{1_000_000 + index}",
            )
            return GeneratedInstanceRecord(
                instance=instance,
                metadata={
                    "seed": 1_000_000 + index,
                    "pressure_type": "balanced",
                    "cost_profile": "balanced_cost",
                },
            )

    monkeypatch.setattr(
        training_module,
        "OnlineInstanceDataset",
        FakeOnlineDataset,
    )
    validation_calls = []

    def fake_evaluate_dataset(
        config,
        *,
        dataset_name,
        policy_name,
        ppo_agent,
        instance_limit,
        **kwargs,
    ):
        assert dataset_name == "validation"
        assert policy_name == "ppo"
        assert ppo_agent is not None
        assert instance_limit == 2
        validation_calls.append(len(validation_calls))
        objective = 10.0 if len(validation_calls) == 1 else 20.0
        return [], [], [], _validation_aggregate(objective)

    monkeypatch.setattr(
        training_module,
        "evaluate_dataset",
        fake_evaluate_dataset,
    )

    run_directory = training_module.train(
        effective_config,
        smoke=True,
        online_instances=True,
        run_name="training_dataset_test",
        parallel_envs=1,
    )

    assert len(validation_calls) == 2
    assert (run_directory / "checkpoint.pt").exists()
    assert (run_directory / "best_checkpoint.pt").exists()
    assert (run_directory / "train_log.csv").exists()
    assert (run_directory / "validation_log.csv").exists()
    summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["episodes"] == 11
    assert summary["unique_instance_count"] == 11
    assert summary["validation_runs"] == 2
    assert summary["best_validation"]["episode"] == 10

    with (run_directory / "train_log.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["instance_seed"]) for row in rows] == list(
        range(1_000_000, 1_000_011)
    )
    with (run_directory / "validation_log.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        validation_rows = list(csv.DictReader(handle))
    assert [int(row["episode"]) for row in validation_rows] == [10, 11]

    best = torch.load(
        run_directory / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert best["metadata"]["best_episode"] == 10


def test_non_smoke_fixed_instance_training_is_rejected(
    config,
    tmp_path,
):
    effective_config = deepcopy(config)
    effective_config["paths"]["result_root"] = str(tmp_path)
    with pytest.raises(ValueError, match="only available with --smoke"):
        training_module.train(
            effective_config,
            smoke=False,
            online_instances=False,
            run_name="invalid_fixed_training",
        )


def test_parallel_training_configuration_contract(config):
    invalid = deepcopy(config)
    invalid["training"]["validation_interval_episodes"] = 10
    with pytest.raises(ValueError, match="must be divisible"):
        training_module.train(
            invalid,
            smoke=False,
            online_instances=True,
            parallel_envs=3,
        )
    with pytest.raises(
        ValueError,
        match="requires online training instances",
    ):
        training_module.train(
            invalid,
            smoke=True,
            online_instances=False,
            parallel_envs=2,
        )


def test_parallel_training_batches_updates_and_writes_update_log(
    config,
    fixed_instance,
    tmp_path,
    monkeypatch,
):
    effective_config = deepcopy(config)
    effective_config["paths"]["result_root"] = str(tmp_path)
    effective_config["training"]["smoke_episodes"] = 11
    effective_config["training"]["smoke_parallel_envs"] = 2
    effective_config["training"]["smoke_validation_instance_limit"] = 2
    effective_config["training"]["validation_interval_episodes"] = 10
    effective_config["training"]["two_stage"][
        "consecutive_validations"
    ] = 1

    class FakeParallelRunner:
        def __init__(self, *, worker_count, **kwargs):
            self.worker_count = worker_count

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def collect_training_batch(
            self,
            agent,
            episode_indices,
            *,
            gamma,
            gae_lambda,
            step_limit,
            reward_phase,
        ):
            assert reward_phase in {"feasibility", "quality", "legacy"}
            episodes = []
            combined = RolloutBuffer(
                preserve_graph=agent.requires_graph_observation
            )
            for index in episode_indices:
                environment = AssemblySchedulingEnv(effective_config)
                observation = environment.reset(fixed_instance)
                mask = environment.get_action_mask()
                action, log_probability, value = agent.act(
                    observation,
                    mask,
                )
                buffer = RolloutBuffer(
                    preserve_graph=agent.requires_graph_observation
                )
                buffer.add(
                    observation,
                    mask,
                    action,
                    log_probability,
                    value,
                    0.0,
                    True,
                )
                buffer.compute_gae(
                    last_value=0.0,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                )
                combined.extend(buffer)
                episodes.append(
                    EpisodeRollout(
                        episode_index=index,
                        instance_id=f"parallel_{1_000_000 + index}",
                        metadata={
                            "seed": 1_000_000 + index,
                            "pressure_type": "balanced",
                            "cost_profile": "balanced_cost",
                        },
                        buffer=buffer,
                        reward_sum=0.0,
                        step_count=1,
                        metrics={
                            "completed_operations": 0,
                            "time": 0.0,
                        },
                        generation_time_seconds=0.01,
                        environment_step_time_seconds=0.01,
                    )
                )
            return TrainingRolloutBatch(
                episodes=episodes,
                buffer=combined,
                sampling_wall_time_seconds=0.1,
                policy_inference_time_seconds=0.02,
            )

    monkeypatch.setattr(
        training_module,
        "ParallelEpisodeRunner",
        FakeParallelRunner,
    )
    validation_calls = []

    def fake_parallel_validation(
        config,
        *,
        dataset_name,
        ppo_agent,
        runner,
        instance_limit,
    ):
        assert dataset_name == "validation"
        assert ppo_agent is not None
        assert runner.worker_count == 2
        assert instance_limit == 2
        validation_calls.append(1)
        objective = 10.0 if len(validation_calls) == 1 else 20.0
        return [], _validation_aggregate(objective)

    monkeypatch.setattr(
        training_module,
        "evaluate_dataset_parallel",
        fake_parallel_validation,
    )
    run_directory = training_module.train(
        effective_config,
        smoke=True,
        online_instances=True,
        run_name="parallel_training_test",
        parallel_envs=2,
    )
    assert len(validation_calls) == 2
    with (run_directory / "update_log.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        updates = list(csv.DictReader(handle))
    assert len(updates) == 6
    assert [int(row["episode_count"]) for row in updates] == [
        2,
        2,
        2,
        2,
        2,
        1,
    ]
    with (run_directory / "train_log.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        training_rows = list(csv.DictReader(handle))
    assert [int(row["instance_seed"]) for row in training_rows] == list(
        range(1_000_000, 1_000_011)
    )
    summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["parallel_envs"] == 2
    assert summary["updates"] == 6
    assert summary["transitions"] == 11
    assert summary["validation_runs"] == 2
    assert (run_directory / "checkpoint.pt").exists()
    assert (run_directory / "best_checkpoint.pt").exists()
    repeat_directory = training_module.train(
        effective_config,
        smoke=True,
        online_instances=True,
        run_name="parallel_training_repeat_test",
        parallel_envs=2,
    )
    first_checkpoint = torch.load(
        run_directory / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    repeat_checkpoint = torch.load(
        repeat_directory / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert first_checkpoint["network"].keys() == repeat_checkpoint[
        "network"
    ].keys()
    assert all(
        torch.equal(
            first_checkpoint["network"][name],
            repeat_checkpoint["network"][name],
        )
        for name in first_checkpoint["network"]
    )


def _assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor)
        assert torch.equal(first, second)
        return
    if isinstance(first, dict):
        assert isinstance(second, dict)
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
        return
    if isinstance(first, (list, tuple)):
        assert isinstance(second, type(first))
        assert len(first) == len(second)
        for first_value, second_value in zip(first, second):
            _assert_nested_equal(first_value, second_value)
        return
    assert first == second


def test_quality_candidate_failure_rolls_back_network_and_optimizer(
    config,
    fixed_instance,
    tmp_path,
    monkeypatch,
):
    effective_config = deepcopy(config)
    effective_config["paths"]["result_root"] = str(tmp_path)
    effective_config["training"]["smoke_episodes"] = 4
    effective_config["training"]["smoke_rollout_steps"] = 1
    effective_config["training"]["validation_interval_episodes"] = 1
    effective_config["training"]["smoke_validation_instance_limit"] = 2

    class FakeOnlineDataset:
        def __init__(self, *, episode_count, **kwargs):
            self.episode_count = episode_count

        def __getitem__(self, index):
            instance = replace(
                fixed_instance,
                instance_id=f"rollback_{1_000_000 + index}",
            )
            return GeneratedInstanceRecord(
                instance=instance,
                metadata={
                    "seed": 1_000_000 + index,
                    "pressure_type": "balanced",
                    "cost_profile": "balanced_cost",
                },
            )

    monkeypatch.setattr(
        training_module,
        "OnlineInstanceDataset",
        FakeOnlineDataset,
    )
    completion_rates = iter((1.0, 1.0, 1.0, 0.5))

    def fake_evaluate_dataset(*args, **kwargs):
        completion_rate = next(completion_rates)
        aggregate = _validation_aggregate(10.0)
        aggregate["completion_rate"] = completion_rate
        aggregate["completed_count"] = int(2 * completion_rate)
        aggregate["truncated_count"] = 2 - aggregate["completed_count"]
        return [], [], [], aggregate

    monkeypatch.setattr(
        training_module,
        "evaluate_dataset",
        fake_evaluate_dataset,
    )
    run_directory = training_module.train(
        effective_config,
        smoke=True,
        online_instances=True,
        run_name="quality_rollback_test",
        parallel_envs=1,
    )
    phase1 = torch.load(
        run_directory / "phase1_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    final = torch.load(
        run_directory / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    _assert_nested_equal(phase1["network"], final["network"])
    _assert_nested_equal(phase1["optimizer"], final["optimizer"])
    summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["formal_training_status"] == "quality_constrained"
    assert summary["training_phase"]["phase_transition_episode"] == 3
    assert summary["training_phase"]["accepted_quality_updates"] == 0
    assert summary["training_phase"]["rejected_quality_updates"] == 1
    assert summary["last_update"]["candidate_status"] == (
        "rejected_rolled_back"
    )


def test_unreached_feasibility_gate_writes_only_provisional_checkpoint(
    config,
    fixed_instance,
    tmp_path,
    monkeypatch,
):
    effective_config = deepcopy(config)
    effective_config["paths"]["result_root"] = str(tmp_path)
    effective_config["training"]["smoke_episodes"] = 1
    effective_config["training"]["smoke_rollout_steps"] = 1
    effective_config["training"]["validation_interval_episodes"] = 1
    effective_config["training"]["smoke_validation_instance_limit"] = 2

    class FakeOnlineDataset:
        def __init__(self, **kwargs):
            pass

        def __getitem__(self, index):
            return GeneratedInstanceRecord(
                instance=replace(
                    fixed_instance,
                    instance_id="unreached_gate",
                ),
                metadata={
                    "seed": 1_000_000,
                    "pressure_type": "balanced",
                    "cost_profile": "balanced_cost",
                },
            )

    monkeypatch.setattr(
        training_module,
        "OnlineInstanceDataset",
        FakeOnlineDataset,
    )

    def fake_evaluate_dataset(*args, **kwargs):
        aggregate = _validation_aggregate(1000.0)
        aggregate["completion_rate"] = 0.5
        aggregate["completed_count"] = 1
        aggregate["truncated_count"] = 1
        return [], [], [], aggregate

    monkeypatch.setattr(
        training_module,
        "evaluate_dataset",
        fake_evaluate_dataset,
    )
    run_directory = training_module.train(
        effective_config,
        smoke=True,
        online_instances=True,
        run_name="unreached_gate_test",
        parallel_envs=1,
    )
    assert not (run_directory / "checkpoint.pt").exists()
    assert not (run_directory / "best_checkpoint.pt").exists()
    assert (run_directory / "last_candidate_checkpoint.pt").exists()
    summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["checkpoint"] is None
    assert summary["best_checkpoint"] is None
    assert summary["formal_training_status"] == "feasibility_not_reached"
