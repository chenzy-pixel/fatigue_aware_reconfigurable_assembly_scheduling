from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

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

    quality_score = (
        0.5 * flow_objective / (1200.0 + flow_objective)
        + 0.3 * 10.0 / 1010.0
        + 0.2 * 2.0 / 52.0
    )
    return {
        "evaluation_schema_version": "3.0.0",
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
            "quality_score": metric(quality_score),
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


def test_ablation_gate_windows_after_filtering_reconfiguration_rows(config):
    effective_config = deepcopy(config)
    effective_config["training"]["ablation_variant"] = "E3"
    gate_config = effective_config["training"]["ablation_gate"]
    gate_config.pop("training_window_episodes", None)
    gate_config["training_window_instances"] = 2

    def training_row(pressure_type: str, completed: bool) -> dict:
        return {
            "pressure_type": pressure_type,
            "terminated": completed,
            "truncated": not completed,
            "reward_identity_error": 0.0,
            "schedule_violation_count": 0,
        }

    rows = [
        training_row("reconfiguration_bottleneck", True),
        training_row("reconfiguration_bottleneck", False),
        *[training_row("balanced", True) for _ in range(10)],
        training_row("reconfiguration_bottleneck", True),
        training_row("balanced", True),
    ]
    validation_rows = [
        {"completion_rate": 1.0, "schedule_violation_count": 0}
    ]
    failure_instance_rows = [
        {
            "instance_id": gate_config["failure_instance_id"],
            "terminated": True,
            "truncated": False,
            "machine_waiting_for_worker_time": 0.0,
        }
    ]
    stability_controller = SimpleNamespace(
        feasibility_rollbacks=0,
        validation_count=1,
        current_learning_rate=1e-4,
    )

    summary = training_module._ablation_gate_summary(
        effective_config,
        rows,
        validation_rows,
        stability_controller,
        failure_instance_rows,
    )

    assert summary is not None
    assert summary["reconfiguration_training_requested_sample_count"] == 2
    assert summary["reconfiguration_training_available_sample_count"] == 3
    assert summary["reconfiguration_training_sample_count"] == 2
    assert summary["reconfiguration_training_completion_rate"] == pytest.approx(
        0.5
    )
    assert not summary["checks"]["reconfiguration_training_completion"]

    gate_config["training_window_episodes"] = gate_config.pop(
        "training_window_instances"
    )
    legacy_summary = training_module._ablation_gate_summary(
        effective_config,
        rows,
        validation_rows,
        stability_controller,
        failure_instance_rows,
    )
    assert legacy_summary is not None
    assert legacy_summary["reconfiguration_training_sample_count"] == 2
    assert legacy_summary[
        "reconfiguration_training_completion_rate"
    ] == pytest.approx(0.5)


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
    sampled_validation_calls = []

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
        if kwargs.get("decode_mode", "greedy") == "sampled":
            sampled_validation_calls.append(kwargs["sampling_seed"])
            return [], [], [], _validation_aggregate(15.0)
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

    assert len(validation_calls) == 3
    sampled_settings = effective_config["training"][
        "validation_control"
    ]["sampled"]
    expected_sampled_seeds = [
        int(effective_config["seed"])
        + int(sampled_settings["seed_offset"])
        + repeat
        for repeat in range(int(sampled_settings["repeats"]))
    ]
    assert sampled_validation_calls == (
        expected_sampled_seeds + [100011, 100012, 100013]
    )
    assert (run_directory / "checkpoint.pt").exists()
    assert (run_directory / "best_checkpoint.pt").exists()
    assert (run_directory / "best_feasibility_checkpoint.pt").exists()
    assert (run_directory / "train_log.csv").exists()
    assert (run_directory / "validation_log.csv").exists()
    summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["episodes"] == 11
    assert summary["unique_instance_count"] == 11
    assert summary["validation_runs"] == 2
    assert summary["best_validation"]["episode"] == 10
    checkpoint_hash = hashlib.sha256(
        (run_directory / "checkpoint.pt").read_bytes()
    ).hexdigest()
    assert checkpoint_hash == summary["checkpoint_sha256"]
    assert checkpoint_hash == summary["final_checkpoint_evaluation"][
        "checkpoint_sha256"
    ]
    assert (run_directory / "checkpoint.pt").read_bytes() == (
        run_directory / "accepted_checkpoint.pt"
    ).read_bytes()
    assert (run_directory / "checkpoint.pt").read_bytes() == (
        run_directory / "best_checkpoint.pt"
    ).read_bytes()

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
    assert validation_rows[0]["sampled_completion_rate"] == ""
    assert validation_rows[1]["sampled_completion_rate"] != ""
    assert summary["sampled_validation_runs"] == 1

    best = torch.load(
        run_directory / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert best["metadata"]["accepted_episode"] == 10
    assert best["metadata"]["checkpoint_role"] == "shadow_best"


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
        **kwargs,
    ):
        assert dataset_name == "validation"
        assert ppo_agent is not None
        assert runner.worker_count == 2
        assert instance_limit == 2
        if kwargs.get("decode_mode", "greedy") == "sampled":
            return [], _validation_aggregate(15.0)
        validation_calls.append(1)
        objective = 10.0 if len(validation_calls) == 1 else 20.0
        return [], _validation_aggregate(objective)

    monkeypatch.setattr(
        training_module,
        "evaluate_dataset_parallel",
        fake_parallel_validation,
    )
    monkeypatch.setattr(
        training_module,
        "_reevaluate_checkpoint_from_disk",
        lambda *args, **kwargs: {"source": "isolated_disk_reload"},
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
    assert (run_directory / "best_feasibility_checkpoint.pt").exists()
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


def test_three_not_promoted_updates_keep_network_and_optimizer_advancing(
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
    effective_config["training"]["two_stage"][
        "consecutive_validations"
    ] = 1

    class FakeOnlineDataset:
        def __init__(self, **kwargs):
            pass

        def __getitem__(self, index):
            return GeneratedInstanceRecord(
                instance=replace(
                    fixed_instance,
                    instance_id=f"continuous_{index}",
                ),
                metadata={
                    "seed": 1_000_000 + index,
                    "pressure_type": "balanced",
                    "cost_profile": "balanced_cost",
                },
            )

    monkeypatch.setattr(
        training_module, "OnlineInstanceDataset", FakeOnlineDataset
    )
    greedy_calls = 0

    def fake_evaluate_dataset(*args, **kwargs):
        nonlocal greedy_calls
        if kwargs.get("decode_mode", "greedy") == "sampled":
            return [], [], [], _validation_aggregate(20.0)
        greedy_calls += 1
        return [], [], [], _validation_aggregate(
            10.0 if greedy_calls == 1 else 20.0
        )

    monkeypatch.setattr(
        training_module, "evaluate_dataset", fake_evaluate_dataset
    )
    monkeypatch.setattr(
        training_module,
        "_reevaluate_checkpoint_from_disk",
        lambda *args, **kwargs: {"source": "isolated_disk_reload"},
    )
    run_directory = training_module.train(
        effective_config,
        smoke=True,
        online_instances=True,
        run_name="continuous_not_promoted_test",
        parallel_envs=1,
    )
    accepted = torch.load(
        run_directory / "accepted_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    last = torch.load(
        run_directory / "last_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert any(
        not torch.equal(accepted["network"][name], last["network"][name])
        for name in accepted["network"]
    )

    def maximum_optimizer_step(payload):
        return max(
            int(value["step"].item())
            for value in payload["optimizer"]["state"].values()
            if "step" in value
        )

    assert maximum_optimizer_step(last) > maximum_optimizer_step(accepted)
    summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["training_phase"]["not_promoted_quality_updates"] == 3
    assert summary["training_phase"]["accepted_quality_updates"] == 0
    assert summary["feasibility_rollbacks"] == 0


def test_quality_nonpromotion_keeps_online_network_and_shadow_best(
    config,
    fixed_instance,
    tmp_path,
    monkeypatch,
):
    effective_config = deepcopy(config)
    training_module._apply_ablation_variant(effective_config, "Q13")
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
    completion_rates = iter((1.0, 1.0, 1.0, 0.5, 1.0))

    def fake_evaluate_dataset(*args, **kwargs):
        if kwargs.get("decode_mode", "greedy") == "sampled":
            return [], [], [], _validation_aggregate(10.0)
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
    sampled_results = []
    for flow in (99.0, 10.0, 10.0):
        sampled = _validation_aggregate(flow)
        sampled["decode_mode"] = "sampled"
        sampled["repeat_count"] = 3
        sampled["unique_instance_count"] = 2
        sampled_results.append(sampled)
    sampled_calls = iter(sampled_results)

    def fake_sampled_validation(*args, **kwargs):
        return next(sampled_calls)

    monkeypatch.setattr(
        training_module,
        "_evaluate_sampled_validation",
        fake_sampled_validation,
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
    last = torch.load(
        run_directory / "last_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert any(
        not torch.equal(phase1["network"][name], last["network"][name])
        for name in phase1["network"]
    )
    summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["formal_training_status"] == "quality_constrained"
    assert summary["training_phase"]["phase_transition_episode"] == 3
    assert summary["training_phase"]["accepted_quality_updates"] == 0
    assert summary["training_phase"]["rejected_quality_updates"] == 1
    assert summary["last_update"]["candidate_status"] == "not_promoted"
    assert summary["last_sampled_validation"]["all_instance_metrics"][
        "flow_time_objective"
    ]["mean"] == pytest.approx(99.0)
    assert summary["final_accepted_sampled_validation"][
        "all_instance_metrics"
    ]["flow_time_objective"]["mean"] == pytest.approx(10.0)
    assert summary["final_accepted_sampled_validation_source"] == (
        "rerun_after_rejected"
    )
    assert summary["final_accepted_checkpoint_episode"] == 3


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
    assert (run_directory / "best_feasibility_checkpoint.pt").exists()
    assert (run_directory / "last_candidate_checkpoint.pt").exists()
    summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["checkpoint"] is None
    assert summary["best_checkpoint"] is None
    assert summary["best_feasibility_checkpoint"] is not None
    assert summary["formal_training_status"] == "feasibility_not_reached"


def test_catastrophic_regression_rolls_back_safe_and_keeps_decayed_lr(
    config,
    fixed_instance,
    tmp_path,
    monkeypatch,
):
    effective_config = deepcopy(config)
    effective_config["paths"]["result_root"] = str(tmp_path)
    effective_config["training"]["smoke_episodes"] = 3
    effective_config["training"]["smoke_rollout_steps"] = 1
    effective_config["training"]["validation_interval_episodes"] = 1
    effective_config["training"]["smoke_validation_instance_limit"] = 2
    effective_config["training"]["validation_control"][
        "learning_rate_plateau"
    ]["patience_validations"] = 1

    class FakeOnlineDataset:
        def __init__(self, **kwargs):
            pass

        def __getitem__(self, index):
            return GeneratedInstanceRecord(
                instance=replace(
                    fixed_instance,
                    instance_id=f"feasibility_rollback_{index}",
                ),
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
    greedy_rates = iter((1.0, 0.9, 0.9))

    def fake_evaluate_dataset(*args, **kwargs):
        aggregate = _validation_aggregate(10.0)
        if kwargs.get("decode_mode", "greedy") == "sampled":
            return [], [], [], aggregate
        completion_rate = next(greedy_rates)
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
        run_name="feasibility_rollback_test",
        parallel_envs=1,
    )

    best = torch.load(
        run_directory / "best_feasibility_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    final = torch.load(
        run_directory / "last_candidate_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    _assert_nested_equal(best["network"], final["network"])
    _assert_nested_equal(
        best["optimizer"]["state"],
        final["optimizer"]["state"],
    )
    best_groups = deepcopy(best["optimizer"]["param_groups"])
    final_groups = deepcopy(final["optimizer"]["param_groups"])
    best_learning_rates = [group.pop("lr") for group in best_groups]
    final_learning_rates = [group.pop("lr") for group in final_groups]
    _assert_nested_equal(best_groups, final_groups)
    assert best_learning_rates == pytest.approx([1e-4])
    assert final_learning_rates == pytest.approx([5e-5])

    summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["best_feasibility_validation"]["episode"] == 1
    assert summary["feasibility_rollbacks"] == 1
    assert summary["learning_rate_decays"] == 1
    assert summary["last_update"]["candidate_status"] == (
        "catastrophic_rolled_back"
    )
