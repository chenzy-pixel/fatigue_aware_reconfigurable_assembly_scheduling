from __future__ import annotations

import csv
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

import eval as evaluation_module
from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent, TypedActorCritic
from data.dataset import load_dataset_split
from environment import AssemblySchedulingEnv
from result.metrics import (
    aggregate_evaluation_rows,
    relative_gap_percent,
    summarize_values,
)
from result.io import write_config, write_csv, write_json


def _aggregate_row(
    *,
    terminated: bool,
    truncated: bool,
    makespan: float,
    total_flow_time: float | None,
    flow_time_objective: float,
) -> dict:
    return {
        "terminated": terminated,
        "truncated": truncated,
        "makespan": makespan,
        "total_flow_time": total_flow_time,
        "flow_time_objective": flow_time_objective,
        "reconfiguration_cost": 10.0,
        "worker_load_variance": 2.0,
        "inference_time_seconds": 0.2,
        "solve_time_seconds": 0.5,
        "inference_time_per_decision_ms": 2.0,
        "relative_heuristic_gap_percent": 5.0,
        "makespan_heuristic_gap_percent": 4.0,
        "reconfiguration_cost_heuristic_gap_percent": 3.0,
        "worker_load_variance_heuristic_gap_percent": 2.0,
        "maximum_worker_fatigue": 0.75,
        "mean_peak_worker_fatigue": 0.5,
        "safe_fatigue_limit": 0.9,
        "fatigue_masked_action_ratio": 0.1,
        "worker_competition_event_count": 3,
        "machine_waiting_for_worker_time": 4.0,
        "completed_reconfigurations": 5,
        "worker_switch_ratio": 0.2,
        "schedule_violation_count": 0,
        "decisions": 100,
    }


def test_gap_and_sample_statistics_contract():
    assert relative_gap_percent(90.0, 100.0) == pytest.approx(-10.0)
    assert relative_gap_percent(110.0, 100.0) == pytest.approx(10.0)
    assert relative_gap_percent(10.0, 0.0) is None
    assert relative_gap_percent(None, 10.0) is None

    summary = summarize_values([1.0, 3.0, None])
    assert summary == {
        "count": 2,
        "mean": 2.0,
        "std": pytest.approx(math.sqrt(2.0)),
    }
    assert summarize_values([2.0]) == {
        "count": 1,
        "mean": 2.0,
        "std": 0.0,
    }


def test_aggregate_uses_completed_and_all_instance_populations():
    rows = [
        _aggregate_row(
            terminated=True,
            truncated=False,
            makespan=100.0,
            total_flow_time=500.0,
            flow_time_objective=500.0,
        ),
        _aggregate_row(
            terminated=False,
            truncated=True,
            makespan=240.0,
            total_flow_time=None,
            flow_time_objective=1_000.0,
        ),
    ]
    aggregate = aggregate_evaluation_rows(
        rows,
        dataset="validation",
        policy="ppo",
        manifest="manifest.json",
    )
    assert aggregate["evaluation_schema_version"] == "3.0.0"
    assert aggregate["completed_count"] == 1
    assert aggregate["completion_rate"] == 0.5
    assert aggregate["truncated_count"] == 1
    assert aggregate["completed_metrics"]["makespan"] == {
        "count": 1,
        "mean": 100.0,
        "std": 0.0,
    }
    assert aggregate["completed_metrics"]["total_flow_time"]["count"] == 1
    objective = aggregate["all_instance_metrics"]["flow_time_objective"]
    assert objective["count"] == 2
    assert objective["mean"] == pytest.approx(750.0)
    assert objective["std"] == pytest.approx(math.sqrt(125_000.0))
    fatigue = aggregate["all_instance_metrics"][
        "maximum_worker_fatigue"
    ]
    assert fatigue == {"count": 2, "mean": 0.75, "std": 0.0}


def test_fixed_validation_evaluation_is_read_only_and_reports_zero_gap(
    config,
    tmp_path,
):
    dataset = load_dataset_split(config, "validation")
    manifest_bytes = dataset.manifest_path.read_bytes()
    instance_paths = [
        Path(config["paths"]["instances_root"])
        / "validation"
        / dataset.manifest["files"][index]["path"]
        for index in range(2)
    ]
    before = [path.read_bytes() for path in instance_paths]

    rows, schedules, reconfigurations, aggregate = (
        evaluation_module.evaluate_dataset(
            config,
            dataset_name="validation",
            policy_name="heuristic",
            instance_limit=2,
        )
    )

    assert dataset.manifest_path.read_bytes() == manifest_bytes
    assert [path.read_bytes() for path in instance_paths] == before
    assert len(rows) == 2
    assert schedules
    assert reconfigurations
    assert aggregate["instance_count"] == 2
    assert aggregate["completion_rate"] == 1.0
    assert aggregate["truncated_count"] == 0
    for row in rows:
        assert row["relative_heuristic_gap_percent"] == pytest.approx(
            0.0, abs=1e-9
        )
        assert row["makespan_heuristic_gap_percent"] == pytest.approx(
            0.0, abs=1e-9
        )
        assert row["inference_time_seconds"] >= 0.0
        assert row["solve_time_seconds"] >= row[
            "inference_time_seconds"
        ]
        assert row["inference_time_per_decision_ms"] >= 0.0

    write_config(tmp_path, config)
    write_json(tmp_path / "metrics.json", aggregate)
    write_csv(tmp_path / "instance_metrics.csv", rows)
    write_csv(tmp_path / "schedule.csv", schedules)
    write_csv(tmp_path / "reconfigurations.csv", reconfigurations)
    assert {
        "config.json",
        "metrics.json",
        "instance_metrics.csv",
        "schedule.csv",
        "reconfigurations.csv",
    } == {path.name for path in tmp_path.iterdir()}
    saved_metrics = json.loads(
        (tmp_path / "metrics.json").read_text(encoding="utf-8")
    )
    assert saved_metrics["evaluation_schema_version"] == "3.0.0"
    with (tmp_path / "instance_metrics.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        columns = set(next(csv.DictReader(handle)).keys())
    assert {
        "instance_id",
        "terminated",
        "truncated",
        "termination_reason",
        "decisions",
        "makespan",
        "total_flow_time",
        "flow_time_objective",
        "reconfiguration_cost",
        "worker_load_variance",
        "inference_time_seconds",
        "solve_time_seconds",
        "inference_time_per_decision_ms",
        "heuristic_flow_time",
        "relative_heuristic_gap_percent",
    } <= columns


def test_representative_diagnostic_preserves_rng_and_training_mode(
    config,
):
    record = load_dataset_split(config, "validation")[0]
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(record.instance)
    network = TypedActorCritic(
        observation.feature_dimensions,
        config["network"]["hidden_dim"],
    )
    agent = PPOAgent(network, config["ppo"], device="cpu")
    network.train()
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    trace = evaluation_module.evaluate_representative_diagnostic(
        config,
        dataset_name="validation",
        ppo_agent=agent,
        instance_index=0,
    )

    assert network.training is True
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert trace["instance_id"] == record.instance.instance_id
    assert trace["schedule"]
    assert trace["fatigue_trace"]
    assert trace["worker_ids"]
    assert all(
        before["time"] <= after["time"]
        for before, after in zip(
            trace["fatigue_trace"],
            trace["fatigue_trace"][1:],
        )
    )


def test_ppo_checkpoint_loads_once_and_validation_preserves_rng(
    config,
    tmp_path,
    monkeypatch,
):
    effective_config = deepcopy(config)
    effective_config["network"] = {
        "encoder_type": "typed_mlp",
        "hidden_dim": int(config["network"]["hidden_dim"]),
    }
    dataset = load_dataset_split(effective_config, "validation")
    environment = AssemblySchedulingEnv(effective_config)
    observation = environment.reset(dataset[0].instance)
    network = TypedActorCritic(
        observation.feature_dimensions,
        int(config["network"]["hidden_dim"]),
    )
    agent = PPOAgent(network, effective_config["ppo"], device="cpu")
    checkpoint = tmp_path / "checkpoint.pt"
    agent.save(checkpoint)

    load_count = 0
    original_load = PPOAgent.load

    def counting_load(self, path, *, load_optimizer=False):
        nonlocal load_count
        load_count += 1
        return original_load(
            self,
            path,
            load_optimizer=load_optimizer,
        )

    monkeypatch.setattr(PPOAgent, "load", counting_load)

    def fast_select(self, observation, environment):
        return HeuristicPolicy().select_action(environment)

    monkeypatch.setattr(
        evaluation_module.EvaluationPolicy,
        "select_action",
        fast_select,
    )
    evaluation_module.evaluate_dataset(
        effective_config,
        dataset_name="validation",
        policy_name="ppo",
        checkpoint=str(checkpoint),
        instance_limit=2,
    )
    assert load_count == 1

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    agent.network.train()
    evaluation_module.evaluate_dataset(
        effective_config,
        dataset_name="validation",
        policy_name="ppo",
        ppo_agent=agent,
        instance_limit=1,
    )
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert agent.network.training


def test_eval_cli_requires_dataset(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eval.py"])
    with pytest.raises(SystemExit) as error:
        evaluation_module.main()
    assert error.value.code == 2
