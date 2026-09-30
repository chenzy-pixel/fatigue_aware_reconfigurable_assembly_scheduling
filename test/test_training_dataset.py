from __future__ import annotations

from copy import deepcopy
import csv

import pytest

import train as training_module
from configs import load_config


def test_training_engine_requires_online_instances(config):
    with pytest.raises(ValueError, match="online instance"):
        training_module.train(config, online_instances=False)


@pytest.mark.parametrize("worker_count", [1, 2])
def test_all_worker_counts_use_the_shared_training_engine(config, worker_count):
    engine = training_module.TrainingEngine(
        deepcopy(config), smoke=True, worker_count=worker_count
    )
    assert engine.worker_count == worker_count
    assert callable(engine.run)


def test_train_entrypoint_forwards_to_training_engine(config, monkeypatch, tmp_path):
    expected = tmp_path / "run"
    calls = []

    def fake_run(self):
        calls.append(self)
        return expected

    monkeypatch.setattr(training_module.TrainingEngine, "run", fake_run)
    result = training_module.train(
        deepcopy(config),
        smoke=True,
        online_instances=True,
        run_name="latest_only",
        parallel_envs=1,
    )
    assert result == expected
    assert len(calls) == 1
    assert calls[0].worker_count == 1


def test_unspecified_update_budget_keeps_previous_worker_cadence(config, monkeypatch, tmp_path):
    captured = {}

    def fake_train(effective, **kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(training_module, "_train_single_stage", fake_train)
    result = training_module.TrainingEngine(
        deepcopy(config), smoke=True, worker_count=2,
    ).run()
    assert result == tmp_path
    assert captured["parallel_envs"] == 2
    assert captured["episodes_per_update"] == 2


def test_update_budget_and_validation_thresholds_with_refilled_workers(tmp_path):
    effective = load_config("configs/e1/single_flow.json")
    effective["device"] = "cpu"
    effective["paths"]["result_root"] = str(tmp_path)
    effective["training"].update({
        "smoke_episodes": 5,
        "smoke_rollout_steps": 2,
        "smoke_validation_instance_limit": 1,
        "validation_interval_episodes": 4,
        "torch_num_threads": 2,
        "validation_parallel_envs": 2,
    })
    effective["training"]["formal_evaluation"]["validation_repeats"] = 1
    effective["training"]["formal_evaluation"]["final_test_repeats"] = 1
    run_directory = training_module.train(
        effective, smoke=True, run_name="update_budget_smoke",
        parallel_envs=2, episodes_per_update=4, validation_parallel_envs=2,
    )
    with (run_directory / "update_log.csv").open(encoding="utf-8-sig", newline="") as handle:
        updates = list(csv.DictReader(handle))
    with (run_directory / "validation_log.csv").open(encoding="utf-8-sig", newline="") as handle:
        validations = list(csv.DictReader(handle))
    assert [int(row["episode_count"]) for row in updates] == [4, 1]
    assert [int(row["episode"]) for row in validations] == [4, 5]
