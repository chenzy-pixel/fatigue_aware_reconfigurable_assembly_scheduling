from __future__ import annotations

from copy import deepcopy

import pytest

import train as training_module


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
