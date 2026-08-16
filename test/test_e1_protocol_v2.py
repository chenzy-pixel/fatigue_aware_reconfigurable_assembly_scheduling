from __future__ import annotations

from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from environment import bounded_quality_score
from result.metrics import (
    CANONICAL_QUALITY_METRIC,
    aggregate_evaluation_rows,
    evaluation_quality_metric,
    quality_metric_sha256,
)
from result.provenance import (
    build_provenance,
    dataset_manifest_snapshot,
    effective_config_snapshot,
    network_weights_sha256,
    source_state_snapshot,
)
from utils import derive_evaluation_sampling_seed


def test_canonical_quality_is_independent_of_reward_variance_scale(config):
    reward_50 = deepcopy(config["reward"])
    reward_20 = deepcopy(reward_50)
    reward_20["variance_scale"] = 20.0
    metric = evaluation_quality_metric(config)
    objectives = (900.0, 400.0, 12.0)

    canonical_50 = bounded_quality_score(*objectives, metric)
    canonical_20 = bounded_quality_score(*objectives, metric)
    assert canonical_20 == canonical_50
    assert bounded_quality_score(*objectives, reward_20) != pytest.approx(
        bounded_quality_score(*objectives, reward_50)
    )


def test_legacy_quality_fallback_and_invalid_definitions(config):
    legacy = deepcopy(config)
    legacy.pop("evaluation")
    assert evaluation_quality_metric(legacy) == CANONICAL_QUALITY_METRIC

    invalid = deepcopy(config)
    invalid["evaluation"]["quality_metric"]["variance_scale"] = 20.0
    with pytest.raises(ValueError, match="immutable"):
        evaluation_quality_metric(invalid)

    invalid = deepcopy(config)
    invalid["evaluation"]["quality_metric"]["quality_weights"]["flow"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        evaluation_quality_metric(invalid)


def test_aggregate_rejects_mixed_quality_metric_hashes():
    metric_hash = quality_metric_sha256(CANONICAL_QUALITY_METRIC)
    base = {
        "terminated": True,
        "truncated": False,
        "makespan": 1.0,
        "total_flow_time": 1.0,
        "flow_time_objective": 1.0,
        "reconfiguration_cost": 1.0,
        "worker_load_variance": 1.0,
        "inference_time_seconds": 0.0,
        "solve_time_seconds": 0.0,
        "inference_time_per_decision_ms": 0.0,
        "relative_heuristic_gap_percent": 0.0,
        "makespan_heuristic_gap_percent": 0.0,
        "reconfiguration_cost_heuristic_gap_percent": 0.0,
        "worker_load_variance_heuristic_gap_percent": 0.0,
        "schedule_violation_count": 0,
        "decisions": 1,
        "quality_metric_sha256": metric_hash,
    }
    conflicting = dict(base, quality_metric_sha256="0" * 64)
    with pytest.raises(ValueError, match="different quality metrics"):
        aggregate_evaluation_rows(
            [base, conflicting],
            dataset="validation",
            policy="ppo",
            manifest="manifest.json",
        )


def test_sampling_seed_derivation_is_instance_and_seed_specific():
    first = derive_evaluation_sampling_seed(100011, "instance-a")
    assert first == derive_evaluation_sampling_seed(100011, "instance-a")
    assert first != derive_evaluation_sampling_seed(100011, "instance-b")
    assert first != derive_evaluation_sampling_seed(100012, "instance-a")


def test_provenance_hashes_are_canonical_and_scoped(tmp_path):
    for directory in ("agent", "configs", "data", "environment", "result"):
        (tmp_path / directory).mkdir()
    (tmp_path / "train.py").write_bytes(b"print('entry')\r\n")
    metrics = tmp_path / "result" / "metrics.py"
    metrics.write_bytes(b"VALUE = 1\r\n")
    initial = source_state_snapshot(tmp_path)["sha256"]

    metrics.write_bytes(b"VALUE = 1\n")
    assert source_state_snapshot(tmp_path)["sha256"] == initial
    metrics.write_text("VALUE = 2\n", encoding="utf-8")
    changed = source_state_snapshot(tmp_path)["sha256"]
    assert changed != initial

    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "cached.py").write_text(
        "VALUE = 99\n", encoding="utf-8"
    )
    (tmp_path / "result" / "runs").mkdir()
    (tmp_path / "result" / "runs" / "generated.py").write_text(
        "VALUE = 99\n", encoding="utf-8"
    )
    assert source_state_snapshot(tmp_path)["sha256"] == changed

    first_config = {"b": 2, "a": 1, "_config_path": "one"}
    second_config = {"a": 1, "_config_path": "two", "b": 2}
    assert effective_config_snapshot(first_config)["sha256"] == (
        effective_config_snapshot(second_config)["sha256"]
    )
    changed_config = {"a": 1, "b": 3}
    assert effective_config_snapshot(changed_config)["sha256"] != (
        effective_config_snapshot(first_config)["sha256"]
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"b": 2, "a": 1}), encoding="utf-8")
    manifest_hash = dataset_manifest_snapshot(manifest)["sha256"]
    manifest.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
    assert dataset_manifest_snapshot(manifest)["sha256"] == manifest_hash
    manifest.write_text(json.dumps({"a": 1, "b": 3}), encoding="utf-8")
    assert dataset_manifest_snapshot(manifest)["sha256"] != manifest_hash


def test_local_sampling_helpers_preserve_global_rng():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(
        derive_evaluation_sampling_seed(100011, "instance-a")
    )
    torch.rand(8, generator=generator)
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_network_weights_hash_tracks_tensor_content_not_mapping_order():
    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    reordered = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    changed = {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}
    assert network_weights_sha256(first) == network_weights_sha256(reordered)
    assert network_weights_sha256(first) != network_weights_sha256(changed)


def test_provenance_falls_back_to_checkpoint_network_hash(config, tmp_path):
    state_dict = {
        "b": torch.tensor([2.0]),
        "a": torch.tensor([1.0]),
    }
    checkpoint = tmp_path / "legacy_checkpoint.pt"
    torch.save(
        {
            "network": state_dict,
            "metadata": {"experiment_suite_version": "legacy"},
        },
        checkpoint,
    )
    expected = network_weights_sha256(state_dict)

    provenance = build_provenance(
        config,
        checkpoint_path=checkpoint,
        checkpoint_metadata={"experiment_suite_version": "legacy"},
        root=tmp_path,
    )
    assert provenance["network_weights_sha256"] == expected

    with pytest.raises(ValueError, match="does not match"):
        build_provenance(
            config,
            checkpoint_path=checkpoint,
            checkpoint_metadata={
                "experiment_suite_version": "legacy",
                "network_weights_sha256": "0" * 64,
            },
            root=tmp_path,
        )
