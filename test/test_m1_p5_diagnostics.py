from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from data import canonical_json_bytes, sha256_bytes
from m1_experiments import (
    M1_METHOD_VERSION,
    _load_ranking_cache,
    _overfit_checks,
    _p5_failure_diagnostics,
    _p5_instance_snapshot,
    _p5_promotion_decision,
    _ranking_acceptance_checks,
    _save_p5_accepted_checkpoint,
    _spearman,
)


def _update(
    index: int,
    *,
    completion: float,
    variance: float,
    quality: float,
) -> dict:
    checks = {
        "completion_rate": completion == 1.0,
        "zero_schedule_violations": True,
        "flow_gap": True,
        "reconfiguration_cost": True,
        "worker_load_variance": variance <= 3.0,
        "quality_score": quality <= 0.2,
    }
    return {
        "stage_update": index,
        "global_update": index,
        "reward_phase": "quality",
        "aggregate": {
            "completion_rate": completion,
            "mean_flow_gap_percent": float(index),
            "mean_reconfiguration_cost": 10.0 + index,
            "mean_heuristic_reconfiguration_cost": 20.0,
            "mean_worker_load_variance": variance,
            "mean_heuristic_worker_load_variance": 3.0,
            "mean_quality_score": quality,
            "mean_heuristic_quality_score": 0.2,
        },
        "checks": checks,
        "stability": {"rollback": index == 2},
        "losses": {"loss": float(index)},
        "instance_rows": [{"instance_id": f"instance_{index}"}],
    }


def test_p5_failure_diagnostics_summarizes_hard_gate_failure() -> None:
    rows = [
        _update(1, completion=1.0, variance=9.0, quality=0.3),
        _update(2, completion=0.8, variance=6.0, quality=0.4),
        _update(3, completion=1.0, variance=12.0, quality=0.35),
    ]

    diagnostics = _p5_failure_diagnostics(rows)

    assert diagnostics["updates"] == 3
    assert diagnostics["never_passed_checks"] == [
        "worker_load_variance",
        "quality_score",
    ]
    assert diagnostics["catastrophic_completion_updates"] == [2]
    assert diagnostics["recorded_rollback_updates"] == [2]
    assert diagnostics["best_quality"]["stage_update"] == 1
    assert diagnostics["best_variance"]["stage_update"] == 2
    assert diagnostics["tradeoff"][
        "best_variance_to_heuristic_ratio"
    ] == pytest.approx(2.0)
    assert diagnostics["tradeoff"]["best_quality_gap_percent"] == pytest.approx(
        50.0
    )


def test_overfit_checks_use_configured_gap_thresholds(config) -> None:
    aggregate = {
        "completion_rate": 1.0,
        "schedule_violation_count": 0,
        "mean_flow_gap_percent": 4.5,
        "mean_reconfiguration_cost_gap_percent": 4.5,
        "mean_worker_load_variance_gap_percent": 4.5,
        "mean_quality_gap_percent": -0.1,
    }
    thresholds = deepcopy(config["m1_gates"]["thresholds"])

    assert all(_overfit_checks(aggregate, thresholds).values())
    thresholds["maximum_variance_gap_percent"] = 4.0

    checks = _overfit_checks(aggregate, thresholds)
    assert not checks["worker_load_variance_gap"]
    assert all(
        value
        for name, value in checks.items()
        if name != "worker_load_variance_gap"
    )


def test_worker_variance_regret_is_a_hard_ranking_gate(config) -> None:
    thresholds = config["m1_gates"]["thresholds"]
    metrics = {
        "production": {
            "mean_spearman": 1.0,
            "mean_regret": 0.0,
        },
        "worker": {
            "mean_spearman": 1.0,
            "mean_regret": 0.0,
            "mean_variance_regret_percent": (
                thresholds["maximum_worker_variance_regret_percent"] + 0.1
            ),
        },
    }

    checks = _ranking_acceptance_checks(metrics, thresholds)

    assert not checks["worker_variance_regret"]
    assert all(
        value
        for name, value in checks.items()
        if name != "worker_variance_regret"
    )


def test_rankable_teacher_with_tied_model_counts_as_zero_spearman() -> None:
    assert _spearman(
        np.asarray([0.0, 0.0]),
        ((0.0,), (1.0,)),
    ) == pytest.approx(0.0)


def test_non_improving_candidate_does_not_overwrite_accepted(
    tmp_path,
) -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self.payload = b"accepted-v1"

        def save(self, path, metadata=None) -> None:
            del metadata
            path.write_bytes(self.payload)

    agent = FakeAgent()
    accepted = tmp_path / "stage" / "accepted_checkpoint.pt"
    accepted.parent.mkdir()
    best = tmp_path / "stage" / "best_checkpoint.pt"
    first = _p5_promotion_decision(
        {"mean_quality_score": 0.20},
        {"completion_rate": True, "quality_gap": False},
        None,
    )
    assert _save_p5_accepted_checkpoint(
        agent,
        accepted_checkpoint=accepted,
        best_checkpoint=best,
        root_directory=tmp_path,
        promotion=first,
        metadata={},
    )
    original = accepted.read_bytes()

    agent.payload = b"worse-candidate"
    worse = _p5_promotion_decision(
        {"mean_quality_score": 0.21},
        {"completion_rate": True, "quality_gap": False},
        0.20,
    )
    assert not _save_p5_accepted_checkpoint(
        agent,
        accepted_checkpoint=accepted,
        best_checkpoint=best,
        root_directory=tmp_path,
        promotion=worse,
        metadata={},
    )
    assert accepted.read_bytes() == original
    assert best.read_bytes() == original


def test_p5_instance_snapshot_reloads_exact_records_and_rejects_v3(
    config,
    instance_generator,
    tmp_path,
) -> None:
    seed = int(config["m1_gates"]["diagnostic_seed_start"])
    record = instance_generator.generate(
        seed=seed,
        split="train",
        pressure_type="easy",
    )
    first = _p5_instance_snapshot(
        tmp_path,
        config,
        [seed],
        [record],
    )
    second = _p5_instance_snapshot(tmp_path, config, [seed])
    expected_hash = sha256_bytes(canonical_json_bytes(first[0].to_dict()))
    actual_hash = sha256_bytes(canonical_json_bytes(second[0].to_dict()))
    assert actual_hash == expected_hash

    manifest_path = tmp_path / "instances" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["method_version"] == M1_METHOD_VERSION
    assert manifest["records"][0]["sha256"] == expected_hash
    manifest["method_version"] = "M1_candidate_graph_v3"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="v3 caches cannot be reused"):
        _p5_instance_snapshot(tmp_path, config, [seed])


def test_v3_ranking_cache_is_rejected(tmp_path) -> None:
    cache = tmp_path / "ranking_states.pt"
    torch.save(
        {
            "method_version": "M1_candidate_graph_v3",
            "policy_head_version": 3,
            "seeds": [1],
            "training_count": 1,
            "maximum_pairs": 2,
            "training_states": [],
            "held_out_states": [],
        },
        cache,
    )

    with pytest.raises(ValueError, match="v3 ranking caches cannot be loaded"):
        _load_ranking_cache(
            cache,
            seeds=[1],
            training_count=1,
            maximum_pairs=2,
        )
