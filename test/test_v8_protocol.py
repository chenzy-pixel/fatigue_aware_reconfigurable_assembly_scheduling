from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import pytest

from configs import load_config, validate_latest_only_config
from configs.normalization import (
    apply_normalization_manifest,
    build_normalization_manifest,
    file_sha256,
    load_normalization_manifest,
    write_immutable_manifest,
)
from environment import PreferenceContext, quality_preference_for_episode, simplex_lattice
from train import _aggregate_formal_rows, _checkpoint_metadata


@pytest.mark.parametrize(
    ("path", "preference"),
    (
        ("configs/v8/specialist_flow.json", [1.0, 0.0, 0.0]),
        ("configs/v8/specialist_cost.json", [0.0, 1.0, 0.0]),
        ("configs/v8/specialist_variance.json", [0.0, 0.0, 1.0]),
        ("configs/e1/single_flow.json", [1.0, 0.0, 0.0]),
        ("configs/e1/single_cost.json", [0.0, 1.0, 0.0]),
        ("configs/e1/single_variance.json", [0.0, 0.0, 1.0]),
    ),
)
def test_single_objective_configs_use_one_quality_preference_from_episode_zero(
    path: str,
    preference: list[float],
):
    config = load_config(path)
    assert "feasibility" not in config["preference"]
    assert config["preference"]["quality"]["fixed"] == preference
    assert quality_preference_for_episode(
        config, algorithm_seed=11, quality_episode_index=0
    ).as_tuple() == tuple(preference)
    assert "two_stage" not in config["training"]


def test_universal_keeps_fixed_66_point_grid_and_training_sequence():
    config = load_config("configs/default.json")
    assert len(simplex_lattice(10, include=())) == 66
    points = [
        quality_preference_for_episode(
            config, algorithm_seed=11, quality_episode_index=index
        ).as_tuple()
        for index in range(20)
    ]
    assert points[:6] == [
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    ]


def test_single_stage_rejects_legacy_reward_weights_and_nonunit_gamma():
    config = deepcopy(load_config("configs/default.json"))
    config.pop("runtime_manifest")
    config["experiment_suite_version"] = "single_stage_progress_quality_v1"
    with pytest.raises(ValueError, match="experiment_suite_version"):
        validate_latest_only_config(config)
    config["experiment_suite_version"] = "single_stage_progress_quality_failure_v2"
    config["reward"]["quality_weights"] = {
        "flow": 1.0,
        "cost": 0.0,
        "variance": 0.0,
    }
    with pytest.raises(ValueError, match="reward.quality_weights"):
        validate_latest_only_config(config)
    config["reward"].pop("quality_weights")
    config["reward"]["terminal_failure_penalty"] = 0.5
    with pytest.raises(ValueError, match="terminal_failure_penalty"):
        validate_latest_only_config(config)
    config["reward"]["terminal_failure_penalty"] = 1.0
    config["ppo"]["gamma"] = 0.99
    with pytest.raises(ValueError, match="gamma"):
        validate_latest_only_config(config)


def test_normalization_manifest_round_trip_stays_outside_training_protocol(tmp_path: Path):
    validation = tmp_path / "validation_manifest.json"
    validation.write_text("{}\n", encoding="utf-8")
    validation_sha = file_sha256(validation)
    rows = []
    for objective_index, objective in enumerate(("flow", "cost", "variance")):
        for seed_index, seed in enumerate((11, 23, 37, 53, 71)):
            checkpoint = tmp_path / f"{objective}_{seed}.pt"
            checkpoint.write_bytes(f"{objective}:{seed}".encode())
            rows.append(
                {
                    "objective": objective,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "raw_objective_mean": 10.0 * (objective_index + 1) + seed_index,
                    "validation_dataset_sha256": validation_sha,
                    "validation_instance_offset": 0,
                    "validation_instance_count": 50,
                }
            )
    manifest = build_normalization_manifest(
        rows, validation_dataset_path=validation
    )
    destination = tmp_path / "normalization.json"
    digest = write_immutable_manifest(destination, manifest)
    loaded = load_normalization_manifest(destination, expected_sha256=digest)
    config = {
        "objective_scalarizer": {
            "scale_source": "frozen_manifest",
            "normalization_manifest": destination.name,
            "normalization_manifest_sha256": digest,
        },
        "network": {},
        "training": {},
    }
    apply_normalization_manifest(config, project_root=tmp_path)
    assert config["objective_scalarizer"]["scales"] == loaded["scales"]
    assert set(
        config["objective_scalarizer"]["endpoint_prediction_upper_bounds"]
    ) == {"flow", "cost", "variance"}
    assert "two_stage" not in config["training"]


def _row(
    preference_key: str,
    quality: float,
    *,
    succeeded: bool = True,
    index: int = 0,
) -> dict:
    return {
        "instance_id": f"instance_{index}",
        "preference_key": preference_key,
        "preference_quality_score": quality if succeeded else 1.0,
        "quality_score": quality,
        "terminated": succeeded,
        "truncated": not succeeded,
        "makespan": 1.0,
        "total_flow_time": 1.0 if succeeded else None,
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
        "operation_progress": 1.0 if succeeded else 0.5,
        "initial_progress": 0.0,
        "single_stage_proxy_return": 1.0 - quality if succeeded else -0.5,
    }


def test_universal_quality_averages_within_preference_then_equally_across_grid():
    config = load_config("configs/default.json")
    keys = [PreferenceContext.from_input(point).key for point in simplex_lattice(10, include=())]
    rows = [_row(key, 0.5, index=index) for index, key in enumerate(keys)]
    rows.extend([_row(keys[0], 0.2, index=100), _row(keys[0], 0.4, index=101)])
    aggregate = _aggregate_formal_rows(
        config,
        rows=rows,
        dataset_name="validation",
        manifest="manifest.json",
        unique_instance_count=1,
        repeat_count=1,
        universal=True,
    )
    assert aggregate["preference_quality_by_key"][keys[0]] == pytest.approx(
        (0.5 + 0.2 + 0.4) / 3
    )
    expected = ((0.5 + 0.2 + 0.4) / 3 + 65 * 0.5) / 66
    assert aggregate["preference_balanced_quality_score"] == pytest.approx(expected)


def test_universal_any_preference_without_success_has_infinite_quality():
    config = load_config("configs/default.json")
    keys = [PreferenceContext.from_input(point).key for point in simplex_lattice(10, include=())]
    rows = [
        _row(key, 0.5, succeeded=index != 7, index=index)
        for index, key in enumerate(keys)
    ]
    aggregate = _aggregate_formal_rows(
        config,
        rows=rows,
        dataset_name="validation",
        manifest="manifest.json",
        unique_instance_count=1,
        repeat_count=1,
        universal=True,
    )
    assert math.isinf(aggregate["preference_quality_by_key"][keys[7]])
    assert math.isinf(aggregate["preference_balanced_quality_score"])
    assert aggregate["completion_rate"] == 0.0


def test_universal_checkpoint_metadata_records_all_fixed_preferences():
    config = load_config("configs/default.json")
    metadata = _checkpoint_metadata(
        config,
        role="best",
        episode=100,
        validation_split="validation",
        validation_instance_limit=2,
    )
    assert metadata["preference_count"] == 66
    assert len(metadata["fixed_preference_set"]) == 66
    assert all(
        sum(preference.values()) == pytest.approx(1.0)
        for preference in metadata["fixed_preference_set"]
    )
