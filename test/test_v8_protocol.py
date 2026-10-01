from __future__ import annotations

import math
import statistics
from copy import deepcopy
from pathlib import Path

import pytest

from configs import load_config, validate_latest_only_config
from configs.formal_preferences import formal_preferences
from configs.normalization import (
    NORMALIZATION_MANIFEST_SCHEMA,
    SELECTED_VALIDATION_MANIFEST_SCHEMA,
    apply_normalization_manifest,
    canonical_json_sha256,
    load_normalization_manifest,
    write_immutable_manifest,
)
from environment import PreferenceContext, normalize_preference, quality_preference_for_episode, simplex_lattice
from result import build_provenance, source_state_snapshot
from train import _aggregate_formal_rows, _checkpoint_metadata
from utils import configured_formal_evaluation_sampling_seeds


@pytest.mark.parametrize(
    ("path", "preference"),
    (
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


def test_universal_uses_13_validation_and_66_final_preferences_with_training_sequence():
    config = load_config("configs/v8/universal.json")
    assert len(formal_preferences(config, "validation")) == 13
    assert len(formal_preferences(config, "final_test")) == 66
    expected_validation = [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
        (1 / 3, 1 / 3, 1 / 3), (0.6, 0.3, 0.1), (0.6, 0.1, 0.3),
        (0.3, 0.6, 0.1), (0.3, 0.1, 0.6), (0.1, 0.6, 0.3),
        (0.1, 0.3, 0.6),
    ]
    for actual, expected in zip(formal_preferences(config, "validation"), expected_validation, strict=True):
        assert actual.as_tuple() == pytest.approx(expected)
    assert config["network"]["worker_flow_time_normalization"] == "candidate_zscore_v1"
    assert config["network"]["worker_flow_time_std_floor"] == 0.001
    assert config["training"]["parallel_envs"] == 20
    assert config["training"]["validation_parallel_envs"] == 20
    assert config["training"]["validation_instance_limit"] == 50
    assert config["training"]["validation_interval_episodes"] == 100
    assert configured_formal_evaluation_sampling_seeds(config, "validation") == [100011, 100012, 100013]
    assert configured_formal_evaluation_sampling_seeds(config, "final_test") == [300011, 300012, 300013]
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


def test_all_final_grid_preferences_survive_float32_observation_round_trip():
    for point in formal_preferences(load_config("configs/v8/universal.json"), "final_test"):
        normalized = normalize_preference(point.as_array())
        assert normalized.as_tuple() == pytest.approx(point.as_tuple(), abs=5e-8)
        assert all(value >= 0.0 for value in normalized.as_tuple())
        assert sum(normalized.as_tuple()) == pytest.approx(1.0)
        for original, actual in zip(point.as_tuple(), normalized.as_tuple(), strict=True):
            if original == 0.0:
                assert actual == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        normalize_preference((0.9, 0.10001, -0.00001))


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
    means = {name: [float(base + i) for i in range(5)] for name, base in (("flow", 10), ("cost", 20), ("variance", 30))}
    manifest = {
        "schema_version": NORMALIZATION_MANIFEST_SCHEMA,
        "sources": {name: {"validation_episodes": [840, 880, 920, 960, 1000], "successful_trajectory_means": values} for name, values in means.items()},
        "scales": {name: values[2] for name, values in means.items()},
    }
    manifest["content_sha256"] = canonical_json_sha256(manifest)
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
    assert config["network"]["normalization_manifest_sha256"] == digest
    assert "two_stage" not in config["training"]
    destination.write_text(destination.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_normalization_manifest(destination, expected_sha256=digest)


def test_frozen_scales_match_committed_tail_means_and_config():
    config = load_config("configs/v8/universal.json")
    manifest = load_normalization_manifest(
        config["objective_scalarizer"]["normalization_manifest"],
        expected_sha256=config["objective_scalarizer"]["normalization_manifest_sha256"],
    )
    assert config["objective_scalarizer"]["scales"] == manifest["scales"]
    assert manifest["schema_version"] == SELECTED_VALIDATION_MANIFEST_SCHEMA
    assert manifest["scales"] == {"flow": 1089.15, "cost": 353.27, "variance": 2.2629}
    for objective, source in manifest["sources"].items():
        assert round(source["successful_trajectory_mean"], source["rounding_digits"]) == manifest["scales"][objective]


def test_legacy_e1_scales_remain_verifiable():
    from configs.normalization import file_sha256
    path = Path("configs/manifests/e1_tail5_scales_20260928.json")
    manifest = load_normalization_manifest(path, expected_sha256=file_sha256(path))
    assert manifest["schema_version"] == NORMALIZATION_MANIFEST_SCHEMA
    for objective, source in manifest["sources"].items():
        assert source["validation_episodes"] == [840, 880, 920, 960, 1000]
        assert statistics.median(source["successful_trajectory_means"]) == manifest["scales"][objective]


@pytest.mark.parametrize("invalid", ["scale", "rounding", "source_hash"])
def test_selected_scales_reject_inconsistent_signed_payload(tmp_path, invalid):
    config = load_config("configs/default.json")
    manifest = load_normalization_manifest(
        config["objective_scalarizer"]["normalization_manifest"],
        expected_sha256=config["objective_scalarizer"]["normalization_manifest_sha256"],
    )
    manifest.pop("content_sha256")
    if invalid == "scale":
        manifest["scales"]["variance"] *= 2
    elif invalid == "rounding":
        manifest["sources"]["flow"]["rounding_digits"] = 4
    else:
        manifest["sources"]["cost"]["validation_log_sha256"] = "unknown"
    manifest["content_sha256"] = canonical_json_sha256(manifest)
    path = tmp_path/"invalid.json"
    digest = write_immutable_manifest(path, manifest)
    with pytest.raises(ValueError, match="normalization"):
        load_normalization_manifest(path, expected_sha256=digest)


def test_current_scales_align_training_reference_quality_and_pareto():
    from analysis.pareto_analysis import normalize_objectives
    from environment.types import bounded_quality_score
    from result.metrics import evaluation_quality_metric, LEGACY_QUALITY_METRIC, quality_metric_sha256
    expected = {"flow":1089.15, "cost":353.27, "variance":2.2629}
    for path in ("configs/default.json", "configs/v8/universal.json",
                 "configs/e1/single_flow.json", "configs/e1/single_cost.json", "configs/e1/single_variance.json",
                 "configs/baselines/mo_alns.json"):
        config = load_config(path)
        assert config["objective_scalarizer"]["scales"] == expected
        reference = evaluation_quality_metric(config)
        assert {name:reference[f"{name}_scale"] for name in expected} == expected
        assert normalize_objectives(list(expected.values()), list(expected.values())) == pytest.approx([.5,.5,.5])
        for preference in ((1,0,0),(0,1,0),(0,0,1)):
            assert bounded_quality_score(*expected.values(), config, preference=preference) == pytest.approx(.5)
        assert bounded_quality_score(1089.15, 2*353.27, 4*2.2629, reference) == pytest.approx(.61)
    assert evaluation_quality_metric({"evaluation":{"quality_metric":LEGACY_QUALITY_METRIC}}) == LEGACY_QUALITY_METRIC
    assert quality_metric_sha256(reference) != quality_metric_sha256(LEGACY_QUALITY_METRIC)


def _row(
    preference_key: str,
    quality: float,
    *,
    succeeded: bool = True,
    index: int = 0,
) -> dict:
    return {
        "instance_id": f"instance_{index}",
        "sampling_repeat": 0,
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
    keys = [point.key for point in formal_preferences(config, "validation")]
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
    expected = ((0.5 + 0.2 + 0.4) / 3 + (len(keys) - 1) * 0.5) / len(keys)
    assert aggregate["preference_balanced_quality_score"] == pytest.approx(expected)


def test_universal_any_preference_without_success_has_infinite_quality():
    config = load_config("configs/default.json")
    keys = [point.key for point in formal_preferences(config, "validation")]
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


def test_universal_validation_uses_only_configured_13_preferences():
    config = load_config("configs/v8/universal.json")
    keys = [point.key for point in formal_preferences(config, "validation")]
    rows = [_row(key, 0.25) for key in keys]
    aggregate = _aggregate_formal_rows(
        config, rows=rows, dataset_name="validation", manifest="manifest.json",
        unique_instance_count=1, repeat_count=1, universal=True, stage="validation",
        strict_counts=True,
    )
    assert aggregate["preference_count"] == 13
    assert aggregate["preference_balanced_quality_score"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="configured preference set"):
        _aggregate_formal_rows(
            config, rows=rows[:-1], dataset_name="validation", manifest="manifest.json",
            unique_instance_count=1, repeat_count=1, universal=True, stage="validation",
            strict_counts=True,
        )


def test_universal_checkpoint_metadata_records_all_fixed_preferences():
    config = load_config("configs/v8/universal.json")
    metadata = _checkpoint_metadata(
        config,
        role="best",
        episode=100,
        validation_split="validation",
        validation_instance_limit=2,
    )
    assert metadata["preference_count"] == 13
    assert metadata["formal_evaluation_stage"] == "validation"
    assert len(metadata["fixed_preference_set"]) == 13
    assert all(
        sum(preference.values()) == pytest.approx(1.0)
        for preference in metadata["fixed_preference_set"]
    )


@pytest.mark.parametrize(("stage", "preference_count"), (("validation", 13), ("final_test", 66)))
def test_formal_aggregation_rejects_duplicate_cells_even_when_counts_match(stage, preference_count):
    config = load_config("configs/v8/universal.json")
    rows = [
        {**_row(point.key, 0.25, index=index), "sampling_repeat": repeat}
        for point in formal_preferences(config, stage)
        for index in range(2)
        for repeat in range(3)
    ]
    arguments = dict(
        config=config, dataset_name="validation" if stage == "validation" else "test",
        manifest="manifest.json", unique_instance_count=2, repeat_count=3,
        universal=True, stage=stage, strict_counts=True,
    )
    aggregate = _aggregate_formal_rows(rows=rows, **arguments)
    assert aggregate["preference_count"] == preference_count
    assert aggregate["cell_count"] == preference_count * 2 * 3
    assert aggregate["completed_cell_count"] == preference_count * 2 * 3
    assert aggregate["completed_count"] == 2
    assert aggregate["formal_evaluation_stage"] == stage
    failed_rows = [dict(row) for row in rows]
    failed_rows[0].update(terminated=False, truncated=True)
    failed = _aggregate_formal_rows(rows=failed_rows, **arguments)
    assert failed["completed_cell_count"] == preference_count * 2 * 3 - 1
    assert failed["completed_count"] == 1
    assert failed["completion_rate"] == pytest.approx(5 / 6)
    rows[1] = dict(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        _aggregate_formal_rows(rows=rows, **arguments)


@pytest.mark.parametrize(("stage", "preference_count"), (("validation", 13), ("final_test", 66)))
def test_provenance_records_the_evaluated_stage_and_moved_sources(stage, preference_count):
    config = load_config("configs/v8/universal.json")
    provenance = build_provenance(config, formal_evaluation_stage=stage)
    assert provenance["formal_evaluation_stage"] == stage
    assert provenance["preference_count"] == preference_count
    assert provenance["ordered_preference_set"] == [
        point.preference.as_dict() for point in formal_preferences(config, stage)
    ]
    assert provenance["objective_scales"] == config["objective_scalarizer"]["scales"]
    assert provenance["normalization_manifest_sha256"] == config["objective_scalarizer"]["normalization_manifest_sha256"]
    assert {"analysis/pareto_analysis.py", "scripts/mo_alns.py", "training/protocol.py"}.issubset(
        set(source_state_snapshot()["paths"])
    )
