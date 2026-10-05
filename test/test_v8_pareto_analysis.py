from __future__ import annotations

import csv
import json
from copy import deepcopy
from itertools import product
from math import prod

import numpy as np
import pytest

from analysis.pareto_analysis import hypervolume_3d, normalize_objectives
from analysis.v8_pareto_analysis import analyze_runs, parse_run_arguments
from configs import load_config, project_path
from configs.formal_preferences import formal_preferences
from result.io import write_config, write_csv, write_json
from result.provenance import build_provenance, dataset_manifest_snapshot
from utils import configured_formal_evaluation_sampling_seeds, derive_evaluation_sampling_seed


def _write_run(path, *, factor=1.0, seed=11, all_failed=False, training=False):
    config = load_config("configs/v8/universal.json")
    config["seed"] = seed
    path.mkdir(parents=True)
    write_config(path, config)
    rows = []
    seeds = configured_formal_evaluation_sampling_seeds(config, "final_test")
    for instance_id in ("test_a", "test_b"):
        for repeat, root_seed in enumerate(seeds):
            for index, preference in enumerate(formal_preferences(config, "final_test")):
                success = not all_failed and not (instance_id == "test_b" and repeat == 1 and index == 0)
                rows.append({
                    "instance_id": instance_id, "sampling_repeat": repeat,
                    "preference_key": preference.key, "sampling_seed": root_seed,
                    "derived_sampling_seed": derive_evaluation_sampling_seed(root_seed, instance_id, preference.key),
                    **{f"preference_{name}": value for name, value in preference.preference.as_dict().items()},
                    "task_succeeded": success, "task_failed": not success,
                    "terminated": True, "truncated": False,
                    "schedule_violation_count": 0, "maximum_worker_fatigue": 0.5,
                    "safe_fatigue_limit": 0.75, "fatigue_mode": "full",
                    "flow_time_objective": 100 * factor, "reconfiguration_cost": 200 * factor,
                    "worker_load_variance": 3 * factor, "action_trace_sha256": f"trace-{instance_id}-{repeat}-{index}",
                })
    provenance = build_provenance(config, dataset_manifest_path=project_path(config["paths"]["manifests_root"]) / "test/manifest.json",
                                  formal_evaluation_stage="final_test")
    metrics = {"evaluation_schema_version": "8.0.0", "decode_mode": "sampled", "dataset": "test",
               "instance_count": 2, "repeat_count": 3, "sampling_seeds": seeds, "provenance": provenance}
    if training:
        write_json(path / "summary.json", {"final_sampled": metrics})
        write_csv(path / "final_sampled_instance_metrics.csv", rows)
    else:
        write_json(path / "metrics.json", metrics)
        write_csv(path / "instance_metrics.csv", rows)
    return config, metrics, rows


def test_v8_complete_66_by_3_grids_produce_matched_per_instance_fronts(tmp_path):
    config, _, _ = _write_run(tmp_path / "first")
    _write_run(tmp_path / "second", factor=0.8, training=True)
    summary = analyze_runs({"baseline": tmp_path / "first", "variant": tmp_path / "second"}, tmp_path / "analysis")
    assert summary["candidate_count"] == 2 * 2 * 66 * 3
    assert summary["instance_count"] == 2
    assert summary["preference_count"] == 66
    assert summary["repeat_count"] == 3
    assert summary["independent_test"]
    scales = [config["objective_scalarizer"]["scales"][name] for name in ("flow", "cost", "variance")]
    expected = prod(1 - value for value in normalize_objectives((100, 200, 3), scales))
    assert summary["methods"]["baseline"]["mean_pooled_hypervolume"]["mean"] == pytest.approx(expected)
    assert summary["methods"]["variant"]["mean_pooled_hypervolume"]["mean"] > expected
    with (tmp_path / "analysis/paired_instances.csv").open(encoding="utf-8-sig") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 2
    assert all(float(row["coverage_variant_over_baseline"]) == 1.0 for row in pairs)
    assert all(float(row["coverage_baseline_over_variant"]) == 0.0 for row in pairs)
    assert all(float(row["delta_pooled_hypervolume"]) > 0.0 for row in pairs)


def test_v8_empty_feasible_set_has_zero_hv_and_reports_completion(tmp_path):
    _write_run(tmp_path / "failed", all_failed=True)
    summary = analyze_runs({"failed": tmp_path / "failed"}, tmp_path / "analysis")
    assert summary["front_row_count"] == 0
    assert summary["methods"]["failed"]["mean_pooled_hypervolume"]["mean"] == 0.0
    assert summary["methods"]["failed"]["mean_completion_rate"]["mean"] == 0.0


@pytest.mark.parametrize("corruption", ("missing", "duplicate", "unsafe", "preference", "seed"))
def test_v8_grid_validates_cells_and_filters_unsafe_points(tmp_path, corruption):
    _, _, rows = _write_run(tmp_path / "run")
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif corruption == "unsafe":
        for row in rows:
            row["maximum_worker_fatigue"] = 0.9
    elif corruption == "preference":
        rows[0]["preference_flow"] = 0.4
    else:
        rows[0]["derived_sampling_seed"] += 1
    write_csv(tmp_path / "run/instance_metrics.csv", rows)
    if corruption == "unsafe":
        result = analyze_runs({"run": tmp_path / "run"}, tmp_path / "out")
        assert result["front_row_count"] == 0
        assert result["methods"]["run"]["mean_completion_rate"]["mean"] > 0.0
    else:
        with pytest.raises(ValueError):
            analyze_runs({"run": tmp_path / "run"}, tmp_path / "out")


@pytest.mark.parametrize("corruption", ("dataset", "scale", "schema", "stage", "config_hash", "neutral"))
def test_v8_analysis_rejects_incompatible_run_identity(tmp_path, corruption):
    config, metrics, _ = _write_run(tmp_path / "first")
    _write_run(tmp_path / "second")
    if corruption == "dataset":
        metrics["provenance"]["dataset_manifest_sha256"] = "different"
    elif corruption == "scale":
        metrics["provenance"]["objective_scales"]["flow"] = 1200.0
    elif corruption == "schema":
        metrics["evaluation_schema_version"] = "4.1.0"
    elif corruption == "stage":
        metrics["provenance"]["formal_evaluation_stage"] = "validation"
    elif corruption == "config_hash":
        metrics["provenance"]["effective_config_sha256"] = "wrong"
    else:
        config["environment"]["fatigue_mode"] = "neutral"
        config.pop("runtime_manifest")
        write_config(tmp_path / "first", config)
    write_json(tmp_path / "first/metrics.json", metrics)
    with pytest.raises(ValueError):
        analyze_runs({"first": tmp_path / "first", "second": tmp_path / "second"}, tmp_path / "out")


def test_v8_groups_multiple_training_seeds_and_requires_matched_seeds(tmp_path):
    _write_run(tmp_path / "a11")
    _write_run(tmp_path / "a23", seed=23, factor=0.9)
    _write_run(tmp_path / "b11", factor=0.8)
    _write_run(tmp_path / "b23", seed=23, factor=0.7)
    runs = {"a": [tmp_path / "a11", tmp_path / "a23"], "b": [tmp_path / "b11", tmp_path / "b23"]}
    summary = analyze_runs(runs, tmp_path / "out")
    assert summary["methods"]["a"]["training_seed_count"] == 2
    assert summary["methods"]["a"]["mean_pooled_hypervolume"]["std"] > 0
    runs["b"].pop()
    with pytest.raises(ValueError, match="unmatched training seeds"):
        analyze_runs(runs, tmp_path / "bad")


def test_hv_sweep_matches_independent_small_grid_union():
    points = np.random.default_rng(51).uniform(0, 1, size=(8, 3))
    coordinates = [sorted({*points[:, dimension], 1.0}) for dimension in range(3)]
    expected = 0.0
    for i, j, k in product(*(range(len(axis) - 1) for axis in coordinates)):
        lower = np.array([coordinates[0][i], coordinates[1][j], coordinates[2][k]])
        if np.any(np.all(points <= lower, axis=1)):
            expected += prod(coordinates[d][n + 1] - coordinates[d][n] for d, n in enumerate((i, j, k)))
    assert hypervolume_3d(points) == pytest.approx(expected, abs=1e-12)


def test_v8_run_cli_accepts_repeated_method_labels():
    runs = parse_run_arguments(["main=seed11", "main=seed23", "variant=variant11"])
    assert len(runs["main"]) == 2
    assert set(runs) == {"main", "variant"}


def test_coverage_counts_equal_front_points_and_handles_missing_front():
    from analysis.v8_pareto_analysis import _coverage
    assert _coverage([(0.1, 0.2, 0.3)], [(0.1, 0.2, 0.3)]) == 1.0
    assert _coverage([], [(0.1, 0.2, 0.3)]) == 0.0
    assert _coverage([(0.1, 0.2, 0.3)], []) is None


@pytest.mark.parametrize("partial_source", ("metadata", "cell"))
def test_v8_analysis_rejects_external_sampling_truncation(tmp_path, partial_source):
    _, metrics, rows = _write_run(tmp_path / "run")
    if partial_source == "metadata":
        metrics.update(evaluation_complete=False, sampling_truncated_count=1)
        write_json(tmp_path / "run/metrics.json", metrics)
    else:
        rows[0].update(terminated=False, truncated=True, sampling_truncated=True,
                       task_succeeded=False, task_failed=False)
        write_csv(tmp_path / "run/instance_metrics.csv", rows)
    with pytest.raises(ValueError, match="incomplete|truncated"):
        analyze_runs({"run": tmp_path / "run"}, tmp_path / "out")
