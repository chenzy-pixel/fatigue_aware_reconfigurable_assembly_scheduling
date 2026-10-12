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
from environment.workload import objective_names, workload_value
from environment import terminal_quality_score
from data.distribution import protocol_hashes
from analysis.workload_dominance import rescore_workload_row, workload_identity
from result.metrics import result_schema_version, evaluation_quality_metric, quality_metric_sha256
from result.io import write_config, write_csv, write_json
from result.provenance import build_provenance, dataset_manifest_snapshot
from utils import configured_formal_evaluation_sampling_seeds, derive_evaluation_sampling_seed


def _write_run(path, *, factor=1.0, seed=11, all_failed=False, training=False, config=None, instance_ids=("test_a", "test_b")):
    config = load_config("configs/v8/universal.json") if config is None else deepcopy(config)
    config["seed"] = seed
    path.mkdir(parents=True)
    write_config(path, config)
    rows = []
    manifest_path = project_path(config["paths"]["manifests_root"]) / "test/manifest.json"
    manifest_hash = dataset_manifest_snapshot(manifest_path)["sha256"]
    scalarizer = config["objective_scalarizer"]
    loads = {"worker_a": 3 * factor, "worker_b": 1 * factor, "worker_c": 1 * factor, "worker_d": 1 * factor}
    workload = workload_value(list(loads.values()), config)
    mean_load = sum(loads.values()) / len(loads)
    variance = sum((load - mean_load) ** 2 for load in loads.values()) / len(loads)
    seeds = configured_formal_evaluation_sampling_seeds(config, "final_test")
    for instance_id in instance_ids:
        for repeat, root_seed in enumerate(seeds):
            for index, preference in enumerate(formal_preferences(config, "final_test")):
                success = not all_failed and not (instance_id == "test_b" and repeat == 1 and index == 0)
                rows.append({
                    "instance_id": instance_id, "sampling_repeat": repeat,
                    "preference_key": preference.key, "sampling_seed": root_seed,
                    "derived_sampling_seed": derive_evaluation_sampling_seed(root_seed, instance_id, preference.key),
                    **{f"preference_{name}": value for name, value in preference.preference.as_dict(objective_names(config)).items()},
                    "task_succeeded": success, "task_failed": not success,
                    "terminated": True, "truncated": False,
                    "terminal_reason": "completed" if success else "infeasible",
                    "makespan": 50 * factor, "total_flow_time": 100 * factor if success else None,
                    "decisions": 1, "inference_time_seconds": 0.0, "solve_time_seconds": 0.0,
                    "inference_time_per_decision_ms": 0.0,
                    "relative_heuristic_gap_percent": None, "makespan_heuristic_gap_percent": None,
                    "reconfiguration_cost_heuristic_gap_percent": None, "worker_load_variance_heuristic_gap_percent": None,
                    "schedule_violation_count": 0, "maximum_worker_fatigue": 0.5,
                    "safe_fatigue_limit": 0.75, "fatigue_mode": "full",
                    "flow_time_objective": 100 * factor, "reconfiguration_cost": 200 * factor,
                    "worker_load_variance": variance, "worker_loads": json.dumps(loads),
                    "workload_objective_value": workload,
                    "worker_load_total": sum(loads.values()), "worker_load_max": max(loads.values()),
                    "candidate_id": f"{instance_id}-{repeat}-{index}",
                    "action_trace_sha256": f"trace-{instance_id}-{repeat}-{index}",
                    "arm": "ppo", "dataset": "test", "algorithm_seed": seed,
                    "result_schema_version": result_schema_version(config), **protocol_hashes(config),
                    "dataset_manifest_sha256": manifest_hash, "subset_sha256": "fixture_subset",
                    "experiment_suite_version": config["experiment_suite_version"],
                    "reward_version": config["reward"]["mode"], **workload_identity(config),
                    "objective_scalarizer_type": scalarizer["type"], "objective_scalarizer_rho": scalarizer["rho"],
                    "normalization_manifest_sha256": scalarizer["normalization_manifest_sha256"],
                    **{f"objective_scale_{name}": scalarizer["scales"][name] for name in objective_names(config)},
                    "quality_metric_sha256": quality_metric_sha256(evaluation_quality_metric(config)),
                    "decode_mode": "sampled", "policy_execution_version": config["training"]["policy_execution_version"],
                    "policy_precision": config["training"]["policy_precision"],
                    "preference_quality_score": terminal_quality_score(100 * factor, 200 * factor, workload, config,
                        preference=preference.preference, terminal_failure=not success),
                })
    provenance = build_provenance(config, dataset_manifest_path=project_path(config["paths"]["manifests_root"]) / "test/manifest.json",
                                  formal_evaluation_stage="final_test")
    metrics = {"evaluation_schema_version": result_schema_version(config), "decode_mode": "sampled", "dataset": "test",
               "instance_count": len(instance_ids), "repeat_count": len(seeds), "sampling_seeds": seeds, "provenance": provenance,
               "policy": "ppo", "manifest": str(manifest_path), "quality_metric": evaluation_quality_metric(config),
               "quality_metric_sha256": quality_metric_sha256(evaluation_quality_metric(config))}
    if training:
        write_json(path / "summary.json", {"final_sampled": metrics})
        write_csv(path / "final_sampled_instance_metrics.csv", rows)
    else:
        write_json(path / "metrics.json", metrics)
        write_csv(path / "instance_metrics.csv", rows)
    return config, metrics, rows


def test_v8_complete_66_by_3_grids_produce_matched_per_instance_fronts(tmp_path):
    config, _, rows = _write_run(tmp_path / "first")
    _write_run(tmp_path / "second", factor=0.8, training=True)
    summary = analyze_runs({"baseline": tmp_path / "first", "variant": tmp_path / "second"}, tmp_path / "analysis")
    assert summary["candidate_count"] == 2 * 2 * 66 * 3
    assert summary["instance_count"] == 2
    assert summary["preference_count"] == 66
    assert summary["repeat_count"] == 3
    assert summary["independent_test"]
    scales = [config["objective_scalarizer"]["scales"][name] for name in objective_names(config)]
    expected = prod(1 - value for value in normalize_objectives((100, 200, rows[0]["workload_objective_value"]), scales))
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


def _explicit_scored_run(path, *, legacy=True):
    from eval import _apply_evaluation_goal
    from test_workload_contracts import _legacy_config

    config = _legacy_config() if legacy else load_config("configs/workload/raw/q075/universal.json")
    config["evaluation"].update(result_schema_version="8.0.0" if legacy else "9.0.0", quality_metric=None)
    config["training"]["formal_evaluation"]["final_test_repeats"] = 1
    config, metrics, rows = _write_run(path, config=config, instance_ids=("test_a",))
    scoring_path = project_path("configs/workload/raw/q075/universal.json")
    rescored, scored_metrics = _apply_evaluation_goal(rows, metrics, scoring_path)
    write_json(path / "evaluation_config.json", load_config(scoring_path))
    write_json(path / "metrics.json", scored_metrics)
    write_csv(path / "instance_metrics.csv", rescored)
    return config, scored_metrics, rescored, rows


@pytest.mark.parametrize("legacy", [True, False])
def test_explicit_scored_formal_outputs_reenter_unified_analysis_preserving_training_scores(tmp_path, legacy):
    config, metrics, _, original = _explicit_scored_run(tmp_path / "rescored", legacy=legacy)
    assert result_schema_version(config) == ("8.0.0" if legacy else "9.0.0")
    assert metrics["evaluation_schema_version"] == "9.0.0"
    assert metrics["evaluation_workload_q"] == .75
    summary = analyze_runs({"rescored": tmp_path / "rescored"}, tmp_path / "out")
    assert summary["evaluation_workload_q"] == .5
    assert summary["candidate_count"] == 66
    assert summary["sources"][0]["evaluation_workload"]["workload_q"] == .75
    with (tmp_path / "out/candidates.csv").open(encoding="utf-8-sig", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    for old, scored in zip(original, candidates, strict=True):
        assert float(scored["training_preference_quality_score"]) == old["preference_quality_score"]
        assert scored["training_quality_metric_sha256"] == old["quality_metric_sha256"]
        assert scored["training_result_schema_version"] == old["result_schema_version"]
        assert float(scored["training_workload_objective_value"]) == old["workload_objective_value"]
        assert float(scored["evaluation_workload_q"]) == .5


@pytest.mark.parametrize("corruption", [
    "missing_config", "config_hash", "config_content", "evaluation_scale", "evaluation_manifest",
    "evaluation_identity", "evaluation_quality_hash", "training_schema", "training_metric",
    "training_preference", "row_scale", "row_evaluation_identity", "row_evaluation_value", "row_evaluation_quality",
])
def test_explicit_scored_output_rejects_tampered_scoring_or_training_identity(tmp_path, corruption):
    directory = tmp_path / "run"
    _, metrics, rows, _ = _explicit_scored_run(directory)
    path = directory / "evaluation_config.json"
    if corruption == "missing_config":
        path.unlink()
    elif corruption == "config_content":
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["training"]["episodes"] += 1
        write_json(path, saved)
    elif corruption == "config_hash":
        metrics["evaluation_config_sha256"] = "wrong"
    elif corruption == "evaluation_scale":
        metrics["evaluation_objective_scales"]["workload"] *= 2
    elif corruption == "evaluation_manifest":
        metrics["evaluation_normalization_manifest_sha256"] = "wrong"
    elif corruption == "evaluation_identity":
        metrics["evaluation_workload_q"] = .5
    elif corruption == "evaluation_quality_hash":
        metrics["quality_metric_sha256"] = "wrong"
    elif corruption == "training_schema":
        metrics["training_evaluation_schema_version"] = "9.0.0"
    elif corruption == "training_metric":
        metrics["training_goal_quality"]["quality_metric_sha256"] = "wrong"
    elif corruption == "training_preference":
        rows[0]["training_preference_quality_score"] += .1
    elif corruption == "row_scale":
        rows[0]["evaluation_objective_scale_workload"] *= 2
    elif corruption == "row_evaluation_identity":
        rows[0]["evaluation_workload_q"] = .5
    elif corruption == "row_evaluation_value":
        rows[0]["workload_objective_value"] += 1
    else:
        rows[0]["preference_quality_score"] += .1
    write_json(directory / "metrics.json", metrics)
    write_csv(directory / "instance_metrics.csv", rows)
    with pytest.raises(ValueError, match="evaluation|candidate"):
        analyze_runs({"run": directory}, tmp_path / "out")


def test_legacy_and_modern_explicit_scoring_outputs_share_one_unified_analysis(tmp_path):
    _explicit_scored_run(tmp_path / "legacy", legacy=True)
    _explicit_scored_run(tmp_path / "modern", legacy=False)
    summary = analyze_runs({"legacy": tmp_path / "legacy", "modern": tmp_path / "modern"}, tmp_path / "out")
    assert summary["candidate_count"] == 132
    assert summary["evaluation_workload_q"] == .5
    assert summary["objective_scales"]["workload"] == 21.03145448655653
    assert {source["training_workload"]["workload_mode"] for source in summary["sources"]} == {"variance", "geometric_owa"}
    assert all(source["evaluation_workload"]["workload_q"] == .75 for source in summary["sources"])


def test_repeated_scoring_preserves_first_training_metadata_and_row_scores(tmp_path):
    from eval import _apply_evaluation_goal

    config, once_metrics, once_rows, original = _explicit_scored_run(tmp_path / "run")
    next_path = project_path("configs/workload/raw/q05/universal.json")
    twice_rows, twice_metrics = _apply_evaluation_goal(once_rows, once_metrics, next_path)
    assert twice_metrics["training_goal_quality"] == once_metrics["training_goal_quality"]
    assert twice_metrics["training_evaluation_schema_version"] == result_schema_version(config)
    assert twice_metrics["evaluation_workload_q"] == .5
    for old, once, twice in zip(original, once_rows, twice_rows, strict=True):
        for field in ("workload_objective_value", "preference_quality_score", "quality_metric_sha256", "result_schema_version"):
            assert twice[f"training_{field}"] == once[f"training_{field}"] == old[field]
        assert rescore_workload_row(twice, load_config(next_path))["training_preference_quality_score"] == old["preference_quality_score"]
    write_json(tmp_path / "run/evaluation_config.json", load_config(next_path))
    write_json(tmp_path / "run/metrics.json", twice_metrics)
    write_csv(tmp_path / "run/instance_metrics.csv", twice_rows)
    assert analyze_runs({"twice": tmp_path / "run"}, tmp_path / "out")["candidate_count"] == 66
