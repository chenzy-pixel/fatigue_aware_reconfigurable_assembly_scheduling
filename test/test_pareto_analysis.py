import pytest
from copy import deepcopy
from configs import load_config
from configs.formal_preferences import formal_preferences
from environment import bounded_quality_score
from result.metrics import EVALUATION_SCHEMA_VERSION, evaluation_quality_metric, quality_metric_sha256
from analysis.pareto_analysis import analyze_rows
from analysis.mo_alns_analysis import analyze_rows as compare_rows
from analysis.pareto_analysis import dominates, vectors_equal, nondominated_indices, normalize_objectives, hypervolume_3d

def test_dominance_equality_and_tolerance() -> None:
    assert dominates((1.0, 2.0, 3.0), (1.0, 2.0, 4.0))
    assert not dominates((1.0, 2.0, 4.0), (1.0, 2.0, 3.0))
    assert vectors_equal((1.0, 2.0, 3.0), (1.0 + 5e-10, 2.0, 3.0))
    assert not dominates((1.0, 2.0, 3.0), (1.0 + 5e-10, 2.0, 3.0))
    assert nondominated_indices([(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]) == [0]


def test_bounded_normalization_is_monotone() -> None:
    lower = normalize_objectives((10.0, 20.0, 30.0), (100.0, 100.0, 100.0))
    upper = normalize_objectives((20.0, 40.0, 60.0), (100.0, 100.0, 100.0))
    assert all(0.0 <= value < 1.0 for value in lower + upper)
    assert all(first < second for first, second in zip(lower, upper, strict=True))


def test_exact_hypervolume_for_single_and_overlapping_boxes() -> None:
    assert hypervolume_3d([(0.2, 0.3, 0.4)]) == pytest.approx(0.8 * 0.7 * 0.6)
    assert hypervolume_3d(
        [(0.2, 0.8, 0.8), (0.8, 0.2, 0.8)]
    ) == pytest.approx(0.056)
    assert hypervolume_3d([(0.2, 0.3, 0.4), (0.4, 0.5, 0.6)]) == pytest.approx(
        0.8 * 0.7 * 0.6
    )


def _formal_candidates(config, stage="validation", arm="ppo"):
    scalarizer = config["objective_scalarizer"]
    rows = []
    repeats = config["training"]["formal_evaluation"]["validation_repeats" if stage == "validation" else "final_test_repeats"] if arm == "ppo" else 1
    for point in formal_preferences(config, stage):
        for repeat in range(repeats):
            rows.append({
                "arm": arm, "dataset": "test", "algorithm_seed": 11, "instance_id": "matrix_instance",
                "terminated": True, "truncated": False, "schedule_violation_count": 0,
                "maximum_worker_fatigue": 0.5, "safe_fatigue_limit": 0.75,
                "flow_time_objective": 1000.0, "reconfiguration_cost": 300.0, "worker_load_variance": 3.0,
                "result_schema_version": EVALUATION_SCHEMA_VERSION,
                "experiment_suite_version": config["experiment_suite_version"],
                "reward_version": config["reward"]["mode"],
                "objective_scalarizer_type": scalarizer["type"], "objective_scalarizer_rho": scalarizer["rho"],
                "normalization_manifest_sha256": scalarizer["normalization_manifest_sha256"],
                **{f"objective_scale_{name}": scalarizer["scales"][name] for name in ("flow", "cost", "variance")},
                "quality_metric_sha256": quality_metric_sha256(evaluation_quality_metric(config)),
                "decode_mode": "sampled" if arm == "ppo" else "solver",
                "policy_execution_version": config["training"]["policy_execution_version"],
                "policy_precision": "float32", "sampling_repeat": repeat,
                **{f"w_{name}": value for name, value in point.preference.as_dict().items()},
                "preference_quality_score": bounded_quality_score(1000.0, 300.0, 3.0, config, preference=point.preference),
            })
    return rows


@pytest.mark.parametrize("stage,count", [("validation", 39), ("final_test", 198)])
def test_current_analysis_validates_stage_specific_sampled_matrix(stage, count):
    config = load_config("configs/default.json")
    rows = _formal_candidates(config, stage)
    annotated, instances, seeds, summary = analyze_rows(rows, config, stage=stage)
    assert len(annotated) == count
    assert instances[0]["ppo_completion_rate"] == 1.0
    assert instances[0]["ppo_hypervolume"] > 0.0
    assert len(seeds) == 1
    assert summary["normalization_manifest_sha256"] == config["objective_scalarizer"]["normalization_manifest_sha256"]
    for malformed in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError, match="incomplete|duplicate"):
            analyze_rows(malformed, config, stage=stage)


@pytest.mark.parametrize("field,value", [("objective_scale_flow", 1200), ("normalization_manifest_sha256", "old_hash"), ("decode_mode", "greedy")])
def test_current_analysis_rejects_mixed_protocol_candidates(field, value):
    config = load_config("configs/default.json")
    rows = deepcopy(_formal_candidates(config))
    rows[0][field] = value
    with pytest.raises(ValueError):
        analyze_rows(rows, config, stage="validation")


def test_comparison_uses_sampled_ppo_and_solver_endpoint_budgets():
    config = load_config("configs/default.json")
    rows = _formal_candidates(config) + _formal_candidates(config, arm="mo_alns")
    annotated, instances, seeds, summary = compare_rows(rows, config, stage="validation")
    assert len(annotated) == 52
    assert instances[0]["ppo_hypervolume"] == pytest.approx(instances[0]["mo_alns_hypervolume"])
    assert summary["analysis_protocol"] == "ppo_mo_alns_solver_budget_v2"
    assert summary["statistics"]["test"]["algorithm_seed_count"] == 1
