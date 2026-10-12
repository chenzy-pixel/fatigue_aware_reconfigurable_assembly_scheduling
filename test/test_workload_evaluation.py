from copy import deepcopy
import math

import pytest

from agent.mo_alns.types import metrics_objectives
from analysis.workload_dominance import (
    actual_worker_loads, labelled_workload_dominance, rescore_workload_row,
    resolve_workload_evaluation_config, workload_identity, analyze_workload_pool,
)
from configs import load_config
from environment.workload import workload_value
from result.metrics import evaluation_quality_metric, quality_metric_sha256, result_schema_version


def _modern(q="q05"):
    return load_config(f"configs/workload/raw/{q}/universal.json")


def _row(candidate, loads, **changes):
    mean = sum(loads.values()) / len(loads)
    row = {"candidate_id": candidate, "method": "policy_a", "algorithm_seed": 11,
           "dataset_manifest_sha256": "fixed_dataset", "environment_config_sha256": "fixed_physics",
           "generator_config_sha256": "fixed_generator", "distribution_contract_sha256": "fixed_distribution",
           "instance_id": "same_instance", "task_succeeded": True, "task_failed": False,
           "terminated": True, "truncated": False, "sampling_truncated": False,
           "maximum_worker_fatigue": .4, "safe_fatigue_limit": .75, "schedule_violation_count": 0,
           "flow_time_objective": 1000., "reconfiguration_cost": 200., "worker_loads": loads,
           "worker_load_variance": sum((value - mean) ** 2 for value in loads.values()) / len(loads),
           "preference_flow": 0., "preference_cost": 0., "preference_workload": 1.,
           "preference_quality_score": .3, "quality_score": .3, "result_schema_version": "9.0.0"}
    return {**row, **changes}


def test_modern_quality_identity_is_explicit_and_frozen():
    config = _modern()
    metric = evaluation_quality_metric(config)
    assert metric["workload_q"] == .5
    assert metric["flow_mode"] == "raw_v1"
    assert metric["workload_rule"] == "normalized_geometric_by_worker_count_v1"
    assert result_schema_version(config) == "9.0.0"
    assert quality_metric_sha256(metric) != quality_metric_sha256(evaluation_quality_metric(_modern("q075")))
    config["evaluation"]["quality_metric"] = {**metric, "workload_scale": metric["workload_scale"] * 2}
    with pytest.raises(ValueError, match="frozen"):
        evaluation_quality_metric(config)


def test_actual_workload_is_used_in_solver_and_rescoring():
    config = _modern()
    row = _row("actual", {"worker_a": 2., "worker_b": 0.},
               workload_objective_value=4 / 3, committed_workload_objective_value=99.)
    assert metrics_objectives(row, config)[2] == pytest.approx(4 / 3)
    scored = rescore_workload_row(row, config)
    assert scored["workload_objective_value"] == pytest.approx(4 / 3)
    assert scored["worker_load_variance"] == 1.
    assert scored["training_workload_objective_value"] == pytest.approx(4 / 3)
    assert scored["training_preference_quality_score"] == .3
    assert scored["evaluation_workload_q"] == .5
    assert scored["worker_load_total"] == 2.
    assert scored["worker_load_max"] == 2.


def test_common_q05_rescores_own_q_while_preserving_originals():
    own = _modern("q075")
    common = resolve_workload_evaluation_config(own)
    assert common["workload_objective"]["q"] == .5
    own_copy = deepcopy(own)
    row = _row("own_q", {"a": 10., "b": 0.}, workload_objective_value=workload_value([10., 0.], own))
    scored = rescore_workload_row(row, common)
    assert scored["workload_objective_value"] == pytest.approx(20 / 3)
    assert scored["training_workload_objective_value"] != scored["workload_objective_value"]
    assert own == own_copy
    assert resolve_workload_evaluation_config(own, own_goal=True) == own


def test_reconstruction_requires_complete_logs_and_all_worker_ids():
    row = _row("historical", {"a": 2., "b": 0., "idle": 0.}, makespan=10., completed_reconfigurations=1)
    row.pop("worker_loads")
    logs = [{"worker_id": "a", "machine_id": "m", "operation_id": "op", "stage": "INS", "start": 3., "end": 5.}]
    with pytest.raises(ValueError, match="complete reconfiguration"):
        actual_worker_loads(row, worker_ids=["a", "b", "idle"], reconfigurations=logs)
    assert actual_worker_loads(row, worker_ids=["a", "b", "idle"], reconfigurations=logs,
                              logs_complete=True) == {"a": 2., "b": 0., "idle": 0.}
    with pytest.raises(ValueError, match="completed installations"):
        actual_worker_loads(row, worker_ids=["a", "b", "idle"], reconfigurations=[], logs_complete=True)
    with pytest.raises(ValueError, match="actual variance"):
        actual_worker_loads({**row, "worker_load_variance": 0}, worker_ids=["a", "b", "idle"],
                            reconfigurations=logs, logs_complete=True)


def test_historical_variance_cannot_substitute_for_actual_owa_loads():
    row = _row("historical", {"a": 2., "b": 0.})
    row.pop("worker_loads")
    with pytest.raises(ValueError, match="actual worker loads unavailable"):
        rescore_workload_row(row, _modern())


def test_labelled_dominance_catches_more_work_even_when_variance_improves():
    rows = [_row("small", {"a": 2., "b": 0.}),
            _row("balanced_more", {"a": 2., "b": 2.}),
            _row("other_policy", {"a": 3., "b": 1.}, method="policy_b")]
    assert rows[1]["worker_load_variance"] < rows[0]["worker_load_variance"]
    result = labelled_workload_dominance(rows)
    assert result["within_policy"]["dominated_candidate_count"] == 1
    assert result["across_policy"]["dominated_candidate_count"] == 1
    assert result["pooled"]["dominated_candidate_count"] == 2
    assert result["pooled"]["dominated_candidate_ratio"] == pytest.approx(2 / 3)
    assert result["within_policy"]["dominated_candidate_ratio"] == pytest.approx(1 / 3)
    assert result["per_policy"]["policy_a"]["within_policy_dominated_ratio"] == .5
    assert result["per_policy"]["policy_b"]["pooled_dominated_ratio"] == 1.
    assert any("small" in pair["dominator_id"] and "balanced_more" in pair["dominated_id"]
               for pair in result["within_policy"]["witnesses"])


def test_worker_labels_physics_and_failure_coverage_are_preserved():
    rows = [_row("left", {"a": 2., "b": 0.}), _row("right", {"a": 0., "b": 3.}),
            _row("different_physics", {"a": 10., "b": 10.}, environment_config_sha256="different"),
            _row("failure", {"a": 10., "b": 10.}, task_succeeded=False, task_failed=True),
            _row("missing", {"a": 10., "b": 10.}, worker_loads=None)]
    result = labelled_workload_dominance(rows)
    assert result["within_policy"]["pair_count"] == 0
    assert result["physical_group_count"] == 2
    assert result["successful_legal_missing_loads_count"] == 1
    assert result["excluded_failed_or_unsafe_count"] == 1
    assert result["task_failed_count"] == 1
    assert result["completion_rate"] == .8
    assert result["pooled"]["eligible_candidate_count"] == 3
    assert result["load_coverage"] == .75
    assert "not a global" in result["interpretation"]


def test_training_explicit_validation_indices_preserve_subset_identity():
    from train import _select_training_validation_subsets
    from data.selection import subset_snapshot
    class Dataset:
        manifest = {"files": [{"path": f"instance_{index}.pkl"} for index in range(6)]}
        def __len__(self):
            return 6
    dataset = Dataset()
    config = {"training": {"validation_instance_indices": [4, 1]}, "generator": {"dataset_pressure_weights": {}}}
    selected = _select_training_validation_subsets(config, dataset, target_count=2, diagnostic_count=0)
    assert selected == {"target": subset_snapshot(dataset, [4, 1], role="target")}
    for indices in ([6, 1], [-1, 1], [1, 1], [], [True, 1]):
        config["training"]["validation_instance_indices"] = indices
        with pytest.raises(ValueError):
            _select_training_validation_subsets(config, dataset, target_count=2, diagnostic_count=0)


def test_common_pool_matches_sampling_budget_and_preserves_failure_denominator():
    first = _row("a", {"a": 2., "b": 0.}, preference_key="0_0_1", sampling_repeat=0)
    second = _row("b", {"a": 3., "b": 1.}, method="policy_b", preference_key="0_0_1", sampling_repeat=0)
    failed = _row("failure", {"a": 0., "b": 0.}, preference_key="0_0_1", sampling_repeat=1,
                  task_succeeded=False, task_failed=True)
    failed_b = {**failed, "method": "policy_b", "candidate_id": "failure_b"}
    result = analyze_workload_pool([first, second, failed, failed_b])
    assert all(summary["completion_rate"] == .5 for summary in result["instances"])
    assert result["evaluation_quality_metric"]["workload_q"] == .5
    assert result["instances"][0]["hypervolume"] > result["instances"][1]["hypervolume"]
    with pytest.raises(ValueError, match="sampling budgets"):
        analyze_workload_pool([first, second, failed])
    with pytest.raises(ValueError, match="evaluation seeds"):
        analyze_workload_pool([{**first, "sampling_seed": 11}, second])


@pytest.mark.parametrize("field", ("dataset_manifest_sha256", "generator_config_sha256",
                                   "environment_config_sha256", "distribution_contract_sha256", "worker_loads"))
def test_common_pool_rejects_unmatched_physical_identity(field):
    first = _row("a", {"a": 2., "b": 0.}, preference_key="0_0_1", sampling_repeat=0)
    second = _row("b", {"a": 3., "b": 1.}, method="policy_b", preference_key="0_0_1", sampling_repeat=0)
    second[field] = {"different_worker": 3., "b": 1.} if field == "worker_loads" else "different"
    with pytest.raises(ValueError, match="physical identity|worker roster"):
        analyze_workload_pool([first, second])


def test_common_pool_rejects_missing_physical_identity():
    row = _row("a", {"a": 2., "b": 0.}, preference_key="0_0_1", sampling_repeat=0)
    row.pop("distribution_contract_sha256")
    with pytest.raises(ValueError, match="complete verified physical identity"):
        analyze_workload_pool([row])
