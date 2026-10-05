from collections import Counter
from copy import deepcopy
from dataclasses import replace

import pytest

from data.dataset import OnlineInstanceDataset, canonical_json_bytes, curriculum_weights_at, load_dataset_split
from data.distribution import PRESSURE_TYPES, protocol_hashes, severity_at, severity_bounds, training_sampling_plan, weighted_labels
from data.generate_orders import InstanceGenerator
from data.selection import resolve_instance_indices, select_validation_subsets, subset_snapshot
from environment.preference import quality_preference_for_episode
from eval import _evaluation_row, evaluate_dataset
from result.metrics import aggregate_evaluation_rows, evaluation_quality_metric
from scripts.prepare_data_v2 import verify_legacy_files
from train import _checkpoint_metadata


@pytest.mark.parametrize("completed,violations,makespan,status", [
    (False, [], 240, "unknown"), (True, [], 180, "observed_feasible"),
    (True, ["invalid schedule"], 180, "unknown"), (True, [], 241, "unknown"),
])
def test_diagnostics_never_resample(config, fixed_instance, monkeypatch, completed, violations, makespan, status):
    generator = InstanceGenerator(fixed_instance, config["generator"], config=config)
    candidate = generator.generate(seed=1_001_000, split="train", pressure_type="balanced").instance
    builds = []
    monkeypatch.setattr(generator, "_build_candidate", lambda **kw: builds.append(kw) or candidate)
    metrics = {"heuristic_completed": completed, "heuristic_truncated": not completed,
               "heuristic_terminal_reason": "completed" if completed else "horizon",
               "heuristic_makespan": makespan, "maximum_worker_fatigue": 0.5,
               "schedule_violations": violations, "ready_configuration_gap_ratio": 0}
    monkeypatch.setattr("data.generate_orders._rollout_metrics", lambda *a: (metrics, None))
    record = generator.generate(seed=2_001_000, split="validation", pressure_type="balanced")
    assert len(builds) == 1
    assert record.metadata["generation_attempt"] == 0
    assert record.metadata["generation_rejection_reasons"] == {}
    assert record.metadata["feasibility_status"] == status
    assert record.metadata["diagnostic_status"] == ("completed" if completed else "truncated")
    assert record.instance == candidate


def test_diagnostic_error_aborts_instead_of_resampling(config, fixed_instance, monkeypatch):
    generator = InstanceGenerator(fixed_instance, config["generator"], config=config)
    candidate = generator.generate(seed=1_001_000, split="train", pressure_type="balanced").instance
    builds = []
    monkeypatch.setattr(generator, "_build_candidate", lambda **kw: builds.append(kw) or candidate)
    def fail(*args):
        raise ValueError("unexpected diagnostic execution error")
    monkeypatch.setattr("data.generate_orders._rollout_metrics", fail)
    with pytest.raises(ValueError, match="unexpected diagnostic") as failure:
        generator.generate(seed=2_001_000, split="validation", pressure_type="balanced")
    assert len(builds) == 1
    assert "seed=2001000" in failure.value.__notes__[0]


def test_cost_stream_is_independent_of_scenario_and_severity(config, fixed_instance):
    generator = InstanceGenerator(fixed_instance, config["generator"], config=config)
    profiles = set()
    for scenario in PRESSURE_TYPES:
        for severity in (.4, 1):
            profiles.add(generator.generate(seed=1_001_002, split="train", pressure_type=scenario,
                severity=severity).metadata["cost_profile"])
    assert len(profiles) == 1


def test_fixed_build_is_worker_count_invariant(config, fixed_instance, tmp_path):
    from data.dataset import InstanceDataset, build_dataset_split
    paths = []
    for workers in (1, 2):
        root = tmp_path / str(workers)
        manifest = build_dataset_split(config=config, template=fixed_instance, split="validation", count=7,
            instances_root=root / "instances", manifests_root=root / "manifests", generation_workers=workers)
        paths.append((manifest, root / "instances"))
    assert paths[0][0].read_bytes() == paths[1][0].read_bytes()
    first, second = [InstanceDataset(manifest, instances_root=root) for manifest, root in paths]
    assert list(first) == list(second)


def test_training_diagnostics_are_disabled_and_seed_reproducible(config, fixed_instance, monkeypatch):
    generator = InstanceGenerator(fixed_instance, config["generator"], config=config)
    def fail(*args):
        raise AssertionError("training diagnostics should not run")
    monkeypatch.setattr("data.generate_orders._rollout_metrics", fail)
    args = dict(seed=1_001_001, split="train", pressure_type="fatigue_bottleneck", severity=0.4)
    first, second = generator.generate(**args), generator.generate(**args)
    assert canonical_json_bytes(first.to_dict()) == canonical_json_bytes(second.to_dict())
    assert first.metadata["heuristic_metrics"] is None
    assert first.metadata["diagnostic_status"] == "not_run"
    assert first.metadata["feasibility_status"] == "unknown"
    profile = config["generator"]["pressure_profiles"][args["pressure_type"]]
    low, high = severity_bounds(profile["order_count"], 0.4, integer=True)
    assert low <= len(first.instance.orders) <= high
    op_low, op_high = severity_bounds(profile["operations_per_order"], 0.4, integer=True)
    assert all(op_low <= len(order.operations) <= op_high for order in first.instance.orders)
    assert all(0.10 <= worker.initial_fatigue <= 0.16 + 1e-9 for worker in first.instance.workers)


def test_curriculum_anchors_windows_partial_window_and_streams(config):
    plan = training_sampling_plan(config, 2037)
    assert len(plan) == 2037
    for at, expected in ((0, .4), (.15, .4), (.375, .7), (.6, 1), (1, 1)):
        assert severity_at(config, at) == pytest.approx(expected)
    for start in range(0, len(plan), 100):
        end = min(start + 100, len(plan))
        expected = {name: 0.0 for name in PRESSURE_TYPES}
        for index in range(start, end):
            weights = curriculum_weights_at(config["generator"]["curriculum"], (index + .5) / len(plan))
            for name in PRESSURE_TYPES:
                expected[name] += weights[name] / (end - start)
        assert Counter(label for label, _ in plan[start:end]) == Counter(weighted_labels(end-start, expected))
    alternate = deepcopy(config)
    alternate["seed"] = 71
    alternate["training"]["parallel_envs"] = 2
    assert training_sampling_plan(alternate, 2037) == plan
    assert all(severity == 1 for _, severity in plan[1223:])
    assert severity_bounds([4, 5], .4, integer=True) == (4, 4)
    assert Counter(weighted_labels(50, config["generator"]["dataset_pressure_weights"])) == dict(zip(PRESSURE_TYPES, (3,18,7,7,5,5,5)))


def test_default_plan_covers_preferences_within_all_scenarios(config):
    coverage = set()
    for index, (label, _) in enumerate(training_sampling_plan(config, 2000)):
        point = quality_preference_for_episode(config, algorithm_seed=11, quality_episode_index=index)
        kind = point.source if point.source.startswith("endpoint_") else "mixed"
        coverage.add((label, kind))
    assert coverage == {(name, kind) for name in PRESSURE_TYPES for kind in ("endpoint_flow", "endpoint_cost", "endpoint_variance", "mixed")}


def test_cache_matches_fresh_and_configuration_changes_isolate(config, fixed_instance, tmp_path):
    current = deepcopy(config)
    current["paths"]["training_instances_cache"] = str(tmp_path / "cache")
    dataset = OnlineInstanceDataset(config=current, template=fixed_instance, episode_count=400)
    first, hit, digest, path = dataset.get_with_cache_info(150)
    assert not hit
    second, hit, other_digest, _ = dataset.get_with_cache_info(150)
    assert hit and digest == other_digest and first == second
    algorithm = deepcopy(current)
    algorithm["seed"] = 71
    assert OnlineInstanceDataset(config=algorithm, template=fixed_instance, episode_count=400)[150] == first
    different = deepcopy(current)
    different["environment"]["max_decisions"] += 1
    assert OnlineInstanceDataset(config=different, template=fixed_instance, episode_count=400).cache_directory != path.parent
    different = deepcopy(current)
    different["generator"]["severity_curriculum"]["anchors"][0]["value"] = .5
    assert OnlineInstanceDataset(config=different, template=fixed_instance, episode_count=400).cache_directory != path.parent
    assert protocol_hashes(different) == protocol_hashes(current)  # Shared fixed pool.


def test_stable_disjoint_validation_subsets_and_checkpoint(config):
    dataset = load_dataset_split(config, "validation")
    weights = config["generator"]["dataset_pressure_weights"]
    subsets = select_validation_subsets(dataset, weights)
    target, diagnostic = subsets["target"], subsets["diagnostic"]
    assert target["pressure_counts"] == dict(zip(PRESSURE_TYPES, (3,18,7,7,5,5,5)))
    assert diagnostic["pressure_counts"] == dict.fromkeys(PRESSURE_TYPES, 7)
    assert set(target["instance_indices"]).isdisjoint(diagnostic["instance_indices"])
    assert select_validation_subsets(dataset, weights) == subsets
    assert subset_snapshot(dataset, target["instance_indices"], role="evaluation")["subset_sha256"] == target["subset_sha256"]
    metadata = _checkpoint_metadata(config, role="best", episode=100, validation_split="validation", validation_instance_limit=50)
    assert metadata["validation_subset"] == target
    assert metadata["validation_instance_seeds"] == [dataset[index].metadata["seed"] for index in target["instance_indices"]]
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_instance_indices(dataset, instance_indices=[1,3], instance_limit=2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_instance_indices(dataset, instance_indices=[1,3], instance_offset=0)
    for invalid in ([1,1], [-1], [len(dataset)]):
        with pytest.raises(ValueError):
            resolve_instance_indices(dataset, instance_indices=invalid)


def test_unknown_instances_in_denominator_and_invalid_gap_is_empty(config, tmp_path):
    rows, _, _, _ = evaluate_dataset(config, dataset_name="validation", policy_name="heuristic", instance_indices=[0])
    dataset = load_dataset_split(config, "validation")
    original = dataset[0]
    from eval import evaluate_instance
    _, metrics = evaluate_instance(config, instance=original.instance, policy_name="heuristic")
    modified = replace(original, metadata={**original.metadata, "feasibility_status": "unknown", "heuristic_metrics": None})
    row = _evaluation_row(modified, metrics, config, evaluation_quality_metric(config))
    assert all(row[name] is None for name in ("relative_heuristic_gap_percent", "makespan_heuristic_gap_percent", "reconfiguration_cost_heuristic_gap_percent", "worker_load_variance_heuristic_gap_percent"))
    failed = {**row, "terminated": True, "truncated": False,
              "task_succeeded": False, "task_failed": True, "task_done": True,
              "sampling_truncated": False, "objective_complete": True,
              "termination_reason": "horizon", "operation_progress": .75}
    aggregate = aggregate_evaluation_rows([row,failed], dataset="validation", policy="heuristic", manifest="manifest.json")
    assert aggregate["completion_rate"] == .5
    assert aggregate["by_feasibility_status"]["unknown"]["count"] == 2
    assert aggregate["failure_reasons"] == {"horizon": 1}
    from result.io import write_csv
    import csv
    path = tmp_path / "rows.csv"
    write_csv(path, [row])
    with path.open(encoding="utf-8-sig") as file:
        assert next(csv.DictReader(file))["relative_heuristic_gap_percent"] == ""
    for field in ("subset_sha256", "distribution_contract_sha256", "objective_scale_flow"):
        with pytest.raises(ValueError, match=field):
            aggregate_evaluation_rows([rows[0], {**rows[0], field: "different"}], dataset="validation", policy="heuristic", manifest="manifest.json")
    with pytest.raises(ValueError, match="result schema"):
        aggregate_evaluation_rows([{**rows[0], "result_schema_version": "7.0.0"}], dataset="validation", policy="heuristic", manifest="manifest.json")


def test_legacy_data_is_preserved_and_loadable():
    import json
    from pathlib import Path
    from configs import load_config
    assert verify_legacy_files() == 565
    # Archived instance bytes remain readable under their recorded data contract.
    legacy = json.loads(Path("configs/archive/data_v1.json").read_text(encoding="utf-8"))
    assert load_dataset_split(legacy, "validation").manifest["schema_version"] == "1.2.0"
    for allow in (False, True):
        with pytest.raises(ValueError, match="failure-v3 reward contract"):
            load_config("configs/archive/data_v1.json", allow_observation_migration=allow)


def test_run_audit_loads_saved_runtime_config(config, tmp_path):
    from configs.config import public_config
    from result.io import write_json
    from scripts.audit_data_v2_runs import load_run_config
    saved = public_config(config)
    path = tmp_path / "config.json"
    write_json(path, saved)
    assert load_run_config(path) == saved
    saved["runtime_manifest"]["observation_schema"] = 0
    write_json(path, saved)
    with pytest.raises(ValueError, match="runtime manifest"):
        load_run_config(path)
