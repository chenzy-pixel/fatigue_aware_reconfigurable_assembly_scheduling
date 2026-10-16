from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from agent.baselines import HeuristicPolicy
from configs import load_config, project_path
from configs.normalization import apply_normalization_manifest
from configs.runtime import runtime_manifest
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv
from utils import action_trace_sha256


V8_BASELINE = Path("test/baselines/v8_golden.json")


def _verified_schema10_flow_config(tmp_path):
    """Restore the pinned legacy variance contract before checking old bytes."""
    baseline = json.loads(V8_BASELINE.read_text(encoding="utf-8"))
    config = json.loads(project_path("configs/default.json").read_text(encoding="utf-8"))
    config.pop("workload_objective", None)
    for field in tuple(config["network"]):
        if field.startswith("workload_") or field in {"objective_scales", "objective_names", "reward_version", "flow_mode", "normalization_manifest_sha256", "global_objective_encoding"}:
            config["network"].pop(field)
    config["network"]["observation_schema_version"] = 10
    config["preference"]["quality"].update(mode="fixed", fixed=[1.0, 0.0, 0.0])
    config["experiment_suite_version"] = "single_stage_progress_quality_failure_v3"
    config["reward"]["mode"] = config["experiment_suite_version"]
    config["evaluation"].update(result_schema_version="8.0.0", quality_metric=None)
    config["objective_scalarizer"].update(
        flow_mode="raw_v1", normalization_manifest="configs/manifests/v2_selected_scales_20261002.json",
        normalization_manifest_sha256=baseline["normalization_manifest_sha256"],
    )
    apply_normalization_manifest(config, project_root=project_path("."))
    config["runtime_manifest"] = runtime_manifest(config)
    saved = tmp_path / "schema10_flow.json"
    saved.write_text(json.dumps(config), encoding="utf-8")
    restored = load_config(saved)
    assert restored["runtime_manifest"] == config["runtime_manifest"]
    assert restored["network"]["observation_schema_version"] == 10
    assert restored["objective_scalarizer"]["scales"] == {"flow": 1089.15, "cost": 353.27, "variance": 2.2629}
    return restored


def _observation_sha256(observation) -> str:
    digest = hashlib.sha256()
    for name in sorted(observation.node_features):
        array = np.ascontiguousarray(observation.node_features[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    for edge_type in sorted(observation.relations):
        relation = observation.relations[edge_type]
        digest.update("|".join(edge_type).encode())
        for value in (relation.edge_index, relation.edge_features):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    digest.update(np.ascontiguousarray(observation.global_features).tobytes())
    digest.update(observation.decision_type.value.encode())
    return digest.hexdigest()


def test_committed_validation_instances_match_manifest_bytes(config):
    manifest_path = Path(config["paths"]["manifests_root"]) / "validation" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances_root = Path(config["paths"]["instances_root"]) / "validation"
    failures = []
    for entry in manifest["files"]:
        instance_path = instances_root / entry["path"]
        if not instance_path.is_file():
            failures.append(f"missing:{entry['path']}")
            continue
        actual = hashlib.sha256(instance_path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            failures.append(
                f"sha256:{entry['path']}:{actual}!={entry['sha256']}"
            )
    assert len(manifest["files"]) == manifest["instance_count"] == 500
    assert failures == []


def test_fixed_instance_schema10_golden_observation_mask_and_trajectory(tmp_path):
    expected = json.loads(V8_BASELINE.read_text(encoding="utf-8"))["fixed_instance"]
    config = _verified_schema10_flow_config(tmp_path)
    instance = load_instance_yaml(project_path(config["paths"]["fixed_instance"]))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    mask = environment.get_action_mask()

    assert _observation_sha256(observation) == expected['initial_observation_sha256']
    # Historical full hashes stay recorded; corrected derived features use schema 10.
    assert expected['schema9_initial_observation_sha256'] == '5a018fa1c1aa84798f56450232fabcda03983ff46fd42f61dd4523f9ebfd6e3d'
    assert hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest() == (
        expected["initial_mask_sha256"]
    )
    actions = []
    rewards = []
    policy = HeuristicPolicy()
    while not (environment.terminated or environment.truncated):
        action = policy.select_action(environment)
        actions.append(action)
        _, reward, _, _, _ = environment.step(action)
        rewards.append(
            [
                reward.flow,
                reward.cost,
                reward.variance,
                reward.operation_progress,
                reward.quality,
                reward.feasibility_shaping,
            ]
        )
    reward_digest = hashlib.sha256(
        json.dumps(rewards, separators=(",", ":")).encode()
    ).hexdigest()
    assert action_trace_sha256(actions) == expected["heuristic_action_trace_sha256"]
    assert len(actions) == expected["heuristic_action_count"]
    assert reward_digest == expected["heuristic_reward_trace_sha256"]
    assert environment.metrics()["terminal_reason"] == expected["terminal_reason"]
    for name, value in expected["metrics"].items():
        assert environment.metrics()[name] == pytest.approx(value)
    assert environment.validate_schedule() == []
    assert _observation_sha256(observation) == expected["initial_observation_sha256"]


def test_modern_workload_and_legacy_variance_preserve_same_physical_trajectory(tmp_path):
    legacy = _verified_schema10_flow_config(tmp_path)
    modern = load_config("configs/controls/raw.json")
    modern["preference"]["quality"].update(mode="fixed", fixed=[1.0, 0.0, 0.0])
    instance = load_instance_yaml(project_path(modern["paths"]["fixed_instance"]))
    first, second = AssemblySchedulingEnv(legacy), AssemblySchedulingEnv(modern)
    old, new = first.reset(instance), second.reset(instance)
    assert old.feature_dimensions["global"] == 9 and new.feature_dimensions["global"] == 10
    assert old.global_feature_names[-1] == "objective_committed_variance_norm"
    assert new.global_feature_names[-1] == "minimum_unfinished_order_slack_norm"
    policy = HeuristicPolicy()
    while not first.task_done:
        assert first.current_time == second.current_time
        assert first.decision_type == second.decision_type
        np.testing.assert_array_equal(first.get_action_mask(), second.get_action_mask())
        left, right = policy.select_action(first), policy.select_action(second)
        assert left == right
        _, old_reward, _, _, _ = first.step(left)
        _, new_reward, _, _, _ = second.step(right)
        assert old_reward.flow == new_reward.flow
        assert old_reward.cost == new_reward.cost
        assert old_reward.scalarize(legacy["reward"]) == new_reward.scalarize(modern["reward"])
        assert first.task_done == second.task_done
    assert first.schedule_log == second.schedule_log
    assert first.reconfiguration_log == second.reconfiguration_log
    for name in ("flow_time_objective", "reconfiguration_cost", "worker_load_variance", "completed_operations"):
        assert first.metrics()[name] == second.metrics()[name]
    assert first.validate_schedule() == second.validate_schedule() == []


ACTIVE_JSON_PATHS = tuple(sorted(
    path for path in project_path("configs").rglob("*.json")
    if "manifests" not in path.parts and "archive" not in path.parts
))


@pytest.mark.parametrize("path", ACTIVE_JSON_PATHS, ids=lambda path: path.relative_to(project_path(".")).as_posix())
def test_supported_training_and_baseline_configs_are_executable(path):
    from agent.ppo.network import build_actor_critic
    from environment.workload import objective_names, workload_config

    raw = json.loads(path.read_text(encoding="utf-8"))
    # Solver-budget manifests are executable through their referenced full
    # config, rather than through the experiment configuration loader itself.
    if "protocol" in raw:
        assert raw["protocol"] == "ppo_mo_alns_solver_budget_v2"
        assert raw["datasets"] and raw["algorithm_seeds"]
        config = load_config(raw["config"])
    else:
        config = load_config(path)
    assert config["runtime_manifest"]["observation_schema"] == 12
    assert config["runtime_manifest"]["objective_names"] == list(objective_names(config))
    assert config["runtime_manifest"]["workload_objective"] == workload_config(config)
    instance = load_instance_yaml(project_path(config["paths"]["fixed_instance"]))
    observation = AssemblySchedulingEnv(config).reset(instance)
    network = build_actor_critic(observation, {**config["network"], "hidden_dim": 16})
    assert network.objectives == ("flow", "cost", "workload")
    assert network.network_spec()["workload_mode"] == workload_config(config)["mode"]
