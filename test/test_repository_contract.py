from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from agent.baselines import HeuristicPolicy
from configs import load_config, project_path
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv
from utils import action_trace_sha256


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


def test_fixed_instance_physics_matches_preserved_golden():
    expected = json.loads(Path("test/baselines/schema12_physics_golden.json").read_text(encoding="utf-8"))["fixed_instance"]
    config = load_config("configs/controls/raw_single_variance.json")
    instance = load_instance_yaml(project_path(config["paths"]["fixed_instance"]))
    env = AssemblySchedulingEnv(config)
    env.reset(instance)
    assert hashlib.sha256(np.ascontiguousarray(env.get_action_mask()).tobytes()).hexdigest() == expected["initial_mask_sha256"]
    actions = []
    policy = HeuristicPolicy()
    while not env.task_done:
        action = policy.select_action(env)
        actions.append(action)
        env.step(action)
    assert action_trace_sha256(actions) == expected["heuristic_action_trace_sha256"]
    assert len(actions) == expected["heuristic_action_count"]
    assert env.metrics()["terminal_reason"] == expected["terminal_reason"]
    for name, value in expected["metrics"].items():
        assert env.metrics()[name] == pytest.approx(value)
    assert env.validate_schedule() == []
