from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from agent.baselines import HeuristicPolicy
from configs import load_config, project_path
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv
from utils import action_trace_sha256


V8_BASELINE = Path("test/baselines/v8_golden.json")


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


def test_fixed_instance_golden_observation_mask_and_trajectory():
    expected = json.loads(V8_BASELINE.read_text(encoding="utf-8"))["fixed_instance"]
    config = load_config("configs/e1/single_flow.json")
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
    assert reward_digest == expected["heuristic_reward_trace_sha256"]
    assert environment.metrics()["terminal_reason"] == expected["terminal_reason"]
    assert environment.validate_schedule() == []
    assert _observation_sha256(observation) == expected["initial_observation_sha256"]


def test_supported_training_and_baseline_configs_are_executable():
    json_files = {
        path.as_posix() for path in Path("configs").rglob("*.json")
        if "manifests" not in path.parts and "archive" not in path.parts
    }
    assert json_files == {
        "configs/default.json",
        "configs/e1/single_flow.json",
        "configs/e1/single_cost.json",
        "configs/e1/single_variance.json",
        "configs/baselines/mo_alns.json",
        "configs/baselines/mo_alns_smoke.json",
        "configs/baselines/mo_alns_manifest.json",
        "configs/v8/universal.json",
        "configs/ablations/no_graph.json",
        "configs/ablations/shared_head.json",
        "configs/ablations/neutral_flow.json",
        "configs/ablations/neutral_cost.json",
        "configs/ablations/neutral_variance.json",
        "configs/ablations/full_flow.json",
        "configs/ablations/full_cost.json",
        "configs/ablations/full_variance.json",
        "configs/flow_excess/universal.json",
        "configs/flow_excess/single_flow.json",
        "configs/flow_excess/mo_alns.json",
        "configs/flow_excess/shared_head/universal.json",
        "configs/flow_excess/shared_head/single_flow.json",
        "configs/flow_excess/shared_head/single_cost.json",
        "configs/flow_excess/shared_head/single_variance.json",
    }
