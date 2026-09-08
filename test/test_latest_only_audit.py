from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from agent.baselines import HeuristicPolicy
from configs import load_config, project_path
from data import load_instance_pickle
from environment import AssemblySchedulingEnv
from utils import action_trace_sha256


BASELINE = Path("test/baselines/latest_only_golden.json")


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


def test_fixed_instance_golden_observation_mask_and_trajectory():
    expected = json.loads(BASELINE.read_text(encoding="utf-8"))["fixed_instance"]
    config = load_config("configs/e1/single_flow.json")
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    mask = environment.get_action_mask()

    assert _observation_sha256(observation) == expected["initial_observation_sha256"]
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
                reward.completion_progress,
                reward.completion_bonus,
                reward.quality,
                reward.truncation,
                reward.unfinished,
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


def test_active_tree_contains_no_removed_experiment_control_strings():
    removed = (
        "E" + "2",
        "e" + "2_",
        "ablation_" + "gate",
        "m1_" + "gates",
        "teacher_" + "kl",
        "tiered_" + "gate",
        "warm_" + "start",
        "typed_" + "mlp",
    )
    roots = [Path("agent"), Path("configs"), Path("environment"), Path("training")]
    offenders = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".json"} or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for token in removed:
                if token in text:
                    offenders.append(f"{path}:{token}")
    assert offenders == []


def test_pre_refactor_expanded_configs_match_archived_fingerprints():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    for objective, expected in baseline["pre_refactor_expanded_config_sha256"].items():
        path = Path("test/baselines/pre_refactor_expanded") / f"{objective}.json"
        expanded = json.loads(path.read_text(encoding="utf-8"))
        payload = json.dumps(
            expanded, sort_keys=True, separators=(",", ":")
        ).encode()
        assert hashlib.sha256(payload).hexdigest() == expected


def test_only_latest_e1_and_mo_alns_configs_are_executable():
    json_files = {
        path.as_posix() for path in Path("configs").rglob("*.json")
    }
    assert json_files == {
        "configs/default.json",
        "configs/e1/single_flow.json",
        "configs/e1/single_cost.json",
        "configs/e1/single_variance.json",
        "configs/baselines/mo_alns.json",
        "configs/baselines/mo_alns_smoke.json",
        "configs/baselines/mo_alns_manifest.json",
        "configs/baselines/mo_alns_manifest.example.json",
    }
