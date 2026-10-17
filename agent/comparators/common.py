"""Shared experiment identity, atomic artifacts and schema-12 adapters."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from environment import ASSEMBLY_NODE_TYPES, CAPABLE_EDGE, SERVICE_CANDIDATE_EDGE, DecisionType

VERSION = "comparison_suite_v1"
METHODS = ("ppo", "dcan", "rule_ddqn", "nsga2_ls", "moead", "mo_alns")
TRAIN_STEPS = 350411
PILOT_STEPS = 35041
SEEDS = (11, 23, 37)
FOUR = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (1/3, 1/3, 1/3))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def method_config(config, method, seed=11, learning_rate=None):
    if method not in METHODS:
        raise ValueError(f"unknown comparison method: {method}")
    result = deepcopy(config)
    result["seed"] = seed
    result.setdefault("comparison", {}).update(version=VERSION, method=method)
    result["comparison"]["delivery_protocol"] = "greedy_single_v1" if method == "rule_ddqn" else "best_of_3_v1" if method in ("ppo", "dcan") else "independent_search_v1"
    if method == "rule_ddqn":
        result["training"]["formal_evaluation"]["sample_aggregation"] = "per_sample_v1"
    result["training"]["forced_action_compression"] = False
    if learning_rate is not None:
        result["ppo"]["learning_rate"] = learning_rate
        result["comparison"]["learning_rate"] = learning_rate
    from configs.runtime import runtime_manifest
    result["runtime_manifest"] = runtime_manifest(result)
    return result


def observation_spec(obs):
    return {"schema": 12, "node_names": {k: list(obs.node_feature_names[k]) for k in ASSEMBLY_NODE_TYPES},
            "global_names": list(obs.global_feature_names), "wait_names": list(obs.action_set_feature_names),
            "edges": {"|".join(k): list(v.feature_names) for k, v in obs.relations.items()}}


def physical_identity(config):
    # Excludes device and algorithm controls, includes every physical/objective input.
    return digest({k: config[k] for k in ("environment", "reward", "objective_scalarizer", "workload_objective")})


def candidates(obs):
    """Sparse candidate -> canonical native action index, never a new action space."""
    production = obs.decision_type == DecisionType.PRODUCTION
    edge = obs.relations[CAPABLE_EDGE if production else SERVICE_CANDIDATE_EDGE]
    left, right = ("operation", "machine") if production else ("machine", "worker")
    width = len(obs.node_features[right])
    ids = edge.edge_index[0] * width + edge.edge_index[1]
    return edge, left, right, ids.astype(np.int64)


def feature_column(edge, name):
    if name not in edge.feature_names:
        raise ValueError(f"comparison feature missing: {name}")
    return edge.edge_features[:, edge.feature_names.index(name)]


def pooled_state(obs):
    parts = []
    for kind in ASSEMBLY_NODE_TYPES:
        values = obs.node_features[kind]
        parts.extend((values.mean(0) if len(values) else np.zeros(values.shape[1]),
                      values.max(0) if len(values) else np.zeros(values.shape[1])))
    parts.extend((obs.global_features, obs.preference,
                  np.array([obs.decision_type == DecisionType.PRODUCTION, obs.decision_type == DecisionType.WORKER]),
                  obs.action_set_features))
    return np.concatenate(parts).astype(np.float32)
