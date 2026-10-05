"""Replay the schema-8 processing alias against the current observation contract."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from configs import load_config
from configs.config import public_config
from data.models import (
    AssemblyInstance, OperationSpec, OrderSpec, instance_to_dict,
    load_instance_yaml, validate_instance,
)
from environment import AssemblySchedulingEnv
from environment.observation_schema import OBSERVATION_SCHEMA_VERSION


def build_alias_instance(template: AssemblyInstance) -> AssemblyInstance:
    """Use standard resources, three waves, and processing times from 8 to 14."""
    blocker_machines = (1, 2, 4, 5, 7)
    machines = tuple(
        replace(machine, initial_module="A3" if index in blocker_machines else "A1",
                module_parameters={module: replace(spec, processing_speed_factor=1.0)
                                   for module, spec in machine.module_parameters.items()})
        for index, machine in enumerate(template.machines)
    )

    def order(name: str, wave: str, release: float, tasks: list[tuple[str, float]]) -> OrderSpec:
        return OrderSpec(name, wave, release, tuple(
            OperationSpec(f"{name}_{sequence}", name, sequence, module, duration)
            for sequence, (module, duration) in enumerate(tasks, start=1)
        ))

    orders = [order("A", "W1", 0, [("A1", 12), ("A3", 12), ("A3", 12)]),
              order("B", "W1", 0, [("A1", 12), ("A3", 8), ("A3", 8), ("A3", 8)])]
    orders += [order(f"C{i}", "W1", 0, [("A3", 14)] * 3) for i in range(5)]
    orders += [order("D", "W1", 2, [("A1", 8)] * 3)]
    orders += [order(f"E{i}", "W2" if i < 2 else "W3", release, [("A2", 8)] * 3)
               for i, release in enumerate((48, 52, 120, 126))]
    waves = {
        name: {"dominant_module": "A3" if name == "W1" else "A2",
               "order_ids": [item.id for item in orders if item.wave == name],
               "release_interval": [min(item.release_time for item in orders if item.wave == name),
                                    max(item.release_time for item in orders if item.wave == name)]}
        for name in ("W1", "W2", "W3")
    }
    instance = replace(template, instance_id="schema8_processing_assignment_alias", instance_type="test",
                       machines=machines, orders=tuple(orders), waves=waves)
    validate_instance(instance)
    return instance


def reachable_histories(config: dict, instance: AssemblyInstance):
    states = []
    paths = []
    for first, second in (("A_1", "B_1"), ("B_1", "A_1")):
        env = AssemblySchedulingEnv(deepcopy(config))
        env.reset(instance, preference=(1/3, 1/3, 1/3))
        path = []

        def step(action):
            assert not env.task_done and not env.get_action_mask()[action]
            path.append(int(action))
            env.step(action, build_observation=False)

        step(env.encode_production_action(instance.operation_index[first], 0))
        for index, machine in enumerate((1, 2, 4, 5, 7)):
            step(env.encode_production_action(instance.operation_index[f"C{index}_1"], machine))
        step(env.wait_action)  # D arrives at t=2.
        step(env.encode_production_action(instance.operation_index[second], 3))
        assert env.current_time == 2.0
        states.append(env)
        paths.append(path)
    return states, paths


def compare_observations(first, second) -> dict[str, bool]:
    return {
        "decision_type": first.decision_type == second.decision_type,
        "node_ids": first.node_ids == second.node_ids,
        "node_feature_names": first.node_feature_names == second.node_feature_names,
        "global_feature_names": first.global_feature_names == second.global_feature_names,
        "action_set_feature_names": first.action_set_feature_names == second.action_set_feature_names,
        "nodes": (first.node_features.keys() == second.node_features.keys() and all(
            np.array_equal(first.node_features[k], second.node_features[k]) for k in first.node_features)),
        "globals": np.array_equal(first.global_features, second.global_features),
        "preference": np.array_equal(first.preference, second.preference),
        "action_set": np.array_equal(first.action_set_features, second.action_set_features),
        "relations": (first.relations.keys() == second.relations.keys() and all(
            first.relations[k].feature_names == second.relations[k].feature_names
            and first.relations[k].bidirectional == second.relations[k].bidirectional
            and np.array_equal(first.relations[k].edge_index, second.relations[k].edge_index)
            and np.array_equal(first.relations[k].edge_features, second.relations[k].edge_features)
            for k in first.relations)),
    }


def observation_sha256(observation) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"phase": observation.decision_type.value,
                             "node_ids": observation.node_ids,
                             "node_names": observation.node_feature_names,
                             "global_names": observation.global_feature_names,
                             "action_names": observation.action_set_feature_names},
                            sort_keys=True).encode())
    arrays = [observation.global_features, observation.preference, observation.action_set_features]
    arrays += [observation.node_features[k] for k in sorted(observation.node_features)]
    for key in sorted(observation.relations):
        edge = observation.relations[key]
        digest.update(json.dumps([key, edge.feature_names, edge.bidirectional]).encode())
        arrays += [edge.edge_index, edge.edge_features]
    for array in arrays:
        digest.update(json.dumps([str(array.dtype), array.shape]).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def audit(config: dict) -> tuple[dict, AssemblyInstance]:
    template = load_instance_yaml(config["paths"]["fixed_instance"])
    instance = build_alias_instance(template)
    envs, paths = reachable_histories(config, instance)
    before = [env.observe() for env in envs]
    from scripts.verify_resource_relation_behavior import legacy_projection
    legacy_before = [legacy_projection(obs) for obs in before]
    legacy_equal = compare_observations(*legacy_before)
    equality = compare_observations(*before)
    masks = [env.get_action_mask() for env in envs]
    equality["action_mask"] = np.array_equal(*masks)
    assignments = [{name: env.operations[instance.operation_index[name]].machine_id
                    for name in ("A_1", "B_1")} for env in envs]
    checks = []
    for action in np.flatnonzero(~masks[0] & ~masks[1]):
        forks = [deepcopy(env) for env in envs]
        outcomes = [env.step(int(action)) for env in forks]
        vectors = [result[1].as_dict() for result in outcomes]
        scalar = [result[1].scalarize(config["reward"]) for result in outcomes]
        next_equal = compare_observations(outcomes[0][0], outcomes[1][0])
        checks.append({"action": int(action), "is_wait": int(action) == envs[0].wait_action,
                       "rewards": vectors, "scalar_rewards": scalar,
                       "reward_difference": scalar[0] - scalar[1],
                       "reward_exactly_equal": vectors[0] == vectors[1],
                       "next_observation_equal": all(next_equal.values()),
                       "next_observation_equality": next_equal,
                       "terminated": [result[2] for result in outcomes],
                       "truncated": [result[3] for result in outcomes],
                       "next_time": [env.current_time for env in forks]})
    return {"observation_schema": OBSERVATION_SCHEMA_VERSION,
            "instance_id": instance.instance_id, "order_count": len(instance.orders),
            "operation_count": len(instance.operations), "current_time": envs[0].current_time,
            "same_config_instance_preference": True,
            "comparison_method": "np.array_equal; all observation fields and action mask",
            "observations_equal": all(equality.values()), "observation_equality": equality,
            "schema8_projection_equal": all(legacy_equal.values()),
            "schema8_projection_hashes": [observation_sha256(o) for o in legacy_before],
            "observation_hashes": [observation_sha256(o) for o in before],
            "legal_histories": paths, "processing_assignments": assignments,
            "current_objectives": [env._objective_vector() for env in envs],
            "same_action_checks": checks,
            "reward_counterexample_found": all(equality.values()) and any(
                abs(item["reward_difference"]) > 1e-10 for item in checks)}, instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output-dir", default="result/analysis/observation_reward_schema9")
    args = parser.parse_args()
    config = load_config(args.config)
    result, instance = audit(config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("counterexample.json", result), ("instance.json", instance_to_dict(instance)),
                        ("config.json", public_config(config))):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
