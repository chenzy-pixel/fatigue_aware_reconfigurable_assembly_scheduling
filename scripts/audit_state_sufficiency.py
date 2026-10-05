"""Audit physical state observability and graph encoding within the current benchmark."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from agent.baselines import HeuristicPolicy, RandomPolicy
from configs import load_config
from configs.config import public_config
from data.dataset import load_generated_record
from data.models import OrderSpec, OperationSpec, load_instance_yaml, validate_instance, instance_to_dict
from environment import AssemblySchedulingEnv, DecisionType, EdgeStore
from environment.observation_schema import OBSERVATION_SCHEMA_VERSION
from scripts.verify_resource_relation_behavior import legacy_projection
from environment.types import OperationState, ReconfigurationStage
from scripts.audit_observation_reward import (
    build_alias_instance, reachable_histories, compare_observations, observation_sha256,
)

PROCESSING = ("operation", "processing_on", "machine")
SERVING = ("machine", "served_by", "worker")
MODES = ("full", "neutral")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def effective(config, mode):
    value = deepcopy(config)
    value["environment"]["fatigue_mode"] = mode
    return value


def current_relations(env):
    processing = [(i, env.instance.machine_index[op.machine_id]) for i, op in enumerate(env.operations)
                  if op.state == OperationState.PROCESSING]
    serving = []
    for record in env.reconfigurations.values():
        worker = (record.disassembly_worker_id if record.stage == ReconfigurationStage.DIS
                  else record.installation_worker_id if record.stage == ReconfigurationStage.INS else None)
        if worker:
            wi = next(i for i, item in enumerate(env.workers) if item.spec.id == worker)
            serving.append((env.instance.machine_index[record.machine_id], wi))

    def store(pairs):
        return EdgeStore(edge_index=np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2).T,
                         edge_features=np.empty((len(pairs), 0), dtype=np.float32),
                         feature_names=(), bidirectional=True)
    return {PROCESSING: store(processing), SERVING: store(serving)}


def virtual_observation(env):
    obs = env.observe()
    actual = current_relations(env)
    if all(kind in obs.relations for kind in actual):
        for kind, expected in actual.items():
            store = obs.relations[kind]
            assert np.array_equal(store.edge_index, expected.edge_index)
            assert store.edge_features.shape == expected.edge_features.shape
            assert store.feature_names == () and store.bidirectional
        return obs
    return replace(obs, relations={**obs.relations, **actual})


def worker_alias_instance(template):
    machines = tuple(template.machines)
    workers = tuple(replace(worker, initial_fatigue=.05) for worker in template.workers)
    definitions = [("A", "W1", 30, "A2"), ("B", "W1", 31, "A2"),
                   ("C", "W2", 36, "A1"), ("D", "W3", 120, "A3")]
    definitions += [(f"E{i}", "W1", 45 + i, "A2") for i in range(6)]
    definitions += [("F", "W2", 60, "A1"), ("G", "W3", 126, "A3")]
    orders = tuple(OrderSpec(name, wave, release, tuple(
        OperationSpec(f"{name}_{s}", name, s, module, 8) for s in (1, 2, 3)))
        for name, wave, release, module in definitions)
    waves = {wave: {"dominant_module": module, "order_ids": [o.id for o in orders if o.wave == wave],
                    "release_interval": [min(o.release_time for o in orders if o.wave == wave),
                                         max(o.release_time for o in orders if o.wave == wave)]}
             for wave, module in (("W1", "A2"), ("W2", "A1"), ("W3", "A3"))}
    instance = replace(template, instance_id="worker_service_assignment_alias", instance_type="test",
                       machines=machines, workers=workers, orders=orders, waves=waves)
    validate_instance(instance)
    return instance


def worker_histories(config, instance):
    states, paths = [], []
    for first, second in ((0, 3), (3, 0)):
        env = AssemblySchedulingEnv(deepcopy(config))
        env.reset(instance, preference=(1/3, 1/3, 1/3))
        path = []

        def step(action):
            assert not env.task_done and not env.get_action_mask()[action]
            path.append(int(action))
            env.step(action, build_observation=False)

        step(env.wait_action)
        step(env.encode_production_action(instance.operation_index["A_1"], 5))
        step(env.wait_action)
        step(env.encode_worker_action(5, 5))
        step(env.wait_action)
        step(env.encode_production_action(instance.operation_index["B_1"], 0))
        step(env.wait_action)
        step(env.encode_worker_action(0, 1))
        step(env.wait_action)
        step(env.wait_action)
        step(env.encode_worker_action(5, first))
        step(env.wait_action)
        step(env.wait_action)
        step(env.encode_worker_action(0, second))
        assert env.current_time == 36
        states.append(env)
        paths.append(path)
    return states, paths


def pair_comparison(states):
    obs = [e.observe() for e in states]
    originals = [legacy_projection(item) for item in obs]
    same = compare_observations(*originals)
    masks = [e.get_action_mask() for e in states]
    same["action_mask"] = np.array_equal(*masks)
    actions = np.flatnonzero(~masks[0] & ~masks[1])
    outcomes = []
    for action in actions:
        forks = [deepcopy(e) for e in states]
        transitions = [e.step(int(action)) for e in forks]
        rewards = [t[1].as_dict() for t in transitions]
        scalar = [t[1].scalarize(e.config["reward"]) for t, e in zip(transitions, forks)]
        next_same = compare_observations(transitions[0][0], transitions[1][0])
        outcomes.append({"action": int(action), "is_wait": int(action) == states[0].wait_action,
                         "rewards": rewards, "scalar_rewards": scalar,
                         "reward_difference": scalar[0] - scalar[1],
                         "next_observation_equal": all(next_same.values()),
                         "next_observation_equality": next_same,
                         "next_action_mask_equal": np.array_equal(forks[0].get_action_mask(), forks[1].get_action_mask()),
                         "terminated": [t[2] for t in transitions],
                         "sampling_truncated": [t[3] for t in transitions],
                         "next_time": [e.current_time for e in forks]})
    proposed = [virtual_observation(e) for e in states]
    return {"observation_schema": OBSERVATION_SCHEMA_VERSION,
            "original_equality": same, "original_equal": all(same.values()),
            "hashes": [observation_sha256(o) for o in originals],
            "current_observation_equal": all(compare_observations(*obs).values()),
            "current_hashes": [observation_sha256(o) for o in obs],
            "virtual_relation_equality": compare_observations(*proposed),
            "current_objectives": [e._objective_vector() for e in states],
            "current_completed_loads": [[w.load for w in e.workers] for e in states],
            "current_committed_loads": [e._committed_worker_loads.tolist() for e in states],
            "common_legal_action_count": len(actions), "outcomes": outcomes}


def encoding_checks(config, fixtures):
    """Check actual resource relations, with virtual types for historical replays."""
    import environment.types as observation_types
    import agent.ppo.network as network_module
    old_types = observation_types.ASSEMBLY_EDGE_TYPES
    old_network_types = network_module.ASSEMBLY_EDGE_TYPES
    old_bidirectional = network_module.BIDIRECTIONAL_EDGE_TYPES
    observation_types.ASSEMBLY_EDGE_TYPES = old_types + tuple(k for k in (PROCESSING, SERVING) if k not in old_types)
    network_module.ASSEMBLY_EDGE_TYPES = old_network_types + tuple(k for k in (PROCESSING, SERVING) if k not in old_network_types)
    network_module.BIDIRECTIONAL_EDGE_TYPES = old_bidirectional | {PROCESSING, SERVING}
    results = []
    try:
        for kind, (instance, builder) in fixtures.items():
            for mode in MODES:
                states, _ = builder(effective(config, mode), instance)
                observations = [virtual_observation(e) for e in states]
                masks = [e.get_action_mask() for e in states]
                for variant in ("hetero_gnn_objective_experts", "hetero_gnn_shared_preference", "node_mlp_pool"):
                    for seed in (0, 1, 11, 23, 37):
                        torch.manual_seed(seed)
                        settings = {**config["network"], "encoder_variant": "node_mlp_pool" if variant == "node_mlp_pool" else "hetero_gnn",
                                    "actor_head_variant": "shared_preference" if variant.endswith("shared_preference") else "objective_experts"}
                        model = network_module.build_actor_critic(observations[0], settings).eval()
                        with torch.no_grad():
                            batch, nodes, _, context = model.encode_graph(observations, device="cpu")
                            logits, values = model.forward_batch(observations, masks, device="cpu")
                            double_model = deepcopy(model).double()
                            double_nodes = {name: double_model.node_projectors[name](features.double())
                                            for name, features in batch.node_features.items()}
                            relations = {key: (indices, features.double(), direction)
                                         for key, (indices, features, direction) in batch.relations.items()}
                            for layer in double_model.message_layers:
                                double_nodes = layer(double_nodes, relations)
                            for layer in double_model.node_mlp_layers:
                                double_nodes = {name: layer[name](features) for name, features in double_nodes.items()}
                            double_context = torch.cat(tuple(double_model._pool_slices(double_nodes[name], batch.node_slices[name])
                                                             for name in network_module.NODE_TYPES)
                                                       + (double_model.global_encoder(batch.global_features.double()),), dim=-1)
                        differences = {}
                        for name, features in nodes.items():
                            first, second = batch.node_slices[name]
                            differences[name] = float((features[first[0]:first[1]]-features[second[0]:second[1]]).abs().max())
                        mask = ~torch.as_tensor(masks[0])
                        results.append({"kind": kind, "mode": mode, "variant": variant, "seed": seed,
                                        "graph_difference_float32": float((context[0]-context[1]).abs().max()),
                                        "graph_difference_float64": float((double_context[0]-double_context[1]).abs().max()),
                                        "critic_difference": float((values[0]-values[1]).abs()),
                                        "legal_actor_difference": float((logits[0, mask]-logits[1, mask]).abs().max()),
                                        "node_differences": differences})
    finally:
        observation_types.ASSEMBLY_EDGE_TYPES = old_types
        network_module.ASSEMBLY_EDGE_TYPES = old_network_types
        network_module.BIDIRECTIONAL_EDGE_TYPES = old_bidirectional
    return results


def boundary_checks(config, template):
    """Exercise ties, partial settlement and external guard behavior via legal steps."""
    rows = []
    for mode in MODES:
        settings = effective(config, mode)
        reference = AssemblySchedulingEnv(settings)
        reference.reset(template, build_observation=False)
        policy = HeuristicPolicy()
        path = []
        partial_path = []
        ties = 0
        while not reference.task_done:
            action = policy.select_action(reference)
            if not partial_path and reference.decision_type == DecisionType.WORKER and action != reference.wait_action:
                mi, wi = reference.decode_worker_action(action)
                rec = reference._pending_reconfiguration(reference.machines[mi].spec.id)
                end = reference.current_tick + reference._stage_duration_ticks(rec, reference.workers[wi])
                if end > reference.current_tick + 1:
                    partial_path = path + [int(action)]
            if action == reference.wait_action:
                opportunity = reference._wait_opportunity()
                if opportunity:
                    ties += int(sum(e[0] == opportunity[0] for e in reference._events) > 1)
            path.append(int(action))
            reference.step(action, build_observation=False)
        assert reference.task_succeeded
        count = len(path)
        for guard, maximum in (("max_decisions", count), ("max_decisions", count-1), ("max_zero_time_actions", 1)):
            guarded_config = deepcopy(settings)
            guarded_config["environment"][guard] = maximum
            guarded = AssemblySchedulingEnv(guarded_config)
            guarded.reset(template, build_observation=False)
            steps = []
            reward_sum = 0.0
            while not guarded.task_done:
                action = policy.select_action(guarded)
                steps.append(int(action))
                _, reward, _, _, _ = guarded.step(action, build_observation=False)
                reward_sum += reward.scalarize(guarded_config["reward"])
            rows.append({"mode": mode, "guard": guard, "limit": maximum,
                         "terminated": guarded.terminated, "sampling_truncated": guarded.truncated,
                         "succeeded": guarded.task_succeeded, "failed": guarded.task_failed,
                         "reason": guarded.terminal_reason, "steps": len(steps),
                         "failure_penalty": guarded.metrics()["terminal_failure_penalty_applied"],
                         "reward_sum": reward_sum})
        # Probe partial real failure on an independently valid fixed-horizon fixture.
        # At the first worker commitment, replay the prefix and force the existing
        # physical failure API at that exact state, separately from legal history evidence.
        env = AssemblySchedulingEnv(settings)
        env.reset(template, build_observation=False)
        for action in partial_path:
            env.step(action, build_observation=False)
        obs = virtual_observation(env)
        recovered = reconstructed_quantities(env, obs)
        ledgers = env._committed_worker_loads.copy()
        active = dict(env._active_committed_worker_tasks)
        env._apply_truncation("horizon")
        assert env.task_failed
        expected = ledgers.copy()
        for (rec_id, phase), (wi, duration) in active.items():
            rec = env.reconfigurations[rec_id]
            prefix = "disassembly" if phase == "DIS" else "installation"
            worked = max(0, min(env.current_tick, getattr(rec, prefix+"_end_tick"))-getattr(rec, prefix+"_start_tick"))*env.resolution
            expected[wi] -= duration-worked
        rows.append({"mode": mode, "kind": "partial_failure_api_probe", "legal_prefix": partial_path,
                     "legal_terminal_transition": False, "coincident_wait_count_reference": ties,
                     "settled_load_max_error": float(np.max(np.abs(expected-env._committed_worker_loads))),
                     "reconstructed_planned_load_max_error": float(np.max(np.abs(recovered["committed"]-ledgers))),
                     "active_task_count": len(active)})
        for phase in ("DIS", "INS", "completion_at_horizon"):
            module = "A1" if phase == "completion_at_horizon" else "A2"
            order = OrderSpec("boundary", "W1", 0, (OperationSpec("boundary_1", "boundary", 1, module, 8),))
            machines = []
            for i, machine in enumerate(template.machines):
                specs = dict(machine.module_parameters)
                if i == 0:
                    specs = {k: replace(v, processing_speed_factor=1) for k, v in specs.items()}
                    specs["A1"] = replace(specs["A1"], disassembly_base_time=5 if phase == "DIS" else 1)
                    specs["A2"] = replace(specs["A2"], installation_base_time=5)
                machines.append(replace(machine, module_parameters=specs))
            fixture = replace(template, instance_id=f"boundary_{phase}", instance_type="test",
                              horizon=8 if phase == "completion_at_horizon" else 4,
                              machines=tuple(machines), orders=(order,),
                              waves={"W1": {"dominant_module": module, "order_ids": [order.id], "release_interval": [0, 0]}})
            validate_instance(fixture)
            env = AssemblySchedulingEnv(settings)
            env.reset(fixture)
            path = []
            reward_sum = 0.0
            while not env.task_done:
                if env.decision_type == DecisionType.PRODUCTION:
                    action = env.encode_production_action(0, 0) if env.operations[0].state == OperationState.READY else env.wait_action
                else:
                    allowed = np.flatnonzero(~env.get_action_mask()[:-1])
                    action = int(allowed[0]) if len(allowed) else env.wait_action
                path.append(action)
                _, reward, _, _, _ = env.step(action)
                reward_sum += reward.scalarize(settings["reward"])
            final = virtual_observation(env)
            final.validate()
            rows.append({"kind": phase, "mode": mode, "legal_terminal_transition": True,
                         "legal_histories": path, "time": env.current_time, "reason": env.terminal_reason,
                         "succeeded": env.task_succeeded, "failed": env.task_failed,
                         "terminated": env.terminated, "sampling_truncated": env.truncated,
                         "worker_loads": [w.load for w in env.workers],
                         "committed_loads": env._committed_worker_loads.tolist(),
                         "actual_processing_edges": final.relations[PROCESSING].num_edges,
                         "actual_serving_edges": final.relations[SERVING].num_edges,
                         "failure_penalty": env.metrics()["terminal_failure_penalty_applied"],
                         "reward_sum": reward_sum})
    return rows


def event_order_checks(config, template):
    rows = []
    for mode in MODES:
        settings = effective(config, mode)
        machines = tuple(replace(machine, module_parameters={k: replace(v, processing_speed_factor=1)
                                                              for k, v in machine.module_parameters.items()})
                         for machine in template.machines)
        orders = (OrderSpec("A", "W1", 0, (OperationSpec("A_1", "A", 1, "A1", 12),)),
                  OrderSpec("B", "W1", 0, (OperationSpec("B_1", "B", 1, "A1", 12),)),
                  OrderSpec("C", "W1", 12, (OperationSpec("C_1", "C", 1, "A1", 8),)))
        instance = replace(template, instance_id="same_tick_event_order", instance_type="test", machines=machines,
                           orders=orders, waves={"W1": {"dominant_module": "A1", "order_ids": [o.id for o in orders],
                                                         "release_interval": [0, 12]}})
        validate_instance(instance)
        states = []
        for ordering in ((0, 1), (1, 0)):
            env = AssemblySchedulingEnv(settings)
            env.reset(instance)
            for oi in ordering:
                env.step(env.encode_production_action(oi, 0 if oi == 0 else 3), build_observation=False)
            states.append(env)
        result = pair_comparison(states)
        assert result["original_equal"]
        assert all(abs(item["reward_difference"]) < 1e-12 and item["next_observation_equal"]
                   for item in result["outcomes"])
        rows.append({"mode": mode, "kind": "irrelevant_event_serial_order", **result})
    return rows


def precision_checks(config, template):
    instance = worker_alias_instance(template)
    changed = replace(instance, workers=(replace(instance.workers[0], labor_cost_per_minute=instance.workers[0].labor_cost_per_minute+1e-8), *instance.workers[1:]))
    validate_instance(changed)
    first = worker_histories(config, instance)[0][0]
    second = worker_histories(config, changed)[0][0]
    equality = compare_observations(first.observe(), second.observe())
    full_relation_equality = compare_observations(virtual_observation(first), virtual_observation(second))
    rewards = [e.step(e.wait_action)[1] for e in (first, second)]
    return {"kind": "float32_cost_precision", "same_instance": False, "same_physical_structure": True,
            "labor_rate_difference": 1e-8, "original_equal": all(equality.values()),
            "virtual_relation_equal": all(full_relation_equality.values()),
            "cost_rewards": [r.cost for r in rewards], "scalar_rewards": [r.scalarize(config["reward"]) for r in rewards],
            "scalar_difference": rewards[0].scalarize(config["reward"])-rewards[1].scalarize(config["reward"])}


def scan_datasets(config, output):
    groups = {}
    constants = {}
    feature_ranges = {"order_count": [], "operation_count": [], "machine_count": [], "worker_count": []}
    manifests = []
    count = 0
    for split in ("validation", "test", "ood", "stress"):
        path = Path(config["paths"]["manifests_root"])/split/"manifest.json"
        raw = path.read_bytes()
        manifest = json.loads(raw)
        manifests.append({"split": split, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                          "count": manifest["instance_count"]})
        split_root = Path(config["paths"]["instances_root"])/split
        for entry in manifest["files"]:
            file = split_root/entry["path"]
            payload = file.read_bytes()
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"], file
            record = json.loads(payload)
            instance, metadata = record["instance"], record["metadata"]
            fixed = {"horizon": instance["time"]["horizon"], "resolution": instance["time"]["resolution"],
                     "fatigue": instance["fatigue"],
                     "unfinished_order_penalty": instance["truncation"]["unfinished_order_penalty_per_order"]}
            # initial fatigue is repeated in the legacy serialization but varies per worker.
            fixed["fatigue"] = {k: v for k, v in fixed["fatigue"].items() if k != "initial_fatigue"}
            fingerprint = json.dumps(fixed, sort_keys=True)
            constants[fingerprint] = constants.get(fingerprint, 0) + 1
            for key, items in (("order_count", instance["orders"]),
                               ("operation_count", [op for order in instance["orders"] for op in order["operations"]]),
                               ("machine_count", instance["machines"]), ("worker_count", instance["workers"])):
                feature_ranges[key].append(len(items))
            cell = (split, metadata["pressure_type"], metadata.get("ood_factor"))
            current = {"split": split, "pressure_type": metadata["pressure_type"],
                       "ood_factor": metadata.get("ood_factor"), "seed": metadata["seed"],
                       "path": str(file), "sha256": entry["sha256"]}
            if cell not in groups or current["seed"] < groups[cell]["seed"]:
                groups[cell] = current
            count += 1
    summary = {"record_count": count, "manifest_checks": manifests,
               "constant_groups": [{"parameters": json.loads(key), "count": value} for key, value in constants.items()],
               "size_ranges": {key: {"min": min(values), "max": max(values)} for key, values in feature_ranges.items()},
               "selected_cells": sorted(groups.values(), key=lambda item: (item["split"], item["pressure_type"], str(item["ood_factor"]))) }
    write_json(output/"dataset_scan.json", summary)
    print(f'Dataset scan: {count} records, {len(groups)} selected cells, {len(constants)} constant groups', flush=True)
    return summary


def reconstructed_quantities(env, obs):
    """Recover reward-relevant quantities using the two proposed relations."""
    horizon = env.instance.horizon
    names = obs.node_feature_names

    def column(kind, name):
        return obs.node_features[kind][:, names[kind].index(name)].astype(np.float64)

    tick_ratio = env.instance.resolution/horizon
    current_tick = int(round(float(obs.global_features[0])/tick_ratio))
    machine_remaining = np.rint(column("machine", "remaining_busy_time_norm")/tick_ratio).astype(np.int64)
    worker_remaining = np.rint(column("worker", "remaining_busy_time_norm")/tick_ratio).astype(np.int64)
    loads = column("worker", "load_norm")*horizon
    committed = loads.copy()
    services = obs.relations[SERVING].edge_index
    locked = obs.relations[("operation", "locked_to", "machine")]
    elapsed_col = locked.feature_names.index("stage_elapsed_time_norm")
    stage_lengths = {}
    for mi, wi in services.T:
        pos = np.flatnonzero(locked.edge_index[1] == mi)
        assert len(pos) == 1
        elapsed = int(round(float(locked.edge_features[pos[0], elapsed_col])/tick_ratio))
        duration = elapsed + int(machine_remaining[mi])
        committed[wi] += duration*env.instance.resolution
        stage_lengths[int(mi)] = duration
        assert worker_remaining[wi] == machine_remaining[mi]
    progress = float(np.mean(column("order", "completion_ratio")))
    scales = env.config["objective_scalarizer"]["scales"]
    objectives = obs.global_features[6:9].astype(np.float64)*[scales[k] for k in ("flow", "cost", "variance")]
    active = int(np.sum(column("order", "released") - column("order", "completed")))
    return {"tick": current_tick, "machine_remaining": machine_remaining, "worker_remaining": worker_remaining,
            "loads": loads, "committed": committed, "progress": progress,
            "objectives": objectives, "active": active, "stage_lengths": stage_lengths}


def reconstructed_events(env, obs, quantities):
    """Recover pending physical event payloads using virtual schema-9 relations."""
    events = []
    tick = quantities["tick"]
    ids = obs.node_ids
    operation_names = obs.node_feature_names["operation"]
    order_names = obs.node_feature_names["order"]
    op_order = obs.relations[("operation", "belongs_to", "order")].edge_index
    release_col = operation_names.index("order_release_time_norm")
    for oi in range(len(ids["order"])):
        if not obs.node_features["order"][oi, order_names.index("released")]:
            operations = op_order[0, op_order[1] == oi]
            release = int(round(float(obs.node_features["operation"][operations[0], release_col])
                                * env.instance.horizon / env.instance.resolution))
            events.append((release, "ORDER_RELEASE", ids["order"][oi]))
    for op, machine in obs.relations[PROCESSING].edge_index.T:
        events.append((tick+int(quantities["machine_remaining"][machine]), "PROCESS_COMPLETE",
                       ids["operation"][op], ids["machine"][machine]))
    locked = obs.relations[("operation", "locked_to", "machine")]
    for machine, worker in obs.relations[SERVING].edge_index.T:
        pos = np.flatnonzero(locked.edge_index[1] == machine)[0]
        op = locked.edge_index[0, pos]
        dis = locked.edge_features[pos, locked.feature_names.index("stage_DIS")] == 1
        events.append((tick+int(quantities["machine_remaining"][machine]), "DIS_COMPLETE" if dis else "INS_COMPLETE",
                       ids["operation"][op], ids["machine"][machine], ids["worker"][worker]))
    return sorted(events)


def actual_events(env):
    events = []
    for tick, _, _, kind, payload in env._events:
        if kind.value == "ORDER_RELEASE":
            events.append((tick, kind.value, payload["order_id"]))
        elif kind.value == "PROCESS_COMPLETE":
            events.append((tick, kind.value, payload["operation_id"], payload["machine_id"]))
        else:
            rec = env.reconfigurations[payload["reconfiguration_id"]]
            events.append((tick, kind.value, rec.operation_id, rec.machine_id, payload["worker_id"]))
    return sorted(events)


def physical_signature(env):
    active = []
    for rec in env.reconfigurations.values():
        if rec.stage == ReconfigurationStage.DONE:
            continue
        active.append((rec.machine_id, rec.operation_id, rec.stage.value, rec.source_module, rec.target_module,
                       env._reconfiguration_stage_start_tick(rec),
                       rec.disassembly_worker_id if rec.stage == ReconfigurationStage.DIS else
                       rec.installation_worker_id if rec.stage == ReconfigurationStage.INS else None))
    data = {"tick": env.current_tick, "phase": env.decision_type.value,
            "processing": [(op.spec.id, op.machine_id) for op in env.operations if op.state == OperationState.PROCESSING],
            "active_stages": sorted(active), "fatigue": [w.fatigue for w in env.workers],
            "loads": [w.load for w in env.workers], "committed": env._committed_worker_loads.tolist(),
            "objectives": env._objective_vector()}
    exact = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    structural = hashlib.sha256(json.dumps({k: data[k] for k in ("tick", "phase", "processing", "active_stages")}, sort_keys=True).encode()).hexdigest()
    return exact, structural


def trace_case(config, instance, mode, label, policy, seen, collisions_found):
    env = AssemblySchedulingEnv(effective(config, mode))
    env.reset(instance, preference=(1/3, 1/3, 1/3))
    maxima = Counter()
    flags = Counter()
    hashes = set()
    collisions = 0
    cross_history_matches = 0
    cross_history_numerical_aliases = 0
    cross_history_structural_aliases = 0
    start = time.perf_counter()
    while not env.task_done:
        obs = env.observe()
        obs.validate()
        prospective = virtual_observation(env)
        physical = reconstructed_quantities(env, prospective)
        assert reconstructed_events(env, prospective, physical) == actual_events(env)
        flags["event_queue_reconstructed"] += 1
        assert physical["tick"] == env.current_tick
        assert physical["active"] == sum(env._order_released[o.id] and o.id not in env._order_completion_tick for o in instance.orders)
        flags[f"phase_{env.decision_type.value}"] += 1
        flags["processing_present"] += int(any(op.state == OperationState.PROCESSING for op in env.operations))
        stages = Counter(r.stage.value for r in env.reconfigurations.values())
        for k in ("WAIT_DIS", "DIS", "WAIT_INS", "INS"):
            flags[f"stage_{k}"] += int(stages[k] > 0)
        flags["coincident_future_events"] += int(len({e[0] for e in env._events}) < len(env._events))
        flags["unreleased_orders"] += int(any(not env._order_released[o.id] for o in instance.orders))
        for key, difference in (("load", np.max(np.abs(physical["loads"]-[w.load for w in env.workers]))),
                                ("committed_load", np.max(np.abs(physical["committed"]-env._committed_worker_loads))),
                                ("progress", abs(physical["progress"]-env.operation_progress())),
                                ("objectives", np.max(np.abs(physical["objectives"]-env._objective_vector())))):
            maxima[key] = max(maxima[key], float(difference))
        for mi, duration in physical["stage_lengths"].items():
            rec = env._active_reconfiguration(env.machines[mi].spec.id)
            prefix = "disassembly" if rec.stage == ReconfigurationStage.DIS else "installation"
            assert duration == getattr(rec, prefix+"_end_tick") - getattr(rec, prefix+"_start_tick")
        digest = observation_sha256(obs)
        if digest in hashes:
            collisions += 1
        hashes.add(digest)
        key = (mode, instance.instance_id, digest)
        exact, structural = physical_signature(env)
        previous = seen.get(key)
        if previous:
            cross_history_matches += 1
            if previous["exact"] != exact:
                numeric = previous["structural"] == structural
                cross_history_numerical_aliases += int(numeric)
                cross_history_structural_aliases += int(not numeric)
                collisions_found.append({"mode": mode, "instance_id": instance.instance_id, "observation_hash": digest,
                                         "previous_policy": previous["policy"], "previous_step": previous["step"],
                                         "current_policy": label, "current_step": env._decision_count,
                                         "type": "numerical" if numeric else "structural"})
        else:
            seen[key] = {"exact": exact, "structural": structural, "policy": label, "step": env._decision_count}
        action = policy.select_action(env)
        is_wait = action == env.wait_action
        _, reward, _, _, _ = env.step(action, build_observation=False)
        flags["wait_actions"] += int(is_wait)
        # Reward targets are exact environment values; numeric reconstruction is separately bounded above.
    return {"instance_id": instance.instance_id, "mode": mode, "policy": label,
            "steps": env._decision_count, "reason": env.terminal_reason,
            "succeeded": env.task_succeeded, "failed": env.task_failed, "sampling_truncated": env.sampling_truncated,
            "reconstruction_max_errors": dict(maxima), "coverage": dict(flags),
            "cross_history_matching_observations": cross_history_matches,
            "cross_history_numerical_aliases": cross_history_numerical_aliases,
            "cross_history_structural_aliases": cross_history_structural_aliases,
            "same_trajectory_observation_repeats": collisions, "wall_seconds": time.perf_counter()-start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="result/analysis/state_sufficiency_schema9")
    parser.add_argument("--skip-traces", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = load_config("configs/default.json")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"config.json", public_config(config))
    template = load_instance_yaml(config["paths"]["fixed_instance"])
    scan = scan_datasets(config, output)
    fixtures = {"processing_assignment": (build_alias_instance(template), reachable_histories),
                "worker_service_assignment": (worker_alias_instance(template), worker_histories)}
    witnesses = []
    for name, (instance, builder) in fixtures.items():
        write_json(output/f"{name}_instance.json", instance_to_dict(instance))
        for mode in MODES:
            states, paths = builder(effective(config, mode), instance)
            result = pair_comparison(states)
            result.update(kind=name, mode=mode, legal_histories=paths)
            witnesses.append(result)
            print(f'Witness {name}/{mode}: original_equal={result["original_equal"]}', flush=True)
    write_json(output/"witnesses.json", witnesses)
    write_json(output/"encoding_checks.json", encoding_checks(config, fixtures))
    write_json(output/"boundary_checks.json", boundary_checks(config, template))
    write_json(output/"event_order_checks.json", event_order_checks(config, template))
    write_json(output/"precision_checks.json", precision_checks(config, template))
    write_json(output/"source_provenance.json", {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in ("environment/env.py", "environment/types.py", "environment/state.py",
                     "environment/time_context.py", "agent/ppo/network.py", "configs/default.json")})
    if not args.skip_traces:
        cases = [("fixed", template)]
        for cell in scan["selected_cells"]:
            cases.append((cell["path"], load_generated_record(cell["path"]).instance))
        rows = []
        seen = {}
        collision_candidates = []
        for index, (path, instance) in enumerate(cases):
            for mode in MODES:
                for label, policy in (("heuristic", HeuristicPolicy()), ("random11", RandomPolicy(11)), ("random23", RandomPolicy(23))):
                    row = trace_case(config, instance, mode, label, policy, seen, collision_candidates)
                    row["source"] = path
                    rows.append(row)
            write_json(output/"trajectory_checks.json", rows)
            print(f'Trajectory cell {index+1}/{len(cases)}: {len(rows)} runs, {sum(r["steps"] for r in rows)} states', flush=True)
        write_json(output/"trace_summary.json", {
            "run_count": len(rows), "state_count": sum(r["steps"] for r in rows), "unique_hash_count": len(seen),
            "mode_count": dict(Counter(r["mode"] for r in rows)),
            "terminal_reasons": dict(Counter(r["reason"] for r in rows)),
            "reconstruction_max_errors": {k: max(r["reconstruction_max_errors"][k] for r in rows)
                                          for k in ("load", "committed_load", "progress", "objectives")},
            "coverage": dict(sum((Counter(r["coverage"]) for r in rows), Counter())),
            "within_trajectory_exact_repeats": sum(r["same_trajectory_observation_repeats"] for r in rows),
            "cross_history_matching_observations": sum(r["cross_history_matching_observations"] for r in rows),
            "cross_history_numerical_aliases": sum(r["cross_history_numerical_aliases"] for r in rows),
            "cross_history_structural_aliases": sum(r["cross_history_structural_aliases"] for r in rows),
        })
        write_json(output/"collision_candidates.json", collision_candidates)


if __name__ == "__main__":
    main()
