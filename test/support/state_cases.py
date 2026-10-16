"""Reusable scheduling regression cases."""
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

import numpy as np

import torch

from agent.baselines import HeuristicPolicy, RandomPolicy

from configs import load_config

from configs.config import public_config

from data.dataset import load_generated_record

from data.models import OrderSpec, OperationSpec, load_instance_yaml, validate_instance, instance_to_dict

from environment import AssemblySchedulingEnv, DecisionType, EdgeStore

from environment.observation_schema import OBSERVATION_SCHEMA_VERSION

from environment.workload import objective_names, workload_value

from support.relation_cases import legacy_projection

from environment.types import OperationState, ReconfigurationStage

from support.observation_cases import (

    build_alias_instance, reachable_histories, compare_observations, observation_sha256,

)

PROCESSING = ("operation", "processing_on", "machine")

SERVING = ("machine", "served_by", "worker")

MODES = ("full", "neutral")

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

                for variant in ("hetero_gnn", "node_mlp_pool"):

                    for seed in (0, 1, 11, 23, 37):

                        torch.manual_seed(seed)

                        settings = {**config["network"], "encoder_variant": "node_mlp_pool" if variant == "node_mlp_pool" else "hetero_gnn",

                                    "actor_head_variant": "shared_preference"}

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

        if obs.decision_type != DecisionType.TERMINAL:

            committed[wi] += duration*env.instance.resolution

        stage_lengths[int(mi)] = duration

        assert worker_remaining[wi] == machine_remaining[mi]

    progress = float(np.mean(column("order", "completion_ratio")))

    scales = env.config["objective_scalarizer"]["scales"]

    encoded = obs.global_features[6:9].astype(np.float64)

    if env.global_objective_encoding == "log2_v1":

        normalized = np.expm1(encoded * np.log(2.0))

    elif env.global_objective_encoding == "bounded_v1":

        # Finite precision can collapse extreme bounded inputs to exactly one.

        if (encoded >= 1).any():

            raise ValueError("bounded global objective saturated; cannot reconstruct a finite value")

        normalized = encoded / (1.0 - encoded)

    else:

        normalized = encoded

    objectives = normalized * [scales[k] for k in objective_names(env.config)]

    active = int(np.sum(column("order", "released") - column("order", "completed")))

    return {"tick": current_tick, "machine_remaining": machine_remaining, "worker_remaining": worker_remaining,

            "loads": loads, "committed": committed, "progress": progress,

            "objectives": objectives, "active": active, "stage_lengths": stage_lengths,

            "reconstructed_workload_objective": workload_value(committed, env.config),

            "actual_workload_objective": workload_value(loads, env.config)}

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
