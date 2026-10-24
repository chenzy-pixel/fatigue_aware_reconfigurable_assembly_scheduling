"""Graph projection and permutation oracles for current regression tests."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from configs import load_config
from data import load_dataset_split
from data.distribution import CELLS
from data.generate_orders import generate_instance
from data.models import instance_time_scale
from environment import AssemblySchedulingEnv, DecisionType
from environment.types import CAPABLE_EDGE, SERVICE_CANDIDATE_EDGE
from environment.resource_projection import ResourceProjector
from environment.time_context import project_wait_state
from environment.workload import workload_value


def check_graph(env):
    obs = env.observe(); obs.validate()
    assert all(np.isfinite(edge.edge_features).all() for edge in obs.relations.values())
    mask = env.get_action_mask()
    kind = CAPABLE_EDGE if env.decision_type == DecisionType.PRODUCTION else SERVICE_CANDIDATE_EDGE
    if env.decision_type != DecisionType.TERMINAL:
        width = len(env.machines) if env.decision_type == DecisionType.PRODUCTION else len(env.workers)
        pairs = {int(a)*width+int(b) for a,b in obs.relations[kind].edge_index.T}
        assert set(np.flatnonzero(~mask[:-1])) <= pairs
    # Optimized grouped edge projection must agree with the individual oracle.
    edge = obs.relations[CAPABLE_EDGE]
    projector = ResourceProjector(env)
    for pos in np.linspace(0, edge.num_edges-1, min(8, edge.num_edges), dtype=int):
        oi, mi = map(int, edge.edge_index[:, pos])
        candidate = projector.candidate(oi, mi)
        expected = 0 if candidate.finish_tick is None else max(0, candidate.finish_tick-env.current_tick)/env.time_scale_tick
        assert np.isclose(edge.edge_features[pos, edge.feature_names.index('predicted_finish_time_norm')], expected, atol=1e-6)
        assert edge.edge_features[pos, edge.feature_names.index('projection_valid')] == float(candidate.finish_tick is not None)
    # Repeated observation is pure, and WAIT features predict a real legal WAIT.
    before = (env.current_tick, deepcopy(env.reconfiguration_log), env._committed_worker_loads.copy())
    if not env.task_done and not mask[-1]:
        certificate = env._wait_certificate()
        projected = project_wait_state(env, certificate['wait_ticks'], settle_terminal=True, certificate=certificate)
        actual = deepcopy(env)
        actual.step(len(mask)-1, build_observation=False)
        assert actual.current_tick == projected.current_tick
        np.testing.assert_allclose(actual._objective_vector(), projected._objective_vector(), atol=1e-8)
    assert env.current_tick == before[0] and env.reconfiguration_log == before[1]
    np.testing.assert_array_equal(env._committed_worker_loads, before[2])
    return {k: len(v) for k,v in obs.node_feature_names.items()}


def check_permutations(config, instance):
    from agent.ppo import build_actor_critic
    a = AssemblySchedulingEnv(config); oa = a.reset(instance)
    network = build_actor_critic(oa, config['network']).eval()
    la, va = network(oa, a.get_action_mask(), device='cpu')
    results = {}
    for name in ('modules', 'machines', 'workers', 'orders'):
        other = replace(instance, **{name: tuple(reversed(getattr(instance, name)))})
        b = AssemblySchedulingEnv(config); ob = b.reset(other)
        lb, vb = network(ob, b.get_action_mask(), device='cpu')
        left = la[:-1].reshape(len(a.operations), len(a.machines))
        right = lb[:-1].reshape(len(b.operations), len(b.machines))
        oi = [other.operation_index[o.spec.id] for o in a.operations]
        mi = [other.machine_index[m.spec.id] for m in a.machines]
        aligned = right[oi][:, mi]
        results[name] = {'logits_max_error': float((left-aligned).abs().max().detach()),
                         'value_error': float((va-vb).abs().max().detach())}
    return results
