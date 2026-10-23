"""Independent randomized physics and graph checks for the schema-13 migration."""
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


def run(output):
    torch.set_num_threads(1); torch.manual_seed(7)
    config = load_config('configs/default.json'); config['device'] = 'cpu'
    rng = np.random.default_rng(927)
    report = {'trajectories': [], 'permutations': {}}
    widths = None
    for cell in CELLS:
        instance = generate_instance(config, cell, 20261023+int(cell.cell_id[1:]))
        env = AssemblySchedulingEnv(config); env.reset(instance, build_observation=False)
        snapshots, reward_sum, steps = 0, 0., 0
        while not env.task_done and steps < 10000:
            if steps % 70 == 0 or (env.decision_type == DecisionType.WORKER and snapshots < 2):
                current = check_graph(env)
                assert widths is None or current == widths
                widths = current; snapshots += 1
            action = int(rng.choice(np.flatnonzero(~env.get_action_mask())))
            _, reward, _, _, _ = env.step(action, build_observation=False)
            reward_sum += reward.scalarize(config); steps += 1
        assert env.task_succeeded, (cell.cell_id, env.terminal_reason, steps)
        assert not env.validate_schedule()
        metrics = env.metrics()
        assert np.isclose(metrics['total_flow_time'], sum(metrics['completion_times'].values()))
        assert np.isclose(metrics['flow_excess_objective'], metrics['total_flow_time']-metrics['flow_processing_lower_bound'])
        assert np.isclose(reward_sum, 1-metrics['preference_quality_score'])
        assert metrics['maximum_worker_fatigue'] <= instance.fatigue.maximum_safe_fatigue+1e-9
        assert instance_time_scale(instance) <= metrics['time']+1e-9
        loads = {w.id: 0. for w in instance.workers}
        labor = fixed = 0.
        workers = {w.id:w for w in instance.workers}
        for row in env.reconfiguration_log:
            duration = row['end']-row['start']
            loads[row['worker_id']] += duration
            labor += duration*workers[row['worker_id']].labor_cost_per_minute
            fixed += row['fixed_cost']
        downtime = sum((rec.installation_end_tick-rec.lock_tick)*instance.resolution * instance.machines[instance.machine_index[rec.machine_id]].downtime_cost_per_minute for rec in env.reconfigurations.values())
        np.testing.assert_allclose(list(loads.values()), [metrics['worker_loads'][k] for k in loads], atol=1e-8)
        assert np.isclose(fixed+labor+downtime, metrics['reconfiguration_cost'])
        assert np.isclose(workload_value(list(loads.values()), config), metrics['workload_objective_value'])
        report['trajectories'].append({'cell':cell.cell_id, 'steps':steps, 'snapshots':snapshots, 'cmax':metrics['time']})
        print(cell.cell_id, steps, snapshots, 'passed', flush=True)
    for index in (0, 13, 26):
        r = load_dataset_split(config, 'validation')[index]
        report['permutations'][r.metadata['cell_id']] = check_permutations(config, r.instance)
    report['node_widths'] = widths
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['permutations'], indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
