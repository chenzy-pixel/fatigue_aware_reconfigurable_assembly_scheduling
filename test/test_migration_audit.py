from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from configs import load_config
from configs.normalization import apply_normalization_manifest
from data import load_dataset_split
from data.dataset import OnlineInstanceDataset
from data.distribution import CELLS, training_sampling_plan
from data.generate_orders import generate_instance
from data.models import validate_instance
from test.support import check_permutations, check_graph
from environment import AssemblySchedulingEnv, DecisionType


@pytest.fixture(scope='module')
def config():
    torch.set_num_threads(1)
    c = load_config('configs/default.json'); c['device'] = 'cpu'
    return c


@pytest.mark.parametrize('index', [0, 13, 26])
def test_all_node_permutations_preserve_policy(config, index):
    instance = load_dataset_split(config, 'validation')[index].instance
    results = check_permutations(config, instance)
    assert all(r['logits_max_error'] < 1e-5 and r['value_error'] < 1e-5 for r in results.values()), results


@pytest.mark.parametrize('kind', ['machines', 'workers', 'modules'])
def test_reject_duplicate_graph_identifiers(config, kind):
    instance = load_dataset_split(config, 'validation')[0].instance
    values = list(getattr(instance, kind))
    values[0] = values[1] if kind == 'modules' else replace(values[0], id=values[1].id)
    with pytest.raises(ValueError, match='unique'):
        validate_instance(replace(instance, **{kind: tuple(values)}))


def test_declared_factor_levels_cannot_silently_drift(config):
    c = deepcopy(config); c['dataset']['factor_levels']['N'] = [1,2,3]
    with pytest.raises(ValueError, match='factor_levels'):
        training_sampling_plan(c, 27)


def test_training_cache_detects_same_cell_corruption(config, tmp_path):
    c = deepcopy(config); c['paths']['training_instances_cache'] = str(tmp_path)
    data = OnlineInstanceDataset(config=c, episode_count=27)
    original, hit, digest, path = data.get_with_cache_info(0)
    assert not hit
    assert data.get_with_cache_info(0)[1:3] == (True, digest)
    raw = json.loads(path.read_text(encoding='utf-8'))
    raw['instance']['orders'][0]['operations'][0]['base_processing_time'] += 1
    path.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='deterministic instance'):
        data.get_with_cache_info(0)
    with pytest.raises(IndexError): data.get_with_cache_info(-1)


def test_normalization_bound_to_actual_calibration(config, tmp_path):
    c = deepcopy(config)
    calibration = tmp_path/'calibration'; calibration.mkdir()
    (calibration/'manifest.json').write_text('{}', encoding='utf-8')
    c['paths']['manifests_root'] = str(tmp_path)
    from configs.config import PROJECT_ROOT
    with pytest.raises(ValueError, match='calibration manifest'):
        apply_normalization_manifest(c, project_root=PROJECT_ROOT)
    c = deepcopy(config); c['generator']['processing_time'] = [9,15]
    with pytest.raises(ValueError, match='generator protocol'):
        apply_normalization_manifest(c, project_root=PROJECT_ROOT)


def test_generator_multiple_seeds_all_cells(config):
    for cell in CELLS:
        for seed in range(10):
            instance = generate_instance(config, cell, seed)
            counts = [sum(op.required_module == a for op in instance.operations) for a in instance.modules]
            assert max(counts)-min(counts) <= 1 and min(counts) >= 2
            assert all(len({op.required_module for op in order.operations}) >= 2 for order in instance.orders)
            assert all(m.initial_module in m.module_parameters for m in instance.machines)
            assert all(len(w.qualified_modules) == 4 for w in instance.workers)


def test_random_intermediate_observations(config):
    instance = generate_instance(config, CELLS[0], 91)
    env = AssemblySchedulingEnv(config); env.reset(instance, build_observation=False)
    rng = np.random.default_rng(53)
    phases = set()
    for step in range(70):
        if env.task_done: break
        phases.add(env.decision_type.value)
        if step % 7 == 0: check_graph(env)
        env.step(int(rng.choice(np.flatnonzero(~env.get_action_mask()))), build_observation=False)
    assert len(phases) == 2


def test_worker_permutation_during_active_reconfiguration(config):
    from agent.ppo import build_actor_critic
    instance = generate_instance(config, CELLS[0], 91)
    other = replace(instance, workers=tuple(reversed(instance.workers)))
    a, b = AssemblySchedulingEnv(config), AssemblySchedulingEnv(config)
    oa = a.reset(instance); b.reset(other)
    network = build_actor_critic(oa, config['network']).eval()
    rng = np.random.default_rng(1); phases = set()
    for step in range(65):
        if a.task_done: break
        assert a.decision_type == b.decision_type
        phases.add(a.decision_type)
        if step % 5 == 0:
            oa, ob = a.observe(), b.observe()
            la, va = network(oa, a.get_action_mask(), device='cpu')
            lb, vb = network(ob, b.get_action_mask(), device='cpu')
            if a.decision_type == DecisionType.WORKER:
                lb = torch.cat((lb[:-1].reshape(len(a.machines),len(a.workers)).flip(1).flatten(), lb[-1:]))
            assert torch.allclose(la, lb, atol=1e-5)
            assert torch.allclose(va, vb, atol=1e-5)
        mask = a.get_action_mask()
        action = int(rng.choice(np.flatnonzero(~mask)))
        mapped = action
        if a.decision_type == DecisionType.WORKER and action != len(mask)-1:
            mi, wi = a.decode_worker_action(action)
            mapped = b.encode_worker_action(mi, other.worker_index[a.workers[wi].spec.id])
        a.step(action, build_observation=False); b.step(mapped, build_observation=False)
        assert a.current_tick == b.current_tick
        assert a.metrics()['worker_loads'] == b.metrics()['worker_loads']
    assert DecisionType.WORKER in phases


@pytest.mark.parametrize('method', ['ppo','dcan','rule_ddqn'])
def test_previous_projection_semantics_checkpoint_is_rejected(config, tmp_path, method):
    from agent.comparators.learning import build_agent
    from agent.comparators.common import method_config
    c = method_config(config, method)
    obs = AssemblySchedulingEnv(c).reset(load_dataset_split(c, 'validation')[0].instance)
    agent = build_agent(c, obs, method)
    path = tmp_path/'model.pt'; agent.save(path)
    payload = torch.load(path, weights_only=False, map_location='cpu')
    spec = payload['network_spec']
    target = spec if method == 'ppo' else spec['observation']
    target['time_context_version'] = 'order_chain_remaining_context_v3'
    torch.save(payload, path)
    with pytest.raises(ValueError): agent.load(path)
