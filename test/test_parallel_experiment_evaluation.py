"""Serial parity, unit-owned RNG, lane refill and shared-pool validation."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from configs.experiment_suite import arm_config, load_suite
from data import load_dataset_split
from environment import AssemblySchedulingEnv
from agent.comparators.learning import build_agent, evaluate_request, validation
from agent.comparators.parallel_evaluation import evaluate_requests, evaluation_pool, reuse_first_sample
from agent.comparators.experiment_evaluation import learning_unit


def assert_same(expected, actual):
    assert actual['selected_repeat'] == expected['selected_repeat']
    for key in ('success', 'quality', 'objectives', 'candidate_count', 'environment_steps', 'delivery_protocol'):
        if key in ('quality', 'objectives'):
            assert actual['summary'][key] == pytest.approx(expected['summary'][key], abs=1e-12)
        else:
            assert actual['summary'][key] == expected['summary'][key]
    assert len(actual['raw']) == len(expected['raw'])
    for a, b in zip(expected['raw'], actual['raw']):
        for key in ('success', 'quality', 'objectives', 'actions', 'trace', 'violations', 'schedule', 'reconfigurations'):
            if key in ('quality', 'objectives'):
                assert a[key] == pytest.approx(b[key], abs=1e-12), key
            else:
                assert a[key] == b[key], key


@pytest.mark.parametrize('arm', list(load_suite()['arms']))
def test_serial_parallel_refill_and_validation(arm, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    c = arm_config(load_suite(), arm, 11)
    c['network'].update(hidden_dim=16, message_passing_layers=1)
    c['training']['validation_parallel_envs'] = 2
    source = load_dataset_split(c, 'calibration')[0]
    target = source.instance.operations[0].required_module
    machines = tuple(replace(m, initial_module=next((a for a in m.module_parameters if a != target), m.initial_module))
                     for m in source.instance.machines)
    records = [SimpleNamespace(instance=replace(source.instance, instance_id=f'parallel_{i}',
                                               machines=machines, orders=source.instance.orders[:i+1]), metadata=source.metadata)
               for i in range(2)]
    method = load_suite()['arms'][arm]['method']
    agent = build_agent(c, AssemblySchedulingEnv(c).reset(records[0].instance), method)
    network = agent.online if method == 'rule_ddqn' else agent.network
    weights = deepcopy(network.state_dict())
    rng = agent.rng.getstate() if method == 'rule_ddqn' else None
    prefs = [(1., 0., 0.), (1/3, 1/3, 1/3)]
    requests = [(r, p) for r in records for p in prefs]
    serial = [evaluate_request(c, agent, r.instance, p, method, seed=71) for r, p in requests]
    with evaluation_pool(c, records[0].instance) as pool:
        progress = []
        parallel, runtime = evaluate_requests(c, agent, requests, method, runner=pool, seed=71, progress_callback=progress.append)
        assert runtime['workers'] == 2
        assert runtime['trajectory_count'] == (4 if method == 'rule_ddqn' else 12)
        assert progress[0]['completed'] == progress[0]['environment_steps'] == 0
        assert progress[-1]['completed'] == progress[-1]['total'] == runtime['trajectory_count']
        assert progress[-1]['active'] == 0
        assert progress[-1]['environment_steps'] == sum(r['summary']['environment_steps'] for r in serial)
        assert [p['completed'] for p in progress] == sorted(p['completed'] for p in progress)
        assert sum(r['summary']['search_seconds'] for r in parallel) == pytest.approx(runtime['wall_seconds'])
        for expected, actual in zip(serial, parallel):
            assert_same(expected, actual)
        # The same workers must remain usable when alternating training validation
        # with captured frontiers. Suppress the standalone pool constructor.
        monkeypatch.setattr('agent.comparators.parallel_evaluation.evaluation_pool', lambda *a, **k: pytest.fail('pool recreated'))
        result, _ = validation(c, agent, records, [0, 1], prefs, method, seed=71, runner=pool)
        assert result['summary']['validation_env_steps'] == sum(r['summary']['environment_steps'] for r in serial)
        assert all(not s['violations'] for r in result['deliveries'] for s in r['raw'])
        if arm == 'full':
            frontier = learning_unit(c, agent, records[0], method, prefs, 71, parallel=True, runner=pool)
            reused = reuse_first_sample(c, frontier)
            single = [evaluate_request(c, agent, records[0].instance, p, method, seed=71, mode='sampled_single') for p in prefs]
            for expected, actual in zip(single, reused['requests']):
                assert_same(expected, actual)
            assert reused['incremental_solve_seconds'] == reused['environment_steps'] == 0
            assert reused['seconds'] == frontier['seconds']
        # Refill and batch size must not enter the request RNG identity.
        c['training']['validation_parallel_envs'] = 1
        one_lane, _ = evaluate_requests(c, agent, requests[:2], method, runner=pool, seed=71)
        for expected, actual in zip(serial[:2], one_lane):
            assert_same(expected, actual)
    assert all(torch.equal(v, network.state_dict()[k]) for k, v in weights.items())
    if method == 'rule_ddqn':
        assert agent.rng.getstate() == rng and agent.steps == agent.updates == 0


def test_default_twenty_workers():
    suite = load_suite()
    assert suite['execution']['evaluation_workers'] == 20
    for arm in suite['arms']:
        assert arm_config(suite, arm, 11)['training']['validation_parallel_envs'] == 20


def test_reused_pool_rejects_capacity_and_physics_mismatch_before_dispatch():
    c = arm_config(load_suite(), 'full', 11)
    record = SimpleNamespace(instance=SimpleNamespace(instance_id='guard'))
    requests = [(record, (1., 0., 0.))]
    pool = SimpleNamespace(worker_count=2, config=deepcopy(c))
    agent = SimpleNamespace(assert_evaluation_config=lambda c: None)
    with pytest.raises(ValueError, match='smaller than requested'):
        evaluate_requests(c, agent, requests, 'ppo', runner=pool)
    pool.worker_count = 20
    pool.config['environment']['max_decisions'] += 1
    with pytest.raises(ValueError, match='physics differs'):
        evaluate_requests(c, agent, requests, 'ppo', runner=pool)


def test_missing_rollout_fails_before_aggregation():
    c = arm_config(load_suite(), 'full', 11)
    pool = SimpleNamespace(worker_count=20, config=deepcopy(c), evaluate_records=lambda *a, **kw: [])
    agent = SimpleNamespace(assert_evaluation_config=lambda c: None)
    requests = [(SimpleNamespace(instance=SimpleNamespace(instance_id='missing')), (1., 0., 0.))]
    with pytest.raises(RuntimeError, match='missing, duplicated'):
        evaluate_requests(c, agent, requests, 'ppo', runner=pool)
