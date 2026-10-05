"""Behavioral regressions for current observations, model contracts, and density."""
from copy import deepcopy
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest
import torch

from agent.baselines import HeuristicPolicy, RandomPolicy
from agent.ppo import PPOAgent, RolloutBuffer
from agent.ppo.network import HeteroGraphActorCritic, build_actor_critic, infer_checkpoint_network_spec
from configs import load_config
from configs.config import public_config
from configs.runtime import validate_latest_only_config
from data.models import OrderSpec, OperationSpec, validate_instance
from data.dataset import load_dataset_split
from environment import AssemblySchedulingEnv
from environment.observation_schema import GLOBAL_FEATURE_NAMES
from environment.types import DecisionType, ReconfigurationStage
from eval import EvaluationPolicy, evaluate_instance
from result.provenance import build_provenance, network_weights_sha256


def _agent(config, observation):
    return PPOAgent(build_actor_critic(observation, {**config['network'], 'hidden_dim': 16}),
                    config['ppo'], device='cpu')


def test_reachable_histories_expose_hidden_cumulative_flow(config, fixed_instance):
    module = fixed_instance.machines[0].initial_module
    machine = replace(fixed_instance.machines[0], module_parameters={name: replace(spec, processing_speed_factor=1.0)
        for name, spec in fixed_instance.machines[0].module_parameters.items()})
    orders = tuple(OrderSpec(name, 'audit', release, (OperationSpec(name+'_op', name, 1, module, duration),))
        for name, release, duration in [('A', 0.0, 10.0), ('B', 0.0, 30.0), ('C', 50.0, 10.0)])
    instance = replace(fixed_instance, instance_type='test', horizon=100.0,
        machines=(machine, *fixed_instance.machines[1:]), orders=orders,
        waves={'audit': {'dominant_module': module, 'order_ids': ['A','B','C'], 'release_interval': [0.0,50.0]}})
    validate_instance(instance)
    observations = []
    for sequence in [(0,1), (1,0)]:
        env = AssemblySchedulingEnv(config)
        env.reset(instance, preference=(1,0,0))
        for index in sequence:
            env.step(env.encode_production_action(index,0))
            env.step(env.wait_action)
        env.step(env.wait_action)
        observations.append(env.observe())
    first, second = observations
    for name in first.node_features:
        np.testing.assert_array_equal(first.node_features[name], second.node_features[name])
    for name in first.relations:
        np.testing.assert_array_equal(first.relations[name].edge_features, second.relations[name].edge_features)
    np.testing.assert_array_equal(first.global_features[:6], second.global_features[:6])
    np.testing.assert_array_equal(first.action_set_features, second.action_set_features)
    scale = config['objective_scalarizer']['scales']['flow']
    assert first.global_features[6] == pytest.approx(50/scale)
    assert second.global_features[6] == pytest.approx(70/scale)
    assert first.global_feature_names == GLOBAL_FEATURE_NAMES


def test_global_objectives_and_transport_match_reward_state(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance)
    rule = HeuristicPolicy()
    for _ in range(90):
        env.step(rule.select_action(env), build_observation=False)
        obs = env.observe()
        scales = config['objective_scalarizer']['scales']
        np.testing.assert_allclose(obs.global_features[6:9], np.asarray(env._objective_vector()) /
            [scales[name] for name in ('flow','cost','variance')], rtol=1e-6)
        buffer = RolloutBuffer(preserve_graph=True)
        buffer.add(obs, env.get_action_mask(), 0, 0, 0, 0, env.terminated)
        for copied in (obs.copy(), pickle.loads(pickle.dumps(obs)), buffer.transitions[0].observation):
            np.testing.assert_array_equal(copied.global_features, obs.global_features)
        if env.task_done:
            break
    agent = _agent(config, obs)
    obs2 = obs.copy()
    obs2.global_features[6:9] += 0.01
    batch = agent.network._collate_graphs([obs,obs2], device='cpu')
    np.testing.assert_array_equal(batch.global_features.numpy(), np.stack([obs.global_features,obs2.global_features]))


@pytest.mark.parametrize('value', [0, -1, 2.5, True])
def test_decision_limits_require_positive_integers(config, value):
    for name in ('max_decisions','max_zero_time_actions'):
        settings = public_config(config)
        settings.pop('runtime_manifest')
        settings['environment'][name] = value
        with pytest.raises(ValueError, match='positive integer'):
            validate_latest_only_config(settings)
        with pytest.raises(ValueError, match='positive integer'):
            AssemblySchedulingEnv(settings)


@pytest.mark.parametrize('dropout', [0.1, -0.1, float('nan')])
def test_dropout_rejected_at_each_boundary(config, fixed_instance, dropout, tmp_path):
    settings = public_config(config)
    settings['network']['dropout'] = dropout
    settings.pop('runtime_manifest')
    with pytest.raises(ValueError, match='dropout = 0'):
        validate_latest_only_config(settings)
    obs = AssemblySchedulingEnv(config).reset(fixed_instance)
    with pytest.raises(ValueError, match='dropout = 0'):
        build_actor_critic(obs, settings['network'])
    with pytest.raises(ValueError, match='dropout = 0'):
        HeteroGraphActorCritic(obs.feature_dimensions, obs.edge_feature_dimensions, obs.action_set_feature_names,
            hidden_dim=16, message_passing_layers=2, dropout=dropout)
    network = build_actor_critic(obs, {**config['network'], 'hidden_dim': 16})
    network.critic[2].p = dropout
    with pytest.raises(ValueError, match='dropout = 0'):
        PPOAgent(network, config['ppo'])
    agent = _agent(config, obs)
    path = tmp_path/'dropout.pt'
    agent.save(path)
    payload = torch.load(path, weights_only=False)
    payload['network_spec']['dropout'] = dropout
    torch.save(payload, path)
    with pytest.raises(ValueError, match='dropout = 0'):
        agent.load(path, allow_observation_migration=True)


def test_unchanged_policy_probabilities_and_values_match(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    obs = env.reset(fixed_instance)
    mask = env.get_action_mask()
    agent = _agent(config, obs)
    action, old_log, old_value = agent.act(obs, mask)
    with torch.no_grad():
        logits, value = agent.network.forward_batch([obs], [mask], device='cpu')
        new_log = torch.distributions.Categorical(logits=logits).log_prob(torch.tensor([action])).item()
    assert np.exp(new_log-old_log) == pytest.approx(1.0, abs=1e-6)
    assert value.item() == pytest.approx(old_value, abs=1e-6)


def test_legacy_schema_rejected_with_optimizer_and_migration_flags(config, fixed_instance, tmp_path):
    obs = AssemblySchedulingEnv(config).reset(fixed_instance)
    agent = _agent(config, obs)
    path = tmp_path/'checkpoint.pt'
    agent.save(path)
    payload = torch.load(path, weights_only=False)
    payload['network_spec']['observation_schema_version'] = 6
    torch.save(payload,path)
    for allow in (False,True):
        with pytest.raises(ValueError,match='requires retraining'):
            agent.load(path,load_optimizer=True,allow_observation_migration=allow)


def test_global_schema_rejects_reordering_and_nonfinite_values(config,fixed_instance):
    obs = AssemblySchedulingEnv(config).reset(fixed_instance)
    names = list(obs.global_feature_names)
    names[-1],names[-2] = names[-2],names[-1]
    with pytest.raises(ValueError,match='global feature'):
        build_actor_critic(replace(obs,global_feature_names=tuple(names)),config['network'])
    corrupted = obs.copy()
    corrupted.global_features[-1] = np.inf
    with pytest.raises(ValueError,match='finite'):
        build_actor_critic(corrupted,config['network'])
    spec = _agent(config,obs).network.network_spec()
    spec['global_feature_names'] = tuple(names)
    with pytest.raises(ValueError,match='global feature'):
        infer_checkpoint_network_spec({'network_spec':spec})


def test_config_snapshot_requires_schema10_and_retraining(config,tmp_path):
    snapshot = public_config(config)
    path = tmp_path/'config.json'
    for schema in (5,6,7,8,9):
        snapshot['runtime_manifest']['observation_schema'] = schema
        path.write_text(json.dumps(snapshot),encoding='utf-8')
        for allow in (False,True):
            with pytest.raises(ValueError,match='schema 10'):
                load_config(path,allow_observation_migration=allow)


@pytest.mark.parametrize('boundary', ['zero','partial','complete'])
def test_completed_reconfiguration_density_uses_true_completion(config,fixed_instance,boundary):
    env = AssemblySchedulingEnv(deepcopy(config))
    env.reset(fixed_instance,build_observation=False)
    rule = HeuristicPolicy()
    rec = None
    for _ in range(300):
        action = rule.select_action(env)
        is_install = False
        if env.decision_type == DecisionType.WORKER and action != env.wait_action:
            mi,_ = env.decode_worker_action(action)
            rec = env._pending_reconfiguration(env.machines[mi].spec.id)
            is_install = rec.stage == ReconfigurationStage.WAIT_INS
        if is_install and boundary == 'zero':
            pass
        env.step(action,build_observation=False)
        if is_install:
            break
    assert rec is not None and rec.stage == ReconfigurationStage.INS
    if boundary == 'zero':
        env._apply_truncation('horizon')
    if boundary != 'zero':
        target = rec.installation_end_tick if boundary == 'complete' else env.current_tick+1
        if boundary == 'partial':
            assert target < rec.installation_end_tick
        env.horizon_tick = target
        env._invalidate_resource_snapshot()
        env._resolve_terminal_or_deadlock()
        while not env.task_done:
            env.step(rule.select_action(env),build_observation=False)
    metrics = env.metrics()
    count = int(boundary == 'complete')
    assert metrics['completed_reconfigurations'] == count
    assert metrics['completed_reconfigurations_per_minute'] == pytest.approx(count/env.current_time)
    completed = sum(op.state.value == 'DONE' for op in env.operations)
    assert metrics['completed_reconfigurations_per_operation'] == pytest.approx(count/completed)
    if boundary != 'complete':
        assert env.reconfiguration_log[-1]['truncated']


def test_zero_density_denominators(config,fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance)
    metrics = env.metrics()
    assert metrics['completed_reconfigurations_per_minute'] is None
    assert metrics['completed_reconfigurations_per_operation'] is None


def test_single_instance_evaluation_restores_mode_on_exception(config,fixed_instance,monkeypatch):
    obs = AssemblySchedulingEnv(config).reset(fixed_instance)
    agent = _agent(config,obs)
    policy = EvaluationPolicy(config,policy_name='ppo',bootstrap_observation=obs,ppo_agent=agent,decode_mode='greedy')
    def fail(*args,**kwargs):
        assert not agent.network.training
        raise RuntimeError('intentional failure')
    monkeypatch.setattr(policy,'select_action',fail)
    with pytest.raises(RuntimeError,match='intentional failure'):
        evaluate_instance(config,instance=fixed_instance,policy_name='ppo',prepared_policy=policy)
    assert agent.network.training


def test_executed_logs_reconstruct_three_objectives_on_success_and_failure(config):
    outcomes = []
    for record in list(load_dataset_split(config,'test'))[:3]:
        for fail_at_horizon, rule in ((False, HeuristicPolicy()), (True, RandomPolicy(311))):
            env = AssemblySchedulingEnv(config)
            env.reset(record.instance,build_observation=False)
            if fail_at_horizon:
                env.horizon_tick = env.current_tick + 1
            while not env.task_done:
                env.step(rule.select_action(env),build_observation=False)
            metrics = env.metrics()
            outcomes.append(env.task_succeeded)
            ends = {row['operation_id']:row['end'] for row in env.schedule_log if not row.get('truncated')}
            flow = sum(max(0,ends.get(order.operations[-1].id,env.current_time)-order.release_time)
                       for order in record.instance.orders)
            flow += sum(order.operations[-1].id not in ends for order in record.instance.orders) * record.instance.unfinished_order_penalty
            loads = {worker.id:0.0 for worker in record.instance.workers}
            rates = {worker.id:worker.labor_cost_per_minute for worker in record.instance.workers}
            cost = 0.0
            for row in env.reconfiguration_log:
                loads[row['worker_id']] += row['duration']
                cost += row['fixed_cost']+row['duration']*rates[row['worker_id']]
            for rec in env.reconfigurations.values():
                end = rec.installation_end_tick if rec.stage == ReconfigurationStage.DONE else env.current_tick
                cost += (end-rec.lock_tick)*env.resolution*env._machine_by_id(rec.machine_id).spec.downtime_cost_per_minute
            assert flow == pytest.approx(metrics['flow_time_objective'],abs=1e-8)
            assert cost == pytest.approx(metrics['reconfiguration_cost'],abs=1e-8)
            assert float(np.var(list(loads.values()))) == pytest.approx(metrics['worker_load_variance'],abs=1e-8)
            assert env.validate_schedule() == []
    assert any(outcomes) and not all(outcomes)
