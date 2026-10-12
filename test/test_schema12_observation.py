"""Deadline context and objective encoding must preserve physics and identity."""
from copy import deepcopy
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest
import torch

from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent, RolloutBuffer, build_actor_critic
from agent.ppo.network import infer_checkpoint_network_spec
from configs import load_config, project_path
from configs.runtime import runtime_manifest
from environment import AssemblySchedulingEnv, DecisionType
from environment.observation_schema import (
    encode_global_objectives, global_feature_names, GLOBAL_OBJECTIVE_ENCODINGS,
)
from environment.time_context import OrderTimeEstimator
from test_order_time_context import _chain_instance, _commit


@pytest.mark.parametrize("encoding", GLOBAL_OBJECTIVE_ENCODINGS)
def test_encoding_math_and_large_finite_inputs(encoding):
    values = encode_global_objectives([0, 3, 3e12], [3, 3, 3], version=12, encoding=encoding)
    expected = [0, 1, np.log2(1+1e12)] if encoding == 'log2_v1' else [0, .5, 1e12/(1+1e12)]
    np.testing.assert_allclose(values, expected, rtol=1e-13)
    huge = encode_global_objectives([0, 1e308, 1e308], [1, 1e-300, 1e308], version=12, encoding=encoding)
    assert np.isfinite(huge).all()
    assert np.all(np.diff(values) > 0)
    assert encode_global_objectives([100]*3, [3]*3, version=12, encoding=encoding)[0] > values[1]


@pytest.mark.parametrize("values,scales", [([-1,0,0],[1]*3),([np.nan,0,0],[1]*3),
    ([np.inf,0,0],[1]*3),([0]*3,[0,1,1]),([0]*3,[-1,1,1]),([0]*3,[np.inf,1,1]),
    ([0]*3,[np.nan,1,1]),([0,0],[1]*3)])
def test_invalid_global_encoding_inputs_rejected(values, scales):
    with pytest.raises(ValueError):
        encode_global_objectives(values,scales,version=12)


def test_global_slack_includes_future_orders_and_reuses_cache(config, fixed_instance, monkeypatch):
    instance = _chain_instance(fixed_instance, second_release=250)
    env = AssemblySchedulingEnv(config)
    calls=[];original=OrderTimeEstimator.finish_ticks
    def counted(self, order_ids=None):
        calls.append(order_ids)
        return original(self,order_ids)
    monkeypatch.setattr(OrderTimeEstimator,'finish_ticks',counted)
    obs=env.reset(instance)
    assert calls == [None]
    assert obs.global_features[-1] < 0
    assert env.minimum_active_order_slack_norm() > 0
    assert obs.global_features[-1] == pytest.approx(min(env.estimated_order_slack_norm(o.id) for o in instance.orders))
    env.observe();env.minimum_unfinished_order_slack_norm()
    assert calls == [None]
    env._invalidate_resource_snapshot();env.observe()
    assert calls == [None,None]
    first,second=instance.orders
    env._order_completion_tick[second.id]=env.current_tick
    assert env.minimum_unfinished_order_slack_norm() == env.estimated_order_slack_norm(first.id)
    env._order_completion_tick[first.id]=env.current_tick
    assert env.minimum_unfinished_order_slack_norm() == 0


@pytest.mark.parametrize('finish_ratio,expected',[(.5,.5),(1,0),(1.2,-.2),(3,-2)])
def test_signed_slack_without_clipping(config,fixed_instance,finish_ratio,expected):
    env=AssemblySchedulingEnv(config);env.reset(fixed_instance)
    env._order_finish_tick_cache={o.id:env.horizon_tick*finish_ratio for o in fixed_instance.orders}
    assert env.minimum_unfinished_order_slack_norm()==pytest.approx(expected)


def _single_module_instance(fixed_instance, durations, *, second_machine=False):
    module=fixed_instance.modules[0]
    base=next(m for m in fixed_instance.machines if module in m.module_parameters)
    machines=(replace(base,initial_module=module,module_parameters={module:replace(base.module_parameters[module],processing_speed_factor=1.)}),)
    if second_machine:
        machines+=(replace(machines[0],id='slow_machine',module_parameters={module:replace(base.module_parameters[module],processing_speed_factor=4.)}),)
    orders=tuple(replace(order,release_time=0.,operations=(replace(order.operations[0],required_module=module,base_processing_time=duration),))
                 for order,duration in zip(fixed_instance.orders,durations))
    return replace(fixed_instance,orders=orders,machines=machines,horizon=150.)


def test_slow_machine_does_not_replace_best_order_route(config,fixed_instance):
    env=AssemblySchedulingEnv(config);obs=env.reset(_single_module_instance(fixed_instance,[100],second_machine=True))
    finishes=[env.estimate_processing_ticks(0,mi) for mi in range(2)]
    assert min(finishes)<env.horizon_tick<max(finishes)
    assert obs.global_features[-1] == pytest.approx((env.horizon_tick-min(finishes))/env.horizon_tick)


def test_independent_orders_are_not_a_joint_feasibility_certificate(config,fixed_instance):
    env=AssemblySchedulingEnv(config);obs=env.reset(_single_module_instance(fixed_instance,[100,100]))
    finishes=env.estimated_order_finish_ticks()
    assert len(set(finishes.values()))==1 and obs.global_features[-1]>0
    assert sum(env.estimate_processing_ticks(i,0) for i in range(2))>env.horizon_tick
    rule=HeuristicPolicy()
    while not env.task_done:env.step(rule.select_action(env),build_observation=False)
    assert env.task_failed


def test_processing_and_locked_orders_use_existing_estimates(config,fixed_instance):
    for mismatch in [False,True]:
        env=AssemblySchedulingEnv(config);env.reset(_chain_instance(fixed_instance,mismatch=mismatch))
        if mismatch:_commit(env)
        else:env.step(int(np.flatnonzero(~env.get_action_mask()[:-1])[0]))
        assert env.minimum_unfinished_order_slack_norm()==pytest.approx(min(
            env.estimated_order_slack_norm(o.id) for o in env.instance.orders))


@pytest.mark.parametrize('encoding',GLOBAL_OBJECTIVE_ENCODINGS)
@pytest.mark.parametrize('head',['objective_experts','shared_preference'])
@pytest.mark.parametrize('mode',['variance','geometric_owa'])
@pytest.mark.parametrize('flow',['raw','excess'])
def test_schema11_and_12_fixed_actions_preserve_physics(encoding,head,mode,flow,fixed_instance):
    paths={('raw','variance'):'configs/e1/single_variance.json',
        ('excess','variance'):'configs/flow_excess/shared_head/single_variance.json',
        ('raw','geometric_owa'):'configs/workload/raw/q05/universal.json',
        ('excess','geometric_owa'):'configs/workload/q05/universal.json'}
    c=load_config(paths[flow,mode]);c['network'].update(global_objective_encoding=encoding,actor_head_variant=head,hidden_dim=16)
    old=deepcopy(c);old['network']['observation_schema_version']=11;old['network'].pop('global_objective_encoding')
    a,b=AssemblySchedulingEnv(old),AssemblySchedulingEnv(c);oa,ob=a.reset(fixed_instance),b.reset(fixed_instance)
    rule=HeuristicPolicy();seen=set()
    for _ in range(65):
        seen.add(a.decision_type)
        np.testing.assert_array_equal(a.get_action_mask(),b.get_action_mask())
        np.testing.assert_array_equal(oa.global_features[:6],ob.global_features[:6])
        np.testing.assert_array_equal(oa.action_set_features,ob.action_set_features)
        for name in oa.node_features:np.testing.assert_array_equal(oa.node_features[name],ob.node_features[name])
        for name in oa.relations:
            np.testing.assert_array_equal(oa.relations[name].edge_index,ob.relations[name].edge_index)
            np.testing.assert_array_equal(oa.relations[name].edge_features,ob.relations[name].edge_features)
        action=rule.select_action(a);oa,ra,ta,xa,_=a.step(action);ob,rb,tb,xb,_=b.step(action)
        assert ra==rb and (ta,xa)==(tb,xb) and a.current_tick==b.current_tick
        assert a.schedule_log==b.schedule_log and a.reconfiguration_log==b.reconfiguration_log
        if a.task_done:break
    assert DecisionType.WORKER in seen
    network=build_actor_critic(ob,c['network']);assert network.global_encoder[0].in_features==10
    observations=[ob,ob.copy(),pickle.loads(pickle.dumps(ob))];masks=[b.get_action_mask()]*3
    logits,values=network.forward_batch(observations,masks,device='cpu')
    for index,item in enumerate(observations):
        single,value=network(item,masks[index],device='cpu')
        torch.testing.assert_close(single,logits[index,:len(masks[index])],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(value,values[index],rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('encoding',GLOBAL_OBJECTIVE_ENCODINGS)
def test_checkpoint_and_buffer_encoding_identity(encoding,config,fixed_instance,tmp_path):
    c=deepcopy(config);c['network'].update(global_objective_encoding=encoding,hidden_dim=16)
    env=AssemblySchedulingEnv(c);obs=env.reset(fixed_instance)
    agent=PPOAgent(build_actor_critic(obs,c['network']),c['ppo']);path=tmp_path/'model.pt';agent.save(path)
    agent.load(path,load_optimizer=False)
    spec=agent.network.network_spec();assert spec['global_objective_encoding']==encoding
    other=deepcopy(c);other['network']['global_objective_encoding']=next(e for e in GLOBAL_OBJECTIVE_ENCODINGS if e!=encoding)
    other_env=AssemblySchedulingEnv(other);other_obs=other_env.reset(fixed_instance)
    other_agent=PPOAgent(build_actor_critic(other_obs,other['network']),other['ppo'])
    with pytest.raises(ValueError,match='global_objective_encoding'):other_agent.load(path,load_optimizer=False)
    for payload in [{**spec,'global_objective_encoding':other['network']['global_objective_encoding']},
                    {k:v for k,v in spec.items() if k!='global_objective_encoding'}]:
        with pytest.raises(ValueError):infer_checkpoint_network_spec({'network_spec':payload})
    buffer=RolloutBuffer();buffer.add(obs,env.get_action_mask(),0,0,0,0,False)
    with pytest.raises(ValueError,match='encodings'):buffer.add(other_obs,other_env.get_action_mask(),0,0,0,0,False)
    second=RolloutBuffer();second.add(other_obs,other_env.get_action_mask(),0,0,0,0,False)
    with pytest.raises(ValueError,match='encodings'):buffer.extend(second)
    with pytest.raises(ValueError,match='global feature'):build_actor_critic(other_obs,c['network'])
    legacy=deepcopy(c);legacy['network']['observation_schema_version']=11;legacy['network'].pop('global_objective_encoding')
    old_env=AssemblySchedulingEnv(legacy);old_obs=old_env.reset(fixed_instance)
    old_agent=PPOAgent(build_actor_critic(old_obs,legacy['network']),legacy['ppo'])
    with pytest.raises(ValueError,match='requires retraining'):old_agent.load(path,load_optimizer=False)
    old_path=tmp_path/'old.pt';old_agent.save(old_path)
    with pytest.raises(ValueError,match='requires retraining'):agent.load(old_path,load_optimizer=False)


def test_old_schema11_snapshot_extending_default_keeps_linear(config,fixed_instance,tmp_path):
    old=deepcopy(config);old['network']['observation_schema_version']=11;old['network'].pop('global_objective_encoding')
    old['runtime_manifest']=runtime_manifest(old);old['extends']=str(project_path('configs/default.json'))
    path=tmp_path/'old.json';path.write_text(json.dumps(old))
    restored=load_config(path);assert restored['runtime_manifest']==old['runtime_manifest']
    assert 'global_objective_encoding' not in restored['network']
    env=AssemblySchedulingEnv(restored);obs=env.reset(fixed_instance)
    assert len(obs.global_features)==9 and obs.global_feature_names==global_feature_names(11)


def test_presets_differ_only_in_encoding_and_name():
    a=load_config('configs/observation/log2/universal.json');b=load_config('configs/observation/bounded/universal.json')
    for c in [a,b]:
        for key in ['_config_path','_config_chain','experiment_name']:c.pop(key)
        c['network'].pop('global_objective_encoding');c['runtime_manifest'].pop('global_objective_encoding')
    assert a==b


@pytest.mark.parametrize('encoding',GLOBAL_OBJECTIVE_ENCODINGS)
@pytest.mark.parametrize('flow',['raw','excess'])
def test_complete_trajectory_preserves_terminal_objectives(encoding,flow,fixed_instance):
    path='configs/workload/raw/q05/universal.json' if flow=='raw' else 'configs/workload/q05/universal.json'
    c=load_config(path);c['network']['global_objective_encoding']=encoding
    old=deepcopy(c);old['network']['observation_schema_version']=11;old['network'].pop('global_objective_encoding')
    instance=_chain_instance(fixed_instance,mismatch=True,second_release=20)
    a,b=AssemblySchedulingEnv(old),AssemblySchedulingEnv(c);a.reset(instance);b.reset(instance)
    rule=HeuristicPolicy()
    while not a.task_done:
        np.testing.assert_array_equal(a.get_action_mask(),b.get_action_mask())
        action=rule.select_action(a)
        _,left,ta,xa,_=a.step(action);_,right,tb,xb,_=b.step(action)
        assert left==right and (ta,xa)==(tb,xb)
    assert a.task_succeeded and b.task_succeeded
    assert a.schedule_log==b.schedule_log and a.reconfiguration_log==b.reconfiguration_log
    for key in ['flow_time_objective','flow_excess_objective','reconfiguration_cost','worker_load_variance',
                'workload_objective_value','training_cumulative_reward','terminal_reason','maximum_worker_fatigue']:
        assert a.metrics()[key]==b.metrics()[key]
    assert b.observe().global_features[-1]==0


@pytest.mark.parametrize('encoding',GLOBAL_OBJECTIVE_ENCODINGS)
def test_both_encodings_preserve_serial_and_parallel_sampling(encoding,tmp_path,fixed_instance):
    from agent.ppo.parallel import ParallelEpisodeRunner
    from data import load_dataset_split
    from environment import PreferenceContext
    from eval import EvaluationPolicy, evaluate_instance
    from test_module_entrypoints import tiny_config
    source=f"configs/observation/{'log2' if encoding=='log2_v1' else 'bounded'}/universal.json"
    c,tiny=tiny_config(tmp_path,fixed_instance,source=source)
    record=load_dataset_split(c,'validation')[0]
    env=AssemblySchedulingEnv(c);obs=env.reset(record.instance)
    agent=PPOAgent(build_actor_critic(obs,c['network']),c['ppo'])
    preferences=[PreferenceContext.from_input(p) for p in [(1,0,0),(0,1,0),(0,0,1)]]
    serial=[]
    for pref in preferences:
        prepared=EvaluationPolicy(c,policy_name='ppo',bootstrap_observation=obs,ppo_agent=agent,
                                  decode_mode='sampled',sampling_seed=100011)
        _,metrics=evaluate_instance(c,instance=record.instance,policy_name='ppo',prepared_policy=prepared,preference=pref)
        serial.append(metrics)
    with ParallelEpisodeRunner(config=c,template=tiny,episode_count=3,worker_count=2) as runner:
        parallel=runner.evaluate_records(agent,[record]*3,max_parallelism=2,deterministic=False,
            sampling_seed=100011,preferences=preferences)
    for first,second in zip(serial,parallel,strict=True):
        assert first['action_trace_sha256']==second.action_trace_sha256
        for key in ['task_succeeded','flow_time_objective','reconfiguration_cost','workload_objective_value',
                    'worker_load_variance','operation_progress','training_cumulative_reward']:
            assert first[key]==pytest.approx(second.metrics[key])
