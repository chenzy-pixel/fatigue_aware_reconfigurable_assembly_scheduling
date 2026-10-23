from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from configs import load_config
from data.dataset import load_dataset_split,OnlineInstanceDataset,canonical_json_bytes,template_sha256
from data.distribution import CELLS,training_sampling_plan
from data.models import instance_to_dict,parse_instance_dict
from data.generate_orders import generate_instance
from data.feasibility import cheap_feasibility_precheck
from environment import AssemblySchedulingEnv,DecisionType
from environment.types import OperationState
from agent.baselines import HeuristicPolicy
from agent.ppo import build_actor_critic,PPOAgent
from agent.ppo.buffer import RolloutBuffer


@pytest.fixture(scope='module')
def config():
    torch.set_num_threads(1)
    c=load_config('configs/default.json');c['device']='cpu';return c


@pytest.fixture(scope='module')
def data(config): return load_dataset_split(config,'validation')


@pytest.mark.parametrize('cell',CELLS,ids=lambda c:c.cell_id)
def test_generated_cell_structure_and_roundtrip(config,cell):
    i=generate_instance(config,cell,17)
    assert (len(i.orders),len(i.operations),len(i.machines),len(i.workers),len(i.modules))==(cell.N,cell.N*cell.K,cell.M,cell.W,cell.A)
    assert len({m.initial_module for m in i.machines})==cell.M
    assert all(sum(a in m.module_parameters for m in i.machines)==2 for a in i.modules)
    assert all(sum(a in w.qualified_modules for w in i.workers)==2 for a in i.modules)
    assert all(len(w.qualified_modules)==4 for w in i.workers)
    counts=[sum(o.required_module==a for o in i.operations) for a in i.modules]
    assert min(counts)>=2 and max(counts)-min(counts)<=1
    assert all(len({o.required_module for o in j.operations})>=2 for j in i.orders)
    raw=instance_to_dict(i)
    assert 'horizon' not in raw['time'] and 'waves' not in raw
    assert parse_instance_dict(raw)==i
    assert cheap_feasibility_precheck(i).passed


def test_stream_prefix_and_split_isolation(config):
    assert training_sampling_plan(config,27)==training_sampling_plan(config,81)[:27]
    assert set(training_sampling_plan(config,27))==set(CELLS)
    a=OnlineInstanceDataset(config=config,episode_count=27)
    b=OnlineInstanceDataset(config=config,episode_count=81)
    assert template_sha256(a[1].instance)==template_sha256(b[1].instance)
    pools=[load_dataset_split(config,s) for s in ('calibration','validation','test')]
    assert [len(d) for d in pools]==[54,27,27]
    sets=[{r.metadata['seed'] for r in d} for d in pools]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))


def test_all_initial_orders_available(config,data):
    e=AssemblySchedulingEnv(config);e.reset(data[0].instance,build_observation=False)
    assert sum(o.state==OperationState.READY for o in e.operations)==12
    assert sum(o.state==OperationState.BLOCKED for o in e.operations)==24
    assert not e._events


def test_completion_beyond_previous_boundary_and_reward(config):
    i=load_dataset_split(config,'test')[20].instance
    e=AssemblySchedulingEnv(config);e.reset(i,build_observation=False);policy=HeuristicPolicy();total=0.
    while not e.task_done:
        _,reward,_,_,_=e.step(policy.select_action(e),build_observation=False)
        total+=reward.scalarize(config)
    m=e.metrics()
    assert e.task_succeeded and e.current_time>240 and not e.validate_schedule()
    assert m['total_flow_time']==pytest.approx(sum(e._order_completion_tick.values())*i.resolution)
    assert m['flow_excess_objective']==pytest.approx(m['total_flow_time']-m['flow_processing_lower_bound'])
    assert total==pytest.approx(1-m['preference_quality_score'])
    assert m['maximum_worker_fatigue']<=.75+1e-9


def test_recovery_finite_search_and_impossible_stage(config,data):
    e=AssemblySchedulingEnv(config);e.reset(data[0].instance,build_observation=False)
    machine=e.machines[0];module=machine.current_module
    wi=next(k for k,w in enumerate(e.workers) if module in w.spec.qualified_modules)
    e.workers[wi].fatigue=.75
    result=e._earliest_safe_stage_projection(0,wi,module,installation=False,earliest_tick=0)
    assert result is not None and 0<result[0]<=1251
    f=replace(e.instance.fatigue,disassembly_accumulation_rate_per_minute=1.)
    impossible=replace(e.instance,fatigue=f)
    assert not cheap_feasibility_precheck(impossible).passed
    e.instance=impossible;e._stage_projection_cache.clear()
    assert e._earliest_safe_stage_projection(0,wi,module,installation=False,earliest_tick=0) is None


def test_truncation_preserves_physics_and_bootstrap(config,data):
    e=AssemblySchedulingEnv(config);e.reset(data[0].instance,build_observation=False);e.decision_limit=1
    _,r,done,cut,_=e.step(HeuristicPolicy().select_action(e),build_observation=False)
    assert cut and not done and not e.task_failed and not e._failure_penalty_applied
    buffer=RolloutBuffer();e.truncated=False;o=e.observe();mask=e.get_action_mask()
    action=int(np.flatnonzero(~mask)[0]);buffer.add(o,mask,action,0.,0.,0.,False)
    buffer.compute_gae(last_value=3.,gamma=1.,gae_lambda=.95)
    assert buffer.transitions[0].return_value==pytest.approx(3.)


def test_mixed_module_counts_and_worker_phase(config,data):
    envs=[AssemblySchedulingEnv(config) for _ in range(4)]
    obs=[e.reset(data[j].instance) for e,j in zip(envs,(0,3,6,0))]
    while envs[3].decision_type!=DecisionType.WORKER:
        obs[3],_,_,_,_=envs[3].step(HeuristicPolicy().select_action(envs[3]))
    assert len({tuple(o.feature_dimensions.items()) for o in obs})==1
    assert len({tuple(o.edge_feature_dimensions.items()) for o in obs})==1
    masks=[e.get_action_mask() for e in envs]
    n=build_actor_critic(obs[0],config['network']).eval()
    batched=n.forward_batch(obs,masks,device='cpu')
    for j,(o,mask) in enumerate(zip(obs,masks)):
        logits,value=n(o,mask,device='cpu')
        assert torch.allclose(logits,batched[0][j,:len(mask)],atol=1e-5)
        assert torch.allclose(value,batched[1][j],atol=1e-5)


def test_machine_permutation_equivariance(config,data):
    i=data[0].instance;j=replace(i,machines=tuple(reversed(i.machines)))
    a,b=AssemblySchedulingEnv(config),AssemblySchedulingEnv(config);oa,ob=a.reset(i),b.reset(j)
    n=build_actor_critic(oa,config['network']).eval()
    la,va=n(oa,a.get_action_mask(),device='cpu');lb,vb=n(ob,b.get_action_mask(),device='cpu')
    assert torch.allclose(va,vb,atol=1e-5)
    assert torch.allclose(la[:-1].reshape(-1,4).flip(1),lb[:-1].reshape(-1,4),atol=1e-5)


def test_checkpoint_rejection_before_mutation(config,data,tmp_path):
    o=AssemblySchedulingEnv(config).reset(data[0].instance)
    a=PPOAgent(build_actor_critic(o,config['network']),config['ppo'],device='cpu')
    p=tmp_path/'checkpoint.pt';a.save(p);a.load(p)
    payload=torch.load(p,map_location='cpu',weights_only=False)
    payload['network_spec']['observation_schema_version']=12;torch.save(payload,p)
    before={k:v.clone() for k,v in a.network.state_dict().items()}
    with pytest.raises(ValueError): a.load(p)
    assert all(torch.equal(v,before[k]) for k,v in a.network.state_dict().items())


def test_reject_historical_and_future_release(config,data):
    raw=instance_to_dict(data[0].instance);raw['time']['horizon']=240
    with pytest.raises(ValueError): parse_instance_dict(raw)
    raw['time'].pop('horizon');raw['orders'][0]['release_time']=10
    with pytest.raises(ValueError): parse_instance_dict(raw)


def test_invalid_projection_is_explicit(config,data,monkeypatch):
    from environment.resource_projection import ResourceProjector
    monkeypatch.setattr(ResourceProjector,'transition',lambda *a,**k:None)
    monkeypatch.setattr(ResourceProjector,'transition_time',lambda *a,**k:None)
    e=AssemblySchedulingEnv(config);o=e.reset(data[0].instance)
    edge=o.relations[('operation','capable_on','machine')]
    valid=edge.edge_features[:,edge.feature_names.index('projection_valid')]
    finish=edge.edge_features[:,edge.feature_names.index('predicted_finish_time_norm')]
    assert np.all(valid==0) and np.all(finish==0)
    assert o.global_features[-1]==1


def test_cache_and_zero_recovery_do_not_change_physics(config,data):
    alternate=deepcopy(config);alternate['training']['resource_projection_cache_entries']=0
    a,b=AssemblySchedulingEnv(config),AssemblySchedulingEnv(alternate)
    a.reset(data[0].instance);b.reset(data[0].instance)
    policy=HeuristicPolicy()
    for _ in range(25):
        assert np.array_equal(a.get_action_mask(),b.get_action_mask())
        action=policy.select_action(a)
        a.step(action);b.step(action)
        assert a.current_tick==b.current_tick
    a.instance=replace(a.instance,fatigue=replace(a.instance.fatigue,idle_recovery_rate_per_minute=0.))
    a._stage_projection_cache.clear()
    module=a.machines[0].current_module
    wi=next(i for i,w in enumerate(a.workers) if module in w.spec.qualified_modules)
    a.workers[wi].fatigue=.75
    assert a._earliest_safe_stage_projection(0,wi,module,installation=False,earliest_tick=a.current_tick) is None


def test_ddqn_actual_update_and_checkpoint(config,data,tmp_path):
    from agent.comparators.ddqn import RuleDDQN,rule_actions
    c=deepcopy(config);c['comparison']={'ddqn':{'warmup':2,'batch_size':2,'update_every':1},'training_steps':24}
    e=AssemblySchedulingEnv(c);o=e.reset(data[0].instance);a=RuleDDQN(o,c)
    rule=int(np.flatnonzero(rule_actions(o,e.get_action_mask())>=0)[0])
    for _ in range(3): a.remember(o,rule,0.,o,e.get_action_mask(),False)
    assert a.updates>=1
    p=tmp_path/'ddqn.pt';a.save(p);b=RuleDDQN(o,c);b.load(p)
    assert all(torch.equal(v,b.online.state_dict()[k]) for k,v in a.online.state_dict().items())


def test_evaluation_adapter_and_controls(config,data):
    from eval import evaluate_instance,build_evaluation_row
    from result import evaluation_quality_metric,aggregate_evaluation_rows
    e,m=evaluate_instance(config,instance=data[0].instance,policy_name='heuristic')
    row=build_evaluation_row(data[0],m,config,evaluation_quality_metric(config))
    assert row['cell_id']=='C01' and row['result_schema_version']=='10.0.0'
    assert not any('wave' in k or 'deadline' in k for k in row)
    summary=aggregate_evaluation_rows([row],dataset='validation',policy='heuristic',manifest=str(data.manifest_path),quality_metric=evaluation_quality_metric(config))
    assert summary['by_cell_id']
    for path in ('configs/ablations/no_graph.json','configs/ablations/neutral_flow.json','configs/controls/single_variance.json'):
        c=load_config(path);o=AssemblySchedulingEnv(c).reset(data[0].instance)
        build_actor_critic(o,c['network'])


def test_module_related_search_operator(config,data):
    import random
    from types import SimpleNamespace
    from agent.mo_alns.solver import _rule_solution,destroy_operations,DEFAULT_SETTINGS
    i=data[0].instance
    solution=_rule_solution(i,'module_group_randomized',random.Random(3))
    modules={o.id:o.required_module for o in i.operations}
    candidate=SimpleNamespace(diagnostics={'operation_module':modules})
    removed=destroy_operations('same_module_related',solution,candidate,i,DEFAULT_SETTINGS,random.Random(3))
    assert len(removed)>=2 and len(set(removed))==len(removed)
    assert set(removed)<=set(modules)
