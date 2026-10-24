"""Scientific interventions, on-policy A2C and strict matrix accounting."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configs.experiment_suite import load_suite, arm_config, preferences, matrix
from data import load_dataset_split
from environment import AssemblySchedulingEnv, DecisionType
from agent.ppo import build_actor_critic, PPOAgent
from agent.comparators.a2c import A2CAgent
from agent.comparators.common import FOUR, digest
from agent.comparators.evolution import EvolutionSolver
from agent.comparators.experiment_evaluation import search_snapshots, search_unit, learning_unit, frontier_metrics, quality
from agent.baselines.experiment_rules import FixedRulePolicy, evaluate_rule, RULES
from agent.search import SearchBudget
from analysis.experiment_suite import stratified_interval, holm, audit_evaluations, instance_tables, common_success_statistics


@pytest.fixture(scope='module')
def config():
    torch.set_num_threads(1)
    c = arm_config(load_suite(),'full',11)
    c['network'].update(hidden_dim=16,message_passing_layers=1)
    return c


@pytest.fixture(scope='module')
def tiny(config):
    instance = load_dataset_split(config,'calibration')[0].instance
    instance = replace(instance,instance_id='tiny_suite',orders=(instance.orders[0],))
    target = instance.operations[0].required_module
    return replace(instance,machines=tuple(replace(m,initial_module=next((a for a in m.module_parameters if a != target),m.initial_module)) for m in instance.machines))


def phase_samples(config,tiny):
    e = AssemblySchedulingEnv(config)
    obs = e.reset(tiny)
    found = {}
    policy = FixedRulePolicy('existing_heuristic')
    while not e.task_done and len(found) < 2:
        found.setdefault(e.decision_type,(obs.copy(),e.get_action_mask().copy()))
        obs,_,_,_,_ = e.step(policy.select_action(e))
    assert set(found) == {DecisionType.PRODUCTION,DecisionType.WORKER}
    return list(found.values())


@pytest.mark.parametrize('switch',['direct_features_enabled','residual_enabled'])
def test_full_defaults_and_checkpoint_intervention_identity(config,tiny,tmp_path,switch):
    e = AssemblySchedulingEnv(config);o=e.reset(tiny);mask=e.get_action_mask()
    torch.manual_seed(17);default=build_actor_critic(o,config['network'])
    c=deepcopy(config['network']);c[switch]=True
    torch.manual_seed(17);explicit=build_actor_critic(o,c)
    assert all(torch.equal(v,explicit.state_dict()[k]) for k,v in default.state_dict().items())
    a,_=default(o,mask,device='cpu');b,_=explicit(o,mask,device='cpu')
    assert torch.equal(a,b)
    agent=PPOAgent(default,config['ppo']);agent.save(tmp_path/'full.pt')
    c[switch]=False
    variant=PPOAgent(build_actor_critic(o,c),config['ppo'])
    with pytest.raises(ValueError,match=switch): variant.load(tmp_path/'full.pt')


@pytest.mark.parametrize('arm',['full','no_graph','no_direct','no_residual'])
def test_batched_and_reference_interventions(config,tiny,arm):
    c=arm_config(load_suite(),arm,11)
    c['network'].update(hidden_dim=16,message_passing_layers=1)
    samples=phase_samples(c,tiny);obs,masks=map(list,zip(*samples))
    network=build_actor_critic(obs[0],c['network'])
    called, hooks = [], []
    if arm == 'no_direct':
        def check(module,args):
            called.append(True)
            assert all(torch.count_nonzero(v)==0 for v in args[1].values())
        hooks=[getattr(network,name).register_forward_pre_hook(check) for name in
               ('production_shared','worker_shared','production_wait_shared','worker_wait_shared')]
    if arm == 'no_residual':
        def fail(*args): raise AssertionError('disabled residual executed')
        hooks=[m.register_forward_pre_hook(fail) for m in
               (network.production_residual,network.worker_residual,network.action_preference_projector)]
        assert not network.production_residual_gate.requires_grad
        agent=PPOAgent(network,c['ppo'])
        trained={id(p) for g in agent.optimizer.param_groups for p in g['params']}
        assert all(id(p) not in trained for p in network.production_residual.parameters())
    with torch.no_grad():
        a,av=network.forward_batch(obs,masks,device='cpu')
        network.execution_mode='reference_v8'
        b,bv=network.forward_batch(obs,masks,device='cpu')
    assert torch.allclose(a,b,atol=2e-5,rtol=2e-5)
    assert torch.allclose(av,bv,atol=2e-5,rtol=2e-5)
    if arm == 'no_direct': assert len(called)==8
    for hook in hooks: hook.remove()


class ToyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__();self.actor=torch.nn.Parameter(torch.tensor([.2,-.1]));self.critic=torch.nn.Parameter(torch.tensor(.3));self.seen=[]
    def forward_batch(self,observations,masks,*,device):
        self.seen.extend(observations)
        x=torch.tensor(observations,dtype=torch.float32)
        return x[:,None]*self.actor[None,:],x*self.critic


def test_a2c_accumulates_exactly_one_fresh_pass_and_update(config,monkeypatch):
    transitions=[SimpleNamespace(observation=float(i),action_mask=np.array([False,False]),action=i%2,
                                 advantage=float(i-2),return_value=float(i)/3,log_probability=9999.) for i in range(1,6)]
    buffer=SimpleNamespace(transitions=transitions)
    models=[]
    for batch in (2,5):
        settings={**config['ppo'],'batch_size':batch,'epochs':99,'max_grad_norm':100.}
        agent=A2CAgent(ToyNetwork(),settings)
        agent.optimizer=torch.optim.SGD(agent.network.parameters(),lr=.1)
        steps=[]; original=agent.optimizer.step
        def step(): steps.append(1);return original()
        monkeypatch.setattr(agent.optimizer,'step',step)
        result=agent.update(buffer)
        assert len(steps)==result['optimizer_updates']==1
        assert result['sample_presentations']==5 and agent.network.seen==[1.,2.,3.,4.,5.]
        assert not torch.equal(agent.network.actor,torch.tensor([.2,-.1]))
        models.append(agent.network)
    for a,b in zip(models[0].parameters(),models[1].parameters()): assert torch.allclose(a,b,atol=1e-7)


def test_a2c_checkpoint_family_before_weight_mutation(config,tiny,tmp_path):
    e=AssemblySchedulingEnv(config);obs=e.reset(tiny)
    ppo=PPOAgent(build_actor_critic(obs,config['network']),config['ppo'])
    ppo.save(tmp_path/'ppo.pt')
    a2c=A2CAgent(build_actor_critic(obs,config['network']),config['ppo'])
    before={k:v.clone() for k,v in a2c.network.state_dict().items()}
    with pytest.raises(ValueError,match='algorithm differs'):a2c.load(tmp_path/'ppo.pt')
    assert all(torch.equal(v,a2c.network.state_dict()[k]) for k,v in before.items())
    a2c.save(tmp_path/'a2c.pt');assert a2c.load(tmp_path/'a2c.pt')['algorithm']=='a2c'


@pytest.mark.parametrize('name',RULES)
def test_fixed_rules_complete_legally_and_deterministically(config,tiny,name):
    first=evaluate_rule(config,tiny,name);second=evaluate_rule(config,tiny,name)
    assert first['success'] and not first['violations'] and first['trace']==second['trace']
    assert first['metrics']['maximum_worker_fatigue']<=first['metrics']['safe_fatigue_limit']+1e-9


def test_projection_fallback_stable_ids_and_wait(config,tiny):
    from agent.comparators.common import candidates
    e=AssemblySchedulingEnv(config);o=e.reset(tiny)
    edge,_,_,ids=candidates(o)
    edge.edge_features[:,edge.feature_names.index('projection_valid')]=0
    legal=[i for i,a in enumerate(ids) if not e.get_action_mask()[a]]
    i=min(legal,key=lambda i:(e.operations[edge.edge_index[0,i]].spec.id,e.machines[edge.edge_index[1,i]].spec.id))
    assert FixedRulePolicy('eft_lf').select_action(e,o)==int(ids[i])
    original=e.get_action_mask
    mask=np.ones_like(original());mask[e.wait_action]=False
    e.get_action_mask=lambda:mask
    assert FixedRulePolicy('srw_lf').select_action(e,o)==e.wait_action
    mask[e.wait_action]=True
    with pytest.raises(ValueError,match='no legal action'): FixedRulePolicy('srw_lf').select_action(e,o)


@pytest.mark.parametrize('method',['nsga2_ls','moead'])
def test_frontier_one_budget_native_feasibility_and_snapshots(config,tiny,method):
    c=deepcopy(config);c['comparison']['evolution']['population_size']=8
    solver=EvolutionSolver(c,method,algorithm_seed=23)
    result=solver.solve_frontier(tiny,FOUR,budget=SearchBudget(max_evaluations=40))
    assert len(result.searches)==1 and len(result.endpoints)==4
    assert result.archive and all(c.feasible for c in result.archive)
    paid=result.searches[0]
    assert paid.accounting['started_evaluations']<=40 and paid.environment_evaluations<=40
    assert paid.operator_statistics['initialization_complete']
    assert all(c.completed_seconds<=paid.search_time_seconds for c in result.archive)
    snaps=search_snapshots(c,paid,FOUR,[0,paid.search_time_seconds])
    assert snaps[0]['paid_hv']==0 and snaps[1]['paid_hv']>0
    with pytest.raises(ValueError,match='duplicate'): solver.solve_frontier(tiny,[FOUR[0],FOUR[0]])


def test_snapshot_excludes_future_completed_candidate(config):
    r=SimpleNamespace(candidate_history=[{'feasible':True,'completed_seconds':61.,'objectives':(1.,2.,3.),'action_trace_sha256':'late'}])
    snapshots=search_snapshots(config,r,FOUR,[60,180])
    assert snapshots[0]['distinct_feasible']==0 and snapshots[0]['paid_hv']==0
    assert snapshots[1]['distinct_feasible']==1 and snapshots[1]['paid_hv']>0


@pytest.mark.parametrize('task',['frontier','independent_request'])
def test_search_artifact_conversion_and_paid_accounting(config,tiny,task):
    c=deepcopy(config);c['comparison']['evolution']['population_size']=8
    record=SimpleNamespace(instance=tiny)
    result=search_unit(c,record,'moead',FOUR,11,task,seconds=5,max_evaluations=12,checkpoints=(0,5))
    assert len(result['deliveries'])==4
    assert len(result['searches'])==(1 if task=='frontier' else 4)
    assert all(d['success'] for d in result['deliveries'])
    assert result['environment_steps']==sum(s['accounting']['environment_steps'] for s in result['searches'])
    assert result['paid_hv']>0


def test_matrix_and_four_preferences_are_explicit(config):
    s=load_suite();m=matrix(s)
    assert len(m['training'])==35 and m['search']=={'frontier':270,'independent_request':1080}
    assert m['validation_trajectories']==666900
    assert len(preferences(config,'frontier'))==66 and len(preferences(config,'independent_request'))==4
    assert FOUR[-1] not in preferences(config,'frontier')
    assert len(set(arm_config(s,a,11)['network'].get('encoder_variant','hetero_gnn') for a in s['arms']))==2


def test_instance_bootstrap_holm_and_known_hv(config):
    result=stratified_interval([2.]*27,['small']*9+['medium']*9+['large']*9)
    assert result=={'mean':2.,'ci_low':2.,'ci_high':2.,'instances':27}
    assert holm([.01,.04,.03])==pytest.approx([.03,.06,.06])
    from analysis.pareto_analysis import hypervolume_3d
    assert hypervolume_3d([(.2,.3,.4)])==pytest.approx(.8*.7*.6)
    assert hypervolume_3d([])==0


def test_common_success_averages_replicates_before_instance_pairing():
    base={'task':'independent_request','instance_id':'i','preference':'p','scale':'small',
          'flow':10.,'cost':20.,'workload':3.}
    rows=[{**base,'arm':'full','seed':11,'success':True,'quality':.2},
          {**base,'arm':'full','seed':23,'success':False,'quality':1.},
          {**base,'arm':'moead','seed':11,'success':False,'quality':1.},
          {**base,'arm':'moead','seed':23,'success':True,'quality':.3}]
    suite={'rules':[],'statistics':{'bootstrap_samples':100,'bootstrap_seed':0}}
    results=common_success_statistics(rows,suite)
    item=next(r for r in results if r['task']=='independent_request' and r['arm']=='moead' and r['metric']=='quality')
    assert item['mean']==pytest.approx(.1) and item['instances']==1 and item['paired_preferences']==1


def test_budget_freeze_requires_full_applicable_evidence(tmp_path,config):
    from scripts.experiment_suite import freeze_budget,check_frozen_budget
    from agent.comparators.common import atomic_json
    suite=load_suite()
    with pytest.raises(ValueError,match='incomplete'):freeze_budget(suite,tmp_path,'id')
    for task in ('frontier','independent_request'):
        for method in suite['search_methods']:
            for cell in suite['search']['development_cells']:
                prefs=preferences(config,task)
                groups=[prefs] if task=='frontier' else [(p,) for p in prefs]
                for i,group in enumerate(groups):
                    payload={'identity':'id','status':'complete','task':task,'arm':method,'cell_id':cell,'seed':11,
                             'deliveries':[{'preference':p} for p in group],
                             'searches':[{'operator_statistics':{'initialization_complete':True,'completed_generations':1,'completed_scans':0},
                                          'snapshots':[{'distinct_feasible':2}]}]}
                    atomic_json(tmp_path/'development'/task/method/f'{cell}_{i}.json',payload)
    freeze_budget(suite,tmp_path,'id')
    assert check_frozen_budget(suite,tmp_path,'id')['status']=='frozen'
    evidence=next((tmp_path/'development').rglob('*.json'))
    evidence.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='evidence differs'):check_frozen_budget(suite,tmp_path,'id')


def test_execution_lock_releases_and_blocks_concurrent_job(tmp_path):
    from scripts.experiment_suite import execution_lock
    with execution_lock(tmp_path):
        with pytest.raises(RuntimeError,match='another suite job'):
            with execution_lock(tmp_path):pass
    with execution_lock(tmp_path):pass


def test_formal_stage_requires_remote_run_flag(monkeypatch,tmp_path):
    import sys
    import scripts.experiment_suite as suite
    monkeypatch.setattr(suite,'suite_identity',lambda *args:'id')
    monkeypatch.setattr(suite,'check_preflight',lambda *args:None)
    monkeypatch.setattr(sys,'argv',['suite','--stage','train','--device','cpu','--output-dir',str(tmp_path)])
    with pytest.raises(SystemExit) as error:suite.main()
    assert error.value.code==2


def fake_units(config):
    record=SimpleNamespace(instance=SimpleNamespace(instance_id='case'),metadata={'cell_id':'C01','scale':'small','N':12,'K':3,'M':4})
    suite={'arms':{'full':{}},'rules':[],'search_methods':[],'seeds':[11]}
    units=[]
    for task in ('frontier','independent_request'):
        for arm in ('full','full_greedy','full_sample1'):
            deliveries=[];requests=[]
            for p in preferences(config,task):
                raw={'objectives':(10.,20.,3.),'success':True}
                deliveries.append({'preference':p,'success':True,'quality':quality(config,raw['objectives'],p),
                                   'objectives':raw['objectives'],'search_seconds':.1})
                requests.append({'raw':[raw]})
            metrics=frontier_metrics(config,[{'objectives':(10.,20.,3.),'success':True}],deliveries)
            units.append({'arm':arm,'task':task,'seed':11,'identity':'id','status':'complete','instance_id':'case',
                          'cell_id':'C01','scale':'small','seconds':1.,'deliveries':deliveries,'requests':requests,**metrics})
    return suite,[record],units


def test_report_rejects_missing_duplicate_and_corrupt_quality(config):
    suite,records,units=fake_units(config)
    rows=audit_evaluations(units,suite,config,records,'id')
    table=instance_tables(rows,units)
    assert len(table)==6 and all(r['success_rate']==1 for r in table)
    with pytest.raises(ValueError,match='incomplete'): audit_evaluations(units[:-1],suite,config,records,'id')
    with pytest.raises(ValueError,match='duplicate'): audit_evaluations(units+units[:1],suite,config,records,'id')
    damaged=deepcopy(units);damaged[0]['deliveries'][0]['quality']+=.1
    with pytest.raises(ValueError,match='quality'): audit_evaluations(damaged,suite,config,records,'id')


def test_report_checks_reused_parent_and_trajectory_identity(config):
    suite, records, units = fake_units(config)
    parent = next(u for u in units if u['arm'] == 'full' and u['task'] == 'frontier')
    derived = next(u for u in units if u['arm'] == 'full_sample1' and u['task'] == 'frontier')
    parent['job_id'] = 'parent'
    parent['content_sha256'] = digest(parent)
    derived.update(cost_reference='full', source_job_id='parent', source_content_sha256=parent['content_sha256'])
    # Streaming file order need not place the parent before the derived unit.
    audit_evaluations(reversed(units), suite, config, records, 'id')
    derived['source_content_sha256'] = 'stale'
    with pytest.raises(ValueError, match='parent identity'):
        audit_evaluations(units, suite, config, records, 'id')
    derived['source_content_sha256'] = parent['content_sha256']
    derived['requests'][0]['raw'][0]['actions'] = [123]
    with pytest.raises(ValueError, match='first trajectories'):
        audit_evaluations(units, suite, config, records, 'id')


def test_complete_report_pipeline_with_synthetic_results(config,tmp_path,monkeypatch):
    """Exercise CSV, statistics and every figure; these are fixture values."""
    import json
    import data
    import analysis.experiment_suite as reporting
    import agent.comparators.learning as comparison
    import result.training_history as summarizer
    import scripts.experiment_suite as executor
    from scripts.experiment_suite import save_unit
    suite=load_suite();suite['seeds']=[11];suite['statistics']['bootstrap_samples']=100
    cases=load_dataset_split(config,'test')
    cases=[cases[i] for i in (0,13,26)]
    monkeypatch.setattr(data,'load_dataset_split',lambda *args:cases)
    monkeypatch.setattr(reporting,'validate_training_streams',lambda *args:{11:'synthetic'})
    monkeypatch.setattr(comparison,'check_learning_checkpoint',lambda *args:None)
    monkeypatch.setattr(summarizer,'export_training',lambda *args:None)
    monkeypatch.setattr(executor,'check_frozen_budget',lambda *args:None)
    monkeypatch.setattr(reporting,'preferences',lambda c,t:FOUR)
    for arm in suite['arms']:
        folder=tmp_path/'training'/arm/'11';folder.mkdir(parents=True)
        for filename,value in [('summary.json',{'training_seconds':1.}),
                               ('training_log.json',[{'epoch':1,'training_steps':10,'episodes_started':20,'training_seconds':1.,'mean_successful_quality':.2}]),
                               ('validations.json',[{'training_steps':10,'summary':{'quality':.2}}])]:
            (folder/filename).write_text(json.dumps(value),encoding='utf-8')
    methods=list(suite['arms'])+suite['rules']+suite['search_methods']+['full_greedy','full_sample1']
    for task in ('frontier','independent_request'):
        for index,arm in enumerate(methods):
            for record in cases:
                seed=None if arm in suite['rules'] else 11
                raw={'success':True,'objectives':(10.+index,20.,3.)}
                deliveries=[{'preference':p,'success':True,'quality':quality(config,raw['objectives'],p),
                             'objectives':raw['objectives'],'search_seconds':.1} for p in FOUR]
                payload={'seconds':1.,'deliveries':deliveries,**frontier_metrics(config,[raw],deliveries)}
                if arm in suite['search_methods']:
                    payload['searches']=[{'preference':None,'archive':[{'objectives':raw['objectives']}],
                                          'snapshots':[{'seconds':t,'paid_hv':payload['paid_hv'],'mean_quality':.2} for t in (60,180,600)]}]
                elif arm in suite['rules']: payload['raw_rule']=raw
                else: payload['requests']=[{'raw':[raw]} for _ in FOUR]
                path=tmp_path/'evaluation'/task/arm/f'{record.instance.instance_id}.json'
                save_unit(path,payload,identity='synthetic',arm=arm,seed=seed,task=task,record=record,job_id='synthetic')
    reporting.report(tmp_path,suite,'synthetic')
    out=tmp_path/'report'
    assert json.loads((out/'audit.json').read_text(encoding='utf-8'))['status']=='passed'
    for filename in ('paired_statistics.csv','seed_variability.csv','table_evolution.csv',
                     'learning_curves.svg','search_curves_frontier.pdf','frontier_C27_seed11.png'):
        assert (out/filename).stat().st_size>0
