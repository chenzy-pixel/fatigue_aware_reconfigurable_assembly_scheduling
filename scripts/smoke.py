"""Small local integration; deliberately separate from all formal artifacts."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json

import torch

from configs.experiment_suite import arm_config, suite_identity
from data import load_dataset_split
from environment import AssemblySchedulingEnv
from agent.baselines.experiment_rules import evaluate_rule
from agent.comparators.common import atomic_json, digest
from agent.comparators.learning import train_request, build_agent, evaluate_request, check_learning_checkpoint
from agent.comparators.experiment_evaluation import search_unit


def run_smoke(suite, root):
    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=True)
    streams, report = [], {'purpose':'local_interface_and_mechanism_verification', 'performance_evidence':False, 'arms':{}}
    source_identity = suite_identity(suite,'cpu')
    root = root/source_identity[:16]
    root.mkdir(parents=True, exist_ok=True)
    report['source_identity'] = source_identity
    for arm, definition in suite['arms'].items():
        c = arm_config(suite,arm,11,'cpu')
        c['training'].update(episodes=2, episodes_per_update=2, validation_interval_epochs=1, validation_parallel_envs=2)
        c['comparison']['ddqn'].update(warmup=2,batch_size=2,update_every=1)
        # Small network only for HGNN smoke, explicitly outside the formal identity.
        c['network'].update(hidden_dim=16,message_passing_layers=1)
        original = load_dataset_split(c,'validation')[0]
        tiny = replace(original.instance, instance_id='smoke_validation', orders=original.instance.orders[:1])
        dataset = [SimpleNamespace(instance=tiny, metadata=original.metadata)]
        folder = root/'training'/arm
        identity = digest(['local_smoke',source_identity,arm,c['network']])
        summary = train_request(c,definition['method'],11,c['ppo']['learning_rate'],dataset,[0],[(1/3,1/3,1/3)],
                                folder,identity,episodes=2,workers=2,smoke_instance=tiny)
        assert summary['epochs'] == summary['validation_count'] == 1 and summary['episodes_started'] == 2
        assert summary['optimizer_updates'] > 0
        episodes = json.loads((folder/'episodes.json').read_text(encoding='utf-8'))
        validations = json.loads((folder/'validations.json').read_text(encoding='utf-8'))
        paid = json.loads((folder/validations[0]['deliveries_path']).read_text(encoding='utf-8'))
        assert sum(len(s['actions']) for d in paid['deliveries'] for s in d['raw']) == summary['validation_env_steps']
        streams.append([(r['record_sha256'],r['preference']) for r in episodes])
        agent = build_agent(c,AssemblySchedulingEnv(c).reset(dataset[0].instance),definition['method'])
        check_learning_checkpoint(folder/'best_checkpoint.pt', c, definition['method'], 11, identity)
        agent.load(folder/'best_checkpoint.pt')
        network = agent.online if definition['method']=='rule_ddqn' else agent.network
        weights = {k:v.clone() for k,v in network.state_dict().items()}
        pref = (1/3,1/3,1/3)
        sampled = evaluate_request(c,agent,dataset[0].instance,pref,definition['method'],seed=11)
        assert all(torch.equal(v,network.state_dict()[k]) for k,v in weights.items())
        if arm == 'full':
            single = evaluate_request(c,agent,dataset[0].instance,pref,'ppo',mode='sampled_single',seed=11)
            assert sampled['raw'][0]['actions'] == single['raw'][0]['actions']
            assert single['summary']['candidate_count'] == 1 and sampled['summary']['candidate_count'] == 3
        report['arms'][arm] = {'training_steps':summary['training_steps'], 'updates':summary['optimizer_updates'],
                               'evaluation_candidates':sampled['summary']['candidate_count'], 'reload_passed':True}
        print(f'smoke {arm}: train/reload/evaluation passed',flush=True)
    assert all(stream == streams[0] for stream in streams)
    c = arm_config(suite,'full',11,'cpu')
    record = load_dataset_split(c,'calibration')[0]
    tiny = replace(record.instance,instance_id='suite_smoke',orders=(record.instance.orders[0],))
    record = SimpleNamespace(instance=tiny,metadata=record.metadata)
    report['rules'] = {}
    for name in suite['rules']:
        r = evaluate_rule(c,tiny,name)
        assert r['success'] and not r['violations']
        report['rules'][name] = {'trace':r['trace'],'steps':len(r['actions'])}
    report['search'] = {}
    c['comparison']['evolution']['population_size'] = 8
    for method in suite['search_methods']:
        for task in ('frontier','independent_request'):
            r = search_unit(c,record,method,[(1.,0.,0.),(1/3,1/3,1/3)],11,task,
                            seconds=3,checkpoints=(.01,1,3),max_evaluations=40)
            assert len(r['searches']) == (1 if task=='frontier' else 2)
            assert all(s['accounting']['started_evaluations'] <= 40 for s in r['searches'])
            assert all(d['success'] for d in r['deliveries'])
            atomic_json(root/f'{method}_{task}.json',r)
            report['search'][f'{method}_{task}'] = {'paid_hv':r['paid_hv'],'seconds':r['seconds']}
    report['paired_training_streams'] = True
    atomic_json(root/'verification.json',report)
    print(f'local smoke evidence: {root}',flush=True)
