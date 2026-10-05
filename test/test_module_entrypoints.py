"""Real subprocess entrypoints on one fixed, legal scheduling instance."""
from dataclasses import replace
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent
from agent.ppo.network import build_actor_critic
from configs import load_config, project_path
from configs.formal_preferences import formal_preferences
from data.dataset import GeneratedInstanceRecord, save_generated_record, template_sha256
from data.models import instance_to_dict, validate_instance
from data.distribution import protocol_hashes
from environment import AssemblySchedulingEnv
from result.io import write_config, write_json


def tiny_config(directory, fixed_instance, source='configs/v8/universal.json', *, training_template=False, environment_updates=None):
    config = load_config(source)
    config['device'] = 'cpu'
    config['environment'].update(environment_updates or {})
    config['network']['hidden_dim'] = 16
    config['paths'].update({name: str(directory/relative) for name,relative in {
        'result_root':'runs', 'instances_root':'instances', 'manifests_root':'manifests',
        'fixed_instance':'template.yaml', 'training_instances_cache':'train_cache'}.items()})
    config['training'].update({'torch_num_threads':1,'parallel_envs':1,'validation_parallel_envs':1,
        'smoke_parallel_envs':1,'smoke_episodes':2,'smoke_rollout_steps':4,'smoke_validation_instance_limit':1})
    config['training']['formal_evaluation'].update({'validation_repeats':1,'final_test_repeats':1})
    config['training']['validation_selection']['diagnostic_instance_limit'] = 0
    config['generator']['dataset_pressure_weights'] = {
        name: float(name == 'easy') for name in config['generator']['dataset_pressure_weights']
    }
    order = fixed_instance.orders[0]
    operation = replace(order.operations[0],base_processing_time=1.0)
    tiny = replace(fixed_instance,instance_id='tiny_template',instance_type='test',
        orders=(replace(order,release_time=0.0,operations=(operation,)),),
        waves={order.wave:{'dominant_module':operation.required_module,'order_ids':[order.id],'release_interval':[0.0,0.0]}})
    validate_instance(tiny)
    directory.mkdir(parents=True,exist_ok=True)
    template = fixed_instance if training_template else tiny
    (directory/'template.yaml').write_text(yaml.safe_dump(instance_to_dict(template)),encoding='utf-8')
    for split,seed in [('test',3000000),('validation',2000000)]:
        instance = replace(tiny,instance_id=f'{split}_tiny_{seed}')
        env = AssemblySchedulingEnv(config)
        env.reset(instance,build_observation=False)
        rule = HeuristicPolicy()
        while not env.task_done:
            env.step(rule.select_action(env),build_observation=False)
        metrics = env.metrics()
        metadata = {'seed':seed,'split':split,'generator_version':config['generator']['version'],
            'template_instance':config['dataset']['template_instance'],'template_sha256':template_sha256(template),
            **protocol_hashes(config), 'severity':1.0,
            'feasibility_status':'unknown',
            'diagnostic_status':'completed' if metrics['task_succeeded'] else 'truncated',
            'diagnostic_terminal_reason':metrics.get('terminal_reason'),
            'pressure_type':'easy','cost_profile':'balanced_cost',
            'pressure_metrics':{'total_effective_load':0.0,'max_module_load':0.0},
            'heuristic_metrics':{'heuristic_completed':metrics['task_succeeded'],'heuristic_makespan':metrics['time'],
                'heuristic_flow_time':metrics['flow_time_objective'],'heuristic_reconfiguration_cost':metrics['reconfiguration_cost'],
                'worker_workload_variance':metrics['worker_load_variance'],'ready_configuration_gap_ratio':0.0,
                'heuristic_reconfiguration_ratio':0.0,'mean_wave_overlap_ratio':0.0}}
        filename = f'instance_{seed}.json'
        digest = save_generated_record(GeneratedInstanceRecord(instance,metadata),directory/'instances'/split/filename)
        path = directory/'manifests'/split/'manifest.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        write_json(path,{'schema_version':config['dataset']['schema_version'],'generator_version':config['generator']['version'],
            'template_instance':config['dataset']['template_instance'],'template_sha256':template_sha256(template),
            **protocol_hashes(config), 'generation_summary':{'pressure_counts':{'easy':1}},
            'split':split,'instance_count':1,'seed_start':seed,'files':[{'path':filename,'seed':seed,'sha256':digest}]})
    write_config(directory,config)
    return config,tiny


def run_cli(command,log_path):
    result = subprocess.run([sys.executable,*command],cwd=project_path('.'),
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True,text=True,timeout=240)
    log_path.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    assert result.returncode == 0, result.stdout+'\n'+result.stderr


@pytest.mark.parametrize('source',['configs/baselines/mo_alns.json','configs/baselines/mo_alns_smoke.json'])
def test_real_baseline_configs_solve_and_write_current_results(source,tmp_path,fixed_instance):
    config,_ = tiny_config(tmp_path,fixed_instance,source)
    run_cli(['scripts/mo_alns.py','--config',str(tmp_path/'config.json'),'--dataset','test',
        '--canonical-only','--instance-limit','1','--parallel-envs','1','--smoke','--run-name','baseline'],tmp_path/'cli.log')
    directory = tmp_path/'runs/baseline'
    metrics = json.loads((directory/'metrics.json').read_text())
    assert metrics['evaluation_schema_version'] == '8.0.0'
    assert metrics['candidate_budget_per_preference'] == 8
    assert metrics['search_scalarizer']['scales'] == config['objective_scalarizer']['scales']
    assert metrics['provenance']['objective_scales'] == config['objective_scalarizer']['scales']
    assert metrics['schedule_violation_count'] == 0
    with (directory/'instance_metrics.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]['single_stage_proxy_return']
    assert rows[0]['method_version'] == 'MO_ALNS_v1'
    for filename in ('config.json','schedule.csv','pareto_archive.csv','search_log.csv','operator_statistics.csv'):
        assert (directory/filename).is_file()


def test_real_baseline_manifest_entrypoint(tmp_path,fixed_instance):
    config, _ = tiny_config(tmp_path,fixed_instance,'configs/baselines/mo_alns_smoke.json')
    manifest = {'protocol':'ppo_mo_alns_solver_budget_v2','config':str(tmp_path/'config.json'),
        'datasets':['test'],'algorithm_seeds':[11],'instance_limit':1,'parallel_envs':1}
    write_json(tmp_path/'manifest.json',manifest)
    run_cli(['scripts/mo_alns_benchmark.py','--manifest',str(tmp_path/'manifest.json'),
        '--output-dir',str(tmp_path/'benchmark')],tmp_path/'manifest_cli.log')
    summary = json.loads((tmp_path/'benchmark/summary.json').read_text())
    assert summary['run_count'] == 1
    with (tmp_path/'benchmark/candidates.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(formal_preferences(config, 'final_test'))
    assert all(row['replay_verified']=='True' for row in rows)


def test_real_schema10_training_update_validation_and_final_grid(tmp_path,fixed_instance):
    config,tiny = tiny_config(tmp_path,fixed_instance,training_template=True)
    run_cli(['train.py','--config',str(tmp_path/'config.json'),'--smoke','--parallel-envs','1',
        '--episodes-per-update','2','--validation-parallel-envs','1','--run-name','ppo'],tmp_path/'train_cli.log')
    directory = tmp_path/'runs/ppo'
    assert (directory/'best_checkpoint.pt').is_file() and (directory/'last_checkpoint.pt').is_file()
    for filename,count in [('sampled_validation_instance_metrics.csv',13),('final_sampled_instance_metrics.csv',66)]:
        with (directory/filename).open(encoding='utf-8-sig',newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == count and all(int(row['schedule_violation_count'])==0 for row in rows)
    payload = torch.load(directory/'best_checkpoint.pt',weights_only=False)
    assert payload['network_spec']['feature_dimensions']['global'] == 9
    assert payload['network_spec']['observation_schema_version'] == 10
    env = AssemblySchedulingEnv(config)
    obs = env.reset(tiny)
    agent = PPOAgent(build_actor_critic(obs,config['network']),config['ppo'])
    agent.load(directory/'best_checkpoint.pt')
    agent.load(directory/'last_checkpoint.pt',load_optimizer=True)
    assert load_config(directory/'config.json')['runtime_manifest']['observation_schema'] == 10


def test_real_training_external_guards_keep_partial_rows_and_last_checkpoint(tmp_path, fixed_instance):
    config, _ = tiny_config(tmp_path, fixed_instance, training_template=True,
                            environment_updates={'max_decisions':1})
    path = tmp_path/'config.json'
    config['training'].update(episodes=2, validation_interval_episodes=1,
                             validation_instance_limit=1, torch_num_threads=1)
    config['training']['formal_evaluation'].update(validation_repeats=1, final_test_repeats=1)
    write_config(tmp_path, config)
    run_cli(['train.py','--config',str(path),'--episodes','2','--parallel-envs','1',
          '--episodes-per-update','2','--validation-parallel-envs','1','--run-name','guard'],
         tmp_path/'guard_cli.log')
    directory = tmp_path/'runs/guard'
    summary = json.loads((directory/'summary.json').read_text())
    assert summary['sampling_attempt_count'] == summary['sampling_truncated_count'] == 2
    assert summary['task_succeeded_count'] == summary['task_failed_count'] == 0
    assert summary['best_checkpoint'] is None and summary['final_sampled'] is None
    assert (directory/'last_checkpoint.pt').is_file()
    with (directory/'sampled_validation_instance_metrics.csv').open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 13
    assert all(row['sampling_truncated'] == 'True' and row['task_failed'] == 'False' for row in rows)
    with (directory/'validation_log.csv').open(encoding='utf-8-sig') as handle:
        validation = list(csv.DictReader(handle))
    assert all(row['evaluation_complete'] == 'False' and row['completion_rate'] == '' for row in validation)
