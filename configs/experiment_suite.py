"""Versioned experiment matrix and identity shared by execution and reports."""
import hashlib
import json
from pathlib import Path

from configs import load_config, project_path
from configs.config import public_config, PROJECT_ROOT
from configs.formal_preferences import formal_preferences
from agent.comparators.common import digest, method_config, FOUR
from agent.baselines.experiment_rules import RULES

DEFAULT_MANIFEST = 'configs/experiment_suite_v1.json'
TASKS = ('frontier', 'independent_request')


def load_suite(path=DEFAULT_MANIFEST):
    suite = json.loads(project_path(path).read_text(encoding='utf-8'))
    if suite['version'] != 'factorial27_experiment_suite_v1':
        raise ValueError('unknown experiment suite version')
    if suite['seeds'] != [11, 23, 37, 53, 71] or len(suite['arms']) != 7:
        raise ValueError('formal matrix requires seven configurations and five seeds')
    if set(suite['arms']) != {'full', 'no_graph', 'no_direct', 'no_residual', 'a2c', 'rule_ddqn', 'dcan'}:
        raise ValueError('formal training arms differ')
    if tuple(suite['rules']) != RULES or suite['search_methods'] != ['nsga2_ls', 'moead']:
        raise ValueError('formal comparison methods differ')
    if suite['training'] != dict(budget_unit='episodes', episodes=2000, episodes_per_update=20, validation_interval_epochs=5):
        raise ValueError('formal training budget differs')
    if suite['search']['seconds'] != 600 or suite['search']['checkpoints'] != [60, 180, 600]:
        raise ValueError('formal search budget differs')
    if suite['execution']['concurrent_jobs'] != 1 or suite['execution']['torch_threads'] != 1:
        raise ValueError('v1 requires serial jobs and one torch thread')
    if int(suite['execution'].get('evaluation_workers', 20)) < 1:
        raise ValueError('evaluation_workers must be positive')
    return suite


def arm_config(suite, arm, seed, device='cpu'):
    definition = suite['arms'][arm]
    c = load_config(project_path(suite['base_config']))
    c['network'].update(definition['network'])
    c['training'].update(suite['training'])
    c['training']['parallel_envs'] = suite['execution']['training_workers']
    c['training']['validation_parallel_envs'] = suite['execution'].get('evaluation_workers', 20)
    c['device'] = device
    if definition['method'] == 'a2c':
        c['ppo']['epochs'] = 1
    c = method_config(c, definition['method'], seed)
    c['comparison']['arm'] = arm
    c['comparison']['validation_storage'] = 'per_epoch_v1'
    return c


def preferences(config, task):
    if task == 'frontier':
        return tuple(p.as_tuple() for p in formal_preferences(config, 'final_test'))
    if task == 'independent_request':
        return FOUR
    raise ValueError(f'unknown evaluation task: {task}')


def identity_payload(suite, device='cpu'):
    from data import load_dataset_split
    from result.provenance import execution_source_files
    c = arm_config(suite, 'full', suite['seeds'][0], device)
    selected = execution_source_files(PROJECT_ROOT)
    sources = {p.relative_to(PROJECT_ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(selected)}
    pools = {split: load_dataset_split(c, split).manifest for split in ('calibration', 'validation', 'test')}
    configs = {arm: public_config(arm_config(suite, arm, suite['seeds'][0], device)) for arm in suite['arms']}
    return {'identity_version':'experiment_execution_v2', 'suite': suite, 'configs': configs, 'sources': sources, 'data': pools}


def suite_identity(suite, device='cpu'):
    return digest(identity_payload(suite, device))


def matrix(suite):
    return {'training': [{'arm': arm, 'method': d['method'], 'seed': seed} for arm, d in suite['arms'].items() for seed in suite['seeds']],
            'search': {task: len(suite['search_methods']) * 27 * len(suite['seeds']) * (4 if task == 'independent_request' else 1) for task in TASKS},
            'rule_solves': 27 * len(suite['rules']), 'search_budget_hours': 225,
            'validation_calls': 700, 'validation_trajectories': 666900,
            'evaluation_workers': suite['execution'].get('evaluation_workers', 20),
            'final_evaluation_unique_trajectories': 189540,
            'reused_frontier_sample1_trajectories': 8910}


def validate_training_streams(root, suite):
    """The same seed must see the same actual instance/preference sequence."""
    reference = {}
    for arm in suite['arms']:
        for seed in suite['seeds']:
            path = Path(root) / 'training' / arm / str(seed)
            summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
            if (not summary.get('budget_completed') or summary['episodes_started'] != 2000
                    or summary['epochs'] != 100 or summary['validation_count'] != 20):
                raise ValueError(f'incomplete formal training: {arm}/{seed}')
            episodes = json.loads((path/'episodes.json').read_text(encoding='utf-8'))
            if [e['episode'] for e in episodes] != list(range(2000)):
                raise ValueError('missing/duplicate training attempts')
            stream = digest([(e['record_sha256'], e['preference']) for e in episodes])
            if seed in reference and stream != reference[seed]:
                raise ValueError(f'cross-method training stream differs: {arm}/{seed}')
            reference[seed] = stream
    return reference
