"""Build, execute and audit the approved factorial-27 experiment matrix.

Formal work is serial and requires --remote-run. A successful preflight with
the same source/config/data identity is required before any formal stage.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import subprocess
import time
from contextlib import contextmanager, nullcontext

import torch

from configs import project_path
from configs.config import public_config
from configs.experiment_suite import (DEFAULT_MANIFEST, TASKS, load_suite, arm_config,
                                     preferences, suite_identity, matrix)
from configs.formal_preferences import formal_preferences
from data import load_dataset_split
from environment import AssemblySchedulingEnv
from agent.baselines.experiment_rules import evaluate_rule
from agent.comparators.common import atomic_json, digest
from agent.comparators.learning import build_agent, train_request, synchronize
from agent.comparators.experiment_evaluation import learning_unit, rule_unit, search_unit, development_pass
from agent.comparators.parallel_evaluation import evaluation_pool, reuse_first_sample


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def cached(path, job_id):
    if not path.exists():
        return None
    value = read_json(path)
    if value.get('job_id') != job_id:
        raise ValueError(f'artifact identity differs: {path}')
    if value.get('status') != 'complete':
        raise ValueError(f'incomplete artifact: {path}')
    if 'content_sha256' in value and value['content_sha256'] != digest({k:v for k,v in value.items() if k != 'content_sha256'}):
        raise ValueError(f'artifact content hash differs: {path}')
    return value


def save_unit(path, payload, *, identity, arm, seed, task, record, job_id):
    value = {'status': 'complete', 'job_id': job_id, 'identity': identity, 'arm': arm, 'seed': seed,
             'task': task, 'instance_id': record.instance.instance_id,
             'cell_id': record.metadata['cell_id'], 'scale': record.metadata['scale'], **payload}
    value['content_sha256'] = digest(value)
    atomic_json(path, value)
    return value


def hardware():
    import os
    return {'python': platform.python_version(), 'platform': platform.platform(), 'cpu': platform.processor(),
            'logical_cpus': os.cpu_count(), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'torch_threads': torch.get_num_threads(),
            'gpu': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def preflight(suite, identity, device):
    from data.distribution import validate_training_distribution, training_sampling_plan
    from data.dataset import protocol_hashes
    configs, probes = {}, {}
    for arm, definition in suite['arms'].items():
        c = arm_config(suite, arm, suite['seeds'][0], device)
        validate_training_distribution(c)
        e = AssemblySchedulingEnv(c)
        o = e.reset(load_dataset_split(c, 'validation')[0].instance)
        agent = build_agent(c, o, definition['method'])
        if definition['method'] == 'rule_ddqn':
            action = agent.act_rules([o], [e.get_action_mask()])[0][0]
            if e.get_action_mask()[action]:
                raise ValueError('DDQN probe selected an illegal action')
        else:
            with torch.no_grad():
                logits, values = agent.network.forward_batch([o], [e.get_action_mask()], device=device)
            if not torch.isfinite(values).all() or not torch.isfinite(logits[0, ~torch.as_tensor(e.get_action_mask(), device=device)]).all():
                raise ValueError('non-finite network probe')
        network = agent.online if definition['method'] == 'rule_ddqn' else agent.network
        probes[arm] = {'finite': True, 'parameter_count': sum(p.numel() for p in network.parameters()),
                       'trainable_parameter_count': sum(p.numel() for p in network.parameters() if p.requires_grad)}
        configs[arm] = public_config(c)
    c = arm_config(suite, 'full', suite['seeds'][0], device)
    pools = {s: load_dataset_split(c, s) for s in ('calibration', 'validation', 'test')}
    if [len(pools[s]) for s in pools] != [54, 27, 27]:
        raise ValueError('fixed dataset counts differ')
    instance_sets = [{r.instance.instance_id for r in d} for d in pools.values()]
    if any(instance_sets[i] & instance_sets[j] for i in range(3) for j in range(i)):
        raise ValueError('dataset splits overlap')
    if len(preferences(c, 'frontier')) != 66 or len(formal_preferences(c, 'validation')) != 13:
        raise ValueError('formal preference contract differs')
    git = subprocess.run(['git','rev-parse','HEAD'], capture_output=True, text=True, check=False)
    return {'status': 'passed', 'identity': identity, 'hardware': hardware(), 'git_commit': git.stdout.strip(),
            'suite': suite, 'configs': configs, 'probes': probes, 'matrix': matrix(suite),
            'data_manifests': {s:d.manifest for s,d in pools.items()}, 'data_protocol': protocol_hashes(c),
            'training_cell_sequence': [x.cell_id for x in training_sampling_plan(c, 2000)],
            'search_budget_status': 'requires_development_check',
            'training_sufficiency': 'not_established', 'search_sufficiency': 'not_established'}


def check_preflight(root, identity):
    path = root/'preflight.json'
    if not path.exists():
        raise ValueError('run the complete same-identity preflight first')
    value = read_json(path)
    if value.get('status') != 'passed' or value.get('identity') != identity:
        raise ValueError('experiment identity differs; use the original Git revision or a new output directory')


@contextmanager
def execution_lock(root):
    """OS locks release on crash; the marker file can safely remain in place."""
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.execution.lock').open('a+b') as handle:
        handle.seek(0,2)
        if handle.tell()==0: handle.write(b'0');handle.flush()
        handle.seek(0)
        if platform.system() == 'Windows':
            import msvcrt
            try: msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as error: raise RuntimeError('another suite job holds the output-directory execution lock') from error
            try: yield
            finally:
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            try: fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as error: raise RuntimeError('another suite job holds the output-directory execution lock') from error
            try: yield
            finally: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def snapshot_sources(root, suite, identity, device):
    import zipfile
    from configs.config import PROJECT_ROOT
    from configs.experiment_suite import identity_payload
    from result.provenance import source_files
    payload = identity_payload(suite,device)
    if digest(payload) != identity:
        raise ValueError('source files changed during preflight')
    archive = root/'source.zip'
    with zipfile.ZipFile(archive.with_suffix('.zip.tmp'),'w',zipfile.ZIP_DEFLATED) as zipped:
        for path in source_files(PROJECT_ROOT):
            zipped.write(path,path.relative_to(PROJECT_ROOT).as_posix())
        for path in sorted((PROJECT_ROOT/'scripts').glob('*.ps*')):
            zipped.write(path,path.relative_to(PROJECT_ROOT).as_posix())
        for name in ('requirements.txt','pytest.ini','README.md','configs/default.json',
                     'configs/comparators/suite.json','configs/experiment_suite_v1.json',
                     'configs/factorial27_normalization.json'):
            zipped.write(PROJECT_ROOT/name,name)
        zipped.writestr('experiment_identity.json',json.dumps(payload,ensure_ascii=False,indent=2))
    archive.with_suffix('.zip.tmp').replace(archive)
    return payload['sources']


def load_agent(root, suite, arm, seed, identity, device, *, record=None):
    from agent.comparators.learning import check_learning_checkpoint
    c = arm_config(suite, arm, seed, device)
    method = suite['arms'][arm]['method']
    path = root/'training'/arm/str(seed)/'best_checkpoint.pt'
    summary = read_json(path.parent/'summary.json')
    if not summary.get('budget_completed') or summary['episodes_started'] != 2000 or summary['epochs'] != 100 or summary['validation_count'] != 20:
        raise ValueError('formal evaluation requires completed training')
    check_learning_checkpoint(path, c, method, seed, identity)
    began = time.perf_counter()
    record = record if record is not None else load_dataset_split(c, 'validation')[0]
    env = AssemblySchedulingEnv(c)
    obs = env.reset(record.instance)
    agent = build_agent(c, obs, method)
    agent.load(path)
    synchronize(c)
    load_seconds = time.perf_counter()-began
    began = time.perf_counter()
    if method == 'rule_ddqn':
        agent.act_rules([obs], [env.get_action_mask()])
    else:
        agent.act(obs, env.get_action_mask(), deterministic=True)
    synchronize(c)
    return c, agent, {'checkpoint': str(path), 'load_seconds': load_seconds,
                      'warmup_seconds': time.perf_counter()-began}


def train(suite, root, identity, device, arms, seeds):
    dataset = load_dataset_split(arm_config(suite, 'full', 11, device), 'validation')
    for arm in arms:
        for seed in seeds:
            c = arm_config(suite, arm, seed, device)
            print(f'train {arm} seed={seed}', flush=True)
            train_request(c, suite['arms'][arm]['method'], seed, c['ppo']['learning_rate'],
                          dataset, list(range(27)),
                          [p.as_tuple() for p in formal_preferences(c, 'validation')], root/'training'/arm/str(seed), identity,
                          episodes=2000, workers=suite['execution']['training_workers'])


def evaluate(suite, root, identity, device, arms, seeds, tasks):
    records = load_dataset_split(arm_config(suite, 'full', 11, device), 'test')
    for arm in arms:
        for seed in seeds:
            modes = [(arm, None)] + ([('full_greedy', 'greedy'), ('full_sample1', 'sampled_single')] if arm == 'full' else [])
            pending = set()
            for task in tasks:
                for record in records:
                    saved_modes = {}
                    for label, mode in modes:
                        path = root/'evaluation'/task/label/str(seed)/f'{record.instance.instance_id}.json'
                        job = digest([identity, task, label, seed, record.instance.instance_id, mode])
                        saved_modes[label] = cached(path, job)
                        if saved_modes[label] is None:
                            pending.add((task, label, record.instance.instance_id))
                    if task == 'frontier' and arm == 'full' and saved_modes['full_sample1'] is not None:
                        parent, derived = saved_modes['full'], saved_modes['full_sample1']
                        if (parent is None or derived.get('source_job_id') != parent['job_id']
                                or derived.get('source_content_sha256') != parent.get('content_sha256')):
                            pending.add((task, 'full_sample1', record.instance.instance_id))
            if not pending:
                continue
            if all(task == 'frontier' and label == 'full_sample1' for task, label, _ in pending):
                c = arm_config(suite, arm, seed, device)
                for record in records:
                    iid = record.instance.instance_id
                    if ('frontier', 'full_sample1', iid) not in pending:
                        continue
                    parent = cached(root/'evaluation'/'frontier'/'full'/str(seed)/f'{iid}.json',
                                    digest([identity, 'frontier', 'full', seed, iid, None]))
                    payload = reuse_first_sample(c, parent)
                    payload['model_timing'] = parent['model_timing']
                    save_unit(root/'evaluation'/'frontier'/'full_sample1'/str(seed)/f'{iid}.json', payload,
                              identity=identity, arm='full_sample1', seed=seed, task='frontier', record=record,
                              job_id=digest([identity, 'frontier', 'full_sample1', seed, iid, 'sampled_single']))
                continue
            c, agent, timing = load_agent(root, suite, arm, seed, identity, device, record=records[0])
            for task in tasks:
                if not any(t == task for t, _, _ in pending):
                    continue
                started = time.perf_counter()
                needs_pool = task == 'frontier' and any(t == task and label != 'full_sample1' for t, label, _ in pending)
                pool_context = evaluation_pool(c, records[0].instance) if needs_pool else nullcontext(None)
                with pool_context as pool:
                    if pool is not None:
                        timing['evaluation_pool_startup_seconds'] = time.perf_counter()-started
                    for record in records:
                        full_unit = None
                        for label, mode in modes:
                            path = root/'evaluation'/task/label/str(seed)/f'{record.instance.instance_id}.json'
                            job = digest([identity, task, label, seed, record.instance.instance_id, mode])
                            if (task, label, record.instance.instance_id) not in pending:
                                if label == 'full' and (task, 'full_sample1', record.instance.instance_id) in pending and task == 'frontier':
                                    full_unit = cached(path, job)
                                continue
                            if task == 'frontier' and label == 'full_sample1':
                                payload = reuse_first_sample(c, full_unit)
                            else:
                                payload = learning_unit(c, agent, record, suite['arms'][arm]['method'], preferences(c, task), seed,
                                                        mode=mode, runner=pool, parallel=task == 'frontier')
                            payload['model_timing'] = timing
                            saved_value = save_unit(path, payload, identity=identity, arm=label, seed=seed, task=task, record=record, job_id=job)
                            if label == 'full':
                                full_unit = saved_value
                            print(f'evaluate {task} {label}/{seed}/{record.instance.instance_id}', flush=True)


def rules(suite, root, identity, device):
    c = arm_config(suite, 'full', 11, device)
    records = load_dataset_split(c, 'test')
    for method in suite['rules']:
        for record in records:
            raw_path = root/'rules'/method/f'{record.instance.instance_id}.json'
            job = digest([identity, 'rule', method, record.instance.instance_id])
            raw = cached(raw_path, job)
            if raw is None:
                raw = {'status': 'complete', 'identity': identity, 'job_id': job,
                       'raw': evaluate_rule(c, record.instance, method)}
                atomic_json(raw_path, raw)
            for task in TASKS:
                path = root/'evaluation'/task/method/'deterministic'/f'{record.instance.instance_id}.json'
                unit_job = digest([job, task])
                if cached(path, unit_job) is not None:
                    continue
                save_unit(path, rule_unit(c, raw['raw'], preferences(c, task)), identity=identity, arm=method,
                          seed=None, task=task, record=record, job_id=unit_job)
            print(f'rule {method}/{record.instance.instance_id}', flush=True)


def search(suite, root, identity, device, seeds, tasks, *, development=False):
    c = arm_config(suite, 'full', 11, device)
    if not development:
        check_frozen_budget(suite,root,identity)
    records = list(load_dataset_split(c, 'calibration' if development else 'test'))
    if development:
        records = [next(r for r in records if r.metadata['cell_id'] == cell) for cell in suite['search']['development_cells']]
        seeds, tasks = [suite['search']['development_seed']], TASKS
    for task in tasks:
        for method in suite['search_methods']:
            for seed in seeds:
                for record in records:
                    # Persist independent preferences separately: interrupted work
                    # never forces completed requests to be paid a second time.
                    prefs = preferences(c, task)
                    groups = [prefs] if task == 'frontier' else [(p,) for p in prefs]
                    for index, group in enumerate(groups):
                        suffix = '' if task == 'frontier' else f'_p{index}'
                        path = root/('development' if development else 'evaluation')/task/method/str(seed)/f'{record.instance.instance_id}{suffix}.json'
                        job = digest([identity, development, task, method, seed, record.instance.instance_id, group])
                        if cached(path, job) is not None:
                            continue
                        payload = search_unit(c, record, method, group, seed, task,
                                              seconds=suite['search']['seconds'], checkpoints=suite['search']['checkpoints'])
                        payload['development_pass'] = development_pass(payload, suite['search']) if development else None
                        save_unit(path, payload, identity=identity, arm=method, seed=seed, task=task, record=record, job_id=job)
                        print(f'search {task} {method}/{seed}/{record.instance.instance_id}{suffix}', flush=True)


def freeze_budget(suite, root, identity):
    paths = sorted((root/'development').rglob('*.json'))
    expected = {(task, method, cell, i) for task in TASKS for method in suite['search_methods']
                for cell in suite['search']['development_cells'] for i in range(4 if task == 'independent_request' else 1)}
    seen, evidence = set(), []
    for path in paths:
        value = read_json(path)
        prefs = value['deliveries']
        index = list(preferences(arm_config(suite, 'full', 11), value['task'])).index(tuple(prefs[0]['preference'])) if value['task'] == 'independent_request' else 0
        key = (value['task'], value['arm'], value['cell_id'], index)
        if (key in seen or key not in expected or value['identity'] != identity or value.get('status') != 'complete'
                or value['seed'] != suite['search']['development_seed']):
            raise ValueError('development coverage/identity differs')
        if not development_pass(value, suite['search']):
            raise ValueError(f'development budget applicability failed: {path}; inspect initialization/iterations/diversity')
        seen.add(key)
        evidence.append({'path': str(path.relative_to(root)), 'digest': digest(value)})
    if seen != expected:
        raise ValueError('development search matrix is incomplete')
    atomic_json(root/'search_budget.json', {'status': 'frozen', 'identity': identity,
                'seconds': suite['search']['seconds'], 'checkpoints': suite['search']['checkpoints'],
                'evidence': evidence, 'budget_sufficiency': 'not_established'})


def check_frozen_budget(suite, root, identity):
    frozen = read_json(root/'search_budget.json')
    if (frozen.get('identity') != identity or frozen.get('status') != 'frozen'
            or frozen.get('seconds') != suite['search']['seconds']
            or frozen.get('checkpoints') != suite['search']['checkpoints'] or len(frozen.get('evidence',[])) != 30):
        raise ValueError('a matching frozen search budget is required')
    paths = set()
    for row in frozen['evidence']:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or path in paths:
            raise ValueError('invalid or duplicate development evidence path')
        paths.add(path)
        if not path.is_file() or digest(read_json(path)) != row['digest']:
            raise ValueError('development budget evidence differs')
    return frozen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('preflight','train','evaluate','rules','develop-search','freeze-budget','search','report','smoke'), default='preflight')
    p.add_argument('--manifest', default=DEFAULT_MANIFEST)
    p.add_argument('--output-dir', default='result/experiments')
    p.add_argument('--device', default='cuda')
    p.add_argument('--arms', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--task', choices=(*TASKS, 'both'), default='both')
    p.add_argument('--training-workers', type=int)
    p.add_argument('--remote-run', action='store_true')
    args = p.parse_args()
    suite = load_suite(args.manifest)
    if args.training_workers is not None:
        if args.training_workers < 1:
            p.error('positive training-workers required')
        suite['execution']['training_workers'] = args.training_workers
    torch.set_num_threads(suite['execution']['torch_threads'])
    root = project_path(args.output_dir)
    if args.stage == 'smoke':
        from scripts.smoke import run_smoke
        run_smoke(suite, Path('.cache/smoke'))
        return
    arms, seeds = args.arms or list(suite['arms']), args.seeds or suite['seeds']
    if len(set(arms)) != len(arms) or any(a not in suite['arms'] for a in arms):
        p.error('unknown or duplicate arm')
    if len(set(seeds)) != len(seeds) or any(s not in suite['seeds'] for s in seeds):
        p.error('unknown or duplicate formal seed')
    identity = suite_identity(suite, args.device)
    tasks = TASKS if args.task == 'both' else [args.task]
    if args.stage == 'preflight':
        if (root/'preflight.json').exists() and read_json(root/'preflight.json').get('identity') != identity:
            p.error('experiment identity differs; use the original Git revision or a new output directory')
        result = preflight(suite, identity, args.device)
        root.mkdir(parents=True,exist_ok=True)
        result['source_hashes'] = snapshot_sources(root,suite,identity,args.device)
        atomic_json(root/'preflight.json', result)
        print(json.dumps({'identity':identity, 'matrix':result['matrix'], 'probes':result['probes']}, indent=2))
        return
    check_preflight(root, identity)
    if args.stage in ('train','evaluate','rules','develop-search','search') and not args.remote_run:
        p.error('formal execution requires --remote-run')
    with execution_lock(root):
        if args.stage == 'train': train(suite, root, identity, args.device, arms, seeds)
        elif args.stage == 'evaluate': evaluate(suite, root, identity, args.device, arms, seeds, tasks)
        elif args.stage == 'rules': rules(suite, root, identity, args.device)
        elif args.stage == 'develop-search': search(suite, root, identity, args.device, seeds, tasks, development=True)
        elif args.stage == 'freeze-budget': freeze_budget(suite, root, identity)
        elif args.stage == 'search': search(suite, root, identity, args.device, seeds, tasks)
        elif args.stage == 'report':
            from analysis.experiment_suite import report
            report(root, suite, identity, args.device)


if __name__ == '__main__':
    main()
