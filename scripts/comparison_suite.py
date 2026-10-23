"""Schema-13 comparison requests with explicit experimental budgets."""
import argparse, json, time
from pathlib import Path
from configs import load_config
from configs.formal_preferences import formal_preferences
from data import load_dataset_split
from agent.comparators.common import atomic_json, digest, method_config, FOUR
from agent.mo_alns import MOALNSSolver, SearchBudget
from agent.comparators.evolution import EvolutionSolver
from environment import PreferenceVector
from agent.comparators.learning import train_request, build_agent, evaluate_request
from agent.comparators.training_budget import training_budget
SEARCH = ('mo_alns', 'nsga2_ls', 'moead')
LEARNING = ('ppo', 'a2c', 'dcan', 'rule_ddqn')

def study_identity(config):
    paths=[p for folder in ('agent','environment','data','configs') for p in Path(folder).rglob('*.py')]
    paths += list(Path('result').glob('*.py'))
    paths += [Path('scripts/comparison_suite.py'), Path('train.py'), Path('scripts/summarize_factorial27.py')]
    sources={str(p):digest(p.read_bytes().hex()) for p in paths}
    return digest({'source':sources,'algorithm':config['ppo'],'network':config['network'],'comparison':config.get('comparison',{}),
                   'training_budget': training_budget(config), 'formal_evaluation': config['training']['formal_evaluation'],
                   'protocol': config['runtime_manifest'], 'generator': config['generator'], 'validation': [r['sha256'] for r in load_dataset_split(config, 'validation').manifest['files']]})


def check_learning_checkpoint(path, config, method, seed, identity):
    """Check request identity before an agent's weights can be changed."""
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)
    meta = saved.get('metadata', {})
    if (meta.get('comparison_identity') != identity or meta.get('algorithm') != method
            or meta.get('seed') != seed or meta.get('training_budget') != training_budget(config)):
        raise ValueError('checkpoint training protocol/identity mismatch')
    return meta

def search_job(job):
    config, record, pref, method, seed, seconds, evaluations, output, identity, *extra = job
    from dataclasses import asdict
    from data.dataset import template_sha256
    job_id=digest([identity,template_sha256(record.instance),method,seed,pref,seconds,evaluations])
    if output and Path(output).is_file():
        previous=json.loads(Path(output).read_text(encoding='utf-8'))
        if previous.get('job_id')!=job_id: raise ValueError('completed search identity mismatch')
        return previous['summary']
    config = method_config(config, method, seed)
    solver = MOALNSSolver(config, algorithm_seed=seed) if method == 'mo_alns' else EvolutionSolver(config, method, algorithm_seed=seed)
    result = solver.solve(record.instance, PreferenceVector(*pref), budget=SearchBudget(max_evaluations=evaluations, seconds=seconds))
    best = result.selected
    row = {'method': method, 'instance_id': record.instance.instance_id, 'cell_id': record.metadata['cell_id'], 'scale': record.metadata['scale'], 'seed': seed, 'preference': pref, 'success': best is not None, 'quality': best.tchebycheff if best else 1.0, 'objectives': best.objectives if best else None, 'search_seconds': result.search_time_seconds, 'environment_evaluations': result.environment_evaluations, 'initial_quality': result.initial_best_tchebycheff, 'identity': identity}
    if output:
        atomic_json(output, {'job_id':job_id,'summary': row,'selected':asdict(best) if best else None})
    return row

def main():
    import sys
    if '--suite' in sys.argv:
        sys.argv.remove('--suite')
        from scripts.experiment_suite import main as run_suite
        return run_suite()
    p = argparse.ArgumentParser()
    p.add_argument('--config',default='configs/default.json')
    p.add_argument('--stage', choices=('preflight', 'train', 'evaluate', 'calibrate-budget'), default='preflight')
    p.add_argument('--method', choices=(*LEARNING, *SEARCH), default='ppo')
    p.add_argument('--remote-run', action='store_true')
    p.add_argument('--budget-manifest')
    p.add_argument('--checkpoint')
    allowance = p.add_mutually_exclusive_group()
    allowance.add_argument('--steps', type=int)
    allowance.add_argument('--episodes', type=int)
    p.add_argument('--seconds', type=float)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--device', default='cuda')
    p.add_argument('--training-workers', type=int, default=20)
    p.add_argument('--output-dir', default='result/factorial27_episode_study')
    args = p.parse_args()
    c = load_config(args.config)
    c['device'] = args.device
    if args.training_workers < 1:
        p.error('--training-workers must be positive')
    if args.stage != 'calibrate-budget' and (args.steps is not None or args.episodes is not None):
        p.error('--steps/--episodes are for calibrate-budget; formal training uses the config')
    if args.method in LEARNING and args.budget_manifest:
        p.error('learning methods use the configured training protocol; --budget-manifest is for search')
    if args.method in SEARCH and (args.steps is not None or args.episodes is not None):
        p.error('search methods use --seconds for calibration')
    if args.method in LEARNING and args.seconds is not None:
        p.error('--seconds is for search calibration')
    if args.method in LEARNING and args.stage == 'evaluate':
        if not args.checkpoint:
            p.error('--checkpoint is required')
    if args.stage == 'calibrate-budget' and args.method in LEARNING and args.steps is None and args.episodes is None:
        p.error('calibrate-budget requires --steps or --episodes for learning methods')
    identity = study_identity(c)
    out = Path(args.output_dir)
    if args.stage == 'preflight':
        from scripts.run_experiments import startup_preflight
        print(json.dumps(startup_preflight({'main': c}, probe_network=True), indent=2))
        return
    if args.stage=='train' and args.method not in LEARNING:
        p.error('train stage is only for learning methods')
    if not args.remote_run:
        p.error('experiment stages require --remote-run')
    from scripts.run_experiments import startup_preflight
    preflight_path = out/'preflight.json'
    if preflight_path.exists():
        previous = json.loads(preflight_path.read_text(encoding='utf-8'))
        if previous.get('main', {}).get('study_identity') != identity:
            p.error('output directory study identity mismatch; use a new directory')
    preflight = startup_preflight({'main':c},probe_network=True)
    budget = {}
    if args.method in SEARCH and args.stage != 'calibrate-budget':
        if not args.budget_manifest:
            p.error('a matching new-protocol budget manifest is required')
        budget = json.loads(Path(args.budget_manifest).read_text(encoding='utf-8'))
        if budget.get('identity') != identity or budget.get('status') != 'frozen':
            p.error('budget identity or status mismatch')
    if args.stage in ('train', 'calibrate-budget') and args.method in LEARNING:
        steps = args.steps if args.stage == 'calibrate-budget' else None
        episodes = args.episodes if args.stage == 'calibrate-budget' else c['training']['episodes']
        try:
            training_budget(c, steps=steps, episodes=episodes)
        except ValueError as exc:
            p.error(str(exc))
        atomic_json(preflight_path, preflight)
        train_request(c, args.method, args.seed, c['ppo']['learning_rate'], steps, load_dataset_split(c, 'calibration' if args.stage == 'calibrate-budget' else 'validation'), list(range(54 if args.stage == 'calibrate-budget' else 27)), [p.as_tuple() for p in formal_preferences(c, 'validation')], out / args.method, identity,
                      episodes=episodes, workers=args.training_workers)
        from scripts.summarize_factorial27 import export_training
        export_training(out / args.method)
    else:
        if args.method in LEARNING:
            check_learning_checkpoint(args.checkpoint, c, args.method, args.seed, identity)
        atomic_json(preflight_path, preflight)
        split = 'calibration' if args.stage == 'calibrate-budget' else 'test'
        dataset = load_dataset_split(c, split)
        if args.method in SEARCH:
            seconds = args.seconds if args.stage == 'calibrate-budget' else budget['search_seconds'][args.method]
            if seconds is None or seconds <= 0:
                p.error('positive search seconds required')
            prefs = FOUR if args.stage == 'calibrate-budget' else [p.as_tuple() for p in formal_preferences(c, 'final_test')]
            for i, r in enumerate(dataset):
                for j, pref in enumerate(prefs):
                    search_job((c, r, pref, args.method, args.seed, seconds, None, out / f'{args.method}_{i}_{j}.json', identity))
        else:
            if not args.checkpoint:
                p.error('--checkpoint is required')
            from environment import AssemblySchedulingEnv
            c = method_config(c, args.method, args.seed)
            agent = build_agent(c, AssemblySchedulingEnv(c).reset(dataset[0].instance), args.method)
            agent.load(args.checkpoint)
            for i, r in enumerate(dataset):
                for j, pref in enumerate(formal_preferences(c, 'final_test')):
                    result = evaluate_request(c, agent, r.instance, pref.as_tuple(), args.method, seed=args.seed)
                    result['summary'].update(seed=args.seed,cell_id=r.metadata['cell_id'],scale=r.metadata['scale'],identity=identity)
                    atomic_json(out / f'{args.method}_{i}_{j}.json', result)
    if args.stage=='evaluate' or (args.stage=='calibrate-budget' and args.method in SEARCH):
        from scripts.summarize_factorial27 import summarize
        summarize(out)

if __name__ == '__main__':
    main()
