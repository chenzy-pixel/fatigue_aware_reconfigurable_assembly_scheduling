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
SEARCH = ('mo_alns', 'nsga2_ls', 'moead')
LEARNING = ('ppo', 'dcan', 'rule_ddqn')

def study_identity(config):
    paths=[p for folder in ('agent','environment','data','configs') for p in Path(folder).rglob('*.py')]
    paths += list(Path('result').glob('*.py'))
    paths += [Path('scripts/comparison_suite.py'), Path('train.py'), Path('scripts/summarize_factorial27.py')]
    sources={str(p):digest(p.read_bytes().hex()) for p in paths}
    return digest({'source':sources,'algorithm':config['ppo'],'network':config['network'],'comparison':config.get('comparison',{}),'protocol': config['runtime_manifest'], 'generator': config['generator'], 'validation': [r['sha256'] for r in load_dataset_split(config, 'validation').manifest['files']]})

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
    p = argparse.ArgumentParser()
    p.add_argument('--config',default='configs/default.json')
    p.add_argument('--stage', choices=('preflight', 'train', 'evaluate', 'calibrate-budget'), default='preflight')
    p.add_argument('--method', choices=(*LEARNING, *SEARCH), default='ppo')
    p.add_argument('--remote-run', action='store_true')
    p.add_argument('--budget-manifest')
    p.add_argument('--checkpoint')
    p.add_argument('--steps', type=int)
    p.add_argument('--seconds', type=float)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output-dir', default='result/factorial27_study')
    args = p.parse_args()
    c = load_config(args.config)
    c['device'] = args.device
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
    atomic_json(out/'preflight.json',startup_preflight({'main':c},probe_network=True))
    budget = {}
    if args.stage != 'calibrate-budget':
        if not args.budget_manifest:
            p.error('a matching new-protocol budget manifest is required')
        budget = json.loads(Path(args.budget_manifest).read_text(encoding='utf-8'))
        if budget.get('identity') != identity or budget.get('status') != 'frozen':
            p.error('budget identity or status mismatch')
    if args.stage in ('train', 'calibrate-budget') and args.method in LEARNING:
        steps = args.steps if args.stage == 'calibrate-budget' else budget['training_steps']
        if not isinstance(steps, int) or steps <= 0:
            p.error('positive training steps required')
        train_request(c, args.method, args.seed, c['ppo']['learning_rate'], steps, load_dataset_split(c, 'calibration' if args.stage == 'calibrate-budget' else 'validation'), list(range(54 if args.stage == 'calibrate-budget' else 27)), [p.as_tuple() for p in formal_preferences(c, 'validation')], out / args.method, identity)
        from scripts.summarize_factorial27 import export_training
        export_training(out / args.method)
    else:
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
