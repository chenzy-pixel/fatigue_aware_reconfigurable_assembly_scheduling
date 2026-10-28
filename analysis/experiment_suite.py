"""Strict matrix audit, instance-paired statistics and reproducible figures."""
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
from scipy.stats import wilcoxon

from configs.experiment_suite import TASKS, arm_config, preferences, validate_training_streams
from agent.comparators.common import atomic_json, digest
from agent.comparators.experiment_evaluation import quality, frontier_metrics
from result.io import write_csv

GROUPS = {'ablation': ('no_graph','no_direct','no_residual'),
          'rules': ('eft_lf','srw_lf','mc_mc','mwl_mwl','existing_heuristic'),
          'learning': ('a2c','rule_ddqn','dcan'), 'evolution': ('nsga2_ls','moead'),
          'decoding': ('full_greedy','full_sample1')}


def stratified_interval(values, scales, *, samples=10000, seed=20261023):
    if not values:
        return {'mean': None, 'ci_low': None, 'ci_high': None, 'instances': 0}
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('non-finite paired statistic')
    rng = np.random.default_rng(seed)
    draws = np.zeros(samples)
    for group in sorted(set(scales)):
        indices = np.flatnonzero(np.asarray(scales) == group)
        draws += x[indices][rng.integers(0, len(indices), (samples, len(indices)))].sum(1) / len(x)
    lo, hi = np.quantile(draws, [.025, .975])
    return {'mean': float(x.mean()), 'ci_low': float(lo), 'ci_high': float(hi), 'instances': len(x)}


def holm(pvalues):
    ordered = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    result, previous = [0.] * len(pvalues), 0.
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1., pvalues[index] * (len(pvalues)-rank)))
        result[index] = previous
    return result


def audit_evaluations(units, suite, config, records, identity, *, compact_units=None, execution_identities=None):
    """Validate complete coverage at delivery granularity, before aggregation."""
    methods = list(suite['arms']) + suite['rules'] + suite['search_methods'] + ['full_greedy','full_sample1']
    cases = {r.instance.instance_id:r for r in records}
    expected = {(task, method, seed, iid, tuple(pref))
                for task in TASKS for method in methods
                for seed in ([None] if method in suite['rules'] else suite['seeds'])
                for iid in cases for pref in preferences(config, task)}
    seen, rows = set(), []
    full_sources, reused_sources = {}, {}
    for unit in units:
        method, task, seed, iid = (unit[k] for k in ('arm','task','seed','instance_id'))
        if unit['identity'] != identity or unit.get('status') != 'complete' or iid not in cases:
            raise ValueError('evaluation identity/status/instance differs')
        expected_execution = (execution_identities or {}).get(method)
        if expected_execution and unit.get('execution_identity') != expected_execution:
            raise ValueError('evaluation execution identity differs')
        if 'content_sha256' in unit and unit['content_sha256'] != digest({k:v for k,v in unit.items() if k != 'content_sha256'}):
            raise ValueError('evaluation content hash differs')
        if task == 'frontier' and method == 'full':
            full_sources[seed, iid] = (unit.get('job_id'), unit.get('content_sha256'),
                                      digest([r['raw'][0] for r in unit['requests']]))
        if unit.get('cost_reference') == 'full':
            if method != 'full_sample1' or task != 'frontier' or any(len(r['raw']) != 1 for r in unit['requests']):
                raise ValueError('invalid reused sample1 artifact')
            reused_sources[seed, iid] = (unit.get('source_job_id'), unit.get('source_content_sha256'),
                                        digest([r['raw'][0] for r in unit['requests']]))
        record = cases[iid]
        if unit['cell_id'] != record.metadata['cell_id'] or unit['scale'] != record.metadata['scale']:
            raise ValueError('evaluation instance metadata differs')
        if not math.isfinite(unit['seconds']) or unit['seconds'] < 0:
            raise ValueError('invalid delivery timing')
        for r in unit['deliveries']:
            key = (task, method, seed, iid, tuple(r['preference']))
            if key not in expected or key in seen:
                raise ValueError('duplicate or unexpected delivery')
            seen.add(key)
            if r['success']:
                q = quality(config, r['objectives'], r['preference'])
                if not math.isclose(q, r['quality'], rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError('public quality and objectives disagree')
                if not all(math.isfinite(v) and v >= 0 for v in r['objectives']):
                    raise ValueError('invalid objective vector')
            elif r['quality'] != 1.:
                raise ValueError('failed requests require the public failure score')
            rows.append({'task': task, 'arm': method, 'seed': seed, 'instance_id': iid,
                         'cell_id': unit['cell_id'], 'scale': unit['scale'],
                         **{k:record.metadata[k] for k in ('N','K','M')},
                         'preference': json.dumps(r['preference']), 'success': r['success'], 'quality': r['quality'],
                         **{name:r['objectives'][i] if r['success'] else None for i,name in enumerate(('flow','cost','workload'))},
                         'seconds':r['search_seconds'], 'timing_basis':r.get('timing_basis','measured_request')})
        # Recompute front metrics from the paid evidence, not saved aggregates.
        if method in suite['search_methods']:
            candidates = [{'success':True,'objectives':v['objectives']} for s in unit['searches'] for v in s['archive']]
        elif method in suite['rules']:
            candidates = [unit['raw_rule']]
        else:
            candidates = [v for request in unit['requests'] for v in request['raw']]
        metrics = frontier_metrics(config, candidates, unit['deliveries'])
        for name in ('paid_hv','delivery_hv','coverage'):
            if not math.isclose(metrics[name], unit[name], abs_tol=1e-10, rel_tol=1e-8):
                raise ValueError(f'saved {name} disagrees with paid candidates')
        if compact_units is not None:
            # Raw trajectories remain on disk. The whole formal suite contains
            # hundreds of thousands of paid rollouts and cannot live in RAM.
            compact = {k:v for k,v in unit.items() if k not in ('requests','searches','raw_rule')}
            compact['requests'] = [{'raw':[{'objectives':v['objectives'],'success':v['success']} for v in r['raw']]}
                                   for r in unit.get('requests',[])]
            compact['searches'] = [{'archive':[{'objectives':v['objectives']} for v in s['archive']],
                                    'snapshots':s['snapshots'],'preference':s['preference']}
                                   for s in unit.get('searches',[])]
            compact_units.append(compact)
    if seen != expected:
        missing = len(expected-seen)
        raise ValueError(f'incomplete evaluation matrix: {missing} missing deliveries')
    for key, source in reused_sources.items():
        if not all(source[:2]) or source != full_sources.get(key):
            raise ValueError('reused sample1 parent identity or first trajectories differ')
    return rows


def instance_tables(rows, units):
    groups, costs = defaultdict(list), defaultdict(list)
    for r in rows: groups[r['task'],r['arm'],r['instance_id']].append(r)
    for u in units: costs[u['task'],u['arm'],u['instance_id']].append(u)
    output = []
    for key, group in sorted(groups.items()):
        paid = costs[key]
        # Independent search units are per-preference files; sum their time by
        # seed before averaging seed replicates. Frontier units are per instance.
        per_seed = defaultdict(float)
        for u in paid: per_seed[u['seed']] += u['seconds']
        result = {'task':key[0], 'arm':key[1], 'instance_id':key[2],
                  **{k:group[0][k] for k in ('cell_id','scale','N','K','M')},
                  'quality': fmean(r['quality'] for r in group),
                  'success_rate': fmean(float(r['success']) for r in group),
                  'request_seconds': fmean(r['seconds'] for r in group),
                  'timing_basis': '|'.join(sorted({r.get('timing_basis', 'measured_request') for r in group})),
                  'total_seconds': fmean(per_seed.values()),
                  'paid_hv': fmean(u['paid_hv'] for u in paid) if key[0] == 'frontier' else None,
                  'delivery_hv': fmean(u['delivery_hv'] for u in paid) if key[0] == 'frontier' else None}
        good = [r for r in group if r['success']]
        result.update({name:fmean(r[name] for r in good) if good else None for name in ('flow','cost','workload')})
        output.append(result)
    return output


def paired_statistics(instances, suite):
    rows = []
    for family, methods in GROUPS.items():
        for task, metric in (('independent_request','quality'), ('frontier','paid_hv'), ('frontier','quality')):
            reference = {r['instance_id']:r for r in instances if r['arm'] == 'full' and r['task'] == task}
            family_rows = []
            for method in methods:
                arm = {r['instance_id']:r for r in instances if r['arm'] == method and r['task'] == task}
                ids = sorted(reference)
                delta = [arm[i][metric]-reference[i][metric] for i in ids]
                stat = stratified_interval(delta, [reference[i]['scale'] for i in ids],
                                           samples=suite['statistics']['bootstrap_samples'], seed=suite['statistics']['bootstrap_seed'])
                p = float(wilcoxon(delta, zero_method='wilcox', alternative='two-sided', method='auto').pvalue) if any(d != 0 for d in delta) else 1.
                family_rows.append({'family':family, 'task':task, 'metric':metric, 'arm':method,
                                    'reference':'full', 'difference':'arm_minus_full', **stat, 'p_value':p})
            for row, adjusted in zip(family_rows, holm([r['p_value'] for r in family_rows])):
                row['p_holm'] = adjusted
            rows.extend(family_rows)
    return rows


def common_success_statistics(rows, suite):
    by_key = defaultdict(list)
    for r in rows: by_key[r['task'],r['arm'],r['instance_id'],r['preference']].append(r)
    result = []
    for task in TASKS:
        for methods in GROUPS.values():
            for method in methods:
                cells = defaultdict(list)
                for (t,arm,iid,pref), reference in by_key.items():
                    if t != task or arm != 'full':
                        continue
                    first = [r for r in reference if r['success']]
                    other = [r for r in by_key.get((task,method,iid,pref),[]) if r['success']]
                    if first and other:
                        cells[iid].append((first, other))
                for metric in ('quality','flow','cost','workload'):
                    ids = sorted(cells)
                    differences = [fmean(fmean(r[metric] for r in b)-fmean(r[metric] for r in a) for a,b in cells[i]) for i in ids]
                    stat = stratified_interval(differences, [cells[i][0][0][0]['scale'] for i in ids],
                                               samples=suite['statistics']['bootstrap_samples'], seed=suite['statistics']['bootstrap_seed'])
                    result.append({'task':task,'arm':method,'metric':metric,'reference':'full', **stat,
                                   'paired_preferences':sum(len(v) for v in cells.values()),
                                   'conditioning':'both methods have a successful replicate for this instance/preference'})
    return result


def descriptive_tables(instances, rows):
    groups, seed_groups = defaultdict(list), defaultdict(list)
    for r in instances:
        for factor in ('all','scale','K','M'):
            groups[r['task'],r['arm'],factor,str(r[factor]) if factor != 'all' else 'all'].append(r)
    for r in rows: seed_groups[r['task'],r['arm'],r['seed']].append(r)
    table = []
    for (task,arm,factor,level), group in sorted(groups.items()):
        out = {'task':task,'arm':arm,'factor':factor,'level':level,'instances':len(group),
               'timing_basis': '|'.join(sorted({r.get('timing_basis', 'measured_request') for r in group}))}
        for metric in ('quality','success_rate','request_seconds','total_seconds','paid_hv','delivery_hv','flow','cost','workload'):
            values = [r[metric] for r in group if r[metric] is not None]
            out[metric] = fmean(values) if values else None
        table.append(out)
    per_seed = [{'task':t,'arm':a,'seed':s,'quality':fmean(r['quality'] for r in g),
                 'success_rate':fmean(float(r['success']) for r in g)} for (t,a,s),g in seed_groups.items()]
    return table, per_seed


def figures(root, suite, units, instances):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analysis.pareto_analysis import normalize_objectives, nondominated_indices
    from configs.formal_preferences import objective_scales
    out = root/'report'
    c = arm_config(suite, 'full', 11)
    scales = objective_scales(c)
    styles = {'full':'#0072B2','a2c':'#D55E00','rule_ddqn':'#009E73','dcan':'#CC79A7','nsga2_ls':'#E69F00','moead':'#56B4E9'}
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    def save(fig, name):
        fig.tight_layout()
        for ext in ('svg','pdf','png'): fig.savefig(out/f'{name}.{ext}', dpi=300)
        plt.close(fig)
    curve_rows = []
    fig, ax = plt.subplots(figsize=(7,4))
    for arm in suite['arms']:
        logs = [json.loads((root/'training'/arm/str(seed)/'training_log.json').read_text(encoding='utf-8')) for seed in suite['seeds']]
        # Interpolate on common actual-step support only; also retain the raw
        # per-epoch data in CSV for curves by attempts and elapsed time.
        for seed, log in zip(suite['seeds'], logs):
            for r in log: curve_rows.append({'arm':arm,'seed':seed,'epoch':r['epoch'],'steps':r['training_steps'],
                                             'attempts':r['episodes_started'],
                                             **{k:r.get(k) for k in ('flow_mean','cost_mean','workload_mean','valid_objective_count')},
                                             'training_seconds':r['training_seconds']})
        validation = [json.loads((root/'training'/arm/str(seed)/'validations.json').read_text(encoding='utf-8')) for seed in suite['seeds']]
        left = max(v[0]['training_steps'] for v in validation)
        right = min(v[-1]['training_steps'] for v in validation)
        if right < left: continue
        x = np.linspace(left, right, 100)
        y = np.stack([np.interp(x,[r['training_steps'] for r in v],[r['summary']['quality'] for r in v]) for v in validation])
        line, = ax.plot(x,y.mean(0),label=arm)
        ax.fill_between(x,y.mean(0)-y.std(0),y.mean(0)+y.std(0),color=line.get_color(),alpha=.12)
    ax.set(xlabel='Training environment steps',ylabel='Validation public Q'); ax.legend(fontsize=8,ncol=2)
    save(fig,'learning_curves');write_csv(out/'learning_curves.csv',curve_rows)
    search_rows = []
    for task in TASKS:
        fig, ax = plt.subplots(figsize=(6,4))
        for method in suite['search_methods']:
            grouped = defaultdict(list)
            for u in units:
                if u['task'] != task or u['arm'] != method: continue
                for search in u['searches']:
                    for snap in search['snapshots']:
                        metric = 'paid_hv' if task == 'frontier' else 'mean_quality'
                        grouped[snap['seconds']].append(snap[metric])
                        search_rows.append({'task':task,'arm':method,'seed':u['seed'],'instance_id':u['instance_id'],
                                            'preference':json.dumps(search['preference']),**snap})
            x = sorted(grouped); y = [fmean(grouped[t]) for t in x]
            ax.plot(x,y,marker='o',label=method,color=styles[method])
        ax.set(xlabel='Paid seconds',ylabel='HV' if task == 'frontier' else 'Public Q');ax.legend()
        save(fig,f'search_curves_{task}')
    write_csv(out/'search_curves.csv',search_rows)
    fig, ax = plt.subplots(figsize=(7,4))
    for method, color in styles.items():
        g = [r for r in instances if r['arm'] == method and r['task'] == 'independent_request']
        x = ['small','medium','large']
        ax.plot(x,[fmean(r['quality'] for r in g if r['scale']==scale) for scale in x],marker='o',label=method,color=color)
    ax.set(xlabel='Instance scale',ylabel='Public Q');ax.legend(fontsize=8,ncol=2);save(fig,'scale_quality')
    fig, ax = plt.subplots(figsize=(9,4))
    for method, color in styles.items():
        group = sorted((r for r in instances if r['arm']==method and r['task']=='independent_request'),key=lambda r:r['cell_id'])
        ax.plot([r['cell_id'] for r in group],[r['quality'] for r in group],label=method,color=color,marker='.',linewidth=1)
    ax.set(xlabel='Design cell',ylabel='Public Q');ax.tick_params(axis='x',labelrotation=90,labelsize=8)
    ax.legend(fontsize=8,ncol=3);save(fig,'instance_quality')
    for cell in suite['representative_cells']:
        fig = plt.figure(figsize=(6,5));ax = fig.add_subplot(111, projection='3d')
        for method, color in styles.items():
            # A fixed seed shows an actual paid front, not a union of five runs.
            unit = next(u for u in units if u['task']=='frontier' and u['arm']==method and u['cell_id']==cell and u['seed']==11)
            if method in suite['search_methods']:
                points = [c['objectives'] for s in unit['searches'] for c in s['archive']]
            else:
                points = [c['objectives'] for r in unit['requests'] for c in r['raw'] if c['success']]
            points = [normalize_objectives(p,scales) for p in points]
            front = np.asarray([points[i] for i in nondominated_indices(points)])
            if len(front): ax.scatter(*front.T,label=method,color=color,s=14)
        ax.set(xlabel='Excess Flow (bounded)',ylabel='Cost (bounded)',zlabel='OWA (bounded)');ax.legend(fontsize=7)
        save(fig,f'frontier_{cell}_seed11')


def report(root, suite, identity, device='cpu'):
    from data import load_dataset_split
    from agent.comparators.learning import check_learning_checkpoint
    from result.training_history import export_training
    from configs.experiment_reuse import arm_execution_identity, registered_execution, resolve_comparison_identity
    root = Path(root)
    registration = registered_execution(root)
    if registration and resolve_comparison_identity(root, registration['execution_identity']) != identity:
        raise ValueError('report comparison identity differs from registered execution')
    from scripts.experiment_suite import check_frozen_budget
    check_frozen_budget(suite,root,identity)
    streams = validate_training_streams(root, suite)
    for arm, definition in suite['arms'].items():
        for seed in suite['seeds']:
            config = arm_config(suite,arm,seed,device)
            execution_identity = arm_execution_identity(root, arm)
            if execution_identity:
                config['comparison']['execution_identity'] = execution_identity
            check_learning_checkpoint(root/'training'/arm/str(seed)/'best_checkpoint.pt',
                                      config, definition['method'],seed,identity)
    c = arm_config(suite,'full',11,device)
    def read_units():
        for p in sorted((root/'evaluation').rglob('*.json')):
            u = json.loads(p.read_text(encoding='utf-8'))
            if 'content_sha256' not in u:
                raise ValueError(f'evaluation missing content hash: {p}')
            yield u
    units = []
    executions = {arm:arm_execution_identity(root, arm) for arm in suite['arms']}
    rows = audit_evaluations(read_units(),suite,c,list(load_dataset_split(c,'test')),identity,
                             compact_units=units,execution_identities=executions)
    instances = instance_tables(rows,units)
    descriptive, seeds = descriptive_tables(instances,rows)
    stats = paired_statistics(instances,suite)
    common = common_success_statistics(rows,suite)
    out = root/'report';out.mkdir(parents=True,exist_ok=True)
    export_training(root/'training')
    write_csv(out/'deliveries.csv',rows);write_csv(out/'per_instance.csv',instances)
    write_csv(out/'summary.csv',descriptive);write_csv(out/'seed_results.csv',seeds)
    variability = []
    for task in TASKS:
        for arm in sorted({r['arm'] for r in seeds}):
            group = [r['quality'] for r in seeds if r['arm']==arm and r['task']==task]
            variability.append({'task':task,'arm':arm,'replicates':len(group),'quality_mean':fmean(group),
                                'quality_seed_std':stdev(group) if len(group)>1 else None})
    write_csv(out/'seed_variability.csv',variability)
    write_csv(out/'paired_statistics.csv',stats);write_csv(out/'common_success.csv',common)
    training = []
    for arm in suite['arms']:
        for seed in suite['seeds']:
            training.append({'arm':arm,'seed':seed,**json.loads((root/'training'/arm/str(seed)/'summary.json').read_text(encoding='utf-8'))})
    from result.training_history import csv_rows
    write_csv(out/'training_summary.csv',csv_rows(training))
    for family, methods in GROUPS.items():
        write_csv(out/f'table_{family}.csv',[r for r in descriptive if r['arm'] in ('full',*methods) and r['factor'] in ('all','scale')])
    figures(root,suite,units,instances)
    atomic_json(out/'audit.json', {'status':'passed','identity':identity,'execution_registration':registration,
                'training_streams':streams,
                'training_runs':len(training),'deliveries':len(rows),'instances':len({r['instance_id'] for r in rows}),
                'statistics_unit':'fixed_instance','training_sufficiency':'not_established',
                'search_sufficiency':'not_established','rule_timing':'shared deterministic solve plus preference rescore'})
    print(f'complete report: {out}',flush=True)
