"""Paid per-instance artifacts for frontiers and independent requests."""
import time
from dataclasses import asdict

from analysis.pareto_analysis import hypervolume_3d, normalize_objectives
from configs.formal_preferences import objective_scales
from environment import PreferenceVector
from environment.types import bounded_quality_score
from agent.comparators.evolution import EvolutionSolver
from agent.comparators.learning import evaluate_request, synchronize
from agent.mo_alns import SearchBudget
from agent.mo_alns.types import preference_key


def quality(config, objectives, preference):
    return bounded_quality_score(*objectives, config, preference=PreferenceVector(*preference))


def frontier_metrics(config, candidates, deliveries):
    scales = objective_scales(config)
    paid = [normalize_objectives(r['objectives'], scales) for r in candidates if r['success']]
    selected = [normalize_objectives(r['objectives'], scales) for r in deliveries if r['success']]
    return {'paid_hv': hypervolume_3d(paid), 'delivery_hv': hypervolume_3d(selected),
            'coverage': sum(r['success'] for r in deliveries)/len(deliveries) if deliveries else 0.}


def learning_unit(config, agent, record, method, prefs, seed, *, mode=None):
    synchronize(config)
    began = time.perf_counter()
    results = [evaluate_request(config, agent, record.instance, pref, method, mode=mode, seed=seed) for pref in prefs]
    deliveries = [r['summary'] for r in results]
    candidates = [raw for r in results for raw in r['raw']]
    synchronize(config)
    elapsed = time.perf_counter()-began
    return {'deliveries': deliveries, 'requests': results, 'seconds': elapsed,
            'environment_steps': sum(r['environment_steps'] for r in deliveries),
            **frontier_metrics(config, candidates, deliveries)}


def rule_unit(config, raw, prefs):
    began = time.perf_counter()
    deliveries = []
    for pref in prefs:
        score_started = time.perf_counter()
        q = quality(config, raw['objectives'], pref) if raw['success'] else 1.
        deliveries.append({'preference': pref, 'success': raw['success'], 'quality': q,
                           'objectives': raw['objectives'], 'search_seconds': raw['seconds'] + time.perf_counter()-score_started,
                           'environment_steps': len(raw['actions']), 'candidate_count': 1,
                           'timing_basis': 'shared_deterministic_solve_plus_rescore'})
    return {'deliveries': deliveries, 'requests': [], 'raw_rule': raw,
            'seconds': raw['seconds']+time.perf_counter()-began,
            'environment_steps': len(raw['actions']), 'quality_replication_unit': 'instance',
            **frontier_metrics(config, [raw], deliveries)}


def search_snapshots(config, result, prefs, checkpoints):
    rows = []
    for seconds in checkpoints:
        history = [r for r in result.candidate_history if r['feasible'] and r['completed_seconds'] <= seconds]
        objectives = [r['objectives'] for r in history]
        rows.append({'seconds': seconds, 'paid_hv': hypervolume_3d([normalize_objectives(v, objective_scales(config)) for v in objectives]),
                     'distinct_feasible': len({r['action_trace_sha256'] for r in history}),
                     'mean_quality': sum(min((quality(config, v, pref) for v in objectives), default=1.) for pref in prefs)/len(prefs),
                     'completed_evaluations': sum(r['completed_seconds'] <= seconds for r in result.candidate_history)})
    return rows


def search_unit(config, record, method, prefs, seed, task, *, seconds=600, checkpoints=(60,180,600), max_evaluations=None):
    began = time.perf_counter()
    solver = EvolutionSolver(config, method, algorithm_seed=seed, dataset_name=task)
    # A frontier request pays for one population; each independent request gets
    # a fresh solver budget and archive. Candidate completion includes validation.
    results, endpoints = [], []
    requests = (None,) if task == 'frontier' else prefs
    for pref in requests:
        request_began = time.perf_counter()
        endpoint_start = len(endpoints)
        budget = SearchBudget(seconds=seconds, max_evaluations=max_evaluations)
        if pref is None:
            grid = solver.solve_frontier(record.instance, prefs, budget=budget)
            result = grid.searches[0]
            selected = [grid.endpoints[preference_key(p)] for p in grid.preferences]
            endpoint_prefs = prefs
        else:
            result = solver.solve(record.instance, PreferenceVector(*pref), budget=budget)
            selected, endpoint_prefs = [result.selected], [pref]
        for preference, candidate in zip(endpoint_prefs, selected):
            endpoints.append({'preference': preference, 'success': candidate is not None,
                              'quality': quality(config, candidate.objectives, preference) if candidate else 1.,
                              'objectives': candidate.objectives if candidate else None,
                              'search_seconds': result.search_time_seconds,
                              'candidate_count': result.environment_evaluations,
                              'environment_steps': result.accounting['environment_steps']})
        request_seconds = time.perf_counter()-request_began
        for endpoint in endpoints[endpoint_start:]:
            endpoint['search_only_seconds'] = endpoint['search_seconds']
            endpoint['search_seconds'] = request_seconds
        results.append({'preference': pref, 'archive': [asdict(c) for c in result.archive],
                        'operator_statistics': result.operator_statistics, 'search_log': result.search_log,
                        'candidate_history': result.candidate_history, 'accounting': result.accounting,
                        'snapshots': search_snapshots(config, result, prefs if pref is None else [pref], checkpoints),
                        'seconds': result.search_time_seconds, 'initial_seconds': result.initial_seconds,
                        'stop_reason': result.stop_reason})
    candidates = [{'success': True, 'objectives': c['objectives']} for r in results for c in r['archive']]
    return {'deliveries': endpoints, 'searches': results, 'seconds': time.perf_counter()-began,
            'environment_steps': sum(r['accounting']['environment_steps'] for r in results),
            **frontier_metrics(config, candidates, endpoints)}


def development_pass(payload, search):
    return all(r['operator_statistics']['initialization_complete']
               and r['operator_statistics']['completed_generations'] + r['operator_statistics']['completed_scans'] >= search['minimum_iterations']
               and r['snapshots'][-1]['distinct_feasible'] >= search['minimum_distinct_feasible']
               for r in payload['searches'])
