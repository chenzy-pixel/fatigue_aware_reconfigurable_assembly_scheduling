"""Fixed request seeds on the shared lane-refill evaluation transport."""
from contextlib import contextmanager
import time

import torch

from agent.ppo.parallel import ParallelEpisodeRunner
from environment import PreferenceVector
from environment.types import metrics_flow_objective
from environment.workload import metrics_workload_objective
from .common import digest, physical_identity


def sampling_seed(seed, instance_id, pref, repeat):
    return int(digest([seed, instance_id, pref, repeat, 'comparison_sampling_v1'])[:15], 16)


@contextmanager
def evaluation_pool(config, template, *, workers=None):
    count = int(workers if workers is not None else config['training'].get('validation_parallel_envs', 20))
    if count < 1:
        raise ValueError('evaluation worker count must be positive')
    with ParallelEpisodeRunner(config=config, template=template, episode_count=1, worker_count=count) as runner:
        yield runner


def summarize(instance_id, pref, method, mode, raw, seconds, *, timing_basis):
    sampled = method != 'rule_ddqn' if mode is None else mode in ('sampled', 'sampled_single')
    selected = min(raw, key=lambda r: (not r['success'], r['quality'] if r['success'] else 1., r['sampling_repeat']))
    return {'summary': {'instance_id': instance_id, 'preference': pref, 'method': method,
                        'delivery_protocol': 'sampled_single_v1' if mode == 'sampled_single' else 'best_of_3_v1' if sampled else 'greedy_single_v1',
                        'success': selected['success'], 'quality': selected['quality'] if selected['success'] else 1.,
                        'objectives': selected['objectives'], 'search_seconds': seconds,
                        'candidate_count': len(raw), 'environment_steps': sum(len(r['actions']) for r in raw),
                        'timing_basis': timing_basis}, 'raw': raw, 'selected_repeat': selected['sampling_repeat']}


def evaluate_requests(config, agent, requests, method, *, runner=None, mode=None, seed=0,
                      capture_trajectories=True, progress_callback=None):
    """Return request-ordered deliveries plus non-overlapping wall-clock timing.

    requests contains (record, preference) pairs. Each repeat owns its RNG,
    independent of lane count, batch composition and refill order.
    """
    requests = list(requests)
    if not requests:
        raise ValueError('at least one evaluation request is required')
    workers = int(config['training'].get('validation_parallel_envs', 20))
    if workers < 1:
        raise ValueError('evaluation worker count must be positive')
    if runner is None:
        with evaluation_pool(config, requests[0][0].instance, workers=workers) as pool:
            return evaluate_requests(config, agent, requests, method, runner=pool, mode=mode,
                                     seed=seed, capture_trajectories=capture_trajectories,
                                     progress_callback=progress_callback)
    if workers > runner.worker_count:
        raise ValueError('evaluation worker pool is smaller than requested parallelism')
    if physical_identity(config) != physical_identity(runner.config):
        raise ValueError('evaluation worker pool physics differs from request config')
    agent.assert_evaluation_config(config)
    sampled = method != 'rule_ddqn' if mode is None else mode in ('sampled', 'sampled_single')
    if method == 'rule_ddqn' and sampled:
        raise ValueError('Rule-DDQN delivery is greedy')
    repeats = 1 if mode == 'sampled_single' else 3 if sampled else 1
    records, prefs, seeds = [], [], []
    for record, pref in requests:
        for repeat in range(repeats):
            records.append(record)
            # Preserve the exact weights accepted by the serial reference. A
            # sequence would be normalized again by PreferenceContext.
            prefs.append(PreferenceVector(*pref))
            seeds.append(sampling_seed(seed, record.instance.instance_id, pref, repeat))
    device = torch.device(config['device'])
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    rollouts = runner.evaluate_records(agent, records, preferences=prefs,
                                      max_parallelism=min(workers, runner.worker_count),
                                      deterministic=not sampled, unit_sampling_seeds=seeds if sampled else None,
                                      capture_trajectories=capture_trajectories, capture_actions=True,
                                      progress_callback=progress_callback)
    if [r.record_index for r in rollouts] != list(range(len(records))):
        raise RuntimeError('evaluation rollouts are missing, duplicated or out of order')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    results = []
    for index, (record, pref) in enumerate(requests):
        raw = []
        for repeat, rollout in enumerate(rollouts[index*repeats:(index+1)*repeats]):
            metrics = dict(rollout.metrics)
            violations = metrics.pop('schedule_violations')
            actions = list(rollout.actions)
            if rollout.decisions != len(actions):
                raise RuntimeError('evaluation action accounting differs')
            raw.append({'success': bool(metrics['task_succeeded'] and not violations),
                        'quality': float(metrics['preference_quality_score']),
                        'objectives': (metrics_flow_objective(metrics, config), metrics['reconfiguration_cost'], metrics_workload_objective(metrics, config)),
                        'sampling_repeat': repeat, 'sampling_seed': seeds[index*repeats+repeat] if sampled else None,
                        'actions': actions, 'trace': digest(actions), 'metrics': metrics, 'violations': violations,
                        'schedule': rollout.schedule, 'reconfigurations': rollout.reconfigurations,
                        'seconds': rollout.solve_time_seconds, 'timing_basis': 'parallel_lane_elapsed',
                        'inference_seconds_share': rollout.inference_time_seconds})
        result = summarize(record.instance.instance_id, pref, method, mode, raw, wall/len(requests),
                           timing_basis='parallel_batch_wall_share')
        result['summary']['lane_elapsed_seconds'] = sum(r['seconds'] for r in raw)
        results.append(result)
    return results, {**runner.last_evaluation_runtime, 'wall_seconds': wall, 'workers': min(workers, runner.worker_count),
                     'trajectory_count': len(rollouts), 'request_count': len(requests),
                     'lane_elapsed_seconds': sum(r.solve_time_seconds for r in rollouts)}


def reuse_first_sample(config, unit):
    """Quality from the already paid prefix; standalone latency is measured separately."""
    from .experiment_evaluation import frontier_metrics
    results = []
    for request in unit['requests']:
        s = request['summary']
        if (s['delivery_protocol'] != 'best_of_3_v1'
                or [r['sampling_repeat'] for r in request['raw']] != [0, 1, 2]):
            raise ValueError('sample1 reuse requires the ordered best-of-three parent')
        results.append(summarize(s['instance_id'], s['preference'], s['method'], 'sampled_single',
                                 [request['raw'][0]], s['search_seconds'], timing_basis='shared_best_of_3_wall'))
    deliveries = [r['summary'] for r in results]
    return {'deliveries': deliveries, 'requests': results, 'seconds': unit['seconds'],
            'incremental_solve_seconds': 0., 'timing_basis': 'shared_best_of_3_wall',
            'cost_reference': 'full', 'environment_steps': 0,
            'source_job_id': unit.get('job_id'), 'source_content_sha256': unit.get('content_sha256'),
            'reused_environment_steps': sum(r['environment_steps'] for r in deliveries),
            **frontier_metrics(config, [r['raw'][0] for r in results], deliveries)}
