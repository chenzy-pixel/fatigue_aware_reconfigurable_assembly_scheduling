"""Compare preserved and optimized Flow observation/evaluation runtimes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import cProfile
import inspect
import json
from pathlib import Path
import pstats
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from agent.baselines import HeuristicPolicy
from agent.ppo.parallel import ParallelEpisodeRunner
from agent.ppo import PPOAgent, build_actor_critic
from configs import load_config
from data.dataset import load_dataset_split
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv
import environment.env as env_module
import agent.ppo.parallel as parallel_module

OUT = ROOT / 'result/audits/flow_runtime_20261008'


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def reference_method(name, namespace):
    import textwrap
    captured = (OUT / name).read_text(encoding='utf-8')
    scope = {}
    exec(compile(textwrap.dedent(captured), str(OUT / name), 'exec'), namespace, scope)
    return next(v for v in scope.values() if callable(v))


@contextmanager
def method(cls, name, value):
    previous = getattr(cls, name)
    setattr(cls, name, value)
    try:
        yield
    finally:
        setattr(cls, name, previous)


def states(config):
    cases = [load_instance_yaml(config['paths']['fixed_instance']),
             load_dataset_split(config, 'validation')[0].instance,
             load_dataset_split(config, 'test')[15].instance]
    result = []
    for instance in cases:
        env = AssemblySchedulingEnv(config)
        env.reset(instance, build_observation=False)
        policy = HeuristicPolicy()
        for index in range(401):
            if index in (0, 25, 100, 180, 280, 380):
                result.append((f'{instance.instance_id}:{index}:{env.decision_type.value}', deepcopy(env)))
            if env.task_done:
                break
            env.step(policy.select_action(env), build_observation=False)
    return result


def compare_observations(a, b):
    assert a.node_features.keys() == b.node_features.keys()
    for name in a.node_features:
        np.testing.assert_array_equal(a.node_features[name], b.node_features[name])
    np.testing.assert_array_equal(a.global_features, b.global_features)
    np.testing.assert_array_equal(a.preference, b.preference)
    np.testing.assert_array_equal(a.action_set_features, b.action_set_features)
    assert a.node_ids == b.node_ids and a.decision_type == b.decision_type
    for name in a.relations:
        left, right = a.relations[name], b.relations[name]
        assert left.feature_names == right.feature_names and left.bidirectional == right.bidirectional
        np.testing.assert_array_equal(left.edge_index, right.edge_index)
        np.testing.assert_array_equal(left.edge_features, right.edge_features)


def cold(env):
    env._invalidate_resource_snapshot()
    return env.observe()


def compare_environment(config):
    original = reference_method('reference_capability_builder.py', env_module.__dict__)
    optimized = AssemblySchedulingEnv._build_capability_relation
    cases = states(config)
    report = []
    for label, env in cases:
        with method(AssemblySchedulingEnv, '_build_capability_relation', original):
            before = cold(env)
            before_mask = env.get_action_mask().copy()
        with method(AssemblySchedulingEnv, '_build_capability_relation', optimized):
            after = cold(env)
            after_mask = env.get_action_mask().copy()
        compare_observations(before, after)
        np.testing.assert_array_equal(before_mask, after_mask)
        timings = {'before': [], 'after': []}
        for repeat in range(9):
            for version in (('before', 'after') if repeat % 2 == 0 else ('after', 'before')):
                with method(AssemblySchedulingEnv, '_build_capability_relation', original if version == 'before' else optimized):
                    start = time.perf_counter()
                    cold(env)
                    timings[version].append(1000 * (time.perf_counter() - start))
        row = {'state': label, 'before_ms': statistics.median(timings['before'][2:]),
               'after_ms': statistics.median(timings['after'][2:]), 'observation_arrays_exact': True}
        row['speedup'] = row['before_ms'] / row['after_ms']
        report.append(row)
        print(json.dumps(row), flush=True)
    write('environment_comparison.json', {'states': report,
        'sum_median_before_ms': sum(r['before_ms'] for r in report),
        'sum_median_after_ms': sum(r['after_ms'] for r in report),
        'measurement': 'Alternating cold full-observation construction on identical preserved states; exact comparison of all node/edge/global arrays and masks; not end-to-end training.'})


def profile(config):
    cases = states(config)
    profiler = cProfile.Profile()
    profiler.enable()
    for _, env in cases:
        cold(env)
    profiler.disable()
    profiler.dump_stats(str(OUT / 'baseline_environment.prof'))
    stats = pstats.Stats(profiler)
    rows = [{'file': Path(file).name, 'line': line, 'name': name, 'calls': calls,
             'own_seconds': own, 'cumulative_seconds': cumulative}
            for (file, line, name), (_, calls, own, cumulative, _) in stats.stats.items()]
    rows.sort(key=lambda r: r['cumulative_seconds'], reverse=True)
    write('baseline_profile.json', {'state_count': len(cases), 'profile': rows[:35]})
    print(json.dumps(rows[:18], indent=2), flush=True)


def compare_evaluation(limit=20, repeats=1, workers=4):
    import torch
    config = load_config(ROOT / 'result/runs/flow_seed11_20261008_130244/config.json')
    torch.set_num_threads(int(config['training']['torch_num_threads']))
    template = load_instance_yaml(config['paths']['fixed_instance'])
    env = AssemblySchedulingEnv(config)
    observation = env.reset(template)
    agent = PPOAgent(build_actor_critic(observation, config['network']), config['ppo'], device=config['device'])
    agent.load(ROOT / 'result/runs/flow_seed11_20261008_130244/best_checkpoint.pt', load_optimizer=False)
    dataset = load_dataset_split(config, 'test')
    indices = (0, 1, 9, 15, 2, 4, 7, 11) if limit == 8 else tuple(range(limit))
    records = [dataset[i] for i in indices]
    preferences = [(1., 0., 0.)] * len(records)
    original = reference_method('reference_evaluate_records.py', parallel_module.__dict__)
    optimized = ParallelEpisodeRunner.evaluate_records
    results = {}
    measurements = []
    with ParallelEpisodeRunner(config=config, template=template, episode_count=len(records), worker_count=workers) as runner:
        runner.evaluate_records(agent, records[:1], deterministic=False, sampling_seed=300013,
                                preferences=preferences[:1])
        actual_act_batch = agent.act_batch
        batch_sizes = []

        def tracked_act_batch(observations, masks, **kwargs):
            batch_sizes.append(len(observations))
            return actual_act_batch(observations, masks, **kwargs)

        agent.act_batch = tracked_act_batch
        was_training = agent.network.training
        agent.network.eval()
        try:
            for repeat in range(repeats):
                versions = ('before', 'after') if repeat % 2 == 0 else ('after', 'before')
                for version in versions:
                    batch_sizes.clear()
                    if agent.device.type == 'cuda':
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    with method(ParallelEpisodeRunner, 'evaluate_records', original if version == 'before' else optimized):
                        rollouts = runner.evaluate_records(agent, records, max_parallelism=workers, deterministic=False,
                            sampling_seed=300013, preferences=preferences)
                    if agent.device.type == 'cuda':
                        torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    measurements.append({'repeat': repeat, 'version': version, 'wall_seconds': seconds,
                        'policy_batches': len(batch_sizes), 'mean_active_lanes': statistics.mean(batch_sizes)})
                    results[version] = rollouts
                    write(f'evaluation_measurements_{limit}.json', {'measurements': measurements})
                    print(json.dumps(measurements[-1]), flush=True)
        finally:
            agent.act_batch = actual_act_batch
            agent.network.train(was_training)
    checked_fields = ('task_succeeded', 'task_failed', 'sampling_truncated', 'operation_progress',
        'flow_time_objective', 'reconfiguration_cost', 'worker_load_variance', 'maximum_worker_fatigue',
        'schedule_violations', 'actual_preference_quality_score', 'training_cumulative_reward')
    rows = []
    for before, after in zip(results['before'], results['after']):
        assert before.record_index == after.record_index
        assert before.action_trace_sha256 == after.action_trace_sha256
        assert before.derived_sampling_seed == after.derived_sampling_seed
        assert before.decisions == after.decisions
        for field in checked_fields:
            assert before.metrics[field] == after.metrics[field], field
        rows.append({'instance_id': records[before.record_index].instance.instance_id,
                     'decisions': before.decisions, 'action_trace_equal': True,
                     'core_metrics_equal': True, 'succeeded': before.metrics['task_succeeded']})
    seconds = {version: statistics.median(r['wall_seconds'] for r in measurements if r['version'] == version)
               for version in ('before', 'after')}
    write(f'evaluation_comparison_{limit}.json', {'measurements': measurements, 'medians_seconds': seconds,
        'speedup': seconds['before'] / seconds['after'], 'rollouts': rows, 'checked_fields': checked_fields,
        'worker_count': workers, 'sampling_seed': 300013, 'test_indices': indices,
        'measurement': 'Same-instance sampled evaluations, same selected Flow weights and optimized environment in both variants; isolates lane refill scheduling, not full training.'})
    print('Trajectory hashes and all checked core metrics match.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('capture', 'compare', 'profile', 'evaluation'))
    parser.add_argument('--evaluation-limit', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    config = load_config('configs/e1/single_flow.json')
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'capture':
        for name, function in [('reference_capability_builder.py', AssemblySchedulingEnv._build_capability_relation),
                               ('reference_evaluate_records.py', ParallelEpisodeRunner.evaluate_records)]:
            path = OUT / name
            if path.exists():
                raise FileExistsError(f'Preserved reference already exists: {path}')
            path.write_text(inspect.getsource(function), encoding='utf-8')
        print('Preserved production methods for same-process comparisons.')
    elif args.mode == 'evaluation':
        compare_evaluation(args.evaluation_limit, args.repeats, args.workers)
    elif args.mode == 'profile':
        profile(config)
    else:
        compare_environment(config)


if __name__ == '__main__':
    main()
