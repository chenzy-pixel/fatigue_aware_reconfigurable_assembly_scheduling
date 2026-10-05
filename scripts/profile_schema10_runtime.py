"""Measure retained runtime optimizations and current CPU projection hot spots."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
from statistics import median
import sys
import time

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from agent.baselines import HeuristicPolicy
from agent.ppo.network import build_actor_critic
from configs import load_config
from data.dataset import load_generated_record
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv, CAPABLE_EDGE
from environment.resource_projection import ResourceProjector
from scripts.audit_observation_reward import compare_observations

OUT = Path('result/analysis/schema10_runtime_profile')


def write_json(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def measure(fn, repetitions=5, synchronize=False):
    fn()
    if synchronize:
        torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        if synchronize:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if synchronize:
            torch.cuda.synchronize()
        samples.append(time.perf_counter()-start)
    return median(samples)


def collect_states(config, instance):
    env = AssemblySchedulingEnv(config)
    env.reset(instance, build_observation=False)
    policy = HeuristicPolicy()
    states = []
    for step in range(61):
        if step in (0, 10, 20, 30, 40, 60):
            saved = deepcopy(env)
            saved._invalidate_resource_snapshot()
            states.append(saved)
        if env.task_done:
            break
        env.step(policy.select_action(env), build_observation=False)
    return states


def environment_timings(states):
    def cold():
        for env in states:
            env._invalidate_resource_snapshot()
            env.observe()

    def cached():
        for env in states:
            env.observe()

    cold_seconds = measure(cold, 3)/len(states)
    cached_seconds = measure(cached, 9)/len(states)
    def mask_cold():
        for env in states:
            env._invalidate_resource_snapshot()
            env.get_action_mask()

    mask_cold_seconds = measure(mask_cold, 3)/len(states)
    mask_cached_seconds = measure(lambda: [env.get_action_mask() for env in states], 9)/len(states)
    # Confirm the cache returns an identical but independently owned graph.
    env = states[0]
    first, second = env.observe(), env.observe()
    assert all(compare_observations(first, second).values())
    assert not np.shares_memory(first.relations[CAPABLE_EDGE].edge_features, second.relations[CAPABLE_EDGE].edge_features)
    groups = []
    for env in states:
        env.observe()
        groups.append(dict(phase=env.decision_type.value, time=env.current_time,
                           edge_count=env._static_edge_indices[CAPABLE_EDGE].shape[1],
                           candidate_cache_entries=len(env._candidate_projection_cache),
                           resource_profile_entries=len(env._production_resource_profile_cache)))
    return dict(states=len(states), cold_observe_ms=1000*cold_seconds,
                cached_observe_ms=1000*cached_seconds, cache_speedup=cold_seconds/cached_seconds,
                cold_mask_ms=1000*mask_cold_seconds, cached_mask_ms=1000*mask_cached_seconds,
                state_groups=groups)


def profile_environment(states):
    profiler = cProfile.Profile()
    profiler.enable()
    for env in states:
        env._invalidate_resource_snapshot()
        env.observe()
    profiler.disable()
    profiler.dump_stats(str(OUT/'cold_observe.prof'))
    stats = pstats.Stats(profiler)
    rows = []
    for (file, line, name), (primitive, total, self_time, cumulative, _) in stats.stats.items():
        if 'resource_projection.py' in file or 'time_context.py' in file or name in (
            '_build_capability_relation', '_build_graph_relations', 'observe', 'var', 'clip', '_worker_fatigue_at_availability'):
            rows.append(dict(file=file, line=line, function=name, calls=total,
                             self_seconds=self_time, cumulative_seconds=cumulative))
    return dict(total_seconds=stats.total_tt,
                functions=sorted(rows, key=lambda row: row['cumulative_seconds'], reverse=True))


def count_repeated_paths(states):
    calls, unique = Counter(), Counter()
    original = ResourceProjector.transition
    seen = set()
    def counted(self, mi, source, target, earliest, resources):
        key = (id(self.env), self.env._state_version, self.env.current_tick,
               mi, source, target, earliest, tuple(resources.workers), tuple(resources.loads.tolist()))
        calls['transition'] += 1
        if key not in seen:
            unique['transition'] += 1
            seen.add(key)
        return original(self, mi, source, target, earliest, resources)
    ResourceProjector.transition = counted
    try:
        for env in states:
            env._invalidate_resource_snapshot()
            env.observe()
    finally:
        ResourceProjector.transition = original
    return dict(calls=calls['transition'], unique_inputs=unique['transition'],
                repeated_inputs=calls['transition']-unique['transition'],
                repeated_fraction=1-unique['transition']/calls['transition'])


def network_timings(config, states, device='cpu'):
    observations = [env.observe() for env in states]
    masks = [env.get_action_mask() for env in states]
    torch.manual_seed(11)
    model = build_actor_critic(observations[0], config['network']).to(device).eval()
    def timed(fn):
        return measure(fn, 7, synchronize=device=='cuda')
    sizes = (1, 8, 20)
    results = []
    with torch.no_grad():
        for size in sizes:
            obs = [observations[i % len(observations)] for i in range(size)]
            mask = [masks[i % len(masks)] for i in range(size)]
            model.execution_mode = 'reference_v8'
            reference_logits, reference_values = model.forward_batch(obs, mask, device=device)
            reference = timed(lambda: model.forward_batch(obs, mask, device=device))
            model.execution_mode = 'phase_batched_v1'
            logits, values = model.forward_batch(obs, mask, device=device)
            torch.testing.assert_close(logits, reference_logits, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(values, reference_values, atol=1e-5, rtol=1e-4)
            optimized = timed(lambda: model.forward_batch(obs, mask, device=device))
            critic = timed(lambda: model.value_batch(obs, mask, device=device))
            packing_reference = timed(lambda: model._collate_graphs_reference(obs, device=device))
            packing = timed(lambda: model._collate_graphs(obs, device=device))
            results.append(dict(batch_size=size, reference_forward_ms=reference*1000,
                                optimized_forward_ms=optimized*1000, forward_speedup=reference/optimized,
                                value_only_ms=critic*1000, full_vs_value_speedup=optimized/critic,
                                packing_reference_ms=packing_reference*1000, packing_optimized_ms=packing*1000))
    return results


def main():
    config = load_config('configs/v8/universal.json')
    torch.set_num_threads(int(config['training']['torch_num_threads']))
    cases = [('fixed', load_instance_yaml(config['paths']['fixed_instance'])),
             ('validation_balanced', load_generated_record(str(Path(config['paths']['instances_root'])/'validation'/'instance_2000000.json')).instance),
             ('stress', load_generated_record(str(Path(config['paths']['instances_root'])/'stress'/'instance_5000000.json')).instance)]
    groups = [(label, collect_states(config, instance)) for label, instance in cases]
    meta = dict(python=sys.version, platform=platform.platform(), logical_cpus=os.cpu_count(),
                torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                threads=torch.get_num_threads(), observation_schema=config['runtime_manifest']['observation_schema'],
                hidden_dim=config['network']['hidden_dim'], message_layers=config['network']['message_passing_layers'],
                parallel_envs=config['training']['parallel_envs'], validation_parallel_envs=config['training']['validation_parallel_envs'],
                forced_action_compression=config['training']['forced_action_compression'],
                measurement='CPU environment wall time; CPU/CUDA inference including packing and transfers; warm-up, medians, CUDA synchronization; not full training throughput')
    if meta['cuda_available']:
        meta['cuda_device'] = torch.cuda.get_device_name(0)
    result = dict(environment=meta, state_timings={})
    for label, states in groups:
        result['state_timings'][label] = environment_timings(states)
        write_json('report.json', result)
        print(f'{label}: cold {result["state_timings"][label]["cold_observe_ms"]:.2f} ms / cached {result["state_timings"][label]["cached_observe_ms"]:.3f} ms', flush=True)
    all_states = [env for _, states in groups for env in states]
    result['profile'] = profile_environment(all_states)
    result['repeated_projection_inputs'] = count_repeated_paths(all_states)
    write_json('report.json', result)
    print('CPU environment profile completed', flush=True)
    result['network_timings'] = network_timings(config, all_states)
    print('CPU network measurement completed', flush=True)
    if meta['cuda_available']:
        result['cuda_network_timings'] = network_timings(config, all_states, device='cuda')
    result['source_hashes'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        'environment/env.py','environment/resource_projection.py','environment/time_context.py',
        'agent/ppo/network.py','agent/ppo/agent.py','agent/ppo/parallel.py','configs/default.json','configs/v8/universal.json')}
    write_json('report.json', result)
    print(json.dumps({'network_timings': result['network_timings'],
                      'cuda_network_timings': result.get('cuda_network_timings'),
                      'repeated_projection_inputs': result['repeated_projection_inputs']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
