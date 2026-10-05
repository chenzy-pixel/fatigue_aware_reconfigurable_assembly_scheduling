"""Measure current cache benefit, inference batching, and projection CPU hot spots."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
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
from environment import AssemblySchedulingEnv
from environment.resource_projection import ResourceProjector


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sample_states(config):
    instances = [load_instance_yaml(config['paths']['fixed_instance']),
                 load_generated_record(str(Path(config['paths']['instances_root'])/'validation'/'instance_2000000.json')).instance]
    rows = []
    for instance in instances:
        env = AssemblySchedulingEnv(config)
        env.reset(instance, build_observation=False)
        policy = HeuristicPolicy()
        for index in range(145):
            if index in (0, 12, 36, 72, 108, 144):
                copy = deepcopy(env)
                copy._invalidate_resource_snapshot()
                rows.append((f'{instance.instance_id}:{index}:{env.decision_type.value}', copy))
            if env.task_done:
                break
            env.step(policy.select_action(env), build_observation=False)
        print(f'Snapshots from {instance.instance_id}', flush=True)
    return rows


def median_ms(function, count=7, device='cpu'):
    for _ in range(2):
        function()
    durations = []
    for _ in range(count):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        if device == 'cuda':
            torch.cuda.synchronize()
        durations.append(1000*(time.perf_counter()-start))
    return statistics.median(durations)


@contextmanager
def count_projection_calls():
    names = ('initial_resources', 'stage_options', 'transition', 'active', 'machine_release', 'candidate')
    originals = {name: getattr(ResourceProjector, name) for name in names}
    counts = Counter()
    def wrap(name):
        original = originals[name]
        def call(self, *args, **kwargs):
            counts[name] += 1
            return original(self, *args, **kwargs)
        return call
    try:
        for name in names:
            setattr(ResourceProjector, name, wrap(name))
        yield counts
    finally:
        for name, original in originals.items():
            setattr(ResourceProjector, name, original)


def environment_benchmark(states):
    results = []
    for label, env in states:
        def cold():
            env._invalidate_resource_snapshot()
            return env.observe()
        cold_ms = median_ms(cold, count=5)
        warm_ms = median_ms(env.observe, count=31)
        def cold_mask():
            env._action_mask_cache = None
            env._action_mask_cache_version = -1
            return env.get_action_mask()
        mask_ms = median_ms(cold_mask, count=7)
        warm_mask_ms = median_ms(env.get_action_mask, count=31)
        with count_projection_calls() as counts:
            env._invalidate_resource_snapshot()
            obs = env.observe()
            call_counts = dict(counts)
            candidate_cache = len(env._candidate_projection_cache)
            group_cache = len(env._production_resource_profile_cache)
            env.observe()
            repeat_extra = dict(Counter(counts)-Counter(call_counts))
        results.append(dict(label=label, phase=env.decision_type.value, operations=len(env.operations),
                            capability_edges=obs.relations[('operation','capable_on','machine')].num_edges,
                            cold_observation_ms=cold_ms, cached_observation_ms=warm_ms,
                            cold_mask_ms=mask_ms, cached_mask_ms=warm_mask_ms,
                            candidate_cache_entries=candidate_cache, group_cache_entries=group_cache,
                            projection_calls=call_counts, repeated_observation_extra_projection_calls=repeat_extra))
        print(f'Environment {label}: {cold_ms:.2f} ms / cached {warm_ms:.3f} ms', flush=True)
    return results


def profile_environment(states, output):
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(2):
        for _, env in states:
            env._invalidate_resource_snapshot()
            env.observe()
    profiler.disable()
    profiler.dump_stats(str(output/'environment.prof'))
    stats = pstats.Stats(profiler)
    entries = []
    for (file, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        entries.append(dict(file=file, line=line, name=name, calls=calls, primitive_calls=primitive,
                            self_seconds=own, cumulative_seconds=cumulative))
    return dict(total_profile_seconds=stats.total_tt,
                top_self=sorted(entries, key=lambda r: r['self_seconds'], reverse=True)[:25],
                top_cumulative=sorted(entries, key=lambda r: r['cumulative_seconds'], reverse=True)[:25])


def network_benchmark(config, states):
    samples = [(env.observe(), env.get_action_mask()) for _, env in states]
    devices = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])
    results = []
    for device in devices:
        torch.manual_seed(11)
        network = build_actor_critic(samples[0][0], config['network']).to(device).eval()
        with torch.no_grad():
            for batch_size in (1, 20):
                observations = [samples[i % len(samples)][0] for i in range(batch_size)]
                masks = [samples[i % len(samples)][1] for i in range(batch_size)]
                reference, fast, critic = {}, {}, {}
                for mode, bucket in (('reference_v8', reference), ('phase_batched_v1', fast)):
                    network.execution_mode = mode
                    bucket['forward_ms'] = median_ms(lambda: network.forward_batch(observations, masks, device=device), count=11, device=device)
                    bucket['logits'], bucket['values'] = network.forward_batch(observations, masks, device=device)
                    bucket['collate_ms'] = median_ms(lambda: (network._collate_graphs_reference if mode=='reference_v8' else network._collate_graphs)(observations, device=device), count=11, device=device)
                for i, mask in enumerate(masks):
                    legal = torch.as_tensor(~mask, device=device)
                    torch.testing.assert_close(fast['logits'][i, :len(mask)][legal], reference['logits'][i, :len(mask)][legal], atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(fast['values'], reference['values'], atol=1e-5, rtol=1e-4)
                network.execution_mode = 'phase_batched_v1'
                critic_ms = median_ms(lambda: network.value_batch(observations, masks, device=device), count=11, device=device)
                torch.testing.assert_close(network.value_batch(observations, masks, device=device), fast['values'], atol=1e-5, rtol=1e-4)
                serial_ms = median_ms(lambda: [network(obs, mask, device=device) for obs, mask in zip(observations, masks)], count=7, device=device)
                results.append(dict(device=device, batch_size=batch_size,
                                    reference_forward_ms=reference['forward_ms'], optimized_forward_ms=fast['forward_ms'],
                                    reference_collate_ms=reference['collate_ms'], optimized_collate_ms=fast['collate_ms'],
                                    value_only_ms=critic_ms, optimized_serial_ms=serial_ms,
                                    reference_over_optimized=reference['forward_ms']/fast['forward_ms'],
                                    serial_over_batch=serial_ms/fast['forward_ms'],
                                    full_over_value_only=fast['forward_ms']/critic_ms))
                print(f'Network {device} batch {batch_size}: ref={reference["forward_ms"]:.2f}, opt={fast["forward_ms"]:.2f}, value={critic_ms:.2f} ms', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='result/analysis/schema10_performance_audit')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config('configs/default.json')
    torch.set_num_threads(int(config['training']['torch_num_threads']))
    result = dict(metadata=dict(observation_schema=config['runtime_manifest']['observation_schema'],
                               time_context=config['runtime_manifest']['time_context'], python=platform.python_version(),
                               torch=torch.__version__, threads=torch.get_num_threads(), logical_cpu=os.cpu_count(),
                               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                               hidden_dim=config['network']['hidden_dim'], layers=config['network']['message_passing_layers'],
                               forced_action_compression=config['training']['forced_action_compression'],
                               worker_local_physical_forced_actions=config['training']['worker_local_physical_forced_actions']))
    states = sample_states(config)
    result['environment'] = environment_benchmark(states)
    write_json(output/'results.json', result)
    result['profile'] = profile_environment(states, output)
    write_json(output/'results.json', result)
    result['network'] = network_benchmark(config, states)
    result['source_hashes'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        'environment/env.py','environment/time_context.py','environment/resource_projection.py',
        'agent/ppo/network.py','agent/ppo/agent.py','agent/ppo/parallel.py','configs/default.json')}
    write_json(output/'results.json', result)
    print('Benchmark complete', flush=True)


if __name__ == '__main__':
    main()
