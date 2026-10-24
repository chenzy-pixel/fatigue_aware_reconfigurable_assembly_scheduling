"""Local matched-trajectory throughput probe, outside formal experiments."""
import argparse
from dataclasses import replace
import platform
import time

import torch

from configs.experiment_suite import arm_config, load_suite
from data import load_dataset_split
from environment import AssemblySchedulingEnv
from agent.comparators.common import atomic_json, FOUR
from agent.comparators.learning import build_agent, evaluate_request, synchronize
from agent.comparators.parallel_evaluation import evaluation_pool, evaluate_requests
from result.provenance import network_weights_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', default='full')
    parser.add_argument('--max-orders', type=int, default=2,
                        help='crop each calibration instance for a bounded local probe; 0 keeps all orders')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--output', default='result/verification/evaluation_parallel_20261024/benchmark.json')
    args = parser.parse_args()
    if args.max_orders < 0:
        parser.error('max-orders must be nonnegative')
    suite = load_suite()
    torch.set_num_threads(1)
    report = {'purpose': 'local_random_weights_throughput_probe', 'formal_performance_evidence': False,
              'platform': platform.platform(), 'torch': torch.__version__, 'device': args.device,
              'gpu': torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
              'workers': args.workers, 'arms': {}}
    report['max_orders'] = args.max_orders
    records = None
    for arm in args.arms.split(','):
        print(f'{arm}: prepare data and model', flush=True)
        c = arm_config(suite, arm, 11, args.device)
        c['training']['validation_parallel_envs'] = args.workers
        c['training']['worker_timeout_seconds'] = 90
        c['training']['worker_stall_timeout_seconds'] = 60
        if records is None:
            dataset = load_dataset_split(c, 'calibration')
            records = [next(r for r in dataset if r.metadata['cell_id'] == cell) for cell in ('C01', 'C14', 'C27')]
            if args.max_orders:
                records = [replace(r, instance=replace(r.instance, instance_id=r.instance.instance_id+f'_probe{args.max_orders}',
                                                      orders=r.instance.orders[:args.max_orders])) for r in records]
        torch.manual_seed(31)
        method = suite['arms'][arm]['method']
        e = AssemblySchedulingEnv(c)
        o = e.reset(records[0].instance)
        agent = build_agent(c, o, method)
        print(f'{arm}: warm model', flush=True)
        if method == 'rule_ddqn':
            agent.act_rules([o], [e.get_action_mask()])
        else:
            agent.act(o, e.get_action_mask(), deterministic=True)
        network = agent.online if method == 'rule_ddqn' else agent.network
        weights_before = network_weights_sha256(network.state_dict())
        requests = [(r, p) for r in records for p in FOUR[:3]]
        started = time.perf_counter()
        print(f'{arm}: start {args.workers} workers', flush=True)
        with evaluation_pool(c, records[0].instance, workers=args.workers) as runner:
            startup = time.perf_counter()-started
            print(f'{arm}: pool ready after {startup:.3f}s; warm worker evaluation', flush=True)
            # Warm both execution paths before timing; pool startup is separate.
            evaluate_requests(c, agent, requests[:1], method, runner=runner, seed=11, capture_trajectories=False)
            synchronize(c)
            started = time.perf_counter()
            serial = []
            for i, (r, p) in enumerate(requests):
                serial.append(evaluate_request(c, agent, r.instance, p, method, seed=11))
                print(f'{arm} serial {i+1}/{len(requests)}', flush=True)
            serial_seconds = time.perf_counter()-started
            parallel, runtime = evaluate_requests(c, agent, requests, method, runner=runner, seed=11)
        for a, b in zip(serial, parallel):
            assert a['selected_repeat'] == b['selected_repeat']
            for x, y in zip(a['raw'], b['raw']):
                for key in ('actions', 'success', 'violations', 'schedule', 'reconfigurations'):
                    assert x[key] == y[key], (arm, key)
                for key in ('objectives', 'quality'):
                    import numpy as np
                    assert np.allclose(x[key], y[key], rtol=1e-12, atol=1e-12), (arm, key)
        assert weights_before == network_weights_sha256(network.state_dict())
        report['arms'][arm] = {'serial_seconds': serial_seconds, 'parallel_seconds': runtime['wall_seconds'],
                               'speedup': serial_seconds/runtime['wall_seconds'], 'pool_startup_seconds': startup,
                               'runtime': runtime, 'requests': len(requests), 'cells': ['C01', 'C14', 'C27'],
                               'trajectory_parity': True, 'network_weights_sha256': weights_before,
                               'environment_steps': sum(r['summary']['environment_steps'] for r in serial)}
        atomic_json(args.output, report)
        print(f'{arm}: {report["arms"][arm]}', flush=True)


if __name__ == '__main__':
    main()
