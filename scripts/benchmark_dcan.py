"""Matched DCAN execution benchmark; separate from formal training results."""
import argparse
from copy import deepcopy
from dataclasses import replace
import platform
from statistics import median
import time

import numpy as np
import torch

from agent.comparators.common import atomic_json, FOUR
from agent.comparators.learning import build_agent, synchronize
from agent.comparators.parallel_evaluation import evaluation_pool, evaluate_requests
from agent.ppo.buffer import RolloutBuffer
from configs.experiment_suite import arm_config, load_suite
from data import load_dataset_split
from environment import AssemblySchedulingEnv, DecisionType
from result.provenance import network_weights_sha256


MODES = ("reference_dcan_v1", "packed_neighbors_v1")


def snapshots(config, records, batch_size, max_orders=0):
    rows = []
    for record in records:
        instance = record.instance
        if max_orders:
            instance = replace(instance, orders=instance.orders[:max_orders])
        target = instance.operations[0].required_module
        machines = tuple(replace(m, initial_module=next(a for a in m.module_parameters if a != target))
                         for m in instance.machines)
        env = AssemblySchedulingEnv(config)
        obs = env.reset(replace(instance, machines=machines))
        rows.append((obs.copy(), env.get_action_mask().copy()))
        for _ in range(30):
            obs, *_ = env.step(int(np.flatnonzero(~env.get_action_mask())[0]))
            if obs.decision_type == DecisionType.WORKER:
                rows.append((obs.copy(), env.get_action_mask().copy()))
                break
        else:
            raise RuntimeError("benchmark did not reach a worker decision")
    return [rows[i % len(rows)] for i in range(batch_size)]


def timed(config, function, repeats):
    samples = []
    for _ in range(repeats):
        synchronize(config)
        started = time.perf_counter()
        function()
        synchronize(config)
        samples.append(time.perf_counter()-started)
    return {"seconds": samples, "median_seconds": median(samples)}


def compare_rows(reference, optimized):
    for a, b in zip(reference, optimized, strict=True):
        assert a["selected_repeat"] == b["selected_repeat"]
        for x, y in zip(a["raw"], b["raw"], strict=True):
            for key in ("actions", "success", "violations", "schedule", "reconfigurations"):
                assert x[key] == y[key], key
            for key in ("objectives", "quality"):
                np.testing.assert_allclose(x[key], y[key], rtol=1e-12, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output", default="result/verification/dcan_optimization_20261028/benchmark.json")
    args = parser.parse_args()
    if min(args.repeats, args.workers) < 1:
        parser.error("repeats and workers must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(31)
    config = arm_config(load_suite(), "dcan", 11, args.device)
    config["training"]["validation_parallel_envs"] = args.workers
    dataset = load_dataset_split(config, "calibration")
    records = [next(r for r in dataset if r.metadata["cell_id"] == cell) for cell in ("C01", "C14", "C27")]
    full_rows = snapshots(config, records, 20)
    agent = build_agent(config, full_rows[0][0], "dcan")
    initial = deepcopy(agent.network.state_dict())
    weights_hash = network_weights_sha256(initial)
    report = {"purpose": "matched_dcan_execution_probe", "formal_performance_evidence": False,
              "platform": platform.platform(), "torch": torch.__version__, "device": args.device,
              "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
              "parameters": sum(p.numel() for p in agent.network.parameters()),
              "network_spec": agent.network.network_spec(), "network_weights_sha256": weights_hash,
              "inference_batch_size": 20, "update_batch_size": config["ppo"]["batch_size"],
              "update_epochs": config["ppo"]["epochs"], "inference_full_instances": True,
              "update_max_orders": 2, "evaluation_max_orders": 2, "workers": args.workers}
    obs, masks = map(list, zip(*full_rows))
    outputs, inference = {}, {}
    for mode in MODES:
        agent.network.execution_mode = mode
        with torch.no_grad():
            outputs[mode] = agent.network.forward_batch(obs, masks, device=args.device)
            inference[mode] = timed(config, lambda: agent.act_batch(obs, masks), args.repeats)
        print(f"inference {mode}: {inference[mode]['median_seconds']:.4f}s", flush=True)
    for a, b in zip(outputs[MODES[0]], outputs[MODES[1]]):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
    inference["speedup"] = inference[MODES[0]]["median_seconds"]/inference[MODES[1]]["median_seconds"]
    report["inference"] = inference
    atomic_json(args.output, report)

    # Fixed real observations and synthetic targets isolate PPO update cost.
    # This probe is not a convergence or solution-quality experiment.
    update_rows = snapshots(config, records, config["ppo"]["batch_size"], max_orders=2)
    update_obs, update_masks = map(list, zip(*update_rows))
    actions, logps, values = agent.act_batch(update_obs, update_masks)
    buffer = RolloutBuffer()
    for i, ((observation, mask), action, logp, value) in enumerate(zip(update_rows, actions, logps, values)):
        buffer.add(observation, mask, action, logp, value, 0., False)
        buffer.transitions[-1].advantage = float(np.sin(i+.5))
        buffer.transitions[-1].return_value = value + .1*float(np.cos(i+.5))
    update, losses = {}, {}
    for mode in MODES:
        agent.network.execution_mode = mode
        seconds = []
        for repeat in range(args.repeats+1):
            agent.network.load_state_dict(initial)
            agent.optimizer = torch.optim.Adam(agent.network.parameters(), lr=config["ppo"]["learning_rate"])
            torch.manual_seed(73+repeat)
            synchronize(config)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            loss = agent.update(buffer)
            synchronize(config)
            elapsed = time.perf_counter()-started
            if repeat:
                seconds.append(elapsed)
            if repeat == 1:
                losses[mode] = loss
                peak = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None
        update[mode] = {"seconds": seconds, "median_seconds": median(seconds), "peak_allocated_bytes": peak}
        print(f"update {mode}: {median(seconds):.4f}s", flush=True)
    update["speedup"] = update[MODES[0]]["median_seconds"]/update[MODES[1]]["median_seconds"]
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "optimizer_updates", "sample_presentations"):
        np.testing.assert_allclose(losses[MODES[0]][key], losses[MODES[1]][key], rtol=3e-4, atol=3e-6)
    report["update"] = update
    report["update_loss_parity"] = True
    atomic_json(args.output, report)
    agent.network.load_state_dict(initial)
    cropped = [replace(r, instance=replace(r.instance, instance_id=r.instance.instance_id+"_dcan_probe2",
                                           orders=r.instance.orders[:2])) for r in records]
    requests = [(record, pref) for record in cropped for pref in FOUR[:3]]
    deliveries, evaluation = {}, {}
    print(f"evaluation: starting {args.workers} workers", flush=True)
    with evaluation_pool(config, cropped[0].instance, workers=args.workers) as runner:
        for mode in MODES:
            agent.network.execution_mode = mode
            evaluate_requests(config, agent, requests[:1], "dcan", runner=runner, seed=11, capture_trajectories=False)
            deliveries[mode], runtime = evaluate_requests(config, agent, requests, "dcan", runner=runner,
                                                         seed=11, capture_trajectories=True)
            evaluation[mode] = runtime
            print(f"evaluation {mode}: {runtime['wall_seconds']:.4f}s", flush=True)
    compare_rows(deliveries[MODES[0]], deliveries[MODES[1]])
    assert weights_hash == network_weights_sha256(agent.network.state_dict())
    evaluation["speedup"] = evaluation[MODES[0]]["wall_seconds"]/evaluation[MODES[1]]["wall_seconds"]
    report.update(evaluation=evaluation, trajectory_parity=True, weights_unchanged=True)
    atomic_json(args.output, report)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
