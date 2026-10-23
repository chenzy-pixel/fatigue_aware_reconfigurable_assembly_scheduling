"""Replay the existing sampled validation protocol, saving complete scalar traces.

Uses the unchanged ParallelEpisodeRunner evaluator. Instrumentation observes
act_batch outputs and worker responses, with no extra inference or RNG draws.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_RUN = ROOT / "result/runs/remaining_flow_experiments_seed11_20261008_225956_shared_head_universal"
DEFAULT_OUT = ROOT / "result/analysis/gae_lambda_v4_20261012"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--roles", nargs="+", default=["last", "best"], choices=["last", "best"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from agent.ppo import PPOAgent, build_actor_critic
    from agent.ppo.parallel import ParallelEpisodeRunner
    from analysis.gae_lambda_diagnostic import aggregate, diagnostic_gate, LAMBDAS
    from configs import load_config, project_path
    from configs.config import public_config
    from data import load_dataset_split, load_instance_yaml
    from environment import AssemblySchedulingEnv, PreferenceContext
    from result.io import write_csv, write_json
    from result.provenance import effective_config_snapshot
    from utils import action_trace_sha256, configured_formal_evaluation_sampling_seeds, set_seed

    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.run / "config.json")
    torch.set_num_threads(int(config["training"]["torch_num_threads"]))
    set_seed(int(config["seed"]))
    dataset = load_dataset_split(config, "validation")
    indices = json.loads((args.run / "validation_subsets.json").read_text())["target"]["instance_indices"]
    if args.instance_limit:
        # Short smoke probes are explicitly separate from the complete diagnostic.
        indices = indices[:args.instance_limit]
    seeds = configured_formal_evaluation_sampling_seeds(config, "validation")[:args.repeats]
    if len(seeds) != args.repeats:
        raise ValueError("requested repeats exceed the archived protocol")
    regions = {"flow": (1, 0, 0), "cost": (0, 1, 0), "variance": (0, 0, 1), "center": (1/3, 1/3, 1/3)}
    prefs = {k: PreferenceContext.from_input(p) for k, p in regions.items()}
    pref_regions = {p.key: k for k, p in prefs.items()}
    records = [dataset[i] for i in indices for _ in regions]
    preferences = [p for _ in indices for p in prefs.values()]
    archive_path = args.run / "sampled_validation_instance_metrics.csv"
    archive = pd.read_csv(archive_path).set_index(["validation_episode", "instance_id", "preference_key", "sampling_repeat"])
    identity = {"source_run": str(args.run.resolve()), "source_config_sha256": sha256(args.run / "config.json"),
                "archive_sha256": sha256(archive_path), "dataset_manifest_sha256": sha256(dataset.manifest_path),
                "effective_config_sha256": effective_config_snapshot(config)["sha256"],
                "checkpoints": {role: sha256(args.run / (role + "_checkpoint.pt")) for role in args.roles},
                "instance_indices": indices, "sampling_seeds": seeds, "regions": regions, "lambdas": LAMBDAS,
                "gamma": config["ppo"]["gamma"], "full_episode_only": True,
                "gate": {"minimum_sign_agreement": 0.95, "maximum_return_rmse": 0.05}}
    provenance_path = args.output / "provenance.json"
    if provenance_path.exists():
        if not args.resume or json.loads(provenance_path.read_text()) != json.loads(json.dumps(identity)):
            raise ValueError("output already exists or resume provenance does not match")
    else:
        write_json(provenance_path, identity)
        write_json(args.output / "frozen_source_config.json", public_config(config))
    bootstrap = AssemblySchedulingEnv(config).reset(dataset[indices[0]].instance)
    agent = PPOAgent(build_actor_critic(bootstrap, config["network"]), config["ppo"], device=config["device"])
    all_episodes = []
    start = time.perf_counter()
    with ParallelEpisodeRunner(config=config, template=load_instance_yaml(project_path(config["paths"]["fixed_instance"])),
                               episode_count=20, worker_count=args.workers) as runner:
        original_act, original_exchange = agent.act_batch, runner._exchange
        for role in args.roles:
            metadata = agent.load(args.run / (role + "_checkpoint.pt"), load_optimizer=False)
            agent.network.eval()
            for repeat, seed in enumerate(seeds):
                chunk = args.output / f"{role}_repeat{repeat}.json"
                if chunk.exists() and args.resume:
                    loaded = json.loads(chunk.read_text())
                    if len(loaded) != len(records):
                        raise ValueError("resume chunk is incomplete")
                    for episode in loaded:
                        episode.update({k: np.asarray(episode[k]) for k in ("rewards", "values", "actions", "legal_counts")})
                    all_episodes.extend(loaded)
                    continue
                active, pending, complete = {}, {}, []
                def capture_act(observations, masks, **kwargs):
                    result = original_act(observations, masks, **kwargs)
                    pending["inference"] = (result, [int((~mask).sum()) for mask in masks])
                    return result

                def capture_exchange(commands):
                    for lane, (command, request) in commands.items():
                        if command == "reset_instance":
                            iid = request.value.instance_id
                            pref = PreferenceContext.from_input(request.preference)
                            region = pref_regions[pref.key]
                            active[lane] = {"role": role, "region": region, "gamma": float(config["ppo"]["gamma"]),
                                "rewards": [], "values": [], "actions": [], "legal_counts": [],
                                "metadata": {"role": role, "region": region, "instance_id": iid,
                                    "preference_key": pref.key, "repeat": repeat, "sampling_seed": int(seed),
                                    "checkpoint_episode": metadata["checkpoint_episode"]}}
                    steps = sorted(lane for lane, (command, _) in commands.items() if command == "step")
                    if steps:
                        (actions, _, values), counts = pending.pop("inference")
                        if len(steps) != len(actions):
                            raise RuntimeError("inference and worker lanes do not align")
                        for lane, action, value, count in zip(steps, actions, values, counts):
                            if int(commands[lane][1]) != int(action):
                                raise RuntimeError("captured action differs from executed action")
                            episode = active[lane]
                            episode["values"].append(float(value)); episode["actions"].append(int(action))
                            episode["legal_counts"].append(count)
                    responses = original_exchange(commands)
                    for lane in steps:
                        response, episode = responses[lane], active[lane]
                        episode["rewards"].append(float(response.reward_vector.scalarize(config["reward"])))
                        if not (response.terminated or response.truncated):
                            continue
                        metrics = response.metrics
                        if metrics is None or not response.terminated:
                            raise RuntimeError("only actual terminal episodes qualify; sampler truncations are rejected")
                        old = archive.loc[(metadata["checkpoint_episode"], episode["metadata"]["instance_id"],
                                           episode["metadata"]["preference_key"], repeat)]
                        digest = action_trace_sha256(episode["actions"])
                        if digest != old.action_trace_sha256:
                            raise RuntimeError("archived action hash did not reproduce: " + str(episode["metadata"]))
                        if bool(metrics["task_succeeded"]) != bool(old.task_succeeded):
                            raise RuntimeError("archived success did not reproduce")
                        error = abs(sum(episode["rewards"]) - float(old.training_cumulative_reward))
                        if error > 1e-7:
                            raise RuntimeError(f"archived cumulative reward differs by {error}")
                        episode["success"] = bool(metrics["task_succeeded"])
                        episode["metadata"].update(task_succeeded=episode["success"], action_hash_match=True,
                            action_trace_sha256=digest, cumulative_reward_error=error,
                            terminal_reason=metrics.get("terminal_reason", metrics.get("termination_reason")),
                            pressure_type=old.pressure_type, total_reward=float(sum(episode["rewards"])))
                        complete.append(deepcopy(episode)); del active[lane]
                        if len(complete) % 20 == 0:
                            progress = {"role": role, "repeat": repeat, "complete_in_chunk": len(complete),
                                        "expected_chunk": len(records), "elapsed_seconds": time.perf_counter()-start}
                            write_json(args.output / "progress.json", progress)
                            print(json.dumps(progress), flush=True)
                    return responses

                agent.act_batch, runner._exchange = capture_act, capture_exchange
                try:
                    results = runner.evaluate_records(agent, records, preferences=preferences, deterministic=False, sampling_seed=seed)
                finally:
                    agent.act_batch, runner._exchange = original_act, original_exchange
                if active or len(complete) != len(records) or len(results) != len(records):
                    raise RuntimeError("complete episode accounting differs")
                complete.sort(key=lambda e: (e["metadata"]["instance_id"], e["region"]))
                write_json(chunk, complete)
                for episode in complete:
                    episode.update({k: np.asarray(episode[k]) for k in ("rewards", "values", "actions", "legal_counts")})
                all_episodes.extend(complete)
    rows, pairs, trajectories = aggregate(all_episodes)
    write_csv(args.output / "gae_metrics.csv", rows)
    write_csv(args.output / "sign_agreement.csv", pairs)
    write_csv(args.output / "trajectory_metrics.csv", trajectories)
    gate = diagnostic_gate(rows, pairs)
    write_json(args.output / "decision.json", gate)
    write_json(args.output / "progress.json", {"stage": "completed", "complete_trajectories": len(all_episodes),
               "steps": sum(len(e["values"]) for e in all_episodes), "elapsed_seconds": time.perf_counter()-start,
               "all_action_hashes_match": True, "requires_training": gate["requires_training"]})
    print(json.dumps({"stage": "completed", "trajectories": len(all_episodes), "requires_training": gate["requires_training"]}), flush=True)


if __name__ == "__main__":
    main()
