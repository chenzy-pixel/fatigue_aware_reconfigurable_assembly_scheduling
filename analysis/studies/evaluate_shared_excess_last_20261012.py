"""Replay archived best controls, then evaluate the saved last on the formal test grid."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUN = ROOT / "result/runs/remaining_flow_experiments_seed11_20261008_225956_shared_head_universal"
OUT = ROOT / "result/analysis/shared_excess_last_20261012"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--reuse-probe", action="store_true", help="Reuse the controls just replayed in this investigation")
    parser.add_argument("--resume", action="store_true", help="Continue saved complete test chunks")
    args = parser.parse_args()
    import pandas as pd
    import torch
    from agent.ppo import PPOAgent, build_actor_critic
    from agent.ppo.parallel import ParallelEpisodeRunner
    from configs import load_config, project_path
    from configs.formal_preferences import formal_preferences
    from data import load_dataset_split, load_instance_yaml
    from data.selection import subset_snapshot
    from environment import AssemblySchedulingEnv, PreferenceContext
    from eval import evaluate_preference_grid_parallel
    from result.io import write_csv, write_json
    from result.provenance import build_provenance, network_weights_sha256
    from train import _aggregate_formal_rows
    from utils import configured_formal_evaluation_sampling_seeds, set_seed

    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(RUN / "config.json")
    config["training"]["validation_parallel_envs"] = args.workers
    torch.set_num_threads(4)
    set_seed(11)
    dataset = load_dataset_split(config, "test")
    bootstrap = AssemblySchedulingEnv(config).reset(dataset[0].instance)
    agent = PPOAgent(build_actor_critic(bootstrap, config["network"]), config["ppo"], device=config["device"])
    original = pd.read_csv(RUN / "final_sampled_instance_metrics.csv").set_index(["instance_id", "preference_key", "sampling_repeat"])
    seeds = configured_formal_evaluation_sampling_seeds(config, "final_test")
    full_selection = subset_snapshot(dataset, list(range(len(dataset))), role="evaluation")
    start = time.perf_counter()
    with ParallelEpisodeRunner(config=config, template=load_instance_yaml(project_path(config["paths"]["fixed_instance"])),
            episode_count=20, worker_count=args.workers) as runner:
        best_metadata = agent.load(RUN / "best_checkpoint.pt", load_optimizer=False)
        if args.reuse_probe:
            probes = pd.read_csv(OUT / "best_replay_rows.csv").to_dict("records")
        else:
            probes, _ = evaluate_preference_grid_parallel(config, dataset_name="test", ppo_agent=agent,
                runner=runner, instance_indices=[0, 9],
                preferences=tuple(PreferenceContext.from_input(p) for p in ((1,0,0), (0,1,0), (0,0,1))),
                sampling_seed=seeds[0])
        checks = []
        for row in probes:
            old = original.loc[(row["instance_id"], row["preference_key"], 0)]
            checks.append({"instance_id": row["instance_id"], "preference_key": row["preference_key"],
                "action_hash_match": row["action_trace_sha256"] == old.action_trace_sha256,
                "success_match": bool(row["task_succeeded"] == old.task_succeeded),
                "maximum_objective_error": max(abs(float(row[f])-float(old[f])) for f in
                    ("flow_time_objective", "flow_excess_objective", "reconfiguration_cost", "worker_load_variance"))})
        write_csv(OUT / "best_replay_checks.csv", checks)
        write_csv(OUT / "best_replay_rows.csv", probes)
        passed = all(c["action_hash_match"] and c["success_match"] and c["maximum_objective_error"] < 1e-7 for c in checks)
        write_json(OUT / "replay_probe.json", {"passed": passed, "controls": checks, "wall_seconds": time.perf_counter()-start,
            "best_checkpoint_episode": best_metadata["checkpoint_episode"]})
        print(json.dumps({"stage": "best_replay_probe", "passed": passed, "wall_seconds": time.perf_counter()-start}), flush=True)
        if not passed:
            raise RuntimeError("Archived best controls did not reproduce; inspect the saved probe before evaluating last")
        if args.probe_only:
            return
        metadata = agent.load(RUN / "last_checkpoint.pt", load_optimizer=False)
        assert metadata["checkpoint_episode"] == 2000
        assert network_weights_sha256(agent.network.state_dict()) == metadata["network_weights_sha256"]
        write_json(OUT / "last_evaluation_config.json", config)
        rows = []
        previous_elapsed = 0.0
        if args.resume:
            import numpy as np
            saved = pd.read_csv(OUT / "last_test_instance_metrics.csv")
            assert not saved.duplicated(["instance_id", "preference_key", "sampling_repeat"]).any()
            assert set(saved.actor_head_variant) == {"shared_preference"}
            assert set(saved.algorithm_seed) == {11}
            required_numbers = ["flow_time_objective", "flow_excess_objective", "reconfiguration_cost",
                "worker_load_variance", "preference_quality_score", "training_cumulative_reward",
                "base_cumulative_reward", "maximum_worker_fatigue", "operation_progress",
                "preference_flow", "preference_cost", "preference_variance"]
            assert np.isfinite(saved[required_numbers].to_numpy(dtype=float)).all()
            # CSV blanks represent optional None fields, not failed numeric observations.
            rows = saved.astype(object).where(pd.notna(saved), None).to_dict("records")
            previous_elapsed = json.loads((OUT/"evaluation_progress.json").read_text())["elapsed_seconds"]
        seen={(r["instance_id"],r["preference_key"],int(r["sampling_repeat"])) for r in rows}
        grid = formal_preferences(config, "final_test")
        eval_start = time.perf_counter()
        for repeat, seed in enumerate(seeds):
            for first in range(0, len(dataset), 2):
                indices = list(range(first, min(first+2, len(dataset))))
                cells={(dataset[i].instance.instance_id,p.key,repeat) for i in indices for p in grid}
                if cells <= seen:
                    continue
                assert not cells.intersection(seen), "Resume only accepts fully saved chunks"
                block_start = time.perf_counter()
                current, summary = evaluate_preference_grid_parallel(config, dataset_name="test", ppo_agent=agent,
                    runner=runner, instance_indices=indices, preferences=grid, sampling_seed=seed)
                for row in current:
                    row["sampling_repeat"] = repeat
                    row["evaluation_chunk_subset_sha256"] = row["subset_sha256"]
                    row["subset_sha256"] = full_selection["subset_sha256"]
                rows.extend(current)
                seen.update(cells)
                write_csv(OUT / "last_test_instance_metrics.csv", rows)
                progress = {"stage": "last_formal_test", "repeat": repeat, "last_instance_index": indices[-1],
                    "completed_trajectories": len(rows), "total_trajectories": len(dataset)*len(grid)*len(seeds),
                    "safe_successes": sum(bool(r["task_succeeded"]) for r in rows),
                    "elapsed_seconds": previous_elapsed+time.perf_counter()-eval_start, "block_seconds": time.perf_counter()-block_start}
                write_json(OUT / "evaluation_progress.json", progress)
                print(json.dumps(progress), flush=True)
        aggregate = _aggregate_formal_rows(config, rows=rows, dataset_name="test", manifest=str(dataset.manifest_path),
            unique_instance_count=len(dataset), repeat_count=len(seeds), universal=True, stage="final_test", strict_counts=True)
        aggregate["checkpoint_episode"] = metadata["checkpoint_episode"]
        aggregate["wall_seconds"] = previous_elapsed+time.perf_counter()-eval_start
        aggregate["provenance"] = build_provenance(config, dataset_manifest_path=dataset.manifest_path,
            checkpoint_path=RUN/"last_checkpoint.pt", checkpoint_metadata=metadata,
            executed_network_state=agent.network.state_dict(), formal_evaluation_stage="final_test",
            evaluation_subset_sha256=full_selection["subset_sha256"])
        # Preserve chunk identities in a dedicated field while stamping the full evaluation matrix.
        aggregate["evaluation_chunking"] = {"instances_per_chunk": 2, "workers": args.workers,
            "instance_ids": [r.instance.instance_id for r in dataset], "sampling_seeds": seeds,
            "control_action_hashes_reproduced": len(checks)}
        write_json(OUT / "last_test_metrics.json", aggregate)
        write_json(OUT / "evaluation_progress.json", {"stage": "completed", "completed_trajectories": len(rows),
            "safe_successes": sum(bool(r["task_succeeded"]) for r in rows), "elapsed_seconds": aggregate["wall_seconds"]})
        print(json.dumps({"stage": "completed", "trajectories": len(rows), "wall_seconds": aggregate["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
