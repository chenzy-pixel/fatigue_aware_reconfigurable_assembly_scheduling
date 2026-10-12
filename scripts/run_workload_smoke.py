"""Run five 40-episode workload integration experiments on validation data."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from configs import load_config
from data import load_dataset_split
from environment import AssemblySchedulingEnv
from agent.ppo import PPOAgent, build_actor_critic
from environment.types import proxy_return_from_metrics
from train import train

ARMS = {
    "variance": "configs/flow_excess/shared_head/single_variance.json",
    "owa_q025": "configs/workload/q025/single_workload.json",
    "owa_q05": "configs/workload/q05/single_workload.json",
    "owa_q075": "configs/workload/q075/single_workload.json",
    "universal_q05": "configs/workload/q05/universal.json",
}


def verify_run(directory: Path, indices: list[int]) -> dict:
    config = load_config(directory / "config.json")
    training_rows = pd.read_csv(directory / "train_log.csv")
    assert len(training_rows) == 40 and training_rows.episode.tolist() == list(range(1, 41))
    assert np.isfinite(training_rows[["reward", "expected_reward", "reward_identity_error"]]).all().all()
    assert (training_rows.reward_identity_error.abs() <= 1e-8).all()
    assert np.allclose(training_rows.reward, training_rows.expected_reward, atol=1e-8, rtol=0)
    assert (training_rows.maximum_worker_fatigue <= training_rows.safe_fatigue_limit + 1e-9).all()
    assert not training_rows.sampling_truncated.any()
    updates = pd.read_csv(directory / "update_log.csv")
    if len(updates) != 2:
        raise AssertionError("40 episodes must produce two 20-episode PPO updates")
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        if name in updates and not np.isfinite(updates[name]).all():
            raise AssertionError(f"nonfinite {name}")
    rows = pd.read_csv(directory / "sampled_validation_instance_metrics.csv")
    validations = pd.read_csv(directory / "validation_log.csv")
    assert validations.episode.tolist() == [20, 40]
    assert set(rows.validation_episode) == {20, 40}
    dataset = load_dataset_split(config, "validation")
    assert set(rows.instance_id) == {dataset[index].instance.instance_id for index in indices}
    for name in ("schedule_violation_count",):
        assert (rows[name] == 0).all()
    assert rows.physical_safety_pass.all() and rows.active_constraint_pass.all()
    assert not rows.sampling_truncated.any()
    failed = rows.loc[rows.task_failed]
    assert failed.termination_reason.notna().all() and (failed.termination_reason != "").all()
    checkpoint = directory / "best_checkpoint.pt"
    assert checkpoint.exists() and (directory / "last_checkpoint.pt").exists()
    env = AssemblySchedulingEnv(config)
    observation = env.reset(dataset[indices[0]].instance)
    agent = PPOAgent(build_actor_critic(observation, config["network"]), config["ppo"], device=config["device"])
    agent.load(checkpoint)
    agent.assert_evaluation_config(config)
    total = 0.
    while not env.task_done:
        action, _, _ = agent.act(observation, env.get_action_mask(), deterministic=True)
        observation, reward, _, _, _ = env.step(action)
        total += reward.scalarize(config["reward"])
    metric = env.metrics()
    assert not env.validate_schedule()
    assert np.isclose(total, proxy_return_from_metrics(metric, config), atol=1e-8)
    return {"run": str(directory.relative_to(ROOT)), "updates": len(updates), "episodes": 40,
            "validation_rows": len(rows), "validation_successes": int(rows.task_succeeded.sum()),
            "checkpoint_reload_verified": True, "deterministic_success": env.task_succeeded,
            "return_identity_verified": True, "workload_mode": config["workload_objective"]["mode"],
            "q": config["workload_objective"].get("q"), "device": config["device"],
            "training_successes": int(training_rows.task_succeeded.sum()),
            "training_terminal_reasons": training_rows.terminal_reason.value_counts().to_dict(),
            "validation_terminal_reasons": rows.termination_reason.value_counts().to_dict(),
            "maximum_training_return_identity_error": float(training_rows.reward_identity_error.abs().max())}


def audit_candidate_pools(completed: dict, output: Path) -> None:
    """Score matched single-objective samples; retain Universal's separate budget."""
    from analysis.workload_dominance import analyze_workload_pool
    from result.io import write_json
    for name, arms in (("single_objective_pool", tuple(arm for arm in ARMS if arm != "universal_q05")),
                       ("universal_pool", ("universal_q05",))):
        if any(arm not in completed for arm in arms):
            continue
        candidates, configs = [], {}
        for arm in arms:
            directory = ROOT / completed[arm]["run"]
            configs[arm] = load_config(directory / "config.json")
            with (directory / "final_sampled_instance_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                candidates.extend({**row, "method": arm} for row in csv.DictReader(stream))
        dataset = load_dataset_split(configs[arms[0]], "validation")
        rosters = {record.instance.instance_id: tuple(worker.id for worker in record.instance.workers) for record in dataset}
        summary = analyze_workload_pool(candidates, training_configs=configs, worker_rosters=rosters)
        summary["purpose"] = "integration_smoke; not evidence of convergence or q selection"
        write_json(output / f"{name}.json", summary)
        print(f"Audited {name}: {len(candidates)} candidates at frozen raw Flow / q=0.5", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="result/audits/workload_owa/smoke")
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    indices = json.loads((ROOT / "data/manifests/v2/validation/subsets.json").read_text())["target"]["instance_indices"][:5]
    selected = ARMS if args.arm == "all" else {args.arm: ARMS[args.arm]}
    completed_path = output / "runs.json"
    completed = json.loads(completed_path.read_text()) if completed_path.exists() else {}
    for arm, path in selected.items():
        if arm in completed:
            completed[arm] = verify_run(ROOT / completed[arm]["run"], indices)
            completed_path.write_text(json.dumps(completed, indent=2), encoding="utf-8")
            print(f"Verified completed {arm}", flush=True)
            continue
        config = deepcopy(load_config(path))
        print(f"Starting {arm}: 40 episodes, updates of 20, fixed validation {indices}", flush=True)
        config["device"] = args.device
        config["seed"] = 11
        training = config["training"]
        training.update(smoke_episodes=40, smoke_rollout_steps=5000, smoke_parallel_envs=4,
                        smoke_validation_instance_limit=5, validation_instance_indices=indices,
                        validation_interval_episodes=20, episodes_per_update=20,
                        validation_parallel_envs=4, torch_num_threads=2)
        training["formal_evaluation"].update(validation_repeats=1, final_test_repeats=1,
                                             final_test_lattice_denominator=1,
                                             validation_preferences=[[1,0,0],[0,1,0],[0,0,1],[.5,.3,.2]])
        directory = train(config, smoke=True, run_name=f"workload_{arm}_smoke_{datetime.now():%Y%m%d_%H%M%S}",
                          parallel_envs=4, validation_parallel_envs=4, visdom_enabled=False)
        completed[arm] = verify_run(directory, indices)
        completed_path.write_text(json.dumps(completed, indent=2), encoding="utf-8")
        print(json.dumps(completed[arm]), flush=True)
    audit_candidate_pools(completed, output)
    print(json.dumps(completed, indent=2), flush=True)


if __name__ == "__main__":
    main()
