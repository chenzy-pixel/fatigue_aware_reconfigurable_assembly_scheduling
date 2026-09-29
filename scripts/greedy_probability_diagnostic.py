"""Replay a PPO checkpoint and inspect production-action probability rankings.

Example:
    python -m scripts.greedy_probability_diagnostic \
        --run result/runs/flow_failure_v2_seed11_500
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from agent.ppo import PPOAgent, build_actor_critic
from configs import project_path
from data import load_dataset_split
from environment import AssemblySchedulingEnv
from utils import action_trace_sha256, set_seed


def _read_csv_by_instance(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["instance_id"]: row for row in csv.DictReader(handle)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("diagnostic produced no rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _production_row(env: AssemblySchedulingEnv, probabilities: np.ndarray, mask: np.ndarray, *,
                    instance_id: str, decision: int, action: int) -> dict:
    legal = np.flatnonzero(~mask)
    classes = {
        "COMMIT_RECONFIG": [],
        "DIRECT_PROCESS": [],
        "WAIT": [],
    }
    for index in legal:
        classes[env._action_type(env.decision_type, int(index))].append(int(index))
    rec = classes["COMMIT_RECONFIG"]
    direct = classes["DIRECT_PROCESS"]
    wait = classes["WAIT"]
    ranking = sorted((int(index) for index in legal), key=lambda index: (-probabilities[index], index))
    top_rec = min(rec, key=lambda index: (-probabilities[index], index)) if rec else None
    top_direct = min(direct, key=lambda index: (-probabilities[index], index)) if direct else None
    remaining = [operation for operation in env.operations if operation.state.value != "DONE"]
    installed = {machine.current_module for machine in env.machines}
    active_targets = {machine.target_module for machine in env.machines if machine.target_module}
    absent = {operation.spec.required_module for operation in remaining} - installed - active_targets
    return {
        "instance_id": instance_id,
        "decision": decision,
        "time": float(env.current_time),
        "selected_action": action,
        "selected_type": env._action_type(env.decision_type, action),
        "selected_probability": float(probabilities[action]),
        "legal_reconfig_count": len(rec),
        "legal_direct_count": len(direct),
        "legal_wait_count": len(wait),
        "reconfig_probability_mass": float(probabilities[rec].sum()) if rec else 0.0,
        "direct_probability_mass": float(probabilities[direct].sum()) if direct else 0.0,
        "wait_probability": float(probabilities[wait[0]]) if wait else 0.0,
        "top_reconfig_action": top_rec if top_rec is not None else "",
        "top_reconfig_probability": float(probabilities[top_rec]) if top_rec is not None else 0.0,
        "top_reconfig_rank": ranking.index(top_rec) + 1 if top_rec is not None else "",
        "top_direct_action": top_direct if top_direct is not None else "",
        "top_direct_probability": float(probabilities[top_direct]) if top_direct is not None else 0.0,
        "reconfig_mass_exceeds_winner": bool(rec and probabilities[rec].sum() > probabilities[action]),
        "reconfig_mass_exceeds_direct_mass": bool(rec and probabilities[rec].sum() > probabilities[direct].sum()),
        "remaining_operation_count": len(remaining),
        "remaining_modules_absent_from_fleet": len(absent),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed training run directory")
    parser.add_argument("--output", help="Diagnostic output directory")
    args = parser.parse_args()

    run = project_path(args.run)
    output = project_path(args.output) if args.output else project_path("result/analysis") / run.name
    output.mkdir(parents=True, exist_ok=True)
    with (run / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    set_seed(int(config["seed"]))
    torch.set_num_threads(int(config["training"].get("torch_num_threads", 4)))
    dataset = load_dataset_split(config, "test")
    reference = _read_csv_by_instance(run / "final_greedy_instance_metrics.csv")
    if len(dataset) != len(reference):
        raise ValueError("test dataset and original Greedy evaluation have different sizes")
    bootstrap = AssemblySchedulingEnv(config)
    first_observation = bootstrap.reset(dataset[0].instance)
    network = build_actor_critic(first_observation, config["network"])
    agent = PPOAgent(network, config["ppo"], device=config["device"])
    agent.load(run / "best_checkpoint.pt")
    agent.network.eval()

    decisions: list[dict] = []
    episode_rows: list[dict] = []
    with torch.no_grad():
        for record in dataset:
            env = AssemblySchedulingEnv(config)
            observation = env.reset(record.instance)
            actions: list[int] = []
            production_rows: list[dict] = []
            while not (env.terminated or env.truncated):
                mask = env.get_action_mask()
                logits, _ = agent.network.forward(observation, mask, device=agent.device)
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                action = int(torch.argmax(logits).item())
                if env.decision_type.value == "PRODUCTION":
                    row = _production_row(env, probabilities, mask, instance_id=record.instance.instance_id,
                                          decision=len(actions), action=action)
                    decisions.append(row)
                    production_rows.append(row)
                actions.append(action)
                observation, _, _, _, _ = env.step(action)

            metrics = env.metrics()
            original = reference[record.instance.instance_id]
            actual_hash = action_trace_sha256(actions)
            if actual_hash != original["action_trace_sha256"]:
                raise AssertionError(f"Greedy action trace differs for {record.instance.instance_id}")
            if int(metrics["commit_reconfig_action_count"]) != int(original["commit_reconfig_action_count"]):
                raise AssertionError(f"Greedy reconfiguration count differs for {record.instance.instance_id}")
            opportunities = [row for row in production_rows if row["legal_reconfig_count"]]
            missed = [row for row in opportunities if row["selected_type"] != "COMMIT_RECONFIG"]
            episode_rows.append({
                "instance_id": record.instance.instance_id,
                "terminated": metrics["terminated"],
                "truncated": metrics["truncated"],
                "unfinished_orders": metrics["unfinished_orders"],
                "decisions": len(actions),
                "commit_reconfig_action_count": metrics["commit_reconfig_action_count"],
                "production_decision_count": len(production_rows),
                "reconfig_opportunity_count": len(opportunities),
                "missed_reconfig_opportunity_count": len(missed),
                "missed_with_reconfig_mass_above_winner": sum(row["reconfig_mass_exceeds_winner"] for row in missed),
                "missed_with_absent_required_module": sum(row["remaining_modules_absent_from_fleet"] > 0 for row in missed),
                "action_trace_sha256": actual_hash,
            })
            print(f"{record.instance.instance_id}: reconfig={metrics['commit_reconfig_action_count']}, "
                  f"opportunities={len(opportunities)}, missed={len(missed)}, "
                  f"mass_above_winner={episode_rows[-1]['missed_with_reconfig_mass_above_winner']}", flush=True)

    _write_csv(output / "greedy_probability_decisions.csv", decisions)
    _write_csv(output / "greedy_probability_instances.csv", episode_rows)
    opportunities = [row for row in decisions if row["legal_reconfig_count"]]
    missed = [row for row in opportunities if row["selected_type"] != "COMMIT_RECONFIG"]
    summary = {
        "run": str(run.resolve()),
        "checkpoint": str((run / "best_checkpoint.pt").resolve()),
        "dataset": "test",
        "instance_count": len(episode_rows),
        "trace_hashes_matched": len(episode_rows),
        "completion_count": sum(row["terminated"] and not row["truncated"] for row in episode_rows),
        "reconfig_action_count": sum(row["commit_reconfig_action_count"] for row in episode_rows),
        "production_decision_count": len(decisions),
        "reconfig_opportunity_count": len(opportunities),
        "missed_reconfig_opportunity_count": len(missed),
        "missed_winner_types": dict(Counter(row["selected_type"] for row in missed)),
        "missed_with_reconfig_mass_above_winner": sum(row["reconfig_mass_exceeds_winner"] for row in missed),
        "missed_with_reconfig_mass_above_direct_mass": sum(row["reconfig_mass_exceeds_direct_mass"] for row in missed),
        "missed_with_absent_required_module": sum(row["remaining_modules_absent_from_fleet"] > 0 for row in missed),
        "missed_mean_reconfig_probability_mass": float(np.mean([row["reconfig_probability_mass"] for row in missed])) if missed else None,
        "missed_mean_top_reconfig_probability": float(np.mean([row["top_reconfig_probability"] for row in missed])) if missed else None,
        "missed_mean_top_reconfig_rank": float(np.mean([row["top_reconfig_rank"] for row in missed])) if missed else None,
    }
    with (output / "greedy_probability_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
