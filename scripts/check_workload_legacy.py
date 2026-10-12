"""Capture or verify exact schema-10 policy behavior on fixed validation data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from agent.ppo import PPOAgent, build_actor_critic
from configs import load_config
from data import load_dataset_split
from environment import AssemblySchedulingEnv


def snapshot(run: Path) -> dict:
    torch.set_num_threads(2)
    config = load_config(run / "config.json")
    config["device"] = "cpu"
    dataset = load_dataset_split(config, "validation")
    env = AssemblySchedulingEnv(config)
    observation = env.reset(dataset[0].instance)
    agent = PPOAgent(build_actor_critic(observation, config["network"]), config["ppo"], device="cpu")
    checkpoint = run / "best_checkpoint.pt"
    agent.load(checkpoint)
    agent.assert_evaluation_config(config)
    rows = []
    for record in (dataset[0], dataset[1]):
        observation = env.reset(record.instance)
        trace = hashlib.sha256()
        actions = []
        while not env.task_done:
            trace.update(observation.global_features.tobytes())
            for name in sorted(observation.node_features):
                trace.update(name.encode())
                trace.update(observation.node_features[name].tobytes())
            for name in sorted(observation.relations):
                edge = observation.relations[name]
                trace.update(str(name).encode())
                trace.update(edge.edge_index.tobytes())
                trace.update(edge.edge_features.tobytes())
            mask = env.get_action_mask()
            trace.update(mask.tobytes())
            action, _, _ = agent.act(observation, mask, deterministic=True)
            actions.append(action)
            observation, reward, _, _, _ = env.step(action)
            trace.update(np.array([reward.flow, reward.cost, reward.variance, reward.scalarize(config["reward"])], dtype=np.float64).tobytes())
        metric = env.metrics()
        rows.append({"instance_id": record.instance.instance_id, "trace_sha256": trace.hexdigest(),
                     "actions": actions, "success": env.task_succeeded,
                     "end_time": env.current_time,
                     "metrics": {key: metric[key] for key in ("flow_time_objective", "reconfiguration_cost", "worker_load_variance")}})
    return {"checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "schema": agent.network.network_spec()["observation_schema_version"], "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="result/runs/flow_seed11_20261008_130244")
    parser.add_argument("--output", default="result/audits/workload_owa/legacy_baseline.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = snapshot(ROOT / args.run)
    path = ROOT / args.output
    if args.verify:
        assert result == json.loads(path.read_text(encoding="utf-8")), "schema10 evaluation changed"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"schema": result["schema"], "instances": len(result["rows"]), "verified": args.verify,
                      "decisions": [len(row["actions"]) for row in result["rows"]]}))


if __name__ == "__main__":
    main()
