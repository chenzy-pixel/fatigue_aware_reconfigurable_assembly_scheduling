"""Capture and verify physical behavior across the schema-9 relation change."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agent.baselines import HeuristicPolicy, RandomPolicy
from configs import load_config
from data.dataset import load_generated_record
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv
from scripts.audit_observation_reward import observation_sha256

ACTUAL_RELATIONS = {("operation", "processing_on", "machine"), ("machine", "served_by", "worker")}


def legacy_projection(obs):
    return replace(obs, relations={key: store for key, store in obs.relations.items() if key not in ACTUAL_RELATIONS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "verify"))
    parser.add_argument("--baseline", default="result/audits/schema9_behavior_reference.json")
    args = parser.parse_args()
    config = load_config("configs/default.json")
    instances = [load_instance_yaml(config["paths"]["fixed_instance"])] + [
        load_generated_record(str(Path(config["paths"]["instances_root"])/"validation"/f"instance_{2000000+i}.json")).instance for i in range(7)
    ]
    results = []
    for instance in instances:
        for mode in ("full", "neutral"):
            for name, policy in (("heuristic", HeuristicPolicy()), ("random11", RandomPolicy(11))):
                settings = deepcopy(config)
                settings["environment"]["fatigue_mode"] = mode
                env = AssemblySchedulingEnv(settings)
                env.reset(instance)
                digest = hashlib.sha256()
                rewards = []
                while not env.task_done:
                    obs = env.observe()
                    digest.update(observation_sha256(legacy_projection(obs)).encode())
                    digest.update(np.ascontiguousarray(env.get_action_mask()).tobytes())
                    action = int(policy.select_action(env))
                    digest.update(str(action).encode())
                    _, reward, _, _, _ = env.step(action, build_observation=False)
                    rewards.append(reward.as_dict())
                digest.update(observation_sha256(legacy_projection(env.observe())).encode())
                results.append({"instance": instance.instance_id, "mode": mode, "policy": name,
                                "digest": digest.hexdigest(), "rewards": rewards,
                                "reason": env.terminal_reason, "objectives": env._objective_vector(),
                                "schedule": env.schedule_log, "reconfigurations": env.reconfiguration_log})
        print(f"Checked {instance.instance_id}: {len(results)} trajectories", flush=True)
    target = Path(args.baseline)
    normalized = json.loads(json.dumps(results))
    if args.mode == "capture":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(normalized), encoding="utf-8")
    else:
        assert json.loads(target.read_text(encoding="utf-8")) == normalized, "schema-8 physical/projection regression"
    print(f"{args.mode}: {len(results)} trajectories, {sum(len(row['rewards']) for row in results)} steps matched", flush=True)


if __name__ == "__main__":
    main()
