"""Common native evaluation and exact-interaction training for learning arms."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.parallel import ParallelEpisodeRunner
from data.models import load_instance_yaml
from configs import project_path
from configs.config import public_config
from environment import AssemblySchedulingEnv, PreferenceVector
from environment.workload import metrics_workload_objective
from environment.types import metrics_flow_objective
from utils import set_seed
from result.training_history import (LOG_VERSION, EPOCH_DEFINITION, instance_row,
                                     aggregate, export_history, save_epoch)
from .common import atomic_json, digest, method_config
from .collection import collect
from .dcan import DCANActorCritic, DCANAgent
from .ddqn import RuleDDQN


def build_agent(config, obs, method):
    if method not in ("ppo", "dcan", "rule_ddqn"):
        raise ValueError(f"unknown learning method: {method}")
    if method == "rule_ddqn":
        return RuleDDQN(obs, config)
    cls = PPOAgent if method == "ppo" else DCANAgent
    network = build_actor_critic(obs, config["network"]) if method == "ppo" else DCANActorCritic(obs, config)
    return cls(network, config["ppo"], device=config["device"])


def synchronize(config):
    if torch.device(config["device"]).type == "cuda":
        torch.cuda.synchronize(torch.device(config["device"]))


def evaluate_request(config, agent, instance, pref, method, *, mode=None, seed=0):
    """A completed delivery includes all paid repeats and terminal validation."""
    sampled = (method != "rule_ddqn") if mode is None else mode == "sampled"
    repeats = 3 if sampled else 1
    synchronize(config)
    began = time.perf_counter()
    raw = []
    for repeat in range(repeats):
        env = AssemblySchedulingEnv(config)
        observation = env.reset(instance, preference=PreferenceVector(*pref))
        key = digest([seed, instance.instance_id, pref, repeat, "comparison_sampling_v1"])
        generator = torch.Generator(device=config["device"]).manual_seed(int(key[:15], 16))
        actions = []
        while not (env.terminated or env.truncated):
            mask = env.get_action_mask()
            if method == "rule_ddqn":
                action = agent.act_rules([observation], [mask])[0][0]
            else:
                action = agent.act(observation, mask, deterministic=not sampled, generator=generator)[0]
            actions.append(int(action))
            observation, _, _, _, _ = env.step(action)
        metrics = env.metrics()
        violations = env.validate_schedule()
        raw.append({"success": bool(metrics["task_succeeded"] and not violations),
                    "quality": float(metrics["preference_quality_score"]),
                    "objectives": (metrics_flow_objective(metrics, config), metrics["reconfiguration_cost"], metrics_workload_objective(metrics, config)),
                    "sampling_repeat": repeat, "actions": actions, "trace": digest(actions),
                    "metrics": metrics, "violations": violations,
                    "schedule": env.schedule_log, "reconfigurations": env.reconfiguration_log})
    selected = min(raw, key=lambda r: (not r["success"], r["quality"] if r["success"] else 1., r["sampling_repeat"]))
    synchronize(config)
    return {"summary": {"instance_id": instance.instance_id, "preference": pref, "method": method,
                         "delivery_protocol": "best_of_3_v1" if sampled else "greedy_single_v1",
                         "success": selected["success"], "quality": selected["quality"] if selected["success"] else 1.,
                         "objectives": selected["objectives"], "search_seconds": time.perf_counter()-began,
                         "candidate_count": repeats}, "raw": raw, "selected_repeat": selected["sampling_repeat"]}


def validation(config, agent, dataset, indices, prefs, method, *, seed):
    rows, deliveries = [], []
    for index in indices:
        for pref in prefs:
            result = evaluate_request(config, agent, dataset[index].instance, pref, method, seed=seed)
            deliveries.append({"summary": result["summary"], "selected_repeat": result["selected_repeat"],
                               "raw": [{k: v for k, v in sample.items() if k not in ("schedule", "reconfigurations")}
                                       for sample in result["raw"]]})
            row = {**result["summary"], "cell_id": dataset[index].metadata["cell_id"],
                   "scale": dataset[index].metadata["scale"],
                   "decision_count": next(r["metrics"]["decision_count"] for r in result["raw"] if r["sampling_repeat"] == result["selected_repeat"]),
                   "raw_safety_pass": all(not r["violations"] and r["metrics"]["maximum_worker_fatigue"] <= r["metrics"]["safe_fatigue_limit"]+1e-9 for r in result["raw"])}
            rows.append(row)
    rates, qualities = [], []
    for pref in prefs:
        group = [r for r in rows if tuple(r["preference"]) == tuple(pref)]
        rates.append(float(np.mean([r["success"] for r in group])))
        good = [r["quality"] for r in group if r["success"]]
        qualities.append(float(np.mean(good)) if good else 1.)
    safe = all(r["raw_safety_pass"] for r in rows)
    summary = {"worst_preference_success": min(rates), "quality": float(np.mean(qualities)),
               "all_preferences_covered": all(any(r["success"] and tuple(r["preference"]) == tuple(p) for r in rows) for p in prefs),
               "safe": safe, "delivery_count": len(rows), "evaluation_seconds": sum(r["search_seconds"] for r in rows)}
    # If any preference has no successful delivery, its quality ranks after all
    # covered validations (the JSON summary remains finite).
    rank = (safe, min(rates), -summary["quality"] if summary["all_preferences_covered"] else -float("inf"))
    return {"summary": summary, "rows": rows, "deliveries": deliveries}, rank


def train_request(base, method, seed, rate, steps, dataset, val_indices, prefs, output, identity, *, workers=20, smoke_instance=None):
    from data.distribution import validate_training_distribution

    validate_training_distribution(base)
    output = Path(output)
    job_id = digest([identity, method, seed, rate, steps, val_indices, prefs, workers, LOG_VERSION])
    summary_path = output / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved.get("job_id") != job_id:
            raise ValueError("completed training identity differs")
        if not (output / "best_checkpoint.pt").exists():
            raise ValueError("completed training is missing checkpoint")
        return saved
    config = method_config(base, method, seed, rate)
    # Exploration uses the formal horizon even in a shorter pilot.
    config["comparison"]["training_steps"] = steps
    from configs.runtime import runtime_manifest
    config["runtime_manifest"]=runtime_manifest(config)
    set_seed(seed)
    torch.set_num_threads(1)
    template = smoke_instance or load_instance_yaml(project_path(config["paths"]["fixed_instance"]))
    env = AssemblySchedulingEnv(config)
    obs = env.reset(template)
    agent = build_agent(config, obs, method)
    atomic_json(output / "config.json", public_config(config))
    # Interrupted training is deliberately restarted, rather than falsely
    # treating weights alone as an exact optimizer/replay/worker resume.
    atomic_json(output / "status.json", {"job_id": job_id, "status": "running", "restart_policy": "restart_incomplete_request"})
    thresholds = sorted({max(1, int(steps*i/5)) for i in range(1, 6)})
    used, episode, validation_index, updates, presentations = 0, 0, 0, 0, 0
    best_rank = None
    history, validations, episode_log = [], [], []
    training_seconds = 0.
    started = time.perf_counter()
    with ParallelEpisodeRunner(config=config, template=template, episode_count=max(steps, 20), worker_count=workers,
                               diagnostic_directory=output / "workers") as runner:
        while used < steps:
            epoch = len(history) + 1
            limit = min(steps-used, thresholds[validation_index]-used)
            rollout = collect(runner, agent, config, episode, limit, ddqn=method == "rule_ddqn",
                              episodes_per_batch=int(config["training"]["episodes_per_update"]))
            if rollout.steps <= 0:
                raise RuntimeError("training made no progress")
            used += rollout.steps
            episode += rollout.episodes_started
            losses = rollout.losses
            update_started = time.perf_counter()
            if method != "rule_ddqn" and len(rollout.buffer):
                losses = agent.update(rollout.buffer)
                updates += int(np.ceil(len(rollout.buffer)/config["ppo"]["batch_size"]))*config["ppo"]["epochs"]
                presentations += len(rollout.buffer)*config["ppo"]["epochs"]
            update_seconds = time.perf_counter()-update_started
            training_seconds += rollout.seconds+update_seconds
            for item in rollout.episodes:
                item.update(epoch=epoch, method=method, training_seed=seed, training_steps=used)
            episode_log.extend(rollout.episodes)
            flat = [instance_row(e, epoch=epoch, method=method, training_seed=seed, training_steps=used) for e in rollout.episodes]
            history.append({"epoch": epoch, "update_id": epoch, "epoch_definition": EPOCH_DEFINITION,
                            "log_schema_version": LOG_VERSION, "method": method, "training_seed": seed,
                            "job_id": job_id, "training_steps": used,
                            "steps": used, "episodes_started": episode, "batch_steps": rollout.steps,
                            "optimizer_epochs_per_batch": config["ppo"]["epochs"] if method != "rule_ddqn" else None,
                            "sample_presentations": agent.sample_presentations if method == "rule_ddqn" else presentations,
                            "update_seconds": update_seconds, "training_seconds": training_seconds,
                            "learning_rate": rate, **aggregate(flat),
                            "seconds": time.perf_counter()-started, "collect_seconds": rollout.seconds,
                            "budget_truncations": sum(e["budget_truncated"] for e in rollout.episodes),
                            "losses": losses, "optimizer_updates": agent.updates if method == "rule_ddqn" else updates})
            save_epoch(output, history[-1], rollout.episodes)
            atomic_json(output / "training_log.json", history)
            atomic_json(output / "episodes.json", episode_log)
            export_history(output, history, episode_log)
            if used == thresholds[validation_index]:
                result, rank = validation(config, agent, dataset, val_indices, prefs, method, seed=seed)
                result.update(training_steps=used, epoch=epoch, method=method, training_seed=seed)
                validations.append(result)
                metadata = {"algorithm": method, "job_id": job_id, "training_steps": used, "seed": seed, "epoch": epoch,
                            "validation": result["summary"], "comparison_identity": identity}
                if result["summary"]["safe"] and (best_rank is None or rank > best_rank):
                    best_rank = rank
                    agent.save(output / "best_checkpoint.pt", metadata)
                    atomic_json(output / "best_validation.json", result)
                agent.save(output / "last_checkpoint.pt", metadata)
                atomic_json(output / "validations.json", validations)
                export_history(output, history, episode_log, validations)
                validation_index += 1
                print(f"{method} seed={seed} steps={used}/{steps} validation={result['summary']}", flush=True)
    if best_rank is None:
        raise RuntimeError("no physically safe eligible checkpoint")
    tail = validations[-2:]
    training_sufficiency = "not_established" if len(tail) < 2 or (
        tail[1]["summary"]["worst_preference_success"] > tail[0]["summary"]["worst_preference_success"]+1e-12 or
        tail[0]["summary"]["quality"]-tail[1]["summary"]["quality"] > .002) else "no_large_gain_in_last_interval"
    summary = {"job_id": job_id, "method": method, "seed": seed, "learning_rate": rate,
               "training_steps": used, "epochs": len(history), "epoch_definition": EPOCH_DEFINITION,
               "log_schema_version": LOG_VERSION, "episodes_started": episode, "validation_count": len(validations),
               "optimizer_updates": agent.updates if method == "rule_ddqn" else updates,
               "sample_presentations": agent.sample_presentations if method == "rule_ddqn" else presentations,
               "seconds": time.perf_counter()-started, "training_sufficiency": training_sufficiency,
               "training_seconds": training_seconds,
               "validation_seconds": sum(v["summary"]["evaluation_seconds"] for v in validations),
               "best_validation": json.loads((output/"best_validation.json").read_text(encoding="utf-8"))["summary"]}
    atomic_json(summary_path, summary)
    atomic_json(output / "status.json", {"job_id": job_id, "status": "complete"})
    return summary
