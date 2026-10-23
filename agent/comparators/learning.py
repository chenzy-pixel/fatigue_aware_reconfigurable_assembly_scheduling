"""Native evaluation and explicit episode/interaction training requests."""
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
from .training_budget import training_budget
from .a2c import A2CAgent


def build_agent(config, obs, method):
    if method not in ("ppo", "a2c", "dcan", "rule_ddqn"):
        raise ValueError(f"unknown learning method: {method}")
    if method == "rule_ddqn":
        return RuleDDQN(obs, config)
    cls = {'ppo': PPOAgent, 'a2c': A2CAgent, 'dcan': DCANAgent}[method]
    network = build_actor_critic(obs, config["network"]) if method in ('ppo', 'a2c') else DCANActorCritic(obs, config)
    return cls(network, config["ppo"], device=config["device"])


def synchronize(config):
    if torch.device(config["device"]).type == "cuda":
        torch.cuda.synchronize(torch.device(config["device"]))


def evaluate_request(config, agent, instance, pref, method, *, mode=None, seed=0):
    """A completed delivery includes all paid repeats and terminal validation."""
    sampled = (method != "rule_ddqn") if mode is None else mode in ('sampled', 'sampled_single')
    repeats = 1 if mode == 'sampled_single' else 3 if sampled else 1
    synchronize(config)
    began = time.perf_counter()
    raw = []
    for repeat in range(repeats):
        repeat_started = time.perf_counter()
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
        synchronize(config)
        raw.append({"success": bool(metrics["task_succeeded"] and not violations),
                    "quality": float(metrics["preference_quality_score"]),
                    "objectives": (metrics_flow_objective(metrics, config), metrics["reconfiguration_cost"], metrics_workload_objective(metrics, config)),
                    "sampling_repeat": repeat, "actions": actions, "trace": digest(actions),
                    "metrics": metrics, "violations": violations,
                    "schedule": env.schedule_log, "reconfigurations": env.reconfiguration_log,
                    "seconds": time.perf_counter()-repeat_started})
    selected = min(raw, key=lambda r: (not r["success"], r["quality"] if r["success"] else 1., r["sampling_repeat"]))
    synchronize(config)
    return {"summary": {"instance_id": instance.instance_id, "preference": pref, "method": method,
                         "delivery_protocol": "sampled_single_v1" if mode == 'sampled_single' else "best_of_3_v1" if sampled else "greedy_single_v1",
                         "success": selected["success"], "quality": selected["quality"] if selected["success"] else 1.,
                         "objectives": selected["objectives"], "search_seconds": time.perf_counter()-began,
                         "candidate_count": repeats,
                         "environment_steps": sum(len(r['actions']) for r in raw)}, "raw": raw, "selected_repeat": selected["sampling_repeat"]}


def validation(config, agent, dataset, indices, prefs, method, *, seed):
    began = time.perf_counter()
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
               "safe": safe, "delivery_count": len(rows), "evaluation_seconds": sum(r["search_seconds"] for r in rows),
               "validation_seconds": time.perf_counter()-began,
               "validation_env_steps": sum(r['environment_steps'] for r in rows)}
    # If any preference has no successful delivery, its quality ranks after all
    # covered validations (the JSON summary remains finite).
    rank = (safe, min(rates), -summary["quality"] if summary["all_preferences_covered"] else -float("inf"))
    return {"summary": summary, "rows": rows, "deliveries": deliveries}, rank


def train_request(base, method, seed, rate, steps, dataset, val_indices, prefs, output, identity, *, workers=20, smoke_instance=None, episodes=None):
    from data.distribution import validate_training_distribution

    validate_training_distribution(base)
    if (steps is None) == (episodes is None):
        raise ValueError("exactly one of steps and episodes is required")
    budget = training_budget(base, steps=steps, episodes=episodes)
    batch_size = budget['episodes_per_update']
    output = Path(output)
    job_id = digest([identity, method, seed, rate, budget, val_indices, prefs, workers, LOG_VERSION])
    summary_path = output / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved.get("job_id") != job_id:
            raise ValueError("completed training identity differs")
        if not (output / "best_checkpoint.pt").exists():
            raise ValueError("completed training is missing checkpoint")
        return saved
    status_path = output / 'status.json'
    if status_path.exists() and json.loads(status_path.read_text(encoding='utf-8')).get('job_id') != job_id:
        raise ValueError("incomplete training identity differs; use a new output directory")
    config = method_config(base, method, seed, rate)
    config['training']['budget_unit'] = budget['budget_unit']
    if episodes is not None:
        config['training']['episodes'] = episodes
    config["comparison"].pop('training_steps', None)
    config["comparison"]["training_budget"] = budget
    if steps is not None:
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
    thresholds = sorted({max(1, int(steps*i/5)) for i in range(1, 6)}) if steps is not None else []
    used, episode, validation_index, updates, presentations = 0, 0, 0, 0, 0
    best_rank = None
    history, validations, episode_log = [], [], []
    training_seconds = validation_seconds = validation_env_steps = 0
    started = time.perf_counter()
    with ParallelEpisodeRunner(config=config, template=template, episode_count=episodes if episodes is not None else max(steps, batch_size), worker_count=workers,
                               diagnostic_directory=output / "workers") as runner:
        while (episode < episodes if episodes is not None else used < steps):
            epoch = len(history) + 1
            limit = min(steps-used, thresholds[validation_index]-used) if steps is not None else None
            count = min(batch_size, episodes-episode) if episodes is not None else batch_size
            if method == 'rule_ddqn' and episodes is not None:
                agent.set_episode_progress(episode, episodes)
            rollout = collect(runner, agent, config, episode, limit, ddqn=method == "rule_ddqn",
                              episodes_per_batch=count)
            if rollout.steps <= 0:
                raise RuntimeError("training made no progress")
            if episodes is not None and (rollout.episodes_started != count or len(rollout.episodes) != count
                                         or any(e['budget_truncated'] for e in rollout.episodes)):
                raise RuntimeError("episode batch ended before all attempts finished")
            used += rollout.steps
            episode += rollout.episodes_started
            losses = rollout.losses
            update_started = time.perf_counter()
            if method != "rule_ddqn" and len(rollout.buffer):
                losses = agent.update(rollout.buffer)
                updates += int(losses.get('optimizer_updates', 1))
                presentations += int(losses.get('sample_presentations', len(rollout.buffer)))
            update_seconds = time.perf_counter()-update_started
            training_seconds += rollout.seconds+update_seconds
            for item in rollout.episodes:
                item.update(epoch=epoch, method=method, training_seed=seed, training_steps=used)
            episode_log.extend(rollout.episodes)
            flat = [instance_row(e, epoch=epoch, method=method, training_seed=seed, training_steps=used) for e in rollout.episodes]
            history.append({"epoch": epoch, "update_id": epoch, "epoch_definition": EPOCH_DEFINITION,
                            "log_schema_version": LOG_VERSION, "method": method, "training_seed": seed,
                            "job_id": job_id, "training_budget": budget, "training_steps": used,
                            "steps": used, "episodes_started": episode, "batch_steps": rollout.steps,
                            "optimizer_epochs_per_batch": 1 if method == 'a2c' else config["ppo"]["epochs"] if method != "rule_ddqn" else None,
                            "sample_presentations": agent.sample_presentations if method == "rule_ddqn" else presentations,
                            "update_seconds": update_seconds, "training_seconds": training_seconds,
                            "validation_env_steps": validation_env_steps, "validation_seconds": validation_seconds,
                            "learning_rate": rate, **aggregate(flat),
                            "seconds": time.perf_counter()-started, "collect_seconds": rollout.seconds,
                            "budget_truncations": sum(e["budget_truncated"] for e in rollout.episodes),
                            "losses": losses, "optimizer_updates": agent.updates if method == "rule_ddqn" else updates})
            if method == 'rule_ddqn':
                history[-1]['exploration_epsilon'] = agent.exploration_epsilon
            save_epoch(output, history[-1], rollout.episodes)
            atomic_json(output / "training_log.json", history)
            atomic_json(output / "episodes.json", episode_log)
            export_history(output, history, episode_log, validations)
            due = (epoch % budget['validation_interval_epochs'] == 0 or episode == episodes) if episodes is not None else used == thresholds[validation_index]
            if due:
                result, rank = validation(config, agent, dataset, val_indices, prefs, method, seed=seed)
                validation_env_steps += result['summary']['validation_env_steps']
                validation_seconds += result['summary']['validation_seconds']
                result.update(training_steps=used, epoch=epoch, method=method, training_seed=seed,
                              validation_env_steps=validation_env_steps, validation_seconds=validation_seconds)
                if config.get('comparison', {}).get('validation_storage') == 'per_epoch_v1':
                    raw_path = output / 'validation_requests' / f'epoch_{epoch:06d}.json'
                    atomic_json(raw_path, result)
                    result = {k:v for k,v in result.items() if k != 'deliveries'}
                    result['deliveries_path'] = str(raw_path.relative_to(output).as_posix())
                validations.append(result)
                metadata = {"algorithm": method, "job_id": job_id, "training_steps": used, "seed": seed, "epoch": epoch,
                            "validation": result["summary"], "comparison_identity": identity,
                            "training_budget": budget, "episodes_started": episode,
                            "validation_env_steps": validation_env_steps, "validation_seconds": validation_seconds}
                if result["summary"]["safe"] and (best_rank is None or rank > best_rank):
                    best_rank = rank
                    agent.save(output / "best_checkpoint.pt", metadata)
                    atomic_json(output / "best_validation.json", result)
                agent.save(output / "last_checkpoint.pt", metadata)
                atomic_json(output / "validations.json", validations)
                validation_index += 1
                print(f"{method} seed={seed} epoch={epoch} episodes={episode}/{episodes} steps={used} validation={result['summary']}", flush=True)
                history[-1].update(validation_env_steps=validation_env_steps, validation_seconds=validation_seconds,
                                   seconds=time.perf_counter()-started)
                save_epoch(output, history[-1], rollout.episodes)
                atomic_json(output / 'training_log.json', history)
                export_history(output, history, episode_log, validations)
    if best_rank is None:
        raise RuntimeError("no physically safe eligible checkpoint")
    summary = {"job_id": job_id, "method": method, "seed": seed, "learning_rate": rate,
               "training_budget": budget, "budget_completed": True,
               "training_steps": used, "epochs": len(history), "epoch_definition": EPOCH_DEFINITION,
               "log_schema_version": LOG_VERSION, "episodes_started": episode, "validation_count": len(validations),
               "optimizer_updates": agent.updates if method == "rule_ddqn" else updates,
               "sample_presentations": agent.sample_presentations if method == "rule_ddqn" else presentations,
               "seconds": time.perf_counter()-started, "training_sufficiency": "not_established",
               "training_seconds": training_seconds,
               "parameter_count": sum(p.numel() for p in (agent.online if method == 'rule_ddqn' else agent.network).parameters()) if hasattr(agent, 'network') or hasattr(agent, 'online') else None,
               "trainable_parameter_count": sum(p.numel() for p in (agent.online if method == 'rule_ddqn' else agent.network).parameters() if p.requires_grad) if hasattr(agent, 'network') or hasattr(agent, 'online') else None,
               "validation_seconds": validation_seconds, "validation_env_steps": validation_env_steps,
               "best_validation": json.loads((output/"best_validation.json").read_text(encoding="utf-8"))["summary"]}
    atomic_json(summary_path, summary)
    atomic_json(output / "status.json", {"job_id": job_id, "status": "complete"})
    return summary
