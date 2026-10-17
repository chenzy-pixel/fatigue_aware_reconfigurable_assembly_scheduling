"""Post-training challenge delivery evaluation and instance-level statistics."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np
import torch

from configs import load_config, project_path
from configs.config import public_config
from configs.formal_preferences import formal_preferences
from configs.runtime import assert_checkpoint_fatigue_mode
from data.challenge import (NAMES, atomic_json, digest, load_suites, read_sealed, seal,
                            successful_witness)
from data.dataset import sha256_file
from environment import AssemblySchedulingEnv, PreferenceContext
from environment.types import metrics_flow_objective, terminal_quality_score
from environment.workload import metrics_workload_objective
from eval import EvaluationPolicy, evaluate_instance
from result.io import write_csv
from result.metrics import failure_reason, successful_row
from result.sample_selection import select_best_of_three
from utils import configured_formal_evaluation_sampling_seeds, set_seed, SAMPLED_EVALUATION_RNG_VERSION


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    """Preserve native Q and original config precision across request resumes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def smoke_records(datasets):
    result = []
    for dataset in datasets:
        if dataset.manifest["suite"] == NAMES[0]:
            seen = set()
            for record in dataset:
                if record.metadata["pressure_type"] not in seen:
                    result.append(record)
                    seen.add(record.metadata["pressure_type"])
        else:
            first = dataset[0].metadata["base_instance_id"]
            result.extend(r for r in dataset if r.metadata["base_instance_id"] == first)
    return result


def delivery_rows(raw):
    """Use the project's native success / Q / repeat selector, including failure Q=1."""
    rows = [{**sample["metrics"], "instance_id": sample["instance_id"],
             "preference_key": sample["preference_key"], "sampling_repeat": sample["sampling_repeat"],
             "schedule_violation_count": len(sample["metrics"]["schedule_violations"])} for sample in raw]
    chosen, _ = select_best_of_three(rows, unique_instance_count=1, preference_keys=[raw[0]["preference_key"]])
    selected_repeat = chosen[0]["sampling_repeat"]
    for sample in raw:
        sample["selected_for_aggregation"] = sample["sampling_repeat"] == selected_repeat
    return selected_repeat


def _sample(env, metrics, record, preference, repeat, runner=None):
    metrics["schedule_violation_count"] = len(metrics["schedule_violations"])
    success = successful_row(metrics)
    if not success:
        metrics["preference_quality_score"] = 1.0
    actions = list(runner._episode_actions) if runner is not None else []
    return {"instance_id": record.instance.instance_id, "preference": list(preference.as_tuple()),
            "preference_key": preference.key, "sampling_repeat": repeat, "success": success,
            "metrics": clean(metrics), "actions": actions,
            "schedule": clean(env.schedule_log), "reconfigurations": clean(env.reconfiguration_log)}


def validate_request(value, key, *, policy, preference_key, iid):
    if value.get("request_sha256") != key:
        raise ValueError("request resume identity mismatch")
    raw = value.get("raw", [])
    expected = 3 if policy == "ppo" else 1
    if len(raw) != expected or {r["sampling_repeat"] for r in raw} != set(range(expected)):
        return False
    for r in raw:
        if r["instance_id"] != iid or r["preference_key"] != preference_key:
            raise ValueError("request cell mismatch")
        if not isinstance(r.get("metrics"), dict) or "schedule" not in r or "reconfigurations" not in r or "actions" not in r:
            return False
    selected = delivery_rows(raw) if policy == "ppo" else 0
    return value.get("selected_repeat") == selected


def rescore_heuristic_sample(reference, preference, config):
    sample = deepcopy(reference)
    sample["source_trajectory_preference"] = list(reference["preference"])
    metrics = sample["metrics"]
    metrics["preference_quality_score"] = terminal_quality_score(
        metrics_flow_objective(metrics, config), metrics["reconfiguration_cost"], metrics_workload_objective(metrics, config),
        config, preference=preference.preference, terminal_failure=not sample["success"])
    metrics["preference"] = preference.preference.as_dict()
    sample.update(preference=list(preference.as_tuple()), preference_key=preference.key, selected_for_aggregation=True)
    return sample


def _provenance(training_config, checkpoint, config, datasets, records, preferences, smoke, policy, seeds):
    source_names = ("analysis/challenge_study.py", "eval.py", "result/sample_selection.py", "utils.py",
                    "agent/ppo/agent.py", "agent/ppo/network.py", "agent/ppo/parallel.py", "environment/env.py", "agent/baselines/policies.py")
    sources = {p: sha256_file(project_path(p)) for p in source_names if project_path(p).exists()}
    value = {"version": "challenge_evaluation_v1", "policy": policy, "smoke": smoke,
             "training_config_sha256": sha256_file(project_path(training_config)),
             "effective_config_sha256": digest(public_config(config)), "training_seed": config["seed"],
             "checkpoint_sha256": sha256_file(project_path(checkpoint)) if checkpoint else None,
             "training_sample_aggregation": config["training"]["formal_evaluation"].get("sample_aggregation", "per_sample_v1"),
             "delivery_sample_aggregation": "best_of_3_v1" if policy == "ppo" else "heuristic_once_rescored_v1",
             "sampling_rng_version": SAMPLED_EVALUATION_RNG_VERSION if policy == "ppo" else None,
             "execution": "native_parallel_trace_v1" if policy == "ppo" else "serial_heuristic_v1",
             "parallel_envs": min(int(config["training"]["validation_parallel_envs"]), len(records)*len(preferences)) if policy == "ppo" else 1,
             "statistical_options": {k: datasets[0].manifest["spec"][k] for k in ("bootstrap_seed", "bootstrap_repeats")},
             "sampling_seeds": seeds, "preferences": [list(p.as_tuple()) for p in preferences],
             "manifest_hashes": {d.manifest["suite"]: sha256_file(d.manifest_path) for d in datasets},
             "instance_ids": [r.instance.instance_id for r in records], "sources": sources,
             "hardware": {"platform": platform.platform(), "torch": torch.__version__, "device": config["device"],
                          "gpu": torch.cuda.get_device_name() if config["device"] == "cuda" else None,
                          "threads": torch.get_num_threads()}}
    return seal(value)


def _parallel_requests(config, agent, records, preferences, seeds, identity, folder):
    from agent.ppo.parallel import ParallelEpisodeRunner
    jobs, paths = [], []
    for record in records:
        for pref in preferences:
            key = digest([identity["content_sha256"], record.instance.instance_id, pref.key])
            path = folder / "requests" / f"{key}.json"
            complete = False
            if path.exists():
                try:
                    complete = validate_request(read_sealed(path), key, policy="ppo", preference_key=pref.key, iid=record.instance.instance_id)
                except ValueError as error:
                    if "identity mismatch" in str(error):
                        raise
                except (KeyError, json.JSONDecodeError):
                    pass
            if complete:
                paths.append({"path": str(path.relative_to(folder)), "sha256": sha256_file(path)})
            else:
                jobs.append((record, pref, key, path))
    if not jobs:
        return paths
    parallelism = identity["parallel_envs"]
    runner = ParallelEpisodeRunner(config=config, template=records[0].instance,
                                   episode_count=int(config["training"]["episodes"]), worker_count=parallelism,
                                   diagnostic_directory=folder/"worker_diagnostics")
    was_training = agent.network.training
    agent.network.eval()
    try:
        for offset in range(0, len(jobs), parallelism):
            chunk = jobs[offset:offset+parallelism]
            samples = [[] for _ in chunk]
            for repeat, seed in enumerate(seeds):
                rollouts = runner.evaluate_records(agent, [j[0] for j in chunk], preferences=[j[1] for j in chunk],
                            max_parallelism=len(chunk), deterministic=False, sampling_seed=seed, capture_trajectories=True)
                if len(rollouts) != len(chunk) or {r.record_index for r in rollouts} != set(range(len(chunk))):
                    raise ValueError("parallel evaluator returned an incomplete request chunk")
                for rollout in rollouts:
                    index = rollout.record_index
                    metrics = dict(rollout.metrics)
                    metrics.update(decisions=rollout.decisions, solve_time_seconds=rollout.solve_time_seconds,
                        inference_time_seconds=rollout.inference_time_seconds, action_trace_sha256=rollout.action_trace_sha256,
                        sampling_seed=rollout.sampling_seed, derived_sampling_seed=rollout.derived_sampling_seed,
                        sampling_rng_version=SAMPLED_EVALUATION_RNG_VERSION, sampling_evaluation_key=rollout.sampling_evaluation_key,
                        policy_execution_version=agent.network.execution_mode, decode_mode="sampled", policy="ppo")
                    env = SimpleNamespace(schedule_log=rollout.schedule, reconfiguration_log=rollout.reconfigurations)
                    trace = SimpleNamespace(_episode_actions=rollout.actions)
                    samples[index].append(_sample(env, metrics, chunk[index][0], chunk[index][1], repeat, trace))
                print(f"ppo chunk {offset//parallelism+1}: repeat {repeat+1}/3, trajectories={len(rollouts)}", flush=True)
            for (record, pref, key, path), raw in zip(chunk, samples):
                selected = delivery_rows(raw)
                atomic_json(path, seal({"request_sha256":key,"identity_sha256":identity["content_sha256"],
                    "metadata":{k:v for k,v in record.metadata.items() if k not in ("heuristic_metrics","diagnostic_actions")},
                    "policy":"ppo","selected_repeat":selected,"raw":raw}))
                paths.append({"path":str(path.relative_to(folder)),"sha256":sha256_file(path)})
    finally:
        agent.network.train(was_training)
        runner.close()
    return paths


def evaluate_suites(manifest, training_config, checkpoint, output, *, policy="ppo", smoke=False, device=None):
    datasets = load_suites(manifest)
    config = load_config(training_config)
    threads = int(config["training"].get("torch_num_threads", 4))
    if threads < 1:
        raise ValueError("torch_num_threads must be positive")
    torch.set_num_threads(threads)
    if device:
        config["device"] = device
    if config["environment"].get("fatigue_mode", "full") != "full":
        raise ValueError("challenge requires full physical fatigue evaluation")
    if any(d.manifest["physical_environment"] != config["environment"] for d in datasets):
        raise ValueError("challenge and checkpoint evaluation environment settings differ")
    records = smoke_records(datasets) if smoke else [r for d in datasets for r in d]
    preferences = ([PreferenceContext.from_input(p) for p in ((1,0,0),(0,1,0),(0,0,1),(1/3,1/3,1/3))]
                   if smoke else list(formal_preferences(config, "final_test")))
    if not smoke and len(preferences) != 66:
        raise ValueError("formal challenge evaluation requires the 66-point universal preference grid")
    seeds = configured_formal_evaluation_sampling_seeds(config, "final_test") if policy == "ppo" else []
    if policy == "ppo" and len(seeds) != 3:
        raise ValueError("challenge delivery requires exactly three sampling seeds")
    output = Path(output)
    arm_output = output / policy
    arm_output.mkdir(parents=True, exist_ok=True)
    identity = _provenance(training_config, checkpoint, config, datasets, records, preferences, smoke, policy, seeds)
    ip = arm_output / "evaluation.json"
    if ip.exists() and read_sealed(ip) != identity:
        raise ValueError("evaluation identity changed; use a new output directory")
    atomic_json(ip, identity)
    atomic_json(arm_output / "training_config.json", public_config(config))
    set_seed(int(config["seed"]))
    if policy == "ppo":
        from agent.ppo import PPOAgent, build_actor_critic
        observation = AssemblySchedulingEnv(config).reset(records[0].instance)
        agent = PPOAgent(build_actor_critic(observation, config["network"]), config["ppo"], device=config["device"])
        checkpoint_metadata = agent.load(project_path(checkpoint))
        assert_checkpoint_fatigue_mode(checkpoint_metadata, config)
        # Keep the complete original checkpoint metadata, without optimizer/weight payloads.
        atomic_json(arm_output / "checkpoint_metadata.json", clean(checkpoint_metadata))
    paths = _parallel_requests(config, agent, records, preferences, seeds, identity, arm_output) if policy == "ppo" else []
    for record in records if policy == "heuristic" else []:
        iid = record.instance.instance_id
        reference = None
        reference_path = arm_output / "heuristic_trajectories" / f"{iid}.json"
        if policy == "heuristic":
            reference_key = digest([identity["content_sha256"], iid, "one_physical_trajectory"])
            if reference_path.exists():
                reference = read_sealed(reference_path)
                if reference["request_sha256"] != reference_key:
                    raise ValueError("heuristic trajectory resume identity mismatch")
            else:
                runner = EvaluationPolicy(config, policy_name="heuristic", bootstrap_observation=AssemblySchedulingEnv(config).reset(record.instance))
                env, metrics = evaluate_instance(config, instance=record.instance, policy_name="heuristic", prepared_policy=runner,
                                                 preference=preferences[0], decode_mode="greedy")
                sample = _sample(env, metrics, record, preferences[0], 0, runner)
                reference = seal({"request_sha256": reference_key, "sample": sample})
                atomic_json(reference_path, reference)
        for pref in preferences:
            key = digest([identity["content_sha256"], iid, pref.key])
            path = arm_output / "requests" / f"{key}.json"
            if path.exists():
                try:
                    cached = read_sealed(path)
                    if validate_request(cached, key, policy=policy, preference_key=pref.key, iid=iid):
                        paths.append({"path": str(path.relative_to(arm_output)), "sha256": sha256_file(path)})
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass
                except ValueError as error:
                    if "identity mismatch" in str(error):
                        raise
                    # A corrupt/incomplete cell is recomputed under the frozen identity.
            raw = []
            sample = rescore_heuristic_sample(reference["sample"], pref, config)
            raw, selected = [sample], 0
            atomic_json(path, seal({"request_sha256": key, "identity_sha256": identity["content_sha256"],
                        "metadata": {k: v for k, v in record.metadata.items() if k not in ("heuristic_metrics", "diagnostic_actions")},
                        "policy": policy, "selected_repeat": selected, "raw": raw}))
            paths.append({"path": str(path.relative_to(arm_output)), "sha256": sha256_file(path)})
        print(f"{policy}: {iid}, cells={len(paths)}/{len(records)*len(preferences)}", flush=True)
    atomic_json(arm_output / "completed.json", seal({"identity_sha256": identity["content_sha256"], "requests": paths,
                "expected_cells": len(records) * len(preferences), "status": "completed"}))
    if smoke and policy == "ppo":
        evaluate_suites(manifest, training_config, None, output, policy="heuristic", smoke=True, device=device)
    summary = report_evaluation(output)
    if smoke and (not summary["all_raw_constraints_pass"] or not summary["all_raw_complete_without_sampling_truncation"]):
        raise RuntimeError("smoke contains invalid or sampling-truncated trajectories; see report.json")
    return summary


def bootstrap_interval(rows, value, *, seed=20261017, repeats=10000):
    groups = defaultdict(list)
    for row in rows:
        groups[row["base_instance_id"] if row["suite"] == NAMES[1] else row["instance_id"]].append(float(row[value]))
    if not groups:
        return {"mean": None, "ci95": None, "independent_units": 0}
    values = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    means = np.mean(values[rng.integers(0, len(values), size=(repeats, len(values)))], axis=1)
    return {"mean": float(values.mean()), "ci95": [float(x) for x in np.quantile(means, [.025, .975])],
            "independent_units": len(values), "ci_scope": "conditional_on_frozen_training_model"}


def _summary(rows, spec):
    success = [r for r in rows if r["success"]]
    return {"cells": len(rows), "successful_cells": len(success),
            "success": bootstrap_interval(rows, "success", seed=spec["bootstrap_seed"], repeats=spec["bootstrap_repeats"]),
            "quality_all_cells_failure_one": bootstrap_interval(rows, "quality", seed=spec["bootstrap_seed"], repeats=spec["bootstrap_repeats"]),
            "completed_objectives": {key: float(np.mean([r[key] for r in success])) if success else None for key in ("flow", "cost", "owa")},
            "mean_request_solve_time_seconds": float(np.mean([r["request_solve_time_seconds"] for r in rows])) if rows else None,
            "mechanism_means_all_cells": {key: float(np.mean([r[key] for r in rows if r.get(key) is not None]))
                 if any(r.get(key) is not None for r in rows) else None for key in (
                     "completed_reconfigurations", "reconfiguration_time", "machine_waiting_for_worker_time",
                     "maximum_worker_fatigue", "fatigue_masked_action_ratio", "worker_competition_event_count")},
            "failure_reasons": dict(Counter(r["failure_reason"] for r in rows if not r["success"]))}


def report_evaluation(output):
    output = Path(output)
    selected, raw_rows, identities = [], [], {}
    spec = None
    for arm in ("ppo", "heuristic"):
        folder = output / arm
        if not (folder / "completed.json").exists():
            continue
        identity = read_sealed(folder / "evaluation.json")
        completed = read_sealed(folder / "completed.json")
        if completed["identity_sha256"] != identity["content_sha256"]:
            raise ValueError("completed evaluation identity mismatch")
        identities[arm] = identity
        config = load_config(folder / "training_config.json")
        expected_cells = {(i, PreferenceContext.from_input(p).key) for i in identity["instance_ids"] for p in identity["preferences"]}
        seen = set()
        for ref in completed["requests"]:
            path = folder / ref["path"]
            if sha256_file(path) != ref["sha256"]:
                raise ValueError("completed request hash mismatch")
            request = read_sealed(path)
            first = request["raw"][0]
            key = digest([identity["content_sha256"], first["instance_id"], first["preference_key"]])
            if not validate_request(request, key, policy=arm, preference_key=first["preference_key"], iid=first["instance_id"]):
                raise ValueError("completed request is incomplete")
            cell = (first["instance_id"], first["preference_key"])
            if cell in seen:
                raise ValueError("duplicate completed cell")
            seen.add(cell)
            md = request["metadata"]
            for sample in request["raw"]:
                m = sample["metrics"]
                row = {"policy": arm, "training_seed": identity["training_seed"], "instance_id": sample["instance_id"],
                       "preference_key": sample["preference_key"], "preference": sample["preference"],
                       "low_cost_preference": sample["preference"][1] <= .2 + 1e-9,
                       **{k: md[k] for k in ("suite", "pressure_type", "cost_profile", "base_instance_id", "reconfiguration_scale", "feasibility_status")},
                       "success": sample["success"], "quality": m["preference_quality_score"] if sample["success"] else 1.,
                       "flow": metrics_flow_objective(m, config), "cost": m["reconfiguration_cost"], "owa": metrics_workload_objective(m, config),
                       "sampling_repeat": sample["sampling_repeat"], "selected_for_aggregation": sample["selected_for_aggregation"],
                       "request_solve_time_seconds": sum(s["metrics"]["solve_time_seconds"] for s in request["raw"]),
                       "failure_reason": None if sample["success"] else failure_reason(m),
                       "active_constraint_pass": not m["schedule_violations"] and m["maximum_worker_fatigue"] <= m["safe_fatigue_limit"] + 1e-9,
                       **{k: m.get(k) for k in ("completed_reconfigurations", "machine_waiting_for_worker_time", "maximum_worker_fatigue", "fatigue_masked_action_ratio", "worker_competition_event_count", "solve_time_seconds", "sampling_seed", "action_trace_sha256")},
                       "reconfiguration_time": sum(float(r.get("end", 0)) - float(r.get("start", 0)) for r in sample["reconfigurations"])}
                raw_rows.append(row)
                if sample["selected_for_aggregation"]:
                    selected.append(row)
        if seen != expected_cells or completed["expected_cells"] != len(expected_cells):
            raise ValueError("completed grid has missing/extra cells")
    if not identities:
        raise ValueError("no completed evaluation to report")
    # Statistics options belong to the frozen suite, not to the trained model.
    spec = next(iter(identities.values()))["statistical_options"]
    groups = defaultdict(list)
    for row in selected:
        groups[(row["policy"], row["suite"], "overall", "all")].append(row)
        for field in ("pressure_type", "cost_profile", "reconfiguration_scale", "feasibility_status", "low_cost_preference", "preference_key"):
            groups[(row["policy"], row["suite"], field, str(row[field]))].append(row)
    summaries = [{"policy": arm, "suite": suite, "group_by": field, "value": value, **_summary(rows, spec)}
                 for (arm, suite, field, value), rows in groups.items()]
    comparisons = []
    if set(identities) == {"ppo", "heuristic"}:
        a, b = identities["ppo"], identities["heuristic"]
        for key in ("training_config_sha256", "manifest_hashes", "instance_ids", "preferences"):
            if a[key] != b[key]:
                raise ValueError("comparison grids/configs differ")
        lookup = {(r["policy"], r["instance_id"], r["preference_key"]): r for r in selected}
        paired = []
        for r in selected:
            if r["policy"] != "ppo":
                continue
            h = lookup["heuristic", r["instance_id"], r["preference_key"]]
            if r["success"] and h["success"]:
                paired.append({**r, "quality_improvement": h["quality"] - r["quality"]})
        for suite in NAMES:
            p = [r for r in paired if r["suite"] == suite]
            comparisons.append({"suite": suite, "common_success_cells": len(p),
                                "quality_improvement": bootstrap_interval(p, "quality_improvement", seed=spec["bootstrap_seed"], repeats=spec["bootstrap_repeats"]),
                                "completed_objectives": {arm: {key: float(np.mean([lookup[arm,r['instance_id'],r['preference_key']][key] for r in p])) if p else None for key in ("flow","cost","owa")} for arm in ("ppo","heuristic")}})
    cert_groups = defaultdict(list)
    for row in selected:
        if row["suite"] == NAMES[1]:
            cert_groups[row["policy"], row["base_instance_id"]].append(row)
    all_certified = {(arm, base) for (arm, base), rows in cert_groups.items()
                     if len({r["reconfiguration_scale"] for r in rows}) == 4 and all(r["feasibility_status"] == "observed_feasible" for r in rows)}
    common_certified = [r for r in selected if r["suite"] == NAMES[1] and (r["policy"],r["base_instance_id"]) in all_certified]
    certified_summaries = []
    for arm in identities:
        for scale in (1.4,1.6,1.8,2.):
            rows = [r for r in common_certified if r["policy"] == arm and r["reconfiguration_scale"] == scale]
            certified_summaries.append({"policy": arm, "scale": scale, **_summary(rows, spec)})
    paired_scales = []
    for arm in identities:
        baseline = {(r["base_instance_id"],r["preference_key"]):r for r in selected if r["policy"]==arm and r["suite"]==NAMES[1] and r["reconfiguration_scale"]==1.4}
        for scale in (1.6,1.8,2.):
            rows = [r for r in selected if r["policy"]==arm and r["suite"]==NAMES[1] and r["reconfiguration_scale"]==scale]
            deltas = [{**r,"quality_delta":r["quality"]-baseline[r["base_instance_id"],r["preference_key"]]["quality"]} for r in rows]
            paired_scales.append({"policy":arm,"scale":scale,"baseline_scale":1.4,"quality_delta_all_cells_failure_one":bootstrap_interval(deltas,"quality_delta",seed=spec["bootstrap_seed"],repeats=spec["bootstrap_repeats"])})
    report = {"version": "challenge_report_v1", "evaluation_identities": {k:v["content_sha256"] for k,v in identities.items()},
              "raw_trajectory_rows": len(raw_rows), "delivery_cells": len(selected),
              "all_raw_constraints_pass": all(r["active_constraint_pass"] for r in raw_rows),
              "all_raw_complete_without_sampling_truncation": all(not s["metrics"].get("sampling_truncated",s["metrics"].get("truncated")) for arm in identities for ref in read_sealed(output/arm/"completed.json")["requests"] for s in read_sealed(output/arm/ref["path"])["raw"]),
              "group_summaries": summaries, "common_success_comparisons": comparisons,
              "all_scale_certified_subset": certified_summaries, "paired_scale_changes": paired_scales,
              "statistical_unit": "instance; extrapolation base group clusters all four scales",
              "bootstrap_seed": spec["bootstrap_seed"], "bootstrap_repeats": spec["bootstrap_repeats"],
              "limitation": "single-training-seed chain validation; not a generalization performance claim"}
    write_csv(output / "raw_samples.csv", raw_rows)
    write_csv(output / "selected_samples.csv", selected)
    atomic_json(output / "report.json", clean(report))
    (output / "report.md").write_text(
        f"# Challenge evaluation\n\nRaw trajectory rows: {len(raw_rows)}; delivered cells: {len(selected)}.\n\n"
        f"All raw constraints pass: {report['all_raw_constraints_pass']}.\n\n"
        "See report.json for per-pressure, cost, scale, feasibility and preference results, paired comparisons and instance-level confidence intervals.\n\n"
        "This is a frozen single-training-seed integration check; performance conclusions require independent models and formal evaluation.\n", encoding="utf-8")
    return {k:v for k,v in report.items() if k not in ("group_summaries","all_scale_certified_subset","paired_scale_changes","common_success_comparisons")}
