"""Reproducible staged MO-ALNS validation and delivery-time comparison.

Run --stage all for mechanism -> curves -> budget -> independent confirmation.
Every completed request is atomically persisted and can be resumed by rerunning.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _thread_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_name] = "1"

from configs import load_config
from configs.config import public_config
from configs.formal_preferences import formal_preferences
from data import load_dataset_split
from data.distribution import PRESSURE_TYPES
from data.selection import subset_snapshot
from environment import PreferenceVector
from agent.mo_alns import MOALNSSolver, SearchBudget
from agent.mo_alns.types import preference_key

CHECKPOINT_DIR = ROOT / "result/runs/schema12_screen_seed11_20261012_201659_owa_log2"
STUDY_VERSION = "mo_alns_independent_requests_v3"
SEEDS = (11, 23, 37)
FOUR = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (1/3, 1/3, 1/3))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def source_hashes():
    paths = (list((ROOT / "agent/mo_alns").glob("*.py"))
             + list((ROOT / "agent/ppo").glob("*.py"))
             + list((ROOT / "environment").glob("*.py"))
             + [ROOT / "agent/baselines/policies.py", ROOT / "data/models.py", ROOT / "utils.py", ROOT / "eval.py",
                Path(__file__), ROOT / "analysis/mo_alns_study.py"])
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def prepare(output):
    config = load_config("configs/baselines/mo_alns.json")
    dataset = load_dataset_split(config, "validation")
    identity = {"study_version": STUDY_VERSION, "source_hashes": source_hashes(),
                "config": public_config(config), "dataset_sha256": fingerprint(dataset.manifest)}
    identity["sha256"] = fingerprint(identity)
    path = output / "identity.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))["sha256"] != identity["sha256"]:
        raise ValueError("study identity changed; use a new output directory")
    atomic_json(path, identity)
    groups = {p: [] for p in PRESSURE_TYPES}
    for index, record in enumerate(dataset):
        groups[record.metadata["pressure_type"]].append((index, record.instance.instance_id))
    selected = {"tuning": [], "confirmation": []}
    for pressure in PRESSURE_TYPES:
        ordered = sorted(groups[pressure], key=lambda x: fingerprint([identity["dataset_sha256"], x[1], "mo_alns_v2_validation"]))
        if len(ordered) < 4:
            raise ValueError("validation pressure stratum has fewer than four instances")
        selected["tuning"].extend(x[0] for x in ordered[:3])
        selected["confirmation"].append(ordered[3][0])
    for name, indices in selected.items():
        selected[name] = sorted(indices)
    atomic_json(output / "subsets.json", {name: subset_snapshot(dataset, indices, role=name) for name, indices in selected.items()})
    return config, dataset, selected, identity["sha256"]


def worker(task):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    config, record, pref, seed, seconds, evaluations, adaptive, identity, path = task
    path = Path(path)
    job_id = fingerprint([identity, record.instance.instance_id, pref, seed, seconds, evaluations, adaptive])
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("job_id") != job_id:
            raise ValueError(f"cached request identity differs: {path}")
        return cached["summary"]
    config = deepcopy(config)
    config["mo_alns"]["adaptive_weights"] = adaptive
    result = MOALNSSolver(config, algorithm_seed=seed, dataset_name="validation").solve(
        record.instance, PreferenceVector(*pref), budget=SearchBudget(evaluations, seconds))
    selected = result.selected
    initial = result.candidate_history[:result.initial_candidate_count]
    summary = {"instance_id": record.instance.instance_id, "pressure_type": record.metadata["pressure_type"],
               "preference": pref, "seed": seed, "seconds_budget": seconds, "evaluation_budget": evaluations,
               "adaptive": adaptive, "status": result.status, "stop_reason": result.stop_reason,
               "success": selected is not None, "quality": selected.tchebycheff if selected else 1.0,
               "objectives": selected.objectives if selected else None,
               "initial_quality": result.initial_best_tchebycheff if result.initial_best_tchebycheff is not None else 1.0,
               "heuristic_quality": initial[0]["quality"] if initial and initial[0]["feasible"] else 1.0,
               "initialization_seconds": result.initial_seconds,
               "search_seconds": result.search_time_seconds,
               "environment_evaluations": result.environment_evaluations,
               "cache_hits": result.cache_hits, **result.operator_statistics, **result.accounting}
    # Every delivered candidate is replayable. No post-deadline replay is needed
    # to produce an endpoint or make a selection.
    payload = {"job_id": job_id, "summary": summary, "candidates": result.candidate_history,
               "events": result.search_log,
               "selected": {"actions": selected.actions, "solution": selected.solution.payload(),
                            "metrics": selected.metrics, "schedule": selected.schedule_log,
                            "reconfigurations": selected.reconfiguration_log} if selected else None,
               "archive": [{"objectives": c.objectives, "action_trace_sha256": c.action_trace_sha256,
                            "completed_seconds": c.completed_seconds} for c in result.archive]}
    atomic_json(path, payload)
    return summary


def run_requests(config, dataset, indices, preferences, seeds, seconds, evaluations, adaptive, identity, folder, workers):
    tasks = []
    for index in indices:
        record = dataset[index]
        for pref in preferences:
            for seed in seeds:
                name = f"{index:04d}_{preference_key(PreferenceVector(*pref))}_{seed}.json"
                tasks.append((config, record, pref, seed, seconds, evaluations, adaptive, identity, folder / name))
    completed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            print(json.dumps({"completed": len(completed), "total": len(tasks), "instance": row["instance_id"],
                              "seed": row["seed"], "quality": row["quality"],
                              "neighbourhoods": row["completed_neighbourhoods"]}), flush=True)
    return completed


def mechanism(config, dataset, identity, output, workers):
    indices = [next(i for i in range(len(dataset)) if dataset[i].instance.instance_id == iid) for iid in
               ("validation_balanced_2000001", "validation_reconfiguration_bottleneck_2000003", "validation_fatigue_bottleneck_2000005")]
    rows = run_requests(config, dataset, indices, (FOUR[-1],), (11,), None, 128, True, identity, output / "mechanism", min(workers, 3))
    checks = [{"instance": r["instance_id"], "passed": (r["success"] and r["completed_neighbourhoods"] >= 20
               and r["weight_updates"] >= 4 and r["initial_temperature"] > 0
               and r["quality"] <= r["initial_quality"] + 1e-12)} for r in rows]
    passed = all(c["passed"] for c in checks)
    atomic_json(output / "mechanism.json", {"passed": passed, "checks": checks, "results": rows})
    if not passed:
        raise RuntimeError("real-instance mechanism gate failed; see mechanism.json")


def compare(config, dataset, indices, identity, output, *, preferences=None, search_seeds=SEEDS):
    import torch
    from eval import EvaluationPolicy, evaluate_instance
    from environment import AssemblySchedulingEnv
    from environment.types import metrics_flow_objective
    from environment.workload import metrics_workload_objective
    from utils import configured_formal_evaluation_sampling_seeds
    budget_path = output / "budget_selection.json"
    selection = json.loads(budget_path.read_text(encoding="utf-8"))
    if selection["needs_extension"]:
        raise RuntimeError("finish the required 1800-second extension before confirmation")
    horizon = selection["selected_seconds"]
    pc = load_config(CHECKPOINT_DIR / "config.json")
    checkpoint = CHECKPOINT_DIR / "best_checkpoint.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != "9afc75e1cce6a210c49fec7b22d00d3e0591e4db6d31ed14480d2ea580d9b766":
        raise ValueError("fixed PPO checkpoint hash differs")
    preferences = ([p.preference.as_tuple() for p in formal_preferences(config, "validation")]
                   if preferences is None else preferences)
    bootstrap = AssemblySchedulingEnv(pc)
    obs = bootstrap.reset(dataset[0].instance)
    seeds = configured_formal_evaluation_sampling_seeds(pc, "validation")
    started = time.perf_counter()
    runners = [EvaluationPolicy(pc, policy_name="ppo", bootstrap_observation=obs,
               checkpoint=str(checkpoint), decode_mode="sampled", sampling_seed=seed) for seed in seeds]
    load_seconds = time.perf_counter() - started
    warm_index = next(i for i in range(len(dataset)) if i not in indices)
    for runner in runners:
        evaluate_instance(pc, instance=dataset[warm_index].instance, policy_name="ppo", prepared_policy=runner,
                          preference=PreferenceVector(*FOUR[-1]))
    atomic_json(output / "ppo_identity.json", {"checkpoint": str(checkpoint), "sha256": checkpoint_hash,
                "config": public_config(pc), "delivery_protocol": "best_of_3_v1", "historical_selection_preserved": True,
                "load_seconds": load_seconds, "sampling_seeds": seeds, "device": torch.cuda.get_device_name(0),
                "python": platform.python_version(), "warmup_index": warm_index})
    for index in indices:
        record = dataset[index]
        for pref in preferences:
            name = f"{index:04d}_{preference_key(PreferenceVector(*pref))}"
            path = output / "ppo" / f"{name}.json"
            if not path.exists():
                trials, triples, hashes = [], [], []
                for _ in range(3):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    rows = []
                    for repeat, runner in enumerate(runners):
                        env, metrics = evaluate_instance(pc, instance=record.instance, policy_name="ppo", prepared_policy=runner,
                                                         preference=PreferenceVector(*pref))
                        metrics["sampling_repeat"] = repeat
                        # An externally truncated rollout is not a completed trial.
                        if metrics.get("sampling_truncated", metrics.get("truncated")):
                            raise RuntimeError("PPO evaluation externally truncated")
                        rows.append(metrics)
                    best = min(rows, key=lambda r: (not r["task_succeeded"], r["preference_quality_score"], r["sampling_repeat"]))
                    torch.cuda.synchronize()
                    trials.append(time.perf_counter() - started)
                    triples.append(rows)
                    hashes.append([r["action_trace_sha256"] for r in rows])
                if hashes[1:] != hashes[:1] * 2:
                    raise RuntimeError("repeated PPO timing changed sampled trajectories")
                chosen = min(triples[0], key=lambda r: (not r["task_succeeded"], r["preference_quality_score"], r["sampling_repeat"]))
                success = bool(chosen["task_succeeded"] and not chosen["schedule_violations"])
                summary = {"instance_id": record.instance.instance_id, "pressure_type": record.metadata["pressure_type"],
                           "preference": pref, "success": success, "quality": chosen["preference_quality_score"] if success else 1.0,
                           "objectives": [metrics_flow_objective(chosen, pc), chosen["reconfiguration_cost"], metrics_workload_objective(chosen, pc)],
                           "search_seconds": median(trials), "timing_trials": trials, "candidate_count": 3}
                atomic_json(path, {"summary": summary, "raw_triple": triples[0], "selected_repeat": chosen["sampling_repeat"],
                                   "identity": identity, "checkpoint_sha256": checkpoint_hash})
            else:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("identity") != identity or cached.get("checkpoint_sha256") != checkpoint_hash:
                    raise ValueError("PPO cached comparison identity differs")
            tppo = json.loads(path.read_text(encoding="utf-8"))["summary"]["search_seconds"]
            # Single-request latency measurements run without competing jobs.
            for label, limit, adaptive in (("confirmed", horizon, True), ("fixed_lns", horizon, False), ("ppo_deadline", tppo, True)):
                for seed in search_seeds:
                    worker((config, record, pref, seed, limit, None, adaptive, identity,
                            output / label / f"{name}_{seed}.json"))
            print(f"confirmed {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("mechanism", "curves", "select", "compare", "report", "formal", "all"), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result/analysis/mo_alns_v2_study")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in [1,8]")
    output = args.output_dir.resolve()
    config, dataset, subsets, identity = prepare(output)
    from analysis.mo_alns_study import select_budget, report
    stage = args.stage
    if stage in ("mechanism", "all"):
        mechanism(config, dataset, identity, output, args.workers)
    if stage in ("curves", "all"):
        if not json.loads((output / "mechanism.json").read_text(encoding="utf-8"))["passed"]:
            raise RuntimeError("mechanism gate has not passed")
        run_requests(config, dataset, subsets["tuning"], FOUR, SEEDS, 600., None, True, identity, output / "curves_600", args.workers)
        selection = select_budget(output)
        if selection["needs_extension"]:
            run_requests(config, dataset, subsets["tuning"], FOUR, SEEDS, 1800., None, True, identity, output / "curves_1800", args.workers)
            select_budget(output)
    if stage == "select":
        select_budget(output)
    if stage in ("compare", "all"):
        compare(config, dataset, subsets["confirmation"], identity, output)
    if stage in ("report", "all"):
        report(output, config)
    if stage in ("formal", "all"):
        selected = json.loads((output / "budget_selection.json").read_text(encoding="utf-8"))
        cfg = public_config(config)
        cfg["mo_alns"]["max_seconds_per_preference"] = selected["selected_seconds"]
        atomic_json(output / "formal_config.json", cfg)
        atomic_json(output / "formal_manifest.json", {"protocol": "ppo_mo_alns_solver_budget_v3", "config": str(output / "formal_config.json"),
                    "datasets": ["test", "ood", "stress"], "algorithm_seeds": list(SEEDS), "parallel_envs": 8,
                    "budget_sufficiency": selected["budget_sufficiency"],
                    "requires_independent_confirmation": (not (output / "comparison_summary.json").exists()
                        or not json.loads((output / "comparison_summary.json").read_text(encoding="utf-8")).get("independent_confirmation_passed", False))})


if __name__ == "__main__":
    main()
