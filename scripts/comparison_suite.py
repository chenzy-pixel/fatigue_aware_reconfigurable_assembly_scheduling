"""Remote staged implementation study of four project-adapted comparators.

Default stage is read-only preflight. Heavy stages require --remote-run.
Completed requests resume by identity; interrupted training restarts from zero.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
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
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import torch

from configs import load_config, project_path
from configs.config import public_config
from configs.formal_preferences import formal_preferences, objective_scales
from data import load_dataset_split
from data.distribution import PRESSURE_TYPES
from data.selection import subset_snapshot
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv, PreferenceVector
from agent.comparators.common import (VERSION, METHODS, SEEDS, FOUR, TRAIN_STEPS, PILOT_STEPS,
                                     atomic_json, digest, method_config)
from agent.comparators.evolution import EvolutionSolver
from agent.comparators.learning import train_request, build_agent, evaluate_request
from agent.mo_alns import MOALNSSolver, SearchBudget
from analysis.comparator_study import snapshot, plateau, report, write_csv

SEARCH = ("nsga2_ls", "moead")
LEARN = ("ppo", "dcan", "rule_ddqn")
RATES = {"ppo": (5e-5, 1e-4, 2e-4), "dcan": (5e-5, 1e-4, 2e-4), "rule_ddqn": (1e-4, 3e-4, 1e-3)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(config, output):
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_split(config, "validation")
    # Preserve the exact existing MO-ALNS subset derivation.
    from scripts.mo_alns_study import fingerprint
    manifest_hash = fingerprint(dataset.manifest)
    groups = {p: [] for p in PRESSURE_TYPES}
    for index, record in enumerate(dataset):
        groups[record.metadata["pressure_type"]].append(index)
    subsets = {"tuning": [], "selection": [], "confirmation": []}
    for group in groups.values():
        ordered = sorted(group, key=lambda i: fingerprint([manifest_hash, dataset[i].instance.instance_id, "mo_alns_v2_validation"]))
        subsets["tuning"].extend(ordered[:3])
        subsets["confirmation"].append(ordered[3])
        remaining = sorted(ordered[4:], key=lambda i: digest([manifest_hash, dataset[i].instance.instance_id, "comparison_selection_v1"]))
        subsets["selection"].extend(remaining[:3])
    subsets = {k: sorted(v) for k, v in subsets.items()}
    if [len(subsets[k]) for k in ("tuning", "selection", "confirmation")] != [21, 21, 7]:
        raise ValueError("insufficient pressure-stratified validation data")
    sources = {}
    for directory in ("agent", "environment", "configs", "data", "training", "result"):
        sources.update({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/directory).rglob("*.py")})
    for name in ("scripts/comparison_suite.py", "scripts/mo_alns_study.py", "analysis/comparator_study.py", "utils.py"):
        p = ROOT/name
        sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    hardware = {"machine": platform.node(), "system": platform.platform(), "cpu": platform.processor(),
                "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    value = {"version": VERSION, "sources": sources, "config": public_config(config),
             "dataset": manifest_hash, "subsets": subsets, "hardware": hardware}
    value["sha256"] = digest(value)
    path = output/"identity.json"
    if path.exists() and read(path)["sha256"] != value["sha256"]:
        raise ValueError("source/config/data/hardware identity changed; choose a new output directory")
    atomic_json(path, value)
    atomic_json(output/"subsets.json", {k: subset_snapshot(dataset, v, role=k) for k, v in subsets.items()})
    return dataset, subsets, value["sha256"]


def search_job(task):
    config, record, pref, method, seed, seconds, evaluations, size, identity, path = task
    torch.set_num_threads(1)
    path = Path(path)
    job_id = digest([identity, record.instance.instance_id, pref, method, seed, seconds, evaluations, size])
    if path.exists():
        cached = read(path)
        if cached.get("job_id") != job_id:
            raise ValueError("cached search identity differs")
        return cached["summary"]
    config = method_config(config, method, seed)
    if method == "mo_alns":
        # Exactly the same parameter set as the independently calibrated arm.
        config["mo_alns"] = load_config("configs/baselines/mo_alns.json")["mo_alns"]
    config["comparison"]["evolution"] = {**config["comparison"].get("evolution", {}), "population_size": size}
    solver = (MOALNSSolver(config, algorithm_seed=seed, dataset_name="validation") if method == "mo_alns"
              else EvolutionSolver(config, method, algorithm_seed=seed, dataset_name="validation"))
    result = solver.solve(record.instance, PreferenceVector(*pref), budget=SearchBudget(evaluations, seconds))
    selected = result.selected
    summary = {"method": method, "instance_id": record.instance.instance_id, "preference": pref, "seed": seed,
               "pressure_type": record.metadata["pressure_type"], "population_size": size,
               "seconds_budget": seconds, "evaluation_budget": evaluations, "success": selected is not None,
               "quality": selected.tchebycheff if selected else 1., "objectives": selected.objectives if selected else None,
               "normalized_objectives": selected.normalized_objectives if selected else None,
               "initial_quality": result.initial_best_tchebycheff,
               "search_seconds": result.search_time_seconds, "initialization_seconds": result.initial_seconds,
               "environment_evaluations": result.environment_evaluations, "stop_reason": result.stop_reason,
               "cache_hits": result.cache_hits, **result.operator_statistics, **result.accounting}
    atomic_json(path, {"job_id": job_id, "summary": summary, "candidates": result.candidate_history, "events": result.search_log,
                       "archive": [{"objectives": c.objectives, "normalized_objectives": c.normalized_objectives,
                                    "actions": c.actions, "trace": c.action_trace_sha256, "completed_seconds": c.completed_seconds} for c in result.archive],
                       "selected": {"actions": selected.actions, "solution": selected.solution.payload(), "metrics": selected.metrics,
                                    "schedule": selected.schedule_log, "reconfigurations": selected.reconfiguration_log} if selected else None})
    return summary


def run_search(config, dataset, indices, prefs, methods, seeds, seconds, identity, folder, sizes, workers):
    tasks = [(config, dataset[i], tuple(p), m, s, seconds, None, sizes[m], identity,
              folder/m/f"{i:04d}_{j:02d}_{s}.json")
             for m in methods for i in indices for j, p in enumerate(prefs) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(search_job, t) for t in tasks]
        rows = []
        for future in as_completed(futures):
            rows.append(future.result())
            print(f"search {folder.name} {len(rows)}/{len(tasks)}", flush=True)
    return rows


def first_per_pressure(dataset, indices):
    chosen = {}
    for i in indices:
        chosen.setdefault(dataset[i].metadata["pressure_type"], i)
    return sorted(chosen.values())


def tune_search(config, dataset, subsets, identity, output, workers):
    chosen = {}
    for method in SEARCH:
        scores = []
        for size in (16, 32, 64):
            rows = run_search(config, dataset, first_per_pressure(dataset, subsets["tuning"]), FOUR, (method,), (11,), 600., identity,
                              output/"tune_search"/str(size), {method: size}, workers)
            scores.append((np.mean([r["success"] for r in rows]), -np.mean([r["quality"] for r in rows]),
                           -np.mean([r["search_seconds"] for r in rows]), -size, size))
        chosen[method] = max(scores)[-1]
    atomic_json(output/"search_selection.json", {"identity": identity, "population_sizes": chosen})


def tune_learning(config, dataset, subsets, identity, output, workers):
    chosen = {}
    for method in LEARN:
        scores = []
        for rate in RATES[method]:
            row = train_request(config, method, 11, rate, PILOT_STEPS, dataset, subsets["tuning"], FOUR,
                                output/"tune_learning"/method/str(rate), identity, workers=workers)
            score = row["best_validation"]
            scores.append((score["safe"], score["worst_preference_success"],
                           -score["quality"] if score["all_preferences_covered"] else -float("inf"), -rate, rate))
        chosen[method] = max(scores)[-1]
    atomic_json(output/"learning_selection.json", {"identity": identity, "learning_rates": chosen})


def select_search_budget(output, method, expected=252):
    latest = output/("curves_1800" if (output/"curves_1800"/method).exists() else "curves_600")/method
    jobs = [read(p) for p in sorted(latest.glob("*.json"))]
    if len(jobs) != expected:
        raise ValueError(f"incomplete curve matrix for {method}: {len(jobs)}/{expected}")
    maximum = 1800 if latest.parent.name == "curves_1800" else 600
    def rows(t):
        return [{**snapshot(j, t), "pressure_type": j["summary"]["pressure_type"]} for j in jobs]
    def mechanisms(rs):
        return all(np.mean([r["iterations"] >= 5 and (method != "nsga2_ls" or r["local_search_evaluations"] > 0)
                            for r in rs if r["pressure_type"] == p]) >= .9 for p in PRESSURE_TYPES)
    end = rows(maximum)
    tail_start = 600 if maximum == 1800 else 180
    stable_tail = plateau(rows(tail_start), end)
    selected = maximum
    sufficient = False
    for t in (60, 180, 600, 1800):
        if t > maximum:
            continue
        if plateau(rows(t), end) and mechanisms(rows(t)) and stable_tail:
            selected, sufficient = t, True
            break
    result = {"method": method, "selected_seconds": selected, "needs_extension": maximum == 600 and not sufficient,
              "budget_sufficiency": "established_on_validation" if sufficient else "not_established",
              "tail_plateau": stable_tail, "mechanism_pass": mechanisms(end)}
    atomic_json(output/f"budget_{method}.json", result)
    curve_rows = [{"method": method, "instance_id": j["summary"]["instance_id"], "pressure_type": j["summary"]["pressure_type"],
                   "seed": j["summary"]["seed"], "preference": j["summary"]["preference"], "seconds": t, **snapshot(j, t)}
                  for j in jobs for t in (10, 30, 60, 180, 600, 1800) if t <= maximum]
    write_csv(output/f"curve_{method}.csv", curve_rows)
    from analysis.comparator_study import clustered_interval
    improvements = [{**j["summary"], "gain": (j["summary"]["initial_quality"] if j["summary"]["initial_quality"] is not None else 1.)-j["summary"]["quality"]} for j in jobs]
    atomic_json(output/f"improvement_{method}.json", clustered_interval(improvements, lambda r: r["gain"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for p in PRESSURE_TYPES:
        times = sorted({r["seconds"] for r in curve_rows})
        ax.plot(times, [np.mean([r["quality"] for r in curve_rows if r["pressure_type"] == p and r["seconds"] == t]) for t in times], label=p)
    ax.set(xlabel="Request budget (s)", ylabel="Mean delivered Q / failure loss", xscale="log")
    ax.legend(fontsize=6)
    fig.tight_layout()
    for extension in ("svg", "pdf", "png"):
        fig.savefig(output/f"curve_{method}.{extension}")
    plt.close(fig)
    return result


def curves(config, dataset, subsets, identity, output, workers):
    sizes = read(output/"search_selection.json")["population_sizes"]
    for method in SEARCH:
        run_search(config, dataset, subsets["tuning"], FOUR, (method,), SEEDS, 600., identity, output/"curves_600", sizes, workers)
        if select_search_budget(output, method)["needs_extension"]:
            run_search(config, dataset, subsets["tuning"], FOUR, (method,), SEEDS, 1800., identity, output/"curves_1800", sizes, workers)
            select_search_budget(output, method)
    # Calibrate ALNS with its own verified mechanism threshold. It is paid as
    # an independent reference study, not assigned an unvalidated 600s cap.
    from scripts.mo_alns_study import prepare as alns_prepare, mechanism, run_requests
    from analysis.mo_alns_study import select_budget
    folder = output/"alns_reference"
    alns_config, alns_dataset, alns_subsets, alns_identity = alns_prepare(folder)
    mechanism(alns_config, alns_dataset, alns_identity, folder, workers)
    run_requests(alns_config, alns_dataset, alns_subsets["tuning"], FOUR, SEEDS, 600., None, True,
                 alns_identity, folder/"curves_600", workers)
    if select_budget(folder)["needs_extension"]:
        run_requests(alns_config, alns_dataset, alns_subsets["tuning"], FOUR, SEEDS, 1800., None, True,
                     alns_identity, folder/"curves_1800", workers)
        select_budget(folder)


def comparison(config, dataset, subsets, prefs, identity, output):
    from analysis.pareto_analysis import normalize_objectives
    sizes = read(output/"search_selection.json")["population_sizes"]
    budgets = {m: read(output/f"budget_{m}.json") for m in SEARCH}
    budgets["mo_alns"] = read(output/"alns_reference"/"budget_selection.json")
    if any(b["needs_extension"] for b in budgets.values()):
        raise ValueError("complete search extension before independent confirmation")
    rates = read(output/"learning_selection.json")["learning_rates"]
    agents = {}
    checkpoint_hashes = {}
    for method in LEARN:
        for seed in SEEDS:
            folder = output/"training"/method/str(seed)
            complete = read(folder/"summary.json")
            if complete["training_steps"] != TRAIN_STEPS:
                raise ValueError("training interaction budget incomplete")
            c = method_config(config, method, seed, rates[method])
            env = AssemblySchedulingEnv(c)
            agent = build_agent(c, env.reset(dataset[subsets["tuning"][0]].instance), method)
            checkpoint = folder/"best_checkpoint.pt"
            metadata = agent.load(checkpoint)
            if metadata.get("comparison_identity") != identity:
                raise ValueError("checkpoint belongs to another comparison study")
            checkpoint_hashes[f"{method}_{seed}"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            # Warm up on a tuning instance, never a confirmation instance.
            evaluate_request(c, agent, dataset[subsets["tuning"][0]].instance, FOUR[-1], method, seed=seed)
            agents[(method, seed)] = (c, agent)
    atomic_json(output/"checkpoint_identity.json", checkpoint_hashes)
    def timed(method, seed, instance, pref, mode):
        c, agent = agents[(method, seed)]
        trials = [evaluate_request(c, agent, instance, pref, method, mode=mode, seed=seed) for _ in range(3)]
        traces = [[r["trace"] for r in t["raw"]] for t in trials]
        if traces[1:] != traces[:1]*2:
            raise RuntimeError("timing repetition changed evaluation candidates")
        result = trials[0]
        result["summary"]["search_seconds"] = median(t["summary"]["search_seconds"] for t in trials)
        result["timing_trials"] = [t["summary"]["search_seconds"] for t in trials]
        return result
    for i in subsets["confirmation"]:
        record = dataset[i]
        for j, pref in enumerate(prefs):
            for method in LEARN:
                for seed in SEEDS:
                    labels = [(method, None)] if method == "rule_ddqn" else [(method, "sampled"), (method+"_greedy", "greedy")]
                    for label, mode in labels:
                        path = output/"confirmation"/label/f"{i:04d}_{j:02d}_{seed}.json"
                        job_id = digest([identity, checkpoint_hashes[f"{method}_{seed}"], i, pref, seed, label])
                        if path.exists():
                            if read(path).get("job_id") != job_id:
                                raise ValueError("cached model delivery identity differs")
                            continue
                        result = timed(method, seed, record.instance, pref, mode)
                        result["job_id"] = job_id
                        result["summary"].update(method=label, seed=seed, pressure_type=record.metadata["pressure_type"],
                            normalized_objectives=normalize_objectives(result["summary"]["objectives"], objective_scales(config)))
                        atomic_json(path, result)
            # Short cap is the median across three independently trained PPOs;
            # each model's triplet latency is also retained for inspection.
            limits = [read(output/"confirmation"/"ppo"/f"{i:04d}_{j:02d}_{s}.json")["summary"]["search_seconds"] for s in SEEDS]
            short = median(limits)
            for method in SEARCH+("mo_alns",):
                full = budgets[method]["selected_seconds"]
                for label, cap in ((method, full), (method+"_ppo_deadline", short)):
                    for seed in SEEDS:
                        path = output/"confirmation"/label/f"{i:04d}_{j:02d}_{seed}.json"
                        row = search_job((config, record, tuple(pref), method, seed, cap, None, sizes.get(method, 32), identity, path))
                        if row["method"] != label:
                            result = read(path)
                            result["summary"]["method"] = label
                            result["summary"]["ppo_model_delivery_times"] = limits
                            atomic_json(path, result)
            print(f"confirmed {i:04d}_{j:02d}", flush=True)


def formal_manifest(config, output, identity):
    summary = read(output/"comparison_summary.json")
    arms = []
    for method in LEARN:
        for seed in SEEDS:
            arms.append({"method": method, "seed": seed, "checkpoint": str(output/"training"/method/str(seed)/"best_checkpoint.pt"),
                         "delivery_protocol": "greedy_single_v1" if method == "rule_ddqn" else "best_of_3_v1"})
    sufficient = True
    for method in SEARCH+("mo_alns",):
        budget = read(output/f"budget_{method}.json") if method in SEARCH else read(output/"alns_reference"/"budget_selection.json")
        sufficient &= budget["budget_sufficiency"] != "not_established"
        arms.append({"method": method, "budget": budget, "seeds": SEEDS})
    atomic_json(output/"formal_manifest.json", {"version": VERSION, "identity": identity, "config": public_config(config), "datasets": ["test", "ood", "stress"],
                "preference_count": 66, "arms": arms, "search_selection": read(output/"search_selection.json"),
                "status": "ready_for_review" if sufficient else "budget_sufficiency_not_established",
                "requires_independent_improvement_review": not all(summary["independent_search_improvement_confirmed"].values()),
                "independent_confirmation_completed": bool(summary), "automatic_execution": False,
                "command": "python -m scripts.comparison_suite --stage formal-evaluate --remote-run --output-dir <same-study-directory>"})


def formal_evaluate(config, output, identity):
    manifest = read(output/"formal_manifest.json")
    if manifest["identity"] != identity or manifest["status"] != "ready_for_review":
        raise ValueError("formal manifest identity/budget sufficiency not ready")
    if manifest.get("requires_independent_improvement_review"):
        raise ValueError("independent search improvement was not confirmed; review before formal expansion")
    preferences = [p.preference.as_tuple() for p in formal_preferences(config, "final_test")]
    rates = read(output/"learning_selection.json")["learning_rates"]
    for split in manifest["datasets"]:
        dataset = load_dataset_split(config, split)
        for arm in manifest["arms"]:
            method = arm["method"]
            for seed in ([arm["seed"]] if method in LEARN else arm["seeds"]):
                if method in LEARN:
                    c = method_config(config, method, seed, rates[method])
                    agent = build_agent(c, AssemblySchedulingEnv(c).reset(dataset[0].instance), method)
                    metadata = agent.load(arm["checkpoint"])
                    if metadata.get("comparison_identity") != identity:
                        raise ValueError("formal checkpoint identity differs")
                for i, record in enumerate(dataset):
                    for j, pref in enumerate(preferences):
                        path = output/"formal"/split/method/f"{i:04d}_{j:02d}_{seed}.json"
                        if method in LEARN:
                            job_id = digest([identity, split, i, pref, method, seed])
                            if path.exists():
                                if read(path).get("job_id") != job_id:
                                    raise ValueError("formal cached identity differs")
                                continue
                            result = evaluate_request(c, agent, record.instance, pref, method, seed=seed)
                            result["job_id"] = job_id
                            result["summary"]["seed"] = seed
                            atomic_json(path, result)
                        else:
                            search_job((config, record, pref, method, seed, arm["budget"]["selected_seconds"], None,
                                manifest["search_selection"]["population_sizes"].get(method, 32), identity, path))


def whole_front(config, dataset, subsets, preferences, output, identity, seconds):
    """One paid population search per instance; kept outside request endpoints."""
    from analysis.pareto_analysis import hypervolume_3d
    from environment.types import bounded_quality_score
    sizes = read(output/"search_selection.json")["population_sizes"]
    rows = []
    for index in subsets["confirmation"]:
        record = dataset[index]
        for method in SEARCH+("mo_alns",):
            for seed in SEEDS:
                path = output/"whole_front"/str(seconds)/method/f"{index:04d}_{seed}.json"
                summary = search_job((config, record, FOUR[-1], method, seed, seconds, None,
                                       sizes.get(method, 32), identity, path))
                payload = read(path)
                archive = payload["archive"]
                endpoints = []
                for pref in preferences:
                    chosen = min(archive, key=lambda c: (bounded_quality_score(*c["objectives"], config, preference=PreferenceVector(*pref)), c["trace"]), default=None)
                    endpoints.append({"preference": pref, "trace": chosen["trace"] if chosen else None,
                                      "quality": bounded_quality_score(*chosen["objectives"], config, preference=PreferenceVector(*pref)) if chosen else 1.})
                payload["whole_front"] = {"endpoints": endpoints, "total_budget_seconds": seconds,
                    "archive_hv": hypervolume_3d([c["normalized_objectives"] for c in archive]),
                    "protocol": "one_search_shared_archive_whole_instance_v1"}
                atomic_json(path, payload)
                rows.append({"method": method, "instance_id": record.instance.instance_id, "seed": seed,
                             "total_seconds": summary["search_seconds"], "budget_seconds": seconds,
                             "archive_hv": payload["whole_front"]["archive_hv"]})
    write_csv(output/"whole_front"/str(seconds)/"summary.csv", rows)


def smoke(config, output):
    config = deepcopy(config)
    config["device"] = "cpu"
    torch.set_num_threads(1)
    instance = load_instance_yaml(project_path(config["paths"]["fixed_instance"]))
    order = replace(instance.orders[0], release_time=0.)
    instance = replace(instance, instance_id="comparison_smoke", orders=(order,),
                       waves={k: {**v, "order_ids": [order.id] if k == order.wave else [], "release_interval": [0., 0.]} for k, v in instance.waves.items()})
    class TinyDataset:
        def __getitem__(self, index):
            from types import SimpleNamespace
            return SimpleNamespace(instance=instance, metadata={"pressure_type": "balanced"})
    identity = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"agent/comparators").glob("*.py")})
    findings = {}
    for method in LEARN:
        findings[method] = train_request(config, method, 11, RATES[method][1], 20, TinyDataset(), [0], (FOUR[-1],),
                                         output/"smoke"/method, identity, workers=2, smoke_instance=instance)
    for method in SEARCH:
        c = method_config(config, method)
        c["comparison"]["evolution"] = {"population_size": 8}
        result = EvolutionSolver(c, method).solve(instance, PreferenceVector(*FOUR[-1]), budget=SearchBudget(max_evaluations=32))
        findings[method] = {"evaluations": result.environment_evaluations, "status": result.status, **result.operator_statistics}
        if result.selected is None:
            raise RuntimeError("search smoke delivered no feasible candidate")
    atomic_json(output/"smoke.json", findings)
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="preflight", choices=("preflight", "smoke", "tune-search", "tune-learning", "train", "curves", "compare", "report", "formal", "formal-evaluate", "front-search", "all"))
    parser.add_argument("--config", default="configs/comparators/suite.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"result/analysis/comparison_suite_v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--training-workers", type=int, default=20)
    parser.add_argument("--remote-run", action="store_true", help="explicitly run heavy stages on the remote experiment host")
    parser.add_argument("--front-seconds", type=float, help="total per-instance budget for the separate front-search stage")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.training_workers <= 20:
        parser.error("workers in [1,8], training-workers in [1,20]")
    heavy = args.stage not in ("preflight", "smoke", "report", "formal")
    if heavy and not args.remote_run:
        parser.error("heavy experiment stages require --remote-run on the remote computer")
    if args.stage == "front-search" and (args.front_seconds is None or not np.isfinite(args.front_seconds) or args.front_seconds <= 0):
        parser.error("front-search needs a positive --front-seconds total budget")
    config = load_config(args.config)
    output = args.output_dir.resolve()
    if args.stage == "smoke":
        text = smoke(config, output)
        print(json.dumps(text, ensure_ascii=False))
        return
    dataset, subsets, identity = prepare(config, output)
    settings = output/"execution_settings.json"
    controls = {"workers": args.workers, "training_workers": args.training_workers}
    if settings.exists() and read(settings) != controls:
        raise ValueError("execution concurrency changed; use a new study directory")
    atomic_json(settings, controls)
    if heavy:
        preflight_path = output/"preflight.json"
        if not preflight_path.exists() or read(preflight_path).get("identity") != identity or not read(preflight_path).get("passed"):
            raise ValueError("run preflight successfully with the same code/config/hardware before heavy stages")
    prefs = tuple(p.preference.as_tuple() for p in formal_preferences(config, "validation"))
    if args.stage == "preflight":
        record = dataset[subsets["tuning"][0]]
        obs = AssemblySchedulingEnv(config).reset(record.instance)
        for method in LEARN:
            agent = build_agent(method_config(config, method), obs, method)
            mask = AssemblySchedulingEnv(config)
            mask.reset(record.instance)
            (agent.act_rules([obs], [mask.get_action_mask()]) if method == "rule_ddqn" else agent.act(obs, mask.get_action_mask()))
        atomic_json(output/"preflight.json", {"passed": True, "identity": identity, "subsets": {k: len(v) for k, v in subsets.items()},
            "training_jobs": 9, "pilot_jobs": 9, "interactions_per_training_job": TRAIN_STEPS,
            "search_tuning_requests": 168, "curve_requests_per_horizon": 756,
            "curve_600_cpu_hours_upper_bound": 126., "extension_1800_cpu_hours_upper_bound": 378.,
            "confirmation_requests_per_arm": 273, "formal_auto_run": False})
        print(f"preflight passed; see {output/'preflight.json'}", flush=True)
        return
    stages = ("tune-search", "tune-learning", "train", "curves", "compare", "report", "formal") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "tune-search":
            tune_search(config, dataset, subsets, identity, output, args.workers)
        elif stage == "tune-learning":
            tune_learning(config, dataset, subsets, identity, output, args.training_workers)
        elif stage == "train":
            rates = read(output/"learning_selection.json")["learning_rates"]
            for method in LEARN:
                for seed in SEEDS:
                    train_request(config, method, seed, rates[method], TRAIN_STEPS, dataset, subsets["selection"], prefs,
                                  output/"training"/method/str(seed), identity, workers=args.training_workers)
            summaries = [read(output/"training"/m/str(s)/"summary.json") for m in LEARN for s in SEEDS]
            atomic_json(output/"training_extension_manifest.json", {"identity": identity, "steps": TRAIN_STEPS*2,
                "requires_review": any(r["training_sufficiency"] == "not_established" for r in summaries),
                "policy": "all methods and seeds restart from zero at 2B if separately requested", "automatic_execution": False})
        elif stage == "curves":
            curves(config, dataset, subsets, identity, output, args.workers)
        elif stage == "compare":
            comparison(config, dataset, subsets, prefs, identity, output)
        elif stage == "report":
            report(output, METHODS)
        elif stage == "formal":
            formal_manifest(config, output, identity)
        elif stage == "formal-evaluate":
            formal_evaluate(config, output, identity)
        elif stage == "front-search":
            whole_front(config, dataset, subsets, prefs, output, identity, args.front_seconds)


if __name__ == "__main__":
    main()
