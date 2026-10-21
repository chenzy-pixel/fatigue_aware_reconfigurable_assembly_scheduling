"""Frozen-instance NSGA-II-LS/MOEA-D search, with a local throughput precheck."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from agent.comparators.common import FOUR, SEEDS, atomic_json, digest
from agent.mo_alns.solver import _rule_solution, decode_solution
from data.dataset import load_generated_record
from environment import PreferenceVector
from scripts.comparison_suite import search_job

VERSION = "heuristic_search_runtime_v1"
SIZES = {"nsga2_ls": 16, "moead": 32}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def host():
    result = {"hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
              "torch": torch.__version__, "numpy": np.__version__, "logical_cpus": os.cpu_count() or 1,
              "physical_cpus": None, "available_memory_bytes": None, "memory_measurement": "unavailable"}
    try:
        import psutil
        result.update(physical_cpus=psutil.cpu_count(logical=False), available_memory_bytes=psutil.virtual_memory().available,
                      memory_measurement="psutil")
    except ImportError:
        pass
    return result


def worker_grid(requested, hardware):
    """Keep one core and an explicit 1 GiB/worker RAM reserve when measurable."""
    cpu = hardware["physical_cpus"] or hardware["logical_cpus"]
    ceiling = max(1, cpu - 1)
    memory = hardware["available_memory_bytes"]
    if memory is not None:
        ceiling = min(ceiling, max(1, int(memory * .75 // 2**30)))
    values = sorted({n for n in requested if 1 <= n <= ceiling})
    return values or [min(ceiling, min(requested))]


def inputs(source, subset, seeds, seconds):
    source = Path(source).resolve()
    previous = read(source / "identity.json")
    subsets = read(source / "subsets.json")
    selected_sizes = read(source / "search_selection.json")["population_sizes"]
    if selected_sizes != SIZES:
        raise ValueError("source population selection differs from the approved N16/N32")
    refs = subsets["tuning"]["files"]
    manifest = ROOT / "data/manifests/v2/validation/manifest.json"
    if sha(manifest) != subsets["tuning"]["dataset_manifest_sha256"]:
        raise ValueError("validation manifest differs from the frozen source")
    records = []
    hashes = {}
    for ref in refs:
        path = ROOT / "data/instances/v2/validation" / ref["path"]
        if sha(path) != ref["sha256"]:
            raise ValueError(f"instance hash differs: {path.name}")
        record = load_generated_record(path)
        records.append(record)
        hashes[record.instance.instance_id] = ref["sha256"]
    if subset == "seven":
        first = {}
        for record in records:
            first.setdefault(record.metadata["pressure_type"], record)
        records = sorted(first.values(), key=lambda r: r.instance.instance_id)
    if len(records) != (7 if subset == "seven" else 21):
        raise ValueError("incomplete pressure-stratified subset")
    # This is the actual source-run config, independent of subsequent training
    # default changes. Fixed physical/objective inputs are carried unchanged.
    config = previous["config"]
    sources = {}
    for directory in ("agent", "environment", "configs", "data"):
        sources.update({p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / directory).rglob("*.py")})
    for p in (ROOT / "result").glob("*.py"):
        sources[p.relative_to(ROOT).as_posix()] = sha(p)
    for name in ("scripts/heuristic_search.py", "scripts/comparison_suite.py", "analysis/comparator_study.py", "utils.py"):
        sources[name] = sha(ROOT / name)
    hardware = host()
    # Available RAM is a live admission observation, not a stable identity field.
    stable_host = {k: v for k, v in hardware.items() if k not in ("available_memory_bytes", "memory_measurement")}
    plan = {"version": VERSION, "source_identity_sha256": previous["sha256"], "source_config": config,
            "source_control_hashes": {name: sha(source / name) for name in ("identity.json", "subsets.json", "search_selection.json")},
            "dataset_manifest_sha256": sha(manifest), "instance_ids": [r.instance.instance_id for r in records],
            "instance_hashes": {r.instance.instance_id: hashes[r.instance.instance_id] for r in records},
            "subset": subset, "preferences": FOUR, "seeds": seeds, "seconds_per_request": seconds,
            "population_sizes": selected_sizes, "sources": sources, "hardware": stable_host,
            "budget_sufficiency": "not_established", "decoder": "exact_native_environment"}
    return config, records, plan, hardware


_BENCH = None


def _bench_init(config, records, ready_queue, release):
    global _BENCH
    torch.set_num_threads(1)
    _BENCH = config, records
    ready_queue.put(os.getpid())
    if not release.wait(120):
        raise RuntimeError("benchmark process startup timed out")


def _bench_job(_):
    config, records = _BENCH
    began, cpu = time.perf_counter(), time.process_time()
    traces = []
    steps = 0
    for record in records:
        for rule in ("earliest_finish", "same_wave_randomized"):
            solution = _rule_solution(record.instance, rule, random.Random(11))
            if rule == "same_wave_randomized":
                solution.resource_modes.clear()
            candidate = decode_solution(config, record.instance, solution, PreferenceVector(1/3, 1/3, 1/3))
            traces.append(candidate.action_trace_sha256)
            steps += len(candidate.actions)
    memory = None
    try:
        import psutil
        memory = psutil.Process().memory_info().rss
    except ImportError:
        pass
    return {"elapsed": time.perf_counter()-began, "cpu_seconds": time.process_time()-cpu, "decodes": len(traces),
            "steps": steps, "traces_sha256": digest(traces), "process_rss_bytes": memory}


def benchmark(config, records, plan, hardware, output, requested):
    identity = digest(plan)
    path = output / "benchmark.json"
    if path.exists():
        cached = read(path)
        if cached["identity"] != identity or cached["requested_grid"] != requested:
            raise ValueError("benchmark source/config/hardware or worker grid changed; use a new output directory")
        print(f"REUSE benchmark: recommended workers={cached['recommended_workers']}", flush=True)
        return cached
    grid = worker_grid(requested, hardware)
    print(f"BENCHMARK grid={grid}; physical={hardware['physical_cpus']}, logical={hardware['logical_cpus']}; RAM guard=1 GiB/worker, 25% reserved", flush=True)
    rows = []
    context = mp.get_context("spawn")
    for workers in grid:
        ready, release = context.Queue(), context.Event()
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_bench_init,
                                 initargs=(config, records, ready, release)) as pool:
            probes = [pool.submit(os.getpid) for _ in range(workers)]
            try:
                for _ in range(workers):
                    ready.get(timeout=120)
            finally:
                release.set()
            for f in probes:
                f.result()
            startup = time.perf_counter()-start
            start = time.perf_counter()
            results = [f.result() for f in [pool.submit(_bench_job, i) for i in range(workers*2)]]
            elapsed = time.perf_counter()-start
        assert len({r["traces_sha256"] for r in results}) == 1, "parallel width changed decoder results"
        row = {"workers": workers, "startup_seconds": startup, "steady_wall_seconds": elapsed,
               "completed_decodes": sum(r["decodes"] for r in results),
               "decodes_per_second": sum(r["decodes"] for r in results)/elapsed,
               "cpu_cores_used": sum(r["cpu_seconds"] for r in results)/elapsed,
               "median_job_seconds": float(np.median([r["elapsed"] for r in results])),
               "max_process_rss_bytes": max((r["process_rss_bytes"] or 0 for r in results), default=0) or None,
               "traces_sha256": results[0]["traces_sha256"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
        ready.close()
    assert len({r["traces_sha256"] for r in rows}) == 1
    best_rate = max(r["decodes_per_second"] for r in rows)
    # Within 5% of best throughput, prefer fewer competing workers.
    selected = min(r["workers"] for r in rows if r["decodes_per_second"] >= .95*best_rate)
    result = {"identity": identity, "requested_grid": requested, "tested_grid": grid, "rows": rows,
              "recommended_workers": selected, "interpretation": "fixed-candidate throughput only; not search-quality evidence"}
    atomic_json(path, result)
    return result


def preflight(plan, hardware, output, workers):
    if workers < 1 or workers > hardware["logical_cpus"]:
        raise ValueError("workers must be between 1 and the host logical CPU count")
    identity = {**plan, "workers": workers}
    identity["sha256"] = digest(identity)
    path = output / "identity.json"
    if path.exists() and read(path)["sha256"] != identity["sha256"]:
        raise ValueError("search identity or concurrency changed; use a new output directory")
    atomic_json(path, identity)
    result = {"identity": identity["sha256"], "passed": True, "workers": workers, "hardware_observation": hardware,
              "requests": len(plan["instance_ids"])*len(FOUR)*len(plan["seeds"])*len(SIZES),
              "seconds_per_request": plan["seconds_per_request"], "budget_sufficiency": "not_established"}
    atomic_json(output / "preflight.json", result)
    print(json.dumps(result), flush=True)
    return identity["sha256"]


def run(config, records, plan, identity, output, workers):
    tasks = [(config, record, tuple(pref), method, seed, plan["seconds_per_request"], None, size, identity,
              output / "requests" / method / f"{record.instance.instance_id}_{j}_{seed}.json")
             for record in records for j, pref in enumerate(FOUR) for seed in plan["seeds"] for method, size in SIZES.items()]
    before = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(search_job, task) for task in tasks}
        completed = 0
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                row = f.result()
                completed += 1
                print(f"DONE {completed}/{len(tasks)} {row['method']} {row['instance_id']} seed={row['seed']} success={row['success']} Q={row['quality']:.6f} evals={row['environment_evaluations']} CPU-cores={row.get('process_cpu_cores_used',0):.2f}", flush=True)
            print(f"HEARTBEAT elapsed={time.perf_counter()-before:.0f}s complete={completed}/{len(tasks)} active<={min(workers,len(pending))} queued>={max(0,len(pending)-workers)}", flush=True)


def report(output, identity):
    jobs = [read(p) for p in sorted((output / "requests").glob("*/*.json"))]
    expected = len(identity["instance_ids"])*len(FOUR)*len(identity["seeds"])*len(SIZES)
    cells = {(j["summary"]["method"], j["summary"]["instance_id"], tuple(j["summary"]["preference"]), j["summary"]["seed"]) for j in jobs}
    expected_cells = {(m, iid, tuple(pref), seed) for m in SIZES for iid in identity["instance_ids"] for pref in FOUR for seed in identity["seeds"]}
    if len(jobs) != expected or cells != expected_cells:
        raise ValueError(f"incomplete result matrix: {len(jobs)}/{expected}")
    for job in jobs:
        row = job["summary"]
        expected_job_id = digest([identity["sha256"], row["instance_id"], row["preference"], row["method"], row["seed"],
                                  identity["seconds_per_request"], None, SIZES[row["method"]]])
        if job["job_id"] != expected_job_id or row["seconds_budget"] != identity["seconds_per_request"]:
            raise ValueError("request source or budget differs from the frozen identity")
        good = [c for c in job["candidates"] if c["feasible"] and c["completed_seconds"] <= row["seconds_budget"]]
        if row["success"] != bool(good) or (good and abs(min(c["quality"] for c in good)-row["quality"]) > 1e-10):
            raise ValueError("delivery differs from the best candidate completed within the budget")
        if row["success"]:
            metrics = job["selected"]["metrics"]
            if not metrics["task_succeeded"] or metrics["sampling_truncated"] or metrics["schedule_violations"] or metrics["maximum_worker_fatigue"] > metrics["safe_fatigue_limit"]+1e-9:
                raise ValueError("selected schedule failed native safety/completion checks")
    summary = {"identity": identity["sha256"], "requests": len(jobs), "budget_sufficiency": "not_established", "methods": {}}
    for method in SIZES:
        rows = [j["summary"] for j in jobs if j["summary"]["method"] == method]
        summary["methods"][method] = {"requests": len(rows), "successes": sum(r["success"] for r in rows),
            "mean_Q": float(np.mean([r["quality"] for r in rows])),
            "mean_complete_evaluations": float(np.mean([r["environment_evaluations"] for r in rows])),
            "mean_process_cpu_cores": float(np.mean([r["process_cpu_cores_used"] for r in rows]))}
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "benchmark", "run", "report"), default="preflight")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "result/analysis/heuristics_seed11_20261019")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result/analysis/heuristics_1800_v1")
    parser.add_argument("--subset", choices=("seven", "tuning21"), default="seven")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--seconds", type=float, default=1800.)
    parser.add_argument("--workers", default="auto", help="auto uses same-identity benchmark recommendation, else 8")
    parser.add_argument("--worker-grid", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--remote-run", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(n < 1 for n in args.worker_grid):
        parser.error("positive seconds/grid and distinct nonempty seeds required")
    if args.stage == "run" and not args.remote_run:
        parser.error("run requires --remote-run on the remote experiment host")
    config, records, plan, hardware = inputs(args.source_dir, args.subset, args.seeds, args.seconds)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "benchmark":
        benchmark(config, records, plan, hardware, output, args.worker_grid)
        return
    if args.workers == "auto":
        cached = read(output / "benchmark.json") if (output / "benchmark.json").exists() else None
        if cached is not None and cached["identity"] != digest(plan):
            raise ValueError("benchmark identity differs; use a new output directory")
        workers = cached["recommended_workers"] if cached else min(8, worker_grid([8], hardware)[0])
    else:
        workers = int(args.workers)
    if args.stage == "preflight":
        preflight(plan, hardware, output, workers)
        return
    identity_path = output / "identity.json"
    precheck_path = output / "preflight.json"
    expected_identity = {**plan, "workers": workers}
    if not identity_path.exists() or not precheck_path.exists():
        raise ValueError("run/report requires a successful same-identity preflight")
    identity = read(identity_path)
    if identity["sha256"] != digest(expected_identity) or read(precheck_path).get("identity") != identity["sha256"] or not read(precheck_path).get("passed"):
        raise ValueError("preflight identity changed; use a new output directory")
    if args.stage == "run":
        run(config, records, plan, identity["sha256"], output, workers)
    report(output, identity)


if __name__ == "__main__":
    main()
