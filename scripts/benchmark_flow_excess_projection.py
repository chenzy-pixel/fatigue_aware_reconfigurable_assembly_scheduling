"""Compare the preserved full WAIT kernel with its optimized isolated copy."""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import inspect
import json
import pickle
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from configs import load_config
from environment import AssemblySchedulingEnv
import environment.env as env_module
import environment.time_context as time_module
from scripts.benchmark_flow_runtime import states, compare_observations
from result.io import write_json

OUTPUT = ROOT / "result/audits/flow_excess_optimization_20261008"


@contextmanager
def projection(function):
    previous = env_module.project_wait_state
    env_module.project_wait_state = function
    try:
        yield
    finally:
        env_module.project_wait_state = previous


def compare_kernel(first, second):
    for field in ("operations", "machines", "workers", "reconfigurations", "_events",
                  "schedule_log", "reconfiguration_log", "_order_released", "_order_completion_tick",
                  "_active_committed_worker_tasks", "_machine_reconfiguration",
                  "_post_reconfiguration_process_count", "_flow_integral", "_flow_penalty",
                  "_reconfiguration_cost", "current_tick", "decision_type", "terminated",
                  "truncated", "terminal_reason", "_wait_masked_states", "_wait_reason_counts",
                  "_wait_mask_reason_counts", "_first_unrecoverable_deadlock_diagnostic"):
        assert getattr(first, field) == getattr(second, field), field
    np.testing.assert_array_equal(first._committed_worker_loads, second._committed_worker_loads)
    assert first.flow_excess_objective() == second.flow_excess_objective()


def main():
    reference_path = OUTPUT / "reference_project_wait_state.py"
    scope = {}
    exec(compile(reference_path.read_text(encoding="utf-8"), str(reference_path), "exec"),
         time_module.__dict__, scope)
    reference = scope["project_wait_state"]
    current = time_module.project_wait_state
    rows = []
    for label, env in states(load_config("configs/flow_excess/universal.json")):
        env._invalidate_resource_snapshot()
        certificate = env._wait_certificate() if not env.task_done else {"allowed": False}
        timing = {name: [] for name in ("before_wait", "after_wait", "before_observe", "after_observe")}
        if certificate["allowed"]:
            original = pickle.dumps(env)
            before = reference(env, certificate["wait_ticks"], settle_terminal=True, certificate=certificate)
            after = current(env, certificate["wait_ticks"], settle_terminal=True, certificate=certificate)
            assert pickle.dumps(env) == original
            compare_kernel(before, after)
        observations = {}
        for repeat in range(9):
            for name, function in (("before", reference), ("after", current)) if repeat % 2 == 0 else (("after", current), ("before", reference)):
                if certificate["allowed"]:
                    start = time.perf_counter()
                    function(env, certificate["wait_ticks"], settle_terminal=True, certificate=certificate)
                    timing[f"{name}_wait"].append(time.perf_counter() - start)
                with projection(function):
                    env._invalidate_resource_snapshot()
                    start = time.perf_counter()
                    observations[name] = env.observe()
                    timing[f"{name}_observe"].append(time.perf_counter() - start)
        compare_observations(observations["before"], observations["after"])
        row = {"state": label, "wait_legal": certificate["allowed"], "kernel_and_observations_equal": True}
        row.update({key: statistics.median(values[2:]) for key, values in timing.items() if values})
        rows.append(row)
    sums = {key: sum(row.get(key, 0) for row in rows) for key in timing}
    report = {"states": rows, "sum_medians_seconds": sums,
              "wait_time_reduction": 1 - sums["after_wait"] / sums["before_wait"],
              "observation_time_reduction": 1 - sums["after_observe"] / sums["before_observe"],
              "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
              "optimized_function_sha256": hashlib.sha256(inspect.getsource(current).encode()).hexdigest(),
              "scope": "same-process alternating exact-kernel/cold-observation comparison; not full training throughput"}
    write_json(OUTPUT / "projection_comparison.json", report)
    print({key: value for key, value in report.items() if key != "states"})


if __name__ == "__main__":
    main()
