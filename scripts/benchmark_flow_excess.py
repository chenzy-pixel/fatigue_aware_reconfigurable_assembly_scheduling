"""Paired deterministic environment timings for the independent Flow experiment."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.baselines import HeuristicPolicy
from configs import load_config
from data import load_dataset_split
from environment import AssemblySchedulingEnv
from result.io import write_csv, write_json


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "result/analysis/flow_excess_integration_20261008")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-episodes", type=int, default=1)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmup_episodes < 0:
        parser.error("repeats must be positive and warmup episodes nonnegative")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    configs = {"raw": load_config("configs/v8/universal.json"),
               "excess": load_config("configs/flow_excess/universal.json")}
    record = load_dataset_split(configs["raw"], "validation")[0]
    rows, traces = [], {}
    policy = HeuristicPolicy()
    for _ in range(args.warmup_episodes):
        for mode in configs:
            warmup = AssemblySchedulingEnv(configs[mode])
            warmup.reset(record.instance)
            while not warmup.task_done:
                warmup.step(policy.select_action(warmup))
    for repeat in range(args.repeats):
        for mode in ("raw", "excess") if repeat % 2 == 0 else ("excess", "raw"):
            env = AssemblySchedulingEnv(configs[mode])
            env.reset(record.instance)
            actions, elapsed = [], 0.0
            while not env.task_done:
                action = policy.select_action(env)
                started = time.perf_counter()
                env.step(action)
                elapsed += time.perf_counter() - started
                actions.append(action)
            traces[(repeat, mode)] = actions
            rows.append({"mode": mode, "repeat": repeat, "instance_id": record.instance.instance_id,
                         "decisions": len(actions), "step_and_observation_seconds": elapsed,
                         "seconds_per_decision": elapsed / len(actions)})
        assert traces[(repeat, "raw")] == traces[(repeat, "excess")]
    write_csv(output / "environment_timing.csv", rows)
    means = {mode: sum(row["seconds_per_decision"] for row in rows if row["mode"] == mode) / args.repeats
             for mode in ("raw", "excess")}
    summary = {"mean_seconds_per_decision": means, "excess_raw_ratio": means["excess"] / means["raw"],
               "repeats": args.repeats, "warmup_episodes_per_mode": args.warmup_episodes,
               "scope": "one fixed instance, alternating paired heuristic traces, step including observation; host-load dependent"}
    write_json(output / "environment_timing_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
