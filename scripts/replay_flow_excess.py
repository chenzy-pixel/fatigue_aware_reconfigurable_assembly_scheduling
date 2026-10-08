"""Closed-form replay of processing lower-bound credits on fixed schedules."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from configs import load_config
from data import load_dataset_split
from environment import AssemblySchedulingEnv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/analysis/flow_excess_replay_20261008"
SUFFIX = "seed11_20261007_232505_102441"
METHODS = ("completed_only", "proportional", "front_loaded")


def ticks(value: float, resolution: float) -> int:
    result = int(round(float(value) / resolution))
    if abs(result * resolution - float(value)) > 1e-7:
        raise ValueError(f"time {value} is not on grid {resolution}")
    return result


@dataclass(frozen=True)
class ProcessingInterval:
    operation_id: str
    order_id: str
    start: int
    observed_end: int
    planned_duration: int
    minimum_duration: int
    completed: bool
    progress_weight: float


@dataclass
class Replay:
    releases: np.ndarray
    completions: np.ndarray
    intervals: list[ProcessingInterval]
    terminal_tick: int
    resolution: float
    failure_flow_penalty: float
    task_failed: bool

    def state(self, grid: np.ndarray) -> dict[str, np.ndarray]:
        """Compute every value from timestamps, never accumulate credits."""
        grid = np.asarray(grid, dtype=np.int64)
        flow_ticks = np.maximum(
            0, np.minimum(grid[:, None], self.completions) - self.releases
        ).sum(axis=1, dtype=np.int64)
        credits = {name: np.zeros(len(grid), dtype=np.float64) for name in METHODS}
        completed_credit = np.zeros(len(grid), dtype=np.int64)
        progress = np.zeros(len(grid), dtype=np.float64)
        for op in self.intervals:
            elapsed = np.clip(grid - op.start, 0, op.observed_end - op.start)
            done = (grid >= op.observed_end) if op.completed else np.zeros(len(grid), dtype=bool)
            completed_credit += done.astype(np.int64) * op.minimum_duration
            progress += done * op.progress_weight
            # The completed contribution is integer-valued. Only active credit is fractional.
            credits["proportional"] += np.where(
                done, 0.0, op.minimum_duration * elapsed / op.planned_duration
            )
            credits["front_loaded"] += np.where(done, 0, np.minimum(elapsed, op.minimum_duration))
        credits["completed_only"] = completed_credit.astype(np.float64)
        credits["proportional"] += completed_credit
        credits["front_loaded"] += completed_credit
        result = {"tick": grid, "time": grid * self.resolution,
                  "raw_flow": flow_ticks * self.resolution, "progress": progress}
        for method, credit in credits.items():
            result[f"credit_{method}"] = credit * self.resolution
            result[f"excess_{method}"] = (flow_ticks - credit) * self.resolution
        return result

    def exact_proportional_credit_at(self, tick: int):
        """Rational tick certificate, independent of vectorized float arithmetic."""
        from fractions import Fraction
        credit = Fraction(0)
        for op in self.intervals:
            elapsed = max(0, min(tick, op.observed_end) - op.start)
            credit += Fraction(op.minimum_duration * elapsed, op.planned_duration)
        return credit


def build_replay(instance, metrics, schedule, native):
    terminal = ticks(metrics["makespan"], instance.resolution)
    n_orders = len(instance.orders)
    counts = {order.id: len(order.operations) for order in instance.orders}
    intervals = []
    seen = set()
    for row in schedule.to_dict("records"):
        opid = row["operation_id"]
        if opid in seen:
            raise ValueError(f"duplicate processing operation {opid}")
        seen.add(opid)
        oi = instance.operation_index[opid]
        mi = instance.machine_index[row["machine_id"]]
        actual = native.estimate_processing_ticks(oi, mi)
        minimum = min(native.estimate_processing_ticks(oi, index)
                      for index, machine in enumerate(instance.machines)
                      if instance.operations[oi].required_module in machine.module_parameters)
        start = ticks(row["start"], instance.resolution)
        end = ticks(row["end"], instance.resolution)
        partial = pd.notna(row.get("truncated")) and bool(row.get("truncated"))
        planned_end = ticks(row["planned_end"], instance.resolution) if partial else end
        assert actual == planned_end - start
        assert 0 < minimum <= actual and start <= end <= terminal
        assert row["order_id"] == instance.operations[oi].order_id
        assert start >= ticks(native._order_by_id(row["order_id"]).release_time, instance.resolution)
        assert end - start == ticks(row["duration"], instance.resolution)
        intervals.append(ProcessingInterval(opid, row["order_id"], start, end, actual,
                                            minimum, not partial, 1 / (n_orders * counts[row["order_id"]])))
    releases, completions = [], []
    for order in instance.orders:
        operations = sorted((op for op in intervals if op.order_id == order.id), key=lambda op: op.start)
        for previous, current in zip(operations, operations[1:]):
            assert previous.completed and previous.observed_end <= current.start
        complete = len(operations) == len(order.operations) and all(op.completed for op in operations)
        releases.append(ticks(order.release_time, instance.resolution))
        completions.append(max(op.observed_end for op in operations) if complete else terminal)
    unfinished = int(metrics["unfinished_orders"])
    failed = bool(metrics["task_failed"])
    penalty = unfinished * instance.unfinished_order_penalty if failed else 0.0
    replay = Replay(np.array(releases), np.array(completions), intervals, terminal,
                    instance.resolution, penalty, failed)
    final = replay.state(np.array([terminal]))
    assert np.isclose(final["raw_flow"][0] + penalty, metrics["flow_time_objective"], atol=1e-7)
    assert np.isclose(final["progress"][0], metrics["operation_progress"], atol=1e-10)
    return replay


def evaluate_grid(replay, grid, scales, identity):
    state = replay.state(grid)
    frame = pd.DataFrame(state)
    frame["failure_flow_penalty"] = 0.0
    frame["failure_reward_penalty"] = 0.0
    if replay.task_failed:
        terminal = frame.iloc[-1].copy()
        terminal["failure_flow_penalty"] = replay.failure_flow_penalty
        terminal["failure_reward_penalty"] = 2.0
        frame = pd.concat([frame, pd.DataFrame([terminal])], ignore_index=True)
    for method in METHODS:
        frame[f"excess_{method}"] += frame.failure_flow_penalty
        excess = frame[f"excess_{method}"].to_numpy()
        assert excess.min() >= -1e-8
        if method != "completed_only":
            assert np.diff(excess).min(initial=0) >= -1e-8
    summaries = []
    progress = frame.progress.to_numpy()
    failure = frame.failure_reward_penalty.to_numpy()
    for scale_name, scale in scales.items():
        for method in METHODS:
            q = frame[f"excess_{method}"].to_numpy() / (scale + frame[f"excess_{method}"].to_numpy())
            quality_reward = q[:-1] - q[1:]
            rewards = np.diff(progress) + quality_reward - failure[1:]
            expected = progress[-1] - progress[0] + q[0] - q[-1] - failure[-1]
            assert np.isclose(rewards.sum(), expected, atol=1e-10)
            if method != "completed_only":
                assert quality_reward.max(initial=0) <= 1e-10
            suffix = f"{scale_name}_{method}"
            frame[f"q_{suffix}"] = q
            frame[f"quality_reward_{suffix}"] = np.r_[0, quality_reward]
            frame[f"reward_{suffix}"] = np.r_[0, rewards]
            summaries.append({**identity, "scale_name": scale_name, "scale": scale,
                              "method": method, "grid_steps": len(rewards),
                              "flow_excess_terminal": frame[f"excess_{method}"].iloc[-1],
                              "credit_terminal": frame[f"credit_{method}"].iloc[-1],
                              "progress_terminal": progress[-1], "return": rewards.sum(),
                              "identity_error": abs(rewards.sum()-expected),
                              "excess_decrease_steps": int((np.diff(frame[f"excess_{method}"]) < -1e-8).sum()),
                              "quality_positive_steps": int((quality_reward > 1e-10).sum()),
                              "quality_reward_variance": np.var(quality_reward),
                              "total_reward_variance": np.var(rewards),
                              "max_abs_quality_reward": np.max(abs(quality_reward), initial=0),
                              "max_abs_total_reward": np.max(abs(rewards), initial=0)})
    return frame, summaries


def summarize(rows):
    frame = pd.DataFrame(rows)
    keys = ["arm", "instance_id", "sampling_repeat", "success", "grid", "scale_name"]
    base = frame[frame.method == "completed_only"].set_index(keys)
    smooth = frame[frame.method == "proportional"].set_index(keys)
    assert base.index.equals(smooth.index)
    pair = base[["scale", "grid_steps", "return"]].rename(columns={"return": "return_completed"})
    pair["return_proportional"] = smooth["return"]
    pair["return_difference"] = smooth["return"] - base["return"]
    for metric in ("quality_reward_variance", "total_reward_variance", "max_abs_quality_reward"):
        pair[f"{metric}_completed"] = base[metric]
        pair[f"{metric}_proportional"] = smooth[metric]
        pair[f"{metric}_ratio"] = smooth[metric] / base[metric].replace(0, np.nan)
    pair.reset_index().to_csv(OUT / "paired_trajectory_statistics.csv", index=False)
    group = pair.reset_index().groupby(["arm", "success", "grid", "scale_name"], dropna=False)
    summary = group.agg(
        trajectories=("return_difference", "size"),
        max_abs_return_difference=("return_difference", lambda values: abs(values).max()),
        quality_variance_ratio_median=("quality_reward_variance_ratio", "median"),
        total_variance_ratio_median=("total_reward_variance_ratio", "median"),
        quality_variance_completed_mean=("quality_reward_variance_completed", "mean"),
        quality_variance_proportional_mean=("quality_reward_variance_proportional", "mean"),
        total_variance_completed_mean=("total_reward_variance_completed", "mean"),
        total_variance_proportional_mean=("total_reward_variance_proportional", "mean"),
        total_variance_improved=("total_reward_variance_ratio", lambda values: int((values < 1).sum())),
    )
    summary.to_csv(OUT / "summary.csv")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-tick-traces", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(parents=True, exist_ok=True)
    config = load_config("configs/default.json")
    native = AssemblySchedulingEnv(config)
    instances = {record.instance.instance_id: record.instance for record in load_dataset_split(config, "test")}
    scale_file = ROOT / "result/analysis/objective_prestage_research_20261008/exploratory_flow_scale.json"
    scale_reference = json.loads(scale_file.read_text())
    scales = {"legacy": 1089.15, "validation_excess": scale_reference["flow_excess_scale"]}
    sources, rows, certificates = [], [], []
    for arm in ("full_flow", "full_cost", "full_variance"):
        directory = ROOT / f"result/runs/ablation_eval_{arm}_{SUFFIX}"
        inputs = {}
        for name in ("instance_metrics", "schedule", "reconfigurations"):
            path = directory / f"{name}.csv"
            inputs[name] = pd.read_csv(path)
            sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        metrics = inputs["instance_metrics"]
        assert len(metrics) == 60 and not metrics.duplicated(["instance_id", "sampling_repeat"]).any()
        for metric in metrics.to_dict("records"):
            iid, repeat = metric["instance_id"], metric["sampling_repeat"]
            instance = instances[iid]
            native.reset(instance, build_observation=False)
            s = inputs["schedule"]
            s = s[(s.instance_id == iid) & (s.sampling_repeat == repeat)]
            r = inputs["reconfigurations"]
            r = r[(r.instance_id == iid) & (r.sampling_repeat == repeat)]
            replay = build_replay(instance, metric, s, native)
            event_ticks = {0, replay.terminal_tick}
            event_ticks.update(tick for tick in replay.releases if tick <= replay.terminal_tick)
            for table in (s, r):
                event_ticks.update(ticks(value, instance.resolution) for field in ("start", "end") for value in table[field])
            identity = {"arm": arm, "instance_id": iid, "sampling_repeat": repeat,
                        "success": bool(metric["task_succeeded"]), "failed": replay.task_failed}
            total_lb_ticks = sum(min(native.estimate_processing_ticks(oi, mi)
                                     for mi, machine in enumerate(instance.machines)
                                     if op.required_module in machine.module_parameters)
                                 for oi, op in enumerate(instance.operations))
            exact_credit = replay.exact_proportional_credit_at(replay.terminal_tick)
            if identity["success"]:
                assert exact_credit == total_lb_ticks
            # Rational certificate for each completion boundary: left active credit equals right DONE credit.
            for op in replay.intervals:
                if op.completed:
                    from fractions import Fraction
                    assert Fraction(op.minimum_duration * (op.observed_end-op.start), op.planned_duration) == op.minimum_duration
            final = replay.state(np.array([replay.terminal_tick]))
            processing_ticks = sum(op.observed_end-op.start for op in replay.intervals)
            flow_ticks = ticks(final["raw_flow"][0], instance.resolution)
            waiting_ticks = flow_ticks-processing_ticks
            assert waiting_ticks >= 0
            from fractions import Fraction
            slow_excess_ticks = sum((Fraction((op.observed_end-op.start)*(op.planned_duration-op.minimum_duration),
                                             op.planned_duration) for op in replay.intervals), Fraction(0))
            assert waiting_ticks + slow_excess_ticks == flow_ticks-exact_credit
            certificates.append({**identity, "raw_flow_terminal": final["raw_flow"][0],
                                 "flow_failure_penalty": replay.failure_flow_penalty,
                                 "flow_reported": metric["flow_time_objective"],
                                 "total_lb_ticks": total_lb_ticks,
                                 "credit_proportional_ticks_numerator": exact_credit.numerator,
                                 "credit_proportional_ticks_denominator": exact_credit.denominator,
                                 "partial_operations": sum(not op.completed for op in replay.intervals),
                                 "waiting_minutes": waiting_ticks*instance.resolution,
                                 "slow_machine_excess_minutes": float(slow_excess_ticks)*instance.resolution,
                                 "proportional_decomposition_error": abs((waiting_ticks+float(slow_excess_ticks))*instance.resolution-final["excess_proportional"][0]),
                                 "failed_credit_difference_minutes": final["credit_proportional"][0]-final["credit_completed_only"][0]})
            for grid_name, grid in (("events", np.array(sorted(event_ticks))),
                                    ("ticks", np.arange(replay.terminal_tick+1))):
                frame, summaries = evaluate_grid(replay, grid, scales, {**identity, "grid": grid_name})
                rows.extend(summaries)
                if grid_name == "events" or args.save_tick_traces:
                    frame.to_csv(OUT / "traces" / f"{arm}_{iid}_repeat{repeat}_{grid_name}.csv", index=False)
                if arm == "full_flow" and iid == "test_easy_3000000" and repeat == 0 and grid_name == "ticks":
                    frame.to_csv(OUT / "representative_tick_trace.csv", index=False)
        print(f"replayed {arm}: 60 trajectories", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "trajectory_statistics.csv", index=False)
    pd.DataFrame(certificates).to_csv(OUT / "terminal_certificates.csv", index=False)
    sources.extend({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in (Path(__file__).resolve(), scale_file, ROOT / "data/manifests/v2/test/manifest.json",
                                ROOT/"environment/env.py",ROOT/"environment/types.py"))
    (OUT / "provenance.json").write_text(json.dumps({"sources": sources, "scale_reference": scale_reference,
        "preference": [1, 0, 0], "failure_reward_penalty": 2,
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "replay_grids": ["all recorded event times", "native processing tick grid"],
        "original_ppo_action_sequence_available": False}, indent=2), encoding="utf-8")
    summary = summarize(rows)
    print(summary.xs("validation_excess", level="scale_name").to_string(), flush=True)
    render_figure()


def render_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    trace = pd.read_csv(OUT/"representative_tick_trace.csv")
    stats = pd.read_csv(OUT/"paired_trajectory_statistics.csv")
    stats = stats[(stats.scale_name=="validation_excess") & (stats.grid=="events") & stats.success]
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(1,2,figsize=(11,3.6),constrained_layout=True)
    axes[0].plot(trace.time,trace.excess_completed_only,color="#D55E00",label="Completed-only",linewidth=1.2)
    axes[0].plot(trace.time,trace.excess_proportional,color="#0072B2",label="Proportional",linewidth=1.5)
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Excess Flow (minutes)")
    axes[0].legend(frameon=False)
    groups=[stats[stats.arm==arm].total_reward_variance_ratio.to_numpy()
            for arm in ("full_flow","full_cost","full_variance")]
    axes[1].boxplot(groups,tick_labels=["Flow\n(n=57)","Cost\n(n=60)","Variance\n(n=57)"],
                    showfliers=True,medianprops={"color":"#0072B2"})
    axes[1].axhline(1,color="0.5",linestyle="--",linewidth=0.8)
    axes[1].set_xlabel("Schedule source; successful trajectories")
    axes[1].set_ylabel("Reward variance ratio\n(proportional / completed-only)")
    fig.savefig(OUT/"replay_comparison.png",dpi=180)
    fig.savefig(OUT/"replay_comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
