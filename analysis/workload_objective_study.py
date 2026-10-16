"""Offline audit of candidate third objectives on versioned historical schedules.

This does not alter training or select scales from the test set. Equal-time
assignments have no recorded decision order, so max-load signal counts are
bounded over all their possible orders. Run from the repository root.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from configs import load_config
from data import load_dataset_split
from environment.fatigue_monitor import audit_fatigue


def main() -> None:
    source = ROOT / "result/analysis/objective_prestage_research_20261008"
    output = ROOT / "result/analysis/workload_objective_study_20261011"
    output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((source / "historical_source_hashes.json").read_text())
    for entry in provenance:
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"], entry["path"]
    old = pd.read_csv(source / "historical_trajectory_metrics.csv").set_index(
        ["objective", "instance_id", "sampling_repeat"]
    )
    config = load_config("configs/default.json")
    instances = {record.instance.instance_id: record.instance for record in load_dataset_split(config, "test")}
    rows = []
    for objective in ("flow", "cost", "variance"):
        directory = ROOT / f"result/runs/ablation_eval_full_{objective}_seed11_20261007_232505_102441"
        metrics = pd.read_csv(directory / "instance_metrics.csv")
        records = pd.read_csv(directory / "reconfigurations.csv")
        for metric in metrics.to_dict("records"):
            iid, repeat = metric["instance_id"], metric["sampling_repeat"]
            instance = instances[iid]
            tasks = records[(records.instance_id == iid) & (records.sampling_repeat == repeat)]
            ids = [worker.id for worker in instance.workers]
            index = {worker_id: i for i, worker_id in enumerate(ids)}
            loads = np.array([tasks.loc[tasks.worker_id == worker_id, "duration"].sum() for worker_id in ids])
            previous = old.loc[(objective, iid, repeat)]
            assert np.isclose(np.var(loads), previous["variance"])
            assert np.isclose(loads.max(), previous["max_load"])
            assert np.isclose(sum(loads**2), len(loads)*np.var(loads) + loads.sum()**2/len(loads))
            row = {"objective": objective, "instance_id": iid, "repeat": repeat,
                   "success": bool(metric["task_succeeded"]), "flow": metric["flow_time_objective"],
                   "cost": metric["reconfiguration_cost"], "makespan": metric["makespan"],
                   "total_load": loads.sum(), "variance": np.var(loads), "max_load": loads.max(),
                   "lp4": np.linalg.norm(loads, ord=4), "lp8": np.linalg.norm(loads, ord=8),
                   "max_plus_005_mean": loads.max() + 0.05*loads.mean(),
                   "stages": len(tasks)}
            monitor, segments = audit_fatigue(instance, tasks.to_dict("records"), metric["makespan"])
            assert np.isclose(monitor["fatigue_monitor_peak"], previous["peak_fatigue"])
            row["peak_fatigue"] = monitor["fatigue_monitor_peak"]
            for threshold in (0.2, 0.3, 0.4):
                shifted = replace(instance, fatigue=replace(instance.fatigue, maximum_safe_fatigue=threshold))
                audit, timeline = audit_fatigue(shifted, tasks.to_dict("records"), metric["makespan"])
                area = audit["fatigue_monitor_over_limit_area"]
                assert np.isclose(area, previous[f"area_{threshold:.2f}"])
                row[f"area_{threshold:.1f}"] = area
                final = {worker.id: worker.initial_fatigue for worker in instance.workers}
                for segment in timeline:
                    final[segment["worker_id"]] = segment["fatigue_end"]
                recovery = instance.fatigue.idle_recovery_rate_per_minute
                assert recovery > 0
                row[f"tail_{threshold:.1f}"] = sum(max(0., f-threshold)**2/(2*recovery) for f in final.values())
            # Successful logs contain completed planned durations. Failed logs
            # can contain truncation, so no committed-load signal inference there.
            if row["success"]:
                prefix = np.zeros(len(ids))
                counts = {name: [0, 0] for name in ("max", "lp8", "variance_reduction")}
                tied = 0
                for _, group in tasks.groupby("start", sort=True):
                    assignments = [(index[t.worker_id], float(t.duration)) for t in group.itertuples()]
                    assert len(assignments) <= len(ids)
                    assert len({i for i, _ in assignments}) == len(assignments)
                    tied += int(len(assignments) > 1)
                    possibilities = {name: [] for name in counts}
                    for order in itertools.permutations(assignments):
                        current = prefix.copy()
                        nonzero = {name: 0 for name in counts}
                        for i, duration in order:
                            before = (current.max(), np.linalg.norm(current, ord=8), np.var(current))
                            current[i] += duration
                            after = (current.max(), np.linalg.norm(current, ord=8), np.var(current))
                            nonzero["max"] += int(after[0]-before[0] > 1e-9)
                            nonzero["lp8"] += int(after[1]-before[1] > 1e-9)
                            nonzero["variance_reduction"] += int(after[2]-before[2] < -1e-9)
                        for name in counts:
                            possibilities[name].append(nonzero[name])
                    for name in counts:
                        counts[name][0] += min(possibilities[name])
                        counts[name][1] += max(possibilities[name])
                    for i, duration in assignments:
                        prefix[i] += duration
                assert np.allclose(prefix, loads)
                row["tied_start_groups"] = tied
                for name, (lower, upper) in counts.items():
                    row[f"{name}_count_lower"] = lower
                    row[f"{name}_count_upper"] = upper
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "trajectories.csv", index=False)
    summary = {"source": "historical seed11 test trajectories; 20 instances x 3 repeats x 3 policies",
               "audit_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "source_hashes_verified": len(provenance), "rows_verified": len(frame), "policies": {}}
    for name, group in frame.groupby("objective"):
        success = group[group.success]
        result = {"requested": len(group), "success": len(success)}
        for field in ("flow", "cost", "total_load", "variance", "max_load", "lp4", "lp8", "peak_fatigue"):
            result[field] = float(success[field].mean())
        for threshold in (0.2, 0.3, 0.4):
            area = success[f"area_{threshold:.1f}"]
            result[f"area_{threshold:.1f}_mean"] = float(area.mean())
            result[f"area_{threshold:.1f}_zero_fraction"] = float((area <= 1e-9).mean())
            result[f"tail_{threshold:.1f}_mean"] = float(success[f"tail_{threshold:.1f}"].mean())
        for field in ("max", "lp8", "variance_reduction"):
            result[f"{field}_signal_fraction_bounds"] = [
                float(success[f"{field}_count_{bound}"].sum()/success.stages.sum())
                for bound in ("lower", "upper")]
        summary["policies"][name] = result
    # Pair by instance and sampling repeat; repeats are not independent training seeds.
    left = frame[(frame.objective == "variance") & frame.success].set_index(["instance_id", "repeat"])
    right = frame[(frame.objective == "cost") & frame.success].set_index(["instance_id", "repeat"])
    keys = left.index.intersection(right.index)
    pair = {"common_success": len(keys), "instances": len(set(k[0] for k in keys)), "metrics": {}}
    for field in ("flow", "cost", "total_load", "max_load", "lp4", "lp8", "max_plus_005_mean", "area_0.2", "area_0.3", "area_0.4"):
        a, b = left.loc[keys, field], right.loc[keys, field]
        pair["metrics"][field] = {"variance_mean": float(a.mean()), "cost_mean": float(b.mean()),
                                  "mean_paired_delta": float((a-b).mean()),
                                  "wins": int((a < b-1e-9).sum()), "ties": int((abs(a-b) <= 1e-9).sum())}
    summary["variance_vs_cost"] = pair
    summary["within_instance_spearman_vs_cost"] = {}
    for field in ("total_load", "max_load", "lp4", "lp8", "area_0.3"):
        correlations = [group[[field, "cost"]].corr(method="spearman").iloc[0, 1]
                        for _, group in frame[frame.success].groupby("instance_id")
                        if group[field].nunique() > 1 and group.cost.nunique() > 1]
        summary["within_instance_spearman_vs_cost"][field] = {"instances": len(correlations),
                                                                  "median": float(np.median(correlations))}
    assert len(frame) == 180
    assert math.isclose(sum(loads**2), len(loads)*np.var(loads)+loads.sum()**2/len(loads))
    (output / "source_hashes.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
