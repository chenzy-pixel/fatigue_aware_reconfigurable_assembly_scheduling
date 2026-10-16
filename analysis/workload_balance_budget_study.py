"""Historical diagnostics for actual-time balance under labor budgets.

No new schedules are generated. Reference totals are minima within the archived
Flow/Cost candidates and are not certified optima or deployment references.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from configs import load_config
from data import load_dataset_split
from environment.dynamics import quantize_to_ticks


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b + 1e-8) and np.any(a < b - 1e-8))


def main() -> None:
    old = ROOT / "result/analysis/objective_prestage_research_20261008"
    out = ROOT / "result/analysis/workload_balance_budget_20261011"
    out.mkdir(parents=True, exist_ok=True)
    hashes = json.loads((old / "historical_source_hashes.json").read_text())
    for entry in hashes:
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    instances = {r.instance.instance_id: r.instance for r in load_dataset_split(load_config("configs/default.json"), "test")}
    rows = []
    for objective in ("flow", "cost", "variance"):
        source = ROOT / f"result/runs/ablation_eval_full_{objective}_seed11_20261007_232505_102441"
        metrics = pd.read_csv(source / "instance_metrics.csv")
        recs = pd.read_csv(source / "reconfigurations.csv")
        for m in metrics.to_dict("records"):
            if not m["task_succeeded"]:
                continue
            iid, repeat = m["instance_id"], m["sampling_repeat"]
            inst = instances[iid]
            tasks = recs[(recs.instance_id == iid) & (recs.sampling_repeat == repeat)]
            by_machine = {machine.id: machine for machine in inst.machines}
            base = 0.
            for task in tasks.itertuples():
                module = task.source_module if task.stage == "DIS" else task.target_module
                spec = by_machine[task.machine_id].module_parameters[module]
                minutes = spec.disassembly_base_time if task.stage == "DIS" else spec.installation_base_time
                minimum = max(1, quantize_to_ticks(minutes, inst.resolution)) * inst.resolution
                assert task.duration >= minimum - 1e-8
                base += minimum
            loads = [float(tasks.loc[tasks.worker_id == worker.id, "duration"].sum()) for worker in inst.workers]
            total = sum(loads)
            assert np.isclose(np.var(loads), m["worker_load_variance"])
            row = {"objective": objective, "instance_id": iid, "repeat": repeat,
                   "flow": m["flow_time_objective"], "cost": m["reconfiguration_cost"],
                   "total": total, "base": base, "fatigue_extension": total-base,
                   "variance": np.var(loads), "reconfigurations": int(tasks.reconfiguration_id.nunique())}
            for h, load in enumerate(loads):
                row[f"load_{h}"] = load
            rows.append(row)
    frame = pd.DataFrame(rows)
    load_columns = sorted(c for c in frame if c.startswith("load_"))
    # Labelled workers are aligned across candidates for the same instance.
    fields = ["flow", "cost", *load_columns]
    references = []
    witnesses = []
    frame["dominated_in_archive"] = False
    for iid, group in frame.groupby("instance_id"):
        efficient = group[group.objective.isin(["flow", "cost"])]
        ref = efficient.sort_values(["total", "flow", "cost", "objective", "repeat"]).iloc[0]
        references.append({"instance_id": iid, "reference_objective": ref.objective,
                           "reference_repeat": int(ref["repeat"]), "total_ref": ref.total,
                           "variance_ref": ref.variance})
        for idx, candidate in group.iterrows():
            for other_idx, other in group.iterrows():
                if idx != other_idx and dominates(other[fields].to_numpy(float), candidate[fields].to_numpy(float)):
                    frame.loc[idx, "dominated_in_archive"] = True
                    witnesses.append({"instance_id": iid, "objective": candidate.objective,
                                      "repeat": int(candidate["repeat"]), "witness_objective": other.objective,
                                      "witness_repeat": int(other["repeat"]),
                                      "candidate_variance": candidate.variance, "witness_variance": other.variance,
                                      "candidate_total": candidate.total, "witness_total": other.total,
                                      "candidate_flow": candidate.flow, "witness_flow": other.flow,
                                      "candidate_cost": candidate.cost, "witness_cost": other.cost,
                                      "candidate_loads": json.dumps(candidate[load_columns].to_list()),
                                      "witness_loads": json.dumps(other[load_columns].to_list())})
                    break
    refs = pd.DataFrame(references)
    frame = frame.merge(refs, on="instance_id", validate="many_to_one")
    frame["labor_over_reference_pct"] = 100*(frame.total/frame.total_ref-1)
    scans, selected = [], []
    for epsilon in (0., .05, .1, .15, .2, .3, .5):
        eligible = frame[frame.total <= frame.total_ref*(1+epsilon)+1e-8]
        for name, group in frame.groupby("objective"):
            scans.append({"epsilon": epsilon, "objective": name, "successful_candidates": len(group),
                          "within_budget": int((group.total <= group.total_ref*(1+epsilon)+1e-8).sum())})
        for iid, group in eligible.groupby("instance_id"):
            # An archive dominator cannot have larger total labor; it remains
            # within the same budget. Keep one nondominated best-balance point.
            allowed = group[~group.dominated_in_archive]
            assert len(allowed) > 0
            choice = allowed.sort_values(["variance", "total", "flow", "cost"]).iloc[0]
            selected.append({"epsilon": epsilon, "instance_id": iid, "objective": choice.objective,
                             "repeat": int(choice["repeat"]), "total": choice.total,
                             "variance": choice.variance, "flow": choice.flow, "cost": choice.cost,
                             "labor_over_reference_pct": choice.labor_over_reference_pct})
    selected = pd.DataFrame(selected)
    # Matched trajectory comparisons for attributing total-time differences.
    variance = frame[frame.objective == "variance"].set_index(["instance_id", "repeat"])
    cost = frame[frame.objective == "cost"].set_index(["instance_id", "repeat"])
    common = variance.index.intersection(cost.index)
    delta = variance.loc[common, ["total", "base", "fatigue_extension", "reconfigurations"]] - cost.loc[common, ["total", "base", "fatigue_extension", "reconfigurations"]]
    assert np.allclose(delta.total, delta.base+delta.fatigue_extension)
    # Explicit counterexamples specify what the proposed guards do and do not guarantee.
    original = np.array([10., 10., 10., 10., 10., 0.])
    partial_padding = np.array([10., 10., 10., 10., 10., 5.])
    full_padding = np.full(6, 10.)
    redistributed = np.array([9., 9., 9., 9., 9., 8.])
    budget = 1.1*original.sum()
    assert partial_padding.sum() <= budget and partial_padding.var() < original.var()
    assert full_padding.sum() > budget and dominates(original, partial_padding)
    assert redistributed.sum() <= budget and redistributed.var() < original.var()
    assert not dominates(original, redistributed) and not dominates(redistributed, original)
    summary = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes_verified": len(hashes), "successful_rows": len(frame),
        "reference_semantics": "min total among successful archived Flow/Cost candidates per instance; diagnostic only",
        "means": frame.groupby("objective")[["total", "base", "fatigue_extension", "variance", "reconfigurations"]].mean().to_dict("index"),
        "variance_minus_cost_matched": {"pairs": len(common), **delta.mean().to_dict()},
        "archive_dominated_counts": frame.groupby("objective").dominated_in_archive.sum().astype(int).to_dict(),
        "dominated_candidates_with_lower_variance_than_witness": sum(w["candidate_variance"] < w["witness_variance"]-1e-8 for w in witnesses),
        "lower_variance_dominated_by_policy": {name: sum(w["objective"] == name and w["candidate_variance"] < w["witness_variance"]-1e-8 for w in witnesses) for name in ("flow", "cost", "variance")},
        "budget_scan": scans,
        "selected_candidate_means": selected.groupby("epsilon")[["total", "variance", "flow", "cost", "labor_over_reference_pct"]].mean().reset_index().to_dict("records"),
        "synthetic_checks": {"budget": budget, "original_variance": original.var(),
                             "partial_padding_variance": partial_padding.var(),
                             "redistributed_variance": redistributed.var(), "passed": True},
    }
    frame.to_csv(out / "candidate_audit.csv", index=False)
    pd.DataFrame(witnesses).to_csv(out / "dominance_witnesses.csv", index=False)
    refs.to_csv(out / "references.csv", index=False)
    pd.DataFrame(scans).to_csv(out / "budget_scan.csv", index=False)
    selected.to_csv(out / "selected_candidates.csv", index=False)
    (out / "source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
