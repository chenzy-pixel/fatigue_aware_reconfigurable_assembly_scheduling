"""Verify production OWA against the recorded 174 successful historical schedules."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from environment.workload import geometric_owa_weights, workload_delta, workload_value
from result.io import write_json


def main() -> None:
    source = ROOT / "result/analysis/workload_balance_budget_20261011"
    hashes = json.loads((source / "source_hashes.json").read_text())
    for item in hashes:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    frame = pd.read_csv(source / "candidate_audit.csv")
    prior = json.loads((ROOT / "result/analysis/owa_objective_audit_20261011/summary.json").read_text())
    columns = [f"load_{i}" for i in range(6)]
    cfg = {"workload_objective": {"mode": "geometric_owa", "q": .5}}
    weights = geometric_owa_weights(6, .5)
    means, steps, error, witnesses = {}, 0, 0., []
    for q in (.25, .5, .75):
        own = {"workload_objective": {"mode": "geometric_owa", "q": q}}
        expected = next(row for row in prior["q_scan"] if row["q"] == q)
        means[str(q)] = {}
        for policy in ("cost", "flow", "variance"):
            values = [workload_value(loads, own) for loads in frame.loc[frame.objective == policy, columns].to_numpy()]
            means[str(q)][policy] = float(np.mean(values))
            assert np.isclose(means[str(q)][policy], expected[policy], rtol=0, atol=1e-10)
    for policy in ("cost", "flow", "variance"):
        directory = ROOT / f"result/runs/ablation_eval_full_{policy}_seed11_20261007_232505_102441"
        stages = pd.read_csv(directory / "reconfigurations.csv")
        for _, row in frame.loc[frame.objective == policy].iterrows():
            records = stages.loc[(stages.instance_id == row.instance_id) & (stages.sampling_repeat == row["repeat"])].sort_values("start", kind="stable")
            prefix, cumulative = np.zeros(6), 0.
            for stage in records.itertuples():
                after = prefix.copy()
                after[int(stage.worker_id[1:]) - 1] += stage.duration
                increment = workload_delta(prefix, after, cfg)
                assert weights[-1] * stage.duration - 1e-10 <= increment <= weights[0] * stage.duration + 1e-10
                cumulative += increment
                prefix = after
                steps += 1
            np.testing.assert_allclose(prefix, row[columns].to_numpy(float), rtol=0, atol=1e-8)
            error = max(error, abs(cumulative - workload_value(prefix, cfg)))
            assert error < 1e-9
    for _, worse in frame.iterrows():
        pool = frame.loc[frame.instance_id == worse.instance_id]
        for _, better in pool.iterrows():
            a = better[["flow", "cost", *columns]].to_numpy(float)
            b = worse[["flow", "cost", *columns]].to_numpy(float)
            if np.all(a <= b + 1e-8) and np.any(a < b - 1e-8):
                oa = np.array([a[0], a[1], workload_value(a[2:], cfg)])
                ob = np.array([b[0], b[1], workload_value(b[2:], cfg)])
                assert np.all(oa <= ob + 1e-8) and np.any(oa < ob - 1e-8)
                witnesses.append({"instance_id": worse.instance_id,
                                  "dominator": f"{better.objective}:repeat_{int(better['repeat'])}",
                                  "dominated": f"{worse.objective}:repeat_{int(worse['repeat'])}"})
    assert len(frame) == prior["successful_trajectories"] == 174
    assert steps == prior["stage_increments_checked"] == 5536
    assert len(witnesses) == prior["dominance_witness_pairs_checked"] == 82
    result = {"successful_trajectories": len(frame), "stage_increments_checked": steps,
              "maximum_telescoping_error": error, "means": means,
              "dominance_witness_pairs_checked": len(witnesses), "witnesses": witnesses,
              "verified_historical_files": len(hashes),
              "production_module_sha256": hashlib.sha256((ROOT / "environment/workload.py").read_bytes()).hexdigest(),
              "input_audit_sha256": hashlib.sha256((source / "candidate_audit.csv").read_bytes()).hexdigest()}
    write_json(ROOT / "result/audits/workload_owa/historical_replay.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"witnesses", "means"}}, indent=2))


if __name__ == "__main__":
    main()
