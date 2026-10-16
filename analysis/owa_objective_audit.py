"""Offline OWA arithmetic, monotonicity and successful-log reconstruction.

No policy replay, new training, or causal attribution is performed. Stage starts
reconstruct committed increments; simultaneous starts use stable log order.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def weights(h: int, q: float) -> np.ndarray:
    w = q ** np.arange(h, dtype=float)
    return w / w.sum()


def owa(loads: np.ndarray, w: np.ndarray) -> float:
    return float(np.sort(loads)[::-1] @ w)


def main() -> None:
    out = ROOT / "result/analysis/owa_objective_audit_20261011"
    out.mkdir(parents=True, exist_ok=True)
    previous = ROOT / "result/analysis/workload_balance_budget_20261011"
    provenance = json.loads((previous / "source_hashes.json").read_text())
    for item in provenance:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    frame = pd.read_csv(previous / "candidate_audit.csv")
    cols = [f"load_{i}" for i in range(6)]
    h = len(cols)
    geometric = weights(h, .5)
    gini = (2*(h-np.arange(1, h+1))+1)/h**2
    loads = frame[cols].to_numpy()
    ordered = np.sort(loads, axis=1)[:, ::-1]
    frame["mean_load"] = loads.mean(axis=1)
    frame["gini_owa"] = ordered @ gini
    frame["geometric_05"] = ordered @ geometric
    frame["lp6"] = np.linalg.norm(loads, ord=6, axis=1)
    frame["max_load"] = loads.max(axis=1)
    # Mean + mean pairwise absolute-difference contribution for the stated Gini weights.
    differences = np.abs(loads[:, :, None]-loads[:, None, :]).sum(axis=(1, 2))
    assert np.allclose(frame.gini_owa, frame.mean_load+differences/(2*h**2))
    fields = ["flow", "cost", *cols]
    diagnostics = {}
    witness_count = 0
    for policy in ("cost", "flow", "variance"):
        subset = frame[frame.objective == policy]
        across = within = 0
        for _, row in subset.iterrows():
            pool = frame[frame.instance_id == row.instance_id]
            x = row[fields].to_numpy(float)
            y = pool[fields].to_numpy(float)
            dominated = np.all(y <= x+1e-8, axis=1) & np.any(y < x-1e-8, axis=1)
            across += int(dominated.any())
            within += int((dominated & (pool.objective.to_numpy() == policy)).any())
            for _, other in pool.loc[dominated].iterrows():
                new_y = np.array([other.flow, other.cost, other.geometric_05])
                new_x = np.array([row.flow, row.cost, row.geometric_05])
                assert np.all(new_y <= new_x+1e-8) and np.any(new_y < new_x-1e-8)
                witness_count += 1
        diagnostics[policy] = {"successful": len(subset), "dominated_across": across,
                               "dominated_within": within,
                               "ratio_total_to_reconfiguration_count": float(subset.total.sum()/subset.reconfigurations.sum()),
                               "mean_episode_ratio_total_to_count": float((subset.total/subset.reconfigurations).mean())}
    max_error = 0.
    steps = 0
    for policy in ("cost", "flow", "variance"):
        source = ROOT / f"result/runs/ablation_eval_full_{policy}_seed11_20261007_232505_102441"
        stages = pd.read_csv(source / "reconfigurations.csv")
        metrics = pd.read_csv(source / "instance_metrics.csv")
        for _, row in frame[frame.objective == policy].iterrows():
            records = stages[(stages.instance_id == row.instance_id) & (stages.sampling_repeat == row["repeat"])].sort_values("start", kind="stable")
            raw = metrics[(metrics.instance_id == row.instance_id) & (metrics.sampling_repeat == row["repeat"])].iloc[0]
            prefix = np.zeros(h)
            accumulated = 0.
            for task in records.itertuples():
                worker = int(task.worker_id[1:])-1
                before = owa(prefix, geometric)
                prefix[worker] += task.duration
                after = owa(prefix, geometric)
                increment = after-before
                assert geometric[-1]*task.duration-1e-10 <= increment <= geometric[0]*task.duration+1e-10
                accumulated -= increment
                steps += 1
            assert np.allclose(prefix, row[cols].to_numpy(float))
            assert np.isclose(prefix.var(), raw.worker_load_variance)
            assert np.isclose(prefix.sum(), row.total)
            err = abs(accumulated+owa(prefix, geometric))
            max_error = max(max_error, err)
            assert err < 1e-9
    rng = np.random.default_rng(37)
    for _ in range(1000):
        x = rng.uniform(0, 100, h)
        q = rng.uniform(.2, .95)
        w = weights(h, q)
        k = int(rng.integers(h))
        d = float(rng.uniform(.001, 10))
        y = x.copy()
        y[k] += d
        difference = owa(y, w)-owa(x, w)
        assert w[-1]*d-1e-10 <= difference <= w[0]*d+1e-10
        a, b = int(x.argmax()), int(x.argmin())
        d = float((x[a]-x[b])*rng.uniform(.001, .49))
        z = x.copy()
        z[a] -= d
        z[b] += d
        assert owa(z, w) < owa(x, w)
        sorted_x = np.sort(x)[::-1]
        concentration = sum((w[k-1]-w[k])*(sorted_x[:k].sum()-k*x.mean()) for k in range(1, h))
        assert np.isclose(owa(x, w), x.mean()+concentration)
    scan = []
    for q in (.25, .5, .75, .9, 1.):
        values = pd.Series(ordered @ weights(h, q), index=frame.index)
        scan.append({"q": q, **{name: float(values[frame.objective == name].mean()) for name in ("cost", "flow", "variance")}})
    summary = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "input_audit_sha256": hashlib.sha256((previous / "candidate_audit.csv").read_bytes()).hexdigest(),
               "original_file_hashes_verified": len(provenance), "successful_trajectories": len(frame),
               "weights_q05": geometric.tolist(), "gini_weights": gini.tolist(),
               "means": frame.groupby("objective")[["mean_load", "gini_owa", "geometric_05", "lp6", "max_load", "variance"]].mean().to_dict("index"),
               "diagnostics": diagnostics, "dominance_witness_pairs_checked": witness_count,
               "stage_increments_checked": steps, "max_telescoping_error": max_error,
               "random_algebra_checks": 1000, "q_scan": scan}
    frame.to_csv(out / "trajectory_metrics.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
