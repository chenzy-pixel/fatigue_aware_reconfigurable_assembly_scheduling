"""Instance-clustered comparison statistics; failures remain visible."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from agent.comparators.common import atomic_json


def snapshot(job, seconds):
    candidates = [c for c in job["candidates"] if c["completed_seconds"] <= seconds and c["feasible"]]
    best = min(candidates, key=lambda c: (c["quality"], c["action_trace_sha256"])) if candidates else None
    iterations = [e for e in job["events"] if e.get("kind") == "iteration" and e["completed_seconds"] <= seconds]
    return {"success": best is not None, "quality": best["quality"] if best else 1.,
            "objectives": best["objectives"] if best else None,
            "iterations": max((max(e.get("completed_generations", 0), e.get("completed_scans", 0)) for e in iterations), default=0),
            "local_search_evaluations": max((e.get("local_search_evaluations", 0) for e in iterations), default=0)}


def plateau(first, second):
    groups = defaultdict(list)
    gains = []
    new_success = 0
    for a, b in zip(first, second):
        groups[a["pressure_type"]].append(a["quality"]-b["quality"])
        gains.append(a["quality"]-b["quality"])
        new_success += bool(b["success"] and not a["success"])
    return bool(gains and not new_success and max(np.mean(g) for g in groups.values()) <= .002
                and np.quantile(gains, .9) <= .01)


def clustered_interval(rows, value, *, samples=10000):
    groups = defaultdict(list)
    pressures = {}
    for row in rows:
        groups[row["instance_id"]].append(float(value(row)))
        if "pressure_type" in row:
            pressures[row["instance_id"]] = row["pressure_type"]
    if not groups:
        return None
    ids = sorted(groups)
    values = np.array([np.mean(groups[i]) for i in ids])
    rng = np.random.default_rng(20261017)
    strata = defaultdict(list)
    for i, iid in enumerate(ids):
        strata[pressures.get(iid, "all")].append(i)
    stratified = len(pressures) == len(ids) and min(map(len, strata.values())) >= 2
    if stratified:
        sampled = [values[np.array(ix)[rng.integers(0, len(ix), (samples, len(ix)))]] for ix in strata.values()]
        means = np.concatenate(sampled, axis=1).mean(1)
    else:
        means = values[rng.integers(0, len(values), (samples, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "instances": len(values), "bootstrap": "pressure_stratified_whole_instance" if stratified else "pooled_whole_instance_small_sample"}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def report(output, methods):
    output = Path(output)
    rows = []
    for path in sorted((output/"confirmation").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "summary" in payload:
            rows.append(payload["summary"])
    if not rows:
        raise ValueError("confirmation deliveries missing")
    expected = {"ppo", "ppo_greedy", "dcan", "dcan_greedy", "rule_ddqn", "nsga2_ls", "moead", "mo_alns",
                "nsga2_ls_ppo_deadline", "moead_ppo_deadline", "mo_alns_ppo_deadline"}
    counts = {method: sum(r["method"] == method for r in rows) for method in expected}
    if any(v != 273 for v in counts.values()) or set(r["method"] for r in rows) != expected:
        raise ValueError(f"incomplete independent confirmation matrix: {counts}")
    keys = [(r["method"], r["instance_id"], tuple(r["preference"]), r["seed"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate independent confirmation delivery")
    per_arm_keys = {m: {(r["instance_id"], tuple(r["preference"]), r["seed"]) for r in rows if r["method"] == m} for m in expected}
    if any(k != per_arm_keys["ppo"] for k in per_arm_keys.values()):
        raise ValueError("confirmation arms have different instance/preference/seed coverage")
    summaries = {}
    for method in sorted({r["method"] for r in rows}):
        own = [r for r in rows if r["method"] == method]
        summaries[method] = {"deliveries": len(own), "success_rate": np.mean([r["success"] for r in own]).item(),
                             "all_coverage_loss": clustered_interval(own, lambda r: r["quality"] if r["success"] else 1.),
                             "mean_seconds": float(np.mean([r["search_seconds"] for r in own]))}
    pairings = []
    # Average training/search replicates within each instance-preference before
    # pairing methods. Replicate IDs are not matched across different algorithms.
    cells = defaultdict(list)
    for row in rows:
        cells[(row["method"], row["instance_id"], tuple(row["preference"]))].append(row)
    reference = {key[1:]: value for key, value in cells.items() if key[0] == "ppo"}
    for method in summaries:
        paired = []
        for key, ref in reference.items():
            arm = cells.get((method, *key))
            if not arm:
                continue
            paired.append({"instance_id": key[0], "delta": float(np.mean([r["quality"] for r in arm])-np.mean([r["quality"] for r in ref])),
                           "objective_delta": (np.mean([r["objectives"] for r in arm], axis=0)-np.mean([r["objectives"] for r in ref], axis=0)).tolist() if all(r["success"] for r in arm+ref) else None,
                           "common_success": all(r["success"] for r in arm+ref)})
        pairings.append({"method": method, "loss_difference_to_ppo": clustered_interval(paired, lambda r: r["delta"]),
                         "common_success_difference": clustered_interval([r for r in paired if r["common_success"]], lambda r: r["delta"]),
                         "common_success_objective_differences": [clustered_interval([r for r in paired if r["common_success"]], lambda r, i=i: r["objective_delta"][i]) for i in range(3)]})
    write_csv(output/"paired_deliveries.csv", rows)
    hv_rows = []
    from analysis.pareto_analysis import hypervolume_3d
    from configs.formal_preferences import objective_scales
    from analysis.pareto_analysis import normalize_objectives
    for (method, iid, seed), group in _by_front(rows).items():
        points = [r["objectives"] for r in group if r["success"]]
        if points:
            # Objectives have already been normalized at the same frozen scales
            # when recorded in the delivery row.
            vectors = [r["normalized_objectives"] for r in group if r["success"]]
            hv = hypervolume_3d(vectors, reference=(1., 1., 1.))
        else:
            hv = 0.
        hv_rows.append({"method": method, "instance_id": iid, "seed": seed, "grid_points": len(group), "delivery_hv": hv})
    write_csv(output/"delivery_hv.csv", hv_rows)
    paid = defaultdict(list)
    costs = defaultdict(float)
    gains = defaultdict(list)
    for path in sorted((output/"confirmation").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["summary"]
        key = (row["method"], row["instance_id"], row["seed"])
        costs[key] += row["search_seconds"]
        if "archive" in payload:
            paid[key].extend(c["normalized_objectives"] for c in payload["archive"])
        else:
            # Every sampled trajectory belongs to the paid pool, not only the
            # delivered one; use the study's fixed physical scales.
            identity = json.loads((output/"identity.json").read_text(encoding="utf-8"))
            scales = objective_scales(identity["config"])
            paid[key].extend(normalize_objectives(r["objectives"], scales) for r in payload["raw"] if r["success"])
        if row["method"] in ("nsga2_ls", "moead", "mo_alns"):
            initial = row["initial_quality"] if row["initial_quality"] is not None else 1.
            gains[row["method"]].append({"instance_id": row["instance_id"], "gain": initial-row["quality"]})
    write_csv(output/"paid_archive_hv.csv", [{"method": k[0], "instance_id": k[1], "seed": k[2],
        "paid_archive_hv": hypervolume_3d(v, reference=(1., 1., 1.)), "total_request_seconds": costs[k]}
        for k, v in sorted(paid.items())])
    improvements = {m: clustered_interval(g, lambda r: r["gain"]) for m, g in gains.items()}
    summary = {"methods": summaries, "paired": pairings, "confirmation_instances": 7,
               "search_improvement": improvements,
               "independent_search_improvement_confirmed": {m: v["ci95"][0] > 0 for m, v in improvements.items()},
               "interpretation": "small independent confirmation; search seeds and training seeds are different replication units"}
    atomic_json(output/"comparison_summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    names = sorted(summaries)
    ax.bar(range(len(names)), [summaries[m]["all_coverage_loss"]["mean"] for m in names])
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Whole-instance mean delivered Q / failure loss")
    fig.tight_layout()
    for extension in ("svg", "pdf", "png"):
        fig.savefig(output/f"comparison.{extension}")
    plt.close(fig)
    lines = ["# 四类对照独立确认", "", "训练种子、搜索种子和采样重复分别报告；以下为小规模确认，不代表正式测试成绩。", "",
             "| 方法 | 交付数 | 成功率 | 全覆盖损失 | 平均秒 |", "|---|---:|---:|---:|---:|"]
    for method, item in summaries.items():
        lines.append(f"| {method} | {item['deliveries']} | {item['success_rate']:.4f} | {item['all_coverage_loss']['mean']:.6f} | {item['mean_seconds']:.3f} |")
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _by_front(rows):
    result = defaultdict(list)
    for row in rows:
        result[(row["method"], row["instance_id"], row["seed"])].append(row)
    return result
