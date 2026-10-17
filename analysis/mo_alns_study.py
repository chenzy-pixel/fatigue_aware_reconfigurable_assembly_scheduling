"""Anytime curves, predeclared budget selection, and delivery comparisons."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analysis.pareto_analysis import hypervolume_3d, normalize_objectives
from configs.formal_preferences import objective_scales
from result.io import write_csv


def read_jobs(folder):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(folder).glob("*.json"))]


def snapshot(job, seconds):
    rows = [r for r in job["candidates"] if r["completed_seconds"] <= seconds and r["feasible"]]
    winner = min(rows, key=lambda r: (r["quality"], r["action_trace_sha256"])) if rows else None
    events = [e for e in job["events"] if e.get("kind") == "neighbour" and e["completed_seconds"] <= seconds]
    s = job["summary"]
    return {"instance_id": s["instance_id"], "pressure_type": s["pressure_type"], "preference": s["preference"],
            "seed": s["seed"], "seconds": seconds, "success": winner is not None,
            "quality": winner["quality"] if winner else 1.0,
            "objectives": winner["objectives"] if winner else None,
            "completed_neighbourhoods": events[-1]["completed_neighbourhoods"] if events else 0,
            "weight_updates": events[-1]["weight_updates"] if events else 0,
            "evaluations": sum(r["completed_seconds"] <= seconds for r in job["candidates"])}


def platform_test(first, second):
    if len(first) != len(second) or not first:
        raise ValueError("budget comparison requires aligned nonempty requests")
    groups = defaultdict(list)
    newly_successful = 0
    for a, b in zip(first, second, strict=True):
        if (a["instance_id"], a["preference"], a["seed"]) != (b["instance_id"], b["preference"], b["seed"]):
            raise ValueError("unaligned request identity")
        newly_successful += int(b["success"] and not a["success"])
        groups[a["pressure_type"]].append(a["quality"] - b["quality"])
    means = {p: float(np.mean(values)) for p, values in groups.items()}
    p90 = float(np.quantile([v for values in groups.values() for v in values], .9))
    return {"passed": newly_successful == 0 and max(means.values()) <= .002 and p90 <= .01,
            "new_success_requests": newly_successful, "group_mean_quality_gain": means, "p90_quality_gain": p90}


def mechanism_test(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["pressure_type"]].append(r["completed_neighbourhoods"] >= 20 and r["weight_updates"] >= 4)
    rates = {p: float(np.mean(v)) for p, v in groups.items()}
    return {"passed": bool(rates) and min(rates.values()) >= .9, "group_pass_rates": rates}


def select_budget(output):
    from scripts.mo_alns_study import atomic_json
    output = Path(output)
    extended = read_jobs(output / "curves_1800")
    base = read_jobs(output / "curves_600")
    if len(base) != 252:
        raise ValueError(f"budget selection requires 252 completed tuning requests; found {len(base)}")
    if extended and len(extended) != 252:
        raise ValueError("1800-second extension is incomplete")
    jobs = extended or base
    horizon = 1800 if extended else 600
    times = (60, 180, 600, 1800) if extended else (60, 180, 600)
    table = {t: [snapshot(job, t) for job in jobs] for t in times}
    previous = 600 if extended else 180
    tail = platform_test(table[previous], table[horizon])
    mechanisms = {t: mechanism_test(table[t]) for t in times}
    needs_extension = not extended and (not tail["passed"] or not mechanisms[horizon]["passed"])
    selected = horizon
    if tail["passed"] and mechanisms[horizon]["passed"]:
        selected = next(t for t in times if mechanisms[t]["passed"] and platform_test(table[t], table[horizon])["passed"])
    sufficient = tail["passed"] and mechanisms[selected]["passed"]
    result = {"selected_seconds": selected, "reference_seconds": horizon,
              "needs_extension": needs_extension,
              "budget_sufficiency": "established_on_tuning_set" if sufficient else "not_established",
              "tail_comparison": tail, "mechanism_checks": mechanisms, "request_count": len(jobs),
              "thresholds": {"group_mean_quality_gain": .002, "p90_quality_gain": .01,
                             "new_success_requests": 0, "group_mechanism_pass_rate": .9}}
    atomic_json(output / "budget_selection.json", result)
    write_csv(output / "budget_curve.csv", [row for t in times for row in table[t]])
    return result


def clustered_improvement(jobs, baseline):
    grouped = defaultdict(list)
    pressures = {}
    for job in jobs:
        s = job["summary"]
        grouped[s["instance_id"]].append(s[baseline] - s["quality"])
        pressures[s["instance_id"]] = s["pressure_type"]
    if not grouped:
        return None
    values = {p: [] for p in sorted(set(pressures.values()))}
    for iid, gains in grouped.items():
        values[pressures[iid]].append(float(np.mean(gains)))
    rng = np.random.default_rng(20261017)
    draws = np.zeros(10000)
    n = len(grouped)
    stratified = min(map(len, values.values())) >= 2
    if stratified:
        for group in values.values():
            draws += rng.choice(group, size=(10000, len(group)), replace=True).sum(axis=1) / n
    else:
        # One confirmation instance per pressure cannot estimate within-stratum
        # variance. Resample whole instances rather than report a false zero CI.
        means = [float(np.mean(v)) for v in grouped.values()]
        draws = rng.choice(means, size=(10000, n), replace=True).mean(axis=1)
    return {"mean_gain": float(np.mean([np.mean(v) for v in grouped.values()])),
            "confidence_interval_95": np.quantile(draws, [.025, .975]).tolist(),
            "instance_count": n, "bootstrap": ("pressure-stratified" if stratified else "pooled") + " instance clusters, 10000 resamples",
            "interval_identifiable": n >= 2,
            "group_mean_gain": {p: float(np.mean(v)) for p, v in values.items()},
            "improved_instance_fraction": float(np.mean([np.mean(v) > 1e-12 for v in grouped.values()]))}


def report(output, config):
    from scripts.mo_alns_study import atomic_json
    output = Path(output)
    selection = json.loads((output / "budget_selection.json").read_text(encoding="utf-8"))
    methods = {name: read_jobs(output / name) for name in ("confirmed", "fixed_lns", "ppo_deadline", "ppo")}
    expected = {"confirmed": 273, "fixed_lns": 273, "ppo_deadline": 273, "ppo": 91}
    counts = {name: len(jobs) for name, jobs in methods.items()}
    if counts != expected:
        raise ValueError(f"confirmation matrix incomplete: {counts}, expected {expected}")
    rows, summaries, hv_rows = [], {}, []
    scales = objective_scales(config)
    for method, jobs in methods.items():
        cells = defaultdict(list)
        for job in jobs:
            s = job["summary"]
            rows.append({"method": method, **s})
            cells[s["instance_id"]].append(s)
        # An instance is the independent unit; preferences/search seeds are averaged within it.
        summaries[method] = {"request_count": len(jobs), "instance_count": len(cells),
                             "completion_rate": float(np.mean([s["success"] for vals in cells.values() for s in vals])),
                             "mean_all_request_loss": float(np.mean([s["quality"] for vals in cells.values() for s in vals])),
                             "median_seconds": float(np.median([s["search_seconds"] for vals in cells.values() for s in vals]))}
        if method != "ppo":
            summaries[method].update(vs_initial=clustered_improvement(jobs, "initial_quality"),
                                     vs_heuristic=clustered_improvement(jobs, "heuristic_quality"))
        for iid, vals in cells.items():
            seeds = sorted({s.get("seed", 11) for s in vals})
            for seed in seeds:
                selected = [s for s in vals if s.get("seed", 11) == seed]
                points = [normalize_objectives(s["objectives"], scales) for s in selected if s["success"]]
                hv_rows.append({"method": method, "instance_id": iid, "seed": seed,
                                "delivery_grid_hv": hypervolume_3d(points), "preference_count": len(selected),
                                "total_request_seconds": sum(s["search_seconds"] for s in selected)})
    # Pair within the same instance and preference, averaging ALNS seeds first.
    ppo_cells = {(j["summary"]["instance_id"], tuple(j["summary"]["preference"])): j["summary"] for j in methods["ppo"]}
    paired = []
    for method in ("confirmed", "fixed_lns", "ppo_deadline"):
        cells = defaultdict(list)
        for job in methods[method]:
            s = job["summary"]
            cells[(s["instance_id"], tuple(s["preference"]))].append(s)
        for key, vals in cells.items():
            ref = ppo_cells[key]
            for s in vals:
                common = ref["success"] and s["success"]
                paired.append({"method": method, "instance_id": key[0], "preference": key[1], "search_seed": s["seed"],
                               "ppo_success": ref["success"], "alns_success": s["success"], "common_success": common,
                               "alns_minus_ppo_all_request_loss": s["quality"] - ref["quality"],
                               "alns_minus_ppo_success_quality": s["quality"] - ref["quality"] if common else None,
                               "flow_delta": s["objectives"][0] - ref["objectives"][0] if common else None,
                               "cost_delta": s["objectives"][1] - ref["objectives"][1] if common else None,
                               "workload_delta": s["objectives"][2] - ref["objectives"][2] if common else None,
                               "seconds_ratio": s["search_seconds"] / ref["search_seconds"]})
    paid_groups, paid_times = defaultdict(list), defaultdict(float)
    for method, jobs in methods.items():
        for job in jobs:
            s = job["summary"]
            key = (method, s["instance_id"], s.get("seed", 11))
            paid_times[key] += s["search_seconds"]
            if method == "ppo":
                from environment.types import metrics_flow_objective
                from environment.workload import metrics_workload_objective
                points = [[metrics_flow_objective(r, config), r["reconfiguration_cost"], metrics_workload_objective(r, config)]
                          for r in job["raw_triple"] if r["task_succeeded"] and not r["schedule_violations"]]
            else:
                points = [r["objectives"] for r in job["archive"]]
            paid_groups[key].extend(points)
    paid_hv = [{"method": key[0], "instance_id": key[1], "seed": key[2],
                "all_paid_candidate_hv": hypervolume_3d([normalize_objectives(p, scales) for p in points]),
                "total_request_seconds": paid_times[key]} for key, points in paid_groups.items()]
    for method, jobs in methods.items():
        if method == "ppo":
            continue
        archive_rows = []
        for job in jobs:
            s = job["summary"]
            archive_rows.append({"method": method, "instance_id": s["instance_id"], "seed": s["seed"],
                                 "preference": s["preference"], "points": [x["objectives"] for x in job["archive"]]})
        atomic_json(output / f"{method}_paid_archives.json", archive_rows)
    confirm_mechanism = mechanism_test([j["summary"] for j in methods["confirmed"]])
    gain = summaries["confirmed"]["vs_initial"]
    confirmation_passed = (confirm_mechanism["passed"] and gain["interval_identifiable"]
                           and gain["confidence_interval_95"][0] > 0)
    result = {"budget": selection, "methods": summaries,
              "independent_confirmation_passed": confirmation_passed,
              "confirmation_mechanism": confirm_mechanism,
              "scope": "7 confirmation validation instances; 13 preferences; one fixed PPO training seed",
              "claim_status": "preliminary validation evidence; no automatic claim of superiority"}
    atomic_json(output / "comparison_summary.json", result)
    write_csv(output / "deliveries.csv", rows)
    write_csv(output / "paired_deliveries.csv", paired)
    write_csv(output / "delivery_hv.csv", hv_rows)
    write_csv(output / "paid_archive_hv.csv", paid_hv)
    tuning = read_jobs(output / ("curves_1800" if selection["reference_seconds"] == 1800 else "curves_600"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    checkpoints = [10, 30, 60, 180, 600] + ([1800] if selection["reference_seconds"] == 1800 else [])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    axes[0].plot(checkpoints, [np.mean([snapshot(j, t)["quality"] for j in tuning]) for t in checkpoints], marker="o")
    axes[1].plot(checkpoints, [np.mean([snapshot(j, t)["success"] for j in tuning]) for t in checkpoints], marker="o")
    axes[0].set_ylabel("Mean delivery loss (failure = 1)")
    axes[1].set_ylabel("Feasible delivery rate")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Seconds per independent request")
        ax.grid(alpha=.25)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(output / f"quality_time.{ext}", dpi=180)
    plt.close(fig)
    lines = ["# MO-ALNS 验证与交付比较", "", f"正式候选时限：{selection['selected_seconds']}秒；预算充分性：{selection['budget_sufficiency']}。", "",
             "本报告只覆盖7个独立确认验证实例、13个偏好及一个固定PPO训练模型。", "",
             "| 方法 | 请求数 | 完成率 | 全请求损失 | 中位求解秒 |", "|---|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['request_count']} | {s['completion_rate']:.4f} | {s['mean_all_request_loss']:.6f} | {s['median_seconds']:.3f} |")
    lines += ["", "共同成功的三目标、Q差值及耗时比见paired_deliveries.csv；交付端点HV见delivery_hv.csv。",
              "短时限结果仅描述PPO交付时限内的可用方案；充分搜索质量和自适应贡献按对应对照分别解释。",
              "失败填充值与成功集合Q同时报告；未满足平台及机制门槛时，不宣布搜索已收敛。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
