"""Describe preference response from saved trajectories on matched instances."""
from __future__ import annotations

from math import gcd
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.studies.schema12_variance_comparison_20261013 import ROOT, RUNS, OLD_LAST, read, safe, PREF, OBJECTIVES

OUT = ROOT / "result/analysis/schema12_variance_comparison_20261013/preference_response"
SOURCES = {Path(__file__), ROOT / "analysis/studies/schema12_variance_comparison_20261013.py"}


def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"schema10_best": RUNS["schema10"] / "final_sampled_instance_metrics.csv",
             "schema12_best": RUNS["schema12"] / "final_sampled_instance_metrics.csv", "schema10_last": OLD_LAST}
    frames = {label: read(path) for label, path in files.items()}
    SOURCES.update(files.values())
    cells, all_means = {}, {}
    for label, frame in frames.items():
        assert len(frame) == 3960 and not frame.duplicated(["instance_id", "preference_key", "sampling_repeat"]).any()
        success = frame[safe(frame)]
        mean = success.groupby(["instance_id", "preference_key"]).agg(
            {**{p: "first" for p in PREF}, **{o: "mean" for o in OBJECTIVES}})
        assert len(mean) == 1320  # at least one successful repeat in every cell
        all_means[label] = mean
        cells[label] = frame.assign(success=safe(frame)).groupby(["instance_id", "preference_key"]).success.sum()
    # Same instance/preference cells with three successes in both primary models.
    primary_complete = (cells["schema10_best"] == 3) & (cells["schema12_best"] == 3)
    complete_index = primary_complete[primary_complete].index
    correlations, levels, endpoints, ray_edges, edge_summary = [], [], [], [], []
    for label, frame in frames.items():
        mean = all_means[label]
        cohorts = {"all_cells_successful_repeats": mean}
        if label != "schema10_last":
            cohorts["both_best_models_three_successes"] = mean.loc[complete_index]
        for cohort, group in cohorts.items():
            for iid, instance in group.groupby(level="instance_id"):
                for pref, objective in zip(PREF, OBJECTIVES):
                    coefficient = spearmanr(instance[pref], instance[objective]).statistic
                    correlations.append({"model": label, "cohort": cohort, "instance_id": iid,
                                         "objective": objective, "preferences": len(instance), "spearman": coefficient})
        mean = mean.reset_index()
        for j, (pref, objective) in enumerate(zip(PREF, OBJECTIVES)):
            for weight, group in mean.groupby(pref):
                levels.append({"model": label, "objective": objective, "weight": weight,
                               "instance_preference_cells": len(group), "mean_target": group[objective].mean(),
                               "median_target": group[objective].median(),
                               "successes": int(safe(frame[frame[pref] == weight]).sum()),
                               "requests": len(frame[frame[pref] == weight])})
            target = frame[frame[pref] == 1]
            endpoints.append({"model": label, "endpoint": pref, "successes": int(safe(target).sum()), "requests": len(target),
                              **{objective: target.loc[safe(target), objective].mean() for objective in OBJECTIVES}})
            for iid, instance in mean.groupby("instance_id"):
                points = instance.copy()
                points["integer_weight"] = np.rint(points[pref] * 10).astype(int)
                other = [p for p in PREF if p != pref]
                endpoint = points[points.integer_weight == 10]
                ratios = {}
                for row in points[points.integer_weight < 10].itertuples(index=False):
                    values = row._asdict()
                    a, b = (int(round(values[p] * 10)) for p in other)
                    d = gcd(a, b)
                    assert d > 0
                    ratios.setdefault((a // d, b // d), []).append(values)
                for ratio, records in ratios.items():
                    records.extend(endpoint.to_dict("records"))
                    sequence = sorted(records, key=lambda r: r["integer_weight"])
                    for first, second in zip(sequence, sequence[1:]):
                        start_key = (iid, first["preference_key"])
                        end_key = (iid, second["preference_key"])
                        same_complete = start_key in complete_index and end_key in complete_index
                        delta = second[objective] - first[objective]
                        tolerance = 1e-8 * max(1, abs(first[objective]), abs(second[objective]))
                        ray_edges.append({"model": label, "instance_id": iid, "objective": objective,
                                          "other_weight_ratio": str(ratio), "weight_start": first[pref], "weight_end": second[pref],
                                          "target_start": first[objective], "target_end": second[objective], "target_delta": delta,
                                          "lower_or_equal": bool(delta <= tolerance), "strict_lower": bool(delta < -tolerance),
                                          "both_best_complete_cells": same_complete})
        boundaries = [(0, 1), (0, 2), (1, 2)]
        for first, second in boundaries:
            third = next(j for j in range(3) if j not in (first, second))
            boundary = mean[np.isclose(mean[PREF[third]], 0)]
            for _, group in boundary.groupby("instance_id"):
                group = group.sort_values(PREF[first])
                for i in (first, second):
                    target = group.sort_values(PREF[i])
                    values = target[OBJECTIVES[i]].to_numpy()
                    correct = np.diff(values) <= 1e-8
                    edge_summary.append({"model": label, "boundary": f"{first}_{second}", "objective": OBJECTIVES[i],
                                         "comparisons": len(correct), "lower_or_equal": int(correct.sum())})
    correlation = pd.DataFrame(correlations)
    corr_summary = correlation.groupby(["model", "cohort", "objective"]).spearman.agg(
        median="median", mean="mean", min="min", max="max", instances="size",
        negative_instances=lambda x: int((x < 0).sum())).reset_index()
    rays = pd.DataFrame(ray_edges)
    ray_summary = []
    for (model, objective), group in rays.groupby(["model", "objective"]):
        for cohort, selected in [("all_cells_successful_repeats", group),
                                  ("both_best_models_three_successes", group[group.both_best_complete_cells])]:
            if model == "schema10_last" and cohort.startswith("both"):
                continue
            ray_summary.append({"model": model, "objective": objective, "cohort": cohort, "comparisons": len(selected),
                                "lower_or_equal_fraction": selected.lower_or_equal.mean(), "strict_lower_fraction": selected.strict_lower.mean()})
    save("instance_correlations", correlations); save("correlation_summary", corr_summary)
    save("weight_levels", levels); save("endpoints", endpoints); save("fixed_ratio_transitions", ray_edges)
    save("fixed_ratio_summary", ray_summary); save("boundary_direction_counts", edge_summary)
    counts = {label: {"successful_requests": int(safe(frame).sum()), "requests": len(frame)} for label, frame in frames.items()}
    summary = {"counts": counts, "primary_complete_cells": len(complete_index), "total_cells": 1320,
               "correlation_summary": corr_summary.to_dict("records"), "fixed_ratio_summary": ray_summary,
               "policy_rollouts_added": 0,
               "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    plot(pd.DataFrame(levels))
    report(corr_summary, pd.DataFrame(ray_summary), pd.DataFrame(levels), len(complete_index))
    print(corr_summary.to_string(index=False))
    print(pd.DataFrame(ray_summary).to_string(index=False))


def plot(levels):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    colors = {"schema10_best": "#6b7280", "schema12_best": "#d97706", "schema10_last": "#2563eb"}
    for ax, objective, label in zip(axes, OBJECTIVES, ("Raw Flow", "Cost", "Worker-load variance")):
        for model, group in levels[levels.objective == objective].groupby("model"):
            ax.plot(group.weight, group.mean_target, marker=".", color=colors[model], label=model)
        ax.set_xlabel("Own objective preference weight"); ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / ("preference_response." + suffix), dpi=180)
    plt.close(fig)


def report(corr, rays, levels, complete_cells):
    text = "# 偏好响应补充分析（2026-10-13）\n\n"
    text += "使用相同20个实例×66偏好×3次采样，每个实例/偏好单元先平均其成功轨迹，再在每个实例内部计算自身权重与目标值的Spearman相关，最后汇报20实例中位数。负相关表示该权重提高时对应目标总体降低；该描述统计不等于最优策略证明。\n\n"
    text += "| 模型 | Flow相关中位数 | Cost相关中位数 | Variance相关中位数 |\n|---|---:|---:|---:|\n"
    for model in ("schema10_best", "schema12_best", "schema10_last"):
        selected = corr[(corr.model == model) & (corr.cohort == "all_cells_successful_repeats")].set_index("objective")
        text += f"| {model} | {selected.loc[OBJECTIVES[0], 'median']:.4f} | {selected.loc[OBJECTIVES[1], 'median']:.4f} | {selected.loc[OBJECTIVES[2], 'median']:.4f} |\n"
    text += f"\n进一步固定双方best均三次成功的 {complete_cells}/1320 个单元，降低不同失败样本造成的响应偏差。cohort=both_best_models_three_successes的结果保存在correlation_summary.csv；方向结论应与全单元结果一起读。旧last只作为补充，不纳入该双best成功集合。\n\n"
    text += "## 固定其余目标权重比例\n\n"
    text += "将两个其他权重的比值约分为整数比例，在同一实例、同比例路径上逐步增加目标自身权重，比较相邻目标均值是否下降或持平。纯目标端点加入每条路径。这样比单纯横向相关更能排除另外两个目标配比变化；但不同偏好具有独立derived采样种子、调度离散且每格仅三次采样，不能要求观测均值逐步严格单调。多个路径共用端点，比较次数不代表独立统计样本。\n\n"
    text += "| 模型 | 目标 | 沿路径下降/持平比例（全单元） |\n|---|---|---:|\n"
    for _, row in rays[rays.cohort == "all_cells_successful_repeats"].iterrows():
        text += f"| {row.model} | {row.objective} | {100*row.lower_or_equal_fraction:.2f}% |\n"
    text += "\n## 响应幅度与形态\n\n"
    text += "| 模型 | 目标 | 权重0时均值 | 权重1时均值 | 下降比例 |\n|---|---|---:|---:|---:|\n"
    for model in ("schema10_best", "schema12_best", "schema10_last"):
        for objective in OBJECTIVES:
            group = levels[(levels.model == model) & (levels.objective == objective)]
            low = group.loc[group.weight == 0, "mean_target"].iloc[0]
            high = group.loc[group.weight == 1, "mean_target"].iloc[0]
            text += f"| {model} | {objective} | {low:.4f} | {high:.4f} | {100*(1-high/low):.2f}% |\n"
    text += "\n权重0时聚合多个另外两目标配比，权重1时只有纯目标端点；两端幅度是描述性量，不是独立因果实验。每层先按实例/偏好平均成功repeat，保持实例等权，所以旧模型纯Variance等不完整成功格的值可能与按所有成功请求池平均略有差异。\n\n"
    text += "新best的Cost、Variance响应强于旧best；Cost平均值随权重增加呈清晰下降。Variance主要在权重0.2–0.5间切换到均衡方案，低权重时的负担集中是其偏好折中的一部分。Flow响应较弱且高权重有波动，纯Flow端点并非最低的整体平均Flow；Variance纯端点也没有保证低于所有邻近偏好。这些现象应视为响应的非平滑性与尚待优化的校准，不能从强相关就宣称每种偏好都最优。\n\n"
    text += "与旧last相比，新best的Cost响应更强，Variance响应相近，Flow响应更弱；旧last的完成率和质量折中见主对照报告。这里的相关只描述响应强弱，不表示对应目标绝对值更好。\n\n"
    text += "新best全测试失败5/3960，三个纯端点均60/60；故本次Cost响应增强不依赖纯Cost牺牲完成率。当前last仍有纯Cost验证退化，不能将best的响应结论直接套用到last。所有结论为单训练seed，复用已保存数据。\n\n"
    text += "![目标权重与目标均值](preference_response.png)\n\n复现：`python -m analysis.studies.schema12_preference_response_20261013`。summary.json记录数据哈希；fixed_ratio_transitions.csv包含单实例路径的方向明细。\n"
    (OUT / "report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
