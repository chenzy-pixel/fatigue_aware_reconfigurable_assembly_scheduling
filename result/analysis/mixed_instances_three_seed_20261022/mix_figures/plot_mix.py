"""Export Mix-only figures from the audited three-seed results; no evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
DATA = OUT.parent
ROOT = DATA.parents[2]
SEEDS = (11, 23, 37)
METHOD = "HGNN-PPO"
COLORS = {11: "#0072B2", 23: "#D55E00", 37: "#009E73"}
MARKERS = {11: "o", 23: "s", 37: "^"}
INPUTS: dict[str, dict] = {}
EXPORTS = []
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--figure", choices=["01_mix_reward", "02_mix_convergence",
                                       "03_mix_preference_response", "04_hgnnppo_heuristic_comparison"],
                    help="Export a single figure and refresh the gallery metadata.")
ARGS = parser.parse_args()

plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
})


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    INPUTS[str(path.relative_to(ROOT)).replace("\\", "/")] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame),
    }
    return frame


def decorate(ax, *, episode=False):
    ax.grid(axis="y", color="#E2E6EA", linewidth=0.7)
    ax.tick_params(length=3, color="#78838C")
    if episode:
        ax.set_xlim(0, 2080)
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.set_xlabel("训练轮次（Episode）")


def seed_legend(fig, *, best=False):
    handles = [Line2D([0], [0], color=COLORS[s], marker=MARKERS[s],
                      linewidth=2, markersize=5, label=f"Seed {s}") for s in SEEDS]
    if best:
        handles.append(Line2D([0], [0], linestyle="", marker="*", markersize=12,
                              color="#424D56", label="协议 best"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               frameon=False, ncol=len(handles), columnspacing=2.5)


def save(fig, name, caption):
    if ARGS.figure and name != ARGS.figure:
        plt.close(fig)
        return {"stem": name, "caption": caption}
    for ext in ("png", "pdf", "svg"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.18)
        EXPORTS.append({"path": path.name, "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    return {"stem": name, "caption": caption}


S = read_csv(DATA / "run_summary.csv").set_index("seed")
V = read_csv(DATA / "validation_curves.csv")
D = read_csv(DATA / "diagnostic_curves.csv")
P = read_csv(DATA / "preference_weight_means.csv")
P = P.loc[P.arm.eq("mixed")].copy()
H = read_csv(DATA / "heuristic_preferences_overall.csv")
assert tuple(sorted(S.index)) == SEEDS
assert len(V) == 60 and len(D) == 12 and len(P) == 99
assert set(V.algorithm_seed) == set(D.algorithm_seed) == set(P.seed) == set(SEEDS)
assert V.sample_aggregation.eq("best_of_3_v1").all()
assert D.sample_aggregation.eq("best_of_3_v1").all()
assert S.test_completion.eq(1).all()
assert not V.duplicated(["algorithm_seed", "episode"]).any()
assert not D.duplicated(["algorithm_seed", "episode"]).any()

T = {}
for seed in SEEDS:
    t = read_csv(ROOT / "result" / "runs" / S.loc[seed, "run"] / "train_log.csv")
    t = t.sort_values("episode")
    assert np.array_equal(t.episode.to_numpy(), np.arange(1, 2001))
    assert np.allclose(
        t.reward,
        t.reward_operation_progress + t.reward_quality + t.reward_failure
        + t.reward_feasibility_shaping, rtol=0, atol=1e-10)
    T[seed] = t

figures = []

# Training logs are single sampled trajectories, distinct from best-of-three evaluation.
fig, axs = plt.subplots(2, 1, figsize=(11.8, 7.6), sharex=True)
fig.suptitle(f"{METHOD}：三 seed 训练 Reward", y=0.985, fontsize=17, fontweight="bold")
seed_legend(fig)
for seed in SEEDS:
    t = T[seed]
    axs[0].plot(t.episode, t.reward.rolling(100, min_periods=100).mean(),
                color=COLORS[seed], linewidth=1.8)
    axs[1].plot(t.episode, t.task_succeeded.rolling(100, min_periods=100).mean() * 100,
                color=COLORS[seed], linewidth=1.8)
axs[0].set_title("(a) 实际训练 Reward", loc="left", pad=10)
axs[0].set_ylabel("Reward（100 轮均值）")
axs[1].set_title("(b) 同批训练轨迹成功率", loc="left", pad=10)
axs[1].set_ylabel("训练成功率（%）")
axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
for ax in axs:
    decorate(ax, episode=True)
axs[0].set_xlabel("")
fig.subplots_adjust(top=0.83, bottom=0.13, left=0.1, right=0.97, hspace=0.34)
caption = "固定混合实例训练；两图均为完整 100 轮窗口的滑动平均，前 99 轮不绘制。每个 seed 训练 2000 轮。"
fig.text(0.1, 0.035, caption, fontsize=10, color="#52616B")
figures.append(save(fig, "01_mix_reward", caption))

# Checkpoint selection is lexicographic in completion then Q, not simply minimum Q.
fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.3), sharex=True)
fig.suptitle(f"{METHOD}：验证集收敛与独立诊断", y=0.985, fontsize=17, fontweight="bold")
seed_legend(fig, best=True)
qmin = min(V.preference_balanced_quality_score.min(), D.preference_balanced_quality_score.min())
qmax = max(V.preference_balanced_quality_score.max(), D.preference_balanced_quality_score.max())
cmin = min(V.completion_rate.min(), D.overall_completion_rate.min()) * 100
for seed in SEEDS:
    v = V.loc[V.algorithm_seed.eq(seed)].sort_values("episode")
    d = D.loc[D.algorithm_seed.eq(seed)].sort_values("episode")
    style = dict(color=COLORS[seed], marker=MARKERS[seed], markersize=4, linewidth=1.6)
    axs[0, 0].plot(v.episode, v.preference_balanced_quality_score, **style)
    axs[0, 1].plot(v.episode, v.completion_rate * 100, **style)
    axs[1, 0].plot(d.episode, d.preference_balanced_quality_score, **style)
    axs[1, 1].plot(d.episode, d.overall_completion_rate * 100, **style)
    best_ep = int(S.loc[seed, "best_episode"])
    best_row = v.loc[v.episode.eq(best_ep)].iloc[0]
    assert np.isclose(best_row.preference_balanced_quality_score, S.loc[seed, "best_val_Q"])
    for ax, y in [(axs[0, 0], best_row.preference_balanced_quality_score),
                  (axs[0, 1], best_row.completion_rate * 100)]:
        ax.scatter([best_ep], [y], marker="*", s=205, color=COLORS[seed],
                   edgecolor="white", linewidth=0.9, zorder=6)
for ax, title in zip(axs.flat, ["(a) 选模验证 Q", "(b) 选模验证：最差偏好完成率",
                               "(c) 独立诊断 Q", "(d) 独立诊断：总体完成率"]):
    ax.set_title(title, loc="left", pad=10)
    decorate(ax, episode=True)
for ax in axs[:, 0]:
    ax.set_ylabel("偏好均衡 Q（↓）")
    ax.set_ylim(qmin - 0.014, qmax + 0.018)
for ax in axs[:, 1]:
    ax.set_ylabel("交付完成率（%）")
    ax.set_ylim(cmin - 1.5, 100.7)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
for ax in axs[0]:
    ax.set_xlabel("")
fig.subplots_adjust(top=0.83, bottom=0.17, left=0.08, right=0.98, hspace=0.36, wspace=0.25)
caption = ("三次采样取优；选模优先比较最差偏好完成率，再比较 Q；Q 统计成功交付。\n"
           "Best：Seed 11 = ep2000，Seed 23 = ep1500，Seed 37 = ep1900。独立诊断不参与选模。")
fig.text(0.08, 0.04, caption, fontsize=10, linespacing=1.7, color="#52616B")
figures.append(save(fig, "02_mix_convergence", caption))

# Responses use all successful formal-test deliveries, aggregated at the same weight.
fig, axs = plt.subplots(1, 3, figsize=(14.0, 5.1))
fig.suptitle(f"{METHOD}：best 模型的偏好响应", y=0.985, fontsize=17, fontweight="bold")
seed_legend(fig)
for ax, objective, title, ylabel in zip(
        axs, ("flow", "cost", "workload"),
        ("(a) Flow 偏好", "(b) Cost 偏好", "(c) OWA 偏好"),
        ("Excess Flow（↓）", "重构成本（↓）", "OWA 工作负荷（↓）")):
    for seed in SEEDS:
        p = P.loc[P.seed.eq(seed) & P.objective.eq(objective)].sort_values("weight")
        assert len(p) == 11 and np.allclose(p.weight, np.linspace(0, 1, 11))
        ax.plot(p.weight, p.value, color=COLORS[seed], marker=MARKERS[seed],
                linewidth=1.8, markersize=4)
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel("对应目标的偏好权重")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.025, 1.025)
    ax.set_xticks(np.linspace(0, 1, 6))
    decorate(ax)
fig.subplots_adjust(top=0.78, bottom=0.25, left=0.07, right=0.98, wspace=0.30)
caption = ("正式测试：20 实例 × 66 偏好，每个模型三次取优；相同目标权重处平均其余偏好组合及实例。\n"
           "每个 seed 均 1320/1320 交付成功；目标值越低越好，曲线展示平均响应，不代表逐实例严格单调。")
fig.text(0.07, 0.045, caption, fontsize=10, linespacing=1.7, color="#52616B")
figures.append(save(fig, "03_mix_preference_response", caption))

# Retain only Mix PPO and the two frozen search comparators.
assert list(H.preference) == ["Flow", "Cost", "OWA", "Equal", "Overall"]
for column in ("mixed_Q", "nsga2_ls", "moead"):
    assert np.isclose(H.loc[4, column], H.loc[:3, column].mean())
fig, ax = plt.subplots(figsize=(12.4, 6.2))
fig.suptitle(f"{METHOD} 与启发式搜索：7 例、4 偏好及总体", y=0.985,
             fontsize=17, fontweight="bold")
x = np.arange(5)
width = 0.24
ax.axvspan(3.53, 4.48, color="#EDF1F4", zorder=0)
for offset, column, label, color in [
    (-1, "mixed_Q", f"{METHOD}（3 seed 均值）", "#009E73"),
    (0, "nsga2_ls", "NSGA-II + LS", "#D55E00"),
    (1, "moead", "MOEA/D", "#CC79A7"),
]:
    bars = ax.bar(x + offset * width, H[column], width=width, label=label,
                  color=color, edgecolor="white", linewidth=0.6)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in H[column]], padding=5, fontsize=9)
ax.set_xticks(x, ["Flow", "Cost", "OWA", "等权", "总体"])
ax.set_ylabel("平均 Q（↓）")
ax.set_ylim(0, 0.65)
ax.set_xlim(-0.6, 4.6)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
decorate(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)
fig.subplots_adjust(top=0.79, bottom=0.25, left=0.08, right=0.98)
caption = (f"{METHOD}：各训练 seed 的 best 模型独立三次取优，再跨 seed 等权平均；总体为 4 偏好等权均值。\n"
           "搜索：冻结的 search seed=11、600 秒/请求。")
fig.text(0.08, 0.055, caption, fontsize=10, linespacing=1.7, color="#52616B")
figures.append(save(fig, "04_hgnnppo_heuristic_comparison", caption))

manifest_path = OUT / "plot_manifest.json"
if ARGS.figure and manifest_path.exists():
    previous_exports = json.loads(manifest_path.read_text(encoding="utf-8"))["exports"]
    updated_paths = {entry["path"] for entry in EXPORTS}
    active_stems = {figure["stem"] for figure in figures}
    EXPORTS = [entry for entry in previous_exports
               if entry["path"] not in updated_paths and Path(entry["path"]).stem in active_stems] + EXPORTS

manifest = {
    "scope": "mixed_instances only; seeds 11, 23, 37; no new training or evaluation",
    "figures": figures,
    "best_episodes": {str(s): int(S.loc[s, "best_episode"]) for s in SEEDS},
    "reward_window": {"episodes": 100, "minimum_periods": 100},
    "inputs": INPUTS,
    "exports": EXPORTS,
}
(OUT / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
lines = [f"# {METHOD} 三 seed 结果图", "", "数据沿用已核验的固定混合实例训练三 seed 汇总。", ""]
for f in figures:
    stem = f["stem"]
    lines += [f"## {stem}", "", f["caption"].replace("\n", "<br>\n"), "",
              f"![{stem}]({stem}.png)", "",
              f"[PNG]({stem}.png) · [PDF]({stem}.pdf) · [SVG]({stem}.svg)", ""]
lines += ["复现：在项目根目录执行 `.\\.venv\\Scripts\\python.exe -B "
          "result/analysis/mixed_instances_three_seed_20261022/mix_figures/plot_mix.py`。", "",
          "输入文件哈希与导出清单见 [plot_manifest.json](plot_manifest.json)。", ""]
(OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"directory": str(OUT), "figures": len(figures), "exports": len(EXPORTS),
                  "inputs": len(INPUTS)}, ensure_ascii=False))
