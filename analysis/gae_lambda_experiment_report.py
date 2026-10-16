"""Live, honest reporting for the frozen lambda matrix and completed diagnosis."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from analysis.pareto_analysis import hypervolume_3d
from result.io import write_csv, write_json


def test_metrics(frame, scales):
    keys = ["instance_id", "preference_key", "sampling_repeat"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate test cells")
    if len(frame) != 20*66*3 or frame.instance_id.nunique() != 20:
        raise ValueError("formal test must contain all 20 x 66 x 3 cells")
    for _, instance in frame.groupby("instance_id"):
        if instance.preference_key.nunique() != 66 or not (instance.groupby("preference_key").size() == 3).all():
            raise ValueError("test preference/repeat coverage is incomplete")
    safe = frame.task_succeeded & (frame.schedule_violation_count == 0)
    safe &= frame.maximum_worker_fatigue <= frame.safe_fatigue_limit + 1e-9
    per_instance = []
    for iid, instance in frame.groupby("instance_id"):
        good = instance[safe.loc[instance.index]]
        fields = ["flow_excess_objective", "reconfiguration_cost", "worker_load_variance"]
        values = good[fields].to_numpy(float)
        pooled = hypervolume_3d(values/(values+scales))
        repeat_hvs = []
        for repeat in range(3):
            points = good[good.sampling_repeat == repeat][fields].to_numpy(float)
            repeat_hvs.append(hypervolume_3d(points/(points+scales)))
        # Average stochastic repeat outcomes at each preference before measuring
        # preference response, so sampling variation is not called preference span.
        flow = good.groupby("preference_key").flow_time_objective.mean()
        span = float(flow.max()-flow.min()) if len(flow) >= 2 else None
        fully_successful = good.groupby("preference_key").size().reindex(instance.preference_key.unique(), fill_value=0)
        per_instance.append({"instance_id": iid, "completion_rate": float(safe.loc[instance.index].mean()),
            "pooled_hv": pooled, "single_repeat_hv_mean": float(np.mean(repeat_hvs)),
            "flow_span_successful_preference_means": span,
            "flow_span_relative_to_mean": span/float(flow.mean()) if span is not None else None,
            "successful_preferences": len(flow), "all_three_successful_preferences": int((fully_successful == 3).sum()),
            "full_grid_all_repeats_successful": bool((fully_successful == 3).all())})
    inst = pd.DataFrame(per_instance)
    return {"completion_rate": float(safe.mean()), "mean_instance_pooled_hv": float(inst.pooled_hv.mean()),
            "mean_instance_single_repeat_hv": float(inst.single_repeat_hv_mean.mean()),
            "mean_instance_flow_span": float(inst.flow_span_successful_preference_means.mean()),
            "mean_instance_relative_flow_span": float(inst.flow_span_relative_to_mean.mean()),
            "full_grid_all_repeats_successful_instances": int(inst.full_grid_all_repeats_successful.sum()),
            "mean_successful_preference_coverage": float(inst.successful_preferences.mean()/66)}, per_instance


def _save_figure(fig, stem):
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))


def build_diagnostic_report(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = pd.read_csv(directory / "gae_metrics.csv")
    signs = pd.read_csv(directory / "sign_agreement.csv")
    progress = json.loads((directory / "progress.json").read_text())
    decision = json.loads((directory / "decision.json").read_text())
    roles = [role for role in ("last", "best") if role in set(metrics.role)]
    pairs = signs[(signs.lambda_left == .95) & (signs.lambda_right == 1) & (signs.scope == "choice_steps")]
    rows = metrics[(metrics["lambda"] == .95) & (metrics.scope == "choice_steps")].merge(
        pairs, on=["role", "region", "outcome", "scope", "trajectories", "instances", "steps"], suffixes=("", "_pair"))
    table = rows[["role", "region", "outcome", "trajectories", "instances", "steps", "advantage_variance", "bias", "mae", "rmse",
                  "nonzero_sign_agreement", "centered_nonzero_sign_agreement"]]
    table.to_csv(directory / "lambda095_vs_mc.csv", index=False, encoding="utf-8-sig")
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"flow": "#0072B2", "cost": "#D55E00", "variance": "#009E73", "center": "#CC79A7"}
    fig, axes = plt.subplots(len(roles), 2, figsize=(11, 3.5*len(roles)), sharex=True, squeeze=False)
    for i, role in enumerate(roles):
        for region, color in colors.items():
            group = metrics[(metrics.role == role) & (metrics.region == region) & (metrics.outcome == "all")
                            & (metrics.scope == "choice_steps")].sort_values("lambda")
            axes[i, 0].plot(group["lambda"], group.advantage_variance, color=color, marker="o", label=region)
            group = signs[(signs.role == role) & (signs.region == region) & (signs.outcome == "all")
                          & (signs.scope == "choice_steps") & (signs.lambda_right == 1)].sort_values("lambda_left")
            axes[i, 1].plot(np.r_[group.lambda_left, 1], np.r_[group.nonzero_sign_agreement, 1], color=color, marker="o", label=region)
        axes[i, 0].set_ylabel(f"{role}: raw advantage variance")
        axes[i, 1].set_ylabel(f"{role}: sign agreement with MC")
        axes[i, 1].set_ylim(0, 1.03)
        axes[i, 1].axhline(.95, color="gray", ls="--", lw=1)
        for ax in axes[i]: ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=10)
    for ax in axes[-1]: ax.set_xlabel("GAE lambda")
    _save_figure(fig, directory / "gae_diagnostic")
    plt.close(fig)
    lines = ["# 冻结 v4 的 GAE λ 诊断", "",
        f"已完成 {progress['complete_trajectories']} 条完整轨迹，共 {progress['steps']:,} 步：原验证子集 50 个实例 × 4 个偏好 × 3 次采样 × {'/'.join(roles)}。所有动作哈希和累计奖励均与归档匹配。",
        "", "奖励、scales、critic、动作序列固定。γ=1；终局 V=0；每条轨迹单独反向递推。horizon/deadlock 是含失败惩罚的真实终局，采样截断不参与。",
        "", "`δ_t = r_t + γ V(s_{t+1}) - V(s_t)`；`A^λ_t = δ_t + γ λ A^λ_{t+1}`；`G_t = r_t + γ G_{t+1}`。完整轨迹上 `A^1_t = G_t - V(s_t)`，因此与 MC 的偏差按 `A^λ_t + V(s_t) - G_t` 计算。",
        "", "下表排除仅一个合法动作的步，这些步对 actor 的梯度为零；它们的奖励和值仍参与完整 GAE。方差是标准化前的总体方差。符号列为排除 1e-8 近零值后的同号率。中心化列先在同一 checkpoint 的整批数据上减去各 λ 的优势均值，近似 PPO 标准化对符号的影响；它不能重建历史训练 minibatch。",
        "", "| checkpoint | 偏好 | 轨迹 | GAE 方差 | 回报偏差 | MAE | RMSE | 原始同号率 | 中心化同号率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table[table.outcome == "all"].itertuples():
        lines.append(f"| {row.role} | {row.region} | {row.trajectories} | {row.advantage_variance:.5f} | {row.bias:+.5f} | {row.mae:.5f} | {row.rmse:.5f} | {row.nonzero_sign_agreement:.1%} | {row.centered_nonzero_sign_agreement:.1%} |")
    lines += ["", "失败轨迹：", "", "| checkpoint | 偏好 | 轨迹/实例 | 回报偏差 | RMSE | 原始同号率 | 中心化同号率 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in table[(table.outcome == "failure") & (table.region != "all")].itertuples():
        lines.append(f"| {row.role} | {row.region} | {row.trajectories}/{row.instances} | {row.bias:+.5f} | {row.rmse:.5f} | {row.nonzero_sign_agreement:.1%} | {row.centered_nonzero_sign_agreement:.1%} |")
    lines += ["", f"诊断判定：{'进入训练对照' if decision['requires_training'] else '保持 0.95'}。预先记录的实用筛查阈值为同号率 ≥95%、回报 RMSE ≤0.05；这不是通用统计定理。",
              "", "GAE/MC 符号冲突说明更新权重受估计器影响，单批 MC 本身也有采样方差，因此不能仅凭符号冲突证明真实策略梯度方向错误。λ=1 的回报偏差为零是代数恒等式，并不说明其训练质量最优。best/last 是同一个训练种子的不同阶段，也不是独立训练重复。",
              "", f"统计单位：总体按步加权，另存逐轨迹指标和等轨迹权重偏差；不同偏好和失败轨迹分别报告。`gae_metrics.csv` 含全部 5 个 λ；`sign_agreement.csv` 含全部 10 对 λ；`trajectory_metrics.csv` 含逐轨迹指标；{len(list(directory.glob('*_repeat*.json')))} 个 JSON chunk 保存所有奖励、value、动作及合法动作数。",
              "", "![GAE 方差与符号诊断](gae_diagnostic.png)", "", "训练对照状态和曲线见同一实验目录内的 training_batch/report.md。"]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(batch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plan = json.loads((batch / "plan.json").read_text())
    state = json.loads((batch / "launch.json").read_text())
    validation, updates, tests, spans, runs = [], [], [], [], []
    config = json.loads(Path(plan["jobs"][0]["config"]).read_text())
    scales = np.array([config["objective_scalarizer"]["scales"][k] for k in ("flow", "cost", "variance")])
    for job in state["jobs"]:
        path = Path(job["run_directory"])
        common = {"lambda": job["lambda"], "seed": job["seed"], "run_name": job["run_name"]}
        run = {**common, "status": job["status"], "completed_training_episodes": 0, "validation_points": 0}
        for name, container in (("validation_log.csv", validation), ("update_log.csv", updates)):
            if not (path / name).exists(): continue
            frame = pd.read_csv(path / name)
            if name == "update_log.csv":
                frame["advantage_variance"] = frame.advantage_std**2
                run["completed_training_episodes"] = int(frame.episode_end.max()+1)
            else: run["validation_points"] = len(frame)
            container.extend([{**common, **r} for r in frame.to_dict("records")])
        # Test results only enter the comparison after the whole run completed.
        if job["status"] == "completed":
            frame = pd.read_csv(path / "final_sampled_instance_metrics.csv")
            values, instance_rows = test_metrics(frame, scales)
            summary = json.loads((path / "summary.json").read_text())
            tests.append({**common, **values, "selected_episode": summary["checkpoint_selection"]["best_episode"]})
            spans.extend([{**common, **r} for r in instance_rows])
        runs.append(run)
    write_csv(batch / "run_status.csv", runs)
    write_csv(batch / "validation_curves.csv", validation)
    write_csv(batch / "update_curves.csv", updates)
    write_csv(batch / "test_summary.csv", tests)
    write_csv(batch / "instance_flow_spans.csv", spans)
    summary_rows = []
    if tests:
        frame = pd.DataFrame(tests)
        measures = ["completion_rate", "mean_instance_pooled_hv", "mean_instance_single_repeat_hv",
                    "mean_instance_flow_span", "mean_instance_relative_flow_span"]
        for lam, group in frame.groupby("lambda"):
            summary_rows.append({"lambda": lam, "completed_seeds": len(group),
                **{key+"_mean": float(group[key].mean()) for key in measures},
                **{key+"_seed_std": float(group[key].std(ddof=1)) if len(group) > 1 else None for key in measures}})
    write_csv(batch / "test_seed_summary.csv", summary_rows)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    colors = {.95: "#0072B2", .99: "#D55E00", 1.0: "#009E73"}
    plots = []
    for data, measures, x, stem in ((validation, ["completion_rate", "preference_balanced_quality_score"], "episode", "validation"),
        (updates, ["advantage_variance", "pre_update_explained_variance", "clip_fraction"], "episode_end", "ppo_diagnostics")):
        if not data: continue
        frame = pd.DataFrame(data)
        fig, axes = plt.subplots(1, len(measures), figsize=(5*len(measures), 3.8), squeeze=False)
        for ax, measure in zip(axes.flat, measures):
            for lam, group in frame.groupby("lambda"):
                for seed, curve in group.groupby("seed"):
                    ax.plot(curve[x], curve[measure], color=colors[lam], alpha=.2, lw=.7)
                # Averages require all prescribed seeds at that episode. Partial
                # results show their seed lines without a misleading n=3 band.
                expected = 1 if lam == 1 else 3
                group = group.groupby(x)[measure].agg(["mean", "std", "count"])
                complete = group[group["count"] == expected]
                if len(complete):
                    ax.plot(complete.index, complete["mean"], color=colors[lam], label=f"lambda={lam}, n={expected}")
                    if expected > 1:
                        ax.fill_between(complete.index, complete["mean"]-complete["std"], complete["mean"]+complete["std"], color=colors[lam], alpha=.15)
                else: ax.plot([], [], color=colors[lam], label=f"lambda={lam}, individual seeds")
            ax.set_xlabel("Training episodes"); ax.set_ylabel(measure.replace("_", " ")); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=9)
        _save_figure(fig, batch / stem); plt.close(fig); plots.append(stem)
    lines = ["# 冻结 v4 的 λ 训练对照", "", f"当前状态：{state['stage']}。正式完成 {sum(j['status']=='completed' for j in state['jobs'])}/7 组。",
        "", "λ=0.95、0.99 各训练种子 11、23、37；λ=1 仅种子 11。全部从头训练，2000 episode，20 episode/update，原验证协议（50 实例 ×13 偏好 ×3 重复，每100 episode）和诊断子集保持一致；best 使用原 completion-first 选择规则。",
        "", "执行代码、scales、验证/测试实例与 manifest 已复制冻结；只有 λ、训练种子和运行名称有实验差异。训练缓存和结果路径指向原位置。预检和 smoke 只证明可运行性；下表和正式 summary 才记录实际训练进度。",
        "", "| λ | seed | 状态 | 已训练 episode | 验证点 |", "|---:|---:|---|---:|---:|"]
    lines += [f"| {r['lambda']} | {r['seed']} | {r['status']} | {r['completed_training_episodes']} | {r['validation_points']} |" for r in runs]
    lines += ["", "测试采用 best checkpoint，在20实例×66网格×3重复上测完成率；HV 以 frozen scales 做 `J/(s+J)`，参考点(1,1,1)，先每实例计算再平均。同时保留三重复合并HV和单重复HV，避免把推断预算影响混为模型质量。",
              "", "Flow 跨度先对同实例、同偏好的成功重复求平均，再在66个偏好间求 max−min；同时报告成功偏好覆盖和整网格全部重复成功实例数。失败结果进入完成率，但不作为可行点计入HV，也不把失败的惩罚Flow混入偏好响应跨度。",
              "", "优势方差为标准化前 `advantage_std²`。explained variance 使用原日志的 GAE return target，不同 λ 的目标也不同，不能把它等同于对 MC 的解释率。clip fraction 保持原 PPO 定义。",
              "", "曲线显示各种子细线，只有该 λ 的全部指定种子都到达同一 episode 才画均值±种子标准差。λ=1 的单种子对照不提供种子方差。"]
    if summary_rows:
        lines += ["", "已完成组的测试结果（多种子组未满3时为阶段性结果）：", "", "| λ | 完成种子 | 完成率 | pooled HV | 单重复 HV | Flow 跨度 |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for r in summary_rows:
            lines.append(f"| {r['lambda']} | {r['completed_seeds']} | {r['completion_rate_mean']:.4%} | {r['mean_instance_pooled_hv_mean']:.5f} | {r['mean_instance_single_repeat_hv_mean']:.5f} | {r['mean_instance_flow_span_mean']:.3f} |")
    for stem in plots: lines += ["", f"![{stem}]({stem}.png)"]
    lines += ["", "原始 run 目录见 plan.json；训练和测试汇总 CSV 随实际进度更新。七组完成前不能得出 λ=0.99 优于0.95 的结论。"]
    temporary = batch / "report.md.tmp"
    temporary.write_text("\n".join(lines)+"\n", encoding="utf-8")
    temporary.replace(batch / "report.md")
    write_json(batch / "report_status.json", {"stage": state["stage"], "completed_runs": len(tests), "run_status": runs})
