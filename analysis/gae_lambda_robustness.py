"""Numerical sensitivity, independent bias decomposition, clustered intervals."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from analysis.gae_lambda_diagnostic import estimates, sign_statistics
from result.io import write_csv, write_json


def supplement(directory):
    episodes = []
    for path in sorted(directory.glob("*_repeat*.json")):
        for e in json.loads(path.read_text()):
            e.update({k: np.asarray(e[k]) for k in ("rewards", "values", "actions", "legal_counts")})
            e["advantages"], e["mc"] = estimates(e["rewards"], e["values"], gamma=e["gamma"])
            episodes.append(e)
    sensitivity, intervals = [], []
    max_decomposition_error = 0.
    rng = np.random.default_rng(20261012)
    for role in sorted({e["role"] for e in episodes}):
        role_episodes = [e for e in episodes if e["role"] == role]
        centers = {lam: np.concatenate([e["advantages"][lam] for e in role_episodes]).mean() for lam in (.95, 1.)}
        for e in role_episodes:
            # Independent identity: e_t = gamma*(1-lambda)*(V_next-G_next)
            #                         + gamma*lambda*e_next.
            for lam in (.9, .95, .97, .99, 1.):
                decomposition = np.zeros_like(e["values"])
                for t in range(len(decomposition)-2, -1, -1):
                    decomposition[t] = e["gamma"]*((1-lam)*(e["values"][t+1]-e["mc"][t+1]) + lam*decomposition[t+1])
                discrepancy = e["advantages"][lam] + e["values"] - e["mc"]
                max_decomposition_error = max(max_decomposition_error, float(np.abs(discrepancy-decomposition).max()))
        for region in ("all", "flow", "cost", "variance", "center"):
            for outcome in ("all", "success", "failure"):
                selected = [e for e in role_episodes if (region == "all" or e["region"] == region)
                            and (outcome == "all" or e["success"] == (outcome == "success"))]
                if not selected: continue
                a = np.concatenate([e["advantages"][.95][e["legal_counts"]>1] for e in selected])
                b = np.concatenate([e["advantages"][1.][e["legal_counts"]>1] for e in selected])
                if not len(a): continue
                common = {"role": role, "region": region, "outcome": outcome, "scope": "choice_steps"}
                for centered in (False, True):
                    left, right = (a-centers[.95], b-centers[1.]) if centered else (a, b)
                    for epsilon in (1e-8, 1e-6, 1e-4, 1e-3):
                        opposite = (left*right<0) & (np.abs(left)>epsilon) & (np.abs(right)>epsilon)
                        sensitivity.append({**common, "centered": centered, "epsilon": epsilon,
                            **sign_statistics(left, right, epsilon=epsilon),
                            "opposite_gae_absolute_mass_fraction": float(np.abs(left[opposite]).sum()/np.abs(left).sum()) if np.abs(left).sum() else None,
                            "opposite_mc_absolute_mass_fraction": float(np.abs(right[opposite]).sum()/np.abs(right).sum()) if np.abs(right).sum() else None})
                by_instance = {}
                for e in selected:
                    mask = e["legal_counts"] > 1
                    left, right = e["advantages"][.95][mask], e["advantages"][1.][mask]
                    error = left-right
                    stats = by_instance.setdefault(e["metadata"]["instance_id"], np.zeros(5))
                    valid = (np.abs(left)>1e-8) & (np.abs(right)>1e-8)
                    stats += [len(error), error.sum(), (error**2).sum(), int(((left*right<0)&valid).sum()), int(valid.sum())]
                clusters = np.array(list(by_instance.values()))
                ids = rng.integers(0, len(clusters), size=(10000, len(clusters)))
                samples = clusters[ids].sum(axis=1)
                derived = {"bias": samples[:,1]/samples[:,0], "rmse": np.sqrt(samples[:,2]/samples[:,0]),
                           "nonzero_sign_agreement": 1-samples[:,3]/samples[:,4]}
                for name, values in derived.items():
                    lower, upper = np.quantile(values, [.025, .975])
                    intervals.append({**common, "measure": name, "instances": len(clusters),
                        "trajectories": len(selected), "lower": float(lower), "upper": float(upper),
                        "bootstrap_repeats": 10000, "seed": 20261012,
                        "interpretation": "fixed validation cohort, instance clusters; small failure groups do not establish population precision"})
    if max_decomposition_error > 1e-9:
        raise AssertionError("independent bootstrap error identity failed")
    write_csv(directory / "sign_sensitivity.csv", sensitivity)
    write_csv(directory / "instance_bootstrap_intervals.csv", intervals)
    trace = Path(__file__).resolve().parents[1] / "result/analysis/shared_excess_last_20261012/trace/decision_trace.csv"
    trace_errors = []
    if trace.exists():
        previous = pd.read_csv(trace)
        for e in episodes:
            if e["region"] != "cost": continue
            old = previous[(previous.role == e["role"]) & (previous.instance_id == e["metadata"]["instance_id"])
                           & (previous.sampling_repeat == e["metadata"]["repeat"])]
            if len(old):
                if len(old) != len(e["values"]): raise AssertionError("prior trace length changed")
                trace_errors.append({"role": e["role"], "instance_id": e["metadata"]["instance_id"],
                    "repeat": e["metadata"]["repeat"], "steps": len(old),
                    "maximum_value_error": float(np.abs(old.critic_value.to_numpy()-e["values"]).max()),
                    "maximum_reward_error": float(np.abs(old.reward.to_numpy()-e["rewards"]).max())})
    write_csv(directory / "prior_trace_checks.csv", trace_errors)
    verification = {"complete_trajectories": len(episodes), "maximum_bootstrap_decomposition_error": max_decomposition_error,
                    "matched_prior_traces": len(trace_errors), "prior_traces": trace_errors}
    write_json(directory / "additional_verification.json", verification)
    with (directory / "report.md").open("a", encoding="utf-8") as stream:
        stream.write("\n附加复核：`sign_sensitivity.csv` 报告 1e-8、1e-6、1e-4、1e-3 近零阈值下的同号率，以及反号步的优势绝对量占比；`instance_bootstrap_intervals.csv` 按完整实例做10000次重采样，保留实例内偏好和重复相关性，失败实例较少的区间不能证明总体精度。\n")
        stream.write(f"\n独立验证恒等式 `e_t=γ(1−λ)(V_{{t+1}}−G_{{t+1}})+γλe_{{t+1}}`（`e=A^λ+V−G`），最大误差 {max_decomposition_error:.3g}。与旧决策 trace 核对 {len(trace_errors)} 条轨迹，详见 `prior_trace_checks.csv`。\n")
        stream.write("\n理论解释：λ=1 使用完整经验回报，λ<1 引入价值函数 bootstrap；取值应权衡偏差与方差。单条失败轨迹的经验回报偏差包含采样随机性，不能全部归因于 critic 的期望偏差。[GAE 原论文 §3](https://arxiv.org/pdf/1506.02438)。\n")
    return verification


if __name__ == "__main__":
    supplement(Path("result/analysis/gae_lambda_v4_20261012"))
