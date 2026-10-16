"""Audit saved preference responses; no policy inference or training.

Run: python -m analysis.studies.schema12_gae099_preference_20261015
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.studies import schema12_owa_comparison_20261014 as audit
from analysis.studies.schema12_gae099_convergence_20261015 import OUT, RUNS

TARGETS = [
    ("Flow", "preference_flow", "flow_time_objective"),
    ("Cost", "preference_cost", "reconfiguration_cost"),
    ("OWA", "preference_workload", "owa_q05"),
]


def cells(frame):
    return frame[audit.safe(frame)].groupby(audit.KEY[:2]).agg(
        {**{p: "first" for p in audit.PREF},
         **{t: "mean" for _, _, t in TARGETS}}
    ).reset_index()


def correlations(frame, model, cohort):
    rows = []
    for name, pref, target in TARGETS:
        values = [spearmanr(g[pref], g[target]).statistic
                  for _, g in frame.groupby("instance_id")]
        rows.append(dict(model=model, cohort=cohort, objective=name,
                         instances=len(values), cells=len(frame),
                         median_spearman=float(np.nanmedian(values)),
                         negative_instances=int((np.array(values) < 0).sum())))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tests, validation = {}, {}
    for model, run in RUNS.items():
        tests[model] = audit.read(run / "final_sampled_instance_metrics.csv", model, model)
        validation[model] = audit.read(run / "sampled_validation_instance_metrics.csv", model, model)
    complete = []
    for frame in tests.values():
        successful = frame.assign(succeeded=audit.safe(frame)).groupby(audit.KEY[:2]).succeeded
        count = successful.agg(["sum", "count"])
        complete.append(set(count[(count["sum"] == 3) & (count["count"] == 3)].index))
    common = complete[0] & complete[1]
    rows, last_cells = [], {}
    for model, frame in tests.items():
        full = cells(frame)
        rows.extend(correlations(full, model, "best_test_all_cells"))
        restricted = full[pd.MultiIndex.from_frame(full[audit.KEY[:2]]).isin(common)]
        rows.extend(correlations(restricted, model, "best_test_both_models_3_of_3_success"))
        last = validation[model]
        last = last[last.validation_episode == 2000]
        last_cells[model] = cells(last)
        rows.extend(correlations(last_cells[model], model, "last_validation_13_preferences"))
    last_common = set.intersection(*[
        set(pd.MultiIndex.from_frame(frame[audit.KEY[:2]]))
        for frame in last_cells.values()
    ])
    for model, frame in last_cells.items():
        subset = frame[pd.MultiIndex.from_frame(frame[audit.KEY[:2]]).isin(last_common)]
        rows.extend(correlations(subset, model, "last_validation_common_available_cells"))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "preference_robustness.csv", index=False, encoding="utf-8-sig")

    # Plot existing per-cell means: every instance/preference receives equal weight.
    levels = pd.read_csv(OUT / "preference_weight_levels.csv")
    amplitudes = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, (name, _, target) in zip(axes, TARGETS):
        for model, color, label in [("lambda095", "#2563eb", "lambda = 0.95 (ep 1900)"),
                                    ("lambda099", "#d97706", "lambda = 0.99 (ep 700)")]:
            data = levels[(levels.model == model) & (levels.target == target)].sort_values("weight")
            ax.plot(data.weight, data["mean"], marker="o", ms=3, color=color, label=label)
            zero = float(data.loc[np.isclose(data.weight, 0), "mean"].iloc[0])
            one = float(data.loc[np.isclose(data.weight, 1), "mean"].iloc[0])
            amplitudes.append(dict(model=model, objective=name, weight_zero_mean=zero,
                                   weight_one_mean=one, reduction_percent=100 * (1 - one / zero)))
        ax.set(title=name + (" (raw)" if name == "Flow" else ""),
               xlabel="Own objective weight", ylabel="Mean objective (lower is better)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "preference_response.png", dpi=180)
    fig.savefig(OUT / "preference_response.svg")
    plt.close(fig)
    amplitude = pd.DataFrame(amplitudes)
    amplitude.to_csv(OUT / "preference_amplitude.csv", index=False, encoding="utf-8-sig")

    text = "# Lambda preference response audit\n\n"
    text += "Best test: 20 instances x 66 preferences x 3 samples. Successful repeats are averaged per instance/preference before correlations; table values are medians across instances. Negative correlations mean lower objective as its own weight increases.\n\n"
    text += "Both models cover all 1320 cells with at least one successful repeat. The common fully successful subset has " + str(len(common)) + " cells (both models succeed in all three repeats). No missing-cell imputation.\n\n"
    text += "| Cohort | Objective | lambda .95 | lambda .99 |\n|---|---|---:|---:|\n"
    for cohort in result.cohort.unique():
        for name, _, _ in TARGETS:
            group = result[(result.cohort == cohort) & (result.objective == name)].set_index("model")
            text += f"| {cohort} | {name} | {group.loc['lambda095', 'median_spearman']:.4f} | {group.loc['lambda099', 'median_spearman']:.4f} |\n"
    text += "\nLast validation uses the same 50 instances x 13 preferences x 3 samples at episode 2000. Lambda .95 has successful repeats in 650 cells; lambda .99 in 648 (two cells fail all three repeats). The matched available-cell subset contains " + str(len(last_common)) + " cells. Its correlations should be compared within that cohort, not as a like-for-like test against the 66-preference best test.\n\n"
    text += "| Objective | lambda .95 weight 0 to 1 reduction | lambda .99 weight 0 to 1 reduction |\n|---|---:|---:|\n"
    for name, _, _ in TARGETS:
        group = amplitude[amplitude.objective == name].set_index("model")
        text += f"| {name} | {group.loc['lambda095', 'reduction_percent']:.2f}% | {group.loc['lambda099', 'reduction_percent']:.2f}% |\n"
    text += "\nWeight-level means average all available combinations of the other two weights at that level. They describe aggregate response, not a single fixed-ratio path. See preference_fixed_ratio.csv for controlled other-weight-ratio directional checks.\n\n"
    text += "![Preference response](preference_response.png)\n\n"
    text += "Conclusion: the .99 selected best has much weaker preference response, especially OWA. Restricting to fully successful cells leaves that conclusion intact. At the last checkpoint, .99 Cost rank response recovers and exceeds .95 on validation, while Flow and OWA remain weaker. Thus the early selected best explains part of the difference; the network does learn preference conditioning later. Rank responsiveness does not establish absolute objective quality. This is one paired training seed and does not establish a universal causal effect of lambda. Flow is displayed in raw units; within an instance, subtracting the fixed processing lower bound does not change rank correlations.\n\n"
    text += "Reproduce: `python -m analysis.studies.schema12_gae099_preference_20261015`. Inputs are the two run directories named in the convergence audit. No policy evaluation is rerun.\n"
    (OUT / "preference_response.md").write_text(text, encoding="utf-8")
    print(json.dumps({"common_complete_cells": len(common), "correlations": rows,
                      "amplitudes": amplitudes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
