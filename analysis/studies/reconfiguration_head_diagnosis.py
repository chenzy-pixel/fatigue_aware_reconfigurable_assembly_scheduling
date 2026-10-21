"""Audit the archived seed11 expert/shared-head runs using saved trajectories.

Run: python -m analysis.studies.reconfiguration_head_diagnosis
All comparisons describe these trained models; checkpoints are not replayed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "result/runs"
OUTPUT = ROOT / "result/analysis/reconfiguration_head_seed11_20261009"
TRAIN_RUNS = {
    "v8": RUNS / "v8_universal_seed11_20261005_004756_510",
    "shared": RUNS / "ablation_structural_shared_head_seed11_20261006_025649_012965",
}
EVAL_RUNS = {
    "v8": RUNS / "ablation_eval_universal_seed11_20261007_232505_102441",
    "shared": RUNS / "ablation_eval_shared_head_seed11_20261007_232505_102441",
}
PRESSURE = "reconfiguration_bottleneck"
BINS = ["cost_le_0.2", "cost_0.2_to_0.5", "cost_gt_0.5"]
CELL_KEYS = ["instance_id", "preference_key", "sampling_repeat"]
METRICS = [
    "completed_reconfigurations", "commit_reconfig_action_count",
    "completed_reconfigurations_per_operation", "reconfigs_per_total_operation",
    "worker_reconfiguration_busy_minutes", "machine_waiting_for_worker_time",
    "maximum_worker_fatigue", "fatigue_masked_action_ratio",
    "worker_competition_event_count", "operation_progress",
    "heuristic_reconfiguration_ratio", "flow_time_objective",
    "reconfiguration_cost", "worker_load_variance",
]
IDENTITIES = [
    "seed", "pressure_type", "cost_profile", "severity", "algorithm_seed",
    "preference_flow", "preference_cost", "preference_variance",
    "derived_sampling_seed", "sampling_seed", "sampling_rng_version",
    "policy_execution_version", "policy_precision", "reward_version",
    "generator_config_sha256", "environment_config_sha256",
    "distribution_contract_sha256", "dataset_manifest_sha256", "subset_sha256",
    "normalization_manifest_sha256", "objective_scale_flow", "objective_scale_cost",
    "objective_scale_variance", "terminal_failure_penalty_configured",
]
SOURCE_FILES: set[Path] = set()


def read_json(path: Path) -> dict:
    SOURCE_FILES.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path, *, trajectories: bool = False) -> pd.DataFrame:
    SOURCE_FILES.add(path)
    selected = set(CELL_KEYS + METRICS + IDENTITIES + [
        "validation_episode", "task_succeeded", "task_failed", "terminated",
        "truncated", "termination_reason", "heuristic_completed", "heuristic_makespan",
        "physical_safety_pass", "active_constraint_pass", "schedule_violation_count",
        "safe_fatigue_limit", "checkpoint_sha256",
        "terminal_failure_penalty_applied",
    ])
    return pd.read_csv(path, usecols=(lambda c: c in selected) if trajectories else None)


def write_csv(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUTPUT / name, index=False, encoding="utf-8-sig")


def flatten(value: object, prefix: str = "") -> dict:
    if isinstance(value, dict):
        return {k: v for name, child in value.items()
                for k, v in flatten(child, f"{prefix}.{name}".strip(".")).items()}
    return {prefix: value}


def add_instance_metrics(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    manifest_path = ROOT / f"data/manifests/v2/{split}/manifest.json"
    manifest = read_json(manifest_path)
    entries = {item["seed"]: item for item in manifest["files"]}
    rows = []
    for instance_id, seed in frame[["instance_id", "seed"]].drop_duplicates().itertuples(index=False):
        item = entries[seed]
        path = ROOT / f"data/instances/v2/{split}" / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
        document = read_json(path)
        instance = document["instance"]
        assert instance["instance_id"] == instance_id
        rows.append({"instance_id": instance_id,
                     "total_operations": sum(len(order["operations"]) for order in instance["orders"]),
                     "horizon": instance["time"]["horizon"]})
    frame = frame.merge(pd.DataFrame(rows), on="instance_id", validate="many_to_one")
    frame = frame.assign(
        cost_bin=pd.cut(frame.preference_cost, [-1, .2, .5, 1.01], labels=BINS),
        reconfigs_per_total_operation=frame.completed_reconfigurations / frame.total_operations,
    )
    # operation_progress averages each order's completion fraction; it is not
    # completed_operations / total_operations when order lengths differ.
    assert frame.operation_progress.between(0, 1 + 1e-12).all()
    assert np.allclose(frame.loc[frame.task_succeeded, "operation_progress"], 1)
    assert (frame.completed_reconfigurations <= frame.total_operations).all()
    assert frame.groupby("instance_id").heuristic_reconfiguration_ratio.nunique().max() == 1
    return frame


def pair(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    assert not left.duplicated(keys).any() and not right.duplicated(keys).any()
    joined = left.merge(right, on=keys, suffixes=("_v8", "_shared"),
                        how="outer", indicator=True, validate="one_to_one")
    assert (joined._merge == "both").all()
    for column in IDENTITIES + ["heuristic_reconfiguration_ratio", "total_operations", "horizon"]:
        pd.testing.assert_series_equal(joined[column + "_v8"], joined[column + "_shared"],
                                       check_names=False)
    joined = joined.drop(columns="_merge").assign(
        cost_bin=joined.cost_bin_v8,
        outcome=np.select([
            joined.task_succeeded_v8 & joined.task_succeeded_shared,
            ~joined.task_succeeded_v8 & joined.task_succeeded_shared,
            joined.task_succeeded_v8 & ~joined.task_succeeded_shared,
        ], ["both_succeeded", "v8_failed_shared_succeeded", "v8_succeeded_shared_failed"],
            default="both_failed"),
    )
    return joined


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    aggregates = {"n": ("task_succeeded", "size"),
                  "successes": ("task_succeeded", "sum"),
                  "completion_rate": ("task_succeeded", "mean")}
    aggregates.update({f"mean_{metric}": (metric, "mean") for metric in METRICS})
    out = frame.groupby(groups, observed=True).agg(**aggregates).reset_index()
    return out.assign(failures=out.n - out.successes)


def paired_summaries(joined: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    instance_rows, tests = [], []
    for cohort, rows in [
        ("all", joined),
        ("both_succeeded", joined[joined.outcome == "both_succeeded"]),
        ("v8_failed_shared_succeeded", joined[joined.outcome == "v8_failed_shared_succeeded"]),
    ]:
        for (instance_id, cost_bin), cells in rows.groupby(["instance_id", "cost_bin"], observed=True):
            row = {"scope": scope, "cohort": cohort, "instance_id": instance_id,
                   "cost_bin": cost_bin, "n": len(cells)}
            for metric in ["task_succeeded"] + METRICS:
                for role in TRAIN_RUNS:
                    row[metric + "_" + role] = cells[metric + "_" + role].mean()
                row[metric + "_delta_v8_minus_shared"] = row[metric + "_v8"] - row[metric + "_shared"]
            instance_rows.append(row)
    per_instance = pd.DataFrame(instance_rows)
    for cost_bin, cells in per_instance[per_instance.cohort == "all"].groupby("cost_bin", observed=True):
        for metric in ["task_succeeded", "completed_reconfigurations", "machine_waiting_for_worker_time"]:
            delta = cells[metric + "_delta_v8_minus_shared"]
            p_value = wilcoxon(delta, method="exact").pvalue if (delta != 0).all() else float("nan")
            tests.append({"scope": scope, "cost_bin": cost_bin, "metric": metric,
                          "instance_count": len(delta), "mean_delta_v8_minus_shared": delta.mean(),
                          "positive_instance_count": int((delta > 0).sum()),
                          "negative_instance_count": int((delta < 0).sum()),
                          "wilcoxon_two_sided_p": p_value,
                          "interpretation": "descriptive_fixed_models_single_training_seed"})
    return per_instance, pd.DataFrame(tests)


def plot_diagnosis(late: pd.DataFrame, curves: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    colors, labels = {"v8": "#D55E00", "shared": "#0072B2"}, {"v8": "V8 experts", "shared": "Shared head"}
    for role in TRAIN_RUNS:
        c = curves[curves.role == role]
        axes[0, 0].plot(c.validation_episode, c.completion_rate, color=colors[role],
                        marker="o" if role == "v8" else "s", label=labels[role])
    axes[0, 0].set(xlabel="Training episode", ylabel="Completion rate", ylim=(0, 1.04))
    axes[0, 0].legend(frameon=False, fontsize=11)
    bar_specs = [(axes[0, 1], "completion_rate", "Completion rate", (0, 1.08)),
                 (axes[1, 0], "mean_completed_reconfigurations", "Completed reconfigurations", (0, 30)),
                 (axes[1, 1], "mean_machine_waiting_for_worker_time", "Machine wait for workers\n(machine-minutes)", (0, 160))]
    for ax, column, label, limits in bar_specs:
        for offset, role in [(-.18, "v8"), (.18, "shared")]:
            rows = late[late.role == role].set_index("cost_bin").loc[BINS]
            bars = ax.bar(np.arange(3) + offset, rows[column], width=.34, color=colors[role], label=labels[role])
            ax.bar_label(bars, fmt="%.3f" if column == "completion_rate" else "%.1f", fontsize=10, padding=3)
        ax.set(xticks=range(3), xticklabels=["≤ 0.2", "0.2–0.5", "> 0.5"],
               xlabel="Cost weight", ylabel=label, ylim=limits)
    for letter, ax in zip("abcd", axes.flat):
        ax.text(0, 1.035, f"({letter})", transform=ax.transAxes, weight="bold")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.savefig(OUTPUT / "diagnosis.png", dpi=300)
    fig.savefig(OUTPUT / "diagnosis.svg")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configs = {role: flatten(read_json(path / "config.json")) for role, path in TRAIN_RUNS.items()}
    differences = [{"field": key, "v8": repr(configs["v8"].get(key)), "shared": repr(configs["shared"].get(key))}
                   for key in sorted(configs["v8"].keys() | configs["shared"].keys())
                   if configs["v8"].get(key) != configs["shared"].get(key)]
    assert {row["field"] for row in differences} == {
        "experiment_name", "network.actor_head_variant", "runtime_manifest.actor_head_variant",
        "runtime_manifest.candidate_ranker"}
    write_csv("config_differences.csv", pd.DataFrame(differences))
    training = {role: read_csv(path / "train_log.csv") for role, path in TRAIN_RUNS.items()}
    matching_columns = ["episode", "instance_id", "seed", "pressure_type", "cost_profile", "severity",
                        "preference_key", "preference_group", "training_cache_fingerprint",
                        "generator_config_sha256", "environment_config_sha256", "distribution_contract_sha256"]
    matching_columns += [column for column in training["v8"] if column.startswith("instance_") and column != "instance_id"]
    pd.testing.assert_frame_equal(training["v8"][matching_columns], training["shared"][matching_columns])
    training_rows = []
    for role, frame in training.items():
        q = frame[frame.pressure_type == PRESSURE]
        for cohort, subset in [("all", q), ("severity_1", q[q.severity >= 1 - 1e-12]),
                               ("ep1001_1500", q[q.episode.between(1001, 1500)]),
                               ("ep1501_2000", q[q.episode.between(1501, 2000)])]:
            training_rows.append({"role": role, "cohort": cohort, "n": len(subset),
                                  "completion_rate": subset.task_succeeded.mean()})
    write_csv("training_completion.csv", pd.DataFrame(training_rows))
    stages = {}
    curves, summaries, per_instance, stats, reason_rows, protocol_rows = [], [], [], [], [], []
    for role, path in TRAIN_RUNS.items():
        frame = add_instance_metrics(read_csv(path / "sampled_validation_instance_metrics.csv", trajectories=True), "validation")
        counts = frame.groupby(["validation_episode", "instance_id"]).size()
        assert len(frame) == 39000 and (counts == 13 * 3).all()
        assert (frame.groupby(["validation_episode", "instance_id", "preference_key"]).size() == 3).all()
        q = frame[frame.pressure_type == PRESSURE].copy()
        curves.append(q.groupby("validation_episode").agg(n=("task_succeeded", "size"),
                       completion_rate=("task_succeeded", "mean")).reset_index().assign(role=role))
        stages.setdefault("validation_ep1000_2000", {})[role] = q[q.validation_episode.between(1000, 2000)].copy()
        stages.setdefault("test_best", {})[role] = add_instance_metrics(
            read_csv(EVAL_RUNS[role] / "instance_metrics.csv", trajectories=True), "test")
        test = stages["test_best"][role]
        assert len(test) == 3960 and (test.groupby("instance_id").size() == 66 * 3).all()
        assert (test.groupby(["instance_id", "preference_key"]).size() == 3).all()
        metadata = read_json(path / "summary.json")["checkpoint_selection"]
        provenance = read_json(EVAL_RUNS[role] / "metrics.json")["provenance"]
        assert test.checkpoint_sha256.nunique() == 1
        checkpoint = path / "best_checkpoint.pt"
        SOURCE_FILES.add(checkpoint)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == provenance["checkpoint_sha256"]
        for field in ["checkpoint_protocol_version", "evaluator_protocol_version", "git", "source_state_sha256"]:
            protocol_rows.append({"role": role, "field": field, "value": str(provenance[field])})
        protocol_rows.append({"role": role, "field": "best_episode", "value": metadata["best_episode"]})
    for scope, frames in stages.items():
        keys = (["validation_episode"] if scope.startswith("validation") else []) + CELL_KEYS
        joined = pair(frames["v8"], frames["shared"], keys)
        joined = joined[joined.pressure_type_v8 == PRESSURE].copy()
        write_csv(f"{scope}_paired_cells.csv", joined)
        write_csv(f"{scope}_paired_outcomes.csv", joined.groupby(["cost_bin", "outcome"], observed=True).size().rename("n").reset_index())
        pi, st = paired_summaries(joined, scope)
        per_instance.append(pi)
        stats.append(st)
        for role, frame in frames.items():
            q = frame[frame.pressure_type == PRESSURE]
            summaries.append(summarize(q, ["cost_bin"]).assign(scope=scope, role=role))
            write_csv(f"{scope}_{role}_preferences.csv", summarize(q, ["preference_flow", "preference_cost", "preference_variance"]))
            write_csv(f"{scope}_{role}_outcome_metrics.csv", summarize(q, ["cost_bin", "task_succeeded"]))
            reason_rows.append(q.groupby(["cost_bin", "termination_reason"], observed=True).size().rename("n").reset_index().assign(scope=scope, role=role))
            assert q.schedule_violation_count.sum() == 0 and q.physical_safety_pass.all()
            assert (q.task_failed == ~q.task_succeeded).all()
            assert np.allclose(q.loc[q.task_failed, "terminal_failure_penalty_applied"], 2)
            assert np.allclose(q.loc[q.task_succeeded, "terminal_failure_penalty_applied"], 0)
    summary = pd.concat(summaries, ignore_index=True)
    curve = pd.concat(curves, ignore_index=True)
    write_csv("summary_by_cost.csv", summary)
    write_csv("validation_curves.csv", curve)
    write_csv("paired_instance_metrics.csv", pd.concat(per_instance, ignore_index=True))
    write_csv("paired_instance_statistics.csv", pd.concat(stats, ignore_index=True))
    write_csv("failure_reasons.csv", pd.concat(reason_rows, ignore_index=True))
    write_csv("protocol_metadata.csv", pd.DataFrame(protocol_rows))
    plot_diagnosis(summary[summary.scope == "validation_ep1000_2000"], curve)
    write_csv("source_hashes.csv", pd.DataFrame([
        {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(SOURCE_FILES)]))
    print(summary[["scope", "role", "cost_bin", "n", "successes", "failures", "completion_rate",
                   "mean_completed_reconfigurations", "mean_machine_waiting_for_worker_time"]].round(6).to_string(index=False))
    print(f"Saved diagnosis to {OUTPUT}")


if __name__ == "__main__":
    main()
