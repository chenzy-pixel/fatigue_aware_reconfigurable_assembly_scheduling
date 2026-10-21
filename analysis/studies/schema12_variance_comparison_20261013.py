"""Audit saved schema12 Variance results against the matched schema10 run.

Uses recorded trajectories only. No training, policy rollout, or new test budget.
Run from the repository root: python -m analysis.studies.schema12_variance_comparison_20261013
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agent.ppo.network import infer_checkpoint_network_spec
from analysis.pareto_analysis import hypervolume_3d
from result.provenance import effective_config_snapshot, network_weights_sha256, dataset_manifest_snapshot
from utils import derive_evaluation_sampling_seed

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "schema10": ROOT / "result/runs/remaining_flow_experiments_seed11_20261008_225956_shared_head_universal",
    "schema12": ROOT / "result/runs/schema12_screen_seed11_20261012_201659_variance_log2",
}
OUT = ROOT / "result/analysis/schema12_variance_comparison_20261013"
OLD_LAST = ROOT / "result/analysis/shared_excess_last_20261012/last_test_instance_metrics.csv"
KEY = ["instance_id", "preference_key", "sampling_repeat"]
PREF = ["preference_flow", "preference_cost", "preference_variance"]
OBJECTIVES = ["flow_time_objective", "reconfiguration_cost", "worker_load_variance"]
SCALES = {"raw": np.array([1089.15, 353.27, 2.2629]), "excess": np.array([368.3143, 353.27, 2.2629])}
MEASURES = [*OBJECTIVES, "flow_excess_objective", "maximum_worker_fatigue", "completed_reconfigurations",
            "worker_reconfiguration_busy_minutes", "machine_waiting_for_worker_time", "production_wait_time",
            "worker_wait_time", "wait_action_count", "makespan", "decision_count"]
COLS = {*KEY, *PREF, *MEASURES, "preference_workload", "validation_episode", "task_succeeded", "task_failed",
        "physical_safety_pass", "schedule_violation_count", "safe_fatigue_limit", "sampling_truncated",
        "sampling_seed", "derived_sampling_seed", "sampling_rng_version", "termination_reason",
        "operation_progress", "initial_progress", "initial_preference_quality_score", "preference_quality_score",
        "actual_preference_quality_score", "terminal_failure_penalty_applied", "training_cumulative_reward",
        "base_cumulative_reward", "flow_processing_lower_bound", "dataset_manifest_sha256", "subset_sha256",
        "action_trace_sha256", "worker_loads", "worker_load_max", "worker_load_total", "workload_objective_value",
        "inference_time_seconds", "solve_time_seconds", "pressure_type", "cost_profile"}
SOURCES: set[Path] = {Path(__file__)}
CHECKS: list[dict] = []


def check(name, value):
    CHECKS.append({"check": name, "passed": bool(value)})
    if not value:
        raise AssertionError(name)


def js(path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path, selected=True):
    SOURCES.add(path)
    frame = pd.read_csv(path, usecols=(lambda c: c in COLS) if selected else None)
    if "preference_workload" in frame:
        frame["preference_variance"] = frame["preference_workload"]
    return frame


def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")


def safe(frame):
    return frame.task_succeeded & frame.physical_safety_pass & (frame.schedule_violation_count == 0)


def quality(frame, mode):
    fields = OBJECTIVES if mode == "raw" else ["flow_excess_objective", *OBJECTIVES[1:]]
    values = frame[fields].to_numpy(dtype=float)
    normalized = values / (values + SCALES[mode])
    weighted = normalized * frame[PREF].to_numpy(dtype=float)
    return pd.Series((weighted.max(axis=1) + .05 * weighted.sum(axis=1)) / 1.05, index=frame.index)


def aggregate(frame):
    good = frame[safe(frame)]
    return {"requests": len(frame), "successes": len(good), "failures": len(frame) - len(good),
            "overall_completion": len(good) / len(frame),
            "minimum_preference_completion": frame.groupby("preference_key").task_succeeded.mean().min(),
            "quality_raw": quality(good, "raw").groupby(good.preference_key).mean().mean(),
            "quality_excess": quality(good, "excess").groupby(good.preference_key).mean().mean(),
            "mean_return": frame.training_cumulative_reward.mean(),
            **{field: good[field].mean() for field in MEASURES if field in good}}


def paired(first, second, label):
    a = first.set_index(KEY).sort_index()
    b = second.set_index(KEY).sort_index()
    check(label + ":request_keys", a.index.equals(b.index))
    check(label + ":derived_seeds", a.derived_sampling_seed.equals(b.derived_sampling_seed))
    common = safe(a) & safe(b)
    outcomes = {"comparison": label, "common_success": int(common.sum()), "total": len(a),
                "common_success_coverage": float(common.mean()),
                "gained_success": int((~safe(a) & safe(b)).sum()),
                "lost_success": int((safe(a) & ~safe(b)).sum()),
                "both_fail": int((~safe(a) & ~safe(b)).sum())}
    measures = []
    for field in MEASURES:
        if field not in a or field not in b:
            continue
        x, y = a.loc[common, field], b.loc[common, field]
        measures.append({"comparison": label, "measure": field, "old_mean": x.mean(), "new_mean": y.mean(),
                         "percent_change": 100 * (y.mean() / x.mean() - 1),
                         "new_lower_fraction": float((y < x - 1e-9).mean())})
    for mode in SCALES:
        x, y = quality(a.loc[common], mode), quality(b.loc[common], mode)
        xm = x.groupby(level="preference_key").mean().mean()
        ym = y.groupby(level="preference_key").mean().mean()
        measures.append({"comparison": label, "measure": "quality_" + mode, "old_mean": xm, "new_mean": ym,
                         "percent_change": 100 * (ym / xm - 1), "new_lower_fraction": float((y < x - 1e-9).mean())})
    requests = pd.DataFrame({"old_success": safe(a), "new_success": safe(b),
                             "old_raw_quality": quality(a, "raw"), "new_raw_quality": quality(b, "raw"),
                             "old_excess_quality": quality(a, "excess"), "new_excess_quality": quality(b, "excess")}).reset_index()
    return outcomes, measures, requests


def hv(frame, label):
    rows = []
    for iid, group in frame.groupby("instance_id"):
        good = group[safe(group)]
        for mode in SCALES:
            fields = OBJECTIVES if mode == "raw" else ["flow_excess_objective", *OBJECTIVES[1:]]
            values = good[fields].to_numpy(dtype=float)
            rows.append({"model": label, "instance_id": iid, "mode": mode,
                         "hv": hypervolume_3d(values / (values + SCALES[mode]))})
    return rows


def best_of_three(frame):
    selected = frame.copy()
    selected["success"] = safe(selected)
    selected["selection_quality_excess"] = quality(selected, "excess")
    return selected.sort_values(["success", "selection_quality_excess", "sampling_repeat"],
                                ascending=[False, True, True]).drop_duplicates(KEY[:2])


def config_diff(old, new, path=""):
    if isinstance(old, dict) and isinstance(new, dict):
        rows = []
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                rows.append({"field": path + key, "old": old.get(key), "new": new.get(key)})
            else:
                rows.extend(config_diff(old[key], new[key], path + key + "."))
        return rows
    return [{"field": path[:-1], "old": old, "new": new}] if old != new else []


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    configs, summaries, tests, validations, train_frames, updates = {}, {}, {}, {}, {}, {}
    checkpoint_evidence, overview, curves, preference_curves, endpoints, failures, hvs = [], [], [], [], [], [], []
    test_manifest = ROOT / "data/manifests/v2/test/manifest.json"
    val_manifest = ROOT / "data/manifests/v2/validation/manifest.json"
    SOURCES.update([test_manifest, val_manifest])
    test_hash = dataset_manifest_snapshot(test_manifest)["sha256"]
    val_hash = dataset_manifest_snapshot(val_manifest)["sha256"]
    for label, directory in RUNS.items():
        c, s = js(directory / "config.json"), js(directory / "summary.json")
        configs[label], summaries[label] = c, s
        frame = read(directory / "final_sampled_instance_metrics.csv")
        validation = read(directory / "sampled_validation_instance_metrics.csv")
        train = read(directory / "train_log.csv", False)
        update = read(directory / "update_log.csv", False)
        log = read(directory / "validation_log.csv", False)
        tests[label], validations[label], train_frames[label], updates[label] = frame, validation, train, update
        p = s["final_sampled"]["provenance"]
        check(label + ":config_hash", effective_config_snapshot(c)["sha256"] == p["effective_config_sha256"])
        check(label + ":2000_episodes", s["episodes"] == len(train) == 2000 and train.episode.tolist() == list(range(1, 2001)))
        check(label + ":100_updates", len(update) == 100 and update.episode_count.sum() == 2000)
        check(label + ":training_not_truncated", not train.sampling_truncated.any())
        check(label + ":training_reward_identity", train.reward_identity_error.abs().max() < 1e-8)
        check(label + ":finite_ppo", np.isfinite(update[["policy_loss", "value_loss", "entropy", "loss", "approx_kl", "gradient_norm"]]).all().all())
        check(label + ":validation_schedule", log.episode.tolist() == list(range(100, 2001, 100)))
        check(label + ":complete_validation", len(validation) == 39000 and (validation.groupby("validation_episode").size() == 1950).all()
              and log.evaluation_complete.all() and log.physical_safety_pass.all() and (log.completion_coverage == 1).all())
        check(label + ":validation_unique", not validation.duplicated(["validation_episode", *KEY]).any())
        selected = log[log.checkpoint_eligible].sort_values(["completion_rate", "preference_balanced_quality_score", "episode"],
                                                          ascending=[False, True, True]).iloc[0]
        check(label + ":selection_reconstruction", int(selected.episode) == s["checkpoint_selection"]["best_episode"])
        check(label + ":complete_test", len(frame) == 3960 and frame.instance_id.nunique() == 20 and frame.preference_key.nunique() == 66
              and not frame.duplicated(KEY).any() and (frame.groupby(KEY[:2]).size() == 3).all() and not frame.sampling_truncated.any())
        check(label + ":test_summary_failure_count", int((~safe(frame)).sum()) == s["final_sampled"]["task_failed_count"])
        check(label + ":manifest_identity", set(frame.dataset_manifest_sha256) == {test_hash} == {p["dataset_manifest_sha256"]}
              and set(validation.dataset_manifest_sha256) == {val_hash})
        for stage, data in [("test", frame), ("validation", validation)]:
            check(label + ":" + stage + ":physical_safe", data.physical_safety_pass.all() and (data.schedule_violation_count == 0).all()
                  and (data.maximum_worker_fatigue <= data.safe_fatigue_limit + 1e-9).all())
            check(label + ":" + stage + ":quality", np.allclose(quality(data[safe(data)], "excess"), data.loc[safe(data), "preference_quality_score"], atol=1e-8))
            check(label + ":" + stage + ":actual_quality", np.allclose(quality(data, "excess"), data.actual_preference_quality_score, atol=1e-8))
            check(label + ":" + stage + ":penalty", np.allclose(data.terminal_failure_penalty_applied, 2 * data.task_failed.astype(int)))
            check(label + ":" + stage + ":reward", np.allclose(data.training_cumulative_reward,
                  data.operation_progress - data.initial_progress + data.initial_preference_quality_score
                  - data.actual_preference_quality_score - data.terminal_failure_penalty_applied, atol=1e-8))
            check(label + ":" + stage + ":finite_objectives", np.isfinite(data[OBJECTIVES + ["flow_excess_objective"]]).all().all())
            check(label + ":" + stage + ":flow_lb", np.allclose(data.loc[safe(data), "flow_excess_objective"],
                  data.loc[safe(data), "flow_time_objective"] - data.loc[safe(data), "flow_processing_lower_bound"], atol=1e-8))
        check(label + ":derived_seeds", all(int(row.derived_sampling_seed) == derive_evaluation_sampling_seed(
              int(row.sampling_seed), row.instance_id, row.preference_key) for row in frame.itertuples()))
        for role in ("best", "last"):
            path = directory / (role + "_checkpoint.pt")
            SOURCES.add(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            spec = infer_checkpoint_network_spec(payload)
            metadata = payload["metadata"]
            weights_hash = network_weights_sha256(payload["network"])
            check(label + ":" + role + ":weights_hash", weights_hash == metadata["network_weights_sha256"])
            check(label + ":" + role + ":episode", metadata["checkpoint_episode"] == (s["checkpoint_selection"]["best_episode"] if role == "best" else 2000))
            check(label + ":" + role + ":schema", spec["observation_schema_version"] == (12 if label == "schema12" else 10))
            check(label + ":" + role + ":shared_excess", spec["actor_head_variant"] == "shared_preference" and spec["flow_mode"] == "excess_proportional_lb_v1")
            if label == "schema12":
                check(label + ":" + role + ":encoding_workload", spec["global_objective_encoding"] == "log2_v1" and spec["workload_mode"] == "variance"
                      and spec["feature_dimensions"]["global"] == 10)
            else:
                check(label + ":" + role + ":nine_globals", spec["feature_dimensions"]["global"] == 9)
            if role == "best":
                check(label + ":checkpoint_file_hash", hashlib.sha256(path.read_bytes()).hexdigest() == p["checkpoint_sha256"])
                check(label + ":provenance_weights", weights_hash == p["network_weights_sha256"])
            checkpoint_evidence.append({"model": label, "role": role, "episode": metadata["checkpoint_episode"],
                                       "parameter_count": sum(t.numel() for t in payload["network"].values()),
                                       "schema": spec["observation_schema_version"], "weights_sha256": weights_hash,
                                       "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            del payload
        overview.append({"model": label, "stage": "best_test", "episode": s["checkpoint_selection"]["best_episode"], **aggregate(frame)})
        hvs.extend(hv(frame, label))
        for episode, group in validation.groupby("validation_episode"):
            item = {"model": label, "episode": int(episode), **aggregate(group)}
            cost = group[group.preference_cost == 1]
            item.update(cost_completion=float(safe(cost).mean()), cost_successes=int(safe(cost).sum()),
                        cost_mean_success=float(cost.loc[safe(cost), "reconfiguration_cost"].mean()),
                        cost_mean_return=float(cost.training_cumulative_reward.mean()),
                        learning_rate=float(log.loc[log.episode == episode, "learning_rate"].iloc[0]))
            curves.append(item)
            for key, request in group.groupby("preference_key"):
                preference_curves.append({"model": label, "episode": int(episode), "preference_key": key,
                                          **dict(zip(PREF, request[PREF].iloc[0].tolist())),
                                          "completion": float(safe(request).mean()), "failures": int((~safe(request)).sum()),
                                          "quality_excess": float(quality(request[safe(request)], "excess").mean())})
        for stage, data in [("best_test", frame), ("best_validation", validation[validation.validation_episode == s["checkpoint_selection"]["best_episode"]]),
                            ("last_validation", validation[validation.validation_episode == 2000])]:
            if stage != "best_test":
                overview.append({"model": label, "stage": stage, "episode": s["checkpoint_selection"]["best_episode"] if stage == "best_validation" else 2000,
                                 **aggregate(data)})
            for name, col in zip(("Flow", "Cost", "Variance"), PREF):
                group = data[data[col] == 1]
                endpoints.append({"model": label, "stage": stage, "endpoint": name, **aggregate(group)})
            bad = data[~safe(data)].copy()
            bad["model"], bad["stage"] = label, stage
            failures.extend(bad.to_dict("records"))
    for field in ("seed", "ppo", "generator", "dataset", "environment", "preference"):
        check("matched_config:" + field, configs["schema10"].get(field) == configs["schema12"].get(field))
    for field in ("fixed_instance_sha256", "generator_config_sha256", "environment_config_sha256", "distribution_contract_sha256",
                  "template_sha256", "dataset_manifest_sha256", "normalization_manifest_sha256", "evaluation_subset_sha256"):
        check("matched_provenance:" + field, summaries["schema10"]["final_sampled"]["provenance"][field] == summaries["schema12"]["final_sampled"]["provenance"][field])
    check("matched_validation_subsets", js(RUNS["schema10"] / "validation_subsets.json") == js(RUNS["schema12"] / "validation_subsets.json"))
    check("matched_training_instance_sequence", train_frames["schema10"][["instance_id", "seed", "preference_key", "pressure_type", "cost_profile"]].equals(
          train_frames["schema12"][["instance_id", "seed", "preference_key", "pressure_type", "cost_profile"]]))
    pairs, paired_measures = [], []
    outcome, measures, requests = paired(tests["schema10"], tests["schema12"], "best_test")
    pairs.append(outcome); paired_measures.extend(measures); save("paired_test_requests", requests)
    a = validations["schema10"]; b = validations["schema12"]
    for stage, ep_a, ep_b in [("selected_validation", 400, 1800), ("last_validation", 2000, 2000)]:
        outcome, measures, _ = paired(a[a.validation_episode == ep_a], b[b.validation_episode == ep_b], stage)
        pairs.append(outcome); paired_measures.extend(measures)
    outcome, measures, _ = paired(b[b.validation_episode == 1800], b[b.validation_episode == 2000], "schema12_best_vs_last_validation")
    pairs.append(outcome); paired_measures.extend(measures)
    diagnostic_rows, return_decomposition = [], []
    for label, directory in RUNS.items():
        diagnostic = read(directory / "sampled_diagnostic_instance_metrics.csv")
        last_diagnostic = diagnostic[diagnostic.validation_episode == 2000]
        target = validations[label][validations[label].validation_episode == 2000]
        check(label + ":independent_diagnostic", len(last_diagnostic) == 1911 and last_diagnostic.instance_id.nunique() == 49
              and not last_diagnostic.duplicated(KEY).any() and not set(last_diagnostic.instance_id) & set(target.instance_id))
        check(label + ":diagnostic_safe", last_diagnostic.physical_safety_pass.all() and (last_diagnostic.schedule_violation_count == 0).all())
        diagnostic_rows.append({"model": label, "stage": "last_independent_validation", **aggregate(last_diagnostic)})
        for ep in (summaries[label]["checkpoint_selection"]["best_episode"], 2000):
            group = validations[label][validations[label].validation_episode == ep]
            return_decomposition.append({"model": label, "episode": int(ep), "progress_mean": group.operation_progress.mean(),
                                         "actual_quality_mean_all_requests": group.actual_preference_quality_score.mean(),
                                         "failure_penalty_mean": group.terminal_failure_penalty_applied.mean(),
                                         "mean_return": group.training_cumulative_reward.mean()})
    indexed = {label: frame.set_index(KEY).sort_index() for label, frame in tests.items()}
    first, second = indexed["schema10"], indexed["schema12"]
    common = safe(first) & safe(second)
    preference_pairs, variance_groups = [], []
    for key, request in first.groupby(level="preference_key"):
        mask = common & (first.index.get_level_values("preference_key") == key)
        x, y = first.loc[mask], second.loc[mask]
        preference_pairs.append({"preference_key": key, **dict(zip(PREF, request[PREF].iloc[0].tolist())),
                                 "common_success": int(mask.sum()),
                                 "old_successes": int(safe(request).sum()),
                                 "new_successes": int(safe(second.loc[request.index]).sum()),
                                 **{"old_" + field: x[field].mean() for field in OBJECTIVES},
                                 **{"new_" + field: y[field].mean() for field in OBJECTIVES},
                                 "old_quality_excess": quality(x, "excess").mean(), "new_quality_excess": quality(y, "excess").mean()})
    for name, weights in [("zero_variance_weight", first.preference_variance == 0),
                          ("positive_variance_weight", first.preference_variance > 0),
                          ("variance_weight_ge_half", first.preference_variance >= .5),
                          ("pure_variance", first.preference_variance == 1)]:
        mask = common & weights
        x, y = first.loc[mask], second.loc[mask]
        variance_groups.append({"preference_group": name, "common_success": int(mask.sum()),
                                "old_variance_mean": x.worker_load_variance.mean(), "new_variance_mean": y.worker_load_variance.mean(),
                                "old_variance_median": x.worker_load_variance.median(), "new_variance_median": y.worker_load_variance.median()})
    budgets = []
    for label, data in tests.items():
        selected = best_of_three(data)
        budgets.append({"model": label, "sampling_budget": 3, "selection": "success_first_then_excess_quality", **aggregate(selected)})
    if OLD_LAST.exists():
        old_last = read(OLD_LAST)
        check("old_last:complete", len(old_last) == 3960 and not old_last.duplicated(KEY).any())
        overview.append({"model": "schema10_last", "stage": "last_test_supplement", "episode": 2000, **aggregate(old_last)})
        outcome, measures, _ = paired(old_last, tests["schema12"], "old_last_vs_new_best_test")
        pairs.append(outcome); paired_measures.extend(measures)
        hvs.extend(hv(old_last, "schema10_last"))
        for name, col in zip(("Flow", "Cost", "Variance"), PREF):
            group = old_last[old_last[col] == 1]
            endpoints.append({"model": "schema10_last", "stage": "last_test_supplement", "endpoint": name, **aggregate(group)})
        budgets.append({"model": "schema10_last", "sampling_budget": 3, "selection": "success_first_then_excess_quality", **aggregate(best_of_three(old_last))})
    curve = pd.DataFrame(curves)
    grouped = []
    for label, frame in tests.items():
        for field in ("pressure_type", "cost_profile", "preference_key"):
            for value, group in frame.groupby(field):
                grouped.append({"model": label, "group_field": field, "group": value, **aggregate(group)})
    hv_frame = pd.DataFrame(hvs)
    by_instance = requests.groupby("instance_id").apply(lambda g: pd.Series({
        "completion_delta": g.new_success.mean() - g.old_success.mean(),
        "raw_quality_delta_common": (g.loc[g.old_success & g.new_success, "new_raw_quality"] - g.loc[g.old_success & g.new_success, "old_raw_quality"]).mean(),
        "excess_quality_delta_common": (g.loc[g.old_success & g.new_success, "new_excess_quality"] - g.loc[g.old_success & g.new_success, "old_excess_quality"]).mean(),
    }), include_groups=False)
    rng = np.random.default_rng(20261013)
    draws = rng.integers(0, len(by_instance), size=(20000, len(by_instance)))
    uncertainty = []
    for field in by_instance:
        values = by_instance[field].to_numpy()
        sampled = values[draws].mean(axis=1)
        low, high = np.quantile(sampled, [.025, .975])
        uncertainty.append({"metric": field, "unit": "test_instance_cluster", "clusters": len(values),
                            "mean_delta": values.mean(), "ci95_lower": low, "ci95_upper": high})
    late = []
    for label, group in curve[curve.episode >= 1600].groupby("model"):
        late.append({"model": label, "validation_points": len(group),
                     "overall_completion_mean": group.overall_completion.mean(),
                     "minimum_preference_completion_mean": group.minimum_preference_completion.mean(),
                     "minimum_preference_completion_low": group.minimum_preference_completion.min(),
                     "cost_completion_mean": group.cost_completion.mean(), "cost_completion_low": group.cost_completion.min()})
    save("overview", overview); save("endpoints", endpoints); save("validation_curve", curves)
    save("preference_validation_curve", preference_curves); save("failures", failures)
    save("paired_counts", pairs); save("paired_metrics", paired_measures); save("best_of_three", budgets)
    save("hypervolume_by_instance", hvs); save("stratified_test", grouped)
    save("cluster_bootstrap", uncertainty); save("late_validation", late)
    save("checkpoint_evidence", checkpoint_evidence); save("checks", CHECKS)
    save("last_independent_validation", diagnostic_rows); save("return_decomposition", return_decomposition)
    save("paired_preferences", preference_pairs); save("variance_preference_groups", variance_groups)
    (OUT / "config_differences.json").write_text(json.dumps(config_diff(configs["schema10"], configs["schema12"]), ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"runs": {label: path.as_posix() for label, path in RUNS.items()}, "overview": overview,
              "paired_counts": pairs, "paired_metrics": paired_measures, "endpoints": endpoints,
              "hypervolume_mean": hv_frame.groupby(["model", "mode"]).hv.mean().unstack().to_dict("index"),
              "best_of_three": budgets, "late_validation": late, "cluster_bootstrap": uncertainty,
              "last_independent_validation": diagnostic_rows, "return_decomposition": return_decomposition,
              "variance_preference_groups": variance_groups,
              "checks_passed": len(CHECKS), "checkpoint_evidence": checkpoint_evidence,
              "runtime_seconds": {k: v["elapsed_seconds"] for k, v in summaries.items()},
              "provenance_git": {k: v["final_sampled"]["provenance"]["git"] for k, v in summaries.items()},
              "new_last_test_available": False, "additional_policy_rollouts": 0,
              "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}}
    (OUT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    plot(curve, hv_frame)
    report(result)
    print(json.dumps({k: result[k] for k in ("checks_passed", "overview", "paired_counts", "hypervolume_mean", "late_validation")}, ensure_ascii=False, indent=2))


def plot(curve, hv_frame):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    for label, group in curve.groupby("model"):
        for ax, field, title in zip(axes.flat,
                ["overall_completion", "minimum_preference_completion", "cost_completion", "quality_excess"],
                ["Overall completion (%)", "Worst preference completion (%)", "Cost endpoint completion (%)", "Successful excess quality (lower better)"]):
            scale = 1 if field == "quality_excess" else 100
            ax.plot(group.episode, group[field] * scale, marker=".", label=label)
            ax.set_xlabel("Training episodes"); ax.set_ylabel(title); ax.grid(alpha=.2)
    axes[0, 0].legend()
    for suffix in ("png", "svg"):
        fig.savefig(OUT / ("validation_comparison." + suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, mode in zip(axes, SCALES):
        for model, group in hv_frame[hv_frame["mode"] == mode].groupby("model"):
            ax.plot(range(1, len(group) + 1), group.sort_values("instance_id").hv, marker=".", label=model)
        ax.set_xlabel("Matched test instance index"); ax.set_ylabel(mode + " pooled HV"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / ("test_hypervolume." + suffix), dpi=180)
    plt.close(fig)


def report(result):
    overview = {(r["model"], r["stage"]): r for r in result["overview"]}
    new, old = overview["schema12", "best_test"], overview["schema10", "best_test"]
    last = overview["schema12", "last_validation"]
    pair = next(r for r in result["paired_counts"] if r["comparison"] == "best_test")
    measures = {r["measure"]: r for r in result["paired_metrics"] if r["comparison"] == "best_test"}
    text = "# Schema12 Variance+log2 与历史 Variance Universal 对照（2026-10-13）\n\n"
    text += "本次只读取已保存模型、日志和轨迹，不新增训练或策略测试。主要对照为两次实验各自按原规则选出的 best。\n\n"
    text += f"新模型 best 为 ep1800，测试 {new['successes']}/3960 成功；旧 best 为 ep400，{old['successes']}/3960 成功。新 best 在这次固定测试上同时提高完成率和平均质量，但新 last 的最差偏好验证完成率仅 {100*last['minimum_preference_completion']:.3f}%，后期稳定性问题仍在。\n\n"
    text += "| 指标 | schema10 best | schema12 best |\n|---|---:|---:|\n"
    for title, field, scale in [("整体测试完成率 (%)", "overall_completion", 100), ("最差偏好测试完成率 (%)", "minimum_preference_completion", 100),
                                ("失败次数", "failures", 1), ("共同 raw 坐标质量损失", "quality_raw", 1), ("共同 excess 坐标质量损失", "quality_excess", 1)]:
        text += f"| {title} | {old[field]*scale:.6f} | {new[field]*scale:.6f} |\n"
    for mode in SCALES:
        text += f"| {mode} 平均逐实例 pooled HV | {result['hypervolume_mean']['schema10'][mode]:.6f} | {result['hypervolume_mean']['schema12'][mode]:.6f} |\n"
    text += "\n## 配对质量\n\n"
    text += f"相同20测试实例、66偏好、3次采样，种子300011/300012/300013，匹配 derived seed。共同成功 {pair['common_success']}/3960（{100*pair['common_success_coverage']:.3f}%）；新增成功 {pair['gained_success']}，失去旧成功 {pair['lost_success']}，双方失败 {pair['both_fail']}。只在共同成功集合比较质量，防止失败请求低成本造成假改善。\n\n"
    text += "| 共同成功指标 | 旧 best | 新 best | 变化 (%) |\n|---|---:|---:|---:|\n"
    for field in [*OBJECTIVES, "flow_excess_objective", "quality_raw", "quality_excess", "completed_reconfigurations", "production_wait_time", "maximum_worker_fatigue"]:
        row = measures[field]
        text += f"| {field} | {row['old_mean']:.6f} | {row['new_mean']:.6f} | {row['percent_change']:+.3f} |\n"
    text += "\n负变化表示该指标下降；疲劳下降不等同于三个训练目标都改善。HV每实例独立计算后平均，成功候选集合为66偏好×3采样，参考点(1,1,1)，没有混合不同实例的Pareto点。共同raw尺度(1089.15,353.27,2.2629)、excess尺度(368.3143,353.27,2.2629)，归一化 q=J/(s+J)，质量损失=(max(wq)+0.05 sum(wq))/1.05。各偏好先平均再等权聚合。\n\n"
    text += "新best总体方差均值显著增大，不能称为三个目标全面改善。按偏好分开看，其承担的折中更清楚：\n\n| 方差偏好分组（共同成功） | 样本数 | 旧方差 | 新方差 |\n|---|---:|---:|---:|\n"
    for row in result["variance_preference_groups"]:
        text += f"| {row['preference_group']} | {row['common_success']} | {row['old_variance_mean']:.4f} | {row['new_variance_mean']:.4f} |\n"
    text += "\n高方差权重请求确有均衡收益，低/零方差权重请求则允许负担更加集中。完整66偏好配对明细见 paired_preferences.csv。不能把不同偏好混合后的方差均值理解为纯Variance目标的性能。\n\n"
    text += "## 端点与后期\n\n| 模型/阶段 | 端点 | 成功/总数 | Cost | 方差 |\n|---|---|---:|---:|---:|\n"
    for row in result["endpoints"]:
        text += f"| {row['model']}/{row['stage']} | {row['endpoint']} | {row['successes']}/{row['requests']} | {row['reconfiguration_cost']:.3f} | {row['worker_load_variance']:.3f} |\n"
    text += "\n端点目标均值只使用该端点成功轨迹。validation_log 的 completion_rate 是最差偏好完成率，整体完成率由1950条轨迹重算；它不是全部轨迹的成功比例。末五点及所有偏好曲线分别见 late_validation.csv 和 preference_validation_curve.csv。\n\n"
    text += f"schema12 last(ep2000)验证整体 {last['successes']}/1950（{100*last['overall_completion']:.3f}%），最差偏好 {100*last['minimum_preference_completion']:.3f}%，excess质量 {last['quality_excess']:.6f}。last 尚未做完整测试，不能把这些验证值写作测试成绩。\n\n"
    new_cost_last = next(r for r in result["endpoints"] if r["model"] == "schema12" and r["stage"] == "last_validation" and r["endpoint"] == "Cost")
    text += f"其中纯Cost {new_cost_last['successes']}/150 成功，承担 {150-new_cost_last['successes']} 次失败；全部last失败 {last['failures']} 次。独立49实例验证提供相同风险信号：\n\n| last模型 | 独立验证成功 | 最差偏好完成率 |\n|---|---:|---:|\n"
    for row in result["last_independent_validation"]:
        text += f"| {row['model']} | {row['successes']}/{row['requests']} | {100*row['minimum_preference_completion']:.3f}% |\n"
    text += "\n新模型best的5次测试失败均为reconfiguration_bottleneck场景的horizon终止，涉及2个实例，失败时疲劳峰值接近0.75且重构21–24次。这是保存轨迹的相关证据，没有动作回放，不能据此断言单一因果机制；安全标志仍全部通过。\n\n"
    text += "schema12 best到last的全请求平均回报分解：\n\n| 轮次 | 平均progress | 平均实际Q（含失败） | 平均罚项 | 平均回报 |\n|---|---:|---:|---:|---:|\n"
    for row in result["return_decomposition"]:
        if row["model"] == "schema12":
            text += f"| {row['episode']} | {row['progress_mean']:.6f} | {row['actual_quality_mean_all_requests']:.6f} | {row['failure_penalty_mean']:.6f} | {row['mean_return']:.6f} |\n"
    text += "\n多数请求的Q下降仍可以抵消新增失败的平均罚项，导致平均回报微升但最差偏好明显退步；纯Cost自己的平均回报实际上下降。全局裕量提供了信息，但没有改变这一期望回报交换机制。completion-first选模保留ep1800符合既定规则。\n\n"
    text += "## 预算与出处\n\n"
    text += "| 模型 | 三次采样至少一次成功单元 | 选择后excess质量 |\n|---|---:|---:|\n"
    for row in result["best_of_three"]:
        text += f"| {row['model']} | {row['successes']}/{row['requests']} | {row['quality_excess']:.6f} |\n"
    text += "\n三次采样择优来自同一已保存数据，以成功优先、再按对应偏好的excess质量选择；这是1320个实例/偏好单元的三次推断预算，不与3960条单次请求完成率混淆。\n\n"
    text += "旧last作为补充对照时，新best也提高完成率，但共同成功样本的excess质量损失高约2.93%，Cost高约32.89%、方差高约118.58%，Flow低约10.58%；其raw/excess HV也低于旧last。新best不能替代旧last的所有折中点。两次best的主要结果属于相同选模规则下的改善，而非支配所有历史模型。\n\n"
    text += f"共 {result['checks_passed']} 项审计通过：2000轮/100次更新、完整验证和测试、保存配置与checkpoint哈希、权重身份、物理安全、罚项和奖励恒等式、冻结尺度、数据/偏好/种子匹配。训练实例ID、生成种子、偏好key和分布顺序一致。奖励代数保持 progress-Q-2F，但新旧观测/协议与代码版本不同，因此是整体版本对照，不能把收益单独归因于裕量或log2。\n\n"
    text += "两次provenance均记录git dirty=true，只保存聚合source_state哈希、没有逐文件内容快照，不能仅凭git commit证明远端执行源码逐文件一致；保存模型、配置、结果身份已核验。config_differences.json记录全部配置差异。旧best CSV没有逐工人load向量，所以这次主对照不编造其OWA、最大个人负担或总劳动；新best虽有load向量，可在后续OWA配对中使用。旧last完整测试仅作为补充，不能冒充旧best。\n\n"
    text += "cluster_bootstrap.csv对20个测试实例做配对簇bootstrap，保留实例内部偏好和采样相关性；它只描述此测试分布的不确定性，不能替代训练多seed。新best对旧best是积极单seed结果，先继续既定OWA/log2对照；编码控制仍在目标选择之后决定。\n\n"
    text += "![验证完成率与质量](validation_comparison.png)\n\n![逐实例HV](test_hypervolume.png)\n\n"
    text += "复现：`python -m analysis.studies.schema12_variance_comparison_20261013`。comparison.json包含计算结果、模型身份、输入文件SHA256；checks.csv记录审计；paired_test_requests.csv、paired_metrics.csv、endpoints.csv、failures.csv提供可核查明细。\n"
    (OUT / "report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
