"""Audit the completed seven-run batch from saved artifacts, without policy replay.

Run: python -m analysis.studies.remaining_batch_review_20261012
The historical variance objective and saved scales are used explicitly. Current
configuration defaults and the current workload environment are not loaded.
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
import torch
from scipy.stats import spearmanr

from analysis.pareto_analysis import hypervolume_3d
from environment.dynamics import quantize_to_ticks
from result.provenance import effective_config_snapshot, network_weights_sha256, dataset_manifest_snapshot
from utils import derive_evaluation_sampling_seed

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "result/runs"
OUT = ROOT / "result/analysis/remaining_batch_review_20261012"
PREFIX = "remaining_flow_experiments_seed11_20261008_225956_"
ROLES = ("main_variance", "main_universal", "flow", "shared_head_flow",
         "shared_head_cost", "shared_head_variance", "shared_head_universal")
OBJECTIVES = ["flow_time_objective", "reconfiguration_cost", "worker_load_variance"]
PREFS = ["preference_flow", "preference_cost", "preference_variance"]
KEYS = ["instance_id", "preference_key", "sampling_repeat"]
SCALES = {"raw": np.array([1089.15, 353.27, 2.2629]),
          "excess": np.array([368.3143, 353.27, 2.2629])}
SOURCES: set[Path] = {Path(__file__)}
CHECKS: list[dict] = []


def check(name, condition):
    CHECKS.append({"check": name, "passed": bool(condition)})
    if not condition:
        raise AssertionError(name)


def js(path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def csv(path, **kwargs):
    SOURCES.add(path)
    return pd.read_csv(path, **kwargs)


def save(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")


def safe(frame):
    return frame.task_succeeded & frame.physical_safety_pass & (frame.schedule_violation_count == 0)


def quality(frame, mode):
    fields = OBJECTIVES if mode == "raw" else ["flow_excess_objective", *OBJECTIVES[1:]]
    values = frame[fields].to_numpy(dtype=float)
    normalized = values / (values + SCALES[mode])
    weighted = normalized * frame[PREFS].to_numpy(dtype=float)
    return (weighted.max(axis=1) + .05 * weighted.sum(axis=1)) / 1.05


def front(points):
    points = np.unique(points, axis=0)
    slack = 1e-9 * np.maximum(1., np.maximum(abs(points[:, None, :]), abs(points[None, :, :])))
    dominates = np.all(points[:, None, :] <= points[None, :, :] + slack, axis=2) & np.any(
        points[:, None, :] < points[None, :, :] - slack, axis=2)
    return points[~dominates.any(axis=0)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    launch = js(RUNS / (PREFIX + "startup") / "launch.json")
    check("launch_order", [j["experiment"] for j in launch["jobs"]] == list(ROLES))
    check("all_jobs_completed", all(j["status"] == "completed" and j["exit_code"] == 0 for j in launch["jobs"]))
    manifest_path = ROOT / "data/manifests/v2/test/manifest.json"
    manifest = js(manifest_path)
    manifest_hash = dataset_manifest_snapshot(manifest_path)["sha256"]
    val_manifest = ROOT / "data/manifests/v2/validation/manifest.json"
    SOURCES.add(val_manifest)
    val_hash = dataset_manifest_snapshot(val_manifest)["sha256"]
    lbs = {}
    for entry in manifest["files"]:
        path = ROOT / "data/instances/v2/test" / entry["path"]
        raw = path.read_bytes()
        check("test_instance_hash:" + entry["path"], hashlib.sha256(raw).hexdigest() == entry["sha256"])
        instance = js(path)["instance"]
        resolution = instance["time"]["resolution"]
        lbs[instance["instance_id"]] = resolution * sum(
            min(max(1, quantize_to_ticks(op["base_processing_time"] * machine["module_parameters"][op["required_module"]]["processing_speed_factor"], resolution))
                for machine in instance["machines"] if op["required_module"] in machine["module_parameters"])
            for order in instance["orders"] for op in order["operations"])

    data, validations, summaries, configs = {}, {}, {}, {}
    overview, runtime, ppo, validation_rows, endpoint_rows, failures, stability = [], [], [], [], [], [], []
    needed = {"validation_episode", *KEYS, *OBJECTIVES, *PREFS, "flow_excess_objective", "preference_quality_score",
              "task_succeeded", "physical_safety_pass", "schedule_violation_count", "sampling_truncated"}
    for role in ROLES:
        directory = RUNS / (PREFIX + role)
        config, summary = js(directory / "config.json"), js(directory / "summary.json")
        configs[role], summaries[role] = config, summary
        universal = role.endswith("universal")
        expected_episodes = 2000 if universal else 1000
        mode = "raw" if role.startswith("main_") else "excess"
        ck_path = directory / "best_checkpoint.pt"
        SOURCES.add(ck_path)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        spec, metadata = ck["network_spec"], ck["metadata"]
        provenance = summary["final_sampled"]["provenance"]
        check(role + ":config_startup", config == js(RUNS / (PREFIX + "startup") / (role + "_config.json")))
        check(role + ":config_hash", effective_config_snapshot(config)["sha256"] == provenance["effective_config_sha256"])
        check(role + ":checkpoint_hash", hashlib.sha256(ck_path.read_bytes()).hexdigest() == provenance["checkpoint_sha256"])
        check(role + ":network_hash", network_weights_sha256(ck["network"]) == provenance["network_weights_sha256"] == metadata["network_weights_sha256"])
        check(role + ":checkpoint_identity", spec["observation_schema_version"] == 10
              and spec["actor_head_variant"] == ("shared_preference" if role.startswith("shared_") else "objective_experts")
              and spec["flow_mode"] == config["objective_scalarizer"].get("flow_mode", "raw_v1")
              and spec["reward_version"] == config["reward"]["mode"])
        best_episode = summary["checkpoint_selection"]["best_episode"]
        check(role + ":checkpoint_episode", metadata["checkpoint_episode"] == best_episode)
        parameter_count = sum(t.numel() for t in ck["network"].values())
        del ck
        train = csv(directory / "train_log.csv")
        updates = csv(directory / "update_log.csv")
        val = csv(directory / "validation_log.csv")
        diag = csv(directory / "diagnostic_log.csv")
        frame = csv(directory / "final_sampled_instance_metrics.csv").sort_values(KEYS).reset_index(drop=True).copy()
        check(role + ":episodes", len(train) == expected_episodes == summary["episodes"] and train.episode.tolist() == list(range(1, expected_episodes + 1)))
        check(role + ":updates", len(updates) == expected_episodes // 20 and int(updates.episode_count.sum()) == expected_episodes)
        check(role + ":validation_schedule", val.episode.tolist() == list(range(100 if universal else 40, expected_episodes + 1, 100 if universal else 40)))
        check(role + ":validation_complete_safe", val.evaluation_complete.all() and val.physical_safety_pass.all()
              and (val.completion_coverage == 1).all() and (val.schedule_violation_count == 0).all())
        check(role + ":sampling_complete", len(frame) == (3960 if universal else 60) and not frame.duplicated(KEYS).any()
              and set(frame.instance_id) == set(lbs) and frame.preference_key.nunique() == (66 if universal else 1)
              and (frame.groupby(KEYS[:2]).size() == 3).all() and not frame.sampling_truncated.any())
        check(role + ":physical_safety", frame.physical_safety_pass.all() and (frame.schedule_violation_count == 0).all()
              and (frame.maximum_worker_fatigue <= frame.safe_fatigue_limit + 1e-9).all())
        check(role + ":test_manifest", set(frame.dataset_manifest_sha256) == {manifest_hash} == {provenance["dataset_manifest_sha256"]})
        check(role + ":val_manifest", set(val.dataset_manifest_sha256) == {val_hash})
        check(role + ":flow_bound", np.allclose(frame.flow_processing_lower_bound, frame.instance_id.map(lbs), atol=1e-8))
        ok = safe(frame)
        check(role + ":flow_excess", np.allclose(frame.loc[ok, "flow_excess_objective"],
              frame.loc[ok, "flow_time_objective"] - frame.loc[ok, "flow_processing_lower_bound"], atol=1e-8))
        check(role + ":quality_reconstruction", np.allclose(quality(frame.loc[ok], mode), frame.loc[ok, "preference_quality_score"], atol=1e-8))
        check(role + ":actual_quality_reconstruction", np.allclose(quality(frame, mode), frame.actual_preference_quality_score, atol=1e-8))
        check(role + ":train_reward_identity", train.reward_identity_error.abs().max() <= 1e-8
              and np.allclose(train.reward, train.expected_reward, atol=1e-8))
        check(role + ":test_reward_identity", np.allclose(frame.training_cumulative_reward,
              frame.operation_progress - frame.initial_progress + frame.initial_preference_quality_score
              - frame.actual_preference_quality_score - frame.terminal_failure_penalty_applied, atol=1e-8))
        check(role + ":test_base_reward_identity", np.allclose(frame.base_cumulative_reward,
              frame.training_cumulative_reward + frame.terminal_failure_penalty_applied, atol=1e-8))
        check(role + ":sampling_seeds", all(int(row.derived_sampling_seed) == derive_evaluation_sampling_seed(
            int(row.sampling_seed), row.instance_id, row.preference_key if universal else None) for row in frame.itertuples()))
        for metric in OBJECTIVES:
            check(role + ":summary_mean:" + metric, abs(frame.loc[ok, metric].mean() - summary["final_sampled"]["completed_metrics"][metric]["mean"]) < 1e-7)
        check(role + ":summary_failure_count", int((~ok).sum()) == summary["final_sampled"]["task_failed_count"])
        # Replay the saved validation rankings, without evaluating a policy.
        eligible = val[val.checkpoint_eligible]
        selected = eligible.sort_values(["completion_rate", "preference_balanced_quality_score", "episode"], ascending=[False, True, True]).iloc[0]
        check(role + ":selection_rank", int(selected.episode) == best_episode)
        samples = csv(directory / "sampled_validation_instance_metrics.csv", usecols=lambda col: col in needed)
        check(role + ":validation_matrix", len(samples) == len(val) * (1950 if universal else 150)
              and not samples.duplicated(["validation_episode", *KEYS]).any())
        for episode, sample in samples.groupby("validation_episode"):
            row = val[val.episode == episode].iloc[0]
            successful = sample[safe(sample)]
            completion = sample.groupby("preference_key").task_succeeded.mean().min()
            q = pd.Series(quality(successful, mode), index=successful.index).groupby(successful.preference_key).mean().mean()
            check(f"{role}:val_aggregate:{episode}", abs(completion - row.completion_rate) < 1e-8
                  and abs(q - row.preference_balanced_quality_score) < 1e-8)
            validation_rows.append({"role": role, "episode": episode, "overall_completion": sample.task_succeeded.mean(),
                "minimum_preference_completion": completion, "own_mode_quality": q,
                "common_raw_quality": pd.Series(quality(successful, "raw"), index=successful.index).groupby(successful.preference_key).mean().mean(),
                "checkpoint_event": row.checkpoint_event, "learning_rate": row.learning_rate,
                **{metric: successful[metric].mean() for metric in OBJECTIVES}})
        # Fixed request sets make late endpoint curves comparable despite failures.
        for j, objective in enumerate(OBJECTIVES):
            endpoint = samples[(samples[PREFS[j]] == 1) & (samples.validation_episode >= val.episode.iloc[-10])].copy()
            if endpoint.empty:
                continue
            counts = endpoint.groupby(KEYS).task_succeeded.agg(["count", "sum"])
            fixed = counts[(counts["count"] == 10) & (counts["sum"] == 10)].index
            chosen = endpoint.set_index(KEYS).loc[fixed].reset_index()
            curve = chosen.groupby("validation_episode")[objective].mean()
            late = curve.iloc[-10:]
            stability.append({"role": role, "objective": objective, "fixed_requests": len(fixed),
                "late_cv": late.std(ddof=0) / late.mean(),
                "last5_vs_previous5_percent": (late.iloc[-5:].mean() / late.iloc[:5].mean() - 1) * 100,
                "last_value": curve.iloc[-1], "minimum_value": curve.min(), "minimum_episode": curve.idxmin()})
        validations[role], data[role] = samples, frame
        frame["quality_common_raw"] = quality(frame, "raw")
        frame["quality_common_excess"] = quality(frame, "excess")
        successful = frame[ok]
        requests = frame.assign(safe=ok).groupby(KEYS[:2]).safe.agg(["sum", "count"])
        overview.append({"order": ROLES.index(role)+1, "role": role, "episodes": len(train), "best_episode": best_episode,
            "flow_mode": mode, "safe_successes": int(ok.sum()), "trajectories": len(frame), "completion_rate": ok.mean(),
            "worst_preference_completion": frame.groupby("preference_key").task_succeeded.mean().min(),
            "requests_with_success": int((requests["sum"] > 0).sum()), "request_count": len(requests),
            "all_repeats_success_requests": int((requests["sum"] == requests["count"]).sum()),
            "own_quality": successful.groupby("preference_key").preference_quality_score.mean().mean(),
            "common_raw_quality": successful.groupby("preference_key").quality_common_raw.mean().mean(),
            "common_excess_quality": successful.groupby("preference_key").quality_common_excess.mean().mean(),
            "training_completion": train.task_succeeded.mean(), "last_half_training_completion": train.iloc[len(train)//2:].task_succeeded.mean(),
            "full_run_hours": summary["elapsed_seconds"]/3600, "state_dict_tensor_count": parameter_count,
            **{metric: successful[metric].mean() for metric in [*OBJECTIVES, "flow_excess_objective"]}})
        for j, objective in enumerate(OBJECTIVES):
            endpoint = frame[frame[PREFS[j]] == 1]
            if endpoint.empty:
                continue
            good = endpoint[safe(endpoint)]
            endpoint_rows.append({"role": role, "objective": objective, "success": len(good), "total": len(endpoint),
                "objective_mean": good[objective].mean(), "best_of3_mean": good.groupby("instance_id")[objective].min().mean(),
                "instances_with_success": good.instance_id.nunique()})
        for row in frame[~ok].to_dict("records"):
            failures.append({"role": role, **{key: row[key] for key in [*KEYS, *PREFS, "termination_reason", "operation_progress", "reconfiguration_cost", "worker_load_variance", "completed_reconfigurations", "wait_action_count"]}})
        sampling = updates.sampling_wall_time_seconds.sum()
        update_time = updates.ppo_update_time_seconds.sum()
        val_time = val.validation_wall_time_seconds.sum()
        diag_time = diag.diagnostic_wall_time_seconds.sum() if "diagnostic_wall_time_seconds" in diag else diag.validation_wall_time_seconds.sum()
        final_time = summary["final_sampled"]["wall_time_seconds"]
        runtime.append({"role": role, "sampling_hours": sampling/3600, "ppo_hours": update_time/3600,
            "selection_validation_hours": val_time/3600, "diagnostic_validation_hours": diag_time/3600,
            "final_test_hours": final_time/3600, "evaluation_fraction": (val_time+diag_time+final_time)/summary["elapsed_seconds"],
            "unallocated_hours": (summary["elapsed_seconds"]-sampling-update_time-val_time-diag_time-final_time)/3600,
            "recorded_inference_ms_per_decision": 1000*frame.inference_time_seconds.sum()/frame.decision_count.sum()})
        numerical = ["policy_loss", "value_loss", "loss", "approx_kl", "clip_fraction", "gradient_norm", "gradient_norm_max", "pre_update_explained_variance"]
        check(role + ":ppo_finite", np.isfinite(updates[numerical].to_numpy()).all())
        ppo.append({"role": role, "kl_mean": updates.approx_kl.mean(), "kl_max": updates.approx_kl.max(),
            "clip_fraction_mean": updates.clip_fraction.mean(), "gradient_clipped_fraction_mean": updates.gradient_clipped_fraction.mean(),
            "gradient_norm_max": updates.gradient_norm_max.max(), "explained_variance_mean": updates.pre_update_explained_variance.mean(),
            "last_learning_rate": updates.learning_rate.iloc[-1]})

    check("same_source_state", len({summaries[r]["provenance"]["source_state_sha256"] for r in ROLES}) == 1)
    train_plan_columns = ["episode", "instance_id", "seed", "pressure_type", "cost_profile", "severity",
                          "generator_config_sha256", "environment_config_sha256", "distribution_contract_sha256"]
    train_plans = {role: csv(RUNS/(PREFIX+role)/"train_log.csv", usecols=train_plan_columns) for role in ROLES}
    check("matched_single_objective_training_instances", all(train_plans[role].equals(train_plans["main_variance"])
          for role in ROLES if not role.endswith("universal")))
    check("matched_universal_training_plan", train_plans["main_universal"].equals(train_plans["shared_head_universal"]))
    controls = ["seed", "pressure_type", "cost_profile", "severity", "algorithm_seed", *PREFS,
                "derived_sampling_seed", "sampling_seed", "dataset_manifest_sha256", "environment_config_sha256"]
    comparisons = [("flow", "shared_head_flow"), ("main_variance", "shared_head_variance"), ("main_universal", "shared_head_universal")]
    paired = []
    for left, right in comparisons:
        a, b = data[left].set_index(KEYS), data[right].set_index(KEYS)
        check(left+":"+right+":matched_cells", a.index.equals(b.index) and a[controls].equals(b[controls]))
        common = safe(a) & safe(b)
        record = {"left": left, "right": right, "common_success": int(common.sum()),
            "right_gained_success": int((~safe(a) & safe(b)).sum()), "right_lost_success": int((safe(a) & ~safe(b)).sum())}
        for metric in [*OBJECTIVES, "quality_common_raw", "quality_common_excess"]:
            av, bv = a.loc[common, metric], b.loc[common, metric]
            record.update({metric+"_left": av.mean(), metric+"_right": bv.mean(),
                metric+"_right_percent_change": 100*(bv.mean()/av.mean()-1), metric+"_right_win_fraction": (bv < av-1e-9).mean()})
            if metric.startswith("quality_"):
                aa, bb = av.groupby(level="preference_key").mean().mean(), bv.groupby(level="preference_key").mean().mean()
                record.update({metric+"_preference_balanced_left": aa, metric+"_preference_balanced_right": bb,
                    metric+"_preference_balanced_right_percent_change": 100*(bb/aa-1)})
        paired.append(record)

    hv_rows, request_rows, response_rows, pressure_rows = [], [], [], []
    for role in ("main_universal", "shared_head_universal"):
        frame = data[role]
        for iid, group in frame.groupby("instance_id"):
            good = group[safe(group)]
            for mode in ("raw", "excess"):
                fields = OBJECTIVES if mode == "raw" else ["flow_excess_objective", *OBJECTIVES[1:]]
                points = good[fields].to_numpy()
                normalized = points / (points+SCALES[mode])
                repeat_hvs = []
                for _, repeat in good.groupby("sampling_repeat"):
                    p = repeat[fields].to_numpy();repeat_hvs.append(hypervolume_3d(p/(p+SCALES[mode])))
                hv_rows.append({"role": role, "instance_id": iid, "mode": mode, "pooled_hv": hypervolume_3d(normalized),
                    "repeat_hv_mean": np.mean(repeat_hvs), "front_points": len(front(points)), "safe_successes": len(good)})
                for key, request in group.groupby("preference_key"):
                    good_request = request[safe(request)]
                    request_rows.append({"role": role, "instance_id": iid, "preference_key": key, "mode": mode,
                        "successful_repeats": len(good_request), "quality_bestof3": good_request["quality_common_"+mode].min() if len(good_request) else np.nan})
            for j, objective in enumerate(OBJECTIVES):
                response_rows.append({"role": role, "instance_id": iid, "objective": objective,
                    "spearman_weight_objective": spearmanr(good[PREFS[j]], good[objective]).statistic})
        for pressure, group in frame.groupby("pressure_type"):
            pressure_rows.append({"role": role, "pressure_type": pressure, "success": int(safe(group).sum()),
                "total": len(group), "completion_rate": safe(group).mean()})

    # Pure endpoints in the late Universal validations explain early model selection.
    late_endpoints = []
    for role in ("main_universal", "shared_head_universal"):
        samples = validations[role]
        for episode, group in samples.groupby("validation_episode"):
            for j, objective in enumerate(OBJECTIVES):
                request = group[group[PREFS[j]] == 1];good = request[safe(request)]
                late_endpoints.append({"role": role, "episode": episode, "objective": objective,
                    "success": len(good), "total": len(request), "completion_rate": len(good)/len(request),
                    "objective_mean": good[objective].mean()})

    for name, rows in (("overview", overview), ("paired_comparisons", paired), ("endpoints", endpoint_rows),
            ("runtime", runtime), ("ppo", ppo), ("validation_curves", validation_rows), ("failures", failures),
            ("late_stability", stability), ("hypervolume_by_instance", hv_rows), ("requests_bestof3", request_rows),
            ("preference_response", response_rows), ("pressure_groups", pressure_rows), ("validation_endpoints", late_endpoints)):
        save(name, rows)

    universal_summary = []
    hvs, requests, responses = pd.DataFrame(hv_rows), pd.DataFrame(request_rows), pd.DataFrame(response_rows)
    for role in ("main_universal", "shared_head_universal"):
        for mode in ("raw", "excess"):
            h = hvs[(hvs.role == role) & (hvs["mode"] == mode)]
            rq = requests[(requests.role == role) & (requests["mode"] == mode)]
            universal_summary.append({"role": role, "mode": mode, "pooled_hv": h.pooled_hv.mean(),
                "repeat_hv": h.repeat_hv_mean.mean(), "bestof3_quality": rq.quality_bestof3.mean(),
                "requests_with_success": int((rq.successful_repeats > 0).sum()), "total_requests": len(rq)})
    save("universal_comparison", universal_summary)
    # The completed raw Flow run preceding this batch is a protocol-level reference.
    # Its source state differs; do not interpret this as a pure normalization ablation.
    raw_directory = RUNS / "flow_seed11_20261008_130244"
    raw_summary = js(raw_directory/"summary.json")
    raw = csv(raw_directory/"final_sampled_instance_metrics.csv").set_index(KEYS).sort_index()
    excess = data["flow"].set_index(KEYS)
    check("raw_flow_reference:matched_cells", raw.index.equals(excess.index) and raw[controls].equals(excess[controls]))
    check("raw_flow_reference:checkpoint_hash", hashlib.sha256((raw_directory/"best_checkpoint.pt").read_bytes()).hexdigest() == raw_summary["final_sampled"]["provenance"]["checkpoint_sha256"])
    SOURCES.add(raw_directory/"best_checkpoint.pt")
    both = safe(raw) & safe(excess)
    raw_reference = {"raw_success": int(safe(raw).sum()), "excess_success": int(safe(excess).sum()),
        "trajectories": len(raw), "common_success": int(both.sum()), "raw_source_state": raw_summary["provenance"]["source_state_sha256"]}
    for field in OBJECTIVES:
        a, b = raw.loc[both,field].mean(), excess.loc[both,field].mean()
        raw_reference.update({field+"_raw": a, field+"_excess": b, field+"_excess_percent_change": 100*(b/a-1)})
    save("raw_flow_reference", [raw_reference])
    save("audit_checks", CHECKS)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}
    result = {"batch_prefix": PREFIX, "audit_count": len(CHECKS), "all_checks_passed": True,
        "overview": overview, "paired_comparisons": paired, "universal": universal_summary,
        "preference_response_medians": responses.groupby(["role", "objective"]).spearman_weight_objective.median().reset_index().to_dict("records"),
        "raw_flow_reference": raw_reference,
        "source_hashes": source_hashes, "limitations": ["One training seed (11); 20 ID test instances; no OOD/stress evaluation.",
            "Main Universal is raw/v3, shared Universal is excess/v4: architecture and Flow protocol change together.",
            "All seven runs record the same dirty source state; Git commit alone cannot reproduce its uncommitted patches.",
            "Historical Universal completion_rate denotes the minimum over preferences; this report separately computes pooled completion."]}
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(validation_rows, late_endpoints, hvs)
    print(json.dumps({k:result[k] for k in ("audit_count", "all_checks_passed", "overview", "paired_comparisons", "universal", "preference_response_medians")}, ensure_ascii=False, indent=2))


def plot(validation_rows, endpoint_rows, hvs):
    curves = pd.DataFrame(validation_rows)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for role, label, color in (("main_universal", "Experts raw", "#0072B2"), ("shared_head_universal", "Shared excess", "#D55E00")):
        sub = curves[curves.role == role]
        axes[0].plot(sub.episode, 100*sub.minimum_preference_completion, label=label+" worst preference", color=color)
        axes[0].plot(sub.episode, 100*sub.overall_completion, linestyle=":", color=color, label=label+" overall")
        axes[1].plot(sub.episode, sub.common_raw_quality, color=color, label=label)
    axes[0].set_ylabel("Validation completion (%)");axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("Common raw preference loss");axes[1].set_xlabel("Training episodes")
    axes[1].set_title("Quality uses successful samples; assess it together with completion", fontsize=9)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT / "universal_validation.png", dpi=160);plt.close(fig)
    endpoints = pd.DataFrame(endpoint_rows)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for j, objective in enumerate(OBJECTIVES):
        for role, label, color in (("main_universal", "Experts raw", "#0072B2"), ("shared_head_universal", "Shared excess", "#D55E00")):
            sub = endpoints[(endpoints.role == role) & (endpoints.objective == objective)]
            axes[0,j].plot(sub.episode, 100*sub.completion_rate, label=label, color=color)
            axes[1,j].plot(sub.episode, sub.objective_mean, color=color)
        axes[0,j].set_title(objective.replace("_", " "), fontsize=10)
        axes[1,j].set_xlabel("Training episodes")
    axes[0,0].set_ylabel("Endpoint completion (%)");axes[1,0].set_ylabel("Mean among successes")
    axes[0,0].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT / "universal_endpoints.png", dpi=160);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, mode in enumerate(("raw", "excess")):
        sub = hvs[hvs["mode"] == mode].pivot(index="instance_id", columns="role", values="pooled_hv")
        axes[j].scatter(sub.main_universal, sub.shared_head_universal, color="#0072B2")
        lim = max(sub.max())*1.05;axes[j].plot([0,lim],[0,lim],color="grey",linestyle="--")
        axes[j].set_xlabel("Experts raw model");axes[j].set_ylabel("Shared excess model")
        axes[j].set_title("Pooled HV in common " + mode + " coordinates")
        axes[j].grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT / "universal_hypervolume.png", dpi=160);plt.close(fig)


if __name__ == "__main__":
    main()
