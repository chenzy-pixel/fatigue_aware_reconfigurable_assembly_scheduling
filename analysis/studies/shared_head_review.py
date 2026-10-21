"""Source-backed review of the available historical shared-head experiment.

Run with python -m analysis.studies.shared_head_review. Uses archived CSVs and saved
checkpoint tensors; does not instantiate or replay the current environment.
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

from analysis.pareto_analysis import (
    dominates, hypervolume_3d, nondominated_indices, vectors_equal,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "result/analysis/shared_head_comprehensive_20261009"
RUNS = ROOT / "result/runs"
ROLES = ("universal", "no_graph", "shared_head")
TRAIN_NAMES = {
    "universal": "v8_universal_seed11_20261005_004756_510",
    "no_graph": "ablation_structural_no_graph_seed11_20261006_025649_012965",
    "shared_head": "ablation_structural_shared_head_seed11_20261006_025649_012965",
}
FIELDS = ("flow_time_objective", "reconfiguration_cost", "worker_load_variance")
WEIGHTS = ("preference_flow", "preference_cost", "preference_variance")
SCALES = np.array([1089.15, 353.27, 2.2629])
KEYS = ["instance_id", "preference_key", "sampling_repeat"]
COLORS = {"universal": "#D55E00", "no_graph": "#009E73", "shared_head": "#0072B2"}
LABELS = {"universal": "V8 experts", "no_graph": "Node MLP + experts", "shared_head": "HGNN + shared head"}
SOURCES: set[Path] = {Path(__file__)}
CHECKS: list[dict] = []


def require(name: str, condition: bool) -> None:
    CHECKS.append({"check": name, "passed": bool(condition)})
    if not condition:
        raise ValueError(name)


def read_json(path: Path) -> dict:
    SOURCES.add(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> pd.DataFrame:
    SOURCES.add(path)
    return pd.read_csv(path).copy()


def write(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def quality(objectives: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = objectives / (SCALES + objectives)
    weighted = normalized * weights
    return (weighted.max(axis=-1) + .05 * weighted.sum(axis=-1)) / 1.05


def unique_front(points: np.ndarray) -> list[tuple]:
    result: list[tuple] = []
    for index in nondominated_indices(points):
        point = tuple(points[index])
        if not any(vectors_equal(point, old) for old in result):
            result.append(point)
    return result


def valid(frame: pd.DataFrame) -> pd.Series:
    return frame.task_succeeded & frame.physical_safety_pass & (frame.schedule_violation_count == 0)


def load_results() -> tuple[dict, dict, dict, dict]:
    trajectories, metrics, training, configs = {}, {}, {}, {}
    prior = read_json(ROOT / "result/analysis/preference_pareto_20261008/summary.json")
    for role in ROLES:
        directory = RUNS / f"ablation_eval_{role}_seed11_20261007_232505_102441"
        for name in ("config.json", "metrics.json", "instance_metrics.csv"):
            path = directory / name
            require(f"prior_source_hash:{role}:{name}", hashlib.sha256(path.read_bytes()).hexdigest()
                    == prior["source_hashes"][str(path.relative_to(ROOT))])
        frame = read_csv(directory / "instance_metrics.csv")
        frame = frame.set_index(KEYS).sort_index().copy()
        require(f"test_matrix:{role}", len(frame) == 3960 and frame.index.is_unique
                and frame.index.get_level_values("instance_id").nunique() == 20
                and (frame.groupby(level=["instance_id", "preference_key"]).size() == 3).all())
        require(f"physical_safety:{role}", frame.physical_safety_pass.all()
                and frame.schedule_violation_count.sum() == 0)
        require(f"historical_failure_flags:{role}", (frame.task_failed == ~frame.task_succeeded).all()
                and (frame.loc[frame.task_failed, "termination_reason"] == "horizon").all())
        success = frame[valid(frame)]
        require(f"quality_reconstruction:{role}", np.allclose(
            quality(success[list(FIELDS)].to_numpy(), success[list(WEIGHTS)].to_numpy()),
            success.preference_quality_score, atol=1e-12))
        trajectories[role] = frame
        metrics[role] = read_json(directory / "metrics.json")
        path = RUNS / TRAIN_NAMES[role]
        configs[role] = read_json(path / "config.json")
        training[role] = {name: read_csv(path / f"{name}_log.csv") for name in ("train", "update", "validation", "diagnostic")}
        training[role]["summary"] = read_json(path / "summary.json")
    controls = ["seed", "pressure_type", "cost_profile", "severity", "algorithm_seed", *WEIGHTS,
                "derived_sampling_seed", "sampling_seed", "normalization_manifest_sha256",
                "dataset_manifest_sha256", "subset_sha256", "reward_version", "policy_execution_version"]
    for role in ROLES[1:]:
        pd.testing.assert_frame_equal(trajectories[ROLES[0]][controls], trajectories[role][controls])
        require(f"matched_test_cells:{role}", True)
        plan = ["episode", "instance_id", "seed", "pressure_type", "cost_profile", "severity", "preference_key"]
        pd.testing.assert_frame_equal(training[ROLES[0]]["train"][plan], training[role]["train"][plan])
        require(f"matched_training_plan:{role}", True)
    return trajectories, metrics, training, configs


def analyze_test(data: dict, metrics: dict) -> tuple[pd.DataFrame, dict]:
    overview, endpoint_rows, pressure_rows, paired_rows, request_rows = [], [], [], [], []
    fronts, hv_rows = {}, []
    for role, frame in data.items():
        ok = frame[valid(frame)]
        prefs = frame.groupby(level="preference_key").task_succeeded.mean()
        units = frame.groupby(level=["instance_id", "preference_key"]).task_succeeded.sum()
        balanced = ok.groupby(level="preference_key").preference_quality_score.mean().mean()
        require(f"balanced_quality:{role}", np.isclose(balanced, metrics[role]["preference_balanced_quality_score"]))
        row = {"model": role, "requested": len(frame), "successful": len(ok),
               "completion_pct": 100 * len(ok) / len(frame), "worst_preference_pct": 100 * prefs.min(),
               "fully_successful_instances": int((frame.groupby(level="instance_id").task_succeeded.sum() == 198).sum()),
               "preference_quality": balanced, "request_units": len(units),
               "units_with_any_success": int((units > 0).sum()), "units_with_three_successes": int((units == 3).sum()),
               **{f"success_mean_{field}": ok[field].mean() for field in FIELDS}}
        for objective, weight in zip(("flow", "cost", "variance"), WEIGHTS):
            group = frame[np.isclose(frame[weight], 1)]
            success = group[valid(group)]
            endpoint_rows.append({"model": role, "objective": objective, "requested": len(group),
                                  "successful": len(success), **{field: success[field].mean() for field in FIELDS}})
        for pressure, group in frame.groupby("pressure_type"):
            pressure_rows.append({"model": role, "pressure_type": pressure,
                                  "n": len(group), "successful": int(group.task_succeeded.sum()),
                                  "completion_pct": 100 * group.task_succeeded.mean()})
        for iid, group in frame.groupby(level="instance_id"):
            success = group[valid(group)]
            raw = success[list(FIELDS)].to_numpy()
            normalized = raw / (raw + SCALES)
            front = unique_front(normalized)
            fronts[(role, iid)] = front
            repeat_hv = [hypervolume_3d(normalized[success.index.get_level_values("sampling_repeat") == repeat]) for repeat in range(3)]
            hv_rows.append({"model": role, "instance_id": iid, "pooled_hv": hypervolume_3d(front),
                            "mean_repeat_hv": np.mean(repeat_hv), "front_size": len(front)})
            # A descriptive oracle over 198 existing multi-preference candidates;
            # it uses a different budget from best-of-three for one preference.
            for pk, cells in group.groupby(level="preference_key"):
                own = cells[valid(cells)]
                w = cells[list(WEIGHTS)].iloc[0].to_numpy(dtype=float)
                losses = quality(raw, w)
                oracle_index = int(losses.argmin())
                best_index = own.preference_quality_score.idxmin() if len(own) else None
                best = own.loc[best_index] if best_index is not None else None
                request_rows.append({"model": role, "instance_id": iid, "preference_key": pk,
                                     **dict(zip(WEIGHTS, w)), "successes_of_three": len(own),
                                     "mean_success_quality": own.preference_quality_score.mean() if len(own) else np.nan,
                                     "best3_quality": best.preference_quality_score if best is not None else np.nan,
                                     "pool_oracle_quality": losses[oracle_index],
                                     "best3_to_pool_gap": best.preference_quality_score - losses[oracle_index] if best is not None else np.nan,
                                     "pool_source_preference": success.iloc[oracle_index].preference,
                                     **{f"best3_{field}": best[field] if best is not None else np.nan for field in FIELDS},
                                     **{f"pool_{field}": raw[oracle_index, i] for i, field in enumerate(FIELDS)}})
        model_hv = pd.DataFrame(hv_rows).query("model == @role")
        row.update(mean_pooled_hv=model_hv.pooled_hv.mean(), mean_repeat_hv=model_hv.mean_repeat_hv.mean(),
                   mean_front_size=model_hv.front_size.mean())
        overview.append(row)
    hv = pd.DataFrame(hv_rows)
    for baseline in ROLES[:2]:
        first, second = data[baseline], data["shared_head"]
        both = valid(first) & valid(second)
        delta = (second.loc[both, "preference_quality_score"] - first.loc[both, "preference_quality_score"])
        per_pref = delta.groupby(level="preference_key").mean()
        per_instance = delta.groupby(level="instance_id").mean()
        paired_rows.append({"baseline": baseline, "variant": "shared_head", "common_success": int(both.sum()),
                            "new_success": int((~valid(first) & valid(second)).sum()),
                            "lost_success": int((valid(first) & ~valid(second)).sum()),
                            "matched_quality_delta": per_pref.mean(),
                            "quality_preference_wins": int((per_pref < 0).sum()),
                            "quality_instance_wins": int((per_instance < 0).sum()),
                            **{f"shared_minus_baseline_{field}": (second.loc[both, field] - first.loc[both, field]).mean() for field in FIELDS}})
    pair_hv = []
    for baseline in ROLES[:2]:
        for iid in data[baseline].index.get_level_values("instance_id").unique():
            a, b = fronts[(baseline, iid)], fronts[("shared_head", iid)]
            pair_hv.append({"baseline": baseline, "instance_id": iid,
                            "hv_delta_shared_minus_baseline": hypervolume_3d(b) - hypervolume_3d(a),
                            "baseline_covers_shared": np.mean([any(dominates(x, y) or vectors_equal(x, y) for x in a) for y in b]),
                            "shared_covers_baseline": np.mean([any(dominates(y, x) or vectors_equal(y, x) for y in b) for x in a])})
    write("test_overview.csv", pd.DataFrame(overview))
    write("endpoint_test.csv", pd.DataFrame(endpoint_rows))
    write("pressure_completion.csv", pd.DataFrame(pressure_rows))
    write("matched_comparisons.csv", pd.DataFrame(paired_rows))
    write("pareto_instances.csv", hv)
    write("pareto_paired_instances.csv", pd.DataFrame(pair_hv))
    requests = pd.DataFrame(request_rows)
    write("best3_and_pool_oracle.csv", requests)
    write("pool_oracle_summary.csv", requests.groupby("model")[["mean_success_quality", "best3_quality", "pool_oracle_quality", "best3_to_pool_gap"]].mean().reset_index())
    for baseline in ROLES[:2]:
        a = requests[requests.model == baseline].set_index(["instance_id", "preference_key"])
        b = requests[requests.model == "shared_head"].set_index(["instance_id", "preference_key"])
        common = a.best3_quality.notna() & b.best3_quality.notna()
        write(f"best3_paired_{baseline}.csv", pd.DataFrame({
            "baseline": [baseline], "common_units": [int(common.sum())],
            "baseline_quality": [a.loc[common, "best3_quality"].mean()],
            "shared_quality": [b.loc[common, "best3_quality"].mean()],
            "shared_unit_win_fraction": [(b.loc[common, "best3_quality"] < a.loc[common, "best3_quality"]).mean()]}))
    # Check new calculations against prior analyses, without modifying them.
    old = read_csv(ROOT / "result/analysis/shared_head_decision_20261008/decision_overview.csv").set_index("model")
    overview_frame = pd.DataFrame(overview).set_index("model")
    for role in ROLES:
        require(f"prior_overview:{role}", np.allclose(
            overview_frame.loc[role, ["completion_pct", "preference_quality", "mean_pooled_hv"]].astype(float),
            old.loc[role, ["completion_pct", "preference_balanced_quality", "mean_pooled_hv"]].astype(float)))
    return overview_frame, fronts


def analyze_preferences(data: dict) -> pd.DataFrame:
    common = np.logical_and.reduce([valid(data[role]).to_numpy() for role in ROLES])
    common_data = {role: frame.loc[common].copy() for role, frame in data.items()}
    tradeoff_rows, slice_rows, correlation_rows = [], [], []
    for role, frame in common_data.items():
        w = frame.preference_variance
        bands = np.select([w == 0, (w > 0) & (w <= .2), (w > .2) & (w <= .5)],
                          ["variance_0", "variance_0_to_0.2", "variance_0.2_to_0.5"], default="variance_gt_0.5")
        frame = frame.assign(band=bands)
        for band, cells in frame.groupby("band"):
            tradeoff_rows.append({"model": role, "band": band, "n": len(cells),
                                 **{field: cells[field].mean() for field in FIELDS}})
        for iid, group in frame.groupby(level="instance_id"):
            pref_means = group.groupby(level="preference_key")[[*WEIGHTS, *FIELDS]].mean()
            for field, weight, objective in zip(FIELDS, WEIGHTS, ("flow", "cost", "variance")):
                correlation_rows.append({"model": role, "instance_id": iid, "objective": objective,
                                         "spearman": spearmanr(pref_means[weight], pref_means[field]).statistic})
    for objective, field, weight in zip(("flow", "cost", "variance"), FIELDS, WEIGHTS):
        cohorts = []
        for role in ROLES:
            by_weight = []
            for own_weight in np.arange(0, 1.01, .2):
                frame = data[role]
                other = [name for name in WEIGHTS if name != weight]
                mask = np.isclose(frame[weight], own_weight)
                for name in other:
                    mask &= np.isclose(frame[name], (1 - own_weight) / 2)
                selected = frame[mask & valid(frame)]
                by_weight.append(set(zip(selected.index.get_level_values("instance_id"), selected.index.get_level_values("sampling_repeat"))))
            cohorts += by_weight
        fixed = set.intersection(*cohorts)
        for role in ROLES:
            for own_weight in np.arange(0, 1.01, .2):
                frame = data[role]
                other = [name for name in WEIGHTS if name != weight]
                mask = np.isclose(frame[weight], own_weight)
                for name in other:
                    mask &= np.isclose(frame[name], (1 - own_weight) / 2)
                selected = frame[mask].reset_index()
                selected = selected[[tuple(row) in fixed for row in selected[["instance_id", "sampling_repeat"]].to_numpy()]]
                require(f"fixed_response_cohort:{objective}:{role}:{own_weight:.1f}", len(selected) == len(fixed))
                means = selected.groupby("instance_id")[list(FIELDS)].mean().mean()
                slice_rows.append({"model": role, "objective": objective, "own_weight": own_weight,
                                   "instances": selected.instance_id.nunique(), "repeat_cells": len(selected),
                                   "target_mean": means[field], **dict(means)})
    slices = pd.DataFrame(slice_rows)
    write("preference_slices.csv", slices)
    write("variance_weight_tradeoffs.csv", pd.DataFrame(tradeoff_rows))
    write("preference_instance_correlations.csv", pd.DataFrame(correlation_rows))
    old = read_csv(ROOT / "result/analysis/preference_pareto_20261008/preference_slices.csv")
    joined = slices.merge(old, on=["model", "objective", "own_weight"], suffixes=("", "_old"))
    # Historical own_weight floats are serialized rounded; match rounded keys.
    if len(joined) != len(slices):
        left, right = slices.copy(), old.copy()
        left.own_weight, right.own_weight = left.own_weight.round(8), right.own_weight.round(8)
        joined = left.merge(right, on=["model", "objective", "own_weight"], suffixes=("", "_old"))
    require("prior_preference_response", len(joined) == 54 and np.allclose(joined.target_mean, joined.target_mean_old))
    return slices


def analyze_training(training: dict, configs: dict, data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    costs, optimizer_rows, validation_rows, component_rows, stability_rows, fixed_curves = [], [], [], [], [], []
    for role, logs in training.items():
        train, updates, val, diag, summary = (logs[k] for k in ("train", "update", "validation", "diagnostic", "summary"))
        require(f"reward_identity:{role}", train.reward_identity_error.abs().max() < 1e-10)
        path = RUNS / TRAIN_NAMES[role] / "best_checkpoint.pt"
        SOURCES.add(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        require(f"archived_schema:{role}", checkpoint["network_spec"]["observation_schema_version"] == 6)
        require(f"checkpoint_role:{role}", checkpoint["network_spec"]["actor_head_variant"]
                == configs[role]["network"].get("actor_head_variant", "objective_experts"))
        counts = {}
        for name, tensor in checkpoint["network"].items():
            if name.endswith(".directions"):
                continue
            if "_experts." in name or "_shared." in name:
                component = "base_actor_heads"
            elif "_residual" in name:
                component = "residual_heads"
            elif name.startswith("critic."):
                component = "critic"
            elif name.startswith(("preference_encoder.", "action_preference_projector.")):
                component = "preference_path"
            else:
                component = "graph_and_candidate_features"
            counts[component] = counts.get(component, 0) + tensor.numel()
        component_rows.extend({"model": role, "component": component, "parameters": count} for component, count in counts.items())
        elapsed = summary["elapsed_seconds"]
        sampling = updates.sampling_wall_time_seconds.sum()
        ppo = updates.ppo_update_time_seconds.sum()
        val_time, diag_time = val.validation_wall_time_seconds.sum(), diag.validation_wall_time_seconds.sum()
        metrics = read_json(RUNS / f"ablation_eval_{role}_seed11_20261007_232505_102441/metrics.json")
        require(f"checkpoint_source:{role}", hashlib.sha256(path.read_bytes()).hexdigest()
                == metrics["provenance"]["checkpoint_sha256"])
        costs.append({"model": role, "parameters": sum(counts.values()), "full_run_hours": elapsed / 3600,
                      "sampling_hours": sampling / 3600, "ppo_hours": ppo / 3600,
                      "selection_validation_hours": val_time / 3600, "diagnostic_hours": diag_time / 3600,
                      "remaining_including_final_test_hours": (elapsed - sampling - ppo - val_time - diag_time) / 3600,
                      "all_validation_pct": 100 * (val_time + diag_time) / elapsed,
                      "train_sampling_ppo_pct": 100 * (sampling + ppo) / elapsed,
                      "train_decisions": int(train.step_count.sum()),
                      "test_decisions": metrics["decision_count"],
                      "recorded_inference_ms_per_decision": 1000 * metrics["total_inference_time_seconds"] / metrics["decision_count"],
                      "recorded_inference_seconds": metrics["total_inference_time_seconds"],
                      "best_episode": summary["checkpoint_selection"]["best_episode"],
                      "training_completion_pct": 100 * train.task_succeeded.mean(),
                      "late_training_completion_pct": 100 * train.loc[train.episode > 1000, "task_succeeded"].mean()})
        for field in ("approx_kl", "clip_fraction", "gradient_norm", "gradient_norm_max",
                      "gradient_clipped_fraction", "entropy", "pre_update_explained_variance", "value_loss"):
            require(f"finite_optimizer:{role}:{field}", np.isfinite(updates[field]).all())
            optimizer_rows.append({"model": role, "field": field, "mean": updates[field].mean(),
                                   "min": updates[field].min(), "max": updates[field].max(),
                                   "late_mean": updates.loc[updates.episode_end > 1000, field].mean()})
        rows = read_csv(RUNS / TRAIN_NAMES[role] / "sampled_validation_instance_metrics.csv")
        validation_rows.append(rows.groupby("validation_episode").agg(
            completion_pct=("task_succeeded", lambda x: 100 * x.mean()),
            n=("task_succeeded", "size")).reset_index().merge(
                val[["episode", "completion_rate", "preference_balanced_quality_score"]],
                left_on="validation_episode", right_on="episode").assign(model=role))
        if role != "shared_head":
            continue
        diagnostic = read_csv(RUNS / TRAIN_NAMES[role] / "sampled_diagnostic_instance_metrics.csv")
        require("diagnostic_holdout_disjoint", not set(rows.instance_id) & set(diagnostic.instance_id))
        write("shared_diagnostic_completion.csv", diagnostic.groupby("validation_episode").agg(
            n=("task_succeeded", "size"), completion_pct=("task_succeeded", lambda x: 100 * x.mean())).reset_index())
        for field, objective, weight in zip(FIELDS, ("flow", "cost", "variance"), WEIGHTS):
            endpoint = rows[np.isclose(rows[weight], 1)].set_index(["validation_episode", *KEYS]).sort_index()
            late = endpoint[endpoint.index.get_level_values("validation_episode") >= 1100]
            successes = late[late.task_succeeded].reset_index().groupby(KEYS).size()
            keys = successes[successes == 10].index
            means = []
            for episode in range(1100, 2001, 100):
                part = endpoint.xs(episode, level="validation_episode")
                fixed = part.loc[keys]
                means.append(fixed[field].mean())
                fixed_curves.append({"objective": objective, "episode": episode, "fixed_cells": len(keys),
                                     "fixed_target_mean": means[-1], "completion_pct": 100 * part.task_succeeded.mean()})
            stability_rows.append({"objective": objective, "fixed_cells": len(keys),
                                   "checkpoint_mean_cv_pct": 100 * np.std(means, ddof=1) / np.mean(means),
                                   "first5_to_last5_change_pct": 100 * (np.mean(means[5:]) / np.mean(means[:5]) - 1)})
    write("runtime_decomposition.csv", pd.DataFrame(costs))
    write("parameter_components.csv", pd.DataFrame(component_rows))
    write("ppo_statistics.csv", pd.DataFrame(optimizer_rows))
    write("validation_curves.csv", pd.concat(validation_rows, ignore_index=True))
    write("endpoint_stability.csv", pd.DataFrame(stability_rows))
    write("endpoint_fixed_curves.csv", pd.DataFrame(fixed_curves))
    return pd.DataFrame(costs), pd.concat(validation_rows, ignore_index=True)


def analyze_heuristic(shared: pd.DataFrame) -> None:
    manifest = read_json(ROOT / "data/manifests/v2/test/manifest.json")
    for entry in manifest["files"]:
        path = ROOT / "data/instances/v2/test" / entry["path"]
        require(f"test_instance_hash:{entry['seed']}",
                hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"])
        document = read_json(path)
        benchmark = document["metadata"]["heuristic_metrics"]
        if benchmark["heuristic_completed"]:
            require(f"heuristic_safe:{entry['seed']}", not benchmark["schedule_violations"]
                    and benchmark["maximum_worker_fatigue"] <= document["instance"]["fatigue"]["maximum_safe_fatigue"] + 1e-9)
    usable = shared[valid(shared) & shared.heuristic_completed & shared.heuristic_comparison_valid]
    raw = usable[["heuristic_flow_time", "heuristic_reconfiguration_cost", "heuristic_worker_load_variance"]].to_numpy()
    require("heuristic_quality_reconstruction", np.allclose(
        quality(raw, usable[list(WEIGHTS)].to_numpy()), usable.heuristic_reward_quality_score))
    rows = []
    for cohort, frame in [("all_matched_success", usable)] + [
        (f"pure_{objective}", usable[np.isclose(usable[weight], 1)])
        for objective, weight in zip(("flow", "cost", "variance"), WEIGHTS)]:
        rows.append({"cohort": cohort, "n": len(frame),
                     "instances": frame.index.get_level_values("instance_id").nunique(),
                     "shared_quality": frame.preference_quality_score.mean(),
                     "heuristic_quality": frame.heuristic_reward_quality_score.mean(),
                     "shared_quality_win_fraction": (frame.preference_quality_score < frame.heuristic_reward_quality_score).mean(),
                     **{f"shared_{field}": frame[field].mean() for field in FIELDS},
                     **{f"heuristic_{field}": frame[other].mean() for field, other in zip(FIELDS,
                         ("heuristic_flow_time", "heuristic_reconfiguration_cost", "heuristic_worker_load_variance"))}})
    write("heuristic_comparison.csv", pd.DataFrame(rows))
    selected_rows = []
    for objective, weight, field, heuristic_field in zip(
        ("flow", "cost", "variance"), WEIGHTS, FIELDS,
        ("heuristic_flow_time", "heuristic_reconfiguration_cost", "heuristic_worker_load_variance")):
        part = usable[np.isclose(usable[weight], 1)]
        means = part.groupby(level="instance_id").agg(
            shared_best=(field, "min"), heuristic=(heuristic_field, "first"))
        selected_rows.append({"objective": objective, "instances": len(means),
                              "shared_best3_target": means.shared_best.mean(),
                              "heuristic_target": means.heuristic.mean(),
                              "shared_best3_instance_wins": int((means.shared_best < means.heuristic).sum())})
    write("heuristic_best3_endpoints.csv", pd.DataFrame(selected_rows))
    hybrid_rows, hybrid_hv = [], []
    for iid, group in shared.groupby(level="instance_id"):
        success = group[valid(group)]
        raw = success[list(FIELDS)].to_numpy()
        points = raw / (raw + SCALES)
        benchmark = group.iloc[0]
        heuristic_valid = bool(benchmark.heuristic_completed and benchmark.heuristic_comparison_valid)
        if heuristic_valid:
            heuristic_raw = benchmark[["heuristic_flow_time", "heuristic_reconfiguration_cost", "heuristic_worker_load_variance"]].to_numpy(dtype=float)
            heuristic_point = heuristic_raw / (heuristic_raw + SCALES)
            extra = np.vstack([points, heuristic_point])
        else:
            extra = points
        hybrid_hv.append({"instance_id": iid, "shared_pooled_hv": hypervolume_3d(points),
                          "shared_plus_heuristic_hv": hypervolume_3d(extra)})
        for pk, cells in group.groupby(level="preference_key"):
            own = cells[valid(cells)]
            best = own.preference_quality_score.min()
            heuristic_q = cells.heuristic_reward_quality_score.iloc[0] if heuristic_valid else np.inf
            hybrid_rows.append({"instance_id": iid, "preference_key": pk,
                                "shared_best3_quality": best,
                                "heuristic_quality": heuristic_q if heuristic_valid else np.nan,
                                "hybrid_quality": min(best, heuristic_q),
                                "heuristic_selected": heuristic_q < best})
    hybrid = pd.DataFrame(hybrid_rows)
    require("hybrid_selection_does_not_increase_loss", (hybrid.hybrid_quality <= hybrid.shared_best3_quality + 1e-12).all())
    write("shared_plus_heuristic_requests.csv", hybrid)
    write("shared_plus_heuristic_hv.csv", pd.DataFrame(hybrid_hv))
    fail = shared[~shared.task_succeeded].reset_index()
    write("shared_failures.csv", fail)
    tails = shared[valid(shared)][[*FIELDS, "completed_reconfigurations", "machine_waiting_for_worker_time",
                                 "maximum_worker_fatigue", "fatigue_masked_action_ratio"]].quantile([.5, .9, .95, .99, 1])
    write("shared_success_tail_metrics.csv", tails.rename_axis("quantile").reset_index())


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=240)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def plots(overview: pd.DataFrame, slices: pd.DataFrame, costs: pd.DataFrame, curves: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for ax, field, label, ylim in zip(axes,
        ("completion_pct", "preference_quality", "mean_pooled_hv"),
        ("Completion (%)", "Preference loss (lower is better)", "Pooled hypervolume"),
        ((94, 101), (0, .46), (0, .30))):
        values = overview.loc[list(ROLES), field]
        if field == "completion_pct":
            ax.scatter(range(3), values, color=[COLORS[role] for role in ROLES], s=80)
            for index, value in enumerate(values):
                ax.annotate(f"{value:.2f}", (index, value), xytext=(0, 8),
                            textcoords="offset points", ha="center")
            ax.set_xlim(-.5, 2.5)
        else:
            bars = ax.bar(range(3), values, color=[COLORS[role] for role in ROLES])
            ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set(xticks=range(3), xticklabels=["V8", "MLP", "Shared"], ylabel=label, ylim=ylim)
    save_figure(fig, "test_overview")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for ax, objective in zip(axes, ("flow", "cost", "variance")):
        for role in ROLES:
            q = slices[(slices.model == role) & (slices.objective == objective)]
            ax.plot(q.own_weight, q.target_mean, marker="o", color=COLORS[role], label=LABELS[role])
        ax.set(xlabel=f"{objective.capitalize()} weight", ylabel=f"{objective.capitalize()} objective")
        if objective == "variance":
            ax.set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, "preference_response")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for role in ROLES:
        q = curves[curves.model == role]
        axes[0].plot(q.validation_episode, q.completion_pct, color=COLORS[role], label=LABELS[role])
        axes[1].plot(q.validation_episode, 100 * q.completion_rate, color=COLORS[role])
        axes[2].plot(q.validation_episode, q.preference_balanced_quality_score, color=COLORS[role])
    for ax, label in zip(axes, ("All validation completion (%)", "Worst-preference completion (%)", "Preference loss")):
        ax.set(xlabel="Training episode", ylabel=label)
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, "learning_curves")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    cost = costs.set_index("model").loc[list(ROLES)]
    bottom = np.zeros(3)
    parts = [("sampling_hours", "Sampling", "#56B4E9"), ("ppo_hours", "PPO update", "#009E73"),
             ("selection_validation_hours", "Selection validation", "#0072B2"),
             ("diagnostic_hours", "Diagnostic validation", "#E69F00"),
             ("remaining_including_final_test_hours", "Other, including final test", "#999999")]
    for field, label, color in parts:
        axes[0].bar(range(3), cost[field], bottom=bottom, label=label, color=color)
        bottom += cost[field].to_numpy()
    axes[0].set(xticks=range(3), xticklabels=["V8", "MLP", "Shared"], ylabel="Recorded full-run hours")
    axes[0].legend(frameon=False, fontsize=8, ncol=2,
                   loc="lower left", bbox_to_anchor=(0, 1.02))
    bars = axes[1].bar(range(3), cost.recorded_inference_ms_per_decision, color=[COLORS[role] for role in ROLES])
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set(xticks=range(3), xticklabels=["V8", "MLP", "Shared"], ylabel="Recorded inference ms/decision", ylim=(0, 5.6))
    save_figure(fig, "runtime")
    endpoints = read_csv(OUT / "endpoint_fixed_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for ax, objective in zip(axes, ("flow", "cost", "variance")):
        q = endpoints[endpoints.objective == objective]
        ax.plot(q.episode, q.fixed_target_mean, color=COLORS["shared_head"], marker="o")
        ax.set(xlabel="Training episode", ylabel=f"Pure {objective} target (fixed cohort)")
    save_figure(fig, "endpoint_stability")
    heuristic = read_csv(OUT / "heuristic_comparison.csv").set_index("cohort")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for ax, objective, field in zip(axes, ("flow", "cost", "variance"), FIELDS):
        q = heuristic.loc[f"pure_{objective}"]
        values = [q[f"heuristic_{field}"], q[f"shared_{field}"]]
        bars = ax.bar(range(2), values, color=["#999999", COLORS["shared_head"]])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set(xticks=range(2), xticklabels=["Heuristic", "Shared"],
               ylabel=f"Pure {objective} target", ylim=(0, max(values) * 1.2))
    save_figure(fig, "heuristic_targets")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, metrics, training, configs = load_results()
    overview, fronts = analyze_test(data, metrics)
    slices = analyze_preferences(data)
    costs, curves = analyze_training(training, configs, data)
    analyze_heuristic(data["shared_head"])
    plots(overview, slices, costs, curves)
    prior_audit = read_json(ROOT / "result/analysis/copied_experiments_20261008/audit.json")
    for path in sorted(SOURCES):
        key = str(path.relative_to(ROOT))
        if key in prior_audit["artifact_hashes"]:
            require(f"prior_artifact_hash:{key}", hashlib.sha256(path.read_bytes()).hexdigest()
                    == prior_audit["artifact_hashes"][key])
    write("source_hashes.csv", pd.DataFrame([
        {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(path for path in SOURCES if OUT not in path.parents)]))
    write("checks.csv", pd.DataFrame(CHECKS))
    print(overview[["completion_pct", "preference_quality", "mean_pooled_hv", "units_with_any_success"]].round(6).to_string())
    print(costs[["model", "full_run_hours", "all_validation_pct", "train_sampling_ppo_pct"]].round(3).to_string(index=False))
    print(f"Passed {len(CHECKS)} checks. Saved review to {OUT}")


if __name__ == "__main__":
    main()
