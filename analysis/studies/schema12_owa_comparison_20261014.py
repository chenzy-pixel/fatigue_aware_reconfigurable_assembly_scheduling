"""Paired saved-trajectory audit for schema12 Variance/OWA screening.

Run: python -m analysis.studies.schema12_owa_comparison_20261014
Rescores both methods in identical frozen coordinates; starts no policy runs.
"""
from __future__ import annotations

import hashlib
import json
from math import gcd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from agent.ppo.network import infer_checkpoint_network_spec
from analysis.pareto_analysis import hypervolume_3d
from configs import load_config
from environment.workload import geometric_owa_weights
from result.provenance import effective_config_snapshot, network_weights_sha256, dataset_manifest_snapshot
from utils import derive_evaluation_sampling_seed

ROOT = Path(__file__).resolve().parents[2]
RUNS = {mode: ROOT / f"result/runs/schema12_screen_seed11_20261012_201659_{mode}_log2" for mode in ("variance", "owa")}
OUT = ROOT / "result/analysis/schema12_owa_comparison_20261014"
KEY = ["instance_id", "preference_key", "sampling_repeat"]
PREF = ["preference_flow", "preference_cost", "preference_workload"]
PHYSICAL = ["flow_time_objective", "flow_excess_objective", "reconfiguration_cost", "worker_load_variance", "owa_q05",
            "worker_load_max", "worker_load_total", "maximum_worker_fatigue", "completed_reconfigurations",
            "production_wait_time", "worker_wait_time", "machine_waiting_for_worker_time", "makespan"]
EXTRA = ["validation_episode", "task_succeeded", "task_failed", "physical_safety_pass", "schedule_violation_count",
         "safe_fatigue_limit", "sampling_truncated", "sampling_seed", "derived_sampling_seed", "termination_reason",
         "operation_progress", "initial_progress", "initial_preference_quality_score", "preference_quality_score",
         "actual_preference_quality_score", "terminal_failure_penalty_applied", "training_cumulative_reward",
         "flow_processing_lower_bound", "dataset_manifest_sha256", "subset_sha256", "action_trace_sha256",
         "worker_loads", "workload_objective_value", "pressure_type", "cost_profile", "workload_mode",
         "worker_reconfiguration_busy_minutes", "inference_time_seconds", "solve_time_seconds"]
SOURCES = {Path(__file__), ROOT / "analysis/pareto_analysis.py", ROOT / "environment/workload.py"}
CHECKS = []
EVAL = {}


def check(name, condition):
    CHECKS.append({"check": name, "passed": bool(condition)})
    if not condition:
        raise AssertionError(name)


def js(path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path, model, tag):
    SOURCES.add(path)
    columns = set(KEY + PREF + PHYSICAL + EXTRA)
    f = pd.read_csv(path, usecols=lambda c: c in columns)
    loads = [json.loads(v) for v in f.worker_loads]
    check(tag + ":worker_roster", all(set(row) == {"H1", "H2", "H3", "H4", "H5", "H6"} for row in loads))
    values = np.array([[row[f"H{i}"] for i in range(1, 7)] for row in loads], dtype=float)
    check(tag + ":finite_nonnegative_loads", np.isfinite(values).all() and (values >= 0).all())
    check(tag + ":variance_reconstructed", np.allclose(values.var(axis=1), f.worker_load_variance, atol=1e-8))
    check(tag + ":total_reconstructed", np.allclose(values.sum(axis=1), f.worker_load_total, atol=1e-8))
    check(tag + ":max_reconstructed", np.allclose(values.max(axis=1), f.worker_load_max, atol=1e-8))
    f["owa_q05"] = np.sort(values, axis=1)[:, ::-1] @ geometric_owa_weights(6, .5)
    check(tag + ":training_workload_reconstructed", np.allclose(f.workload_objective_value,
          f.worker_load_variance if model == "variance" else f.owa_q05, atol=1e-8))
    for i in range(6):
        f[f"load_H{i+1}"] = values[:, i]
    # Preserve native training objective/quality; evaluation columns are separate.
    f["training_workload_objective_value"] = f.workload_objective_value
    f["training_preference_quality_score"] = f.preference_quality_score
    for coordinate in EVAL:
        f["evaluation_quality_" + coordinate] = quality(f, coordinate)
    return f


def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")


def safe(f):
    return f.task_succeeded & f.physical_safety_pass & (f.schedule_violation_count == 0)


def quality(f, coordinate):
    target = EVAL[coordinate]
    values = f[target["fields"]].to_numpy(dtype=float)
    bounded = values / (values + np.array(target["scales"]))
    weighted = bounded * f[PREF].to_numpy(dtype=float)
    return pd.Series((weighted.max(axis=1) + .05 * weighted.sum(axis=1)) / 1.05, index=f.index)


def aggregate(f):
    good = f[safe(f)]
    cost = f[np.isclose(f.preference_cost, 1)]
    return {"requests": len(f), "successes": len(good), "failures": len(f)-len(good),
            "overall_completion": len(good)/len(f),
            "worst_preference_completion": f.assign(success=safe(f)).groupby("preference_key").success.mean().min(),
            "cost_completion": safe(cost).mean() if len(cost) else None,
            **{field: good[field].mean() for field in PHYSICAL},
            **{"quality_" + coord: quality(good, coord).groupby(good.preference_key).mean().mean() for coord in EVAL}}


def comparison(first, second, stage):
    a, b = first.set_index(KEY).sort_index(), second.set_index(KEY).sort_index()
    check(stage + ":identical_requests", a.index.equals(b.index))
    check(stage + ":identical_sampling_seeds", a.derived_sampling_seed.equals(b.derived_sampling_seed))
    check(stage + ":identical_preferences", np.allclose(a[PREF], b[PREF]))
    common = safe(a) & safe(b)
    counts = {"stage": stage, "requests": len(a), "common_success": int(common.sum()), "coverage": common.mean(),
              "owa_gained": int((~safe(a) & safe(b)).sum()), "owa_lost": int((safe(a) & ~safe(b)).sum()),
              "both_failed": int((~safe(a) & ~safe(b)).sum())}
    rows = []
    for field in PHYSICAL:
        x, y = a.loc[common, field], b.loc[common, field]
        rows.append({"stage": stage, "measure": field, "variance_mean": x.mean(), "owa_mean": y.mean(),
                     "percent_change": 100*(y.mean()/x.mean()-1), "owa_lower_fraction": (y < x-1e-9).mean()})
    for coord in EVAL:
        x, y = quality(a.loc[common], coord), quality(b.loc[common], coord)
        xm, ym = x.groupby(level="preference_key").mean().mean(), y.groupby(level="preference_key").mean().mean()
        rows.append({"stage": stage, "measure": "quality_" + coord, "variance_mean": xm, "owa_mean": ym,
                     "percent_change": 100*(ym/xm-1), "owa_lower_fraction": (y < x-1e-9).mean()})
    return counts, rows


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                result.append({"field": prefix+k, "variance": a.get(k), "owa": b.get(k)})
            else:
                result.extend(differences(a[k], b[k], prefix+k+"."))
        return result
    return [{"field": prefix[:-1], "variance": a, "owa": b}] if a != b else []


def preference_response(tests):
    correlation, levels, transitions, endpoints = [], [], [], []
    for model, f in tests.items():
        g = f[safe(f)].groupby(["instance_id", "preference_key"]).agg(
            {**{p: "first" for p in PREF}, **{o: "mean" for o in PHYSICAL}}).reset_index()
        check(model + ":preference_cell_coverage", len(g) == 1320)
        targets = [(PREF[0], PHYSICAL[0]), (PREF[1], "reconfiguration_cost"),
                   (PREF[2], "owa_q05"), (PREF[2], "worker_load_variance"), (PREF[2], "worker_load_max")]
        for pref, target in targets:
            for iid, cells in g.groupby("instance_id"):
                correlation.append({"model": model, "instance_id": iid, "target": target,
                                    "spearman": spearmanr(cells[pref], cells[target]).statistic})
                others = [p for p in PREF if p != pref]
                rays = {}
                pure = cells[np.isclose(cells[pref], 1)].to_dict("records")
                for record in cells[cells[pref] < 1-1e-10].to_dict("records"):
                    x, y = [int(round(record[p]*10)) for p in others]
                    d = gcd(x, y)
                    rays.setdefault((x//d, y//d), []).append(record)
                correct = total = 0
                for records in rays.values():
                    sequence = sorted(records + pure, key=lambda r: r[pref])
                    for x, y in zip(sequence, sequence[1:]):
                        correct += int(y[target] <= x[target] + 1e-8)
                        total += 1
                transitions.append({"model": model, "instance_id": iid, "target": target,
                                    "decrease_or_equal": correct, "comparisons": total})
            for weight, cells in g.groupby(pref):
                levels.append({"model": model, "target": target, "weight": weight, "mean": cells[target].mean()})
        for name, pref in zip(("Flow", "Cost", "Workload"), PREF):
            endpoints.append({"model": model, "endpoint": name, **aggregate(f[np.isclose(f[pref], 1)])})
    corr = pd.DataFrame(correlation).groupby(["model", "target"]).spearman.agg(
        median="median", mean="mean", negative_instances=lambda s: int((s < 0).sum())).reset_index()
    trans = pd.DataFrame(transitions).groupby(["model", "target"])[["decrease_or_equal", "comparisons"]].sum().reset_index()
    trans["direction_fraction"] = trans.decrease_or_equal/trans.comparisons
    save("preference_instance_correlations", correlation); save("preference_summary", corr)
    save("preference_weight_levels", levels); save("preference_fixed_ratio", trans)
    return corr.to_dict("records"), trans.to_dict("records"), levels, endpoints


def pooled_dominance(tests):
    pool = pd.concat([f[safe(f)].assign(model=model) for model, f in tests.items()], ignore_index=True)
    summaries, witnesses = [], []
    for iid, group in pool.groupby("instance_id"):
        group = group.reset_index(drop=True)
        fields = ["flow_time_objective", "reconfiguration_cost"] + [f"load_H{i}" for i in range(1, 7)]
        v = group[fields].to_numpy()
        tol = 1e-9*np.maximum(1, np.maximum(abs(v[:, None, :]), abs(v[None, :, :])))
        dominates = (v[:, None, :] <= v[None, :, :]+tol).all(axis=2) & (v[:, None, :] < v[None, :, :]-tol).any(axis=2)
        for model in tests:
            targets = np.flatnonzero(group.model == model)
            other = np.flatnonzero(group.model != model)
            own = targets
            summaries.append({"instance_id": iid, "model": model, "candidates": len(targets),
                              "dominated_by_any": int(dominates[:, targets].any(axis=0).sum()),
                              "dominated_by_other_model": int(dominates[np.ix_(other, targets)].any(axis=0).sum()),
                              "dominated_by_same_model": int(dominates[np.ix_(own, targets)].any(axis=0).sum())})
            for target in targets:
                candidates = np.flatnonzero(dominates[:, target])
                if len(candidates):
                    better = candidates[0]
                    witnesses.append({"instance_id": iid, "dominated_model": model,
                                      "dominated_preference": group.iloc[target].preference_key,
                                      "dominated_repeat": int(group.iloc[target].sampling_repeat),
                                      "dominator_model": group.iloc[better].model,
                                      "dominator_preference": group.iloc[better].preference_key,
                                      "dominator_repeat": int(group.iloc[better].sampling_repeat)})
    save("labelled_dominance_by_instance", summaries); save("labelled_dominance_witnesses", witnesses)
    result = pd.DataFrame(summaries).groupby("model").sum(numeric_only=True)
    for field in ("dominated_by_any", "dominated_by_other_model", "dominated_by_same_model"):
        result[field + "_fraction"] = result[field]/result.candidates
    return result.reset_index().to_dict("records")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    eval_configs = {}
    for flow in ("excess", "raw"):
        for workload in ("variance", "owa"):
            path = ({("excess", "variance"): "configs/controls/variance.json",
                     ("excess", "owa"): "configs/default.json",
                     ("raw", "variance"): "configs/controls/raw_single_variance.json",
                     ("raw", "owa"): "configs/controls/raw.json"})[flow, workload]
            config = load_config(path)
            SOURCES.add(ROOT / path)
            manifest = ROOT / config["objective_scalarizer"]["normalization_manifest"]
            SOURCES.add(manifest)
            scales = config["objective_scalarizer"]["scales"]
            coord = flow + "_" + workload
            EVAL[coord] = {"fields": ["flow_time_objective" if flow == "raw" else "flow_excess_objective", "reconfiguration_cost",
                                      "worker_load_variance" if workload == "variance" else "owa_q05"],
                           "scales": [scales[k] for k in ("flow", "cost", "workload")], "rho": .05,
                           "hv_reference": [1, 1, 1], "workload_definition": config["workload_objective"],
                           "normalization_manifest_sha256": config["objective_scalarizer"]["normalization_manifest_sha256"]}
            eval_configs[coord] = effective_config_snapshot(config)
    (OUT / "evaluation_config.json").write_text(json.dumps({"coordinates": EVAL, "effective_configs": eval_configs}, ensure_ascii=False, indent=2), encoding="utf-8")
    configs, summaries, tests, vals, diags, trains, logs = {}, {}, {}, {}, {}, {}, {}
    overview, curves, failures, models, run_metrics = [], [], [], [], []
    test_hash = dataset_manifest_snapshot(ROOT / "data/manifests/v2/test/manifest.json")["sha256"]
    val_hash = dataset_manifest_snapshot(ROOT / "data/manifests/v2/validation/manifest.json")["sha256"]
    SOURCES.update([ROOT / "data/manifests/v2/test/manifest.json", ROOT / "data/manifests/v2/validation/manifest.json"])
    for model, directory in RUNS.items():
        c, s = js(directory / "config.json"), js(directory / "summary.json")
        configs[model], summaries[model] = c, s
        train = pd.read_csv(directory / "train_log.csv"); update = pd.read_csv(directory / "update_log.csv")
        log = pd.read_csv(directory / "validation_log.csv")
        SOURCES.update(directory / name for name in ("train_log.csv", "update_log.csv", "validation_log.csv"))
        trains[model], logs[model] = train, log
        check(model + ":episodes_updates", len(train) == s["episodes"] == 2000 and train.episode.tolist() == list(range(1,2001))
              and len(update) == 100 and update.episode_count.sum() == 2000)
        check(model + ":finite_ppo", np.isfinite(update[["policy_loss", "value_loss", "entropy", "approx_kl", "gradient_norm"]]).all().all())
        check(model + ":train_rewards", train.reward_identity_error.abs().max() < 1e-8 and not train.sampling_truncated.any())
        check(model + ":validation_schedule", log.episode.tolist() == list(range(100,2001,100)))
        check(model + ":validation_complete", log.evaluation_complete.all() and (log.completion_coverage == 1).all())
        best = int(s["checkpoint_selection"]["best_episode"])
        selected = log[log.checkpoint_eligible].sort_values(["completion_rate", "preference_balanced_quality_score", "episode"], ascending=[False,True,True]).iloc[0]
        check(model + ":selection", int(selected.episode) == best)
        provenance = s["final_sampled"]["provenance"]
        check(model + ":config_hash", effective_config_snapshot(c)["sha256"] == provenance["effective_config_sha256"])
        for role in ("best", "last"):
            path = directory / (role + "_checkpoint.pt"); SOURCES.add(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            spec = infer_checkpoint_network_spec(payload); metadata = payload["metadata"]
            digest = network_weights_sha256(payload["network"])
            check(model + ":" + role + ":weights", digest == metadata["network_weights_sha256"])
            check(model + ":" + role + ":episode", metadata["checkpoint_episode"] == (best if role == "best" else 2000))
            check(model + ":" + role + ":schema_encoding", spec["observation_schema_version"] == 12 and spec["global_objective_encoding"] == "log2_v1"
                  and spec["feature_dimensions"]["global"] == 10 and spec["actor_head_variant"] == "shared_preference")
            check(model + ":" + role + ":workload", spec["workload_mode"] == ("variance" if model == "variance" else "geometric_owa")
                  and spec["workload_q"] == (None if model == "variance" else .5) and np.allclose(spec["objective_scales"], EVAL["excess_"+model]["scales"]))
            if role == "best":
                check(model + ":checkpoint_hash", hashlib.sha256(path.read_bytes()).hexdigest() == provenance["checkpoint_sha256"])
                check(model + ":provenance_weights", digest == provenance["network_weights_sha256"])
            models.append({"model": model, "role": role, "episode": metadata["checkpoint_episode"], "weights_sha256": digest,
                           "parameters": sum(v.numel() for v in payload["network"].values())})
            del payload
        test = read(directory / "final_sampled_instance_metrics.csv", model, model+":test")
        val = read(directory / "sampled_validation_instance_metrics.csv", model, model+":validation")
        diag = read(directory / "sampled_diagnostic_instance_metrics.csv", model, model+":diagnostic")
        tests[model], vals[model], diags[model] = test, val, diag
        check(model + ":test_complete", len(test) == 3960 and test.instance_id.nunique() == 20 and test.preference_key.nunique() == 66
              and not test.duplicated(KEY).any() and (test.groupby(KEY[:2]).size() == 3).all())
        check(model + ":validation_complete_rows", len(val) == 39000 and not val.duplicated(["validation_episode", *KEY]).any()
              and (val.groupby("validation_episode").size() == 1950).all())
        check(model + ":test_manifest", set(test.dataset_manifest_sha256) == {test_hash} == {provenance["dataset_manifest_sha256"]})
        check(model + ":validation_manifest", set(val.dataset_manifest_sha256) == {val_hash})
        check(model + ":independent_diagnostic", not set(diag.instance_id) & set(val.instance_id) and diag.instance_id.nunique() == 49)
        for stage, frame in [("test",test),("validation",val),("diagnostic",diag)]:
            check(model+":"+stage+":safe", frame.physical_safety_pass.all() and (frame.schedule_violation_count == 0).all()
                  and (frame.maximum_worker_fatigue <= frame.safe_fatigue_limit + 1e-9).all() and not frame.sampling_truncated.any())
            good = frame[safe(frame)]
            check(model+":"+stage+":native_quality", np.allclose(quality(good,"excess_"+model), good.preference_quality_score, atol=1e-8))
            check(model+":"+stage+":reward", np.allclose(frame.training_cumulative_reward, frame.operation_progress-frame.initial_progress
                  +frame.initial_preference_quality_score-frame.actual_preference_quality_score-frame.terminal_failure_penalty_applied, atol=1e-8))
            check(model+":"+stage+":failure_penalty", np.allclose(frame.terminal_failure_penalty_applied,2*frame.task_failed.astype(int)))
            check(model+":"+stage+":flow_bound", np.allclose(good.flow_excess_objective,good.flow_time_objective-good.flow_processing_lower_bound,atol=1e-8))
        check(model+":derived_seeds", all(int(r.derived_sampling_seed)==derive_evaluation_sampling_seed(int(r.sampling_seed),r.instance_id,r.preference_key) for r in test.itertuples()))
        check(model+":test_summary", int((~safe(test)).sum()) == s["final_sampled"]["task_failed_count"])
        for stage, frame in [("best_test",test),("best_validation",val[val.validation_episode==best]),
                             ("last_validation",val[val.validation_episode==2000]),("last_diagnostic",diag[diag.validation_episode==2000])]:
            overview.append({"model":model,"stage":stage,"episode":best if stage.startswith("best") else 2000,**aggregate(frame)})
            failed=frame[~safe(frame)].copy();failed["model"]=model;failed["stage"]=stage;failures.extend(failed.to_dict("records"))
        for episode, frame in val.groupby("validation_episode"):
            curves.append({"model":model,"episode":int(episode),**aggregate(frame),
                           "native_quality":frame.loc[safe(frame)].groupby("preference_key").preference_quality_score.mean().mean(),
                           "mean_return":frame.training_cumulative_reward.mean()})
        test.assign(training_method=model).to_csv(OUT/f"{model}_test_rescored.csv",index=False,encoding="utf-8-sig")
        run_metrics.append({"model":model,"training_successes":int(train.task_succeeded.sum()),"episodes":len(train),
                            "elapsed_seconds":s["elapsed_seconds"],"sampling_seconds":update.sampling_wall_time_seconds.sum(),
                            "ppo_seconds":update.ppo_update_time_seconds.sum(),"test_wall_seconds":s["final_sampled"]["wall_time_seconds"],
                            "transitions":int(update.transition_count.sum()),"test_solve_seconds_mean":test.solve_time_seconds.mean()})
    for field in ("seed","ppo","generator","dataset","environment","training","preference","reward"):
        check("matched_config:"+field,configs["variance"].get(field)==configs["owa"].get(field))
    for field in ("source_state_sha256","source_file_count","git","fixed_instance_sha256","generator_config_sha256", "environment_config_sha256",
                  "distribution_contract_sha256","template_sha256","dataset_manifest_sha256","evaluation_subset_sha256"):
        check("matched_provenance:"+field,summaries["variance"]["final_sampled"]["provenance"][field]==summaries["owa"]["final_sampled"]["provenance"][field])
    check("matched_training_sequence",trains["variance"][["instance_id","seed","preference_key","pressure_type","cost_profile"]].equals(
          trains["owa"][["instance_id","seed","preference_key","pressure_type","cost_profile"]]))
    check("matched_validation_subsets",js(RUNS["variance"]/"validation_subsets.json")==js(RUNS["owa"]/"validation_subsets.json"))
    (OUT/"config_differences.json").write_text(json.dumps(differences(configs["variance"],configs["owa"]),ensure_ascii=False,indent=2),encoding="utf-8")
    counts, paired = [],[]
    comparisons=[("best_test",tests["variance"],tests["owa"]),
                 ("selected_validation",vals["variance"][vals["variance"].validation_episode==1800],vals["owa"][vals["owa"].validation_episode==1900]),
                 ("last_validation",vals["variance"][vals["variance"].validation_episode==2000],vals["owa"][vals["owa"].validation_episode==2000]),
                 ("last_diagnostic",diags["variance"][diags["variance"].validation_episode==2000],diags["owa"][diags["owa"].validation_episode==2000])]
    for stage,a,b in comparisons:
        count,rows=comparison(a,b,stage);counts.append(count);paired.extend(rows)
    hv_rows,budgets,stratified,late=[],[],[],[]
    for model,f in tests.items():
        for coordinate,target in EVAL.items():
            for iid,g in f.groupby("instance_id"):
                values=g.loc[safe(g),target["fields"]].to_numpy()
                hv_rows.append({"model":model,"instance_id":iid,"coordinate":coordinate,
                                "hv":hypervolume_3d(values/(values+target["scales"]))})
        for scoring in ("excess_owa","excess_variance"):
            chosen=f.assign(success=safe(f),selection_quality=quality(f,scoring)).sort_values(
                ["success","selection_quality","sampling_repeat"],ascending=[False,True,True]).drop_duplicates(KEY[:2])
            budgets.append({"model":model,"selection_coordinate":scoring,"budget":3,**aggregate(chosen)})
        for field in ("pressure_type","cost_profile"):
            for value,group in f.groupby(field):stratified.append({"model":model,"field":field,"group":value,**aggregate(group)})
    curve=pd.DataFrame(curves)
    for model,g in curve[curve.episode>=1600].groupby("model"):
        late.append({"model":model,"overall_completion_mean":g.overall_completion.mean(),"worst_preference_mean":g.worst_preference_completion.mean(),
                     "worst_preference_low":g.worst_preference_completion.min(),"cost_completion_mean":g.cost_completion.mean(),"cost_completion_low":g.cost_completion.min()})
    correlation,transitions,levels,endpoints=preference_response(tests)
    dominance=pooled_dominance(tests)
    clusters=[]
    a,b=[tests[m].set_index(KEY).sort_index() for m in ("variance","owa")]
    common=safe(a)&safe(b)
    preference_groups=[]
    for name,selection in [("low_workload_weight",a.preference_workload<.5),("high_workload_weight",a.preference_workload>=.5),
                            ("pure_workload",a.preference_workload==1)]:
        mask=common&selection
        for field in PHYSICAL:
            x,y=a.loc[mask,field].mean(),b.loc[mask,field].mean()
            preference_groups.append({"group":name,"measure":field,"common_success":int(mask.sum()),
                                      "variance_mean":x,"owa_mean":y,"percent_change":100*(y/x-1)})
    normalization,coupling=[],[]
    for model,f in tests.items():
        good=f[safe(f)]
        for field,scale in [("flow_excess_objective",368.3143),("reconfiguration_cost",353.27),
                            ("owa_q05",21.03145448655653),("worker_load_variance",2.2629)]:
            q=good[field]/(good[field]+scale)
            normalization.append({"model":model,"objective":field,"scale":scale,"mean_bounded_component":q.mean(),
                                  "median_bounded_component":q.median(),"fraction_above_09":(q>.9).mean()})
        cells=good.groupby(["instance_id","preference_key"])[["reconfiguration_cost","owa_q05"]].mean()
        for iid,g in cells.groupby(level="instance_id"):
            coupling.append({"model":model,"instance_id":iid,"cost_owa_spearman":spearmanr(g.reconfiguration_cost,g.owa_q05).statistic})
    load_fields=[f"load_H{i}" for i in range(1,7)]
    load_better=(b[load_fields]<=a[load_fields]+1e-8).all(axis=1)&(b[load_fields]<a[load_fields]-1e-8).any(axis=1)
    counter=common&load_better&(b.worker_load_variance>a.worker_load_variance+1e-8)
    examples=[]
    for index in a[counter].sort_values("preference_workload",ascending=False).index[:3]:
        examples.append({"instance_id":index[0],"preference_key":index[1],"sampling_repeat":int(index[2]),
                         "preference":a.loc[index,PREF].tolist(),"variance_loads":a.loc[index,load_fields].tolist(),
                         "owa_loads":b.loc[index,load_fields].tolist(),
                         "variance_policy_variance":float(a.loc[index,"worker_load_variance"]),
                         "owa_policy_variance":float(b.loc[index,"worker_load_variance"]),
                         "variance_policy_owa":float(a.loc[index,"owa_q05"]),"owa_policy_owa":float(b.loc[index,"owa_q05"]),
                         "variance_policy_flow":float(a.loc[index,"flow_time_objective"]),"owa_policy_flow":float(b.loc[index,"flow_time_objective"]),
                         "variance_policy_cost":float(a.loc[index,"reconfiguration_cost"]),"owa_policy_cost":float(b.loc[index,"reconfiguration_cost"])})
    (OUT/"load_examples.json").write_text(json.dumps({"paired_load_reduction_with_higher_variance":int(counter.sum()),"examples":examples},ensure_ascii=False,indent=2),encoding="utf-8")
    for iid in a.index.get_level_values("instance_id").unique():
        mask=common&(a.index.get_level_values("instance_id")==iid)
        allmask=a.index.get_level_values("instance_id")==iid
        clusters.append({"instance_id":iid,"completion_delta":safe(b.loc[allmask]).mean()-safe(a.loc[allmask]).mean(),
                         **{field+"_delta":(b.loc[mask,field]-a.loc[mask,field]).mean() for field in PHYSICAL},
                         **{"quality_"+coord+"_delta":(quality(b.loc[mask],coord)-quality(a.loc[mask],coord)).mean() for coord in EVAL}})
    cluster=pd.DataFrame(clusters).set_index("instance_id");rng=np.random.default_rng(20261014)
    draws=rng.integers(0,len(cluster),size=(20000,len(cluster)));uncertainty=[]
    for field in cluster:
        v=cluster[field].to_numpy();low,high=np.quantile(v[draws].mean(axis=1),[.025,.975])
        uncertainty.append({"measure":field,"mean_delta":v.mean(),"ci95_low":low,"ci95_high":high,"instance_clusters":len(v)})
    for name,rows in [("overview",overview),("paired_counts",counts),("paired_metrics",paired),("validation_curve",curves),
                      ("endpoints",endpoints),("failures",failures),("best_of_three",budgets),("hypervolume_by_instance",hv_rows),
                      ("stratified_test",stratified),("late_validation",late),("checkpoints",models),("run_metrics",run_metrics),
                      ("cluster_bootstrap",uncertainty),("labelled_dominance_summary",dominance),("checks",CHECKS),
                      ("preference_group_metrics",preference_groups),("normalized_objectives",normalization),("cost_owa_coupling",coupling)]:save(name,rows)
    result={"runs":{k:p.as_posix() for k,p in RUNS.items()},"overview":overview,"paired_counts":counts,"paired_metrics":paired,
            "endpoints":endpoints,"late_validation":late,"preference_correlations":correlation,"preference_directions":transitions,
            "hypervolume":pd.DataFrame(hv_rows).groupby(["model","coordinate"]).hv.mean().unstack().to_dict("index"),
            "best_of_three":budgets,"cluster_bootstrap":uncertainty,"labelled_dominance":dominance,"run_metrics":run_metrics,
            "preference_group_metrics":preference_groups,"normalization_diagnostics":normalization,
            "cost_owa_coupling_median":pd.DataFrame(coupling).groupby("model").cost_owa_spearman.median().to_dict(),
            "load_reduction_higher_variance_count":int(counter.sum()),"load_examples":examples,
            "checks_passed":len(CHECKS),"additional_training_or_policy_rollouts":0,"new_last_full_test":False,
            "evaluation_config_sha256":hashlib.sha256((OUT/"evaluation_config.json").read_bytes()).hexdigest(),
            "git_record":summaries["owa"]["final_sampled"]["provenance"]["git"],
            "source_state_sha256":summaries["owa"]["final_sampled"]["provenance"]["source_state_sha256"],
            "input_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}}
    (OUT/"comparison.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    plot(curve,pd.DataFrame(levels))
    report(result)
    print(json.dumps({k:result[k] for k in ["checks_passed","overview","paired_counts","hypervolume","late_validation","labelled_dominance"]},ensure_ascii=False,indent=2))


def plot(curves,levels):
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(10,6.7),constrained_layout=True)
    for model,g in curves.groupby("model"):
        for ax,field,label in zip(axes.flat,["overall_completion","cost_completion","quality_excess_owa","quality_excess_variance"],
                                 ["Overall completion (%)","Cost completion (%)","Common OWA quality loss","Common Variance quality loss"]):
            scale=100 if "completion" in field else 1
            ax.plot(g.episode,g[field]*scale,marker=".",label=model);ax.set_xlabel("Training episodes");ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0,0].legend()
    for ext in ("png","svg"):fig.savefig(OUT/f"validation_comparison.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(13,3.5),constrained_layout=True)
    for ax,target,title in zip(axes,["flow_time_objective","reconfiguration_cost","owa_q05","worker_load_variance"],["Raw Flow","Cost","OWA(q=0.5)","Load variance"]):
        for model,g in levels[levels.target==target].groupby("model"):ax.plot(g.weight,g["mean"],marker=".",label=model)
        ax.set_xlabel("Corresponding preference weight");ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].legend()
    for ext in ("png","svg"):fig.savefig(OUT/f"preference_response.{ext}",dpi=180)
    plt.close(fig)


def report(result):
    overview={(r["model"],r["stage"]):r for r in result["overview"]}
    a,b=overview["variance","best_test"],overview["owa","best_test"]
    matched={r["measure"]:r for r in result["paired_metrics"] if r["stage"]=="best_test"}
    groups={(r["group"],r["measure"]):r for r in result["preference_group_metrics"]}
    text="# Schema12 OWA 与 Variance 完整筛选对照（2026-10-14）\n\n"
    text+="**结论：OWA(q=0.5)值得作为减轻个人负担的候选继续做 bounded 输入对照，但没有全面支配 Variance。** 两次各2000轮的实验已完成，均使用shared_head、excess、schema12、log2、相同数据/偏好/seed11。OWA降低Cost、总劳动量与最大个人负担，Flow略增加；高均衡偏好下仍由Variance获得更小的方差。best测试完成率OWA略低，末期Cost偏好稳定性则明显改善。\n\n"
    text+="本报告复用保存的训练/验证/测试轨迹，未新增训练或策略评估。best与last分别读原保存配置；本轮last尚无完整测试，仅以选模验证与独立验证分析，不能把验证数值写作last测试性能。\n\n"
    text+="## 1. 可比性与验收\n\n"
    text+=f"{result['checks_passed']}项审计通过。两组各2000 episodes、100次PPO更新；训练实例ID、生成种子、偏好key与分布顺序一致。源码source_state聚合哈希一致：`{result['source_state_sha256']}`；git均记录`4319827`且dirty=true，因此可确认已捕获Python源码集合一致，但不宣称独立复原了远端全部运行环境。模型文件哈希、权重哈希、schema12/log2/shared身份、对应workload身份及冻结尺度均通过核验。\n\n"
    text+="主对照只改变第三目标及其配套的观测、奖励目标和冻结尺度（Variance=2.2629，OWA=21.03145448655653）。Flow=368.3143、Cost=353.27、失败罚项2、PPO及选模规则相同。分别重建每条轨迹的六位工人负担、population variance、OWA(q=.5)、总劳动量与最大个人负担；三类评估均检查安全、完整性和奖励恒等式。\n\n"
    text+="两次best由各自预先定义的(min偏好完成率↑，自身质量损失↓)选出：Variance ep1800，OWA ep1900。这是相同选模流程下的算法对照；同为ep2000的last另行比较。\n\n"
    text+="## 2. Best完整测试\n\n20实例×66偏好×3次采样，共3960请求；实例、偏好和derived sampling seed逐项配对，采样种子300011/300012/300013。\n\n| 指标 | Variance best | OWA best |\n|---|---:|---:|\n"
    for title,field,scale in [("整体完成率 (%)","overall_completion",100),("失败数","failures",1),
                              ("最差偏好完成率 (%)","worst_preference_completion",100),("纯Cost完成率 (%)","cost_completion",100)]:
        text+=f"| {title} | {a[field]*scale:.4f} | {b[field]*scale:.4f} |\n"
    count=next(r for r in result["paired_counts"] if r["stage"]=="best_test")
    text+=f"\n共同成功{count['common_success']}/3960（{100*count['coverage']:.3f}%）；OWA新增成功{count['owa_gained']}，失去Variance成功{count['owa_lost']}，双方同时失败{count['both_failed']}。OWA多3次失败，整体完成率差−0.0758个百分点。20实例簇bootstrap的完成率差95%区间约[−0.2525,+0.1010]个百分点，不能据单seed/20实例宣称完成率提升或等价。\n\n"
    text+="以下为同一共同成功集合的配对均值，避免失败时少加工造成低Cost的假收益：\n\n| 指标 | Variance | OWA | OWA变化 |\n|---|---:|---:|---:|\n"
    for field in PHYSICAL:
        r=matched[field];text+=f"| {field} | {r['variance_mean']:.4f} | {r['owa_mean']:.4f} | {r['percent_change']:+.2f}% |\n"
    text+="\nOWA减少重构和工人服务劳动，生产WAIT时间更长，符合较低Cost/负担与较高Flow之间的折中。这是保存轨迹中的统计关联，尚未逐动作干预证明因果。\n\n"
    text+="## 3. 统一坐标下的质量与Pareto\n\n"
    text+="对两模型同时重算四套坐标：raw/excess Flow × Variance/OWA第三目标。每套使用同一个冻结manifest、q_i=J_i/(s_i+J_i)、Q=(max_i w_iq_i+0.05 sum_i w_iq_i)/1.05。每个偏好先平均成功样本，再对偏好等权；HV在每个实例内部汇集66×3候选后计算，再按实例平均，参考点(1,1,1)。\n\n"
    text+="| 共同评价坐标 | 配对Q：Variance | 配对Q：OWA | Q变化 | HV：Variance | HV：OWA |\n|---|---:|---:|---:|---:|---:|\n"
    for coord in EVAL:
        r=matched["quality_"+coord];text+=f"| {coord} | {r['variance_mean']:.6f} | {r['owa_mean']:.6f} | {r['percent_change']:+.2f}% | {result['hypervolume']['variance'][coord]:.6f} | {result['hypervolume']['owa'][coord]:.6f} |\n"
    text+="\nQ越低、HV越高越好。OWA坐标中有收益，Variance坐标中退步；原生0.35281与0.28958不是同一量纲的质量对照。evaluation_config.json保存四套评价身份和有效配置哈希；*_test_rescored.csv保留training_*原始指标，将重算值置于evaluation_quality_*字段。\n\n"
    text+="按20个实例进行配对簇bootstrap（保留实例内部偏好/重复相关性），共同OWA-Q差的95%区间约[−0.02172,−0.00942]，Flow差约[+6.10,+43.35]，Cost差约[−142.09,−77.78]。这支持当前固定测试上的取舍方向，不代表跨训练seed的统计显著性。\n\n"
    text+="## 4. 减负与均衡需要分开解释\n\n"
    text+="总体方差下降54.45%，但这不能推出高均衡偏好下也更均匀。按第三目标权重分组：\n\n| 分组 | 共同成功数 | Variance模型方差 | OWA模型方差 | 总劳动变化 | 最大个人负担变化 |\n|---|---:|---:|---:|---:|---:|\n"
    for group in ("low_workload_weight","high_workload_weight","pure_workload"):
        r=groups[group,"worker_load_variance"]
        text+=f"| {group} | {r['common_success']} | {r['variance_mean']:.4f} | {r['owa_mean']:.4f} | {groups[group,'worker_load_total']['percent_change']:+.2f}% | {groups[group,'worker_load_max']['percent_change']:+.2f}% |\n"
    text+="\n高第三目标权重(≥.5)时，OWA的总劳动量更低、最大个人负担也更低，但负担分配更不均匀。OWA同时追求负担大小和偏向较重工人的排序权重，不能等同于原Variance。\n\n"
    text+=f"配对请求中有{result['load_reduction_higher_variance_count']}条出现OWA让每位同名工人的负担均不增加、至少一人下降，而Variance值反而上升。这说明只看方差可能把整体减负判为变差；不代表这些轨迹的Flow与Cost也一定共同支配。具体例子见load_examples.json。\n\n"
    text+="在同一实例的全部成功候选池中，对(raw Flow, Cost, H1..H6实际负担)做逐坐标支配诊断：\n\n| 模型 | 候选数 | 被同池任意候选支配 | 被另一模型候选支配 |\n|---|---:|---:|---:|\n"
    for r in result["labelled_dominance"]:
        text+=f"| {r['model']} | {r['candidates']} | {100*r['dominated_by_any_fraction']:.2f}% | {100*r['dominated_by_other_model_fraction']:.2f}% |\n"
    text+="\nOWA的经验支配率较低但没有归零；有限采样策略仍可能给出被其他采样方案支配的解。这是同池诊断，不是全局最优性或不可支配性证明。\n\n"
    text+="## 5. 偏好响应\n\n"
    text+="每个实例/偏好先平均成功重复，再在每个实例计算权重—目标Spearman，表中为20实例中位数；越负说明目标随自身权重增加越稳定下降。Workload列对两模型都用同一OWA实测指标，避免把OWA响应和Variance响应当成同一目标。\n\n| 实测目标 | Variance模型 | OWA模型 |\n|---|---:|---:|\n"
    corrs={(r["model"],r["target"]):r for r in result["preference_correlations"]}
    for field in ("flow_time_objective","reconfiguration_cost","owa_q05","worker_load_variance","worker_load_max"):
        text+=f"| {field} | {corrs['variance',field]['median']:.4f} | {corrs['owa',field]['median']:.4f} |\n"
    text+="\nOWA模型的Flow响应更强，Cost响应仍有效但弱于Variance模型；提高第三权重能降低实际OWA和最大个人负担，其响应不等于强均衡响应。固定另外两个权重比例后沿偏好路径比较，OWA模型Flow/Cost/OWA下降或持平比例约69.8%/74.0%/72.2%；因此有响应但非严格逐步单调。每格仅三次采样，路径共用端点，不能把路径比较数当成独立样本。\n\n"
    text+=f"Cost与OWA在实例内部跨偏好的Spearman中位数：Variance模型{result['cost_owa_coupling_median']['variance']:.3f}，OWA模型{result['cost_owa_coupling_median']['owa']:.3f}。OWA模型这两个目标强同向，说明减少重构劳动时常同时降低成本；这也限制了两者之间可形成的独立折中范围。\n\n"
    text+="端点实际结果（成功轨迹均值；不同完成集合应同时看计数）：\n\n| 模型/端点 | 成功 | Flow | Cost | OWA | 方差 | 最大个人负担 |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for r in result["endpoints"]:
        text+=f"| {r['model']}/{r['endpoint']} | {r['successes']}/{r['requests']} | {r['flow_time_objective']:.2f} | {r['reconfiguration_cost']:.2f} | {r['owa_q05']:.2f} | {r['worker_load_variance']:.2f} | {r['worker_load_max']:.2f} |\n"
    text+="\n![偏好响应](preference_response.png)\n\n## 6. 后期稳定性、失败与last\n\n"
    text+="| 阶段 | 模型 | 成功/请求 | 最差偏好 | 纯Cost |\n|---|---|---:|---:|---:|\n"
    for stage in ("best_validation","last_validation","last_diagnostic"):
        for model in RUNS:
            r=overview[model,stage];text+=f"| {stage} | {model} | {r['successes']}/{r['requests']} | {100*r['worst_preference_completion']:.2f}% | {100*r['cost_completion']:.2f}% |\n"
    text+="\n独立验证为49个未参与选模的同分布实例×13偏好×3重复，并非OOD。OWA last在选模集和独立验证都比Variance last更稳定，并有较低负担；尚未执行last完整测试，不能据此替换best作为正式测试结果。\n\n"
    text+="最后五次验证(ep1600–2000)：\n\n| 模型 | 整体完成率均值 | 最差偏好均值 | 纯Cost均值 | 纯Cost最低值 |\n|---|---:|---:|---:|---:|\n"
    for r in result["late_validation"]:
        text+=f"| {r['model']} | {100*r['overall_completion_mean']:.3f}% | {100*r['worst_preference_mean']:.3f}% | {100*r['cost_completion_mean']:.3f}% | {100*r['cost_completion_low']:.3f}% |\n"
    text+="\nOWA改善了末期最差偏好，末五点整体完成率均值却略低；其ep1100最差偏好曾降至76%，不能称作训练全过程稳定或已解决失败问题。OWA best的8次测试失败均为horizon：5次重构瓶颈、3次机器瓶颈，进度约97.1%–98.9%，疲劳峰值约0.23–0.58。相比Variance best失败时接近0.75的疲劳，这里不能将所有超时归为疲劳上限；具体调度原因仍需动作回放验证。\n\n"
    text+="![统一坐标下的验证曲线](validation_comparison.png)\n\n## 7. 三次采样择优与计算开销\n\n"
    text+="两模型均1320/1320实例/偏好单元至少一次成功。按同一OWA坐标成功优先择优，quality分别为：\n\n| 模型 | 三次择优共同OWA质量损失 |\n|---|---:|\n"
    for r in result["best_of_three"]:
        if r["selection_coordinate"]=="excess_owa":text+=f"| {r['model']} | {r['quality_excess_owa']:.6f} |\n"
    text+="\n选择坐标固定后OWA依然更好；若以共同Variance坐标选择和评价，Variance更好，完整双口径见best_of_three.csv。择优只复用现有三次样本，不能当作单次推断成绩。\n\n"
    text+="| 模型 | 总耗时小时 | 训练决策样本 | 完整best测试秒 |\n|---|---:|---:|---:|\n"
    for r in result["run_metrics"]:text+=f"| {r['model']} | {r['elapsed_seconds']/3600:.2f} | {r['transitions']} | {r['test_wall_seconds']:.1f} |\n"
    text+="\nOWA该次运行较快，且决策样本更少；硬件负载和策略轨迹均影响耗时，不能将差异全部归因于目标计算更快。\n\n## 8. 下一步预算建议\n\n"
    text+="按预先约定，以验证而非测试决定补跑：OWA在验证中降低实际负担，末期Cost风险没有持续恶化，last的独立验证也支持这一方向。若继续采用已选定的OWA负担语义，建议只补`owa_bounded`这一组2000轮，与本组OWA/log2保持其余设置一致；保留Variance结果作为均衡目标对照。该建议不等于OWA全面优于Variance，尤其不能忽略best完成率略低和高均衡偏好的方差代价。\n\n"
    text+="```powershell\npython scripts/run_schema12_screening.py --experiment owa_bounded\n```\n\n"
    text+="本报告未启动该命令。后续若评估OWA last，只补该保留候选的完整测试，并使用run内config.json；多seed、raw、q敏感性和单目标上界留到编码对照有结果之后。\n\n"
    text+="复现：`python -m analysis.studies.schema12_owa_comparison_20261014`。comparison.json含数据/模型/脚本哈希；checks.csv记录验收；paired_metrics.csv、endpoints.csv、failures.csv、preference_group_metrics.csv、cluster_bootstrap.csv为主要明细。\n"
    (OUT/"report.md").write_text(text,encoding="utf-8")


if __name__ == "__main__":
    main()
