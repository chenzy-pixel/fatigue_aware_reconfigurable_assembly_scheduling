"""Saved-run diagnosis and last checkpoint quality, using declared historical metrics."""
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
from analysis.studies.remaining_batch_review_20261012 import quality, safe, OBJECTIVES, PREFS, SCALES
from result.provenance import network_weights_sha256

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "result/runs/remaining_flow_experiments_seed11_20261008_225956_shared_head_universal"
OUT = ROOT / "result/analysis/shared_excess_last_20261012"
KEYS = ["instance_id", "preference_key", "sampling_repeat"]
SOURCES: set[Path] = {Path(__file__)}
MEASURES = [*OBJECTIVES, "flow_excess_objective", "completed_reconfigurations", "worker_reconfiguration_busy_minutes",
    "machine_waiting_for_worker_time", "maximum_worker_fatigue", "fatigue_masked_action_ratio", "production_wait_time",
    "worker_wait_time", "wait_action_count", "operation_progress", "decision_count", "direct_process_action_count",
    "commit_reconfig_action_count", "context_override_rate", "ranker_top_selection_rate"]
COLS = {*KEYS, *MEASURES, *PREFS, "validation_episode", "task_succeeded", "task_failed", "physical_safety_pass",
    "schedule_violation_count", "safe_fatigue_limit", "termination_reason", "training_cumulative_reward", "base_cumulative_reward",
    "preference_quality_score", "actual_preference_quality_score", "terminal_failure_penalty_applied", "initial_progress",
    "initial_preference_quality_score", "pressure_type", "cost_profile", "derived_sampling_seed", "sampling_seed", "action_trace_sha256"}


def read(path, selected=True):
    SOURCES.add(path)
    return pd.read_csv(path, usecols=(lambda col: col in COLS) if selected else None)


def write(name, records):
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(OUT/(name+".csv"), index=False, encoding="utf-8-sig")


def aggregate(frame):
    good = frame[safe(frame)]
    indexed = frame.set_index(KEYS)
    groups = indexed.groupby(level="preference_key")
    return {"trajectories": len(frame), "successes": len(good), "overall_completion": len(good)/len(frame),
        "minimum_preference_completion": groups.task_succeeded.mean().min(),
        "quality_excess": pd.Series(quality(good,"excess"), index=good.index).groupby(good.preference_key).mean().mean(),
        "quality_raw": pd.Series(quality(good,"raw"), index=good.index).groupby(good.preference_key).mean().mean(),
        "mean_training_return": frame.training_cumulative_reward.mean(),
        "failure_count": len(frame)-len(good), **{field: good[field].mean() for field in MEASURES},
        "all_safe": bool(frame.physical_safety_pass.all() and frame.schedule_violation_count.sum()==0)}


def compare(first, second, stage):
    a, b = first.set_index(KEYS).sort_index(), second.set_index(KEYS).sort_index()
    assert a.index.equals(b.index) and a.derived_sampling_seed.equals(b.derived_sampling_seed)
    common = safe(a)&safe(b)
    counts = {"stage": stage, "common_success": int(common.sum()),
        "last_gained": int((~safe(a)&safe(b)).sum()), "last_lost": int((safe(a)&~safe(b)).sum()),
        "both_fail": int((~safe(a)&~safe(b)).sum())}
    paired = []
    for field in MEASURES:
        aa,bb=a.loc[common,field].mean(), b.loc[common,field].mean()
        paired.append({"stage":stage,"measure":field,"best_mean":aa,"last_mean":bb,
            "percent_change":100*(bb/aa-1) if abs(aa)>1e-12 else np.nan,
            "last_lower_fraction":(b.loc[common,field]<a.loc[common,field]-1e-9).mean()})
    for mode in ("raw","excess"):
        aq=pd.Series(quality(a.loc[common],mode),index=a.loc[common].index)
        bq=pd.Series(quality(b.loc[common],mode),index=b.loc[common].index)
        aa=aq.groupby(level="preference_key").mean().mean();bb=bq.groupby(level="preference_key").mean().mean()
        paired.append({"stage":stage,"measure":"quality_"+mode,"best_mean":aa,"last_mean":bb,
            "percent_change":100*(bb/aa-1),"last_lower_fraction":(bq<aq-1e-9).mean()})
    outcomes = pd.DataFrame({"best_success":safe(a),"last_success":safe(b)})
    return counts, paired, outcomes.reset_index()


def hv(frame, role, stage):
    rows=[]
    good=frame[safe(frame)]
    for iid,g in good.groupby("instance_id"):
        for mode in ("raw","excess"):
            fields=OBJECTIVES if mode=="raw" else ["flow_excess_objective",*OBJECTIVES[1:]]
            points=g[fields].to_numpy();rows.append({"stage":stage,"role":role,"instance_id":iid,"mode":mode,
                "pooled_hv":hypervolume_3d(points/(points+SCALES[mode]))})
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    val=read(RUN/"sampled_validation_instance_metrics.csv")
    diagnostic=read(RUN/"sampled_diagnostic_instance_metrics.csv")
    updates=read(RUN/"update_log.csv",False)
    train=read(RUN/"train_log.csv",False)
    metadata={}
    for role,episode in (("best",400),("last",2000)):
        path=RUN/(role+"_checkpoint.pt");SOURCES.add(path)
        ck=torch.load(path,map_location="cpu",weights_only=False)
        assert ck["metadata"]["checkpoint_episode"]==episode
        assert network_weights_sha256(ck["network"])==ck["metadata"]["network_weights_sha256"]
        metadata[role]={"episode":episode,"file_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "network_weights_sha256":ck["metadata"]["network_weights_sha256"],"identity":ck["metadata"]["runtime_manifest"]}
        del ck
    best=val[val.validation_episode==400].copy();last=val[val.validation_episode==2000].copy()
    assert len(best)==len(last)==1950
    assert np.allclose(val.loc[val.task_succeeded,"preference_quality_score"],quality(val[val.task_succeeded],"excess"),atol=1e-8)
    assert (val.terminal_failure_penalty_applied==2*val.task_failed.astype(int)).all()
    overview=[{"stage":"selection_validation","role":"best",**aggregate(best)},
              {"stage":"selection_validation","role":"last",**aggregate(last)},
              {"stage":"independent_diagnostic_validation","role":"last",**aggregate(diagnostic[diagnostic.validation_episode==2000])}]
    curves=[];preferences=[];failure_rows=[];failure_measures=[];instance_failures=[]
    for episode,g in val.groupby("validation_episode"):
        curves.append({"episode":episode,**aggregate(g)})
        for key,request in g.groupby("preference_key"):
            good=request[safe(request)];bad=request[~safe(request)]
            preferences.append({"episode":episode,"preference_key":key,**{p:request[p].iloc[0] for p in PREFS},
                "successes":len(good),"total":len(request),"completion":len(good)/len(request),
                "quality":good.preference_quality_score.mean(), "mean_training_return":request.training_cumulative_reward.mean(),
                **{field:good[field].mean() for field in OBJECTIVES}})
        requests=g.assign(safe=safe(g)).groupby(KEYS[:2]).safe.agg(["sum","count"])
        curves[-1]["request_count"]=len(requests)
        curves[-1]["bestof3_requests_covered"]=int((requests["sum"]>0).sum())
        curves[-1]["bestof3_worst_preference_coverage"]=(requests["sum"]>0).groupby(level="preference_key").mean().min()
        for mode in ("raw","excess"):
            successful=g[safe(g)].copy();successful["common_quality"]=quality(successful,mode)
            minima=successful.groupby(KEYS[:2]).common_quality.min().reindex(requests.index)
            curves[-1]["bestof3_quality_"+mode]=minima.groupby(level="preference_key").mean().mean()
            curves[-1]["bestof3_failure_aware_quality_"+mode]=minima.fillna(1).groupby(level="preference_key").mean().mean()
        for (iid,pressure),g1 in g.groupby(["instance_id","pressure_type"]):
            instance_failures.append({"episode":episode,"instance_id":iid,"pressure_type":pressure,
                "failures":int((~g1.task_succeeded).sum()),"total":len(g1)})
        bad=g[~safe(g)]
        failure_rows.extend(bad.to_dict("records"))
        for is_cost,group in (("pure_cost",g[g.preference_cost==1]),("all",g)):
            for success,sub in group.groupby("task_succeeded"):
                failure_measures.append({"episode":episode,"preference_subset":is_cost,"success":success,"count":len(sub),
                    **{field:sub[field].mean() for field in MEASURES}})
    counts,paired,outcomes=compare(best,last,"selection_validation")
    hvs=hv(best,"best","selection_validation")+hv(last,"last","selection_validation")
    endpoints=[]
    for role,group in (("best",best),("last",last)):
        for pref,objective in zip(PREFS,OBJECTIVES):
            request=group[group[pref]==1];good=request[safe(request)]
            endpoints.append({"stage":"selection_validation","role":role,"objective":objective,
                "successes":len(good),"total":len(request),"mean":good[objective].mean(),
                "bestof3":good.groupby("instance_id")[objective].min().mean()})
    # Expected-return and worst-preference completion optimize different rankings.
    reward_analysis=[]
    for name,pref in (("all",None),("pure_cost","preference_cost")):
        a,b=(best,last) if pref is None else (best[best[pref]==1],last[last[pref]==1])
        base_delta=b.base_cumulative_reward.mean()-a.base_cumulative_reward.mean()
        failure_delta=b.task_failed.mean()-a.task_failed.mean()
        reward_analysis.append({"subset":name,"best_base_return":a.base_cumulative_reward.mean(),
            "last_base_return":b.base_cumulative_reward.mean(),"best_training_return":a.training_cumulative_reward.mean(),
            "last_training_return":b.training_cumulative_reward.mean(),"best_failure_probability":a.task_failed.mean(),
            "last_failure_probability":b.task_failed.mean(),"offline_penalty_tie":base_delta/failure_delta if failure_delta>0 else np.nan})
    model_response=[]
    for role,group in (("best",best),("last",last)):
        for iid,g in group[safe(group)].groupby("instance_id"):
            for pref,obj in zip(PREFS,OBJECTIVES):
                model_response.append({"stage":"selection_validation","role":role,"instance_id":iid,
                    "objective":obj,"spearman":spearmanr(g[pref],g[obj]).statistic})
    # Fixed matched requests for low/high Cost show the behavioral shift without success-subset bias.
    cost_slices=[]
    a,b=best.set_index(KEYS).sort_index(),last.set_index(KEYS).sort_index()
    common=safe(a)&safe(b)
    for label,mask in (("cost_zero",a.preference_cost==0),("cost_0_to_half",(a.preference_cost>0)&(a.preference_cost<=.5)),("cost_above_half",a.preference_cost>.5)):
        selected=common&mask
        for role,frame in (("best",a),("last",b)):
            cost_slices.append({"role":role,"cost_slice":label,"matched_successes":int(selected.sum()),
                **{field:frame.loc[selected,field].mean() for field in MEASURES}})
    last_test=OUT/"last_test_instance_metrics.csv"
    complete=(OUT/"last_test_metrics.json").exists()
    test_comparison=None
    expert_comparison=None
    if complete:
        fresh=read(last_test);archived=read(RUN/"final_sampled_instance_metrics.csv")
        assert len(fresh)==len(archived)==3960
        test_comparison,test_pairs,test_outcomes=compare(archived,fresh,"final_test")
        paired.extend(test_pairs);write("last_test_paired_outcomes",test_outcomes)
        overview.extend([{"stage":"final_test","role":"best",**aggregate(archived)},
                         {"stage":"final_test","role":"last",**aggregate(fresh)}])
        hvs.extend(hv(archived,"best","final_test")+hv(fresh,"last","final_test"))
        for role,group in (("best",archived),("last",fresh)):
            for pref,objective in zip(PREFS,OBJECTIVES):
                request=group[group[pref]==1];good=request[safe(request)]
                endpoints.append({"stage":"final_test","role":role,"objective":objective,"successes":len(good),
                    "total":len(request),"mean":good[objective].mean(),"bestof3":good.groupby("instance_id")[objective].min().mean()})
        write("last_test_failures",fresh[~safe(fresh)])
        expert_path=ROOT/"result/runs/remaining_flow_experiments_seed11_20261008_225956_main_universal/final_sampled_instance_metrics.csv"
        expert=read(expert_path)
        expert_comparison,expert_pairs,expert_outcomes=compare(expert,fresh,"experts_raw_vs_shared_last_test")
        write("last_vs_experts_paired_metrics",pd.DataFrame(expert_pairs).rename(columns={"best_mean":"expert_mean"}))
        write("last_vs_experts_paired_outcomes",expert_outcomes.rename(columns={"best_success":"expert_success"}))
        overview.append({"stage":"final_test","role":"experts_raw_best",**aggregate(expert)})
        hvs.extend(hv(expert,"experts_raw_best","final_test"))
    hvs=pd.DataFrame(hvs)
    for name,records in (("checkpoint_overview",overview),("validation_diagnosis_curves",curves),
        ("validation_by_preference",preferences),("validation_failures",failure_rows),
        ("validation_failure_measures",failure_measures),("validation_by_instance",instance_failures),
        ("last_vs_best_paired_metrics",paired),("last_vs_best_validation_outcomes",outcomes),
        ("last_vs_best_endpoints",endpoints),("expected_return_ranking",reward_analysis),
        ("preference_response_checkpoints",model_response),("matched_cost_slices",cost_slices),("checkpoint_hypervolumes",hvs)):
        write(name,records)
    hv_means=hvs.groupby(["stage","role","mode"]).pooled_hv.mean().reset_index().to_dict("records")
    late=val[val.validation_episode>=1000]
    late_failed=late[~safe(late)]
    summary={"checkpoints":metadata,"last_test_complete":complete,"validation_pairing":counts,"test_pairing":test_comparison,
        "expert_test_pairing":expert_comparison,
        "overview":overview,"hypervolume_means":hv_means,"reward_ranking":reward_analysis,
        "late_validation_trajectories":len(late),"late_failures":len(late_failed),
        "late_failures_pure_cost":int((late_failed.preference_cost==1).sum()),
        "late_failures_cost_above_half":int((late_failed.preference_cost>.5).sum()),
        "late_failure_reasons":late_failed.termination_reason.value_counts().to_dict(),
        "ppo_finite":bool(np.isfinite(updates[["loss","approx_kl","gradient_norm","gradient_norm_max","pre_update_explained_variance"]]).all().all()),
        "training_preference_counts":train.preference_group.value_counts().to_dict(),
        "training_completion_by_preference":train.groupby("preference_group").task_succeeded.mean().to_dict(),
        "source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}}
    (OUT/"diagnosis_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    plot(pd.DataFrame(curves),pd.DataFrame(preferences),pd.DataFrame(failure_measures))
    print(json.dumps({k:summary[k] for k in ["last_test_complete","validation_pairing","overview","hypervolume_means","reward_ranking","late_failures","late_failures_pure_cost","late_failures_cost_above_half","late_failure_reasons"]},ensure_ascii=False,indent=2))


def plot(curves,prefs,measures):
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    axes[0,0].plot(curves.episode,100*curves.overall_completion,label="Overall")
    axes[0,0].plot(curves.episode,100*curves.minimum_preference_completion,label="Worst preference")
    axes[0,0].set_ylabel("Completion (%)");axes[0,0].legend()
    axes[0,1].plot(curves.episode,curves.quality_excess,label="Successful preference loss")
    axes[0,1].plot(curves.episode,curves.mean_training_return,label="All-request training return")
    axes[0,1].set_ylabel("Loss / return");axes[0,1].legend(fontsize=8)
    for j,(pref,label) in enumerate(zip(PREFS,["Pure Flow","Pure Cost","Pure Variance"])):
        sub=prefs[prefs[pref]==1];axes[1,0].plot(sub.episode,100*sub.completion,label=label)
    axes[1,0].set_ylabel("Endpoint completion (%)");axes[1,0].legend(fontsize=8)
    for successful,label in ((True,"Successful Cost requests"),(False,"Failed Cost requests")):
        sub=measures[(measures.preference_subset=="pure_cost")&(measures.success==successful)]
        axes[1,1].plot(sub.episode,sub.completed_reconfigurations,label=label)
    axes[1,1].set_ylabel("Mean completed reconfigurations");axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.set_xlabel("Training episodes");ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/"late_completion_diagnosis.png",dpi=200);fig.savefig(OUT/"late_completion_diagnosis.svg");plt.close(fig)
    matrix=prefs.pivot(index="preference_key",columns="episode",values="completion")
    labels=[]
    for key in matrix.index:
        g=prefs[prefs.preference_key==key].iloc[0];labels.append("("+", ".join(f"{g[p]:.2g}" for p in PREFS)+")")
    fig,ax=plt.subplots(figsize=(12,6));im=ax.imshow(100*matrix.to_numpy(),aspect="auto",vmin=75,vmax=100,cmap="viridis")
    ax.set_yticks(range(len(labels)),labels);ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=45)
    ax.set_xlabel("Training episodes");ax.set_ylabel("Preference (Flow, Cost, Variance)")
    fig.colorbar(im,ax=ax,label="Completion (%)");fig.tight_layout();fig.savefig(OUT/"preference_completion_heatmap.png",dpi=200);plt.close(fig)


if __name__=="__main__":main()
