"""Saved-result audit and convergence readout for the matched OWA lambda pair.

Run: python -m analysis.studies.schema12_gae099_convergence_20261015
No model training or policy rollouts are started by this analysis.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agent.ppo.network import infer_checkpoint_network_spec
from analysis.pareto_analysis import hypervolume_3d
import analysis.studies.schema12_owa_comparison_20261014 as audit
from result.provenance import effective_config_snapshot,network_weights_sha256,dataset_manifest_snapshot
from utils import derive_evaluation_sampling_seed

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"result/analysis/schema12_gae099_convergence_20261015"
RUNS={"lambda095":ROOT/"result/runs/schema12_screen_seed11_20261012_201659_owa_log2",
      "lambda099":ROOT/"result/runs/schema12_screen_seed11_20261014_185917_owa_log2_lambda099"}
KEY=audit.KEY;PREF=audit.PREF;PHYSICAL=audit.PHYSICAL
SOURCES={Path(__file__),ROOT/"analysis/schema12_owa_comparison_20261014.py",ROOT/"analysis/pareto_analysis.py"}
CHECKS=[]


def check(name,value):
    CHECKS.append({"check":name,"passed":bool(value)})
    if not value:raise AssertionError(name)


def js(path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def csv(path):
    SOURCES.add(path)
    return pd.read_csv(path)


def save(name,rows):
    pd.DataFrame(rows).to_csv(OUT/(name+".csv"),index=False,encoding="utf-8-sig")


def source_hash(ref):
    packages={"agent","analysis","configs","data","environment","scripts","training"}
    paths=subprocess.check_output(["git","ls-tree","-r","--name-only",ref],text=True).splitlines()
    files=sorted(p for p in paths if p.endswith(".py") and ("/" not in p or p.split("/")[0] in packages or (p.startswith("result/") and p.count("/")==1)))
    raw=subprocess.check_output(["git","cat-file","--batch"],input=("\n".join(ref+":"+p for p in files)+"\n").encode())
    offset=0;digest=hashlib.sha256()
    for path in files:
        end=raw.index(b"\n",offset);size=int(raw[offset:end].split()[-1]);content=raw[end+1:end+1+size];offset=end+size+2
        content=content.decode("utf-8-sig").replace("\r\n","\n").replace("\r","\n").encode();name=path.encode()
        digest.update(len(name).to_bytes(8,"big"));digest.update(name);digest.update(len(content).to_bytes(8,"big"));digest.update(content)
    return {"commit":ref,"python_file_count":len(files),"source_state_sha256":digest.hexdigest()}


def paired(a,b,stage):
    a,b=a.set_index(KEY).sort_index(),b.set_index(KEY).sort_index()
    check(stage+":matching_keys",a.index.equals(b.index))
    check(stage+":matching_seeds",a.derived_sampling_seed.equals(b.derived_sampling_seed))
    common=audit.safe(a)&audit.safe(b)
    counts={"stage":stage,"requests":len(a),"common_success":int(common.sum()),"coverage":common.mean(),
            "lambda099_gained":int((~audit.safe(a)&audit.safe(b)).sum()),"lambda099_lost":int((audit.safe(a)&~audit.safe(b)).sum()),
            "both_failed":int((~audit.safe(a)&~audit.safe(b)).sum())}
    metrics=[]
    for field in PHYSICAL:
        x,y=a.loc[common,field],b.loc[common,field]
        metrics.append({"stage":stage,"metric":field,"lambda095_mean":x.mean(),"lambda099_mean":y.mean(),
                        "percent_change":100*(y.mean()/x.mean()-1),"lambda099_lower_fraction":(y<x-1e-9).mean()})
    for coord in audit.EVAL:
        x,y=audit.quality(a.loc[common],coord),audit.quality(b.loc[common],coord)
        xm,ym=x.groupby(level="preference_key").mean().mean(),y.groupby(level="preference_key").mean().mean()
        metrics.append({"stage":stage,"metric":"quality_"+coord,"lambda095_mean":xm,"lambda099_mean":ym,
                        "percent_change":100*(ym/xm-1),"lambda099_lower_fraction":(y<x-1e-9).mean()})
    return counts,metrics


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    audit.OUT=OUT
    for flow,scale in [("excess",368.3143),("raw",1089.15)]:
        audit.EVAL[flow+"_owa"]={"fields":["flow_excess_objective" if flow=="excess" else "flow_time_objective","reconfiguration_cost","owa_q05"],
                               "scales":[scale,353.27,21.03145448655653]}
    evaluation={"coordinates":audit.EVAL,"scalarizer":"normalized_augmented_tchebycheff_v1","rho":.05,"hv_reference":[1,1,1],
                "workload":{"mode":"geometric_owa","q":.5,"weights":[32/63,16/63,8/63,4/63,2/63,1/63]},"decode":"sampled","temperature":1}
    (OUT/"evaluation_config.json").write_text(json.dumps(evaluation,indent=2),encoding="utf-8")
    configs,summaries,tests,vals,diags,trains,updates={},{},{},{},{},{},{}
    overview,curves,optimization,convergence,models,failures,endpoints,source_evidence=[],[],[],[],[],[],[],[]
    test_manifest=ROOT/"data/manifests/v2/test/manifest.json";val_manifest=ROOT/"data/manifests/v2/validation/manifest.json"
    SOURCES.update([test_manifest,val_manifest]);test_hash=dataset_manifest_snapshot(test_manifest)["sha256"];val_hash=dataset_manifest_snapshot(val_manifest)["sha256"]
    for model,path in RUNS.items():
        config=js(path/"config.json");summary=js(path/"summary.json");configs[model]=config;summaries[model]=summary
        train,update,log=[csv(path/name) for name in ["train_log.csv","update_log.csv","validation_log.csv"]]
        trains[model],updates[model]=train,update
        lam=.95 if model=="lambda095" else .99
        check(model+":gamma_lambda",config["ppo"]["gamma"]==1 and config["ppo"]["gae_lambda"]==lam)
        check(model+":2000_episodes",len(train)==summary["episodes"]==2000 and train.episode.tolist()==list(range(1,2001)))
        check(model+":100_updates",len(update)==100 and update.episode_count.sum()==2000)
        check(model+":native_training_reward",train.reward_identity_error.abs().max()<1e-8 and not train.sampling_truncated.any())
        fields=["policy_loss","value_loss","entropy","approx_kl","clip_fraction","gradient_norm","pre_update_explained_variance"]
        check(model+":finite_updates",np.isfinite(update[fields]).all().all())
        check(model+":validation_schedule",log.episode.tolist()==list(range(100,2001,100)) and log.evaluation_complete.all() and (log.completion_coverage==1).all())
        best=int(summary["checkpoint_selection"]["best_episode"])
        selected=log[log.checkpoint_eligible].sort_values(["completion_rate","preference_balanced_quality_score","episode"],ascending=[False,True,True]).iloc[0]
        check(model+":selection_rebuilt",int(selected.episode)==best)
        provenance=summary["final_sampled"]["provenance"]
        check(model+":effective_config_hash",effective_config_snapshot(config)["sha256"]==provenance["effective_config_sha256"])
        source=source_hash(provenance["git"]["commit"])
        check(model+":git_python_source_rebuilt",source["source_state_sha256"]==provenance["source_state_sha256"] and source["python_file_count"]==provenance["source_file_count"])
        source_evidence.append({"model":model,**source})
        for role in ["best","last"]:
            file=path/(role+"_checkpoint.pt");SOURCES.add(file);payload=torch.load(file,map_location="cpu",weights_only=False)
            spec=infer_checkpoint_network_spec(payload);meta=payload["metadata"];weights=network_weights_sha256(payload["network"])
            check(model+":"+role+":weights",weights==meta["network_weights_sha256"])
            check(model+":"+role+":episode",meta["checkpoint_episode"]==(best if role=="best" else 2000))
            check(model+":"+role+":network_identity",spec["observation_schema_version"]==12 and spec["global_objective_encoding"]=="log2_v1"
                  and spec["actor_head_variant"]=="shared_preference" and spec["workload_mode"]=="geometric_owa" and spec["workload_q"]==.5
                  and np.allclose(spec["objective_scales"],audit.EVAL["excess_owa"]["scales"]))
            if role=="best":
                check(model+":best_file_hash",hashlib.sha256(file.read_bytes()).hexdigest()==provenance["checkpoint_sha256"])
                check(model+":best_provenance_weights",weights==provenance["network_weights_sha256"])
            models.append({"model":model,"role":role,"episode":int(meta["checkpoint_episode"]),"weights_sha256":weights})
            del payload
        test=audit.read(path/"final_sampled_instance_metrics.csv",model,model+":test")
        val=audit.read(path/"sampled_validation_instance_metrics.csv",model,model+":validation")
        diag=audit.read(path/"sampled_diagnostic_instance_metrics.csv",model,model+":diagnostic")
        SOURCES.update([path/"final_sampled_instance_metrics.csv",path/"sampled_validation_instance_metrics.csv",path/"sampled_diagnostic_instance_metrics.csv"])
        tests[model],vals[model],diags[model]=test,val,diag
        check(model+":test_grid",len(test)==3960 and test.instance_id.nunique()==20 and test.preference_key.nunique()==66
              and not test.duplicated(KEY).any() and (test.groupby(KEY[:2]).size()==3).all())
        check(model+":validation_grid",len(val)==39000 and not val.duplicated(["validation_episode",*KEY]).any() and (val.groupby("validation_episode").size()==1950).all())
        check(model+":test_failure_count",int((~audit.safe(test)).sum())==summary["final_sampled"]["task_failed_count"])
        check(model+":dataset_manifest",set(test.dataset_manifest_sha256)=={test_hash}=={provenance["dataset_manifest_sha256"]} and set(val.dataset_manifest_sha256)=={val_hash})
        check(model+":diagnostic_disjoint",diag.instance_id.nunique()==49 and not set(diag.instance_id)&set(val.instance_id))
        for stage,data in [("test",test),("validation",val),("diagnostic",diag)]:
            check(model+":"+stage+":safety",data.physical_safety_pass.all() and (data.schedule_violation_count==0).all()
                  and (data.maximum_worker_fatigue<=data.safe_fatigue_limit+1e-9).all() and not data.sampling_truncated.any())
            check(model+":"+stage+":quality",np.allclose(audit.quality(data[audit.safe(data)],"excess_owa"),data.loc[audit.safe(data),"preference_quality_score"],atol=1e-8))
            check(model+":"+stage+":reward",np.allclose(data.training_cumulative_reward,data.operation_progress-data.initial_progress+
                  data.initial_preference_quality_score-data.actual_preference_quality_score-data.terminal_failure_penalty_applied,atol=1e-8))
            check(model+":"+stage+":failure_penalty",np.allclose(data.terminal_failure_penalty_applied,2*data.task_failed.astype(int)))
        check(model+":test_derived_seeds",all(int(r.derived_sampling_seed)==derive_evaluation_sampling_seed(int(r.sampling_seed),r.instance_id,r.preference_key) for r in test.itertuples()))
        for stage,data in [("best_test",test),("best_validation",val[val.validation_episode==best]),("last_validation",val[val.validation_episode==2000]),("last_diagnostic",diag[diag.validation_episode==2000])]:
            overview.append({"model":model,"stage":stage,"episode":best if stage.startswith("best") else 2000,**audit.aggregate(data)})
            failed=data[~audit.safe(data)].copy();failed["model"]=model;failed["stage"]=stage;failures.extend(failed.to_dict("records"))
            for name,pref in zip(["Flow","Cost","OWA"],PREF):
                endpoints.append({"model":model,"stage":stage,"endpoint":name,**audit.aggregate(data[np.isclose(data[pref],1)])})
        for episode,data in val.groupby("validation_episode"):
            row={"model":model,"episode":int(episode),**audit.aggregate(data),"mean_return":data.training_cumulative_reward.mean(),
                 "mean_progress":data.operation_progress.mean(),"mean_actual_quality":data.actual_preference_quality_score.mean(),
                 "mean_failure_penalty":data.terminal_failure_penalty_applied.mean()}
            for name,pref in zip(["Flow","Cost","OWA"],PREF):row[name+"_completion"]=audit.safe(data[np.isclose(data[pref],1)]).mean()
            curves.append(row)
        update=update.assign(model=model,completed_episodes=update.episode_end+1)
        update.to_csv(OUT/(model+"_update_metrics.csv"),index=False,encoding="utf-8-sig")
        for phase,group in [("early_1_400",update[update.completed_episodes<=400]),("middle_401_1400",update[(update.completed_episodes>400)&(update.completed_episodes<=1400)]),
                            ("late_1401_2000",update[update.completed_episodes>1400])]:
            optimization.append({"model":model,"phase":phase,"updates":len(group),
                                 **{field:group[field].mean() for field in fields+["return_std","advantage_std","gradient_clipped_fraction"]},
                                 "explained_variance_min":group.pre_update_explained_variance.min(),"value_loss_max":group.value_loss.max(),
                                 "KL_max":group.approx_kl.max(),"clip_fraction_max":group.clip_fraction.max()})
        check(model+":constant_lr",np.allclose(log.learning_rate,.0001))
    ca,cb=deepcopy(configs["lambda095"]),deepcopy(configs["lambda099"])
    cb["experiment_name"]=ca["experiment_name"];cb["ppo"]["gae_lambda"]=.95
    check("config_only_lambda_and_name",effective_config_snapshot(ca)["sha256"]==effective_config_snapshot(cb)["sha256"])
    check("identical_training_sequence",trains["lambda095"][["instance_id","seed","preference_key","pressure_type","cost_profile"]].equals(
          trains["lambda099"][["instance_id","seed","preference_key","pressure_type","cost_profile"]]))
    check("identical_validation_subsets",js(RUNS["lambda095"]/"validation_subsets.json")==js(RUNS["lambda099"]/"validation_subsets.json"))
    changed_source=subprocess.check_output(["git","diff","--name-only",source_evidence[0]["commit"],source_evidence[1]["commit"]],text=True).splitlines()
    check("code_change_is_launcher_only",[p for p in changed_source if p.endswith('.py')]==['scripts/run_schema12_screening.py'])
    counts,metrics,hvs,budgets=[],[],[],[]
    for stage,a,b in [("best_test",tests["lambda095"],tests["lambda099"]),
                      ("selected_validation",vals["lambda095"][vals["lambda095"].validation_episode==1900],vals["lambda099"][vals["lambda099"].validation_episode==700]),
                      ("last_validation",vals["lambda095"][vals["lambda095"].validation_episode==2000],vals["lambda099"][vals["lambda099"].validation_episode==2000]),
                      ("last_diagnostic",diags["lambda095"][diags["lambda095"].validation_episode==2000],diags["lambda099"][diags["lambda099"].validation_episode==2000])]:
        count,measures=paired(a,b,stage);counts.append(count);metrics.extend(measures)
    for model,f in tests.items():
        for coord,target in audit.EVAL.items():
            for iid,g in f.groupby("instance_id"):
                v=g.loc[audit.safe(g),target["fields"]].to_numpy()
                hvs.append({"model":model,"coordinate":coord,"instance_id":iid,"hv":hypervolume_3d(v/(v+target["scales"]))})
        chosen=f.assign(success=audit.safe(f),selection_quality=audit.quality(f,"excess_owa")).sort_values(
            ["success","selection_quality","sampling_repeat"],ascending=[False,True,True]).drop_duplicates(KEY[:2])
        budgets.append({"model":model,"budget":3,**audit.aggregate(chosen)})
    curve=pd.DataFrame(curves)
    for model,g in curve.groupby("model"):
        for phase,selected in [("first_100_700",g[g.episode<=700]),("late_1400_2000",g[g.episode>=1400]),("last_five_1600_2000",g[g.episode>=1600])]:
            convergence.append({"model":model,"phase":phase,"points":len(selected),
                                "quality_mean":selected.quality_excess_owa.mean(),"quality_std":selected.quality_excess_owa.std(ddof=0),
                                "quality_min":selected.quality_excess_owa.min(),"quality_max":selected.quality_excess_owa.max(),
                                "quality_linear_change_per100":np.polyfit(selected.episode,selected.quality_excess_owa,1)[0]*100,
                                "worst_pref_mean":selected.worst_preference_completion.mean(),"worst_pref_low":selected.worst_preference_completion.min(),
                                "cost_mean":selected.Cost_completion.mean(),"cost_low":selected.Cost_completion.min(),"overall_mean":selected.overall_completion.mean(),
                                "overall_linear_change_per100":np.polyfit(selected.episode,selected.overall_completion,1)[0]*100})
    milestones=[]
    for model,g in curve.groupby("model"):
        for name,condition in [("Q_le_035",g.quality_excess_owa<=.35),("Q_le_030",g.quality_excess_owa<=.30),
                               ("Q_le_028",g.quality_excess_owa<=.28),("worst_completion_ge_098",g.worst_preference_completion>=.98-1e-10),
                               ("worst_ge_098_and_Q_le_030",(g.worst_preference_completion>=.98-1e-10)&(g.quality_excess_owa<=.30))]:
            eligible=g[condition];milestones.append({"model":model,"criterion":name,"first_episode":int(eligible.episode.min()) if len(eligible) else None,"qualifying_validations":len(eligible)})
    correlations,transitions,levels,_=audit.preference_response(tests)
    a,b=[tests[m].set_index(KEY).sort_index() for m in RUNS];common=audit.safe(a)&audit.safe(b)
    clusters=[]
    for iid in a.index.get_level_values("instance_id").unique():
        own=a.index.get_level_values("instance_id")==iid;mask=common&own
        clusters.append({"instance_id":iid,"completion_delta":audit.safe(b.loc[own]).mean()-audit.safe(a.loc[own]).mean(),
                         "quality_delta":(audit.quality(b.loc[mask],"excess_owa")-audit.quality(a.loc[mask],"excess_owa")).mean(),
                         **{field+"_delta":(b.loc[mask,field]-a.loc[mask,field]).mean() for field in PHYSICAL}})
    cluster=pd.DataFrame(clusters).set_index("instance_id");rng=np.random.default_rng(20261015);draws=rng.integers(0,20,size=(20000,20));ci=[]
    for field in cluster:
        v=cluster[field].to_numpy();low,high=np.quantile(v[draws].mean(axis=1),[.025,.975]);ci.append({"metric":field,"mean_delta":v.mean(),"ci95_low":low,"ci95_high":high,"instance_clusters":20})
    all_checks=CHECKS+audit.CHECKS
    for name,rows in [("overview",overview),("paired_metrics",metrics),("paired_counts",counts),("validation_curve",curves),("optimization_phases",optimization),
                      ("convergence_phases",convergence),("milestones",milestones),("failures",failures),("endpoints",endpoints),("checkpoints",models),
                      ("best_of_three",budgets),("hypervolume_by_instance",hvs),("bootstrap",ci),("checks",all_checks)]:save(name,rows)
    SOURCES.update(audit.SOURCES-{Path(audit.__file__)})
    result={"runs":{k:p.as_posix() for k,p in RUNS.items()},"overview":overview,"paired_counts":counts,"paired_metrics":metrics,
            "hypervolume_mean":pd.DataFrame(hvs).groupby(["model","coordinate"]).hv.mean().unstack().to_dict("index"),
            "convergence":convergence,"optimization":optimization,"milestones":milestones,"endpoints":endpoints,"best_of_three":budgets,
            "source_reconstruction":source_evidence,"preference_correlations":correlations,"confidence_intervals":ci,"checks_passed":len(all_checks),
            "gamma":1.0,"lambdas":[.95,.99],"full_last_test_available":False,"new_policy_rollouts":0,
            "elapsed_hours":{k:s["elapsed_seconds"]/3600 for k,s in summaries.items()},
            "input_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}}
    (OUT/"comparison.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    plot(curve,updates)
    report(result)
    print(json.dumps({k:result[k] for k in ["checks_passed","overview","hypervolume_mean","convergence","optimization"]},ensure_ascii=False,indent=2))


def plot(curve,updates):
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    colors={"lambda095":"#2563eb","lambda099":"#d97706"}
    fig,axes=plt.subplots(2,2,figsize=(10,6.8),constrained_layout=True)
    for model,g in curve.groupby("model"):
        for ax,field,label in zip(axes.flat,["quality_excess_owa","worst_preference_completion","Cost_completion","overall_completion"],
                                 ["OWA quality loss (lower better)","Worst preference completion (%)","Cost completion (%)","Overall completion (%)"]):
            mult=1 if field.startswith("quality") else 100
            ax.plot(g.episode,g[field]*mult,marker=".",label=model,color=colors[model]);ax.set_xlabel("Training episodes");ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0,0].legend()
    for ext in ["png","svg"]:fig.savefig(OUT/f"convergence.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),constrained_layout=True)
    for model,frame in updates.items():
        for ax,field,label in zip(axes.flat,["pre_update_explained_variance","advantage_std","approx_kl","value_loss"],
                                 ["Pre-update explained variance","Raw GAE advantage std","Approximate KL","Value loss"]):
            ax.plot(frame.episode_end+1,frame[field].rolling(5,min_periods=1).mean(),label=model,color=colors[model]);ax.set_xlabel("Training episodes");ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0,0].legend();axes[1,1].set_yscale("log")
    for ext in ["png","svg"]:fig.savefig(OUT/f"optimization.{ext}",dpi=180)
    plt.close(fig)


def report(result):
    overview={(r['model'],r['stage']):r for r in result['overview']}
    a,b=overview['lambda095','best_test'],overview['lambda099','best_test']
    metrics={r['metric']:r for r in result['paired_metrics'] if r['stage']=='best_test'}
    count=next(r for r in result['paired_counts'] if r['stage']=='best_test')
    text='# OWA/log2：GAE λ=0.99 与 λ=0.95 收敛对照（2026-10-15）\n\n'
    text+='**结论：λ=0.99的best稍偏向完成率与Flow，但质量更差；在2000轮预算内没有表现出更好的综合收敛，后期优势估计与价值拟合波动明显增大。建议保留λ=0.95作为当前默认，保留λ=0.99的best作为完成率取向候选。**\n\n'
    text+='本分析读取已保存的模型、训练/验证/测试文件，不新增训练、策略回放或last完整测试。比较对象均为schema12、shared_head、excess、OWA(q=.5)、log2、seed11、gamma=1、2000轮/100次更新。\n\n'
    text+='## 1. 是否只改变了λ\n\n'
    text+=f"有效配置核对：除实验名称和gae_lambda(.95→.99)外完全一致，训练实例ID、生成种子、偏好key与场景顺序一致，验证子集完全一致。奖励、冻结尺度、网络、PPO其他超参数和每20轮更新保持匹配。共{result['checks_passed']}项核验通过。\n\n"
    text+='虽然provenance均标记git dirty=true，本次从各自记录的Git提交重建114个Python源文件的规范化哈希，两次均与保存的source_state精确匹配。两提交间唯一Python变化是启动器增加实验选项；训练、环境与网络源码一致。模型文件、权重哈希、schema/编码/workload身份及原配置均核验通过。这比仅看分支名更严格，但不宣称复原了所有第三方库或硬件执行细节。\n\n'
    text+='## 2. 相同选模规则下的best测试\n\n相同20实例×66偏好×3采样(3960请求)，逐请求核对derived seed。质量损失Q越低越好，HV越高越好，均使用共同OWA尺度(368.3143,353.27,21.03145448655653)。\n\n| 指标 | λ=.95 | λ=.99 |\n|---|---:|---:|\n'
    for title,field,scale in [('best轮次','episode',1),('成功请求','successes',1),('失败请求','failures',1),('整体完成率(%)','overall_completion',100),
                              ('最差偏好完成率(%)','worst_preference_completion',100),('excess质量损失','quality_excess_owa',1),('raw质量损失','quality_raw_owa',1)]:
        text+=f"| {title} | {a[field]*scale:.6f} | {b[field]*scale:.6f} |\n"
    for coordinate in audit.EVAL:
        text+=f"| {coordinate}逐实例平均pooled HV | {result['hypervolume_mean']['lambda095'][coordinate]:.6f} | {result['hypervolume_mean']['lambda099'][coordinate]:.6f} |\n"
    text+=f"\n共同成功{count['common_success']}/3960，覆盖{100*count['coverage']:.3f}%；λ=.99新增成功{count['lambda099_gained']}、失去成功{count['lambda099_lost']}。完成率提高0.0758个百分点，20实例簇bootstrap的95%区间约[−0.1263,+0.3535]个百分点，不足以作为跨seed的稳定完成率收益。\n\n"
    text+='共同成功配对的目标比较：\n\n| 指标 | λ=.95 | λ=.99 | 变化 |\n|---|---:|---:|---:|\n'
    for field in ['flow_time_objective','reconfiguration_cost','owa_q05','worker_load_max','worker_load_total','completed_reconfigurations','maximum_worker_fatigue','quality_excess_owa']:
        r=metrics[field];text+=f"| {field} | {r['lambda095_mean']:.6f} | {r['lambda099_mean']:.6f} | {r['percent_change']:+.2f}% |\n"
    text+='\n新best更多重构、劳动和个人负担更高，Flow略小；是不同的保守/成本折中，而不是三个目标全面进步。配对Q差的实例簇95%区间约[+0.04068,+0.05937]，在本测试分布上质量退步方向明确；该区间仍不能替代训练多seed。HV每实例独立计算后平均，候选来自66偏好×3采样。\n\n'
    text+='## 3. 收敛速度要分开看完成率和质量\n\n'
    text+='| 描述性阈值 | λ=.95首次达到 | λ=.99首次达到 |\n|---|---:|---:|\n'
    milestones={(r['model'],r['criterion']):r for r in result['milestones']}
    for name in ['Q_le_035','Q_le_030','Q_le_028','worst_completion_ge_098','worst_ge_098_and_Q_le_030']:
        text+=f"| {name} | {milestones['lambda095',name]['first_episode']} | {milestones['lambda099',name]['first_episode']} |\n"
    text+='\n这些阈值是事后描述曲线，不是新选模规则或已验证早停策略。λ=.99更早达到高最差偏好完成率，但质量下降更慢；首次同时满足98%最差偏好与Q≤.30出现在ep900，却只维持一个验证点。ep700选出的best不表示ep700后模型停止学习，也不代表质量已最优。\n\n'
    text+='λ=.99在ep900的Q=.299751、ep1200的Q=.278457低于best(ep700)的.343281，但最差偏好完成率分别为98%、90%，低于ep700的99.333%，因此按既定严格字典序不替换best。只有best/last权重被保存，中间轮次日志不能冒充其完整测试性能。\n\n'
    text+='## 4. 后期是否进入稳定平台\n\n'
    text+='ep1400–2000七个验证点：\n\n| 模型 | Q均值 | Q标准差 | 最差偏好均值 | 整体完成率均值 | Q每100轮线性趋势 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in result['convergence']:
        if r['phase']=='late_1400_2000':
            text+=f"| {r['model']} | {r['quality_mean']:.6f} | {r['quality_std']:.6f} | {100*r['worst_pref_mean']:.3f}% | {100*r['overall_mean']:.3f}% | {r['quality_linear_change_per100']:+.6f} |\n"
    text+='\nλ=.95后期Q大致位于较低的平台，但完成率仍有波动；λ=.99从ep1200的低Q回升，后段质量更差且波动更大，不能认为已稳定收敛到更优策略。七点线性趋势只是描述，不构成长期外推或收敛证明。\n\n'
    text+='相同ep2000的last及不参与选模的49实例诊断：\n\n| 阶段/模型 | 成功数 | 最差偏好完成率 | Cost完成率 | 共同OWA质量损失 |\n|---|---:|---:|---:|---:|\n'
    for stage in ['last_validation','last_diagnostic']:
        for model in RUNS:
            r=overview[model,stage];text+=f"| {stage}/{model} | {r['successes']}/{r['requests']} | {100*r['worst_preference_completion']:.3f}% | {100*r['cost_completion']:.3f}% | {r['quality_excess_owa']:.6f} |\n"
    text+='\n因此性能差异并非仅由λ=.99选中了较早best造成：相同2000轮的质量、整体完成率和独立验证也较差。纯Cost的last确实改善至150/150，但失败转移到了Flow和混合偏好，不能只看Cost端点宣称整体稳定性提升。last尚无完整测试，不将其验证值当成测试值。\n\n'
    text+='![完成率与质量收敛](convergence.png)\n\n## 5. 优势估计与优化诊断\n\n'
    text+='最后600训练轮、30次更新的均值：\n\n| 指标 | λ=.95 | λ=.99 |\n|---|---:|---:|\n'
    opt={r['model']:r for r in result['optimization'] if r['phase']=='late_1401_2000'}
    for field in ['advantage_std','return_std','pre_update_explained_variance','value_loss','approx_kl','clip_fraction','gradient_norm','gradient_clipped_fraction']:
        text+=f"| {field} | {opt['lambda095'][field]:.6f} | {opt['lambda099'][field]:.6f} |\n"
    text+='\nλ=.99的原始优势标准差约3.05倍，value loss约4.91倍，预更新explained variance从.951降至.731；KL、clip fraction和梯度范数均值接近。更符合长GAE目标增加估计波动、critic更难拟合的迹象，没有明显证据指向单纯KL或更新幅度爆炸。\n\n'
    text+='GAE系数为γλ，γ=1时100决策后的TD残差权重由.95^100≈.00592增至.99^100≈.36603。它扩大了远期误差传播，也可能扩大噪声传播。上表的critic目标随λ改变，不能把value loss的倍数单独当作因果证明；应结合优势分布、验证质量回退和多seed一起判断。两组学习率均保持1e−4，未发生学习率衰减，不能把差异归因于实际用了不同LR调度。\n\n'
    text+='![优化与价值拟合](optimization.png)\n\n## 6. 端点、失败和采样预算\n\n'
    text+='| 模型/端点 | 成功请求 | Flow | Cost | OWA负担 |\n|---|---:|---:|---:|---:|\n'
    for r in result['endpoints']:
        if r['stage']=='best_test':text+=f"| {r['model']}/{r['endpoint']} | {r['successes']}/{r['requests']} | {r['flow_time_objective']:.2f} | {r['reconfiguration_cost']:.2f} | {r['owa_q05']:.2f} |\n"
    text+='\nλ=.99三个纯端点均60/60，但Cost、OWA端点的目标值更差。新best的5次失败均为horizon，其中3次balanced、2次重构瓶颈，疲劳峰值约.24–.75；即时物理安全仍全部通过。完整偏好响应数据保存在preference_summary.csv、preference_fixed_ratio.csv，响应不等于绝对质量。\n\n'
    text+='两组对1320个实例/偏好单元，三次采样均至少一次成功。共同OWA标准成功优先择优后的质量损失：\n\n| 模型 | 三次择优Q |\n|---|---:|\n'
    for r in result['best_of_three']:text+=f"| {r['model']} | {r['quality_excess_owa']:.6f} |\n"
    text+='\n同样推断预算下λ=.95仍更优；择优结果不能与单次请求完成率混淆。\n\n## 7. 当前决策\n\n'
    text+='保留gamma=1、lambda=.95作为当前OWA/log2基线，不直接用.99替换。lambda=.99的best可作为较高完成率、较高成本/负担的候选保留；本次结果不支持仅靠加长训练预算便能追上.95。若后续确需继续研究λ，优先用相同seed配对复核已有差异，再决定是否调整价值拟合或其他训练项，避免同时改尺度和奖励。\n\n'
    text+='这是一组受控单训练seed比较，不能推导lambda=.99普遍更差。正式训练耗时分别约20.95h、23.70h；时间也受硬件负载和策略轨迹影响。\n\n'
    text+='复现：`python -m analysis.studies.schema12_gae099_convergence_20261015`。comparison.json含来源哈希及Git源码重建；checks.csv、paired_metrics.csv、validation_curve.csv、optimization_phases.csv、milestones.csv、bootstrap.csv提供可核查明细。\n'
    (OUT/'report.md').write_text(text,encoding='utf-8')


if __name__=="__main__":main()
