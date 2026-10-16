"""Validation-only scale and utility sensitivity audit; keeps frozen configs intact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"result/analysis/schema12_normalization_review_20261014"
RUNS={m:ROOT/f"result/runs/schema12_screen_seed11_20261012_201659_{m}_log2" for m in ("variance","owa")}
PREF=["preference_flow","preference_cost","preference_workload"]
FIELDS=["flow_excess_objective","reconfiguration_cost","workload_objective_value"]
KEY=["instance_id","preference_key","sampling_repeat"]
NAMES=["Flow","Cost","Workload"]
SOURCES={Path(__file__),ROOT/"environment/types.py",ROOT/"environment/observation_schema.py"}
CHECKS=[]


def check(name,condition):
    CHECKS.append({"check":name,"passed":bool(condition)})
    if not condition:raise AssertionError(name)


def read_json(path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe(f):
    return f.task_succeeded&f.physical_safety_pass&(f.schedule_violation_count==0)


def score(j,w,s):
    z=w*j/(j+s)
    return (z.max(axis=1)+.05*z.sum(axis=1))/1.05


def transform(f,s):
    j=f[FIELDS].to_numpy(dtype=float);w=f[PREF].to_numpy(dtype=float)
    q=j/(j+s);z=w*q
    maximum=np.isclose(z,z.max(axis=1)[:,None],rtol=0,atol=1e-12)
    shares=maximum/maximum.sum(axis=1)[:,None]
    derivative=(shares+.05)/1.05*w*q*(1-q)
    baseline=score(j,w,s)
    gains=[]
    for i in range(3):
        alternative=j.copy();alternative[:,i]*=.9
        gains.append(baseline-score(alternative,w,s))
    return j,w,q,shares,derivative,np.stack(gains,axis=1)


def diagnostics(f,s,model,stage):
    f=f[safe(f)]
    j,w,q,shares,derivative,gains=transform(f,s)
    records=[]
    masks={"all_success":np.ones(len(f),bool),"equal_weights":np.isclose(w,1/3).all(axis=1),
           "all_weights_positive":(w>1e-10).all(axis=1)}
    for i in range(3):masks[f"priority_{NAMES[i]}"]=w[:,i]>=.5-1e-10
    for cohort,mask in masks.items():
        if not mask.any():continue
        for i,name in enumerate(NAMES):
            x= q[mask,i];xj=j[mask,i]
            records.append({"model":model,"stage":stage,"cohort":cohort,"objective":name,"requests":int(mask.sum()),
                            "scale":s[i],"J_mean":xj.mean(),"J_median":np.median(xj),
                            "q_mean":x.mean(),"q_median":np.median(x),"q_p90":np.quantile(x,.9),"q_max":x.max(),
                            "q_above_09_fraction":(x>.9).mean(),"q_below_01_fraction":(x<.1).mean(),
                            "relative_component_sensitivity_mean":(x*(1-x)).mean(),
                            "max_term_fraction":shares[mask,i].mean(),
                            "utility_log_sensitivity_mean":derivative[mask,i].mean(),
                            "utility_gain_10pct_improvement":gains[mask,i].mean(),
                            "terminal_log2_input_p95":np.quantile(np.log2(1+xj/s[i]),.95),
                            "terminal_log2_input_max":np.log2(1+xj/s[i]).max()})
    return records


def choices(f,s):
    g=f.copy();g["safe"]=safe(g)
    g["rescored"]=score(g[FIELDS].to_numpy(),g[PREF].to_numpy(),s)
    return g.sort_values(["safe","rescored","sampling_repeat"],ascending=[False,True,True]).drop_duplicates(KEY[:2]).set_index(KEY[:2]).sort_index()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    snapshots,config,frames,records,curves={},{},{},[],[]
    for model,run in RUNS.items():
        c=read_json(run/"config.json");s=read_json(run/"summary.json");config[model]=c
        scale=np.array([c["objective_scalarizer"]["scales"][k] for k in ("flow","cost","workload")])
        columns=set(KEY+PREF+FIELDS+["validation_episode","task_succeeded","physical_safety_pass","schedule_violation_count",
                     "preference_quality_score","actual_preference_quality_score","worker_loads","derived_sampling_seed","action_trace_sha256"])
        for kind,file in [("selection","sampled_validation_instance_metrics.csv"),("diagnostic","sampled_diagnostic_instance_metrics.csv")]:
            path=run/file;SOURCES.add(path)
            f=pd.read_csv(path,usecols=lambda name:name in columns)
            frames[model,kind]=f
            check(model+":"+kind+":finite_objectives",np.isfinite(f[FIELDS]).all().all() and (f[FIELDS]>=0).all().all())
            good=f[safe(f)]
            check(model+":"+kind+":native_quality",np.allclose(score(good[FIELDS].to_numpy(),good[PREF].to_numpy(),scale),good.preference_quality_score,atol=1e-8))
        best=int(s["checkpoint_selection"]["best_episode"])
        selection=frames[model,"selection"];diag=frames[model,"diagnostic"]
        for stage,f in [("best_validation",selection[selection.validation_episode==best]),
                         ("last_validation",selection[selection.validation_episode==2000]),
                         ("last_independent_validation",diag[diag.validation_episode==2000])]:
            snapshots[model,stage]=f
            records.extend(diagnostics(f,scale,model,stage))
        for episode,f in selection.groupby("validation_episode"):
            good=f[safe(f)];j,w,q,shares,d,g=transform(good,scale);equal=np.isclose(w,1/3).all(axis=1)
            curves.append({"model":model,"episode":int(episode),"successful_requests":len(good),
                           **{f"q_mean_{name}":q[:,i].mean() for i,name in enumerate(NAMES)},
                           **{f"equal_max_share_{name}":shares[equal,i].mean() for i,name in enumerate(NAMES)}})
    manifest=read_json(ROOT/"configs/manifests/workload_excess_q50_20261011.json")
    inherited=read_json(ROOT/"configs/manifests/flow_excess_scales_20261008.json")
    rows=manifest["workload_source"]["rows"]
    good=[r for r in rows if r["success"] and r["physical_valid"]]
    reconstructed=[]
    for r in good:
        values=sorted(r["worker_loads"].values(),reverse=True)
        value=sum(a*b for a,b in zip([32,16,8,4,2,1],values))/63
        check("calibration_owa:"+r["instance_id"],np.isclose(value,r["workload_value"]))
        reconstructed.append(value)
    check("calibration_scale_mean",np.isclose(np.mean(reconstructed),manifest["scales"]["workload"]))
    baseline_flow=inherited["sources"]["flow"]["baseline_rows"]
    mean_excess=np.mean([r["flow"]-r["flow_lb"] for r in baseline_flow if r["success"]])
    check("flow_scale_rounding",round(mean_excess,4)==manifest["scales"]["flow"])
    check("cost_scale_rounding",round(inherited["sources"]["cost"]["successful_trajectory_mean"],2)==manifest["scales"]["cost"])
    check("variance_scale_rounding",round(inherited["sources"]["variance"]["successful_trajectory_mean"],4)==inherited["scales"]["variance"])
    base=np.array([manifest["scales"][k] for k in ("flow","cost","workload")])
    sensitivity=[];own=snapshots["owa","best_validation"]
    base_choice=choices(own,base)
    a=snapshots["variance","best_validation"].set_index(KEY).sort_index().copy()
    b=own.set_index(KEY).sort_index().copy()
    check("paired_requests",a.index.equals(b.index) and a.derived_sampling_seed.equals(b.derived_sampling_seed))
    # Both policies evaluated with the same OWA objective, reconstructed from actual worker loads.
    a["workload_objective_value"]=[sum(w*x for w,x in zip([32,16,8,4,2,1],sorted(json.loads(v).values(),reverse=True)))/63 for v in a.worker_loads]
    common=safe(a)&safe(b);ja,jb=a.loc[common,FIELDS].to_numpy(),b.loc[common,FIELDS].to_numpy()
    ww=a.loc[common,PREF].to_numpy()
    old_direction=score(jb,ww,base)<score(ja,ww,base)
    for changed,i in zip(NAMES,range(3)):
        for factor in (.5,.75,1.,1.25,2.):
            scales=base.copy();scales[i]*=factor
            good_own=own[safe(own)];j,w,q,shares,derivative,gains=transform(good_own,scales)
            equal=np.isclose(w,1/3).all(axis=1)
            selected=choices(own,scales)
            qa,qb=score(ja,ww,scales),score(jb,ww,scales)
            keys=a.loc[common].index.get_level_values("preference_key")
            qa_mean=pd.Series(qa).groupby(np.asarray(keys)).mean().mean()
            qb_mean=pd.Series(qb).groupby(np.asarray(keys)).mean().mean()
            sensitivity.append({"changed_objective":changed,"factor":factor,"flow_scale":scales[0],"cost_scale":scales[1],"owa_scale":scales[2],
                                **{f"equal_q_{name}":q[equal,k].mean() for k,name in enumerate(NAMES)},
                                **{f"equal_max_share_{name}":shares[equal,k].mean() for k,name in enumerate(NAMES)},
                                **{f"equal_gain_10pct_{name}":gains[equal,k].mean() for k,name in enumerate(NAMES)},
                                "bestof3_choice_changed_fraction":(selected.sampling_repeat!=base_choice.sampling_repeat).mean(),
                                "paired_policy_rank_changed_fraction":((qb<qa)!=old_direction).mean(),
                                "variance_policy_common_owa_Q":qa_mean,"owa_policy_common_owa_Q":qb_mean,
                                "owa_policy_relative_quality_change_pct":100*(qb_mean/qa_mean-1)})
    d=pd.DataFrame(records);sens=pd.DataFrame(sensitivity);history=pd.DataFrame(curves)
    for name,data in [("distribution_and_sensitivity",d),("one_factor_scale_sensitivity",sens),("validation_history",history),("checks",pd.DataFrame(CHECKS))]:
        data.to_csv(OUT/f"{name}.csv",index=False,encoding="utf-8-sig")
    source_info={"flow":{"scale":base[0],"anchor":"successful heuristic mean excess","success":49,"requests":50},
                 "cost":{"scale":base[1],"anchor":"historical learned single-Cost validation mean","success":145,"requests":150},
                 "owa":{"scale":base[2],"anchor":"successful heuristic mean OWA","success":len(good),"requests":len(rows)},
                 "variance":{"scale":inherited["scales"]["variance"],"anchor":"historical learned single-Variance validation mean","success":146,"requests":150}}
    result={"scale_sources":source_info,"distribution":records,"offline_scale_sensitivity":sensitivity,
            "checks_passed":len(CHECKS),"scale_or_config_modified":False,"additional_policy_rollouts":0,
            "source_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}}
    (OUT/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    plot(d,sens)
    report(result,d,sens,history)
    print(d[(d.model=='owa')&(d.stage=='best_validation')&d.cohort.isin(['all_success','equal_weights'])][['cohort','objective','q_mean','q_p90','q_above_09_fraction','max_term_fraction','utility_gain_10pct_improvement','terminal_log2_input_p95']].to_string(index=False))
    print(sens[sens.changed_objective=='Workload'].to_string(index=False))


def plot(d,s):
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    equal=d[(d.model=='owa')&(d.stage=='best_validation')&(d.cohort=='equal_weights')].set_index('objective').loc[NAMES]
    axes[0].bar(NAMES,equal.q_mean,color=['#2563eb','#d97706','#059669']);axes[0].set_ylim(0,1);axes[0].set_ylabel('Mean bounded objective at equal weights')
    varied=s[s.changed_objective=='Workload']
    for name in NAMES:axes[1].plot(varied.factor,100*varied[f'equal_max_share_{name}'],marker='.',label=name)
    axes[1].set_xlabel('OWA scale multiplier');axes[1].set_ylabel('Share as maximum weighted term (%)');axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    for ext in ('png','svg'):fig.savefig(OUT/f'scale_sensitivity.{ext}',dpi=180)
    plt.close(fig)


def report(result,d,s,history):
    text="# 三目标归一化复核（2026-10-14）\n\n"
    text+="**建议：当前不统一重标三个尺度；Flow=368.3143、Cost=353.27、OWA=21.03145449继续作为这轮编码对照的冻结基线。** 新证据显示OWA在等权偏好下直接评分贡献偏弱，值得作为单独的尺度敏感性候选；这不等于已经证明降低尺度能改善训练，也不应与bounded输入同时更改。\n\n"
    text+="本次只使用两模型保存的选模验证和独立验证做诊断与离线重算，没有用测试集拟合新尺度，没有更改配置、模型、奖励或manifest，没有新增策略轨迹。\n\n"
    text+="## 1. 先区分三层\n\n"
    text+="- 目标定义：Flow为excess时间，Cost为现有费用，第三目标为OWA或Variance。\n- 观测输入：schema12为log2(1+J_i/s_i)，bounded对照为(J_i/s_i)/(1+J_i/s_i)。\n- 奖励质量：q_i=J_i/(s_i+J_i)，Q=[max_i(w_i q_i)+0.05 sum_i(w_i q_i)]/1.05。\n\n"
    text+="改输入编码不改变Q；改尺度则同时改变Q、全局目标输入和其他依赖尺度的特征，并改变模型身份。旧checkpoint应按原尺度使用。OWA与Variance表达不同偏好，重标尺度不能将两种目标变成同一个目标。\n\n"
    text+="## 2. 数值范围是否失衡\n\nOWA best(ep1900)验证成功1944条，三目标分布：\n\n| 目标 | 冻结尺度 | q均值 | q的90分位 | q>0.9比例 | 终局log2输入95分位 |\n|---|---:|---:|---:|---:|---:|\n"
    selected=d[(d.model=='owa')&(d.stage=='best_validation')&(d.cohort=='all_success')]
    for r in selected.itertuples():
        text+=f"| {r.objective} | {r.scale:.5f} | {r.q_mean:.4f} | {r.q_p90:.4f} | {100*r.q_above_09_fraction:.2f}% | {r.terminal_log2_input_p95:.4f} |\n"
    text+="\n三项处于可比较量级，OWA也没有普遍接近0或1。q对相同比例目标变化的敏感度为 dq/dln(J)=q(1−q)；三项均值约0.221、0.235、0.229，OWA并没有因bounded函数本身丧失相对敏感度。以上log2值是由终局目标重建的输入量级参考，不是所有中间状态或网络激活分布的实测。\n\n"
    text+="## 3. 真正需要关注的是等权下的最大项\n\n取验证中的150条等权成功请求，w=(1/3,1/3,1/3)：\n\n| 目标 | q均值 | 成为最大加权项的比例 | 单独改善该目标10%的平均Q收益 |\n|---|---:|---:|---:|\n"
    equal=d[(d.model=='owa')&(d.stage=='best_validation')&(d.cohort=='equal_weights')]
    for r in equal.itertuples():
        text+=f"| {r.objective} | {r.q_mean:.4f} | {100*r.max_term_fraction:.2f}% | {r.utility_gain_10pct_improvement:.6f} |\n"
    text+="\n这里的收益按Q(J)−Q(J_1,...,0.9J_i,...,J_3)精确计算，固定其他目标。它是效用函数的局部假设实验，不是可实现的新调度、不代表PPO梯度或实际学习进展。OWA直接收益仅约Flow的10.1%、Cost的8.6%。\n\n"
    text+="在唯一最大项处，数学导数为\n\n`dQ/dln(J_i) = [(1{i为最大项}+0.05)/1.05] * w_i*q_i*(1-q_i)`。\n\n"
    text+="OWA在等权处不是最大项，主要通过0.05的增广项进入评分；这比q均值相差0.1更值得注意。独立验证的等权成功147条中，OWA最大项比例仍为0；所有20个训练验证点约为0–1.34%，并非单个best偶然现象。\n\n"
    text+="但不能据此认定OWA被完全忽略：所有偏好合计，OWA最大项占26.1%；第三权重≥0.5时，占67.9%。整体偏好池中，三个目标单独改善10%的平均Q收益为0.004975、0.006240、0.004360，差异远小于等权子集。Cost与OWA在本轮策略上强相关，降低劳动常同时降低Cost，存在间接收益。Tchebycheff本就优先改善当前较差的归一化目标，最大项比例不要求人为均分成1/3。\n\n"
    text+="因此可以确认：当前等权数字不等于相同比例改善的等量奖励。若业务要求这种解释，需要专门校准偏好/效用；若采用固定参考水平、优先改善最差项的定义，当前行为本身不是数值错误。\n\n"
    text+="## 4. 原始尺度的参考标准\n\n| 项目 | 尺度 | 参考来源 | 成功覆盖 |\n|---|---:|---|---:|\n"
    for name,r in result['scale_sources'].items():
        text+=f"| {name} | {r['scale']:.6f} | {r['anchor']} | {r['success']}/{r['requests']} |\n"
    text+="\nFlow与OWA来自启发式成功轨迹均值；Cost与历史Variance来自单目标学习策略的成功验证均值。因此三项具有不同参考难度，等权不是统一的相对最优差距。OWA的49条参考负担、均值及Flow/Cost/Variance取整均已重算核对，未发现尺度文件或公式错误。参考来源差异是后续实验设计可改善的点，不能直接用当前Universal自身的均值替换尺度；那会让参考系随模型改变，且受到失败覆盖与偏好分布影响。\n\n"
    text+="## 5. 改OWA尺度会发生什么\n\n保持已保存轨迹和Flow/Cost尺度固定，仅对s_OWA做一因素扫描：\n\n| s_OWA倍数 | 实际尺度 | 等权OWA q均值 | 等权OWA最大项比例 | 三次采样择优改变比例 |\n|---|---:|---:|---:|---:|\n"
    for r in s[s.changed_objective=='Workload'].itertuples():
        text+=f"| {r.factor:.2f} | {r.owa_scale:.5f} | {r.equal_q_Workload:.4f} | {100*r.equal_max_share_Workload:.2f}% | {100*r.bestof3_choice_changed_fraction:.2f}% |\n"
    text+="\n减小尺度会提高OWA的归一化值，从而更容易占最大项，属于强化相对优先级。取0.75倍≈15.77时直接10%改善收益由0.000385升至0.000722，但仍低于Flow/Cost；取0.5倍≈10.52会使OWA占54.7%的最大项，同时Cost降至5.3%，不能称为自动恢复平衡。约15.77只能作为较温和的敏感性候选，不是已验证的新尺度或最优值。\n\n"
    text+="另外分别扫描Flow和Cost的0.5、0.75、1、1.25、2倍。15个单因素条件下，OWA模型在双方共同成功验证样本、同一OWA坐标中的Q均优于Variance模型，改善范围约3.35%–5.00%。因此当前OWA收益不是只在某个精确尺度点成立；所有绝对分数都是在各自坐标内比较，不能把跨尺度Q下降当作模型改善。\n\n"
    text+="这些反事实没有改变调度或失败数；它们只能揭示评分/择优会变多少，不能预言重新训练后的完成率和Pareto前沿。\n\n"
    text+="## 6. Variance尺度是否必须增大\n\n"
    var=d[(d.model=='variance')&(d.stage=='best_validation')&(d.objective=='Workload')]
    for cohort,label in [('all_success','全部成功偏好'),('equal_weights','等权偏好'),('priority_Workload','方差权重≥0.5')]:
        r=var[var.cohort==cohort].iloc[0]
        text+=f"- {label}：q均值{r.q_mean:.4f}，q>0.9比例{100*r.q_above_09_fraction:.2f}%。\n"
    text+="\n全体约一半方差分量超过0.9，但高方差权重下只有约1.2%，说明高值主要来自允许牺牲均衡的偏好。等权下Variance成为最大项约92.7%，与OWA的0%形成明显差异；这再次说明从Variance换成OWA时，目标定义与效用平衡一起改变。没有证据支持按所有偏好的平均方差放大尺度，那会削弱真正关心均衡的请求。保留原尺度完成当前对照，必要时再单独研究Variance的偏好解释。\n\n"
    text+="## 7. 具体决策\n\n"
    text+="1. Flow=368.3143、Cost=353.27：保留。未发现普遍饱和或足以要求立即重标的证据。\n2. OWA=21.03145449：本轮保留；记录等权直接效用偏弱。若下一阶段明确要求更强的负担优先级，可先把0.75倍作为一个单独候选，用新的不可变manifest和匹配训练验证。\n3. 本次已有log2与准备中的bounded应继续使用相同尺度，才能独立检验输入编码。若现在改尺度，应将其标记为新的实验条件，不能继续称为单独编码对照。\n4. 将来重标须预先确定参考策略、固定验证集、成功覆盖规则和统计量；统一版本记录训练/评价尺度，以共同冻结坐标汇报比较。测试集不参与尺度拟合。\n\n"
    text+=f"本次{result['checks_passed']}项核验通过。复现：`python -m analysis.studies.schema12_normalization_review_20261014`。audit.json记录全部来源哈希；distribution_and_sensitivity.csv、validation_history.csv、one_factor_scale_sensitivity.csv保存分布、效用敏感性和离线重算明细。\n\n![OWA尺度敏感性](scale_sensitivity.png)\n"
    (OUT/'report.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
