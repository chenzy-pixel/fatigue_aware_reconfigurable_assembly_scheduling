"""Build the completed investigation report after all 3960 last test requests finish."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"result/analysis/shared_excess_last_20261012"


def table(frame,columns,formats=None):
    formats=formats or {}
    header="| "+" | ".join(label for _,label in columns)+" |\n"
    header+="|"+"---|"*len(columns)+"\n"
    for _,row in frame.iterrows():
        values=[formats.get(field,lambda x:str(x))(row[field]) for field,_ in columns]
        header+="| "+" | ".join(values)+" |\n"
    return header


def main():
    from analysis.studies.shared_excess_last_study_20261012 import main as analyze
    from analysis.studies.remaining_batch_review_20261012 import quality,safe
    analyze()
    summary=json.loads((OUT/"diagnosis_summary.json").read_text());assert summary["last_test_complete"]
    fresh=pd.read_csv(OUT/"last_test_instance_metrics.csv");assert len(fresh)==3960
    assert not fresh.duplicated(["instance_id","preference_key","sampling_repeat"]).any()
    assert fresh.instance_id.nunique()==20 and fresh.preference_key.nunique()==66
    assert (fresh.groupby(["instance_id","preference_key"]).size()==3).all()
    assert set(fresh.sampling_repeat)=={0,1,2} and not fresh.sampling_truncated.any()
    assert np.allclose(quality(fresh[safe(fresh)],"excess"),fresh.loc[safe(fresh),"preference_quality_score"],atol=1e-8)
    metrics=json.loads((OUT/"last_test_metrics.json").read_text())
    assert metrics["checkpoint_episode"]==2000
    assert metrics["provenance"]["network_weights_sha256"]==summary["checkpoints"]["last"]["network_weights_sha256"]
    assert metrics["provenance"]["checkpoint_sha256"]==summary["checkpoints"]["last"]["file_sha256"]
    assert fresh.physical_safety_pass.all() and fresh.schedule_violation_count.sum()==0
    assert (fresh.maximum_worker_fatigue<=fresh.safe_fatigue_limit+1e-9).all()
    checks=pd.read_csv(OUT/"best_replay_checks.csv");traces=pd.read_csv(OUT/"trace/trace_summary.csv")
    assert checks.action_hash_match.all() and traces.action_hash_match.all() and traces.all_safe.all()
    overview=pd.read_csv(OUT/"checkpoint_overview.csv")
    hv=pd.read_csv(OUT/"checkpoint_hypervolumes.csv")
    test=overview[overview.stage=="final_test"].copy()
    test["hv_raw"]=test.role.map(hv[(hv.stage=="final_test")&(hv['mode']=="raw")].groupby("role").pooled_hv.mean())
    test["hv_excess"]=test.role.map(hv[(hv.stage=="final_test")&(hv['mode']=="excess")].groupby("role").pooled_hv.mean())
    test["label"]=test.role.map({"best":"共享头 best ep400","last":"共享头 last ep2000","experts_raw_best":"专家头 raw best ep1900"})
    val=overview[overview.stage=="selection_validation"].copy();val["label"]=val.role.map({"best":"best ep400","last":"last ep2000"})
    endpoints=pd.read_csv(OUT/"last_vs_best_endpoints.csv")
    endpoint_test=endpoints[endpoints.stage=="final_test"].copy()
    endpoint_test["label"]=endpoint_test.role.map({"best":"共享头 best","last":"共享头 last"})
    names={"flow_time_objective":"Flow raw","reconfiguration_cost":"Cost","worker_load_variance":"Variance"}
    endpoint_test["objective_label"]=endpoint_test.objective.map(names)
    units=[]
    for role,path in (("shared_best",ROOT/"result/runs/remaining_flow_experiments_seed11_20261008_225956_shared_head_universal/final_sampled_instance_metrics.csv"),
            ("shared_last",OUT/"last_test_instance_metrics.csv"),
            ("experts_raw_best",ROOT/"result/runs/remaining_flow_experiments_seed11_20261008_225956_main_universal/final_sampled_instance_metrics.csv")):
        f=pd.read_csv(path);rq=f.assign(valid=safe(f)).groupby(["instance_id","preference_key"]).valid.agg(["sum","count"])
        good=f[safe(f)].copy();good["q_raw"]=quality(good,"raw")
        min_q=good.groupby(["instance_id","preference_key"]).q_raw.min().reindex(rq.index)
        units.append({"role":role,"requests":len(rq),"covered":int((rq['sum']>0).sum()),
            "all3_success":int((rq['sum']==3).sum()),"bestof3_quality_raw":min_q.mean(),
            "failure_aware_bestof3_quality_raw":min_q.fillna(1).mean()})
    pd.DataFrame(units).to_csv(OUT/"last_test_bestof3_comparison.csv",index=False,encoding="utf-8-sig")
    p=lambda x:f"{100*float(x):.3f}%";number=lambda x:f"{float(x):.6f}";integer=lambda x:str(int(x))
    val_table=table(val,[("label","模型"),("successes","成功/1950"),("overall_completion","整体完成率"),
        ("minimum_preference_completion","最差偏好完成率"),("quality_excess","excess质量损失"),("mean_training_return","平均训练回报")],
        {"successes":integer,"overall_completion":p,"minimum_preference_completion":p,"quality_excess":number,"mean_training_return":number})
    test_table=table(test,[("label","模型"),("successes","成功/3960"),("overall_completion","整体完成率"),
        ("minimum_preference_completion","最差偏好完成率"),("quality_raw","统一raw质量损失"),("hv_raw","统一raw HV"),("hv_excess","统一excess HV")],
        {"successes":integer,"overall_completion":p,"minimum_preference_completion":p,"quality_raw":number,"hv_raw":number,"hv_excess":number})
    endpoint_table=table(endpoint_test,[("label","模型"),("objective_label","纯偏好目标"),("successes","成功/60"),("mean","成功目标均值"),("bestof3","三次择优目标均值")],
        {"successes":integer,"mean":lambda x:f"{x:.4f}","bestof3":lambda x:f"{x:.4f}"})
    pair=summary["test_pairing"];expert_pair=summary["expert_test_pairing"]
    neural=pd.read_csv(OUT/"trace/decision_trace.csv").query("legal_count>1").groupby(["role","phase"]).shared_pair_saturation_fraction.mean()
    last=test[test.role=="last"].iloc[0];best=test[test.role=="best"].iloc[0]
    quality_change=100*(last.quality_raw/best.quality_raw-1)
    hv_change=100*(last.hv_raw/best.hv_raw-1)
    completion_change=100*(last.overall_completion-best.overall_completion)
    budget_table=table(pd.DataFrame(units),[("role","模型"),("covered","有成功解/1320请求"),
        ("all3_success","三次均成功请求"),("bestof3_quality_raw","三次择优raw质量损失")],
        {"covered":integer,"all3_success":integer,"bestof3_quality_raw":number})
    document=f"""**完整测试确认：shared_head excess 的 last 比 ep400 best 有更强的多目标质量，但最差偏好的完成风险仍集中在 Cost 一侧。**

2026-10-12。完成本次新增完整测试 3960 条，并审计原训练、选模验证与独立诊断数据。研究对象是七项批次的共享头 Universal：历史 schema10、HGNN、shared_preference、excess/v4、seed11、2000 轮。第三目标仍为 worker_load_variance。

**last 的实际测试质量。** 本次重载 last ep2000，20 固定测试实例×66 偏好×3 sampled repeat，种子 300011/300012/300013，温度1，与保存的 best 使用同一实例、偏好和采样流。先复现 best 的 6 条控制轨迹，再逐动作复现 12 条验证诊断轨迹；18 条动作哈希均匹配。模型张量哈希与保存 metadata 相符，未进行观测迁移。

{test_table}
last 相对共享 best，统一raw质量损失变化{quality_change:+.2f}%，raw HV变化{hv_change:+.2f}%，整体完成率变化{completion_change:+.3f}个百分点。这是完整测试的实际结果；验证集的完成率排序不能代替测试结论。

Q 越小越好，HV 越大越好。Q 先对每个偏好的成功样本平均，再对全部偏好等权；HV 按每个实例独立计算后平均，不混合不同实例。两套共同坐标分别使用 raw=(1089.15,353.27,2.2629)、excess=(368.3143,353.27,2.2629)，q_i=J_i/(s_i+J_i)，Q=(max_i w_iq_i+0.05Σ_iw_iq_i)/1.05，HV参考点(1,1,1)。

共享 last 相对共享 best：共同成功 {pair['common_success']} 条，新获成功 {pair['last_gained']} 条，失去原成功 {pair['last_lost']} 条，两侧都失败 {pair['both_fail']} 条。相对专家头 raw best：共同成功 {expert_pair['common_success']} 条，新获成功 {expert_pair['last_gained']} 条，失去原成功 {expert_pair['last_lost']} 条。匹配成功集合的质量/目标差异见 last_vs_best_paired_metrics.csv 和 last_vs_experts_paired_metrics.csv。

{endpoint_table}
单目标端点值只平均成功请求。best-of-3 的三模型请求覆盖与质量如下，覆盖和条件性质量应一起读取。

{budget_table}
全部新增测试保存的物理安全标志通过，调度违规为0，疲劳峰值不超过上限。

![完整测试比较]({OUT.as_posix()}/last_test_comparison.png)

**“完成率上不去”首先涉及日志口径。** validation_log.csv 的 completion_rate 为13种偏好的最低完成率，每偏好50实例×3重复=150条；它与1950条的整体完成率不同。

{val_table}
last 整体仅比 best 多一条失败，但失败从不同偏好集中到了 Cost。ep600–2000 连续15次验证，最低完成率都由纯 Cost 给出。ep1000–2000共21450条，414次失败，其中纯Cost147次、w_cost>0.5的229次；413次horizon，1次unrecoverable_deadlock。低点ep1700纯Cost仅113/150成功。全部失败已实际扣除2.0终局惩罚。

last 的独立49实例诊断为1901/1911成功，整体99.477%，最差偏好95.918%，excess质量0.354485。这是与选模集不重叠的同分布验证，不是OOD/stress。

**已确认的机制一：平均回报允许质量与少量失败交换，选模则优先保护最差偏好。**

本批 r_t=Δprogress−ΔQ−2·terminal_failure，gamma=1，故总回报=最终progress−最终Q−2F（初始项在本批为0）。每条成功轨迹的回报优于失败轨迹，但期望回报仍可能通过大幅降低多数请求的Q、承担少量额外失败而上升。

纯Cost验证：best全部150条成功、成功Cost均值809.5202，平均回报0.33759；last146条成功、成功Cost均值343.3014，平均回报0.50880。整个验证集回报0.58253→0.64172。这样，训练平均回报上升与最差偏好完成率下降可以同时发生。离线保持这些轨迹固定，纯Cost平均回报排序在惩罚约8.42处才相等；这是回报分解，不是推荐的训练惩罚值。

严格字典序选模采用(min_w完成率↑，Q↓)，因此ep400的98.667%压过last的97.333%，即使后者Q降低约14.83%。该选择符合规则。验证完成率/质量前沿为ep400、600、1500、2000；阈值97%时，已有验证中last质量最好，阈值98%时为ep600。只有best/last模型被保留，不能用中间轮次CSV冒充其测试成绩。

如果应用使用每请求三次采样后择优，需另看请求覆盖：ep400、600、1500、2000均650/650单元至少一次成功，而ep1700为648/650。该推断协议的选模指标可在后续实验中预先声明；当前正式单轨迹指标保持原口径。

![后期完成率与行为]({OUT.as_posix()}/late_completion_diagnosis.png)
![各偏好的完成率]({OUT.as_posix()}/preference_completion_heatmap.png)

**已确认的机制二：纯Cost存在持续等待和工人服务尾部风险。**

回放四条last纯Cost失败及各自相同请求的best，另回放last的其他成功repeat；12条均与历史动作哈希一致。四个实例都存在best的完整安全解，因此这四条失败是策略行为问题。

- 2000177：last5次重构、疲劳峰值0.278；best16次重构并在187.4分钟完成。last在211.4、215.6、218.3分钟以约88%概率选择WAIT，最后O_R15_5到236.3才加工，确定结束时间246.7超过240分钟horizon。这里的瓶颈是过晚调度。
- 2000121：last仅6次重构，O_R1_5到231.0分钟才开始，结束245.4；另一次重构承诺在232.1分钟，乐观完工也超过horizon。低Cost路径延迟了关键工序。
- 2000182：last疲劳峰值0.7202；best0.3526。last最终工序在239.7分钟开始，来不及完成。
- 2000330：last20次重构、疲劳峰值0.75；best19次、峰值0.474并完成。这里还存在工人负荷集中、疲劳恢复与重构服务拖延，不能把所有失败都归为重构次数过多。

生产mask核对READY、机器空闲和模块能力，WAIT由进展证书允许；它们保障即时动作合法，不保证所有剩余工序在horizon内完成。模型具有预测完工时间、horizon slack和图状态信息，但纯Cost仍会在合法动作中选择超时风险路径。

**动作干预验证及反例。** 只对上述四条失败进行诊断，共16条变体，不用于正式性能排名：

| 干预 | 恢复完成/4 |
|---|---:|
| 最后1/4时域内，将推进时间的WAIT改成模型最高评分合法pair | 3 |
| 只替换一次末段WAIT | 0 |
| 同一工人阶段改派明显更低预计疲劳的合法工人 | 0 |
| 原last权重使用greedy解码 | 0 |

持续末段干预能救回3条，说明累积时间裕量损失具有实质作用；2000121仍失败，提示更早的路线/机器配置选择也需处理。低疲劳改派未单独救回失败，疲劳与时间路径应共同分析。四条greedy诊断的完成进度仅约42%–58%，其高概率WAIT是学到的倾向；正式sampled协议中的随机配对选择帮助部分请求完成。推广这些策略需要对全协议检验。

固定同一物理状态，仅改变偏好：2000177在186.5分钟，last的WAIT概率为Flow0.369、Cost0.827、Variance0.682。偏好通道有效，纯Cost的等待倾向更强。所测固定状态上Flow与Cost动作分布L1平均为best0.179、last0.536；last的响应比早期模型强。

**有实测依据的学习机制候选：末端失败反馈与价值估计。**

代码使用gamma=1、GAE lambda=0.95。A_t=Σ_l(γλ)^lδ_(t+l)，δ_t=r_t+γV(s_(t+1))−V(s_t)。100决策之前的单个末端TD残差系数0.95^100≈0.00592；依靠Value才能及时表示长程失败风险。[GAE原论文](https://arxiv.org/html/1506.02438v6)指出lambda小于1时，近似Value误差会引入偏差；增大lambda降低这种依赖，同时增加估计方差。

在四条复现失败中，327个可选择其他动作的WAIT，有87个得到正的未标准化GAE(0.95)，即使各轨迹最终回报为负；重算lambda=0.99或1.0后，这些WAIT的优势全部非正。它们是验证轨迹的离线诊断，并非历史训练batch梯度；实际更新还受全batch优势标准化、PPO clip和其他样本影响，不能据此宣称每个WAIT在历史训练中都被强化。

初始Value校准覆盖50实例×13偏好×两模型：last总体预测均值0.64363、实测三采样回报均值0.64172，总体偏差很小；纯Cost四个有失败repeat的实例，预测均值0.55160，三个repeat实测均值−0.25023。尾部误差被总体指标掩盖，但每请求只有三次采样，真实期望仍有统计不确定性。

共享头输出为2tanh(score)。所测非强制决策中，last工人pair饱和比例约{100*neural.loc[('last','WORKER')]:.2f}%、生产pair约{100*neural.loc[('last','PRODUCTION')]:.2f}%，best这两项为0。tanh饱和会缩小局部导数，是敏感性/更新能力的机制候选，尚未通过单因素训练证明其造成失败。共享Universal平均梯度裁剪比例44.35%，记录loss/KL/梯度有限；当前证据没有直接测量目标梯度冲突。

训练覆盖也需重视：2000轮只有200条纯Cost，后1000轮的两段纯Cost训练均45/50成功；后期训练难度提高至severity=1。训练在线实例未全部具有可行性证明，该训练统计是现象，固定验证的失败与上述成对回放提供更直接证据。

本批共享头Cost单目标也存在失败：58/60完成、成功Cost均值206.6321。因此成本与期限风险的交换并非只出现在Universal。纯Cost的w_flow=0，excess对Flow目标的缩放不直接改变其标量Q；模式仍能通过其他偏好的训练、观测特征和共享参数影响端点策略。当前未进行固定其他条件、仅改变Flow模式或评分头的Universal配对训练，不能把上述失效全部归因于excess或共享梯度冲突。

**后续实验优先级。**

1. 用同协议控制gamma=1，仅比较GAE lambda=0.95/0.99/1.0，补报纯Cost和压力类型的Value误差、最差偏好完成率与质量。现有回放给这一消融提供了具体理由，实际收益仍需训练验证。
2. 给所有偏好共同表达期限风险：优先比较末段时间裕量/关键订单完成风险的软评分或势函数引导。仓库已存在关闭的feasibility_shaping接口；使用时检查gamma=1下回报恒等式和终端势函数。将预测完工时间用作硬mask前，需要必要条件或可行性证书，避免误删可完成路线。
3. 将纯Cost的尾部完成风险纳入训练采样、价值诊断或逐偏好风险约束，评价提高失败惩罚的收益与质量代价。[CPO原论文](https://arxiv.org/html/1705.10528)提供了奖励与期望成本约束分开的框架；本项目还需按偏好处理风险，平均约束不自动保证最差偏好。
4. 明确部署的单次/三次择优预算，再预先声明选模准则，并保留完成率/质量前沿checkpoint。当前last可以用实际测试数据评价，ep1500等中间模型则需要未来运行保留。

本批仅seed11、20个ID测试实例；专家raw与共享excess同时改变评分头和Flow/reward协议。历史训练来自带未提交改动的源树，本次使用当前兼容代码进行控制回放和完整评测；18条控制动作一致支持这些已检验轨迹的可比性，不能视为全状态等价证明。以上确认了本模型的具体失效路径与last质量，架构或excess模式的一般因果结论仍需匹配协议、多训练种子验证。

关键代码依据：[验证完成率聚合]({ROOT.as_posix()}/train.py:281)、[严格字典序选模]({ROOT.as_posix()}/training/protocol.py:111)、[GAE递推]({ROOT.as_posix()}/agent/ppo/buffer.py:93)、[全batch优势标准化]({ROOT.as_posix()}/agent/ppo/agent.py:284)、[共享头2tanh输出]({ROOT.as_posix()}/agent/ppo/network.py:431)、[WAIT进展证书]({ROOT.as_posix()}/environment/env.py:3778)。

复核入口：analysis/studies/shared_excess_last_study_20261012.py；完整测试入口analysis/studies/evaluate_shared_excess_last_20261012.py；回放、干预、Value校准入口分别为analysis/studies/trace_shared_excess_last_20261012.py、analysis/studies/intervene_shared_excess_last_20261012.py、analysis/studies/calibrate_shared_excess_critic_20261012.py。所有原始新增行、配对统计、动作记录、checkpoint哈希与图均在本目录。
"""
    (OUT/"report.md").write_text(document,encoding="utf-8")
    plot_test(test)
    from analysis.studies.verify_shared_excess_last_20261012 import main as independent_verify
    independent_verify()
    independent=json.loads((OUT/"independent_verification.json").read_text())
    failure_groups=pd.read_csv(OUT/"last_test_failure_groups.csv")
    pressure_counts=failure_groups.groupby("pressure_type").failures.sum().to_dict()
    intervals=pd.read_csv(OUT/"instance_cluster_intervals.csv")
    intervals["label"]=intervals.reference.map({"best":"共享头 best","experts_raw_best":"专家头 raw best"})
    intervals["metric_label"]=intervals.metric.map({"completion_delta":"整体完成率差",
        "paired_success_quality_raw_delta":"共同成功raw质量差","pooled_hv_raw_delta":"raw HV差"})
    uncertainty_table=table(intervals,[("label","参照模型"),("metric_label","last减参照"),
        ("last_minus_reference","实测差"),("instance_cluster_95_low","95%区间下限"),
        ("instance_cluster_95_high","95%区间上限")],
        {"last_minus_reference":number,"instance_cluster_95_low":number,"instance_cluster_95_high":number})
    appendix=f"""

**独立复核与不确定性。** 独立实现沿z积分xy矩形并集，复核三个模型×20实例×两套坐标的120个HV，最大绝对误差{independent['maximum_volume_error']:.3g}。完整last测试终止失败原因：{independent['failure_reasons']}；各实例和偏好的失败统计见last_test_failure_groups.csv、last_test_completion_by_preference.csv。

测试失败按压力类型分布为{pressure_counts}；机器瓶颈3000014单独有13次失败。last成功条件下的质量与HV收益明确，但整体完成率相对best仅多4条成功，实例bootstrap区间跨0，不能据此推断它具有更高的一般完成可靠性。纯Cost从best的60/60变为last的55/60，是最差偏好下降的直接来源。

{uncertainty_table}
上述区间以整个实例为抽样单元进行10000次配对bootstrap，保留同一实例内的偏好和重复相关性。完成率差采用0–1比例单位；Q差只使用两模型共同成功的匹配请求。这些区间描述这20个ID实例的重采样不确定性，不能代表训练种子不确定性或OOD泛化。
"""
    (OUT/"report.md").write_text(document+appendix,encoding="utf-8")
    artifact_files=[Path(__file__), ROOT/"analysis/studies/verify_shared_excess_last_20261012.py",
        *(ROOT/"analysis/studies"/name for name in ("evaluate_shared_excess_last_20261012.py",
            "trace_shared_excess_last_20261012.py","intervene_shared_excess_last_20261012.py",
            "calibrate_shared_excess_critic_20261012.py")),
        *(OUT/name for name in ("last_test_metrics.json","trace/trace_summary.csv",
            "trace/decision_trace.csv","interventions/outcomes.csv",
            "critic_initial_calibration.csv","independent_verification.json"))]
    verification={"last_test_trajectories":len(fresh),"safe_successes":int(fresh.task_succeeded.sum()),
        "best_control_hash_matches":int(checks.action_hash_match.sum()),"validation_replay_hash_matches":int(traces.action_hash_match.sum()),
        "all_test_physical_safety_pass":bool(fresh.physical_safety_pass.all()),"schedule_violation_count":int(fresh.schedule_violation_count.sum()),
        "source_hashes":summary["source_hashes"],
        "artifact_hashes":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact_files},
        "independent_hv_checks":independent['volume_cross_checks'],
        "maximum_hv_error":independent['maximum_volume_error'],
        "report_sha256":hashlib.sha256((OUT/"report.md").read_bytes()).hexdigest()}
    (OUT/"final_verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
    print(test[["role","successes","overall_completion","minimum_preference_completion","quality_raw","hv_raw","hv_excess"]].to_string(index=False))


def plot_test(test):
    labels=test.label.tolist();colors=["#777777","#D55E00","#0072B2"]
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,field,title,percent in zip(axes.flat,["overall_completion","minimum_preference_completion","quality_raw","hv_raw"],
        ["Overall completion","Worst-preference completion","Common raw preference loss","Common raw pooled HV"],[True,True,False,False]):
        values=test[field].to_numpy()*(100 if percent else 1)
        if percent:
            ax.scatter(range(len(values)),values,color=colors,s=100,zorder=3)
            ax.set_xlim(-.5,len(values)-.5)
        else:
            ax.bar(range(len(values)),values,color=colors)
            ax.set_ylim(0,float(values.max())*1.15)
        ax.set_xticks(range(len(values)),["Shared best 400","Shared last 2000","Experts raw best"],fontsize=9)
        ax.set_ylabel(title+(" (%)" if percent else ""));ax.grid(axis="y",alpha=.2)
        if percent:ax.set_ylim(max(0,float(values.min())-3),100.5)
        for index,value in enumerate(values):ax.annotate(f"{value:.3f}" if percent else f"{value:.4f}",(index,value),xytext=(0,5),textcoords="offset points",ha="center",fontsize=10)
    fig.tight_layout();fig.savefig(OUT/"last_test_comparison.png",dpi=200);fig.savefig(OUT/"last_test_comparison.svg");plt.close(fig)


if __name__=="__main__":main()
