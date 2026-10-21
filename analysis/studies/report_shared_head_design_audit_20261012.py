"""Render the head/mode/global/reward audit from inspected evidence."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'result/analysis/shared_head_design_audit_20261012'
LABELS={'old_experts':'旧专家头 raw','old_shared':'旧共享头 raw','new_experts_raw':'本批专家头 raw',
        'new_shared_best':'本批共享头 excess best','new_shared_last':'本批共享头 excess last'}


def table(frame,columns,formatters=None):
    fmt=formatters or {};lines=['| '+' | '.join(label for _,label in columns)+' |','|'+'---|'*len(columns)]
    for _,row in frame.iterrows():lines.append('| '+' | '.join(fmt.get(key,str)(row[key]) for key,_ in columns)+' |')
    return '\n'.join(lines)


def main():
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    overview=pd.read_csv(OUT/'universal_comparison.csv');overview['label']=overview.role.map(LABELS)
    endpoints=pd.read_csv(OUT/'universal_endpoints.csv');tail=pd.read_csv(OUT/'matched_cost_tail.csv')
    probe=pd.read_csv(OUT/'global_probe_summary.csv');ties=pd.read_csv(OUT/'penalty_break_even.csv')
    old=overview[overview.role=='old_shared'].iloc[0];new=overview[overview.role=='new_shared_last'].iloc[0]
    fmt={'episode':lambda v:str(int(v)),'successes':lambda v:str(int(v)),
         'overall_completion':lambda v:f'{100*v:.3f}%','worst_preference_completion':lambda v:f'{100*v:.3f}%',
         'quality_raw':lambda v:f'{v:.6f}','raw_hv':lambda v:f'{v:.6f}'}
    main_table=table(overview,[('label','模型'),('episode','选中轮次'),('successes','成功/3960'),
        ('overall_completion','整体完成率'),('worst_preference_completion','最差偏好完成率'),('quality_raw','共同raw质量↓'),('raw_hv','共同raw HV↑')],fmt)
    endpoint=endpoints[endpoints.role.isin(['old_shared','new_shared_last'])].copy();endpoint['label']=endpoint.role.map(LABELS)
    endpoint['target']=endpoint.objective.map(dict(zip(['flow_time_objective','reconfiguration_cost','worker_load_variance'],['Flow','Cost','Variance'])))
    endpoint_table=table(endpoint,[('label','模型'),('target','纯偏好'),('successes','成功/60'),('mean','成功目标均值')],{'successes':lambda v:str(int(v)),'mean':lambda v:f'{v:.4f}'})
    tail['label']=tail.role.map(LABELS)
    tail_table=table(tail,[('label','模型'),('mean','平均Cost'),('median','中位数'),('p95','P95'),('maximum','最大值'),('mean_bounded_cost','平均J/(s+J)')],{k:lambda v:f'{v:.4f}' for k in ['mean','median','p95','maximum','mean_bounded_cost']})
    prefix=ROOT.as_posix();out=OUT.as_posix()
    text=f"""**审核结论：旧共享头的端点和可靠性更强；本批last的综合权衡质量与HV更强。全局向量是有证据支持的敏感因素，尚未单独证明它导致退化。失败罚项已正确执行，但平均回报确实允许用部分完成率换取更低的有界质量损失。**

本报告回答2026-10-12关于旧/新shared_head、专家头、raw/excess及失败惩罚的追问。数据均为历史Variance目标，旧schema6与本批schema10；不混入当前默认schema11的OWA目标。

**一、先纠正比较对象与“质量下降”的口径。**

旧shared_head来自`ablation_structural_shared_head_seed11_20261006_025649_012965`，raw/v2，best=last=ep2000。本批shared_head为excess/v4，best=ep400，last=ep2000。

{main_table}

五个模型均训练2000轮；test相同20实例×66偏好×3采样。重新逐单元核对实例、偏好、重复、派生种子、manifest与subset哈希，并核对各运行2000条训练的实例、压力、severity、偏好计划，全部一致。质量按成功样本的偏好等权共同raw尺度计算，HV按实例独立计算后平均。

本批last相对旧shared：综合质量损失变化{100*(new.quality_raw/old.quality_raw-1):+.2f}%，HV变化{100*(new.raw_hv/old.raw_hv-1):+.2f}%，整体完成率下降{100*(old.overall_completion-new.overall_completion):.3f}个百分点，最差偏好下降3.333个百分点。共有3905条匹配成功请求，质量差−0.007125；12/20实例、35/66偏好改善，HV在19/20实例改善。以整个实例为单位bootstrap，匹配质量差95%区间为[{audit['old_new_uncertainty']['paired_quality_raw_95_interval'][0]:.6f},{audit['old_new_uncertainty']['paired_quality_raw_95_interval'][1]:.6f}]；这不估计训练种子不确定性。

旧三个纯端点的实际目标均值确实更好：

{endpoint_table}

本批last的纯Variance实际目标比旧模型高约51.55%；纯Cost均值接近但多5次失败。综合损失较低主要来自混合偏好区域：36个内部偏好在共同成功集的平均损失0.313951→0.303030，27个边界混合偏好0.377048→0.372246；三个纯端点平均则0.464612→0.482143。应同时看端点、按偏好质量、HV与完成率。

![历史与本批比较]({out}/historical_current_comparison.png)

**二、全局向量的具体改变，以及能下到哪一层结论。**

checkpoint权重直接确认全局输入11→9维：`global_encoder.0.weight`从128×11变为128×9。旧基线源码`git show c83b515:environment/env.py`给出旧11维顺序；当前schema10路径在[env.py]({prefix}/environment/env.py:796)。

| 改动 | 内容 |
|---|---|
| 保留6维 | 当前时间、待分配重构比例、工序完成比例、生产阶段标志、工人匹配缺口、最小备选工人比例 |
| 删除5维 | 活跃订单比例、READY工序比例、工人阶段标志、安全工人比例、最小候选完工裕量 |
| 新增3维 | 当前奖励Flow/s_F、累计Cost/s_C、承诺负荷Variance/s_V |

工人阶段标志与生产阶段基本冗余；其他信息仍部分存在于节点、候选与WAIT特征。最小候选裕量从全局显式聚合中移除，可能增加模型识别期限风险的学习难度，但候选的horizon_slack和WAIT后的订单裕量依然存在，期限信息没有全部消失。

更具体的数值问题是：新增累计目标使用无界线性`J/s`，并非奖励中的有界`J/(s+J)`。本批last成功终局Variance/s_V均值12.50、P95=65.59、最大149.59；大多数旧全局量在[0,1]，旧最小裕量截断在[-1,1]。这些终局统计提示范围不一致，不等价于动作决策状态分布。

为核实活跃状态，额外回放两条已经复现的纯Cost失败，选取31个非强制状态，固定图、候选、mask、偏好与权重，只变换全局累计目标编码。原始状态概率与旧回放记录逐点匹配。last在这些状态上的Variance/s_V均值22.23、最大79.50；仅把第9维改为`x/(1+x)`，动作分布L1平均改变0.2873，7/31状态的最高概率动作改变，WAIT平均概率0.5591→0.4523。best同样敏感。

这确认目标全局输入会实质影响纯Cost决策，是值得训练消融的机制候选。对已经训练的网络临时变换输入属于分布外敏感性测试，不能视为恢复旧11维，也没有证明修订后完成率会上升。

同时存在反例：本批专家头使用相同9维定义，Variance/s_V的P95也高达60.95，却达到99.823%完成率。因此无法从量级单独推出共享头回归的原因。

旧到新还同时发生以下变化：图关系12→14，增加实际加工`processing_on`与实际服务`served_by`；time_context v1→v2及顺序资源投影；HGNN带属性消息加入joint ReLU；工程截断改为自举；本次共享头还切换raw→excess。两个新增关系带来66048个参数，全局维数减少256个参数，合计模型1017673→1083465。维数变化不是这段版本差异的全部。

旧/new来源都记录dirty=true，旧checkpoint未记录消息激活身份；提交历史可以定位上述实现改变，但不能完整恢复旧运行的所有未提交源码。全局维数、关系集合、时序版本及新消息身份则有checkpoint证据。详见audit.json。

**三、专家头还是共享头：目前最干净的结构证据仍支持共享头。**

旧raw对照的配置差异只有实验名、actor_head_variant及其派生身份；源码状态哈希、训练计划、观测与奖励一致。共享头相对专家头完成3856→3939，质量0.408421→0.347041，HV0.181379→0.253228。3843条共同成功请求中，66/66偏好和20/20实例的平均质量都改善。这是一个训练种子的直接结构对照。

本批同schema10、同excess的Flow单目标也提供结构对照：两份配置仅评分头及派生身份/实验名不同，来源哈希相同。共享头60/60成功，专家头59/60；在59条共同成功请求上Flow从1131.51降到1104.14，约改善2.42%。该结果不能直接推广到Universal所有偏好。

专家头有分目标、固定方向直接评分的归纳约束；共享头把全部目标特征与偏好交给同一个条件MLP，更灵活。前者也有上下文和残差项，不能据直接评分器宣称整个策略严格单调。源码见[直接专家评分]({prefix}/agent/ppo/network.py:360)、[共享评分]({prefix}/agent/ppo/network.py:431)。

实际选模型时：本批专家头raw更适合优先可靠单次完成；本批共享last更适合优先综合质量、Pareto候选或三次采样择优。作为架构研究方向，现有单变量证据不支持因本次结果就改回专家头。

**四、raw还是excess：Universal的2×2对照尚不完整。**

| schema10 Universal | raw | excess |
|---|---|---|
| objective_experts | 已有2000轮 | 缺完整同协议训练 |
| shared_head | 缺完整同协议训练 | 已有2000轮 |

旧shared raw属于schema6，不能填进这个矩阵。本批专家raw与共享excess是同时改变两个因素的对角线，无法分离评分头、模式及交互效应。

专家头Flow单目标的已有raw/excess参考：raw57/60成功、成功Flow1116.54；excess59/60、1131.51。在56条共同成功请求上raw1108.21，excess1124.05，excess的Flow反而高1.43%。三次择优Flow1099.37与1099.69接近。两次来源哈希不同，且只有seed11；这组证据不支持宣布excess在目标质量上优于raw。

对同一实例的成功终局，F_excess=F_raw−固定加工下界，故纯Flow的确定性解排序相同。但两套冻结尺度不同，有界非线性变换及多目标权衡都会变化，随机策略的期望排序也未必相同。纯Cost时w_flow=0，Flow缩放不直接进入标量Q，却仍可通过全局输入、WAIT特征与其他偏好的共享训练影响行为。

**五、失败罚项：实现正确，平均回报与完成率优先的目标不一致。**

旧shared、本批shared和本批专家raw的117000条选模验证记录全部满足回报恒等式，最大误差小于1.5e−15；每次真实失败都恰好扣除2.0。旧新均为gamma=1、GAE lambda=0.95、关闭feasibility shaping，Cost尺度353.27、Variance尺度2.2629未改变。没有发现漏扣罚项或把原始货币数值直接加入标量奖励。

另外独立重建旧/新专家头和共享头四次训练的8000条实际训练回报，全部满足同一恒等式与失败罚项；结果见training_reward_checks.csv。

记p为最终完成进度，Q为有界偏好损失，I_fail为失败指示，本批初始项为0：

`R = p − Q − κ I_fail`，当前`κ=2`。

`E[R] = E[p−Q] − κ Pr(failure)`。

Q在[0,1]内，成功轨迹回报在[0,1]；失败轨迹因p<1，回报小于−1。罚项足以让单条失败比单条成功差，但不能保证整个随机策略按完成概率严格排序。多数成功请求的质量收益仍可盖过少量额外失败。

| 纯Cost验证，固定原轨迹 | ep400 best | ep2000 last |
|---|---:|---:|
| 完成 | 150/150 | 146/150 |
| 平均p−Q | 0.337593 | 0.562134 |
| κ=2时平均回报 | 0.337593 | 0.508801 |
| κ=5时离线回报 | 0.337593 | 0.428801 |
| κ=10时离线回报 | 0.337593 | 0.295468 |

last的无失败罚项回报增益0.224541，额外失败概率4/150，仅扣去0.053333，平均回报仍增加0.171208。保持策略和轨迹不动，两者在κ≈8.4203处打平。测试集上的相同计算则为κ≈2.4750；不能用某个测试阈值直接选择训练超参数。

在整个验证集，last只比best多失败1/1950，却有较大的多数请求质量收益，离线打平需要κ≈117.43。这个数不是推荐值，而是说明平均完成率惩罚和严格最差偏好选模的目标差异很大。任意有限κ都不能在所有可能策略间保证严格字典序：额外失败概率可以任意小。

但罚项也不是全部原因。旧shared使用同样κ=2，纯Cost测试60/60成功且成本338.54；旧纯Cost平均回报0.53633，高于本批last的0.39736。旧策略展示了更可靠且低成本的可达行为。需要同时处理学习、表示与尾部信用分配，而不是仅增加一个常数。

![纯Cost的离线罚项敏感性]({out}/failure_penalty_sensitivity.png)

**六、有界归一化还改变了对成本尾部的偏好。**

纯Cost的Q=J/(s+J)，s=353.27，二阶导数`q''(J)=−2s/(s+J)^3<0`。它对单个确定成本保持单调，但`E[q(J)]`与`q(E[J])`不相等，也不能把最小化E[q]当作最小化平均原始成本。

在旧shared与本批last共同成功的同一55条纯Cost请求上：

{tail_table}

平均实际Cost略升、P95升高，但平均有界损失下降。新策略的中位数降低，高成本尾部扩大；有界变换压缩高成本损失，能够产生这一排序。这是标量化的数学性质，旧版也存在，并非excess独有的实现错误。若目标还包括平均实际Cost或尾部成本，需要把这些指标与完成率一起纳入训练/验证目标，而不能只读有界Q。

**后续选择与最小验证顺序。**

1. 先锁定本批schema10、Variance定义、冻结尺度及2000轮预算，补齐共享raw与专家excess，形成同版本2×2；保存并同协议评估best、last和预先定义的验证前沿checkpoint。
2. 全局输入消融保留累计目标信息，分别检验“补回显式最小裕量”和“累计目标改为有界或log1p编码”，其他图关系、投影、消息与奖励固定。临时改输入的探针结果只能用于挑选假设，必须重训验证。
3. 对选定共享配置单独比较κ=2/5/10，先固定GAE lambda；使用验证集的逐偏好完成率、原始目标及尾部指标决策。提高κ可能减轻失败，也可能牺牲质量或增加Value学习难度，当前没有重训收益保证。
4. 如果实际要求是“每个偏好完成率至少98%再优化质量”，应明确这个约束，并使用逐偏好的风险估计/约束或自适应罚项；统一平均罚项无法自动保证最差偏好。GAE lambda与Value尾部校准可随后单独消融。

当前模型选择建议：单次完成可靠性优先，选本批专家raw；质量与三采样候选优先，保留共享last；三个纯端点与完成表现，旧共享raw仍是有价值的历史参照。结构研发继续保留共享头；raw作为控制基线，excess收益等待同版本对照确认。

**复核入口。** [统计审计脚本]({prefix}/analysis/studies/shared_head_design_audit_20261012.py)、[固定状态全局探针]({prefix}/analysis/studies/probe_global_features_20261012.py)。audit.json保存模型身份、来源哈希与配对检查；config_differences.csv保存完整配置差异；penalty_sensitivity.csv保存20轮验证×罚项网格的离线重评分；matched_cost_tail.csv保存共同成功请求的成本尾部。全部为seed11证据，跨版本行为变化的单因素因果关系仍未由重训实验证实。
"""
    (OUT/'report.md').write_text(text,encoding='utf-8')
    plot(overview,ties)
    files=[Path(__file__),ROOT/'analysis/studies/shared_head_design_audit_20261012.py',ROOT/'analysis/studies/probe_global_features_20261012.py',
           OUT/'audit.json',OUT/'global_fixed_state_probe.csv',OUT/'global_probe_controls.csv',OUT/'global_probe_verification.json',
           OUT/'report.md',OUT/'matched_cost_tail.csv',OUT/'historical_current_comparison.png',OUT/'failure_penalty_sensitivity.png']
    verification={'matched_tests':audit['checks'],'reward_rows_checked':sum(r['trajectories'] for r in audit['reward_identity']),
                  'training_reward_rows_checked':sum(r['trajectories'] for r in audit['training_reward_identity']),
                  'global_probe':json.loads((OUT/'global_probe_verification.json').read_text()),
                  'artifact_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print('Report, figures and verification saved:',OUT)


def plot(overview,ties):
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    labels=['Old experts\nraw','Old shared\nraw','Current experts\nraw','Current shared\nbest 400','Current shared\nlast 2000']
    colors=['#999999','#0072B2','#009E73','#E69F00','#D55E00'];fig,axes=plt.subplots(1,3,figsize=(16,5.1))
    for ax,field,label in zip(axes,['quality_raw','worst_preference_completion','raw_hv'],['Common raw preference loss','Worst-preference completion (%)','Common raw pooled HV']):
        y=overview[field].to_numpy()*(100 if field=='worst_preference_completion' else 1)
        if field=='worst_preference_completion':ax.scatter(range(5),y,c=colors,s=90);ax.set_ylim(87,102);ax.set_xlim(-.5,4.5)
        else:ax.bar(range(5),y,color=colors);ax.set_ylim(0,y.max()*1.2)
        ax.set_xticks(range(5),labels,rotation=28,ha='right',fontsize=9);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(y):ax.annotate(f'{v:.2f}' if field=='worst_preference_completion' else f'{v:.3f}',(i,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'historical_current_comparison.png',dpi=300);fig.savefig(OUT/'historical_current_comparison.svg');plt.close(fig)
    grid=np.linspace(0,12,241);fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,stage in zip(axes,['validation','test']):
        row=ties[(ties.stage==stage)&(ties.subset=='pure_cost')].iloc[0]
        ax.plot(grid,row.reference_base_return-grid*row.reference_failure_probability,label='Current shared best 400',color='#777777')
        ax.plot(grid,row.candidate_base_return-grid*row.candidate_failure_probability,label='Current shared last 2000',color='#D55E00')
        if stage=='test':
            old=ties[(ties.stage=='old_vs_new_last_test')&(ties.subset=='pure_cost')].iloc[0]
            ax.plot(grid,old.reference_base_return-grid*old.reference_failure_probability,label='Old shared raw 2000',color='#0072B2')
        ax.axvline(2,color='black',linestyle=':',alpha=.7);ax.axvline(row.penalty_tie,color='#D55E00',linestyle='--',alpha=.6)
        ax.set_xlabel('Failure penalty kappa (fixed trajectories)');ax.set_ylabel(f'Pure Cost {stage} mean return');ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'failure_penalty_sensitivity.png',dpi=300);fig.savefig(OUT/'failure_penalty_sensitivity.svg');plt.close(fig)


if __name__=='__main__':main()
