"""Author the aligned three-seed readout from verified tables."""
from pathlib import Path
import json,re
import pandas as pd
OUT=Path(__file__).parent;ROOT=OUT.parents[2]
S=pd.read_csv(OUT/'comparison_summary.csv');G=pd.read_csv(OUT/'group_summary.csv');R=pd.read_csv(OUT/'run_summary.csv')
P=pd.read_csv(OUT/'preference_correlations_matched.csv');L=pd.read_csv(OUT/'preference_monotonicity_matched.csv');E=pd.read_csv(OUT/'preference_cost_workload_endpoints.csv')
H=pd.read_csv(OUT/'heuristic_preferences_overall.csv',index_col=0);HS=pd.read_csv(OUT/'heuristic_by_seed.csv');V=json.loads((OUT/'unified_verification.json').read_text(encoding='utf-8'))

def table(cols,rows):return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(str(x) for x in r)+' |' for r in rows])
def stat(seed,arm):return S[(S.seed==seed)&(S.arm==arm)].iloc[0]
def response(seed,arm,obj):return P[(P.seed==seed)&(P.arm==arm)&(P.objective==obj)].iloc[0].rho_mean
def monotonic(seed,arm,obj):return L[(L.seed==seed)&(L.arm==arm)&(L.objective==obj)].iloc[0].rate
old=G[G.arm=='progressive'].iloc[0];new=G[G.arm=='mixed'].iloc[0]
test_q_delta=100*(new.Q_mean/old.Q_mean-1);heur_q_delta=100*(H.loc['Overall','mixed_Q']/H.loc['Overall','progressive_Q']-1)
lines=['# 固定混合与渐进课程：三 seed 统一检查与比较','',
'更新：2026-10-22，UTC+8。该报告替代截至2026-10-21的两seed阶段性汇总。两种课程均对齐seed11、23、37，各模型先在同一实例/偏好内三次取优，再跨训练seed等权汇总；不在seed之间取最优。','',
'结论：mix三seed的原测试交付均全部完成，平均Q接近但略高于渐进课程；原始单次可靠性和后期波动的改善并不覆盖所有seed。偏好可控性均值略强，但存在方向与seed取舍。7例启发式补评已补齐，PPO成本/负载优势保持，Flow仍是搜索更强。没有新训练或根据测试重选checkpoint。','',
'## 1. 完整性与身份','',
f'三组mix均完成2000episodes、100次PPO外层更新、20次选模验证、4次独立诊断和最终测试。正式原始日志重算通过{V["training_audit_checks"]}项检查，包括完整采样三元组、原生质量公式、取优标记、选模事件、配置与checkpoint身份、物理约束和工程截断。原始验证/诊断/测试合计151812条，训练合计6000回合。','',
table(['mix seed','worker事件起止 UTC+8','耗时','best ep'],[[int(x.seed),f'{x.start_utc8} → {x.end_utc8}',f'{x.elapsed_hours:.2f}h',int(x.best_episode)] for _,x in R.iterrows()]),'',
'三个mix归档源码指纹一致，训练基点为58d5845且dirty。三个渐进归档源码指纹一致，基点8e1dfd7且dirty；新旧训练源码指纹不同。保存的实际配置除experiment_name、类别课程和severity课程锚点外一致，固定验证/诊断子集也一致。协议匹配不等于完全相同源码的严格因果实验。','',
'当前工作区已另行整合默认训练与搜索优化（bc48ad2）。本报告按归档配置和产物核验；读取历史课程配置不调用当前训练入口。旧训练序列检查已改为核对实际日志中每100轮5/35/15/15/10/10/10配额和全程severity=1。','',
'## 2. 原测试集：20例×66偏好','',
'每seed原始3960条轨迹，形成1320个交付组合。正式Q按每偏好的成功交付均值再等权平均；Q越低越好，失败覆盖另列。Q不是最优性gap。','',
table(['seed','课程','best ep','测试Q ↓','交付完成率','最差偏好完成率','原始完成率'],[[int(x.seed),x.arm,int(x.best_episode),f'{x.test_Q:.6f}',f'{100*x.test_completion:.3f}%',f'{100*x.test_worst_completion:.1f}%',f'{100*x.raw_completion:.3f}%'] for _,x in S.iterrows()]),'',
table(['课程','Q均值±seed样本标准差','交付成功','原始完成率均值'],[[x.arm,f'{x.Q_mean:.6f} ± {x.Q_sample_sd:.6f}',('3960/3960' if x.arm=='mixed' else '3956/3960'),f'{100*x.raw_completion:.3f}%'] for _,x in G.iterrows()]),'',
f'mix正式平均Q高{test_q_delta:.2f}%，交付完成率100%对99.899%。将失败交付辅助记为Q=1后，全覆盖均值为渐进{V["test_group"][0]["full_coverage_loss"]:.6f}、mix{V["test_group"][1]["full_coverage_loss"]:.6f}。官方成功样本Q与该辅助损失不可混用。','',
'补入seed37后，不能再概括为“mix各seed单次可靠性都更好”：seed11/23原始成功率改善，但seed37从98.005%降至97.045%；三次取优弥补了这一差异。','',
'同seed共同成功配对结果见test_paired.csv。seed37整体Q仅降低0.03%，但跨所有偏好的物理目标均值分别为：Flow 521.771→575.503、Cost 232.392→272.511、OWA 9.152→10.214。物理目标总体均值不携带各偏好权重，不能用它们反推非线性、偏好条件的Q；需结合端点与逐偏好结果评价。','',
'## 3. 收敛与reward','',
'![三seed收敛对照](curriculum_comparison.png)','',
table(['mix seed','best验证Q','ep2000验证Q','ep2000最差偏好完成率','判断'],[[seed,f'{stat(seed,"mixed").best_val_Q:.6f}',f'{stat(seed,"mixed").last_val_Q:.6f}',f'{100*stat(seed,"mixed").last_val_worst_completion:.1f}%',{11:'最后一轮才获best，尚有学习与波动',23:'last Q更低但2格失败，保留1500轮best',37:'last较1900轮best的Q差8.86%，尚未稳定'}[seed]] for seed in [11,23,37]]),'',
table(['seed','后期验证Q标准差：渐进→mix','后800轮reward标准差：渐进→mix','后800轮训练失败：渐进→mix'],[[seed,f'{stat(seed,"progressive").late_validation_Q_sd:.5f} → {stat(seed,"mixed").late_validation_Q_sd:.5f}',f'{stat(seed,"progressive").late_train_sd:.4f} → {stat(seed,"mixed").late_train_sd:.4f}',f'{int(stat(seed,"progressive").late_train_failures)} → {int(stat(seed,"mixed").late_train_failures)}'] for seed in [11,23,37]]),'',
'验证后期口径为ep1300–2000的8个检查点；reward后期口径为ep1201–2000的800条单次训练轨迹。两种标准差不是同一统计量。mix对验证Q波动的改善出现在seed23/37，对训练reward波动的改善仅出现在seed23；不能混称“mix整体更平稳”。','',
'![mix三seed训练回报](mixed_reward_three_seeds.png)','',
'上图为100回合滚动reward，下图为同批轨迹账面扣除失败惩罚影响的回报，不是新奖励实验。mix消除了渐进课程severity=1处集中释放整数上界的共同台阶，但任务/偏好差异、少量失败和策略更新仍导致平台波动。所有学习率仍为1e-4，未触发衰减，不能视为已经完成小学习率精调。','',
'独立诊断末轮mix Q为seed11/23/37的0.243641/0.241014/0.269395，低于渐进0.255810/0.271000/0.277552；但mix前两seed分别有1/637与3/637交付失败，seed37全部完成，渐进末轮三seed均全完成。质量与覆盖需一起报告。','',
f'平均每seed环境步数：渐进{old.mean_steps:.0f}、mix{new.mean_steps:.0f}（mix多{100*(new.mean_steps/old.mean_steps-1):.1f}%）；总耗时均值{old.mean_total_hours:.2f}h与{new.mean_total_hours:.2f}h。相同episodes与外层updates不意味着相同交互步数、minibatch梯度步数或计算预算。','',
'## 4. Best偏好响应','',
'正式结果使用20例×66偏好的三次取优交付。相关性为同实例内目标权重与该目标值的Spearman，再平均20实例；下表使用同seed两课程共同成功组合（seed23为1316格，其余1320格）。负值越大表示总体方向更符合增大权重、降低目标的预期，不能代表绝对目标值更优。','',
table(['seed','课程','Flow','Cost','OWA'],[[seed,arm,*[f'{response(seed,arm,obj):.3f}' for obj in ['flow','cost','workload']]] for seed in [11,23,37] for arm in ['progressive','mixed']]),'',
'三seed相关系数均值：Flow渐进-0.798→mix-0.805，Cost -0.431→-0.445，OWA -0.391→-0.422。均值略强不等于各seed/方向均改善：seed37 OWA明显增强而Cost变弱，seed11 OWA略弱，seed23 Flow略弱。','',
table(['seed','课程','局部Flow不变差','局部Cost不变差','局部OWA不变差'],[[seed,arm,*[f'{100*monotonic(seed,arm,obj):.1f}%' for obj in ['flow','cost','workload']]] for seed in [11,23,37] for arm in ['progressive','mixed']]),'',
'局部检查固定其余两权重相等，自身权重按0,.2,.4,.6,.8,1递增，配对同实例相邻、两种模型均成功的结果。seed23 Flow/OWA分母99/98，其余100。有限采样下仍有反向变化，不能声称严格单调。','',
table(['seed','课程','纯Cost负载','纯OWA负载','切换后负载降低','切换后成本变化'],[[int(x.seed),x.arm,f'{x.workload_at_cost:.3f}',f'{x.workload_at_workload:.3f}',f'{x.workload_drop_percent:+.2f}%',f'{x.cost_increase_percent:+.2f}%'] for _,x in E.iterrows()]),'',
'mix三个seed纯OWA端点的负载都低于各自纯Cost端点，修正了渐进seed23/37的平均反转。但seed37 mix纯OWA成本也比纯Cost低14.13%，提示成本端点仍有未充分优化之处；这种端点结果不能证明纯偏好已接近对应单目标最优。所有纯端点均20/20完成。','',
'![三seed偏好对照](preference_comparison.png)','',
'曲线按自身权重分组平均，单纯形上其他权重也随之变化；seed23成功集合有小差异。匹配相关性与局部配对另有严格共同样本表。','',
'## 5. 对齐后的7例×4偏好启发式比较','',
'沿用既有7例：balanced_2000223、easy_2000056、fatigue_2000124、arrival_2000222、machine_2000182、reconfiguration_2000066、worker_2000018；4偏好为三个端点及等权。PPO每模型各自三次取优，再平均三个训练seed。NSGA-II+LS N=16、MOEA/D N=32，均为此前冻结的搜索seed11、每请求600秒结果。','',
table(['偏好','渐进PPO 3seed','mix PPO 3seed','NSGA-II+LS','MOEA/D'],[[idx,*[f'{row[k]:.6f}' for k in ['progressive_Q','mixed_Q','nsga2_ls','moead']]] for idx,row in H.iterrows()]),'',
table(['seed','渐进Q','mix Q','mix相对变化','mix胜NSGA/28','mix胜MOEA/D/28'],[[int(x.seed),f'{x.progressive_Q:.6f}',f'{x.mixed_Q:.6f}',f'{x.relative_change_percent:+.2f}%',int(x.nsga2_ls_wins),int(x.moead_wins)] for _,x in HS.iterrows()]),'',
f'mix总体Q较渐进高{heur_q_delta:.2f}%；较NSGA-II+LS低{100*(1-H.loc["Overall","mixed_Q"]/H.loc["Overall","nsga2_ls"]):.2f}%，较MOEA/D低{100*(1-H.loc["Overall","mixed_Q"]/H.loc["Overall","moead"]):.2f}%。这些是归一化偏好代价差，不是物理目标同比降幅。优势仍主要在Cost/OWA，Flow平均Q仍由搜索占优。','',
'![启发式总体比较](heuristic_overall_comparison.png)','',
f'本次新增seed37共84条PPO采样轨迹，复用seed11/23此前168条；统一为252条→84份交付，原始成功{V["mixed_backfill_raw_success"]}/252，交付{V["mixed_backfill_delivery_success"]}/84。逐请求保存动作、排程、指标与取优证据；6个新旧模型的配对数据均核验。','',
'补评源码变更已显式处理：本地较上次新增在线生成器延迟创建和环境查询优化。seed37所有84条动作轨迹又在58d5845原环境源码中重放，全部动作合法，Flow/Cost/OWA/Q/疲劳指标最大误差0。三个mix模型与历史选模日志重叠的12格Q、取优repeat和动作哈希全部一致；前两seed已有请求原样保留。','',
'该7例是搜索调参子集，每类压力仅1例，搜索只有1个seed，PPO选模集与之重叠1例。不能作为独立确认结果。600秒搜索预算充分性未被本次证明；不同机器与并发方式不用于正式加速比。实例bootstrap只衡量这7例中的实例不确定性，不能代替训练seed或搜索seed不确定性。','',
'## 6. 对此前判断的统一修订','',
'- “等待seed37再做三seed结论”已完成；新的权威表是本报告，不再混用渐进3seed与mix2seed。',
'- “mix单次完工率提升”收窄为跨seed均值提升、seed11/23改善，seed37下降。',
'- “mix更稳定”不成立为普遍结论：两seed验证Q波动略降，但reward波动仅seed23降低。',
'- “mix偏好更合理”限于总体趋势和部分端点；局部反转、成本端点不足与不同seed取舍仍在。',
'- 两套模型均保留原协议best，不根据新增补评重选。结果支持完工、质量与偏好可控性之间的权衡，尚不支持宣称固定混合全面优于渐进课程。','',
'## 7. 复核与复用','',
'`audit.json`为三seed正式日志核验；`unified_verification.json`为统一补评及输入保护核验。`comparison_summary.csv`、`comparison_validation.csv`、`reward_windows.csv`、`preference_correlations_matched.csv`、`preference_monotonicity_matched.csv`、`heuristic_preferences_overall.csv`为本文数值依据。','',
'`unified_raw_samples.csv`含完整三seed252条补评，`unified_selected_samples.csv`含84份交付，`unified_paired_comparison.csv`含168行与两搜索的逐模型配对。当前目录原始`raw_samples.csv`/`selected_samples.csv`是新增seed37单模型的84/28条，不能误作三seed汇总；其余原始请求保留在20261021目录。','',
'执行脚本：audit_mixed.py、compare_training.py、align_details.py、run_seed37_eval.py、replay_previous_environment.py、finalize.py、write_report.py。新增补评身份冻结，原有请求文件和checkpoint均保持原哈希。所有图另存PDF用于导出。','']
(OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
assert all((OUT/x).is_file() for x in re.findall(r'\]\(([^)]+)\)', '\n'.join(lines)))
print('REPORT',OUT/'report.md','Q relative',test_q_delta,'heuristic relative',heur_q_delta)
