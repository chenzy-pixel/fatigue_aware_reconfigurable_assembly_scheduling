# 默认正式评估与选模：三次采样取优

PPO主线新实验的 `training.formal_evaluation.sample_aggregation` 默认为 `best_of_3_v1`。每个相同实例、相同偏好仍采样三次，不增加原有PPO正式评估的请求预算。2026-10-17新增套件的PPO/DCAN主交付也采用这一选择规则；DDQN贪心单次、搜索单请求交付另有协议，见下文。

先在三个完整采样中选择一份交付方案，排序为成功完成优先、原生偏好质量损失Q更低优先、采样序号更小优先。三次都失败的组合仍计为失败。原始请求中的调度违规、活动约束失败、采样截断、缺失或重复不能通过取优隐藏；不完整评估不能选checkpoint。全疲劳主线检查物理疲劳安全；疲劳中性对照按其活动约束准入，并独立报告原始物理疲劳暴露。真实任务失败的终局报表Q固定为1，三次均真实失败时按采样序号保留。

原有PPO入口的Universal模型先按取优后的最差偏好完成率选checkpoint，再按取优后的偏好平衡Q比较。同一个偏好内仅对成功交付组合平均Q，再对偏好等权平均；某偏好无成功组合时Q为无穷大。单目标模型以取优后的整体完成率优先。实现按1e-12容差比较，同分保留更早的checkpoint。

原有PPO入口的验证、独立诊断、最终测试和独立正式评估使用相同聚合函数。其学习率平台控制仍由best改善事件驱动，因选模口径改变，其实际衰减时机可能改变。训练PPO使用的轨迹、GAE和奖励不进行三选一。新套件复用选择规则，但使用独立训练/验证实现及预算，不从这些描述推定其调度器或日志格式与原入口相同。

## 数据和身份

- `validation_log.csv` 的主完成率和Q、控制台及Visdom主指标均使用三次取优。另记录原始整体/最差偏好完成率、原始Q、成功0/1/2/3次的组合数量。
- `request_count` 是全部采样请求数，`delivery_count` 是取优后的实例—偏好组合数。例如PPO主线50实例正式验证1950次请求产生650个交付组合；20实例测试3960次请求产生1320个交付组合。
- `raw_sampled` 保留原口径的完整汇总。总推理耗时、总求解耗时、决策数和worker时间累计所有三次请求，而非仅获选请求。
- 保留原始 `sampled_validation_instance_metrics.csv`、`sampled_diagnostic_instance_metrics.csv` 和 `final_sampled_instance_metrics.csv`，增加 `selected_for_aggregation` 标记。对应被选全量行保存在 `selected_validation_instance_metrics.csv`、`selected_diagnostic_instance_metrics.csv`、`final_selected_instance_metrics.csv`，包含采样序号和动作哈希。
- 独立评估保存全部 `instance_metrics.csv` 及 `selected_instance_metrics.csv`。显式重评分仍保留按原生目标选出的方案，不根据新评分偷偷重选。
- 配置、checkpoint元数据和正式评估provenance记录聚合方式与采样预算。新训练协议以 `_best_of_3_v1` 结尾，结果的 `aggregation_protocol_version` 为 `best_of_3_v1`；物理逐轨迹结果schema、奖励和网络观察空间身份不变。

## 新对照套件的协议范围

[comparison_suite_v1](comparator_implementation_spec.md)将方法身份和交付方式分别保存：

| 方法/用途 | 每实例—偏好交付 | 验证与记录 |
|---|---|---|
| 新匹配PPO、DCAN适配版主表 | sampled三次，按成功/Q/repeat取优，best_of_3_v1 | 每次选模21例×13偏好，819条原始轨迹、273份交付 |
| PPO/DCAN的greedy诊断 | greedy单次，greedy_single_v1 | 使用同一best权重，独立臂记录，不反向选模 |
| Rule-DDQN | 贪心规则选择单次，greedy_single_v1 | 每次选模273条原始轨迹/交付，不标为best-of-three |
| NSGA-II-LS、MOEA/D、MO-ALNS | independent_search_v1，本请求截止前最优可行候选 | 完整搜索成本计费；初始化和搜索结果分别可查 |

suite的Rule-DDQN配置将`sample_aggregation`显式设为`per_sample_v1`以兼容配置契约，交付身份仍明确为`greedy_single_v1`，不能据字段名称误读成三次随机采样。suite以逐请求JSON保留`raw`、`selected_repeat`和summary，不承诺输出原PPO入口同名CSV。

三个训练种子11/23/37是三个独立模型，每个模型各自三次取优；搜索的三个种子是随机搜索重复。计时时每模型固定同一组三条采样轨迹重复运行三次，只取时延中位数，交付仍从原组三条中选，不能视为九次选优。短时限搜索另使用三个PPO模型交付时延的中位数。

suite学习率试验和正式训练的选模均使用各方法自身交付协议；正式训练在五个等交互档位验证。训练预算、21/21/7互斥划分及远端执行见[实验协议](experiment_protocol.md)和[启动说明](comparator_remote_startup.md)。三次取优性质仅约束相应候选选择，不证明跨方法预算等价、训练收敛或方法优劣。

## 独立挑战集的交付

独立[challenge套件](challenge_suite_protocol.md)也复用原生三次取优函数，并沿用保存配置的final_test采样种子；旧checkpoint的per_sample选模身份保持原样。其`raw_samples.csv`、`selected_samples.csv`与请求JSON单独保存。启发式单条物理轨迹重评分不属于三次采样；并行PPO评价的请求耗时不作为对照套件的独占硬件交付时限。

## 历史兼容与远端队列

无 `sample_aggregation` 字段的历史完整配置按 `per_sample_v1` 解释，有效配置哈希与原runtime manifest保持不变。即使历史保存配置仍extends默认配置，也不会继承新的聚合方式。需要显式旧口径的新配置可设置 `per_sample_v1`；该模式保留原重复次数约定。

原PPO正式入口的best-of-three模式要求验证和最终测试均恰好3次采样。仅一次采样的独立诊断应使用显式旧口径配置；suite中独立的greedy诊断按上表记录。任何单次交付都不能标记为三次取优正式评估。

历史运行保留其原提交、配置和协议。当前本地可核实lambda=.95/.97/.99的seed11旧协议结果；远端队列的实时状态需另行读取，不能由本地文件推断。本地历史启动脚本已归入[整理快照](project_cleanup.md)，新实验使用通用启动器。

## 验证

单元与入口测试覆盖三选一、失败与质量排序、同分、完整性/安全门槛、旧配置、新配置重载、best与学习率事件、完整原始与被选CSV及独立评估。真实保存日志可执行：

```powershell
.\.venv\Scripts\python.exe -m analysis.verify_best_of_three_protocol
```

该复算仅读取既有lambda=.95/.97/.99、seed11记录。新口径分别重选ep1800/1900/1400；已有原best测试的三次取优Q为.262400/.324509/.313682，均1320/1320成功。中间候选权重没有保存，因此复算不构成这些新候选已完成测试的证据。

## 证明与实验边界

[选模证明](../data/docs/SELECTION.md) 给出候选成功标记、成功候选集内Q最优性、最差偏好完成率及选模规则。它不证明每个原始目标同时改善、PPO优化了三次最大回报，或旧训练改用新聚合后等价于新协议训练。学习率控制与中间权重保存的差异尤其需要保留。

现有 analysis.pareto_analysis / analysis.mo_alns_analysis 的主要指标基于原始候选池，analysis.ablation_analysis 按原始sampling_repeat配对。原PPO入口的交付主指标取训练summary、独立评价metrics及selected文件；新`analysis.comparator_study`已支持suite确认集交付配对，并分别输出13点交付HV和全部付费档案HV。全部消融及正式split汇总的剩余范围见[实验清单](experiment_runlist.md)。不能把原始池HV或逐请求完成率改名为三次取优交付指标。
