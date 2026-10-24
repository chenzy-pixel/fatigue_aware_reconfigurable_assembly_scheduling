# 实验设计规格：三因子27例 v1

2026-10-23。完整实施合同与远端命令见[实验体系v1](experiment_suite_v1.md)，机器清单见[experiment_suite_v1.json](../configs/experiment_suite_v1.json)，本地证据见[实施验收](experiment_suite_verification_20261023.md)。

## 6A．评价设计

固定数据、因子定义与目标见[factorial27_spec.md](factorial27_spec.md)。标定54例、验证27例、测试27例，测试三个规模组各9例；数据schema3、观察schema13，尺度在标定集冻结。7种学习配置×5种子，每次2000算例尝试、100个Epoch、20次验证。首轮超参数冻结。

前沿任务为66偏好，进化算法每实例600秒；独立请求为三端点及精确等权，每请求600秒。开发检查30次、额外5小时，正式两类搜索共1350次、预算225小时；训练与推理另计。充分性待正式证据。

## 6B．论断、实验与数据映射

下列均为待检验问题，实现和本地验收不构成性能结论。

|问题|实验配置|主要落盘数据|对应表图|
|---|---|---|---|
|图传播、直接特征和残差分支的贡献|Full及三个单因素变体，各5种子|逐实例交付、配对差值、原始目标和HV|`table_ablation.csv`、`paired_statistics.csv`|
|相对固定规则的质量与时间|Full及五规则|规则轨迹、重评分、完成率、Q及计时口径|`table_rules.csv`、`instance_quality.*`|
|不同学习方法的行为与训练效率|Full、A2C、Rule-DDQN、DCAN-adapted|训练/验证步数、真实更新次数、训练时间、各目标、逐实例结果|`training_summary.csv`、`table_learning.csv`、`learning_curves.*`|
|进化算法质量与求解时间|NSGA-II-LS、MOEA/D，前沿及独立请求分别计费|完整档案、候选完成时刻、搜索计数和60/180/600秒快照|`table_evolution.csv`、`search_curves_*.*`、`frontier_C*_seed11.*`|
|Full解码策略收益与成本|共用Full模型的greedy、sample1、best-of-3|各候选、选中repeat、全部付费时间|`table_decoding.csv`|

统一交付表为`deliveries.csv`；实例汇总为`per_instance.csv`；总体/规模/K/M汇总为`summary.csv`；共同成功结果为`common_success.csv`；独立种子统计为`seed_results.csv`和`seed_variability.csv`。数据路径以运行根目录下`report/`为前缀。

训练源数据包括`training_log.json/csv`、`episodes.json/csv`、`epochs/`、`validations.json`、`validation_requests/`；每次运行保存有效配置、检查点、源码/数据身份和硬件。报告以实例为统计单位，bootstrap和Holm规则以实施合同为准。

## 变更记录

- 2026-10-24复核：补充派生sample1与母结果内容哈希/轨迹对应校验，过期派生缓存自动重建；更新完成后释放训练观察缓存；复用评测池要求环境配置一致且容量足够。实验预算、采样及目标定义保持原合同。
- 2026-10-24：恢复训练验证及前沿质量评测20路环境、集中批量推理、空闲通道续填；验证复用训练worker池。固定每条轨迹的采样种子，保留逐步动作和物理审计。前沿记录批次墙钟及通道用时，独立请求继续独占串行测时。复用Full前沿首样本、缓存固定数据、合并同Epoch重复导出；源码及运行身份随之更新，使用新的输出目录。正式实验仍由用户在远端手动启动。
- 2026-10-23：确认7×5训练矩阵，采用相同2000算例尝试，分别报告实际交互步数；冻结首轮参数及600秒搜索上限。建立两类搜索入口、预算开发门槛、统计与可追溯产物。完成本地验收，正式实验由用户在远端手动启动。
