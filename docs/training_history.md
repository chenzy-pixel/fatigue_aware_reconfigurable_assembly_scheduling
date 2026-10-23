# Epoch 与三目标训练记录

2026-10-23，日志契约 `epoch_history_v1`，补充三因子27例实验协议。

## 计数口径

- `epoch` 从1开始，指一次采样批次及其学习更新。默认每批最多启动20个算例，预算边界处可能不足20个；它不表示固定数据集完整遍历一次。
- `episode` 从1开始，对应一次在线生成算例的调度尝试；`episode_index` 是内部从0开始的索引。实例ID、生成种子、参数组合和记录哈希一并保存。
- `sampling_round` 每27次算例启动增加1，对应一次27类参数组合覆盖。它与 epoch 独立；完成求解的覆盖情况应结合成功率检查。
- `optimizer_epochs_per_batch` 是 PPO/DCAN 对一批经验重复优化的轮数；`optimizer_updates` 是累计梯度更新次数，`sample_presentations` 是累计训练样本呈现次数。DDQN 另记实际 TD loss。
- `training_steps` 是累计实际环境交互步数。逐算例另含 `environment_steps` 和环境报告的 `decision_count`。WAIT 计入交互，记录快照不推进环境。

PPO/DCAN 训练目标来自本 epoch 更新前的采样策略，随后进行学习更新；固定验证在更新后进行。DDQN 在采样过程中学习。跨方法对照同时使用 `training_steps`，同一 epoch 不代表相同交互预算。

## 文件与字段

正式入口 `scripts.comparison_suite` / `scripts.run_experiments` 的每个训练目录产生：

| 文件 | 内容 |
|---|---|
| `epochs/epoch_000001.json` 等 | 每 epoch 原子写入的汇总及完整逐算例指标 |
| `episodes.json` / `episodes.csv` | 每算例的 epoch、三目标、决策次数、参数、偏好、奖励、耗时、终止状态 |
| `training_log.json` / `training_log.csv` | 每 epoch 汇总、loss、学习率、累计交互/梯度次数、耗时 |
| `epoch_objectives.csv` | 每 epoch 全体、规模、参数组合、偏好及组合×偏好的三目标统计 |
| `validations.json` | 验证请求、全部采样候选及被选中结果的原始证据 |
| `validation_instances.csv` | 验证 epoch、固定算例、偏好、被选中三目标及决策次数 |
| `validation_objectives.csv` | 固定验证集的分组三目标曲线数据 |
| `summary.json` / checkpoint metadata | 总 epoch 数、累计交互步数；checkpoint 对应的 epoch |

`objective_flow / objective_cost / objective_workload` 分别是完整合法调度的 Excess Flow、重构费用、工人负荷目标。默认负荷目标为 geometric OWA，variance 对照由 `metric_workload_mode` 区分。

阶段值保存为 `snapshot_flow / snapshot_cost / snapshot_workload`；`snapshot_time` 是当前仿真时间，只有完整合法解才填写 `cmax`。`objective_valid` 必须为真才能纳入完整解目标曲线。失败、工程截断或安全校验失败时完整目标列留空，不能填0。某 epoch 没有完整合法解时，目标均值留空。

环境指标另以 `metric_` 前缀保留，含完成比例、重构/复用/等待次数、最大疲劳、疲劳阈值、工人负荷及峰值疲劳分布、奖励分项、终止原因等。字典和列表在 CSV 内用 JSON 编码；原始 JSON 保留嵌套结构。耗时包含采样、学习更新、生成、重置、环境步进、观测与指标计算等；并行实例墙钟耗时不能直接相加当作整轮耗时。

`train.py --smoke` 保留 `train_log.csv` / `update_log.csv` 命名，增加相同 epoch、有效目标列、`epoch_objectives.csv` 和逐 epoch 原始记录；验证明细增加 epoch 和累计环境步数。

正式训练统一通过 `scripts.comparison_suite` / `scripts.run_experiments` 核验预算清单身份并执行准确交互额度；`train.py` 不接受通过任意非空 `budget_manifest` 开启正式训练。

## 论文作图

训练曲线：筛选 `epoch_objectives.csv` 的 `group_by=scale` 或 `cell_id`，以 epoch 为横轴，以 `flow_mean / cost_mean / workload_mean` 为纵轴，三目标分别画图。同步检查 `valid_delivery_rate`、有效样本数和截断数。保留三目标原始单位，不使用加权分数替代三个目标。

在线训练实例及偏好持续变化，混合均值的变化也可能来自样本构成。论文中的学习效果应同时给出固定验证集曲线，建议进一步固定偏好和规模/参数组合。多训练种子的均值和误差带应由独立 run 计算；epoch 内实例间标准差不是训练种子间误差。

当前正式入口沿用交互预算20%、40%、60%、80%、100%的验证节点；每 epoch 都记录训练目标，验证目标仅在真实验证的 epoch 有数据。不得把5个验证点复制或插值成逐 epoch 实测结果。后续需要更密的固定验证曲线时，应明确增加验证频率并记录额外验证耗时。

CSV 每批刷新，单文件原子替换，逐 epoch JSON 可独立读取。中断请求仍按现有规则从头重启，不声明优化器、回放池和工作进程精确恢复。训练结束后可重新导出：

```powershell
.venv/Scripts/python.exe -m scripts.summarize_factorial27 <训练输出目录>
```

旧运行无法补出当时未记录的快照或 epoch 数据；新运行使用新日志身份。
