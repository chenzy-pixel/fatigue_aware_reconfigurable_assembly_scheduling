# 五组消融实验

2026-10-03。五组消融使用当前 V2 算例、schema-6 时间上下文、冻结尺度
`(1089.15,353.27,2.2629)` 和全局失败惩罚 `2.0`。默认算法种子为 11。

| 变体 | 对照 | 改变的组件 | 训练轮数 | 验证间隔 |
|---|---|---|---:|---:|
| no_graph | Universal | 节点 MLP 与池化替代异质图消息传播 | 2000 | 100 |
| shared_head | Universal | 共享偏好动作评分头替代目标专家头 | 2000 | 100 |
| neutral_flow | 单目标 Flow | 疲劳中性动力学 | 1000 | 40 |
| neutral_cost | 单目标 Cost | 疲劳中性动力学 | 1000 | 40 |
| neutral_variance | 单目标 Variance | 疲劳中性动力学 | 1000 | 40 |

所有变体继承对照的训练、验证各 20 个 worker、每次更新 20 个 episode、PPO 参数及数据课程。
结构消融使用 13 点验证和 66 点最终测试，每个实例每个偏好采样 3 次。
三个疲劳中性变体与各自单目标对照使用同一 one-hot 偏好。

## 疲劳中性动力学与审计

疲劳中性模式将初始疲劳、拆装时间的疲劳系数、疲劳积累与恢复率置为零。
工人资格、机器适用性、资源占用、工序先后关系及时间上限继续约束动作。
原始算例完整保留，固定数据及在线缓存依据完整物理基准构造；诊断和候选接受使用完整疲劳模型。
`fatigue_mode` 记录在实际运行配置与 checkpoint 元数据中，实例分布与缓存指纹共用物理基准设置。

环境依据原始疲劳参数对实际执行的拆装时间线进行独立审计，记录峰值、超限工人比例、
超限时间和超限面积，以及工人忙碌/空闲与重构次数指标。该审计描述疲劳中性调度的条件性暴露，
其拆装持续时间来自中性动力学，不能解释为完整疲劳系统中的同一条可执行调度。

`active_constraint_pass` 表示所选模型的活动约束通过；`physical_safety_pass` 额外检验原始疲劳安全线。
疲劳中性模型按活动约束、完成率优先规则选 best，同时报告原始疲劳下的安全结果。
checkpoint 加载检查网络变体及疲劳模式，历史完整模型按默认 HGNN、专家头和完整疲劳识别。
部分拆装任务的回报由 `reward_preference_quality_score` 重建，成功轨迹继续使用真实终局目标。

## 启动与结果

在项目根目录执行：

```powershell
python scripts/run_10_ablation_smoke.py
python scripts/run_11_train_structural.py
python scripts/run_12_train_neutral.py
```

第一条为五组小规模 PPO 更新、验证和重载检查；后两条分别顺序训练两组结构消融和三组疲劳中性消融。
Universal 和完整疲劳的三个单目标基线分别训练。各入口支持 `--seed`、`--device` 和 `--dry-run`；
默认正式入口使用配置中的设备，冒烟入口自动选择可用 CUDA 或 CPU。
每批结果带时间戳，清单写入 `result/runs/ablation_batches/`，保存各变体训练目录与 checkpoint 路径。

完成九个 seed11 模型后执行：

```powershell
python scripts/run_13_evaluate_ablations.py
python scripts/run_14_summarize_ablations.py
```

评估入口先查找全部九个已完成、协议匹配的 best checkpoint，按当前配置重算测试集结果。
查找检查算法种子、训练预算、网络变体、疲劳模式、数据生成规则、验证池哈希、完整标量化参数、训练初始化、尺度、惩罚及 checkpoint 文件哈希。
评估清单记录对应训练及评估目录；单目标评估额外导出 `fatigue_segments.csv`。

汇总先核验九个角色的训练配置、评估配置、checkpoint 和 CSV 来源哈希，要求完整测试集、偏好与重复矩阵。
随后按相同实例、偏好、采样重复及派生随机种子配对，检查数据与尺度哈希以及失败惩罚。
完成数统计全部轨迹，目标差仅统计两侧共同成功的轨迹。产出 `paired_cells.csv`、`summary.csv`、
来源清单和 `report.md`。默认汇总最新完成的 seed11 评估清单，也可显式指定 `--manifest`。
一个训练种子的结果属于描述性证据；跨训练种子的稳定性需使用匹配的多种子实验。
