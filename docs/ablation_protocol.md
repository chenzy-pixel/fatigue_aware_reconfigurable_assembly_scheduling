# 对照与消融

2026-10-23：当前schema13三因子27例实验使用[实验体系v1](experiment_suite_v1.md)及`configs/experiment_suite_v1.json`，包括35次按2000算例尝试训练、五规则和两类搜索。以下保留历史协议记录。

更新：2026-10-17。原有PPO单因素消融与新增跨方法比较分别按下述契约解释。

主线为 HGNN + shared-head + excess Flow + OWA(q=.5)。[当前消融清单](../configs/manifests/ablation_mainline.json) 定义八组训练和四组配对，配置路径见[迁移说明](project_cleanup.md)。

| 对照 | baseline → variant | 改变因素 |
|---|---|---|
| 图传播 | Universal → no_graph | HGNN → node MLP pooling |
| 疲劳 | full → neutral，分别针对 Flow/Cost/Workload | fatigue_mode |

所有配对保持其余物理、目标、尺度、PPO 和正式评估设置相同；工人中性疲劳仍独立重建反事实疲劳暴露。`configs.ablation_protocol` 校验配对配置、模型、来源和评测完整性。

```powershell
python -m scripts.run_experiments --suite structural --seed 11 --preflight-only
python -m scripts.run_experiments --suite structural --seed 11
python -m scripts.run_experiments --suite fatigue --seed 11
python -m scripts.ablation_protocol --seed 11
python -m analysis.ablation_analysis --seed 11
```

`configs/controls/` 还提供 raw Flow、variance、q=.25/.75、bounded 编码和 lambda=.97/.99 等覆盖项。新参数扫描通过通用启动器显式列出配置；跨方法使用相同物理环境、固定实例和公共目标，交付预算与训练机制按方法分别记录。

## 新增跨方法对照

[comparison_suite_v1](comparator_implementation_spec.md)通过独立入口实现NSGA-II-LS、MOEA/D、DCAN适配版、Rule-DDQN，并重新训练匹配PPO。该套件不加入既有八组PPO消融清单，也不由`configs.ablation_protocol`自动证明单因素可比性。

| 比较 | 能检验什么 | 解释边界 |
|---|---|---|
| 匹配PPO ↔ DCAN适配版 | 同环境、目标、交互预算下的表示及条件注意力方案 | 网络结构和候选建模均改变；是方法对照，不能单独归因为某个注意力项 |
| 匹配PPO ↔ Rule-DDQN | 原生候选PPO与规则选择Double-DQN的整体效果及效率 | 状态池化、规则动作、优化和交付协议不同；不能只归因为PPO与DDQN更新公式 |
| MO-ALNS ↔ NSGA-II-LS ↔ MOEA/D | 自适应邻域、精英种群和分解搜索的交付及覆盖 | 共享原生环境/公共评分；NSGA-II有局部搜索、MOEA/D首版无局部搜索，不是仅更换种群选择算子 |
| MO-ALNS ↔ 固定概率LNS | 同邻域/修复及预算下自适应权重的贡献 | 位于单独MO-ALNS研究；保留其历史PPO参照身份，不混作新套件训练重复 |

学习三方法各用11/23/37从头训练，预算为每次350411次实际env.step，学习率试验另计；更新次数、样本重用和墙钟仍可能不同。PPO/DCAN按best-of-three选模和主交付，另列greedy诊断；DDQN按greedy单次选模和交付。对照不要求解码预算相同，报告时必须保留这些差异。

搜索先在独立调参集完成参数及600→必要时1800秒预算曲线，再到7例确认集比较充分搜索与PPO实测交付时限。短时限只得到初始化解，不等价于充分搜索无收益。是否获得合理发挥机会，由平台、机制次数及相对初始化改善证据判断；不以必须战胜PPO为门槛。远端手动命令、21/21/7数据隔离和正式扩展入口见[启动说明](comparator_remote_startup.md)。

## 交付与原始请求统计

原有PPO训练与独立正式评估按best_of_3_v1选模并保存selected交付。`analysis.ablation_analysis`仍按原始sampling_repeat对齐比较，其结果表示原始请求层面的配对统计。交付层比较应按实例和偏好对齐，允许两臂选中不同repeat，并保留全部原始请求的活动约束和预算记录。

新增`analysis.comparator_study`已提供suite确认集的3003份交付覆盖检查、方法自身重复平均后的配对统计、实例bootstrap、13点交付HV及全部付费档案HV。搜索种子不当作独立训练模型。该报告器不自动覆盖原有全部消融或正式split；完整矩阵、剩余统计范围及执行顺序见[实验清单](experiment_runlist.md)。当前只有[本机实现验证](comparator_verification_20261017.md)，远端正式结果尚待实际产物。
