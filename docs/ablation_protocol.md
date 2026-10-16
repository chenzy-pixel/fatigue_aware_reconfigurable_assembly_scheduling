# 对照与消融

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

`configs/controls/` 还提供 raw Flow、variance、q=.25/.75、bounded 编码和 lambda=.97/.99 等覆盖项。新参数扫描通过通用启动器显式列出配置；MO-ALNS 和启发式使用相同数据、目标与报告协议，并记录其实际求解预算。

## 交付与原始请求统计

当前训练与独立正式评估按best_of_3_v1选模并保存selected交付。`analysis.ablation_analysis`仍按原始sampling_repeat对齐比较，其结果表示原始请求层面的配对统计。交付层比较应按实例和偏好对齐，允许两臂选中不同repeat，并保留全部原始请求的活动约束和预算记录。完整矩阵、统计缺口及执行顺序见[实验清单](experiment_runlist.md)。
