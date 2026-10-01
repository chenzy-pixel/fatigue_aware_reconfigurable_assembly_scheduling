# V2 归一化尺度更新

2026-10-02，按用户指定，全部目标归一化统一为：

| 目标 | 尺度 |
|---|---:|
| Flow | 1089.15 |
| Cost | 353.27 |
| Variance | 2.2629 |

主配置、单目标/Universal/MO-ALNS 继承配置、目标相关图特征、偏好奖励、固定参考质量和 Pareto/HV 使用同一组尺度。
`q_i=J_i/(s_i+J_i)` 及偏好增强切比雪夫公式不变；固定参考质量的权重仍为 `(0.5,0.3,0.2)`，版本更新为 `canonical_bounded_quality_v2`。

清单：`configs/manifests/v2_selected_scales_20261002.json`。
SHA256：`9d4829c467f592d9a319251f0f779e7354cce70e1511d3bc90ab19b78fa4cce3`。
原始验证均值、轮次、成功数、舍入位数与来源日志哈希均写入清单。

## 在运行训练的电脑上同步代码

主分支合并后，在项目根目录同步 Git 代码，然后用原 `(drl)` 环境检查生效配置：

```powershell
git switch main
git pull --ff-only origin main
python -c "from configs import load_config; c=load_config('configs/default.json'); print(c['objective_scalarizer']['scales']); print(c['evaluation']['quality_metric'])"
```

默认配置更新目标尺度、对应清单加载和质量指标版本。历史归档配置、原 E1 清单和已有运行结果保留。

## checkpoint 和结果解释

新的归一化哈希进入 network spec。旧 checkpoint 仍使用原尺度，按新配置直接加载会拒绝不匹配。
如要从旧权重进入新尺度，应建立单独的迁移训练阶段并记录旧/新尺度、来源权重和重新验证结果。
`continue_v2.py` 对应已经完成的原尺度追加阶段，会继承源 run 的有效配置；它不是新尺度迁移入口。
新旧 reward 数值不直接拼接；原始 Flow/Cost/Variance 和完成率仍可按固定评测矩阵比较。

## 验证

- 71 项相关测试通过，覆盖清单完整性、配置继承、偏好奖励、参考质量、评估、Pareto、MO-ALNS 和 checkpoint 哈希。
- 三个目标在原始目标等于对应尺度时，单目标归一化代价均为 0.5。
- 固定算例的新旧尺度比较：动作掩码、211 步启发式动作序列、原始三目标和完成状态一致；归一化观测和奖励哈希按新尺度更新。
- V2 生成器仍为 2.0.0、结果 schema 仍为 8.0.0、固定数据分布协议哈希不变。
