# 合法候选评分与独立价值查询

2026-10-02：V8 HGNN 的默认 `phase_batched_v1` 执行路径按阶段只编码和评分合法 pair，再按原始动作编号 scatter 回完整 logits。
WAIT 保留自己的向量及 mask；工人时长标准化和残差尺度只统计合法动作。HGNN 仍编码完整图，候选稀疏化不改变图信息。

`HeteroGraphActorCritic.value_batch()` 只执行图编码、偏好编码与共享 critic。
PPO 的 rollout cutoff 自举通过该入口求值，跳过动作 embedding、专家评分和偏好残差。
两条路径共享 `_critic_values()`，不增加模型参数。`reference_v8` 保留完整候选参考计算，用于数值、梯度与诊断对照。

## 验证口径

回归覆盖生产/工人混批、变长实例、仅 WAIT、WAIT 被 mask、单一合法 pair、非零专家 context、CPU/CUDA logits/概率/梯度一致性。
模块 hook 检查动作头输入行数等于合法 pair 数；诊断中的动作编号保持原编号。
价值查询与完整 actor-critic 的值和共享编码器梯度对照，并通过 hook 确认动作头不执行。

性能测量使用相同 schema-6 观测与同一份权重，比较主线原完整 pair 阶段批处理和新的稀疏执行。
原网络仅扩展输入维度接收同一份时间特征，用于隔离执行路径差异；它不作为 schema-6 的正式模型保存。
测量包含观测打包、神经网络与动作诊断，不包含环境推进、订单时间估计、多进程或 PPO 更新。
CPU 使用 2 个 PyTorch 线程；每条路径预热 2 次、交替测量 7 次取中位数，GPU 在计时边界同步。

## 本机测量

固定算例启发式轨迹的 16 个状态（8 个生产、8 个工人）中，完整 pair 行数为 4224，实际合法 pair 行数为 30，减少 99.29%。

| 路径 | 原实现中位数 | 新实现中位数 | 比值 |
|---|---:|---:|---:|
| CPU actor+critic | 101.03 ms | 76.64 ms | 1.32× |
| CPU 价值查询 | 104.48 ms | 51.69 ms | 2.02× |
| RTX 4060 Laptop GPU actor+critic | 105.52 ms | 91.83 ms | 1.15× |
| RTX 4060 Laptop GPU 价值查询 | 110.31 ms | 15.14 ms | 7.29× |

同一权重的最大合法 logit 差为 CPU 1.49e-8、GPU 2.98e-8；价值差为 CPU 0、GPU 4.47e-8。
这些是该 batch 的观测值，不代表全部状态或整个训练耗时的加速比例。
PyTorch 2.7.1+cu126；原始交替计时保留在本地 `result/analysis/time_context_inference_20261002/benchmark.json`。
可复跑固定主线基准提交 `ba08892` 的比较：

```powershell
python scripts/benchmark_v8_inference.py --baseline-ref ba08892
```

该脚本的输出默认保存到 `result/audits/inference_20261002`，明确排除环境推进和订单时间估计。

## 集成验证记录

`python -m pytest -q`：263 项通过，2 项长耗时审计按既有默认设置跳过（397.67 秒）。
CPU/CUDA 的前向、概率与梯度对照均通过；schema-5 的原观测通道、mask、启发式动作及奖励基线保持一致。
随后补充迁移后保存/重载的来源哈希检查，相关时间上下文、PPO 和单目标 checkpoint 测试 28 项通过（8.99 秒）。
三个原始 V2 单目标 run 的 `continue_v2.py --check` 均通过，验证原输入权重逐值保留、新时间列为零及原尺度/验证子集契约。

`python -m scripts.run_00_smoke` 在 CUDA 上运行 2 个 16 步 rollout、1 次 PPO 更新，保存并重载 best checkpoint。
13 点验证完成 13/13 条轨迹，66 点最终采样完成 61/66 条，5 条在 horizon 未完成；全部 66 条调度约束检查为零违规。
本次缩小的冒烟复用 1 个验证实例，不是独立测试集上的性能实验；实例聚合的完成计数与逐轨迹完成数口径不同。
本地记录为 `result/runs/universal_protocol_smoke_20261002_022317`。
冒烟确认运行链路与安全检查，调度效果需要正式同预算实验评估。
