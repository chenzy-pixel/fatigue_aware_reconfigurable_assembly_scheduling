# 合法候选计算与独立价值路径

2026-10-01 已完成两项 V8 执行路径优化。

本页测量记录对应 schema-5 网络。随后加入的 schema-6 时间上下文见
[订单时间上下文](order_time_context.md)；下面的权重兼容性和数值对照描述的是这次执行路径优化。

## 实现

`agent/ppo/network_v8.py` 的默认 actor 路径只对 mask 中合法的 pair 和 WAIT 计算动作 embedding、专家分数和偏好残差。工人候选时长标准化继续只读取合法 pair，残差尺度继续只读取合法动作；被 mask 的 WAIT 不参与这两项统计。

候选的原始动作编号、graph ID 和稀疏 mask 按阶段统一传输，生产与工人阶段各通过一次 scatter 将 logits 写回原 batch。非法动作与不同实例尺寸产生的 padding 保持原有最小浮点填充值，策略诊断继续报告原始动作编号。共享图编码使用原有完整图。

网络新增 `value_batch()`，由 `PPOAgent.value_batch()` 调用。价值查询仅执行图编码、偏好编码与共享 critic；rollout cutoff 自举通过此路径求值。两条路径共用 `_critic_values()`。

网络参数、`network_spec` 和 checkpoint schema 保持原结构。已用旧 checkpoint 的同一份权重严格加载修改前后网络并比对规格。`reference_v8` 保留完整候选计算，用于数值和梯度回归。

## 测量

使用 `configs/v8/universal.json`，128 维、两层 HGNN、dropout=0。同一份已保存 V8 Flow checkpoint 权重用于修改前后模型。32 个观测来自固定实例和 `validation/instance_2000000.json` 的启发式轨迹，生产与工人阶段各 16 个。

该 batch 的 pair 计算行数从 **9,472 减少到 54**，下降 **99.43%**。WAIT 仍单独计算。

| 测量项 | 修改前 | 修改后 | 加速比 |
|---|---:|---:|---:|
| CPU actor+critic 推断 | 143.11 ms | 103.49 ms | 1.38× |
| CPU 价值查询 | 143.13 ms | 67.00 ms | 2.14× |
| CPU 前向+反向 | 323.84 ms | 201.84 ms | 1.60× |
| RTX 4060 Laptop GPU actor+critic 推断 | 155.61 ms | 145.37 ms | 1.07× |
| RTX 4060 Laptop GPU 价值查询 | 156.98 ms | 14.60 ms | 10.75× |
| RTX 4060 Laptop GPU 前向+反向 | 63.82 ms | 52.60 ms | 1.21× |

GPU 前向+反向的额外峰值张量内存从 **186.69 MiB 降至 107.28 MiB**，减少 **42.53%**。这不包含测试前已经分配的模型参数，也不是整个训练进程的显存占用。

测量设置：CPU 2 个 PyTorch 线程；推断预热 3 次、交替测量 15 次取中位数；前向+反向预热 2 次、交替测量 7 次取中位数。GPU 在计时边界同步。推断包含原有动作诊断，前向+反向使用价值平方损失与合法动作负 log-probability，不包含环境推进和优化器 step。

同一权重下，观测到的最大合法 logit 绝对差约 **1.05e-7**，价值路径的最大差约 **1.20e-7**。这些是浮点计算的观测差异；不是对后续全部训练轨迹完全一致的保证。具体耗时取决于候选稀疏度、batch、硬件与诊断开销，上表不能直接换算为完整训练加速比。

本地原始记录与保存的修改前源码位于 `result/audits/legal_candidates_20261001/`，按仓库规则被 Git 忽略。可在当前工作区重跑：

```powershell
.\.venv\Scripts\python.exe result\audits\legal_candidates_20261001\benchmark.py
```

当前脚本以 schema-5 输入视图对照新增时间输入补零后的 schema-6 网络，输出
`benchmark_schema6_migration.json`；本页原始测量仍保存在 `benchmark.json`。

## 验证

`test/test_ppo_batching.py` 扩展以下保护：

- CPU/CUDA 上与完整候选参考路径的 logits、动作概率、价值及参数梯度对照；覆盖 HGNN、node-MLP、共享偏好头和非零专家 context 权重。
- 不同实例尺寸、生产/工人混批、只有 WAIT、WAIT 被 mask、只有一个合法 pair、整个阶段没有合法 pair。
- 用模块 hook 确认 pair 动作头实际处理的行数就是合法 pair 数。
- 价值路径的输出与共享编码器/critic 梯度一致，动作头和动作上下文 projector 不执行。
- mask 形状和合法性校验，以及动作诊断的原始编号和浮点统计。

网络、PPO、时长标准化与消融变体的 **58 项核心回归通过**；图观测与 rollout collector 的 **22 项回归通过**，合计 **80 项**。collector 检查包含 cutoff 自举、真实失败终止、串并行 greedy/sample 轨迹一致性和随机数状态保持。本次未执行该文件的训练缓存持久化及 20-worker 缓存压力测试。

CUDA 端到端小规模训练也通过：2 个 16 步 cutoff rollout、1 次 PPO update、1 条 sampled validation，以及选中 best checkpoint 后重载执行 1 条独立 sampled final-test。验证和最终测试均完成，安全检查通过，最终调度违例为 0。该检查用时约 55 秒，产物在 `result/runs/legal_candidate_smoke_20261001/`；它验证训练链路可运行，不构成调度质量的统计实验。
