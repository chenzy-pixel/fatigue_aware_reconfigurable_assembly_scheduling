# 当前代码架构

本文描述单阶段 PPO 主线的模块边界与运行契约。实验数值以
`configs/default.json` 和目标覆盖配置为准。

## 1. 分层与依赖

```text
configs → data → environment → agent/ppo → training + train.py → result
                         ↘ agent/mo_alns ↗
```

- `configs/` 装配有效配置并生成只读 runtime manifest。
- `data/` 提供实例模型、在线训练实例和 manifest 固定评测集。
- `environment/` 实现离散事件、约束、mask、观测、奖励和指标。
- `agent/ppo/` 实现 V8 HGNN actor-critic、GAE、PPO 与并行 collector。
- `training/` 实现 sampled 验证的字典序 checkpoint 选择器。
- `result/` 负责 schema、日志、provenance、checkpoint 和可视化。

环境不依赖 PPO；PPO 与 MO-ALNS 共用相同实例、约束和指标口径。

## 2. 固定运行身份

`configs/runtime.py` 生成当前实现身份：

| 领域 | 实现 |
|---|---|
| 生产动作 | `pair_plus_wait_v1` |
| 工人动作 | `pair_plus_wait_v1` |
| 策略头 | V8 objective experts |
| Pair 可行性 | `instant_physical_pair_mask_v1` |
| WAIT mask | `progress_certified_wait_v2` |
| Observation | schema 5 |
| Reward | `single_stage_progress_quality_failure_v2` |
| 训练协议 | `single_stage_lexicographic_failure_v2` |

## 3. 环境契约

`AssemblySchedulingEnv` 对外提供：

| 接口 | 契约 |
|---|---|
| `reset(instance, preference=...)` | 固定订单/工序进度分母，初始化事件、状态和 `P_0,Q_0` |
| `observe()` | 生成 schema-5 异质图与三目标偏好字段 |
| `get_action_mask()` | 返回当前生产或工人阶段的精确合法动作 mask |
| `step(action)` | 执行动作、推进事件并返回 `RewardVector` 与真实任务终止状态 |
| `metrics()` | 返回目标、进度、质量、终止、安全和资源诊断 |
| `validate_schedule()` | 检查先后约束、资源冲突和调度一致性 |

### 3.1 工序进度

reset 时建立不可变映射
`_progress_order_operation_indices: tuple[tuple[int, ...], ...]`，覆盖算例全部
订单。任意状态的进度为

\[
P_t=\frac1N\sum_i\frac{\#\{o\in i:o.state=\mathrm{DONE}\}}{n_i}.
\]

订单释放只更新运行状态。一次 WAIT 会处理目标 tick 的全部事件，因此同 tick
完成的多道工序共同进入一个 `operation_progress` 增量。

### 3.2 质量与奖励

`environment/types.py` 中的 `bounded_quality_score` 延续当前归一化 augmented
Tchebycheff 标量器。每个 episode 使用自身的固定偏好。环境 step 返回：

```text
RewardVector
  flow, cost, variance          原始诊断差分
  operation_progress           P(t+1)-P(t)
  quality                      -(Q(t+1)-Q(t))
  failure                      仅任务失败终止步骤为 -1
  feasibility_shaping          可选势函数差分
```

当前训练标量为 `operation_progress + quality + failure`。势函数需要的资源余量与
时间裕量计算仍服务于资源诊断和可选实验配置。

`proxy_return_from_metrics` 按一般形式重算：

\[
P_T-P_0+Q_0-Q_T-I.
\]

collector 在 episode 结束时检查累计基础奖励与代理回报在 `1e-8` 内一致。

### 3.3 终止顺序

`task_succeeded` 表示全部工序完成；`task_failed` 表示环境层失败。事件推进顺序是：

```text
推进到目标 tick
  → 处理该 tick 全部事件
  → 刷新 READY/DONE 状态
  → 检查全部工序完成
  → 检查 horizon 与其他失败条件
```

训练奖励始终使用实际 `Q_T`；环境失败另扣一次 1。正式评测仍可把失败质量标记为
1，且该字段不进入 v2 回报重建。PPO collector 只对真实任务成功或失败提交
`done=True` 和 `last_value=0`。采集步数 cutoff 保持 `done=False`，并从实际下一
observation 调用 `value_batch` 自举。

## 4. 网络与 PPO

`agent/ppo/network.py` 的 V8 网络包含：

```text
PolicyObservation
  → node-type encoders
  → 2 层 heterogeneous message passing
  → Flow / Cost / Variance action experts
  → preference-conditioned expert mixture
  → production / worker / WAIT logits + state value
```

偏好是 episode 级三维字段，不拼入图全局特征。PPO 使用 clipped policy loss、
value loss、entropy、GAE 和梯度裁剪。正式配置强制 `gamma=1`。

`agent/ppo/parallel.py` 对任意 worker 数使用同一进程协议：

- episode `k` 对应确定性的在线实例 seed；
- 从 episode 0 调用当前质量偏好生成器；
- 串行与并行 rollout 共享动作采样与 GAE 语义；
- sampled 评测为每个 instance-preference 单元派生独立 RNG seed；
- worker 超时、异常和诊断通过统一响应协议返回。

## 5. 训练与 checkpoint

`train.py` 的训练循环：

```text
collect_training_batch
  → PPOAgent.update
  → 固定 manifest sampled validation
  → LexicographicCheckpointSelector.observe
  → 条件写 best_checkpoint.pt
  → 始终写 last_checkpoint.pt
  → 从磁盘重载 best
  → 独立 final-test sampled
```

`training/protocol.py` 只维护以下状态：安全合格次数、最佳 sampled 完成率、最佳
偏好等权质量、最佳 episode、改善次数与平局次数。首个安全候选建立 best；之后
先比较完成率，再在 `1e-12` 完成率平局时比较质量。完全平局不写 best 文件。

学习率 plateau controller 只更新优化器学习率。`--initial-checkpoint` 是显式的
网络权重初始化入口。

## 6. Universal 聚合

Universal 正式验证固定为 66 点 step-0.1 simplex：

1. 对每个偏好统计 sampled 成功率；排名完成率取 66 个值的最小值。
2. 每条成功轨迹按自身偏好读取 `preference_quality_score`。
3. 先在每个偏好内部求成功轨迹质量均值。
4. 再对 66 个偏好等权平均。
5. 任一偏好没有成功轨迹时，聚合质量为 `+inf`。

单目标配置复用同一聚合函数，其偏好集合只有对应 one-hot 点。

## 7. 可复现评测

`training.formal_evaluation` 固定 sampled decoding 和 temperature `1.0`：

- validation root seeds：`algorithm_seed + 100000 + repeat`；
- final-test root seeds：`algorithm_seed + 300000 + repeat`；
- 派生 seed：RNG 版本、root seed、instance ID、preference key 的 SHA256 前 8 字节。

checkpoint metadata 保存数据 manifest hash、实例顺序、实例 seed、重复次数、固定
偏好集合、root seeds、派生规则和温度。`result/provenance.py` 进一步保存有效配置、
源码/Git 状态、数据集和网络权重指纹。

## 8. 结果文件

每个训练 run 包含：

- `config.json`、`terminal.log`、`summary.json`；
- `train_log.csv`、`update_log.csv`、`validation_log.csv`；
- sampled validation 逐轨迹 CSV；
- `last_checkpoint.pt`；
- 安全候选存在时的 `best_checkpoint.pt`；
- sampled final-test 逐轨迹 CSV。

`summary.json` 记录 final sampled 聚合、偏好质量、工序进度回报和失败
轨迹进度分布。

## 9. 测试边界

| 测试 | 保护内容 |
|---|---|
| `test_single_stage_reward.py` | 固定分母、精确进度增量、一般回报恒等式、失败质量 |
| `test_wait_action.py` | WAIT 证书与 horizon 临界完成 |
| `test_parallel_rollout.py` | cutoff 自举、真实终止、串并行确定性 |
| `test_e1_single_objective.py` | checkpoint 初始化、改善、平局、安全性和 metadata |
| `test_v8_protocol.py` | 66 点聚合、偏好内均值、偏好间等权、`+inf` 规则 |
| `test_latest_only_audit.py` | manifest 字节哈希、黄金 observation/mask/action/reward |
| `test_ppo*.py` | GAE、batching、更新与 checkpoint 兼容 |

固定数据的 manifest 顺序和文件 SHA256 是正式评测协议的一部分。
