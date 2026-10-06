# 当前代码架构

本文描述单阶段 PPO 主线的模块边界与运行契约。2026-10-02 更新的主实验协议为：
Flow/Cost/工人负荷方差尺度 `(1089.15, 353.27, 2.2629)`，训练验证固定 13 点、
每 100 轮一次，最终评估固定 66 点。完整来源和偏好表见
[主实验协议](experiment_protocol.md)。

Universal 的可执行配置为 `configs/v8/universal.json`，工人 Flow 输入使用
`candidate_zscore_v1`，标准差下限为 `0.001`。网络规格与 checkpoint 同时记录这些值。

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
| Observation | schema 10，9 维全局输入、14 类关系与 order_chain_action_context_v2 |
| Reward | `single_stage_progress_quality_failure_v3` |
| 训练协议 | `single_stage_lexicographic_failure_v3` |

## 3. 环境契约

`AssemblySchedulingEnv` 对外提供：

| 接口 | 契约 |
|---|---|
| `reset(instance, preference=...)` | 固定订单/工序进度分母，初始化事件、状态和 `P_0,Q_0` |
| `observe()` | 生成 schema-10 异质图与三目标偏好字段 |
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
  failure                      仅任务失败终止步骤为 -lambda（当前默认 2）
  feasibility_shaping          可选势函数差分
```

当前训练标量为 `operation_progress + quality + failure`。势函数需要的资源余量与
时间裕量计算仍服务于资源诊断和可选实验配置。

`proxy_return_from_metrics` 按一般形式重算：

\[
P_T-P_0+Q_0-Q_T-\lambda I.
\]

collector 在 episode 结束时检查累计基础奖励与代理回报在 `1e-8` 内一致。

已确认的主实验使用 `q_i=J_i/(s_i+J_i)`，固定尺度为 Flow=1089.15、
Cost=353.27、工人负荷方差=2.2629。三项来自用户选定的 V2 续训验证参考值，
取值轮次分别为 500、500、360，按指定精度舍入。来源、取值方式和哈希
记录于 `configs/manifests/v2_selected_scales_20261002.json`、有效配置及 provenance。
`evaluation.quality_metric` 的 v2 参考分数共用这三个尺度，与使用偏好增强切比雪夫公式的
`preference_quality_score` 分别报告。

### 3.3 终止顺序

`task_succeeded` 表示全部工序完成；`task_failed` 表示环境层失败。事件推进顺序是：

```text
推进到目标 tick
  → 处理该 tick 全部事件
  → 刷新 READY/DONE 状态
  → 检查全部工序完成
  → 检查 horizon 与其他失败条件
```

训练奖励始终使用实际 `Q_T`；环境失败另扣一次配置的惩罚（当前默认 2）。正式评测仍可把失败质量标记为
1，且该字段不进入 v3 回报重建。PPO collector 只对真实任务成功或失败提交
`done=True` 和 `last_value=0`。采集步数 cutoff 保持 `done=False`，并从实际下一
observation 调用 `value_batch` 自举。

## 4. 网络与 PPO

`agent/ppo/network.py` 的 V8 网络包含：

```text
HeterogeneousGraphObservation
  → node-type encoders
  → 2 层 heterogeneous message passing
  → Flow / Cost / Variance action experts
  → preference-conditioned expert mixture
  → production / worker / WAIT logits + state value
```

图包含六类节点、十四种关系和两层 HGNN 消息传递。偏好是 episode 级三维字段，
经 `3 → 32 → ReLU → 32` 编码后供 actor 与 critic 共用，不拼入图全局特征。
有属性关系的消息为 `ReLU(Linear([邻居表示, 边属性]))`，正向和反向均在
`index_add` 前计算；零属性关系继续使用 Linear。所有进入节点的关系消息统一按总度数
求均值，再做原有残差、LayerNorm 和 ReLU，图读出按节点类型求均值。
`configs/network_contract.py` 统一生成 HGNN 的 `attributed_joint_relu_v1` /
`total_degree_mean_v1` 消息身份；node-MLP 两项为 `not_applicable`。
身份进入 normalized network config、network spec 与 runtime manifest。
图消融校验允许该身份随 encoder_variant 派生变化，其他配对严格一致。
PPO 使用 clipped policy loss、
value loss、entropy、GAE 和梯度裁剪。配置强制 `gamma=1`、`dropout=0`。
全局输入共 9 维：当前时间、待重构比例、工序完成比例、生产阶段标志、工人匹配缺口、
最小工人备选比例，以及累计 Flow/尺度、累计成本/尺度、承诺负荷方差/尺度。
三目标读取奖励使用的 `_objective_vector()`，不裁剪。完整名称与顺序进入 network spec，
actor、critic 和 WAIT 上下文共享这一输入。order 节点继续保留 released、completed；
其差值的均值表达全系统活跃订单比例。

训练、评估与配置快照要求 schema 10，旧 schema 5/6/7/8/9 模型须重新训练。
同为 schema 10 的旧消息结构检查点若缺少或不匹配当前计算身份，也需重新训练。
加载在网络和 optimizer 状态修改前完成校验，迁移开关不能绕过该边界。
决策保护触发 sampling_truncated，保留末状态并做价值自举；真实完成、期限失败及死锁
使用真实终止标志。实际加工和工人服务使用独立、零属性的双向 processing_on/served_by
关系，严格核对两端实体状态。候选时间、费用、方差和订单时间上下文共用安全顺序
投影，见 [schema 10 说明](schema10_sequential_projection.md)。
生产、工人和 WAIT 动作各有三目标专家；直连 ranker 使用固定符号的归一化
`softplus(theta)` 权重，偏好残差以合法动作基础 logit 的标准差缩放。
Universal 的工人 Flow 专家将合法候选工期标准化为 `candidate_zscore_v1`，
标准差下限 `0.001`；动作与上下文编码仍保留绝对工期。

当前时间上下文将整个订单预计裕量放入订单节点和生产候选边，工人候选另带当前待拆/待装阶段的等待年龄，
WAIT 向量记录下一已知事件后的最小裕量及其变化。它们经动作 embedding 进入各目标专家独立的 context MLP。
估计纳入已知资源占用和疲劳恢复，未来未确定的跨订单竞争仍可能使其偏乐观；字段定义见
[订单时间上下文](order_time_context.md)。

默认阶段批处理只编码和评分合法 pair，再按原动作编号恢复完整 logits；HGNN 编码完整图。
`value_batch()` 仅运行图编码、偏好编码和共享 critic，用于采集 cutoff 自举。
执行对照和本机计时见 [推理优化](graph_network_performance.md)。


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

Universal 按评估阶段提供偏好集合：

- 训练验证：每 100 个 episode，固定 50 个验证实例、3 次 sampled 采样、13 个偏好，
  共 1950 条轨迹。偏好为 3 个端点、3 个边中点、等权中心和 `(0.6,0.3,0.1)` 的 6 种排列，
  顺序见主实验协议。
- 最终评估：重载固定 best checkpoint，以独立采样种子在测试集上评估 66 点 step-0.1
  simplex，每实例每偏好采样 3 次；结果用于最终报告。

聚合使用当前评估阶段的完整偏好集合：

1. 对每个偏好统计 sampled 成功率；训练 checkpoint 排名完成率取 13 个验证偏好值的最小值。
2. 每条成功轨迹按自身偏好读取 `preference_quality_score`。
3. 先在每个偏好内部求成功轨迹质量均值。
4. 再对该阶段的全部偏好等权平均（验证 13 个、最终评估 66 个）。
5. 任一偏好没有成功轨迹时，聚合质量为 `+inf`。

`cell_count`、`completed_cell_count` 统计逐轨迹单元；`instance_count` 统计独立实例，
`completed_count` 统计该阶段全部偏好和采样重复都成功的实例。

单目标配置复用同一聚合函数，其偏好集合只有对应 one-hot 点。

`eval.py` 与 `train.py` 从配置读取当前阶段偏好集合，并校验逐偏好单元完整性。
训练中的端点/Sobol 偏好采样独立于评估网格。

## 7. 可复现评测

`training.formal_evaluation` 固定 sampled decoding 和 temperature `1.0`：

- validation root seeds：`algorithm_seed + 100000 + repeat`；
- final-test root seeds：`algorithm_seed + 300000 + repeat`；
- 派生 seed：RNG 版本、root seed、instance ID、preference key 的 SHA256 前 8 字节。

checkpoint metadata 保存数据 manifest hash、实例顺序、实例 seed、重复次数、固定
偏好集合、root seeds、派生规则和温度。`result/provenance.py` 进一步保存有效配置、
源码/Git 状态、数据集和网络权重指纹。
checkpoint metadata 记录选模所用的 13 点验证集合，最终评估 provenance
记录所用的 66 点测试集合，并共同记录固定尺度的来源与哈希。

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
| `test_v8_protocol.py` | 13 点验证、66 点最终评估、偏好等权聚合、固定尺度清单 |
| `test_repository_contract.py` | manifest 字节哈希、黄金 observation/mask/action/reward、支持的配置集合 |
| `test_ppo*.py` | GAE、batching、更新与 checkpoint 兼容 |

固定数据的 manifest 顺序和文件 SHA256 是正式评测协议的一部分。
