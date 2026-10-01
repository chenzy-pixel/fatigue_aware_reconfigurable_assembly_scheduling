# V8 图网络冗余检查

本文记录 2026-10-01 优化前的审计结果。合法候选计算与独立价值路径已随后实现；
实现说明与验证结果见[优化记录](graph_network_performance.md)。下文的计算行数和调用关系为审计基线。

检查日期：2026-10-01。范围：`configs/v8/universal.json` 对应的 schema-5、128 维、两层 HGNN、三目标专家 actor-critic。

结论：存在明确的恒零输入、重复特征、无效权重和多余计算。完整节点类型与专家分支都有使用路径；本次证据不足以认定某一整类节点或某一消息关系对调度质量无用。最值得先优化的是 actor 对被 mask 的候选的计算。

## 检查证据与边界

- 实际执行网络为 `agent/ppo/network_v8.py`。`agent/ppo/network.py:1728` 重新导入 V8 公共接口，上面的 V7 实现不是当前模型。
- 使用固定实例和 `instance_2000000.json` 至 `instance_2000006.json` 七种压力类型的验证实例，分别执行启发式与 seed=11 随机策略，共 16 条轨迹、5,099 个非终止决策状态。15 条完成，1 条随机轨迹到达 horizon。
- 32 个生产/工人观测上的初始化网络反向传播检查显示，当前各模块的参数均接入计算图。该检查只证明计算连接，不能作为训练后特征重要性或策略效果的证据；专家 context 输出层初始化为零，前层初始梯度为零是预期行为。
- 现有 `test_graph_observation.py`、`test_hetero_gnn.py`、`test_v8_policy.py` 共 **25 项测试通过**，用时 18.73 秒。
- 原始数据在 `result/analysis/graph_network_audit_20261001/audit.json`。此目录按仓库规则被 Git 忽略。
- 本次只做审计，网络与环境源码、配置、checkpoint 均未修改。

## 1. 确定的无效输入与参数

### 四个恒零输入列

| 位置 | 特征 | 证据 | 对应无效输入权重数 |
|---|---|---|---:|
| machine 节点 | `target_module_A0` | 初始模块均为有效模块；重构 target 是工序要求模块；拆卸时也保留该有效 target | 128 |
| locked 边 | `source_module_A0` | 可提交生产动作要求机器当前模块不是 A0；该 source 保存提交时的模块 | 256 |
| locked 边 | `target_module_A0` | target 来自工序要求模块，属于实际模块集合 | 256 |
| WAIT 特征 | `estimated_load_variance_delta_if_wait` | `env.py:1546` 构造值固定为 `0.0` | 128 |

在 5,099 个状态中这四列均为 0；其原因可由当前合法状态构造规则确认。两层消息传递下合计 **768 个输入权重**不会影响输出，梯度恒为 0。此结论适用于当前合法实例与可达仿真状态。

WAIT 方差的直接专家并非输出 0：`_signed_unit(0)=0.5`，经过负号和 tanh 后为约 **-0.462117**。删去原始零输入时，应保留或等价迁移这个直接项，不能将整个 WAIT 方差专家一起删掉；其 context 分支仍可学习等待的长期效果。

### 九个单特征 ranker 参数

`network_v8.py:238` 的权重为 `softplus(theta) / sum(softplus(theta))`。一个特征时该权重恒为 1，theta 的梯度恒为 0。

涉及生产 Variance、工人 Flow、工人 Variance，以及生产 WAIT 和工人 WAIT 的各三个专家，共 **9 个参数**。修改 theta 并反向传播已验证权重为 1、梯度为 0。可用固定权重实现同一函数。

上述 768+9=777 个确定无效参数，占本次网络 1,130,468 个参数约 **0.069%**；主要收益是输入与实现更清晰，参数节省很小。

### 恒为 1 的边属性

`precedence`、`can_install.qualified`、`can_disassemble.qualified`、`belongs_to_order`、`belongs_to_wave`、`requires_module`、`qualified_for_module` 共七列，在存在的边上都为 1。

`network_v8.py:344` 的消息变换是带 bias 的 Linear。常数列贡献可以并入对应关系的 bias，边属性本身没有新增信息。两层、128 维时，可消除 **1,792 个冗余的边属性权重自由度**。

应保留边拓扑及关系种类，它们仍决定邻接、消息来源和聚合度。迁移时需折叠 bias，以保持原模型行为。

## 2. 可以由其他输入精确恢复的信息

| 输入 | 等价关系 | 轨迹最大绝对误差 | 判断 |
|---|---|---:|---|
| WAIT `wait_duration_norm` 与 `next_event_delta_norm` | 两列都直接写入 `wait_ticks / horizon_tick` | 0 | 明确重复，可合并首层权重 |
| WAIT 与 global 的 `active_order_ratio` | 同一活跃订单计数和分母 | 0 | 跨编码分支重复；删一个路径前需考虑作用位置 |
| capable `horizon_slack_norm` 与 `predicted_finish_time_norm` | `slack = 1 - finish`，当前两者裁剪范围也对应 | 5.96e-8 | 保留一个原始输入即可，另一项按需恢复 |
| wave-module `remaining_operation_ratio` | `released_operation_ratio + future_operation_ratio` | 1.49e-8 | 明确线性依赖，可合并消息变换权重 |
| global `production_decision` 与 `worker_decision` | 非终止状态两列之和为 1 | 0 | 一个标志即可表达阶段；还可由 `decision_type` 恢复 |

生产 Flow 直接专家同时使用预测完成时间和 slack，因此同一完成时间信息以两种形式参与归一化权重。可以从一个原始输入恢复两项，避免存储重复；直接减少专家项数会改变 simplex 权重分配及非线性，须重新验证。

工人 Flow 的候选 z-score 与动作编码中的绝对时长有不同作用：前者表示候选间差异，后者保留绝对时间尺度。两者不能据此判为无用重复。

## 3. 重复编码，但需要消融判断

- 工序 `required_module_*` 与 `operation → module` 的 requires 边重复表达模块归属。
- 机器 `supports_module_*` 与 machine-module supports 拓扑重复表达兼容性。
- 工人 `qualified_module_*` 与 worker-module qualified_for 拓扑重复表达资格。
- machine 的 current/target 模块、machine-module 的 installed 标志、locked 边的 source/target 模块描述相关的重构状态。
- order、wave、module、wave-module 与 global 的部分进度/剩余工作量统计，可以从基础工序状态聚合得到。

`can_install` 边可由 `worker → module` 与 `operation → module` 的连接关系精确生成。本次每个状态都验证了对应邻接矩阵等式，最大误差为 0。它不引入新的资格事实，但提供工人到工序的一跳消息路径；现有两层 HGNN 的消息变换、ReLU 和平均聚合使其不等价于简单删除后的两跳路径。

建议优先做资格 one-hot 与 `can_install` 关系消融，再判断 wave/module 聚合统计是否需要缩减。完成/未释放工序虽无法立刻选取，仍表达已完成进度、后续需求和资源规划，不能仅凭动作被 mask 就整体删去其图消息。

## 4. 明显的计算冗余

### actor 计算全部笛卡尔积候选

`network_v8.py:744` 先生成所有 operation-machine 或 machine-worker 组合，经过 edge encoder、action encoder、专家和 residual 网络，最后在 `forward_batch` 中 mask。

| 阶段 | 状态数 | 计算的 pair 行数 | 合法 pair 行数 | 被 mask 比例 |
|---|---:|---:|---:|---:|
| 生产 | 3,255 | 1,867,440 | 4,683 | **99.75%** |
| 工人 | 1,844 | 88,512 | 2,279 | **97.43%** |

优先考虑只对合法 pair 和 WAIT 执行动作头，再把结果填回完整 logits。当前 direct 标准化、residual 标准差和诊断只取合法动作，且默认 dropout=0，具备保持现有有效输出的条件；实际改造仍需验证数值、梯度、batch 和 checkpoint 契约。

这里的比例是动作头候选计算的浪费比例，不是端到端提速比例。图编码、环境推进仍有独立成本。图中的非法动作关联边可能仍提供规划上下文，应把动作头稀疏化与图拓扑裁剪分开评估。

### 仅求 value 时也计算 actor

`agent/ppo/agent.py:221` 的 `value_batch()` 调用完整 `forward_batch()` 并丢弃 logits。这会在 cutoff 自举等场景执行不需要的候选头和诊断。可增加 encode_graph + preference_encoder + critic 的直接价值路径。

### 观测构建中的未使用统计

`environment/env.py:1341` 至 `1360` 计算 `remaining_workload_by_module`、`total_remaining_workload` 和 `qualified_worker_count`；后面的 service features 与返回观测未引用它们。这些是确定的死计算，可清理。`remaining_workload_by_module` 仅用于继续计算未被使用的 `total_remaining_workload`。

## 5. 单目标配置与历史实现

E1 单目标配置固定 one-hot 偏好。混合器只读取其中一个专家，但仍计算三个专家。在该固定偏好下，其余八个专家（四类动作头 × 两个非活动目标）约 **13.31 万参数**不影响 actor 输出、得到零梯度，占总参数约 11.78%。这属于单目标运行的专门冗余；Universal 的三目标专家都有使用场景。固定偏好编码也只是一个可学习常量，可在单目标专用架构中简化，但需考虑实验结构一致性。

`network.py` 保留的 V7 实现增加阅读负担，公共构造入口已经被 V8 覆盖。这是历史源码冗余，不是当前模型运行时又叠加了一套 V7 网络。若需要清理，可迁入明确的 archive 文件并核验历史分析入口。

## 建议实施顺序

1. 清理未使用统计；增加直接 critic/value 路径。
2. 稀疏计算合法 actor 候选，保留完整动作编号与 mask 接口。
3. 删除或折叠恒零列、重复 WAIT 时长、单特征 ranker theta、恒定边属性；同步迁移网络规格和 checkpoint。
4. 按匹配训练预算、多个 seed、同一评测集，对资格重复编码、`can_install`、聚合统计做消融，再决定删减图结构。

本次没有执行重训练或完整消融，因此尚不能把“信息可推导”进一步解释为“删掉后策略效果不变”。
