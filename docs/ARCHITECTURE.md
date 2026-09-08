# 当前代码架构

本文描述仓库当前可执行 E1 主线的模块边界、文件职责和主要调用关系。它是代码导航文档，不重复实验参数；具体数值以 `configs/default.json` 和目标配置为准。

## 1. 总体分层

```text
配置层 configs
    ↓ 装配有效配置、固定 runtime manifest
数据层 data
    ↓ 提供 AssemblyInstance / 固定数据集 / 在线训练实例
环境层 environment
    ↓ observation + action mask + reward + terminal metrics
算法层 agent
    ↓ HGNN actor-critic + PPO + rollout collector
训练协议层 training
    ↓ 候选窗口、独立 audit、accepted checkpoint
入口层 train.py / eval.py / mo_alns.py
    ↓
结果层 result
      配置、checkpoint、CSV、指标、provenance、可视化
```

核心依赖方向是 `configs → data → environment → agent → training/entrypoints → result`。环境不依赖 PPO；PPO 只消费环境公开的 observation、mask、reward 和终止信息。MO-ALNS 与 PPO 共用同一个环境和数据模型，以保证约束与指标口径一致。

## 2. 当前固定实现

`configs/runtime.py` 生成只读 `runtime_manifest`，用于结果追溯，不作为运行时分支开关：

| 领域 | 当前实现 |
|---|---|
| 主线 | `e1_latest` |
| 生产动作 | `pair_plus_defer_v1` |
| 策略头 | V7 |
| 候选排序 | `bounded_ranker_scale_v7` |
| 工人可行性 | `temporal_matching_admission_recovery_v3` |
| Temporal search | `strict_recovery_frontier_transposition_budget_v1` |
| Defer shield | `deadline_progress_viability_shield_v2` |
| Observation | schema 3 |
| 训练协议 | `v7_e1_single_objective_protocol_v4` |

## 3. 根目录入口

| 文件 | 主要组件或逻辑 | 被谁调用 / 输出 |
|---|---|---|
| `train.py` | `TrainingEngine`；训练装配；周期验证；候选 checkpoint 落盘与重新加载；独立 audit；resume；训练日志 | CLI `train.py --config ...`；写入 `result/runs/<run>` |
| `eval.py` | `EvaluationPolicy`；单实例、固定数据集、串行/并行评测；greedy/sample 策略；诊断 trace | 训练验证和独立评测共用；输出评测 CSV/JSON |
| `mo_alns.py` | 在固定数据集和 preference grid 上运行 MO-ALNS；重放候选并校验目标值 | 输出 MO-ALNS candidate rows |
| `mo_alns_benchmark.py` | 按 manifest 批量执行 MO-ALNS 基线 | 读取 `configs/baselines/*manifest*.json` |
| `mo_alns_analysis.py` | 汇总 E1 与 MO-ALNS 的候选质量、配对统计和 preference/objective 关系 | 生成离线分析表与报告 |
| `pareto_analysis.py` | 非支配集、三维 hypervolume、候选审计和科研绘图 | 仅用于离线评测，不参与 PPO 晋升 |
| `single_objective_analysis.py` | 分析单目标训练曲线、阶段标记、audit 结果并生成图表/报告 | 消费已有 run 目录 |
| `e1_reproducibility_audit.py` | E1 复现检查入口 | 对配置、数据、checkpoint 和输出做审计 |
| `benchmark_parallel.py` | collector 并行吞吐基准 | 性能诊断工具 |
| `utils.py` | 全局 seed、动作轨迹哈希、评测采样 seed 等通用函数 | 训练、评测和 collector 共用 |

## 4. 配置层 `configs/`

| 文件 | 组件或职责 |
|---|---|
| `configs/config.py` | 递归加载 JSON、处理 `extends`、深度合并、解析项目路径，并在最后附加 runtime manifest |
| `configs/runtime.py` | 保存唯一实现身份；拒绝已经移除的实现选择字段；返回 manifest 副本 |
| `configs/default.json` | 最新 E1 的完整公共参数：实例、环境数值预算、网络尺寸、PPO、数据集、训练和评测参数 |
| `configs/e1/single_flow.json` | flow-time 单目标覆盖 |
| `configs/e1/single_cost.json` | cost 单目标覆盖 |
| `configs/e1/single_variance.json` | workload-variance 单目标覆盖 |
| `configs/baselines/mo_alns.json` | MO-ALNS 正式基线参数 |
| `configs/baselines/mo_alns_smoke.json` | 最小基线冒烟参数 |
| `configs/baselines/mo_alns_manifest*.json` | 基线批量运行清单及示例 |

配置允许改变真实实验参数，例如 reward weights、temporal search budget、deadline reserve、网络宽度、PPO 超参数、数据规模、seed 和 worker 数；动作语义、策略头版本、matching 版本及 shield 版本由代码固定。

## 5. 数据层 `data/`

| 文件 | 主要组件或逻辑 |
|---|---|
| `data/models.py` | `AssemblyInstance` 及订单、工序、机器、模块、工人、疲劳和成本 dataclass；JSON/YAML/pickle 序列化；实例结构校验 |
| `data/generate_orders.py` | `InstanceGenerator`；按 seed 和 profile 生成训练/验证/测试实例；构造订单波次、工艺路线、机器、工人和模块成本 |
| `data/feasibility.py` | 最大匹配；静态资源可行性分析；生成前的 cheap feasibility precheck |
| `data/dataset.py` | `GeneratedInstanceRecord`、`InstanceDataset`、`OnlineInstanceDataset`；split seed 合约；缓存；manifest/hash 校验；发布固定数据集 |
| `data/instances/fixed_instance.yaml` | 黄金测试使用的小型确定性实例 |
| `data/instances/{validation,test,ood,stress}/` | 固定评测实例，不在训练过程中重采样 |
| `data/manifests/{validation,test,ood,stress}/manifest.json` | 数据集成员、seed、hash 和 profile 元数据 |

训练时 `OnlineInstanceDataset` 根据确定性 seed 生成或读取缓存；验证和评测通过 manifest 加载固定实例，因此不同策略看到相同问题集合。

## 6. 环境层 `environment/`

### 6.1 文件职责

| 文件 | 主要组件或逻辑 |
|---|---|
| `environment/actions.py` | `ActionCodec`；稳定编码/解码 `operation × machine`、`machine × worker`，并定义 production defer 与 worker advance 的动作 ID |
| `environment/state.py` | `OperationRuntime`、`MachineRuntime`、`WorkerRuntime`、`ReconfigurationRuntime`；episode 内可变状态 |
| `environment/dynamics.py` | 时间与 tick 的确定性量化转换、数值容差 |
| `environment/temporal.py` | Temporal oracle 的不可变输入/输出结构、worker/task snapshot、搜索预算和预算耗尽异常 |
| `environment/preference.py` | 三目标 `PreferenceVector`、归一化、默认权重和 simplex lattice；供 reward 标量化及 MO-ALNS 离线网格使用 |
| `environment/types.py` | 状态枚举；`RewardVector`；异质图节点/边类型；`EdgeStore`；`HeterogeneousGraphObservation`；`PolicyObservation` |
| `environment/env.py` | `AssemblySchedulingEnv` 门面及当前离散事件内核：reset、观测、mask、step、事件推进、加工/重构/疲劳、temporal v3、defer shield v2、reward、metrics 和 schedule validation |
| `environment/__init__.py` | 环境包的稳定导出面；入口代码应优先从这里导入公共类型 |

### 6.2 环境公开契约

外部训练器和评测器依赖以下六个行为接口：

| 方法 | 契约 |
|---|---|
| `reset(instance, ...)` | 装载实例、初始化事件队列和运行时状态，返回初始 observation |
| `observe()` | 生成 schema 3 的异质图 observation 与动作候选特征 |
| `get_action_mask()` | 返回当前决策阶段的合法动作布尔 mask |
| `step(action)` | 解码动作、推进生产/工人决策或离散事件，返回 observation、reward、done 和 info |
| `metrics()` | 汇总 flow time、cost、load variance、安全性、matching 和 defer 诊断 |
| `validate_schedule()` | 对最终 schedule 做先后约束、资源占用和一致性校验，返回错误列表 |

`env.py` 目前仍承载 temporal search、defer certificate、事件推进和图构建的具体私有方法；已经抽出的 `actions/state/dynamics/temporal/types` 是稳定领域边界。后续继续瘦身时，应保持上述六个接口、动作 ID、observation schema 和 reward 恒等式不变。

### 6.3 一次环境决策

```text
observe()
  └─ 构建异质图节点/关系/候选特征
get_action_mask()
  ├─ 静态资源与 precedence 快速检查
  ├─ 静态 matching
  ├─ 必要时 temporal oracle v3
  └─ 对 defer 动作生成 completion-viability shield v2 certificate
step(action)
  ├─ production: pair 或 defer
  ├─ worker: machine-worker pair 或 advance
  ├─ 推进离散事件与疲劳/成本/负载
  └─ 生成 RewardVector、终止原因和诊断
```

## 7. 算法层 `agent/`

### 7.1 PPO 主线

| 文件 | 主要组件或逻辑 |
|---|---|
| `agent/networks/ranker.py` | `BoundedRanker`；V7 候选相对特征的有界线性排序器 |
| `agent/networks/heads.py` | 标量 MLP head 构造函数；供 actor/critic 子头复用 |
| `agent/ppo/network.py` | `HeterogeneousMessagePassingLayer`、`HeteroGraphActorCritic`；图 batch 拼接；HGNN message passing；production/worker logits；V7 bounded context residual；value head；checkpoint network spec |
| `agent/ppo/buffer.py` | `Transition`、`RolloutBuffer`；保存 on-policy transition，计算 GAE 和 returns |
| `agent/ppo/agent.py` | `PPOAgent`；采样/贪心动作、batched act/value、clipped PPO 更新、熵和 value loss、checkpoint 严格保存/加载 |
| `agent/ppo/parallel.py` | `ParallelEpisodeRunner`；worker 进程协议；训练 batch 收集；固定数据集评测；forced-action 聚合；timeout/failure 传播；`worker_count=1` 也走同一路径 |
| `agent/ppo/__init__.py` | PPO 包的公共导出面 |

网络前向链路：

```text
PolicyObservation
  → graph batch/collation
  → node-type encoders
  → heterogeneous message passing
  → production head / worker head
  → bounded ranker context residual
  → action mask
  → Categorical(logits) + state value
```

PPO 更新只包含 clipped policy loss、value loss、entropy、GAE、梯度裁剪和学习率控制。

### 7.2 基线算法

| 文件 | 主要组件或逻辑 |
|---|---|
| `agent/baselines/policies.py` | `HeuristicPolicy` 与 `RandomPolicy`，用于环境/评测基准 |
| `agent/mo_alns/types.py` | MO-ALNS solution、candidate、search result 和 objective tuple |
| `agent/mo_alns/archive.py` | 非支配 archive、归一化目标和 augmented Tchebycheff 比较 |
| `agent/mo_alns/solver.py` | 解码、destroy/repair operators、matching-safe repair、operator 权重更新和 `MOALNSSolver` 主循环 |

## 8. 训练协议层 `training/`

| 文件 | 主要组件或逻辑 |
|---|---|
| `training/protocol.py` | `TrainingPhaseController`；single-objective guarded v1 候选窗口、验证观察、独立 audit、接受/拒绝状态和 resume 状态序列化 |
| `training/__init__.py` | 仅导出当前单目标协议常量和 controller |

职责边界：`TrainingEngine` 决定何时收集、更新、验证、保存与重载；`TrainingPhaseController` 只根据验证/audit 证据推进协议状态，不执行环境 rollout 或网络更新。

## 9. 结果层 `result/`

| 文件 | 主要组件或逻辑 |
|---|---|
| `result/io.py` | 创建 run 目录；原子化写 JSON/CSV/config；统一评测产物布局 |
| `result/metrics.py` | 当前结果 schema；三目标聚合；upper-tail summary；相对 gap；lexicographic selection key |
| `result/provenance.py` | 源码状态、Git 状态、有效配置、数据 manifest、checkpoint/network hash 和环境信息快照 |
| `result/terminal_log.py` | tee stdout/stderr 到 run 日志，同时保留终端显示 |
| `result/visdom_dashboard.py` | 在线训练/验证面板、事件日志、诊断快照和 schedule Gantt SVG |
| `result/visdom_replay.py` | 从持久化日志重放 Visdom 曲线 |
| `result/runs/` | 训练 checkpoint、配置、CSV、audit 和评测产物；属于实验资产，不是源代码 |
| `result/analysis/` | 离线统计、Pareto/HV 与绘图产物 |

## 10. 训练与评测调用链

### 10.1 训练

```text
train.py main
  → load_config
  → OnlineInstanceDataset / fixed validation dataset
  → build_actor_critic → PPOAgent
  → ParallelEpisodeRunner
  → TrainingEngine.run
      → collect_training_batch
      → RolloutBuffer.compute_gae
      → PPOAgent.update
      → fixed validation
      → TrainingPhaseController.observe_validation
      → candidate checkpoint
      → independent audit
      → accepted checkpoint
  → result/io + provenance + dashboard
```

### 10.2 评测

```text
eval.py main
  → load_config + dataset manifest
  → EvaluationPolicy
      ├─ PPOAgent + accepted checkpoint
      ├─ HeuristicPolicy
      └─ RandomPolicy
  → ParallelEpisodeRunner.evaluate_records
  → aggregate_evaluation_rows
  → metrics.json + instance_metrics.csv + provenance
```

### 10.3 MO-ALNS

```text
mo_alns.py
  → load fixed dataset
  → MOALNSSolver
  → AssemblySchedulingEnv decode/validate
  → ParetoArchive
  → candidate CSV
  → mo_alns_analysis.py / pareto_analysis.py
```

## 11. 测试与架构约束

| 测试文件 | 主要保护内容 |
|---|---|
| `test/test_latest_only_audit.py` | 配置清单、runtime identity、旧模式零引用、黄金 observation/mask/action/reward、checkpoint hash |
| `test/test_temporal_matching_v3.py` | 静态快路径、temporal recovery、预算耗尽、horizon 边界 |
| `test/test_production_defer.py` | pair/defer 动作和 shield certificate |
| `test/test_graph_observation.py`、`test/test_hetero_gnn.py` | schema 3 图结构、关系和 HGNN 前向 |
| `test/test_v7_policy.py` | V7 logits、ranker、checkpoint 严格兼容和冻结输出 |
| `test/test_ppo*.py` | GAE、batching、PPO 更新、保存和恢复 |
| `test/test_parallel_rollout.py`、`test/test_reproducibility.py` | 单/多 worker 的 seed、实例序列和采样复现契约 |
| `test/test_e1_single_objective.py` | 候选窗口、独立 audit、accepted checkpoint 和 resume |
| `test/test_mo_alns.py`、`test/test_pareto_analysis.py` | 基线求解与离线 Pareto/HV |

黄金基准位于 `test/baselines/latest_only_golden.json`；重构前展开配置快照位于 `test/baselines/pre_refactor_expanded/`。它们只用于回归比较，不是可执行实现选择。

## 12. 当前架构检查结论

- 配置、数据、环境、agent、训练协议和结果模块之间的顶层边界清晰，训练与评测共享相同 collector 和环境语义。
- 动作 codec、运行时状态、temporal 数据契约、网络 building blocks、训练协议和 provenance 已形成独立组件。
- `environment/env.py` 仍是最大的实现聚合点，门面与事件推进、图构建、temporal search、defer shield 逻辑位于同一个类中。它是下一轮等价拆分的主要候选，但不影响当前公开契约和黄金结果。
- `train.py` 同时承担入口装配、验证稳定性、checkpoint audit 和训练循环；后续可把验证/audit orchestration 进一步移入 `training/`，同时保持 `train.py --config ...` 不变。
- `agent/ppo/network.py` 集中了图批处理、message passing 和两类动作 head。当前 checkpoint key 兼容性依赖这一组织方式，继续拆分时需要保留 module attribute 名和 `state_dict` key。
