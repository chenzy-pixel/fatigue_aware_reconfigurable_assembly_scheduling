# 当前实验协议

更新日期：2026-10-03。`configs/default.json` 是当前 Universal 协议的主配置，`configs/v8/universal.json` 继承它。单目标和 MO-ALNS 配置复用相同环境、奖励与冻结尺度。

2026-10-03 奖励参数变更：全局失败终止惩罚设为 **2.0**，由 `configs/default.json`
统一提供，单目标、Universal 和 MO-ALNS 入口继承。每条失败轨迹只在终止步扣除一次，
生效值记录于运行配置、runtime manifest 和 checkpoint 元数据。

2026-10-02 时间上下文与执行路径变更：从桌面提交 `06218ef` 移植完整订单裕量、工人候选裕量/阶段等待、WAIT 后最小裕量及变化，观测升级至 schema 6，三项目标专家通过各自上下文评分使用新增信息。
动作评分按合法 pair 稀疏计算，价值自举使用独立共享 critic 路径；沿用 V2 数据、单阶段奖励、当前冻结尺度及评测矩阵。
同一归一化版本的 schema-5 checkpoint 新增输入列补零，保留已有权重，并记录迁移；归一化哈希校验仍执行。

2026-10-02 实验变更：按用户指定，全部目标归一化统一为 Flow=1089.15、Cost=353.27、Variance=2.2629。
新增 V2 验证参考尺度清单，偏好奖励、图中的目标相关特征、固定评估质量和 Pareto/HV 分析使用该组尺度。
固定评估质量升级为 `canonical_bounded_quality_v2`，权重保持 `(0.5,0.3,0.2)`。
原 E1 清单、已完成运行的配置、日志和 checkpoint 保留原版本；尺度变更后的训练单独记录。

2026-10-01 增加 V2 单目标独立微调阶段，具体变更、参数和执行入口见 [V2 追加训练](v2_continuation.md)。该入口要求 V2 项目代码及原运行结果。

2026-10-03 消融变更：接入节点 MLP 编码器、共享偏好评分头和三个疲劳中性单目标变体。
五组继承当前 V2 数据、schema-6 时间上下文、冻结尺度、惩罚 2.0 和父配置训练预算；
原始疲劳监测与活动约束分别报告。训练、评估及配对汇总入口见[五组消融协议](ablation_protocol.md)。

## 固定尺度和质量

| 目标 | 冻结尺度 |
|---|---:|
| Flow | 1089.15 |
| Cost | 353.27 |
| 工人负荷方差 | 2.2629 |

清单为 `configs/manifests/v2_selected_scales_20261002.json`，文件 SHA256 为 `9d4829c467f592d9a319251f0f779e7354cce70e1511d3bc90ab19b78fa4cce3`。加载配置时校验清单及其哈希，训练期间保持固定。

尺度由用户指定，来源为 V2 追加训练中的成功轨迹验证均值：Flow 第 500 轮的 1089.1547297297298 保留两位小数，Cost 第 500 轮的 353.2652386890254 保留两位小数，Variance 第 360 轮的 2.262897640791476 保留四位小数。对应完成数为 148/150、145/150、146/150。清单记录实际均值、取值轮次、舍入精度和日志哈希；这些参考值不表示三个目标来自同一完成率或同一 best checkpoint。

旧 E1 尺度清单 `configs/manifests/e1_tail5_scales_20260928.json` 保留，用于读取原实验有效配置和核验历史结果。

偏好质量采用 `q_i = J_i/(s_i+J_i)`，再计算 `(max_i(w_i*q_i) + rho*sum_i(w_i*q_i))/(1+rho)`，其中 `rho=0.05`。该公式同时用于 PPO 偏好奖励、MO-ALNS 选解和结果分析。

`quality_score` 使用固定参考指标 `canonical_bounded_quality_v2`：尺度 `(1089.15,353.27,2.2629)`、权重 `(0.5,0.3,0.2)` 的有界加权和。`preference_quality_score` 使用同一组冻结尺度和每条轨迹自身偏好的增强切比雪夫质量；两个指标的聚合公式仍不同。旧 `canonical_bounded_quality_v1` 仅用于按历史有效配置复核原结果。

## 网络、奖励和预算

网络为 V8 HGNN actor-critic，schema-6 图包含六类节点和十二类关系。生产与工人采用 pair-plus-WAIT；WAIT 由精确进展证书控制。工人 Flow 专家采用 `candidate_zscore_v1`，标准差下限 `0.001`。正式执行模式为 `phase_batched_v1`，精度为 `float32`。

所有训练入口使用 `single_stage_progress_quality_failure_v2`：`r_t = delta_progress + Q_t - Q_(t+1) - failure_penalty`。任务失败仅在终止步扣 2，实际质量保持可重建；`gamma=1`、feasibility shaping 关闭。rollout cutoff 使用 critic 自举。

| 配置 | 训练轮数 | 验证间隔 | 训练 worker | 验证 worker | 每次 PPO 更新 episode |
|---|---:|---:|---:|---:|---:|
| Universal | 2000 | 100 | 20 | 20 | 20 |
| Flow | 1000 | 40 | 20 | 20 | 20 |
| Cost | 1000 | 40 | 20 | 20 | 20 |
| Variance | 1000 | 40 | 20 | 20 | 20 |

独立算法种子为 `11,23,37,53,71`。Universal 训练采用 20-episode 偏好块：三个单目标端点各两次，加十四个 scrambled-Sobol simplex 偏好。单目标训练、验证与最终评估使用对应 one-hot 偏好。

## 训练验证

Universal 每次验证为新版 500 实例验证池中固定分层选取的 50 个实例 × 13 个偏好 × 3 次 sampled 解码，共 1950 条轨迹。温度为 1.0，有序偏好坐标依次为 `(Flow,Cost,Variance)`：

| 编号 | 偏好 |
|---|---|
| 1 | (1, 0, 0) |
| 2 | (0, 1, 0) |
| 3 | (0, 0, 1) |
| 4 | (0.5, 0.5, 0) |
| 5 | (0.5, 0, 0.5) |
| 6 | (0, 0.5, 0.5) |
| 7 | (0.333333333333, 0.333333333333, 0.333333333333) |
| 8 | (0.6, 0.3, 0.1) |
| 9 | (0.6, 0.1, 0.3) |
| 10 | (0.3, 0.6, 0.1) |
| 11 | (0.3, 0.1, 0.6) |
| 12 | (0.1, 0.6, 0.3) |
| 13 | (0.1, 0.3, 0.6) |

checkpoint 采用 completion-first：先最大化所有验证偏好中的最低成功率；完成率在 `1e-12` 内平局时，最小化偏好等权质量。质量先在每个偏好的成功轨迹内求均值，再对偏好等权平均。任一偏好没有成功轨迹时取 `+inf`。完全平局保留已有 best；每个 run 保存 last。

验证采样根种子为 `algorithm_seed+100000+repeat`，每个 instance-preference 单元另通过 SHA256 派生独立 Torch RNG。

## 最终评估和分析

最终评估重载固定 best checkpoint，采用 `{(i/10,j/10,k/10): i+j+k=10}` 的 66 点网格，每实例每偏好采样三次，温度为 1.0。根种子为 `algorithm_seed+300000+repeat`。测试集、OOD 和 stress 均只读加载已物化数据，实例数量由实际评测命令确定并写入 provenance。

MO-ALNS 使用同阶段的偏好集合和冻结尺度，默认每偏好最多 300 次环境评估，输出每偏好一个档案选解。它使用求解器预算；PPO 使用每偏好三次采样预算。报告保存两种预算和求解时间。

结果 schema 为 `8.0.0`。逐轨迹 CSV 记录原始目标、偏好、成功/失败、安全检查、算法种子、重复索引、执行模式、尺度及清单哈希。分析校验每个实例、方法、种子的完整偏好/重复矩阵，拒绝重复单元、缺失单元和不匹配的协议数据。

Pareto/HV 使用安全成功候选、有界归一化和参考点 `(1,1,1)`。统计先按实例计算，再按算法种子聚合；方法配对以匹配的种子均值为单位。

```powershell
python -m analysis.pareto_analysis --candidate-csv result/runs/v8_universal_seed11/final_sampled_instance_metrics.csv --output-dir result/analysis/universal
python -m scripts.mo_alns --config configs/baselines/mo_alns.json --dataset test --algorithm-seed 11
python -m analysis.mo_alns_analysis --ppo-candidate-csv result/analysis/universal/candidates.csv --mo-alns-candidate-csv result/runs/mo_alns_test/instance_metrics.csv --output-dir result/analysis/ppo_mo_alns
```

训练、评估和分析从配置读取偏好集合、重复次数及尺度。固定数据和尺度清单的原始字节保持不变。

V2 参考清单由 `python -m scripts.build_v2_normalization_manifest --output 新路径` 从已转移的续训日志重建，输出不可覆盖。历史 E1 中位数清单仍可由 `scripts.build_normalization_manifest` 重建。

新尺度的 checkpoint 使用新的归一化哈希。旧 checkpoint 默认加载至新配置会拒绝哈希不匹配；从旧模型转入新尺度属于单独的权重迁移实验，需记录来源并重新验证。`continue_v2.py` 记录的原始追加阶段继承源 run 的旧尺度，属于历史协议。

训练电脑的 Git 同步方式和尺度版本说明见 [V2 归一化更新](v2_normalization_update.md)。

## 算例协议 v2

详细参数、采样规则与验收见 [算例生成与训练分布 v2 实验规格](data_v2_specification.md)。默认数据目录为 `data/instances/v2` 和 `data/manifests/v2`，在线缓存为 `data/instances/train_cache/v2`。生成器与数据 schema 为 `2.0.0`；旧配置及数据哈希见 `configs/archive`。

候选只依据合法性、场景结构和必要可行性预检查接受。启发式诊断不筛选实例；截断实例标记 `unknown` 并保留，训练默认关闭诊断。主验证子集按目标场景比例分配为 3/18/7/7/5/5/5；49 个诊断实例每场景 7 个，与主验证互不重叠。诊断每五次主验证及训练结束运行，不参与选模。所有子集保存实际索引和实例、子集哈希。

训练采用 100-episode 窗口配额和固定线性课程，压力上限从 0.4 逐步增到 1.0。完成能力评测包含全部选中实例及未知实例；heuristic gap 仅在策略和参考均完整、安全完成时有效。成功质量、失败进度及原因分别报告；结果聚合校验数据协议、子集和尺度哈希。

## Worker 运行日志

默认 `logging.worker_progress.debug_steps=false`。运行目录中的 `worker_progress.jsonl` 只记录算例任务开始（`instance_start`）、结束（`instance_end`）、异常（`error`）和慢任务（`slow_task`）。生成与调度分别记录任务，任务编号关联起止事件；结束记录保存算例标识、在线种子、总步数、耗时和结束原因，包含 rollout 步数截断、异常及提前关闭。

将 `logging.worker_progress.debug_steps` 设为 `true` 可额外记录每个 worker step 的响应（`response`），用于逐步调试。心跳始终保存在内存中，继续刷新 `training.worker_stall_timeout_seconds` 检测；`training.worker_timeout_seconds` 仍限制单次命令总耗时。异常记录包含最后一次心跳及 traceback。

慢任务阈值沿用 `training.slow_instance_seconds`，作用于单次 worker 命令耗时和整个算例任务耗时。每个算例任务最多记录一次慢任务提示，同时写入 `slow_instances.jsonl`；算例结束记录保留最终耗时。
