# 当前项目性能与训练效率检查（2026-10-03）

当前项目仍有明确的优化空间。优先顺序为：**推断诊断批量计算 → 环境恢复搜索与状态缓存 → 验证任务持续补位 → 并行度调档 → PPO 数据打包与学习参数试验**。

诊断统计批量计算和环境恢复搜索/WAIT 证书缓存已落实，完整诊断与完整轨迹的修改前后测量见[实施记录](runtime_performance_optimizations_20261003.md)。下文保留本次实施前的审计数据与建议。

## 检查范围与证据

检查对象是当前工作区，HEAD 为 `06218effb331a7a3897cd2dd1e6170c797a6c2b2`，包含现有未提交配置改动。运行时公共网络入口最终导入 `agent/ppo/network_v8.py`；本次测量的是该实现。硬件为 RTX 4060 Laptop 8 GiB，Python 3.12.14、PyTorch 2.7.1+cu126，主进程使用 4 个 Torch 线程。

本次新增审计脚本、原型与结果文件；训练实现和配置保持检查时的状态。测量使用固定实例及 V2 验证集前两个实例，每个实例重放相同的 100 步前缀。GPU 计时在预热后同步 CUDA，取多次中位数。网络输入取这三条路径的 60 个观测，较大的 batch 会循环使用这些观测。原型只在独立审计进程中运行。

原始数据位于 `result/audits/performance_review_20261003/`：`environment.json`、`environment_candidates.json`、`inference.json`、`ppo.json`、`history.json`、`metadata.json` 及三个 `.prof` 文件。复跑入口为该目录的 `benchmark.py`、`environment_candidates.py` 和 `history.py`。

## 1. 推断诊断统计是首要优化机会

`agent/ppo/network_v8.py:942` 在无梯度推断中逐图调用 `_record_components()`（定义于第 1255 行），计算每个目标的均值、标准差、RMS、饱和率和动作排序。虽然结果已经合并为一次 host 传输，这些统计本身仍逐图执行大量小 Tensor 操作。

20 图推断的 CPU profiler 中，`_record_components()` 累计耗时约 57.6 ms，整次 `act_batch()` 约 79.7 ms；这组计时包含 profiler 开销，用于定位，吞吐以以下预热测量为准。

| 图 batch | 当前完整 forward | 诊断占位对照 forward | 局部加速比 |
|---:|---:|---:|---:|
| 1 | 14.98 ms | 9.76 ms | 1.53× |
| 20 | 72.46 ms | 16.51 ms | 4.39× |
| 40 | 135.80 ms | 19.21 ms | 7.07× |
| 64 | 214.24 ms | 25.39 ms | 8.44× |

对照原型只将诊断生成替换为占位行，actor/critic 数学计算保持原路径；logits/value 通过 `rtol=2e-5, atol=2e-6` 检查，实测最大 logit 差为 `1.49e-8`。CUDA scatter 归约在重复前向中可能产生末位差异，因此使用浮点容差。

**建议实现**：用已有 `merged_graph_ids` 对 direct/context/expert 的计数、和、平方和及饱和计数做分段归约，批量得到全部图的诊断；pair 的 top-action 保留原动作编号和原平局规则。保留现有诊断字段，验证其数值与当前实现一致。对照中的诊断占位不能直接用于正式实验，也不能据此宣称整次训练提速 4.39×；它量化的是这部分开销可被优化的空间。

PPO 有梯度更新已经跳过这些动作诊断，收益主要落在训练 rollout 和正式验证。

## 2. 环境存在可保持现有特征的优化

完整图观测让同一 100 步轨迹从约 0.65–0.77 s 增至 1.91–2.13 s，观测相关增量约占完整环境耗时的 63%–67%。环境 profiler 的主要热点如下，累计时间有调用嵌套，不能相加解释占比：

- `_earliest_worker_pair_recovery_tick()`（`environment/env.py:3752`）逐 tick 搜索安全恢复点。同一候选在每个 tick 重复检查资格、当前可行性和阶段参数。
- `_wait_certificate()`（第 3857 行）在 mask、观测和仿真路径重复计算。300 步 profile 调用 903 次。
- `OrderTimeEstimator.finish_ticks()`（`environment/time_context.py:148`）累计约占该环境 profile 的 34%；一次观测通常涉及当前状态和 WAIT 投影状态的订单链估计。
- `AssemblyInstance.operation_index`（`data/models.py:91`）每次访问都重新展开所有工序并构建字典。300 步 profile 调用约 17.4 万次，累计约占 12.7%。

独立原型用现有安全投影函数，对恢复时间做整数二分；WAIT 证书按状态版本、tick 和决策阶段缓存。疲劳随恢复时间下降，阶段时长系数和累积率非负，使安全谓词单调；保留原量化、安全阈值及 EPSILON。

| 100 步实例 | 当前环境 | 二分恢复搜索 + WAIT 证书缓存 | 加速比 |
|---|---:|---:|---:|
| 固定 15×4 | 1.832 s | 1.172 s | 1.56× |
| validation_balanced_2000000 | 2.090 s | 1.410 s | 1.48× |
| validation_machine_bottleneck_2000001 | 2.086 s | 1.320 s | 1.58× |

三条路径合计 300 步：完整节点、边、global、action-set 特征、动作 mask、奖励、终止标志与物理 tick 均一致。二分单独提速 1.37–1.46×；证书缓存单独提速 1.16–1.20×。

**建议实现**：先落地这两个原型，补全 horizon、零恢复率、无合格工人、多个并行任务、phase handoff 与 WAIT 投影的回归检查；随后在 reset 时缓存工序索引、机器查找表、静态加工 tick、模块阶段参数与工人资格列表。对订单链预测可复用阶段参数和相同输入的局部投影，动态缓存需跟随状态失效。原型结果目前只覆盖这三条路径，尚未测量完整训练或全验证集。

## 3. 验证调度值得优化，正式协议保持完整

当前 Universal 每 100 episode 验证 `50 × 3 × 13 = 1950` 条轨迹，2000 episode 共 20 次、39,000 条验证轨迹；最终测试还需另外执行。E1 每 40 episode 验证 150 条轨迹。当前入口已集中调用 `act_batch()`。

历史完整运行的日志显示，正式验证占总耗时约 68%–77%；若存在独立诊断验证，它还占约 13%。这些记录来自各自不同提交与配置，作为瓶颈方向证据，不能解释为本次工作区的完整运行耗时：

| 历史 run | 总耗时 | 正式验证 | 独立诊断验证 | PPO 更新 |
|---|---:|---:|---:|---:|
| flow_v2_seed11_1000 | 5.11 h | 68.0% | 13.2% | 3.6% |
| cost_v2_seed11_1000 | 3.83 h | 67.9% | 13.0% | 3.9% |
| variance_v2_seed11_1000 | 5.96 h | 68.7% | 13.0% | 3.5% |
| e1_flow_seed11_ep1000_20261002_192603 | 5.32 h | 68.4% | 13.6% | 3.0% |

`ParallelEpisodeRunner.evaluate_records()`（`agent/ppo/parallel.py:1551`）分 chunk 执行：快轨迹结束后，对应 lane 空闲，直到整个 chunk 结束才载入下一批。建议参考训练 collector 的持续补位方式，在一个验证 episode 完成后立即分配下一个实例/偏好任务。评估期间网络保持固定，每条任务保留 `instance_id + preference_key + repeat` 派生 RNG，按 record index 恢复输出顺序。该调度改造的收益尚未测量。

协议中已确定的 50 个实例、3 次采样、13 个偏好、温度和选模规则继续作为正式实验约束。提高验证速度可先从上述实现优化获得。

## 4. 并行数、PPO 与学习效率

**并行调档需要固定更新预算。** `TrainingEngine.run()` 默认将 `episodes_per_update` 取为 worker 数（`train.py:542` 附近）。把训练 worker 从 20 改成 40，默认每次 PPO 的 episode 数也会翻倍；同样 2000 episode 的更新次数从 100 降为 50。因此要分开记录环境并行数和每次更新 episode 数。若固定每次更新为 20，训练 worker 超过 20 通常也无法充分使用，优先独立调 `validation_parallel_envs`。当前主机 24 个逻辑 CPU，40 worker 是否划算应实测，不能沿用另一台机器的结论。

**PPO 打包为次优先级。** 当前每个 epoch/minibatch 都重新 `_collate_graphs()` 和传输动作、旧 log-prob、returns；可把固定 rollout 数据预先打包，预存这些标量张量，在多个 epoch 复用。当前 `bool(torch.isfinite(loss))` 每个 minibatch 同步 GPU，`torch.randperm(..., device=cuda).tolist()` 每个 epoch 又传回 host。应在保留非有限值检测的前提下减少同步。128 条合成 transition、2 epoch、batch 64 的预热更新约 0.259 s；该微测量不能用于推算真实 PPO 更新速度或收敛。

**学习率调度值得单独试验。** Universal 的 plateau patience 为 15 次验证、间隔 100 episode，即连续 1500 episode 无改进才降学习率，2000 episode 中调度机会有限。历史 E1 尾部 KL 约 `5e-4–3e-3`、clip fraction 约 `0.5%–3.4%`，可以作为比较学习率或更新预算的监测指标；这些旧配置、旧奖励与单目标记录不能证明当前 Universal 更新不足。需要在相同训练 episode、固定验证协议和多个 seed 下比较完成率、偏好平衡质量及达到目标质量的墙钟时间。增加 epochs、缩小网络或改变奖励不应被当作已验证的提速方案。

**AMP/compile 排在主要瓶颈之后。** 这类改造需要检查混合精度下归约、mask、分布和 PPO ratio 的数值，并记录首次编译开销。一般性的同步、AMP 和 profiler 指引可参考 [PyTorch 性能调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)、[AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) 和 [CUDA 计时语义](https://docs.pytorch.org/docs/stable/notes/cuda)。当前实测主要空间在 CPU 环境与逐图诊断。

## 验证状态

独立原型的 300 步完整观测与物理轨迹比较通过；推断诊断占位对照的 actor/critic 数值检查通过。当前版本的 PPO batch、订单时间上下文与 PPO checkpoint 测试共 **37 项通过**（11.35 s），记录在审计目录 `pytest.xml`。

后续端到端验收应固定同一配置和采样预算，分项记录 rollout、正式验证、PPO、IPC、GPU 峰值显存和总墙钟，并核验评估输出和 checkpoint 选择。本文的局部加速比不相乘，也不代表算法解质量已经改善。
