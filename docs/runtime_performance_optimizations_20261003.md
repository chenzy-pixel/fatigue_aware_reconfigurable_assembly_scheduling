# 诊断统计与环境运行优化（2026-10-03）

已完成诊断统计批量计算，以及恢复时间二分搜索和 WAIT 证书缓存。网络参数结构、network_spec、checkpoint schema、观测字段、奖励公式和 PPO 配置保持原定义。

## 实现

`agent/ppo/network.py` 的阶段批量路径调用 `_record_components_grouped()`，以 graph ID 一次归约整批合法动作的均值、RMS、饱和率和计数。标准差采用中心化后的第二次归约，避免近似常量分数的平方和相减误差。生产和工人阶段分别归约。

完整保留原诊断字段、Python 返回类型与输入图顺序。WAIT 的合法性单独读取；pair 排名排除 WAIT，使用原动作编号，在分数平局时取原编号最小的候选。`contribution_*_rms` 延续参考路径最终保存的偏好加权 expert RMS。诊断计算只在无梯度推断中执行；`reference_v8` 的逐图统计作为回归参考保留。

`environment/env.py` 的 `_earliest_worker_pair_recovery_tick()` 对每个合格、空闲且当前尚不安全的工人/阶段组合做整数二分，复用原安全投影函数、时间量化和安全阈值。非负疲劳系数与累积率使安全谓词随恢复时间单调；按全部组合的最早 tick 返回。无恢复率、超过 horizon 或最终仍不安全时返回原有的不可恢复结果。

WAIT 证书按 `(state_version, current_tick, horizon_tick, decision_type)` 缓存，并在现有资源状态失效入口清除。缓存、返回值与最近一次证书分别保存独立字典；reset、阶段切换与 WAIT 投影状态均受失效规则保护。

## 修改前后实际测量

RTX 4060 Laptop 8 GiB，PyTorch 2.7.1+cu126，4 个 Torch 线程。下表来自性能提交 `5f2216c` 对应工作区的修改前后对照，使用保存的两个源码文件和同一有效配置快照；主线集成验证单独记录如下。

网络输入来自三条环境路径的 60 个观测，包含生产与工人阶段；大于 60 的 batch 循环使用观测。同一参数严格加载前后网络，并将专家 context 与 residual 权重设为非零以覆盖学习后的评分。计时包含 forward 和全部诊断的取回，预热后交替测量 7 次，CUDA 边界同步，取中位数。

| 图 batch | 修改前 | 修改后 | 加速比 |
|---:|---:|---:|---:|
| 1 | 14.00 ms | 12.67 ms | 1.11× |
| 20 | 74.24 ms | 22.98 ms | 3.23× |
| 40 | 127.03 ms | 27.67 ms | 4.59× |
| 64 | 199.75 ms | 36.32 ms | 5.50× |

全部诊断字段一致，浮点诊断最大观测差为 `5.96e-8`；最大 logit 差为 `2.98e-8`。独立随机数流下四个 batch 的 sampled 动作均匹配。前后 state_dict 键和 network_spec 一致。浮点归约的末位差异不构成所有后续训练权重逐位一致的保证。

环境对照采用三条完整启发式轨迹，交替计时 3 次取中位数；计时包含 reset、step 与完整图观测。

| 实例 | 完整步数 | 修改前 | 修改后 | 加速比 |
|---|---:|---:|---:|---:|
| 固定 15×4 | 211 | 3.192 s | 1.880 s | 1.70× |
| validation_balanced_2000000 | 288 | 4.197 s | 2.976 s | 1.41× |
| validation_machine_bottleneck_2000001 | 289 | 4.441 s | 2.979 s | 1.49× |

共 788 步的完整节点/边/global/action-set 特征、偏好、动作 mask、奖励、终止标志、step info、物理 tick、调度日志、重构日志及最终 metrics 全部一致。

这些是网络和环境的局部加速比，尚未测量完整正式训练的总墙钟；不能相乘估计总加速比。

## 验证

新增 `test/test_runtime_performance.py` 覆盖 CPU/CUDA 诊断对照、objective-expert/shared-head 两种诊断字段、混合偏好、排名平局、WAIT-only、pair-only、masked WAIT 和近似常量统计。环境检查覆盖拆卸/安装恢复的穷举 tick 对照、horizon 边界、多个工人和阶段、零恢复率、资格/忙碌限制、不可恢复阶段、对数级投影调用，以及缓存隔离、失效、reset 和 WAIT 投影。

与环境、动作 mask、已有性能优化、golden 观测轨迹、PPO、V8 策略、批量数值/梯度、时间上下文和奖励回归一起，**96 项测试通过**（34.29 s）。扩展回归同时修正了 V8 奖励测试中写死的失败惩罚期望，改为读取有效配置；奖励实现与当前全局惩罚 2.0 保持原值。

CUDA 训练冒烟通过：两个 16 步 cutoff rollout、一次 PPO update、一条 sampled validation、best checkpoint 保存与重载、一条独立 sampled final evaluation。验证和最终评估完成率均为 100%，安全检查通过，最终调度违例为 0。该冒烟约 72.54 s，只验证执行链路；正式训练配置未调整。

原始测量、修改前源码、JUnit XML 和冒烟产物保存在被忽略的 `result/audits/performance_implementation_20261003/`。复跑测量入口：

```powershell
.\.venv\Scripts\python.exe result\audits\performance_implementation_20261003\verify_and_benchmark.py
```

## 主线集成验证

以远端 `main` 的 `29873f0` 为基准，将诊断归约接入主线现有 `agent/ppo/network.py`。主线 V2 数据、冻结尺度、失败惩罚 2.0、启动脚本及消融协议沿用现有配置。

整合后，运行时/PPO/协议回归 **103 项通过**，无疲劳、结构消融和消融协议回归 **94 项通过**，合计 **197 项通过**。主线 CUDA 冒烟约 **46.06 s**，完成 rollout、PPO、sampled 验证、checkpoint 保存重载与独立最终评估；验证和最终评估均完成，安全检查通过，最终调度违例为 0。JUnit 与冒烟产物保存在本地审计目录。
