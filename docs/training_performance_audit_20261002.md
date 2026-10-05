# 训练耗时排查（2026-10-02）

## 结论

主线 `main`（`84191c2`）保留批量 PPO、阶段批量动作头、集中批量推断及合法候选评分优化。2026-10-02 再次通过 `git ls-remote origin refs/heads/main` 核实，GitHub 主线也指向 `84191c20da51189d76154afd9722f01476279cb1`。

用户提供的远端外层目录位于提交 `a748ccf`，早于性能优化提交 `3f241ab`。远端工作区没有已跟踪文件的改动，存在一个未跟踪的同名子目录 `fatigue_aware_reconfigurable_assembly_scheduling/`。这证明该外层 Git checkout 仍旧，但实际训练源码的绝对路径尚未直接读取；不能只凭外层 Git 状态认定用户没有下载最新主线。

用户给出的启动位置为 `C:\Users\Administrator\Projects\fatigue_aware_reconfigurable_assembly_scheduling`，命令直接运行该目录的 `train.py`。在没有另行改变代码路径的情况下，这会执行旧实现。最新代码可能解压进同名子目录，但该子目录内容尚未远端核实。

## 性能优化来源与合并历史

| 时间 | 提交 | 变更与主线关系 |
|---|---|---|
| 2026-09-28 18:52 | `3f241ab` | 在 `codex/sampled-formal-evaluation` 上实现阶段批量动作头、批量 sampled 验证、批量图打包及减少 GPU 到 CPU 的同步 |
| 2026-09-28 20:50 | `75f1c31` | Flow 相对时间配置采用训练 40 worker、验证 40 worker、每次更新 40 episode |
| 2026-09-28 22:30 | `db9822b` | 3.82 h Flow 运行记录的提交；已含上述优化 |
| 2026-09-29 12:59 | `e93600d` | PR #8 将优化分支合入 `main`，`3f241ab`、`75f1c31`、`db9822b` 都是当前主线祖先 |
| 2026-09-29 14:00 | `c0532df` | 3.80 h Flow 运行记录的提交；沿用批处理优化，另将 E1 失败惩罚设为 5 |
| 2026-09-30 17:33 | `09d31f0` | 将 V8 网络合并到 `agent/ppo/network.py`，并将 E1 配置统一为 20 / 20 / 20；PR #9 在 `ff40142` 合入主线 |
| 2026-10-02 03:01 | `476bb79` | 主线集成 schema-6 时间上下文、稀疏合法候选评分及独立 critic 求值；PR #12 在 `84191c2` 合入 |

两次 Flow 运行分别为：

- `e1_flow_relative_time_seed11_ep1000_rerun_20260928_223621`：3.8233 h，Git 记录 `db9822b`。
- `e1_flow_relative_time_seed11_ep1000_penalty5_20260929_140255`：3.7955 h，Git 记录 `c0532df`。

虽然运行来源都标记 `dirty=true`，按仓库 provenance 算法重新计算对应 Git 提交的 57 个 Python 文件指纹，分别与运行记录的 `source_state_sha256` 完全一致。这排除了“提速只存在于未提交 Python 改动”的疑点。配置快照另行记录，Flow 两次均为 40 / 40 / 40 且执行版本 `phase_batched_v1`。

通过 Python AST 比较函数体，`PPOAgent.act_batch()` 和 `PPOAgent.update()` 从 `3f241ab` 到当前 `main` 完全一致。`HeteroGraphActorCritic._collate_graphs()` 与 `_phase_logits_grouped()` 从旧 `network_v8.py` 到清理提交的 `network.py` 函数体也完全一致。当前主线进一步升级为稀疏候选实现，默认仍为 `phase_batched_v1`，sampled 验证仍集中调用 `act_batch()`。

`c0532df` 本身不是当前主线祖先；其失败惩罚 5.0 配置没有按原提交合入，主线当前默认惩罚为 1.0。这与性能优化是否合并是不同事项：性能优化的祖先提交 `3f241ab` 已通过 PR #8 合入，且实现仍在。

原始历史核对结果：`result/audits/training_performance_20261002/git_history.json`；复跑脚本为同目录的 `git_history.py`。

## 交叉证据

- 远端 `git rev-parse --short HEAD` 返回 `a748ccf`。
- 该提交的 V8 `forward_batch()` 逐图调用 `_phase_logits()`，尚未包含 `3f241ab` 引入的阶段批量动作头。
- 该提交的 sampled 验证逐条调用 `agent.act()`；优化版本集中调用 `agent.act_batch()`。
- 该提交每次正式验证附加 greedy 验证，控制台显示 `greedy complete`；用户贴出的训练日志符合这一版本。本地主线当前入口没有这条输出。
- 该提交仅在训练结束后写 `update_log.csv` 和 `validation_log.csv`。远端在 episode 800 时查不到这两个文件，与旧实现一致。
- 该提交默认 generator 为 `1.3.0`、数据目录为 `data/manifests`，而当前主线为 V2。run 名称中的 `v2_timecontext` 不会启用 V2 数据或 schema-6 时间上下文。

## 历史完整运行

从本地保存的远端 `summary.json`、配置快照及 CSV 读取，时间为完整运行实测。

| Run | 训练/验证 worker | 完整耗时 |
|---|---:|---:|
| `e1_flow_relative_time_seed11_ep1000`（优化前） | 20 / 20 | 13.11 h |
| `e1_flow_relative_time_seed11_ep1000_penalty5_20260929_140255`（优化后） | 40 / 40 | 3.80 h |
| `flow_v2_seed11_1000`（时间上下文新增前） | 20 / 20 | 5.11 h |
| `cost_v2_seed11_1000` | 20 / 20 | 3.83 h |
| `variance_v2_seed11_1000` | 20 / 20 | 5.96 h |

V2 Flow 的 5.11 h 中：训练采样与 PPO 更新合计 0.90 h，正式验证 3.48 h，独立诊断验证 0.67 h。总耗时主要在验证路径。

`09d31f0` 将 E1 对齐到 20 个训练/验证 worker、每次 PPO 更新 20 个 episode。旧 Flow 的 40 / 40 / 40 配置不再是主线默认值。这是另一项性能配置变化，但不是远端执行旧代码的替代解释。

## 当前主线时间上下文的独立测量

在本机比较 `main:environment/env.py` 和新增时间上下文之前的 `ba08892:environment/env.py`，共重放 3 个实例、452 个相同动作。每条动作的 mask、奖励、终止状态及物理 tick 一致。每条路径预热一次，交替测量三次，取中位数。

| 实例 | 原环境 | 当前主线 | 变慢比例 |
|---|---:|---:|---:|
| 固定 15×4 实例，160 步 | 2.140 s | 2.965 s | 1.39× |
| V2 easy 验证实例，132 步 | 1.660 s | 2.286 s | 1.38× |
| V2 balanced 验证实例，160 步 | 1.897 s | 2.969 s | 1.57× |

环境 CPU profiler 中，订单链估计 `finish_ticks()` 累计 6.43 s，占整段 profile 23.85 s 的约 27%。这是新增的可优化开销，但上述局部环境结果不等于完整训练耗时，也不能用于宣称远端新版本需要 12 h。此次远端运行尚未提供完整耗时文件，其提交与输出首先指向旧版本。

## 修复方向

核实远端最新代码所在目录，确认实际启动的 `train.py`、配置与模块都来自同一份最新主线；再以新 run 名运行。当前 PowerShell 中用分号串联的三次训练会依次启动，后两个命令仍会使用该启动目录。

如果决定更新外层 Git checkout，应在当前训练序列结束或明确停止后更新，再重启；运行中的 Python 主进程不会因为下载新代码自动换成新实现。历史 run 应保留各自配置、模型与结果，不应仅改名后解释为新版本实验。

本次只执行只读代码/历史排查和独立重放测量，未修改训练实现、运行配置或远端进程。

本地原始记录：`result/audits/training_performance_20261002/history.json`、`environment_benchmark.json`、`main_environment.prof`；复跑脚本为同目录的 `audit.py`。
