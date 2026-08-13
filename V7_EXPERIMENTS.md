# v7 E0–E5 实验套件

本套件的代码版本为 `policy-head-v7-code`，协议固定为
`v7_e0_e5_protocol_v1`，结果 schema 为 `4.0.0`。历史 v6 checkpoint
及其结果目录只读保留；新训练全部从头开始。P5/E6 不在本套件中，所有 arm
都强制保持 `training.forced_action_compression=false`，并保留 non-delay
派工约束。

## 一键启动

在项目根目录用 PowerShell 执行：

```powershell
.\run_all_experiments.ps1 -Stage All -ParallelEnvs 10
```

可选阶段为 `Smoke`、`Screen`、`Formal`、`Audit`、`All`。先检查命令而不执行：

```powershell
.\run_all_experiments.ps1 -Stage All -ParallelEnvs 10 -DryRun
```

编排器在 CPU-only 环境中始终串行启动训练进程；`ParallelEnvs` 只控制每个
训练进程内部的并行环境数。状态原子写入
`result/experiments/v7_e0_e5_protocol_v1_<hash>/state.json`。代码、协议、配置或
固定数据 manifest 任一哈希变化都会生成新的套件目录，不会复用旧结果。失败
任务默认重试一次；已完成且产物有效的任务按 input hash 跳过。

`All` 的顺序是：v7 单元测试与七个 arm 冒烟、E0 五个历史 v6 checkpoint
审计、21 个 600-episode 筛选训练、安全门、35 个 2000-episode 正式训练，
最后在 test/OOD/stress 上执行 greedy 和 sampled seeds
100011/100012/100013 的统一评估与汇总。

## Arm 与配置

| arm | 配置 | 唯一处理变化 |
|---|---|---|
| C0 | `configs/v7/c0_v6_control.json` | v6 网络按 v7 守门协议重训 |
| E1 | `configs/v7/e1_context_exception.json` | ranker 尺度有界 residual，gate=-2 |
| E2 | `configs/v7/e2_commit_set.json` | production commit-set logit |
| E3 | `configs/v7/e3_future_value.json` | 未来复用与资格稀缺特征 |
| E4 | `configs/v7/e4_conditional_wait.json` | non-delay 下的条件式 worker wait |
| E5 | `configs/v7/e5_variance_scale.json` | `variance_scale: 50→20` |
| FULL | `configs/v7/full_v7.json` | 合并 E1–E5 |

编排器会在启动前将各 arm 与 C0 做递归配置差异比较，任何不在白名单中的变化
都会拒绝运行。E5 另有严格断言，除实验名称外只能改变 `reward.variance_scale`。

## 输出与统计

每次训练仍写入 `result/runs/<run-name>/`；正式 checkpoint 记录生效配置、网络与
特征 schema、源码状态 SHA-256、算法 seed、协议和结果 schema。汇总套件目录包含：

- `e0_checkpoint_audit.json`：五个历史 v6 checkpoint schema/哈希审计；
- `screen_gate.json`：21 个筛选运行的安全检查；
- `formal_per_seed.csv`：按算法 seed 汇总的正式指标；
- `formal_wilcoxon.csv`：同 seed treatment−C0 的 exact Wilcoxon；
- `formal_aggregate.json`、`REPORT.md` 和 `formal_comparison.svg`。

正式推断以五个算法 seed 为独立单位。先求同 seed 的 treatment−C0，再做双侧
exact Wilcoxon。五个非零配对时最小双侧 p 值为 0.0625；逐实例结果只能作为
探索性证据。
