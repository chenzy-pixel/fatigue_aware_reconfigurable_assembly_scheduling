# NSGA-II＋LS / MOEA/D：效率预检与1800秒搜索

本入口用于已返回600秒批次的预算扩展。默认复用相同7个实例、4个偏好和搜索种子11/23/37，固定NSGA-II＋LS种群16、MOEA/D种群32。每个实例—偏好—搜索种子独立获得1800秒，包含初始化、完整原生环境解码、局部搜索和档案更新。正式请求不共享其他请求或预检产生的候选。

1800秒是用户批准的下一档预算，充分性尚未确认。旧600秒产物保持原样；新代码和并行度生成新身份。本入口只运行这两种启发式，不启动训练或MO-ALNS。

## 已完成的效率核验

本机2026-10-21固定候选测试覆盖7种压力、5种解码规则，共35条轨迹。优化前后动作、原始指标、排程、重构、诊断和目标结果一致；总解码耗时65.42→18.10秒，约3.61倍。两种搜索在固定40次完整评价时，候选顺序、交付方案、算子与预算计数也保持一致。该测试证明实现等价及本机提速，不证明1800秒质量或远端加速比。

热点与修改：

- WAIT完成估计原来在每个工序查询中重建整份工序索引；改为一次查询内复用。
- WAIT时长改善检查原来重复计算同一机器—模块组合，并在空闲工人全部恢复后继续扫描。改为去重，并在下一事件或完全恢复后一个保护tick处截止；原时长函数、量化与最早改善判据保持不变。
- 解码动作按工序优先级作为词典序第一项。先保留真正可能获选的工序，再评分其资源，避免计算必败候选。

本机并行基准排除了进程启动时间：1/2/4进程分别约2.75/3.71/6.53次完整解码/秒，消耗约0.95/1.90/3.84个CPU核，各档动作摘要一致。4进程吞吐约为1进程的2.37倍；该结果不替远端选择并行度。

纯Python搜索中的每个进程主要使用一个CPU核。旧入口的8进程上限在32逻辑CPU机器上对应约25%的整机占用量级；父进程低占用或GPU低占用也不能直接证明搜索空转。应同时看所有worker、每请求CPU时间和完整解码吞吐。

可核查产物位于`result/analysis/search_efficiency_20261021/`：CPU profile、35条候选前后快照、固定评价次数的搜索快照、并行吞吐、测试日志及新入口的小预算链路验证。

## 远端启动

先同步本轮源码，保留远端固定V2数据及原600秒结果目录。默认源目录为`result/analysis/heuristics_seed11_20261019`，必须包含`identity.json`、`subsets.json`和`search_selection.json`。入口从原运行身份读取完整配置，避免当前训练分布默认值变更影响本轮配对；同时核对validation manifest和21个源实例文件哈希。

默认只预检：

```powershell
.\scripts\run_heuristics_1800.ps1 -Python python
```

在远端手动执行完整批次：

```powershell
.\scripts\run_heuristics_1800.ps1 -Stage Run -Python python
```

`Run`在`Workers=auto`时依次完成并行基准→身份预检→1800秒搜索→汇总。基准默认尝试4/8/12进程，按可测物理核及可用内存筛选；内存准入采用每进程预留1GiB、保留25%余量的工程估计。没有可用硬件监测库时，记录未知字段并按逻辑CPU筛选。各档先等待所有进程就绪，再测固定候选吞吐；选择达到最高吞吐95%以上的最少进程数。此入口允许超过8进程，实际并行度由远端基准决定。

也可显式指定并行度，跳过自动基准：

```powershell
.\scripts\run_heuristics_1800.ps1 -Stage Run -Python python -Workers 8
```

其他源目录、21例调参集或只跑seed11可显式指定：

```powershell
.\scripts\run_heuristics_1800.ps1 -Stage Run -Python python -SourceDir result/analysis/heuristics_seed11_20261019 -OutputDir result/analysis/heuristics_1800_tuning21_v1 -Subset tuning21 -Seeds 11
```

以上示例的输出身份不同，结果分开保存。默认7例完整矩阵为168个请求，预算合计84单进程小时；21例×3种子则为504个请求、252单进程小时。并行基准、进程启动和写盘成本单独发生，不把预算总和当成可靠墙钟ETA。

## 进度、恢复与结果

运行每30秒打印完成数、活跃进程上限和队列下限；每个完成请求打印成功交付Q、完整评价次数及实际CPU核使用量。`process_cpu_cores_used≈1`表示单进程接近持续使用一个CPU核，并非占满整机。

默认输出目录`result/analysis/heuristics_1800_v1/`包含：

- `benchmark.json`：各并行档的启动时间、稳态吞吐、CPU核用量、可测RSS及建议并行度；这是效率测试，不是算法质量实验。
- `identity.json`、`preflight.json`：原批次身份、实际保存配置、数据/源码哈希、硬件软件、1800秒预算和并行度。
- `requests/<method>/*.json`：完整候选记录、完成时刻、搜索代/扫描、选中方案、动作、排程和原生约束检查；原子写入。
- `summary.json`：完整矩阵核验后的成功率、平均Q、完整评价数和CPU核用量，保持`budget_sufficiency=not_established`。

汇总核对精确请求集合、请求身份、预算截止前最优候选及选中排程的原生安全/完成指标。解码次数、`env.step`、局部搜索、代/扫描和耗时均保留；不同种子的结果独立汇总，不跨种子再取优。

相同命令可恢复：完整请求按身份跳过，未完整写出的请求从头执行。源码、数据、配置、预算、种子、子集或并行度变化时使用新输出目录。自动并行基准的缓存也只在相同源码/配置/数据/硬件身份下复用。

若要单独运行Python入口：

```powershell
python -u -m scripts.heuristic_search --stage benchmark --output-dir result/analysis/heuristics_1800_v1
python -u -m scripts.heuristic_search --stage preflight --output-dir result/analysis/heuristics_1800_v1
python -u -m scripts.heuristic_search --stage run --remote-run --output-dir result/analysis/heuristics_1800_v1
```

更长预算可用`-Seconds`显式指定并另建目录；当前批准批次使用默认1800秒。充分性仍需分析600→1800及末段收益，并在独立确认集验证；提高求解吞吐不等于证明启发式收敛。
