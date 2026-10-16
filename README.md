# Fatigue-aware reconfigurable assembly scheduling

疲劳感知的可重构装配调度。PPO 基于异质图观测，联合选择生产工序—机器配对和重构机器—工人配对；三目标为订单 Flow、重构费用及工人实际劳动负担。环境负责事件推进、动作掩码、资源互斥与疲劳安全。

## 当前主线

`schema12 + log2 + HGNN shared-head + excess Flow + geometric OWA(q=.5)`。

- PPO：`gamma=1`、`gae_lambda=.95`，每采集20 episodes更新；额外势函数塑形关闭。
- 冻结尺度：`(368.3143, 353.27, 21.03145448655653)`，来源为[excess/OWA清单](configs/manifests/workload_excess_q50_20261011.json)。
- 全局目标输入为 `log2(1+J/s)`；训练质量坐标为 `J/(s+J)`。
- 正式评价：同实例、同偏好三次采样，成功优先、原生Q更低优先取一份交付；所有请求的成本、安全与完整性均保留。
- Universal选模：交付后的最差偏好完成率优先，再比较偏好平衡Q。训练buffer、奖励和GAE保持原始单轨迹口径。

| 配置 | episodes | 每次更新 | 验证间隔 |
|---|---:|---:|---:|
| [Universal](configs/default.json) | 2000 | 20 | 100 |
| [Flow](configs/e1/single_flow.json) | 1000 | 20 | 40 |
| [Cost](configs/e1/single_cost.json) | 1000 | 20 | 40 |
| [Workload](configs/e1/single_workload.json) | 1000 | 20 | 40 |

固定V2数据：500个验证实例，test/OOD/stress各20个。主验证取固定分层50实例×13偏好×3次采样；Universal测试为20实例×66偏好×3次采样。当前数据规模与更大规模发布方案分开记录，详见[实验清单](docs/experiment_runlist.md)。

## 安装与检查

本机 `.venv` 链接到共享环境 `C:\Users\chenz\myenv`。新环境按[依赖文件](requirements.txt)安装；以下命令在项目根目录运行。

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.run_experiments --suite all --seed 11 --preflight-only
.\.venv\Scripts\python.exe -m scripts.run_00_smoke
```

预检验证配置、数据、冻结身份、硬件和网络前向；smoke检查训练/保存/重载/评价链路。正式完成以summary、checkpoint及完整结果行核实。

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.t
```

旧V1清单按 `.gitattributes` 使用CRLF换行，原归档哈希保持不变；当前实验使用 `data/manifests/v2`。运行时只接受schema12配置与模型。

## 训练

当前先运行种子 **11、23、37**。三个seed的Universal批次：

```powershell
.\scripts\run_three_seeds.ps1 -Experiment universal
```

该脚本先后执行三个种子，任一失败停止；`-PreflightOnly`只预检，`-DryRun`只打印命令，`-List`列出26个实验。每个实验的单独命令见[三种子逐项清单](docs/experiment_three_seeds.md)。

脚本优先使用已激活的 virtualenv/Conda 环境，其次使用项目`.venv`，最后查找 PATH 中的`python`；`-Python`可显式指定解释器路径或命令名。

一次Universal训练示例：

```powershell
.\.venv\Scripts\python.exe train.py --config configs/default.json --algorithm-seed 11 --run-name universal_seed11
```

一个种子的完整核心矩阵包括Universal、无图、三个单目标及其疲劳中性对照，共8次训练：

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiments --suite all --seed 11
```

两条命令是不同启动方式；完整矩阵已包含Universal。通用启动器支持 `--config` 列出对照、`--seed` 指定算法种子，预检全部配置后顺序启动，任一训练失败停止。`--parallel-envs`调整并行宽度，保留每次更新20 episodes的预算。

## 评估与分析

使用训练目录保存的配置读取模型：

```powershell
.\.venv\Scripts\python.exe eval.py --config result/runs/universal_seed11/config.json --policy ppo --checkpoint result/runs/universal_seed11/best_checkpoint.pt --dataset test --decode-mode sampled --preference-set final_test --run-name universal_seed11_test
```

训练末尾已自动测试选出的best；独立评估用于复核或补齐产物。`eval.py`还支持启发式和随机策略；MO-ALNS入口为 `python -m scripts.mo_alns`。其预算、适用参数和完整比较命令见[实验执行清单](docs/experiment_runlist.md)。

结果保留有效配置、源码/数据/尺度/权重身份、训练与验证日志、全部原始请求和selected交付行。历史逐采样结果与新协议结果分别标记；原始样本池Pareto/HV和三次取优交付指标分别报告。

## 代码与文档

`agent`负责策略与求解器，`environment`负责物理与观测，`data`负责实例，`configs`负责配置/身份，`training`负责选模，`result`负责指标和产物；`scripts`提供运行入口，`analysis`及`analysis/studies`提供分析和历史复核。

- [全部实验及执行顺序](docs/experiment_runlist.md)
- [主线实验协议](docs/experiment_protocol.md) · [三次取优与选模操作协议](docs/default_best_of_three.md) · [消融协议](docs/ablation_protocol.md)
- [数学证明目录](data/docs/README.md) · [选模证明](data/docs/SELECTION.md)
- [当前架构](docs/ARCHITECTURE.md) · [整理与路径迁移](docs/project_cleanup.md)
- [工业背景](docs/research_background_cesa3r.md) · [fluid搜索研究设计](docs/fluid_bottleneck_search_ppo_research_20261014.md)
