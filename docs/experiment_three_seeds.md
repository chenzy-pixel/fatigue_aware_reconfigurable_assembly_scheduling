# 三个seed逐实验运行清单

本阶段种子固定为 **11、23、37**。每行命令运行一个实验的三个seed；命令在项目根目录的PowerShell执行。脚本默认仅运行Universal，不自动展开全部对照。

[启动脚本](../scripts/run_three_seeds.ps1)复用scripts.run_experiments，逐实验、逐seed串行运行，失败立即停止。物理/网络/目标配置和每次20 episodes更新预算沿用对应JSON。重复执行会创建新run，不会自动续训或跳过已有结果。

| 编号 | 分组 | 实验 | 每seed episodes | 三seed命令 |
|---|---|---|---:|---|
| 01 | 核心 | [Universal主线](../configs/default.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment universal` |
| 02 | 核心 | [Flow单目标](../configs/e1/single_flow.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment single_flow` |
| 03 | 核心 | [Cost单目标](../configs/e1/single_cost.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment single_cost` |
| 04 | 核心 | [Workload单目标](../configs/e1/single_workload.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment single_workload` |
| 05 | 核心 | [无图编码对照](../configs/ablations/no_graph.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment no_graph` |
| 06 | 核心 | [Flow疲劳中性](../configs/ablations/neutral_flow.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment neutral_flow` |
| 07 | 核心 | [Cost疲劳中性](../configs/ablations/neutral_cost.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment neutral_cost` |
| 08 | 核心 | [Workload疲劳中性](../configs/ablations/neutral_workload.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment neutral_workload` |
| 09 | 单因素 | [raw Flow](../configs/controls/raw.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment raw` |
| 10 | 单因素 | [Variance目标](../configs/controls/variance.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment variance` |
| 11 | 单因素 | [bounded编码](../configs/controls/bounded.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment bounded` |
| 12 | 单因素 | [GAE lambda=.97](../configs/controls/lambda097.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment lambda097` |
| 13 | 单因素 | [GAE lambda=.99](../configs/controls/lambda099.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment lambda099` |
| 14 | 单因素 | [OWA q=.25](../configs/controls/owa_q025.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment owa_q025` |
| 15 | 单因素 | [OWA q=.75](../configs/controls/owa_q075.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment owa_q075` |
| 16 | 条件扩展 | [raw × bounded](../configs/controls/raw_bounded.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment raw_bounded` |
| 17 | 条件扩展 | [Variance × bounded](../configs/controls/variance_bounded.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment variance_bounded` |
| 18 | 条件扩展 | [单目标Variance](../configs/controls/single_variance.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment single_variance` |
| 19 | 条件扩展 | [单目标OWA q=.25](../configs/controls/single_owa_q025.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment single_owa_q025` |
| 20 | 条件扩展 | [单目标OWA q=.75](../configs/controls/single_owa_q075.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment single_owa_q075` |
| 21 | 条件扩展 | [raw × OWA q=.25](../configs/controls/raw_owa_q025.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment raw_owa_q025` |
| 22 | 条件扩展 | [raw × OWA q=.75](../configs/controls/raw_owa_q075.json) | 2000 | `.\scripts\run_three_seeds.ps1 -Experiment raw_owa_q075` |
| 23 | 条件扩展 | [raw × 单目标Variance](../configs/controls/raw_single_variance.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment raw_single_variance` |
| 24 | 条件扩展 | [raw × 单目标OWA q=.25](../configs/controls/raw_single_owa_q025.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment raw_single_owa_q025` |
| 25 | 条件扩展 | [raw × 单目标OWA q=.5](../configs/controls/raw_single_owa_q05.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment raw_single_owa_q05` |
| 26 | 条件扩展 | [raw × 单目标OWA q=.75](../configs/controls/raw_single_owa_q075.json) | 1000 | `.\scripts\run_three_seeds.ps1 -Experiment raw_single_owa_q075` |

核心8项共24次训练；单因素7项共21次；条件扩展11项共33次。当前核心加单因素为45次训练，全部展开为78次。条件扩展按研究问题选择，不默认全部执行。

## 先预览与预检

```powershell
.\scripts\run_three_seeds.ps1 -Experiment universal -DryRun
.\scripts\run_three_seeds.ps1 -Experiment universal -PreflightOnly
.\scripts\run_three_seeds.ps1 -Experiment universal
```

## 选择多个实验或控制资源

```powershell
.\scripts\run_three_seeds.ps1 -Experiment single_flow,single_cost,single_workload
.\scripts\run_three_seeds.ps1 -Experiment universal -Device cuda -ParallelEnvs 20 -ValidationParallelEnvs 20
```

默认Python为项目`.venv/Scripts/python.exe`；其他环境可用`-Python C:\path\to\python.exe`。资源参数省略时沿用配置。可用`-List`查看实验名和配置映射。

## 其他实验逐项清单

以下不是本PPO训练脚本的实验名，执行依赖和缺口见[完整实验清单](experiment_runlist.md)。

| 编号 | 实验 | 当前三seed范围/预算 |
|---|---|---|
| B01 | MO-ALNS求解器 | seeds11/23/37 × test/OOD/stress，共9个数据集作业；每偏好最多300候选评价 |
| B02 | 启发式基线 | test/OOD/stress各一次确定性物理轨迹，再按公共偏好评分 |
| B03 | 随机策略基线 | 三个种子 × 三个数据集；完整网格包装待补 |
| E01 | Universal OOD全偏好评估 | 三个best，66偏好×3请求；正式CLI包装待补 |
| E02 | Universal stress全偏好评估 | 三个best，66偏好×3请求；正式CLI包装待补 |
| M01 | 偏好响应 | 原始及selected结果，按目标/偏好/完成覆盖分析 |
| M02 | OWA劳动机制 | 总劳动、最大劳动、Variance、OWA及集中项 |
| M03 | 三次取优增益 | 同一checkpoint的一次采样与三次交付比较 |
| M04 | 选模口径敏感性 | 完整验证日志重选episode；需要相应权重才可测试 |
| M05 | 新旧训练控制协议 | 可选新增per_sample配置×3seed，未包含在78次中 |
| M06 | critic/GAE诊断 | 冻结策略轨迹，比较估计误差和方差 |
| M07 | 批量/投影效率 | 输出等价前提下比较多次计时及端到端成本 |
| M08 | best/last稳定性 | last作为预先声明的独立诊断，不反向选模 |

MO-ALNS现有manifest已改为三个种子，可单独执行：

```powershell
.\.venv\Scripts\python.exe -m scripts.mo_alns_benchmark --manifest configs/baselines/mo_alns_manifest.json --output-dir result/analysis/mainline_mo_alns_three_seeds
```

本清单与脚本的交付不表示正式训练已经启动。
