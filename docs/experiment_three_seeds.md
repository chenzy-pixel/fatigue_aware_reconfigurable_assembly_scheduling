# 三个seed逐实验运行清单

更新：2026-10-17。本阶段种子固定为 **11、23、37**。前26行对应原有PPO实验，每行命令运行一个实验的三个seed；命令在项目根目录的PowerShell执行。脚本默认仅运行Universal，不自动展开全部对照。新匹配PPO、DCAN、DDQN和搜索使用后文独立套件，不能作为本脚本的`-Experiment`名称传入。

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

默认优先使用已激活的 virtualenv 或 Conda 环境，其次使用项目`.venv/Scripts/python.exe`，最后查找 PATH 中的`python`。可用`-Python C:\path\to\python.exe`或`-Python python`显式指定；同名命令有多个匹配时使用第一个。资源参数省略时沿用配置。可用`-List`查看实验名和配置映射。

## 独立评估

PPO主线训练结束时已自动测试验证选出的best。需要复核或补齐测试产物时，使用训练目录中的保存配置与权重；将下例`<实际run_name>`替换为该次训练的真实目录名，在已激活Python环境的项目根目录执行：

```powershell
$evaluationRunDir = 'result/runs/<实际run_name>'
python eval.py --config "$evaluationRunDir/config.json" --policy ppo --checkpoint "$evaluationRunDir/best_checkpoint.pt" --dataset test --decode-mode sampled --preference-set final_test --run-name universal_test_review
```

此例适用于Universal的66偏好test复核；新套件的DCAN/DDQN及其冻结模型使用下文套件入口。原有PPO入口的OOD/stress范围与公共坐标要求见[实验清单](experiment_runlist.md)。

## 其他实验逐项清单

以下不是本PPO训练脚本的实验名，执行依赖和缺口见[完整实验清单](experiment_runlist.md)。

| 编号 | 实验 | 当前三seed范围/预算 |
|---|---|---|
| B01 | MO-ALNS v2 | 搜索种子11/23/37；21例×4偏好曲线，600秒起步、必要时1800秒；先选预算，再确认及生成正式manifest |
| B02 | 启发式基线 | test/OOD/stress各一次确定性物理轨迹，再按公共偏好评分 |
| B03 | 随机策略基线 | 预先声明算法种子和采样预算；完整网格包装待补 |
| C01 | 新匹配PPO / DCAN / Rule-DDQN | 各3训练种子，共9次正式训练，每次350411次env.step；另有各3学习率、seed11的9次35041步试验 |
| C02 | NSGA-II-LS / MOEA/D | 各种群16/32/64验证调参，冻结后21例×4偏好×3搜索种子跑600→必要时1800秒曲线 |
| C03 | 独立确认与短时限 | 7例×13偏好×3种子；每臂273份交付，11臂3003份；充分预算与PPO实测交付时限分列 |
| E01 | Universal OOD全偏好评估 | P1三个best仍待原CLI包装；suite的冻结模型已有formal-evaluate入口，66偏好 |
| E02 | Universal stress全偏好评估 | 同上；suite的all不自动展开正式split |
| E03 | 独立瓶颈与重构倍率challenge | 已物化100+80例；每个冻结PPO模型11880交付/35640原始轨迹；三模型各自执行，当前CLI仅PPO/启发式 |
| M01 | 偏好响应 | 原始及selected结果，按目标/偏好/完成覆盖分析 |
| M02 | OWA劳动机制 | 总劳动、最大劳动、Variance、OWA及集中项 |
| M03 | 三次取优增益 | 同一checkpoint的一次采样与三次交付比较 |
| M04 | 选模口径敏感性 | 完整验证日志重选episode；需要相应权重才可测试 |
| M05 | 新旧训练控制协议 | 可选新增per_sample配置×3seed，未包含在78次中 |
| M06 | critic/GAE诊断 | 冻结策略轨迹，比较估计误差和方差 |
| M07 | 批量/投影效率 | 输出等价前提下比较多次计时及端到端成本 |
| M08 | best/last稳定性 | last作为预先声明的独立诊断，不反向选模 |

## 独立挑战集评价

[challenge_v1协议](challenge_suite_protocol.md)给出build/audit/smoke/evaluate/report命令。已有冻结数据可直接审计并评价；三个训练种子须分别指定各自`--training-config`、`--checkpoint`和`--output-dir`，不能把默认历史seed11权重重复评价记作三个训练模型。该入口不接受`run_three_seeds.ps1 -Experiment`名称，也不经`eval.py --dataset test_hard`调用。

每模型在100例test_hard与80例倍率配对上各跑66偏好，PPO每偏好三次取优；启发式每实例只跑一次物理轨迹并重评分。模型与参数冻结后才使用这些集合，结果按实例/配对基础组统计。现有9例smoke只用于链路验收。

## 新对照套件：远端单独启动

以上C01–C03及套件内MO-ALNS重新校准使用[远端启动说明](comparator_remote_startup.md)，不计入原78次PPO矩阵。远端激活Python/CUDA环境后，先预检：

```powershell
.\scripts\run_comparison_suite.ps1 -Python python -OutputDir result/analysis/comparison_suite_remote_v1
```

预检通过后执行：

```powershell
.\scripts\run_comparison_suite.ps1 -Stage all -RemoteRun -Python python -OutputDir result/analysis/comparison_suite_remote_v1 -Workers 8 -TrainingWorkers 20
```

该批次在21调参/21选模/7确认的互斥验证划分上完成调参、匹配训练、曲线和独立确认，最后生成正式manifest。PPO/DCAN主交付为三次取优，DDQN为贪心单次；三训练种子与三次采样分别记录。完整test/OOD/stress在预算及独立改善门槛满足后，另行执行`--stage formal-evaluate --remote-run`，保持同一输出目录。

单独的旧参照MO-ALNS研究仍有[分阶段入口](mo_alns_v2_implementation_20261017.md)，固定的是历史seed11 PPO，不能替代新套件三模型匹配比较。默认600秒或旧300候选设定均不作为已确认的正式预算。用户在远端手动启动，本机已取消的长实验不再续跑；文档、预检和smoke不代表正式成绩。
