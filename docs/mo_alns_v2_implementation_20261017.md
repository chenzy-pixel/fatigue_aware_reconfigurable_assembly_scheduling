# MO-ALNS v2 实施与验证记录

本次实现 `MO_ALNS_v2 / mo_alns_solver_budget_v3`。物理约束和公共目标采用现有项目环境；默认配置中的600秒是验证起点，最终正式预算由分层曲线确定。

执行约定更新：用户已取消本机长实验，后续在远端手动启动。包含新匹配PPO、NSGA-II-LS、MOEA/D、DCAN和DDQN的主比较使用[统一套件](comparator_remote_startup.md)；它会在同一远端重新校准ALNS。本文保留首轮本机机制证据和单独的历史固定PPO研究，不能把这些记录当作新套件的正式成绩。

## 实现

- `agent/mo_alns/budget.py` 统一管理时间、启动/完成/中断评价数、实际环境步数及预测调用数；初始化、解码和修复内部均检查截止时间。
- 原始确定性启发式直接运行一次，保存真实动作轨迹、目标及日志作为可交付初值；后续扰动重新评价。缓存哈希排除origin，保留动态资源模式及可重放轨迹身份。
- 每邻域先完成主修复解，再评价至多4个替代方案及1个matching repair；最多6次完整评价。插入预测上限24次，达到上限后合法补齐，避免用大量完整仿真做regret插入枚举。
- 插入选项按编码语义及预测的实际分配去重，regret-3使用不同插入代价之差的累加；真实重复轨迹仍计入计算成本。
- 轻量串行预测复用环境的安全阶段路径、疲劳恢复、DIS/INS时长和重构成本公式，并计算实际OWA或variance定义。真实解码时动态资源模式根据当前状态选择合法机器和工人。
- 退火从首轮启用，按完整邻域降温；每5轮更新权重，停滞后最多重热2次。所有已付费候选立即登记，matching repair前后的候选均保留。
- 单偏好端点只从本请求档案选择，合并档案单独输出；没有按时得到可行解时显式记录无交付。最终审计重放单列，不参与截止时选优。
- `analysis/flow_evaluation.py` 同时检查评价上限、时限、局部预算和自适应设置，避免将不同搜索预算当作相同对照。

## 测试和初次机制证据

定向目标、分析、预算与统计回归62项通过；之后完善的最新预算/统计测试15项通过；原有MO-ALNS测试6项通过，真实CLI入口3项通过。各集合存在重叠，不相加作为总测试数。XML位于`result/analysis/mo_alns_v2_*tests.xml`。

较广回归还发现10个既有配置的Flow模式与冻结归一化清单不一致，与本次修改无继承依赖；明细见`result/analysis/mo_alns_v2_preexisting_failures.json`。未将这些失败隐藏为通过。原先另一个“预算必须为整数”的旧测试已改为时间预算匹配检查并通过。

首轮真实机制验收：三例均为seed11、等权偏好、128次完整评价，结果见`result/analysis/mo_alns_v2_acceptance/mechanism.json`。

| 实例 | 完整邻域 | 权重更新 | 接受非改进候选 | 初始Q | 最终Q | 求解秒 |
|---|---:|---:|---:|---:|---:|---:|
| balanced_2000001 | 22 | 4 | 0 | 0.162534 | 0.162534 | 192.89 |
| fatigue_bottleneck_2000005 | 22 | 4 | 8 | 0.261147 | 0.254029 | 338.86 |
| reconfiguration_bottleneck_2000003 | 21 | 4 | 1 | 0.274688 | 0.270161 | 458.24 |

这证明完整搜索循环、权重更新及退火已实际执行。两例Q下降分别约2.73%与1.65%；不代表各目标同时改善，也不支持全验证集稳定优势。首轮验收之后进一步加入了“预测实际分配相同”的选项去重和报告完善；完整管线按最终源码身份重新执行机制门槛，避免混用不同版本证据。

单请求比较链路检查见`result/analysis/mo_alns_v2_compare_smoke/`：固定λ=.95 PPO模型、balanced实例、等权；PPO三次取优的三次重复计时使用同一三元组，实际交付中位耗时约15.85秒，Q=0.107865；30秒ALNS与固定概率LNS均完成6个邻域，Q=0.162534；PPO时限内ALNS完成2个邻域并交付同一初值。deadline overshoot约0.00008–0.0023秒。这是链路和超时行为检查，不是经预算选择的正式质量比较。

## 单独的历史固定PPO验证入口

如需继续这个独立研究，在远端项目根目录及已激活Python环境执行，使用新的远端输出目录：

```powershell
python -u -m scripts.mo_alns_study --stage all --output-dir result/analysis/mo_alns_v2_remote_v1 --workers 8
```

管线依次执行机制检查、21例×4偏好×3搜索种子的600秒曲线、必要时相同矩阵1800秒扩展、预算选择、7例×13偏好的独立确认和PPO比较，最后生成报告与正式批次清单。源码、模型、配置和数据哈希核对后，可重跑同一命令跳过已完成请求。时间限制在每个独立请求内生效；调参总时间包含所有已运行预算档位，另外记录。

PPO模型固定为`result/runs/schema12_screen_seed11_20261012_201659_owa_log2/best_checkpoint.pt`，SHA256为`9afc75e1cce6a210c49fec7b22d00d3e0591e4db6d31ed14480d2ea580d9b766`。保留历史选模身份，新增评价采用三次取优。充分预算ALNS、固定概率LNS和PPO短时限ALNS使用相同邻域定义，短时限只改变截止时间。

该单独脚本需要远端具有上述权重及同目录保存配置，并保留其原选模身份。与之不同，新`comparison_suite`从头训练三个PPO参照，只在`alns_reference`调用ALNS机制/曲线/预算接口，不依赖旧权重。两种研究的PPO训练重复和时限不可混用。

也可按阶段执行`--stage mechanism`、`curves`、`select`、`compare`、`report`、`formal`。曲线和预算未完整时，选择器拒绝发布预算；需要扩展时，比较入口要求先完成扩展。全部正式test/OOD/stress批次由生成的manifest单独启动。

主要产物：`identity.json`、`subsets.json`、`mechanism.json`、逐请求JSON、`budget_curve.csv`、`budget_selection.json`、原始PPO三元组与selected交付、`paired_deliveries.csv`、交付/付费档案HV、质量—时间SVG/PDF/PNG、`comparison_summary.json`和`report.md`。

确认集每个压力类别只有一个实例，无法估计组内bootstrap方差；此时置信区间采用整个实例为单位的pooled bootstrap，仍另报各压力类别结果。预算选择集有每组3例，采用分层实例bootstrap。任何区间和胜率都标为小规模验证证据。

## 判定与边界

平台条件、机制门槛及最长1800秒档位遵循`docs/experiment-spec.md`。最长预算仍有明显收益或机制轮数不足时，报告`budget_sufficiency=not_established`。确认集未支持稳定收益时，正式清单保留待确认状态。PPO短时限下只能交付初始解，是该时限下的结果，不能推广为充分搜索ALNS的整体强弱。
