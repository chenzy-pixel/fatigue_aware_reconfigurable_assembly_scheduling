# Schema 9 修改后复核

后续已按本次范围实施前两项及关联费用/方差投影，见
[schema 10 方法说明](schema10_sequential_projection.md)。本页数值和发现保留审计时的版本身份。

日期：2026-10-05。复核范围为观测构造、候选投影、图消息与 critic、并行采样结束处理、
单目标分析和正式评估入口。本轮新增复核脚本和报告，未修改环境、网络或训练生产逻辑。

确认五项问题，其中前三项与状态信息的表达或利用直接相关。新增 processing_on、
served_by 已补齐先前两处实际关联；本轮没有找到新的同实例、相同完整原始观测和 mask、
同一动作产生显著不同奖励的反例。下面的编码碰撞发生在原始边属性不同的图之间。

## 1. 忙碌机器的候选时间没有遵守已有承诺（P2）

位置：`environment/env.py:3524` 的 `_compute_production_resource_profile()`，以及
`environment/env.py:1157` 的 predicted_finish 构造。

当机器当前模块与目标模块相同，resource profile 直接返回 current_tick，未纳入
busy_until_tick。另一路 earliest_start 已纳入机器占用，导致同一边上的两个时间矛盾。

未经修改的标准固定实例，只执行合法动作 `[0, 480]`，在 t=3 得到：

| READY 工序 O_J2_1 → M1 的信息 | 分钟 |
|---|---:|
| 当前加工剩余占用对应的释放时刻 | 9.9 |
| 该候选加工时长 | 9.0 |
| earliest_start_time | 9.9 |
| predicted_finish_time | **12.0** |
| 物理最早完成时刻 | **18.9** |

该候选此刻被 mask，不能非法执行；但全部兼容边仍参与共享 HGNN 编码，错误值仍可
影响其他动作上下文和 critic。masked action 的安全性不能证明其边特征无影响。

活动重构也有相似问题：保留标准机器参数及工人资格的合法服务反例，在 t=36 时 M6
的安装确定于 t=38 结束，A_1 随后加工到 t=46.4；其 capable_on 却给出 240.1 的
“无可行投影”值。原因是 A0 阶段仍按新的 A0 拆卸尝试投影，而没有采用实际锁定任务。

建议统一物理时间入口，明确区分当前实际工序、当前锁定工序和未来候选：
实际任务采用已承诺的结束时刻，其他候选从机器完成全部承诺后的时刻/模块开始估计。
验收需覆盖 READY→忙碌机器、LOCKED→实际机器，以及 WAIT_DIS/DIS/WAIT_INS/INS。

## 2. 拆卸和安装投影没有串接同一工人的疲劳（P2）

位置：`environment/env.py:3549`、`:3573` 的两阶段 resource profile；
同类读取见 `environment/env.py:2284` 的候选负荷方差估计。

安装投影仅设置 earliest_tick=disassembly_end，仍从原环境读取工人状态；若该工人
也是假设的拆卸工人，它会被当作此前一直空闲并恢复疲劳，没有累加刚执行的拆卸。

合法构造实例保留标准机器参数和工人资格，仅调整订单释放和初始疲劳：H5=0.5，
其他工人=0.74。M6 从 A3 改装到 A2，执行合法动作 `[5, 288, 34, 48]` 后：

| 比较 | 结果 |
|---|---:|
| 原 profile 预计开始加工 | **8.5 分钟** |
| H5 实际拆卸完成 | 4.3 分钟 |
| H5 实际拆卸后疲劳 | 0.6505 |
| 此刻立即安全安装的工人 | **0 人** |
| 拆卸后最早安全安装完成 | **13.6 分钟** |
| 已有 OrderTimeEstimator 串接投影 | 13.6 分钟 |

真实 mask 会阻止不安全服务；问题是合法生产候选的预计完成时间偏乐观，且两种
观测时间估计互相矛盾。初始疲劳值是合法构造值，本轮不声称该精确参数组合来自
已持久化的数据集，也不据此推断实际训练收益损失。

建议复用带临时工人状态的顺序投影，枚举拆卸工人时保存其阶段后疲劳、负荷和
可用时刻，再估计安装。方差增量应使用同一串行服务路径。本轮数值反例直接验证
的是时间/安全投影；方差估计的同类问题由代码读取路径确认，尚未单独量化输出误差。

## 3. 线性消息仍会丢失边属性与邻居的配对（P2，模型表达边界）

位置：`agent/ppo/network_v8.py:405`、`:413` 的消息函数；critic 入口 `:777`。

当前消息为 W_h h_neighbor + W_e e + b，随后聚合。若对若干边做属性交换，保持
每个端点的属性和不变，所有节点嵌入在精确算术下仍相同。新实际分配关系解决了
拓扑关联，但不自动解决 capable_on、supports 等有属性关系中的配对丢失。

两份通过 validate_instance() 的构造实例：节点、global、偏好、动作上下文和拓扑
完全相同；只在 M1/M4 的 A1/A2 参数中交叉交换加工速度。capable_on 和 supports
的每个端点属性和都完全一致。实例包含两道 150 分钟工序，部分重构基时为 15 分钟，
这些值超出当前常用生成配置，因此该反例证明网络结构限制，不证明当前基准内的
实际碰撞频率。

五种子 0/1/11/23/37 的结果：

| 检查 | 结果 |
|---|---|
| 原始完整观测 | 边属性不同 |
| 每个端点的两类边属性和 | 差值 0 |
| 图上下文差，float64 | 4.44e−16～8.88e−16 |
| critic 输出差，float64 | 0～5.55e−17 |
| 固定策略先 A→M1、B→M3，再 WAIT 至完成 | Flow 270 / 300 |
| 对应 Flow 偏好采样回报 | 0.81015454 / 0.79341822 |

这是同一个固定派发规则下不同回报的例子；两条实际 WAIT 次数因结束事件不同而不同。
它不证明两状态最优价值不同。actor 还直接读取候选边，可能区分此刻动作；critic
只读取图/global/偏好，无法利用已被聚合消掉的配对。增加层数也不能恢复第一层前
已经按端点求和抹掉的信息。

建议对有属性关系采用聚合前的非线性邻居—边联合消息，如小型 MLP([h_neighbor,e])，
或由 e 调制邻居消息的门控。先针对这类反例验证节点表示及 critic 可区分，再做
匹配消融。仍不能将有限维消息/池化宣称为对任意图的注入编码。

## 4. runner 的 step_limit 漏记采样截断（P2）

位置：`agent/ppo/parallel.py:1392`～`:1453`，reset 强制动作分支 `:1191`；
实际入口 `train.py:678` 的 smoke_rollout_steps。

直接运行并行采样 collect_training_batch(step_limit=2)：两步后结束该段并重置实例，
buffer 末项 done=False，学习路径进入价值自举；但返回的指标为：

```
terminated=False, truncated=False, task_done=False
task_succeeded=False, task_failed=False, sampling_truncated=False
objective_complete=False, terminal_reason=None
```

因此一次已停止的采样没有归入成功、失败或外部截断。当前正式训练未传 step_limit，
默认受影响入口是 smoke，直接使用 runner 的用户也可触发；环境本身两个工程保护
路径不属于此问题。不能据此声称正式训练的工程截断自举失效。

建议统一 runner 截断与环境截断的结果契约，返回 sampling_truncated=True、明确
rollout_step_limit 原因和实际步数；保留原物理末状态，done=False 并自举。
reset 的纯强制动作段也应记录截断状态，但不制造策略样本。

## 5. 单目标绘图仍假设每次验证完整（P2）

位置：`analysis/single_objective_analysis.py:52` 的字段读取及 `:177` 的绘图。

真实环境 max_decisions=1 的截断指标，加上当前验证格式中的
evaluation_complete=False / completion_rate 空值，可复现：

```
ValueError: could not convert string to float: ''
```

analyze_run 的验证摘要已跳过不完整质量值，但 plot_run 仍直接 float(空字符串)。
load_plot_rows 也丢掉 sampling_truncated、objective_complete，部分进度被标成 terminal
progress，部分目标与完整目标画在同一序列中而无法区分。

建议保留明确的状态字段；绘图为不完整验证留空点并标注覆盖率/截断数量，分开表示
部分采样目标与完整调度目标。实际采样回报仍可显示。本轮检查的正式 Pareto/HV 和
匹配消融入口已有不完整评估拒绝逻辑，未复现截断样本混入正式比较的问题。

## 实施顺序与边界

建议先统一两类候选物理投影，再修复 runner 和分析端的截断处理；图消息非线性交互
作为单独的模型修改验证。保持九维全局向量即可，这五项问题均不需要增加全局统计量。
边特征计算口径改变应更新观测语义契约；消息架构改变需更新网络规格和检查点边界，
重新训练后才能评估效果。

本轮运行的是针对性复核，没有重新宣称全仓回归通过或证明完整 Markov 充分性。
合法路径、构造实例、五种子编码输出、真实 runner 结果和绘图异常都已保存。

复现命令：

```powershell
.\.venv\Scripts\python.exe scripts/audit_schema9_followup.py
```

脚本：[audit_schema9_followup.py](../scripts/audit_schema9_followup.py)。
证据：[findings.json](../result/analysis/schema9_followup/findings.json)，同目录保存两个边属性
实例、单目标绘图输入以及生产源文件 SHA-256。输出属于局部构造审计，不是性能实验。
