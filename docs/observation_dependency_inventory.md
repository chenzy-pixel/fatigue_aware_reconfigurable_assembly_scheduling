# 观测充分性：状态依赖与可恢复性清单

审计对象为实施修复前的 schema 8、六类节点、十二类关系、九维全局向量；固定实验配置内，
比较有效物理状态的奖励、合法动作及转移。下列“可恢复”按理想精度解释；实际 float32
误差和工程采样边界另列。拟新增关系仅在审计进程内构造，未改变生产环境。
本页保留原审计判定；实际加工/服务关系已在后续 schema 9 实施，见
[当前契约](schema9_resource_relations.md)。

## 动态依赖

| 依赖与读取位置 | 当前表达 | 结论及恢复方式 |
|---|---|---|
| current_tick、horizon_tick；时间推进、释放和期限 | global current_time_norm；horizon/resolution 在当前基准固定 | 可恢复：round(time_norm × horizon / resolution)，全部代表状态精确还原 tick |
| decision_type；生产/工人动作分支 | 独立字段与 production_decision | 直接可见 |
| 工序状态；READY、LOCKED、PROCESSING、DONE、BLOCKED、UNRELEASED | operation 状态 one-hot | 直接可见 |
| 工序—订单、工序前后继、工序要求模块 | belongs_to、precedes、requires；序号与模块 one-hot | 直接可见；序列长度由订单邻接恢复 |
| 正在加工工序所属机器；PROCESS_COMPLETE 对象 | 内核 operation.machine_id，图仅有兼容边 | **结构缺口 F1**：新增 processing_on |
| 机器当前模块、目标模块、占用类型 | machine one-hot；supports 边 installed/target 标志 | 直接可见；同源重复不是额外状态缺口 |
| 机器和工人 busy_until_tick；下一完成事件 | 各自 remaining_busy_time_norm | 固定时间格下可恢复：current_tick + round(remaining_norm × horizon / resolution) |
| 重构锁定工序、source/target module、当前阶段 | locked_to 拓扑、source/target/stage one-hot | 直接可见；保留用于 WAIT_DIS/DIS/WAIT_INS/INS |
| 当前重构阶段开始时刻 | locked_to.stage_elapsed_time_norm | 可恢复：current_tick − round(elapsed_norm × horizon / resolution) |
| 活动 DIS/INS 阶段的实际工人 | 内核 disassembly_worker_id / installation_worker_id；候选服务边只覆盖等待阶段 | **结构缺口 F2**：新增 served_by，候选边无法替代实际服务关系 |
| 活动阶段原计划时长 | 内核 start/end ticks；对应机器 elapsed+remaining 已可见 | 补齐 served_by 后可恢复，duration_ticks=elapsed_ticks+remaining_ticks |
| 工人 current fatigue | worker.fatigue_ratio；活动任务期间保持开始时疲劳，结束时才累加 | 可见到 float32 精度；补齐阶段时长和服务关系后，可恢复预计释放疲劳 |
| 工人已完成 load | worker.load_norm | 直接可见到 float32 精度 |
| 每名工人 committed load | 内核 _committed_worker_loads；全局只有方差 | 补齐 served_by 后可恢复：completed_load + active_stage_planned_duration；当前每人最多一个活动任务 |
| _active_committed_worker_tasks 账本 | 内核任务 ID → 工人/计划时长 | 由 locked_to + served_by + 阶段时长恢复；任务 ID 是标签，无需输入网络 |
| 工人资格、当前可服务任务、安全性 | worker-module 拓扑及 one-hot，状态、疲劳、阶段参数，service_candidate 和 mask | 原始事实可见；有 F2 时未来释放后的事实不充分，补齐实际关系后恢复 |
| 每个订单是否释放/完成 | order.released/completed | 直接可见 |
| 订单释放时间 | 每道工序的 order_release_time_norm | 通过 belongs_to 取该订单任一道工序，恢复未来 ORDER_RELEASE 事件 |
| _order_completion_tick | 仅 completed 标志可见，实际历史完成时刻未暴露 | 不再影响未来转移；历史 Flow 已进入累计目标，完成时刻用于日志/终局指标 |
| 活跃订单数；Flow 增量 | order released/completed | sum(released−completed)；积分增量=active_count × physical_time_delta |
| 每个订单完成工序数和总工序数；progress | 工序 DONE、belongs_to；order.completion_ratio | P=mean(order.completion_ratio)；进度权重 1/(订单数×该订单工序数) |
| _flow_integral；非线性奖励历史 | global objective_flow_norm | 乘冻结尺度恢复到观测精度；有效非终止状态下 _flow_penalty=0 |
| _reconfiguration_cost；历史固定费/人工/停机费 | global objective_cost_norm | 乘冻结尺度恢复到观测精度，不能用当前资源状态重算历史成本 |
| _committed_load_variance；非线性奖励 | global objective_committed_variance_norm | 当前值直接可见；后续增量依赖个体负荷，按上述公式恢复 |
| 当前活动重构的人工/停机成本增量 | worker.labor_cost_rate_norm、machine.downtime_cost_rate_norm 与状态 | 费率=norm × cost_scale/horizon；按忙碌阶段积分 |
| 真实失败的部分阶段结算 | start/end、服务工人、完成负荷、承诺账本 | 补齐 served_by 后可恢复：worked_ticks=min(current,end)−start，回退未执行承诺时长并累加实际疲劳/负荷 |
| 实际失败时未完成订单罚项 | order.completed、实例 unfinished_order_penalty | unfinished_count × 固定罚项；与工程截断分开 |
| _failure_penalty_applied | 内部一次性开关 | 标准有效非终止状态恒 false；进入真实终止一次施加，不能继续 step |
| _events 的物理内容 | 内部堆，部分下一事件上下文可见 | 两种实际关系补齐后：未来释放+实际加工结束+实际 DIS/INS 结束，能恢复时间、种类和对象 |
| 同时事件的 priority 与 serial | 不进入网络 | priority 固定；不同资源的完成及释放无在线分配选择，批次末统一刷新 READY；serial 重排合法测试结果相同 |
| _decision_count、_zero_time_actions | 工程保护内部变量 | 外部采样控制；不影响物理奖励，不属于此次九维策略状态；若要求完整 step 五元组的截断位相同，必须额外固定这些计数 |
| _initial_objectives、_initial_progress、initial quality | reset 时保存 | 正常从头启动固定为零；用于累计回报核算，不是额外未来状态 |

## 静态参数及实验常量

| 参数 | 观测或实验来源 | 判定 |
|---|---|---|
| 工序加工基时 | operation.base_processing_time_norm | 乘固定 horizon 恢复 |
| 机器—模块加工速度、拆卸/安装基时 | supports 关系的 processing_speed_factor、disassembly_base_time_norm、installation_base_time_norm | 直接表达所有可用模块参数 |
| 模块固定拆卸/安装费 | module 节点两项成本 | 乘冻结 cost_scale 恢复 |
| 机器停机费、工人人工费 | 相应节点 rate_norm | 费率均直接表达 |
| 机器模块支持、工人资格 | supports/qualified_for 与 one-hot | 拓扑可恢复；候选关系重复表达并提供消息路径 |
| horizon、resolution、六项疲劳动力学常量、未完成订单罚项 | 固定模板和所有持久化数据 | 560 个实际数据记录中只有一个参数组；full/neutral 分开检查，neutral 在 reset 中固定为零疲劳动力学 |
| 偏好、标量器类别/rho/尺度、失败罚系数 | preference 字段与固定运行配置 | 充分性判断以配置相同为前提；不同实验配置不要求产生同一奖励 |
| 实例规模 | 各节点数组长度、邻接结构 | 原始观测可见，数据范围为 8 机器/6 工人、12–18 订单、39–87 工序 |
| 波次时间窗和剩余需求 | wave 和 wave-module、订单释放时间 | 观测已表达；实际释放以逐订单时间为准 |

## 诊断、缓存与可排除的历史

操作历史日志、peak fatigue、maximum_fatigue_seen、动作计数、等待计数、资源压力计数、
重构复用计数、post_reconfiguration_process_count、已完成重构的 ID 和历史机器分配均
只服务诊断/日志。它们不改变当前物理奖励、合法动作或未来资源演化。
本结论针对当前已禁用的 feasibility_shaping 配置；重新启用奖励分支应重新审计依赖。

各 cache、state_version、预计算 capability arrays 和资源投影缓存属于派生实现状态。
恢复后的语义状态应重新生成并清空缓存，不能将缓存值当作可观测性证明的一部分。

## 编码层清单

| 路径 | 当前情况 | 处理结论 |
|---|---|---|
| attributed relation → Linear([neighbor,edge]) → sum/mean | 边属性与邻居特征在聚合前没有非线性交互 | 结构上仅保留 incident edge attribute sums；已有加工标志反例证明简单追加标志无法保留实际配对 |
| processing_on / served_by 动态邻接 | 拟新增，邻接直接选择实际资源，边无属性 | 已用原型验证两种状态对的 HGNN 表示可区分；不要求所有不同状态都编码成不同向量 |
| 两层消息传递 | 当前生产关系与 order、locked、supports 路径参与计算 | 实际机器信息可到工序再到订单；阶段信息可到机器再到实际工人 |
| 图上下文的类型均值池化 | 有限维、非注入映射，理论上可合并不同图 | 作为表达能力限制记录；当前审计未找到新的合法、任务相关池化反例，不据此盲目扩充统计量 |
| critic | 使用六类节点池化、global、preference；不直接使用 action_set_features | 派生 WAIT 上下文不能代替 critic 缺失的实际资源关联；新增关系通过共享编码器进入 critic |
| production actor | 合法候选的 operation/machine、capable edge、global、graph context | 直连候选信息可区分当前候选；正在执行的工序不会作为合法候选进入动作头，必须通过图关系传递 |
| worker actor | machine/worker、锁定工序嵌入、service_candidate、graph context | 候选服务边与实际服务边分别承担分配选择和活动任务状态的角色 |
| WAIT actor | action_set 与 graph context | 三目标专家 context 末层零初始化，初始输出接近不能证明路径无效；审计分别报告 latent 和输出 |
| shared_preference actor | 同一共享图编码器，普通共享动作头 | 纳入相同关联探针，能看到拟新增关系造成的差异 |
| node_mlp_pool | 主动移除图消息；critic 不看边 | 两种反例均无法区分，是该消融的预期信息损失，结果不能代表主 HGNN 的状态充分性 |

## 精度与结论条件

“直接表达”仍受 float32 压缩约束；更高精度内核的所有实数状态不可能从这些数组逐位恢复。
合法参数微扰 1e-8 的人工费反例给出约 4.82e-13 的标量奖励差，并在补齐关系后仍成立。
所以修复目标是明确关联、精确时间格与受控数值误差，不能承诺 float32 输入下跨不同实数
算例的奖励逐位相等。阈值/量化边界可能放大微小误差，需列入专门测试。

依赖推导支持：在固定配置、固定动力学参数、有效非终止状态、理想精度下，补齐两种实际
关联后，清单内影响奖励/动作/事件演化的物理变量均有表达或恢复路径。
这不等价于有限轨迹覆盖证明，也不等价于任意初始化或训练后的有限维编码器具有注入性。
