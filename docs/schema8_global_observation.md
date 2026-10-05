# Schema 8：9 维全局观测与外部采样截断

本页保留 schema 8 的实现与验收记录。当前生产观测已升级为
[schema 9](schema9_resource_relations.md)，九维全局和外部采样截断定义继续沿用。

当前观测由六类节点、十二类关系、候选动作上下文、三维偏好及九维全局向量构成。
全局顺序集中定义于 `environment/observation_schema.py`：

| 索引 | 名称 | 口径 |
|---:|---|---|
| 0 | current_time_norm | 当前时间 / horizon |
| 1 | pending_reconfiguration_ratio | WAIT_DIS、WAIT_INS 任务数 / 机器数 |
| 2 | completed_operation_ratio | DONE 工序数 / 总工序数 |
| 3 | production_decision | 生产阶段 1，工人阶段 0 |
| 4 | worker_matching_deficit_norm | (任务数 − 最大安全匹配数) / 任务数 |
| 5 | minimum_worker_alternative_ratio | 最小安全备选工人数 / 工人数 |
| 6 | objective_flow_norm | 奖励使用的当前 Flow / 冻结尺度 |
| 7 | objective_cost_norm | 奖励使用的当前成本 / 冻结尺度 |
| 8 | objective_committed_variance_norm | 奖励使用的承诺负荷方差 / 冻结尺度 |

计数分母至少为 1；无待分配任务时匹配缺口为 0、最小备选比例为 1。
累计三目标采用线性尺度归一化，直接读取奖励目标向量。
order 的 released、completed 继续表达每个订单是否活跃，其差值均值给出全局比例；
WAIT 与 wave 保留各自的统计上下文。

## 结束与训练接口

| 原因 | terminated | truncated | task_succeeded | task_failed |
|---|---|---|---|---|
| 全部工序完成 | true | false | true | false |
| horizon / 不可恢复死锁 | true | false | false | true |
| decision_limit / zero_time_action_limit | false | true | false | false |

动作执行后先判断真实终止，再检查工程保护。两个保护同时满足时使用 decision_limit。
工程截断保留当前物理状态、承诺任务、决策阶段及合法 mask，不结算失败罚项或部分任务。
返回 sampling_truncated=true、objective_complete=false，以及两项计数。
环境结束后禁止继续 step；工程截断仍允许 observe/get_action_mask。

PPO 真实终止转移的 done=true、末状态价值为零；工程截断 done=false，
以 V(final_observation) 自举。每段独立计算 GAE 后合并。强制动作段没有策略转移时
只记录实际采样回报与指标。training.episodes 表示采样启动次数，截断占一次；
后续启动新实例。日志同时记录成功、真实失败、完整终止和采样截断计数。
算例生成与反事实预检查使用按工序数、资源数和时间 tick 数推导的保守保护上限，
接受条件读取 task_succeeded，从而保持物理可行性预检查独立于训练采样截止点。
stress 算例的失败比例限制继续使用既有配置键 max_truncated_fraction；计数读取
heuristic_failed，历史算例则由 heuristic_completed 恢复任务结果。

## 结果与模型

结果 schema 为 8.0.0，奖励/实验协议为 single_stage_progress_quality_failure_v3；
策略头仍为 V8。checkpoint、网络规格与配置快照要求 observation schema 8。
旧 schema 5/6/7 必须重新训练，迁移开关不能覆盖此要求。

外部截断行的目标值属于部分轨迹，objective_complete=false。评估保留全部请求行，
汇总 evaluation_complete=false、completion_coverage 与 sampling_truncated_count；
正式成功率、质量汇总为空，不参与选模及 Pareto/HV。bootstrap 仅进入价值学习目标，
不写入实际采样奖励。

## 验证边界

独立回归比较固定实例及七类验证实例，各使用 heuristic/random 两种策略。
修改前后 16 条轨迹、5,099 步的节点、边、动作上下文、合法 mask、动作、九维投影、
奖励和终局目标完全一致。该验证证明未触发工程保护时的行为保持，
不替代正式训练消融或整个图观测的 Markov 充分性证明。

2026-10-05 的[奖励充分性审计](observation_reward_audit_20261005.md)发现实际加工关联缺口：
两条合法历史的完整观测相同，执行同一个 WAIT 的进度奖励不同。该问题需要补充局部
实际加工映射；九维全局向量的契约和此前行为回归通过，不等于观测具有严格充分性。

后续[全面充分性审计](observation_sufficiency_audit.md)另确认活动工人服务关联缺口，
完成 162 条代表轨迹的事件/承诺负荷恢复核对，并给出
[Schema 9 统一实施规格](schema9_observation_repair_spec.md)。当前生产观测仍为 schema 8。

验收记录（2026-10-05）：

- 全量回归：365 passed、3 skipped，402.96 秒；记录为
  `result/audits/schema8_verified.xml`。跳过项沿用项目的慢速/历史来源检查条件。
- 集成回归覆盖 PPO 更新、best/last checkpoint 重载、13 点验证及 66×3 点 CLI 评估；
  32 项通过，记录为 `result/audits/schema8_integration.xml`。
- 工程保护训练实测：2 次启动均外部截断，13 个验证单元完整保存，last checkpoint
  保存成功，best checkpoint 选择跳过；记录为 `result/audits/schema8_truncated_training.xml`。
- `git diff --check` 通过；固定算例与 manifest 文件保留其原始内容。
