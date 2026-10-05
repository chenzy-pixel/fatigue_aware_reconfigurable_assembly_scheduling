# Schema 9：实际资源关系

本页保留 schema 9 的关系构造和验收记录。当前生产观测采用
[schema 10 顺序投影](schema10_sequential_projection.md)，十四类关系和九维全局继续沿用。

Schema 9 保留六类节点、九维全局向量和 V8 策略头，将关系从十二种扩为十四种。
新增关系表达正在执行的实际资源分配，使审计确认的加工/工人服务状态对在原始输入
和 HGNN 编码中都能区分。

| 关系 | 生命周期 | 属性和方向 |
|---|---|---|
| operation → processing_on → machine | 直接加工或安装完成后创建；工序完成后移除 | E×0，双向 |
| machine → served_by → worker | DIS/INS 开始后创建；该阶段结束后移除 | E×0，双向 |

两种关系追加到 ASSEMBLY_EDGE_TYPES 末尾，从 environment 公共接口导出。
没有实际分配时，索引为 2×0，属性为 0×0；全部索引稳定排序。
capable_on 表示兼容性，locked_to 表示重构锁定，service_candidate 表示待分配候选。
DONE 工序及等待重构阶段不生成对应实际分配边。

## 一致性与表示

环境构造时要求每个 PROCESSING 工序/机器恰好配对；每个活动 DIS/INS 机器/工人
恰好配对。实际工人资格、模块支持、锁定工序、阶段和 busy_until_tick 必须一致。
观测 validate 核对零属性、双向、排序、索引范围、唯一性、状态覆盖及阶段/时长。
发现不一致时给出 processing_on/served_by 错误，不产生部分关联。

关系随现有状态缓存失效机制重建；复制、pickle、buffer 和并行 worker 保留完整关系。
网络将其加入两种图批处理路径和共享双向关系 Linear，宽度为 hidden_dim。
消息沿实际邻接选择资源；actor、WAIT 和 critic 通过共享图上下文使用这些关系。
node_mlp_pool 去掉图消息，继续保留其关联信息损失的消融定义。

在有效非终止状态下，每名工人的承诺负荷可恢复为：

\[
L_w^{commit}=L_w^{completed}+\sum_{m\to w}(d_m^{elapsed}+d_m^{remaining}).
\]

所有项使用一致时间单位，归一化时均除以 horizon。elapsed 来自 locked_to，
remaining 来自 machine 节点，配对来自 served_by。无需新增全局或工人统计特征。
物理事件时间格、类型及语义对象也能由未来释放信息和两种实际关联恢复。

工程截断保留原物理状态和实际关系，最终观测用于价值自举。真实失败执行原有部分
任务结算；终态关系按内核保留的实体状态构造，已结算账本不再视为未来计划，
终态不能继续 step。所有物理动作、奖励、mask 和目标结算函数沿用原实现。

## 模型与实验契约

观测版本集中定义为 9；runtime manifest、network spec、保存重载和配置快照使用同一
常量。network spec 保存十四种关系及属性宽度，两种实际关系必须为零宽。
schema 5/6/7/8 的 checkpoint 与配置快照明确拒绝，迁移开关不能绕过，新模型从头训练。
策略头仍为 V8，结果格式仍为 8.0.0，奖励协议仍为 single_stage_progress_quality_failure_v3。
持久化基准、manifest 和冻结目标尺度保持原文件。

修复目标为消除已发现的两类结构缺口。有限维池化没有全图注入性保证，float32 输入
也不能逐位恢复任意实数参数。连续恢复采用归一化 atol=rtol=1e−6，时间格和事件对象
要求精确一致；安全阈值两侧、ceil 工期边界和费用微扰保留回归测试。

## 验收记录

加工反例 full/neutral 的 WAIT 奖励差仍为 1/144；工人反例即时奖励仍相同、下一状态
仍不同。新增关系使原来的观测碰撞消失。五种子、两种模式、两种头的 HGNN 图上下文
双精度差均大于 1e−8，相关节点差大于 1e−6。单动作 softmax/初始 logits 不作为
充分性标准。证据保存在 `result/analysis/schema9_relation_verification/`。

行为基线由修改前 schema 8 保存，修改后剥离两种新关系再比较：
`scripts/verify_resource_relation_behavior.py capture/verify` 覆盖固定模板和七类验证实例，
分别用 full/neutral 与 heuristic/random11，共 **32 条轨迹、9,889 步**。
旧观测投影、mask、动作、所有奖励分量、累计目标和执行日志完全一致。
基线文件为 `result/audits/schema9_behavior_reference.json`。
黄金观测保留 schema8_initial_observation_sha256，新哈希反映新增关系。

生命周期、运输、恢复、编码、损坏输入、真实失败和工程截断回归见
`test/test_schema9_relations.py`；历史反例回放见 `test/test_state_sufficiency_audit.py`。
2026-10-05 最终验收：

- 全量运行 395 项：389 通过、3 跳过。另 3 项首次在在线缓存文件写入处受 Windows
  长路径限制失败；使用短临时目录复验后 3 项全部通过，共 **392 个有效用例通过**。
  记录为 `result/audits/schema9_full.xml` 与 `schema9_integration.xml`，保留初次失败记录。
- 最后对两种实际关系的非空混合批次补验了参考/优化前向及全部参数梯度，并要求
  新关系 Linear 的梯度非零；连同反例及黄金观测回归 **26 项通过，16.36 秒**。
  记录为 `result/audits/schema9_final_relations.xml`。
- 真实微型 PPO 训练完成更新、best/last checkpoint 保存和重载、13 点验证及 66 点
  最终评估；外部截断训练保留部分轨迹、跳过 best 选择，在线缓存与 worker 进度通过。
  可检查的配置、检查点、指标及 acceptance.json 保存于
  `result/audits/schema9_integration_evidence/`；其中模型仅为微型集成验收模型。
- schema 5/6/7/8 的模型/配置快照在迁移开关开/关时均拒绝；观测和模型采用 schema 9。
- `git diff --check` 通过。基准数据、manifest 及冻结尺度的原始哈希校验通过。

本次训练为集成冒烟；正式训练和多种子消融仍按实验协议另行执行。
