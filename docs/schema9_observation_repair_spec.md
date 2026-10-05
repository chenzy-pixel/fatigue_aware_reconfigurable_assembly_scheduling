# Schema 9：实际资源关联统一修改规格

本规格已实施为生产 schema 9：保持九维全局向量、奖励协议与工程采样截断语义，
补齐已确认的实际加工和工人服务关联。方法与验收记录见
[schema 9 实际资源关系](schema9_resource_relations.md)。

## 最终观测接口

六类节点、原有节点特征和九维全局向量保持现有定义。关系由 12 种扩为 **14 种**：

| 常量 | EdgeType | 边属性/维度 | 方向及含义 |
|---|---|---|---|
| PROCESSING_ON_EDGE | (operation, processing_on, machine) | 空，E×0 | 双向，PROCESSING 工序所属实际机器 |
| SERVED_BY_EDGE | (machine, served_by, worker) | 空，E×0 | 双向，DIS/INS 机器当前使用的实际工人 |

将两个常量追加到 ASSEMBLY_EDGE_TYPES 元组末尾，并从 environment 公共接口导出。
原有 capable_on 保留兼容/候选语义，locked_to 保留重构锁定语义，service_candidate
保留待分配候选语义。全部新增边按源、目标索引排序；没有活动关系时，索引为 2×0，
属性为 0×0。不增加常数属性或用数值型机器/工人 ID 作输入。

processing_on 由 PROCESSING 工序的 machine_id 构造；不包含 READY、LOCKED、DONE。
served_by 只读取当前 DIS 的 disassembly_worker_id 或 INS 的 installation_worker_id；
WAIT_DIS、WAIT_INS、DONE 均不生成边。拆卸到等待安装时移除服务边；安装完成时移除
服务边和重构锁定边，并生成该工序的加工边；加工完成时移除加工边。
工程截断保留实际关系，真实失败后的关系由结算后的实际状态决定，不按日志裁短标志推断。

## 数据完整性与网络

环境构造阶段校验：每个 PROCESSING 工序恰有一条加工边，每台 PROCESSING 机器
恰对应一个工序；每个忙碌 DIS/INS 工人和机器恰有一条服务边。两端状态、模块支持、
阶段、busy_until_tick 必须一致。发现违反不变量时抛出清晰错误，不生成部分关联。

原始观测 validate 校验名称、关系集合、E×0 维度、双向标志、排序、索引范围、唯一性、
与节点状态一致性。工人实例无需新增 committed_load_norm 或其他统计列：

\[
L_w^{commit}=L_w^{completed}+\sum_{m\to w}\frac{d_m^{elapsed}+d_m^{remaining}}{horizon}.
\]

式中负荷均按 horizon 归一化，elapsed 来自该机器对应 locked_to 的当前阶段字段。
当前每名工人最多执行一个任务。worker node 的 load_norm 继续表示已完成负荷。

共享 HGNN 将两种关系加入双向批处理和关系 Linear，输入宽度为 hidden_dim+0；
沿用当前每层的聚合、残差和 LayerNorm。实际邻接负责选择邻居，无需在兼容边上另做
标志门控，也不在本次改动中普遍替换其余关系的消息函数。
默认两层下，机器信息传到实际加工工序；locked 阶段信息传到机器再传到实际工人。
actor、WAIT 和 critic 均通过现有共享图上下文获得这些信息。
node_mlp_pool 继续忽略图消息，作为既有消融如实记录其关联信息损失。

新关系在每次物理状态改变后重建，使用现有统一缓存失效入口；不建立独立的持久缓存。
更新观察复制、pickle、buffer、并行 worker、参考/优化两种图打包路径和网络规格校验。
不调整动作编号、合法 mask 或任何物理执行函数。

## 版本与兼容

- observation schema 从 8 升至 **9**，集中引用版本常量，移除当前加载路径的硬编码 8。
- schema 5/6/7/8 模型及配置快照在当前入口明确拒绝；保留的迁移 CLI 开关不能绕过。
  当前训练从头初始化，不提供权重/优化器迁移。
- network spec 写入完整 14 种关系、每项属性维度和 observation schema 9。
  策略头仍为 V8，奖励/实验协议仍为 single_stage_progress_quality_failure_v3，
  结果格式仍为 8.0.0，因为本次仅改变观测结构。
- 基准数据、manifest 和冻结尺度文件保持原始内容；黄金观测更新只针对新增关系，
  保存 schema-8 原始哈希，原动作/reward/mask 基线必须仍一致。
- 更新当前架构文档、方法说明与审计脚本：能回放保存的 schema-8 反例并核对 schema-9
  输入已区分；历史证据文件保留其明确的版本身份。

## 验收用例与数值口径

1. **两个反例、两种模式：** 加工反例的 WAIT 奖励仍相差 1/144，新增加工边使输入不同；
   工人反例即时奖励相同、下一 worker 状态不同，新增服务边使输入不同。
2. **编码：** 默认两层 128 维，种子 0/1/11/23/37；objective_experts/shared_preference
   均要求图上下文双精度最大差 >1e-8，至少一个相关节点差 >1e-6。
   不要求初始化 actor 输出必须不同，单一合法动作的 softmax 本来无需不同。
3. **生命周期：** reset、直接加工、重构锁定、开始/结束 DIS、等待 INS、开始/结束 INS、
   加工完成、同时释放与完成、部分阶段真实失败、工程截断均核对关系与实体状态。
4. **恢复：** 原始观测加固定实验常量能恢复时间格、所有未来物理事件时间/种类/对象、
   活动阶段计划时长和个体承诺负荷。tick、事件对象与离散状态要求精确一致；
   归一化连续值用 atol=1e-6、rtol=1e-6，阶段时长以四舍五入恢复整数 tick。
   内核/固定动作轨迹的奖励回归仍采用原 1e-8 恒等式标准。
5. **观测精度：** 保留 float32 特征和高精度内核；费用微扰探针按舍入限制记录，
   不用逐位奖励相等作跨实数参数的验收条件。安全阈值两侧、ceil 工期边界各有专测。
6. **管线：** batch/reference 数值及梯度对比、buffer 与 worker 完整运输、critic 自举、
   模型保存重载与旧版本拒绝；实际微型 PPO 更新、13 点验证、66 点评估及全量回归。
7. **行为：** 固定动作回放的调度、mask、目标、奖励与 schema 8 相同；剥离新增两种关系
   后原始节点/边/全局/动作特征逐项相同。

交付应将依赖清单、反例与恢复测试一起保留。可声称已修复这两类结构缺口；不得将五个
初始化种子、有限轨迹通过或原型表示可区分解释为全状态空间证明或训练性能改善。
