# HGNN 联合消息实施记录 · 2026-10-06

改动基线为 `43add12`。有属性关系采用 `ReLU(Linear([neighbor, edge]))` 后再聚合；双向关系的两个方向执行同一规则。零属性消息、总度数 mean、残差、LayerNorm、节点更新及按类型 mean 图读出维持当前定义。

计算身份由 `configs/network_contract.py` 统一生成：

| Encoder | message_function | message_aggregation |
|---|---|---|
| hetero_gnn | attributed_joint_relu_v1 | total_degree_mean_v1 |
| node_mlp_pool | not_applicable | not_applicable |

身份是实现生成的记录，不提供运行时结构选择。它进入 normalized network config、network spec 和 runtime manifest；外部传入值必须与当前 encoder 的身份一致。checkpoint 中两项均为必需字段。

图消融的消息身份允许随 encoder_variant 派生变化，完整/neutral 配对和 actor-head 消融仍要求一致。原始 observation schema 为 10，策略头为 V8，参数名、张量形状、参数量与此前相同。固定 hidden_dim=128 的默认模型仍为 1,197,796 个参数。

## Checkpoint 与配置

旧 schema-10 模型若没有新身份，或身份不匹配，加载在修改网络/optimizer 之前拒绝。`allow_observation_migration` 不绕过检查；旧配置快照由 runtime manifest 比较识别。新模型支持正常网络及 optimizer 往返加载。旧线性消息模型需要重新训练，权重形状相同不代表执行语义相同。

本次是共享编码器计算语义的更新，不改变数据、物理调度或奖励。新结构的正式训练、调度质量和多种子性能结论需由后续实验建立。

## 验收命令与产物

```powershell
.\.venv\Scripts\python.exe -m pytest -q test/test_joint_messages.py test/test_structural_ablations.py test/test_hetero_gnn.py test/test_ablation_protocol.py --basetemp=.pytest_tmp/joint_target --junitxml=result/audits/joint_messages_20261006/targeted.xml
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp/joint_all --junitxml=result/audits/joint_messages_20261006/full.xml
.\.venv\Scripts\python.exe scripts/benchmark_joint_messages.py
```

`test_joint_messages.py` 覆盖确定权重下的联合 ReLU、零属性负消息、双向计算、不同关系共用总度数、空关系、节点重排、合法资源配对反例、计算身份和 checkpoint 失败原子性。既有结构测试覆盖变长/阶段混合 batch、CPU/CUDA 输出与梯度以及独立 critic 查询；既有真实入口测试覆盖微型 PPO 更新、验证、checkpoint 重载与最终偏好网格。

`scripts/benchmark_joint_messages.py` 在隔离的旧线性参考和当前结构上使用完全相同的参数，不增加生产结构选择。batch=20 的状态来自固定、validation 和 stress 实例的合法启发式轨迹；两层、hidden_dim=128、CPU 四线程，GPU 在计时边界同步。三轮预热、九轮计时，方法顺序交替，报告中位数。时间包含图打包及网络计算，研究损失的前向/反向不含 optimizer 或完整 PPO 更新。

性能原始数据为 `result/audits/joint_messages_20261006/performance.json`；旧参考只用于表达与性能对照，不保存为生产 checkpoint。超过原实现 15% 的组件开销需单独解释，不能外推为整体训练耗时。

## 验证结果

- 首轮针对性回归：**109 passed，56.27 秒**。五个初始化种子的 float64 反例全部可区分；隔离旧线性参考仍在 1e−12 以内碰撞。默认完整模型参数仍为 1,197,796，全部 state-dict 键与权重张量逐项一致。
- CPU/CUDA 结构执行对照包含输出、梯度、变长及阶段混合 batch、独立 critic 查询；checkpoint 身份缺失/错误在两个迁移开关设置下均拒绝，目标网络及已初始化的 Adam 状态保持逐项一致。
- 真实微型入口：2 个训练 episode、1 次 PPO 更新，best/last 文件存在；保存的配置和 checkpoint 记录新消息身份。13 条验证和 66 条最终网格评估记录均零调度违规，测试另外重载 best 的网络权重，以及 last 的网络和 optimizer 状态。证据为审计目录 `integration.json`，原始运行目录记录在该文件中。
- 完整常规回归：**545 passed、2 skipped、1 failed，677.60 秒**。唯一失败为既有 `test_legacy_data_is_preserved_and_loadable`：四个 v1 manifest 的本地 LF 字节与冻结 CRLF 哈希不同，仅在内存中转为 CRLF 后均准确匹配。没有新增回归失败；两个 slow 审计按既有规则跳过。固定原始文件保留，未以重写数据消除失败。
- 最终验收索引为 `result/audits/joint_messages_20261006/acceptance.json`，保存两组 JUnit 计数、源码指纹、运行身份、微型链路、性能阈值及历史换行问题的独立核查。

本机组件计时：

| 设备 / 组件 | 旧线性参考 ms | 当前联合 ReLU ms | 当前/参考 |
|---|---:|---:|---:|
| CPU critic | 18.32 | 15.85 | 0.865 |
| CPU actor+critic 前向/反向 | 61.42 | 57.02 | 0.928 |
| RTX 4060 Laptop critic | 11.63 | 12.32 | 1.060 |
| RTX 4060 Laptop actor+critic 前向/反向 | 49.87 | 51.49 | 1.032 |

全部组件均未超过 15% 开销阈值。GPU critic 中位数约增加 6.0%，前向/反向约增加 3.2%。CPU 结果存在时序和频率噪声，不能据此宣称结构改动带来整体加速；这些组件计时也不是正式训练吞吐。
