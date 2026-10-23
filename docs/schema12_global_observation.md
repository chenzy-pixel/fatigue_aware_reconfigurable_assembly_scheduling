> 历史 schema12 说明。2026-10-23 起的默认主线采用[三因子批次调度协议](factorial27_migration.md)；本文以下内容保留用于解释历史实现和产物。

# Schema12：订单完工裕量与累计目标输入编码

schema12 使用十维全局观测。前六维结构特征沿用原顺序，第6–8索引为编码后的累计奖励目标，第9索引为全部未完成订单的预计完工裕量。策略头、workload目标、Flow模式和奖励定义均由既有配置选择。

## 完工裕量

对所有已知且未完成的订单，令 `S_j=(horizon-estimated_finish_j)/horizon`，全局量取 `min_j S_j`。未释放订单也参与，释放时间由现有顺序资源投影处理；完成订单不参与，无未完成订单返回0。

该值保持正负号，不裁剪、不乘偏好权重。它只复用订单节点已经计算的完整完工时间缓存，不额外投影。WAIT特征继续使用已释放活跃订单的口径。

订单估计使用剩余工序链、机器选择、承诺状态与工人服务；各订单独立推演，逐工序选择最快候选。正裕量不是联合可行性证明，负裕量不是动作屏蔽依据。

## 累计目标编码

令 `x_i=J_i/s_i`，其中目标来自奖励目标向量，尺度来自既有冻结manifest。Flow使用raw或excess；第三目标使用配置的Variance或geometric OWA。

| network.global_objective_encoding | 编码 | 用途 |
|---|---|---|
| log2_v1 | log2(1+x) | schema12默认 |
| bounded_v1 | x/(1+x) | 对照 |

仅编码全局累计目标，不变换候选特征、裕量或奖励Q。实现使用float64及数值稳定公式，要求有限非负目标和有限正尺度，输出转为float32观测。对数实现避免先计算极大比值而溢出；有界实现避免大目标与大尺度求和溢出。

对照配置为 `configs/default.json` 与 `configs/controls/bounded.json`，继承同一Universal主线，仅编码与实验名称不同。两种编码需要各自训练。默认的第三目标与Flow模式由主线继承，可按既有配置机制选择其他协议。

## 身份与兼容

累计目标特征名称包含log2/bounded标识，两种编码均为10维但名称不同。编码写入网络规格、runtime manifest和评估provenance，并随有效配置参与哈希。网络加载、观测校验与rollout buffer拒绝跨编码混用。

当前配置、观测、checkpoint 和 rollout buffer 统一使用 schema12。旧版本在配置继承或参数加载前拒绝，避免九维观测被静默改为十维输入。历史模型复核使用原代码版本。

## 验收

`test/test_schema12_observation.py`覆盖编码数学、极值与非法输入，订单集合、有符号裕量、缓存、加工和锁定状态、机器候选以及独立订单估计的限制；同动作对照覆盖raw/excess、shared-head、Variance/OWA，检查mask、奖励、物理日志、节点和关系一致性。

`test/test_module_entrypoints.py`对两种编码分别运行两轮训练、PPO更新、13偏好验证、66偏好最终评估和best/last模型重载，核对编码身份与provenance。短程验收不用于证明质量或完成率提高。
