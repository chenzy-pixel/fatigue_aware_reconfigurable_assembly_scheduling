# 当前代码架构

## 端到端路径

`configs.load_config` 展开配置并验证尺度与 runtime manifest → `data` 提供在线训练实例或固定评测实例 → `AssemblySchedulingEnv` 构造 schema12 图观测与合法动作 → `agent/ppo` 批量推断、采集与更新 → `training` 选择 checkpoint → `result` 保存指标和来源。

`train.py` 与 `eval.py` 是训练、评估入口；`scripts.run_experiments` 对指定配置或消融矩阵预检并串行启动训练。`analysis` 复用保存的完整记录；`analysis/studies` 保存具体历史实验的复核程序。

## 环境与图网络

环境负责生产 `(operation,machine)+WAIT` 和工人 `(machine,worker)+WAIT` 决策。`ResourceProjector` 服务于图特征、等待预测和资源可行性检查；软时间估计的边界见 [时间上下文](order_time_context.md)。动作合法性由环境重新计算。

schema12 包含 operation、machine、worker、order、module、wave 节点，十四类关系、十维全局输入和三维偏好。三个累计目标采用 log2 编码，另包含所有未完成订单的最小估计裕量，详见 [观测契约](schema12_global_observation.md)。

HGNN 使用按关系共享的消息函数、总入度均值聚合、残差更新和分类型池化。默认 `shared_preference` actor 对合法候选联合使用图上下文与偏好评分；生产、工人和 WAIT 动作使用对应候选特征，critic 输出状态价值。`node_mlp_pool` 为图编码对照。

## 目标、奖励与终止

默认目标为 proportional excess Flow、重构费用和实际/承诺重构劳动的 geometric OWA(q=.5)。冻结尺度为 `(368.3143, 353.27, 21.03145448655653)`。观测编码和奖励标量化分别定义，见 [实验协议](experiment_protocol.md)。

基础奖励为 `ΔP - ΔQ - 2·I_failure`；P 是全部已知订单的工序完成比例均值，Q 是有界偏好质量。固定偏好、尺度、gamma=1 且关闭附加势函数项时，完整回报满足 `G=P_T-P_0+Q_0-Q_T-2I`。

任务成功、死锁或 horizon 失败结束环境并关闭自举；工程决策保护或 rollout 截断保留 critic 自举。到达 horizon 时先处理该 tick 的全部事件，再检查完成与失败。

## 训练与结果身份

PPO 使用原始采集轨迹及 GAE；正式验证与测试采用 [三次采样取优](default_best_of_three.md)。原始请求和被选交付分别落盘，违规和缺失请求继续受完整性与安全门槛约束。

保存配置、网络规格、runtime manifest、数据与尺度哈希共同定义模型身份。当前运行时只接受 schema12 配置与模型；版本不符在加载前拒绝。固定数据、原始实验记录和模型均在原位置保留。
