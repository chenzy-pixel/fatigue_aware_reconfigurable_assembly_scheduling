# 当前代码架构

更新：2026-10-17。当前包含原有PPO主线与独立对照套件，二者共用物理环境和公共目标。

## 端到端路径

`configs.load_config` 展开配置并验证尺度与 runtime manifest → `data` 提供在线训练实例或固定评测实例 → `AssemblySchedulingEnv` 构造 schema12 图观测与合法动作 → `agent/ppo` 批量推断、采集与更新 → `training` 选择 checkpoint → `result` 保存指标和来源。

`train.py` 与 `eval.py` 是训练、评估入口；`scripts.run_experiments` 对指定配置或消融矩阵预检并串行启动训练。`analysis` 复用保存的完整记录；`analysis/studies` 保存具体历史实验的复核程序。

新增`scripts.comparison_suite`读取`configs/comparators`，经`agent/comparators`训练/搜索和原生环境评价，由`analysis/comparator_study.py`生成预算曲线及独立确认报告。它复用PPO优化器与worker传输，但有独立的精确交互采集、训练控制和交付JSON；不通过原`train.py/eval.py`切换DCAN或DDQN。默认阶段为预检，完整运行由用户在远端显式启动。

## 环境与图网络

环境负责生产 `(operation,machine)+WAIT` 和工人 `(machine,worker)+WAIT` 决策。`ResourceProjector` 服务于图特征、等待预测和资源可行性检查；软时间估计的边界见 [时间上下文](order_time_context.md)。动作合法性由环境重新计算。

普通加工只占用机器；工人服务于换模块时的DIS/INS，分别检查源模块/目标模块资质，并按当前疲劳及恢复计算阶段时长和安全性。搜索代理和派工规则只能提出动作，最终可行性与目标始终由同一原生环境裁决，不能以固定setup常数替代重构过程。

schema12 包含 operation、machine、worker、order、module、wave 节点，十四类关系、十维全局输入和三维偏好。三个累计目标采用 log2 编码，另包含所有未完成订单的最小估计裕量，详见 [观测契约](schema12_global_observation.md)。

HGNN 使用按关系共享的消息函数、总入度均值聚合、残差更新和分类型池化。默认 `shared_preference` actor 对合法候选联合使用图上下文与偏好评分；生产、工人和 WAIT 动作使用对应候选特征，critic 输出状态价值。`node_mlp_pool` 为图编码对照。

## 对照方法模块

| 模块 | 当前实现与职责 |
|---|---|
| `agent/mo_alns` | MO_ALNS_v2 / mo_alns_solver_budget_v3；统一时限、原始启发式初始化、轻量动态修复、每邻域至多6次完整精评/24次插入预测及自适应循环 |
| `agent/comparators/evolution.py` | NSGA-II-LS与MOEA/D；共享拓扑工序顺序、机器/DIS/INS工人排序、WAIT位及动态资源模式编码，均用原生解码器。NSGA-II精英保留并做局部搜索；MOEA/D按分解方向协作，首版无局部搜索 |
| `agent/comparators/dcan.py` | 独立DCAN条件注意力网络，128维/2层/4头；扩展机器、工人服务竞争及辅助节点上下文，学习生产/工人/WAIT评分；公共标量critic与PPO更新 |
| `agent/comparators/ddqn.py` | mean/max状态池化、256×256线性Q头；两阶段各7条规则映射到原生合法动作并去重，含WAIT；Double-Q、经验回放与目标网络 |
| `agent/comparators/collection.py`、`learning.py` | 动作发放前预留全局env.step额度；计入WAIT、关闭forced compression、预算边界bootstrap；五档验证及按方法选择best/last，保存交互、更新和样本重用 |
| `agent/comparators/common.py` | 方法及交付协议身份、物理身份、候选适配、原子文件；不把不同模型家族解释为同一种checkpoint |

DCAN是作者结构在本项目环境中的适配，使用本项目奖励和优化，不宣称原论文完整复现。Rule-DDQN的规则接口与PPO的原生候选评分不同，性能差异属于整体方法比较。具体算法、参数和边界见[实施契约](comparator_implementation_spec.md)。

## 独立挑战集与轨迹导出

`data/challenge.py`根据`configs/challenge_suite.json`生成、加载并审计独立冻结的`challenge_v1`：100例五类瓶颈实例，以及20组×4档重构倍率实例。它保存候选台账、物理内容哈希与动作证书，使用独立loader，原V1/V2 split契约保持不变。`data/generate_orders.py`新增可选倍率覆盖和未量化参数记录，用同一组潜在参数生成各倍率档；默认生成路径保留既有随机流和浮点运算顺序。

`scripts.challenge_suite`提供build/audit/smoke/evaluate/report，`analysis/challenge_study.py`负责冻结PPO和启发式评价、原生三次取优及按实例/基础组统计。当前入口仅支持PPO与启发式；其他对照方法尚未接入。数据和已执行检查见[挑战集协议](challenge_suite_protocol.md)。

`ParallelEpisodeRunner.evaluate_records(..., capture_trajectories=True)`可返回完整动作、调度和重构日志；默认关闭。challenge的PPO评价按保存配置聚批执行，启发式每实例只跑一条物理轨迹后按偏好重评分。并行记录的请求耗时不等于独占硬件单请求延迟，不用于替换对照套件串行测量的PPO短时限。

## 目标、奖励与终止

默认目标为 proportional excess Flow、重构费用和实际/承诺重构劳动的 geometric OWA(q=.5)。冻结尺度为 `(368.3143, 353.27, 21.03145448655653)`。观测编码和奖励标量化分别定义，见 [实验协议](experiment_protocol.md)。

基础奖励为 `ΔP - ΔQ - 2·I_failure`；P 是全部已知订单的工序完成比例均值，Q 是有界偏好质量。固定偏好、尺度、gamma=1 且关闭附加势函数项时，完整回报满足 `G=P_T-P_0+Q_0-Q_T-2I`。

任务成功、死锁或 horizon 失败结束环境并关闭自举；工程决策保护或 rollout 截断保留 critic 自举。到达 horizon 时先处理该 tick 的全部事件，再检查完成与失败。

## 训练与结果身份

PPO使用原始采集轨迹及GAE；原有PPO正式验证与测试采用[三次采样取优](default_best_of_three.md)。新套件PPO/DCAN主交付也三次取优，DDQN贪心单次，搜索独立请求交付；原始候选、安全、失败、初始化及全部计算成本分别保留。

保存配置、网络规格、runtime manifest、数据与尺度哈希共同定义模型身份。当前环境契约为schema12；原有PPO入口加载其shared-head家族，suite按方法分派PPO、DCAN或DDQN家族并核对身份，不能互换权重。仅显式comparison配置增加新方法元数据，历史配置解释保持独立。

suite每方法每训练种子精确350411次env.step，调参35041步；固定验证分为21调参/21选模/7确认。搜索默认600秒只是曲线起点，充分性不足再扩展1800秒；另设PPO实际交付时限，整前沿共享档案独立记录。详见[实验协议](experiment_protocol.md)。

套件保存源码/配置/数据/硬件哈希，并锁定搜索与训练并行宽度。完整请求可按身份跳过；中断训练从零重跑，不凭权重伪恢复优化器/replay。远端阶段和完整产物见[启动说明](comparator_remote_startup.md)；本机检查仅证明集成，性能仍待远端训练、曲线和确认。
