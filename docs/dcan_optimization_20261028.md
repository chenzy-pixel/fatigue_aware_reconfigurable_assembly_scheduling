# DCAN批处理优化与验证

2026-10-28。优化当前同环境适配版的计算方式。网络仍为128维、2层、4头、740252个参数；参数名称、检查点规格、物理环境、动作映射、奖励、PPO超参数和2000次算例尝试预算沿用当前合同。

## 实现

- 多图按节点类型打包成互不连通的图，并统一计算线性投影和关系消息。各图保留独立的邻居索引、度数、偏好、池化分段和动作位置。
- 条件注意力只构造实际邻居与偏好槽位。工序链通常为前驱、自身、后继及偏好；打包器也支持更高节点度数，去重并保留全部邻居。资源注意力仍使用当前竞争强度定义；偏好强度继续取所有同图资源的均值。
- 保留原偏好更新：原对角槽位替换为偏好嵌入，额外槽位替换为工序/资源嵌入。各阶段的候选评分和WAIT评分也改为批量计算，最后恢复每个观察的原始动作索引和掩码。
- 默认执行方式为`packed_neighbors_v1`。原稠密逐图实现保留为`reference_dcan_v1`，供数值和性能核对，不增加参数或改变保存的`network_spec`。

主要实现：[dcan.py](../agent/comparators/dcan.py)。浮点求和和矩阵运算的分组改变，数值按容差核对；不承诺长期训练逐位一致或相同的随机轨迹。正式算法质量仍需完整实验验证。

## 验证范围

[新增回归](../test/test_dcan_batching.py)覆盖条件注意力输出与输入/参数梯度、CPU float32/float64、CUDA、不同规模及偏好的混合决策阶段、批次重排与单图对照、240工序完整观察、空候选和空辅助节点、输入检查及参考检查点权重/优化器加载。另在CUDA上以64个完整240工序观察执行实际PPO更新，检查两次优化、128次样本呈现、有限损失及权重更新。

CPU float32整网输出/梯度容差为`rtol=2e-5, atol=2e-6`；float64为`rtol=1e-10, atol=1e-11`。CUDA输出为`rtol=3e-5, atol=3e-6`，梯度为`rtol=2e-4, atol=1e-5`。检查点可加载表示权重契约兼容，不等于训练器具有完整断点恢复能力。

全仓库回归：**149 passed**，含13项新增DCAN检查，耗时136.41秒；证据：[regression.xml](../result/verification/dcan_optimization_20261028/regression.xml)。单独的多进程轨迹测试见[worker_parity.xml](../result/verification/dcan_optimization_20261028/worker_parity.xml)。初次受限环境中的20路worker启动超过60秒；在允许Windows多进程通信的环境中完成验证，未修改正式worker超时或训练配置。

七配置CUDA预检全部通过，DCAN参数量仍为740252；[预检记录](../result/verification/dcan_optimization_20261028/preflight/preflight.json)的工作区执行身份为`a73533efb6d4f4ffddd9b49157b718735438cd8ffddbd63df096a2e6317ba6bd`。预检命令会在运行目录生成源码快照。本次只执行本地检查和吞吐探针，未启动正式训练矩阵。

## 本机实测

Windows 11、RTX 4060 Laptop GPU、Torch 2.14.1+cu132；Torch线程数1。同一组未训练权重、相同输入和PPO配置，对两个执行方式预热后计时。推理/更新各取3次中位数；完整求解计时不含worker池启动，但包含环境推进、观察和动作传输及终态校验。

|测试|参考执行|优化执行|加速比|
|---|---:|---:|---:|
|20图策略推理，C01/C14/C27完整规模，混合生产/工人阶段|0.4353秒|0.0301秒|14.45×|
|64样本PPO更新，2遍，每实例截取2订单|5.6957秒|0.2036秒|27.97×|
|20路完整求解，9请求/27轨迹，每实例截取2订单|25.5965秒|3.7489秒|6.83×|

更新计时使用真实观察及固定的合成优势/回报，目的为比较执行开销；不作为训练收敛实验。两条路径的PPO损失、优化次数及样本呈现计数通过核对。完整求解的全部动作、调度日志、成功状态、违规、选中repeat及目标通过核对，权重哈希保持不变。原始计时和配置见[benchmark.json](../result/verification/dcan_optimization_20261028/benchmark.json)。

以上倍数对应表中工作负载，不能直接换算正式2000次训练及27×13×3验证的总时长，也不构成模型质量提升结论。

复现：

```powershell
python -m pytest test -q
python -m scripts.benchmark_dcan --device cuda --workers 20 --repeats 3
```

## 运行应用

在原项目停止旧进程，清理原实验根目录下`training/dcan`及`evaluation/<任务>/dcan`的旧DCAN结果，保留`preflight.json`、`source.zip`及其他算法产物。更新源码后，对原根目录执行兼容预检，再从第1轮启动DCAN。以当前实验根目录`result/experiment_suite_v1_parallel20`为例：

```powershell
python -m scripts.experiment_suite --stage preflight --device cuda --output-dir result/experiment_suite_v1_parallel20 --refresh-dcan
python -m scripts.experiment_suite --stage train --device cuda --arms dcan --seeds 11 --output-dir result/experiment_suite_v1_parallel20 --remote-run
```

训练结果继续写入`result/experiment_suite_v1_parallel20/training/dcan/11`。上述命令只启动DCAN seed 11；全部正式矩阵仍按[实验体系](experiment_suite_v1.md)运行。

兼容预检逐项比较原始源码快照与当前数据、配置及源码。只接纳`configs/dcan_execution_reuse_v1.json`列明的已审查版本，并执行七配置网络探针。原比较身份、原始预检及源码快照保持原样；当前执行的预检、快照和兼容记录写入`executions/<执行身份>`，`execution_reuse.json`登记当前版本。DCAN检查点、训练任务及评测缓存另外绑定当前执行身份，汇总检查这些来源；其他方法继续复用原产物。后续正常启动入口会识别该登记。

运行中的Python进程不会自动载入新的计算路径。既有训练器按`restart_incomplete_request`处理未完成请求；此入口执行从头训练。兼容预检不会删除训练文件，旧DCAN结果未清理、原始快照缺失、数据/配置改变或其他源码不匹配时会停止并列出原因。

同目录刷新验证：全仓库**170项回归通过**，覆盖原结果保留、已审查版本准入、配置/数据变化拒绝、DCAN任务与检查点执行身份、缓存及汇总审计。另复制上次本机真实预检建立临时验证根目录，完成七配置CUDA探针；原预检、原源码快照和代表其他方法结果的测试文件逐字节保持一致。证据见[回归记录](../result/verification/dcan_same_root_20261028/regression.xml)和[同目录验证记录](../result/verification/dcan_same_root_20261028/verification.json)。
