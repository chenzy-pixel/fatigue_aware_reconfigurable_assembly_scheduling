# 20路评测恢复与效率核查

2026-10-24。本轮完成代码修订和本地验证；正式35次训练及搜索由用户按新身份启动。

后续实现复核修复了派生缓存校验、训练观察释放和评测池入口检查，当前预检身份及回归证据见[实现复核](evaluation_parallel_review_20261024.md)。下文保留首次吞吐探针及当时的验证记录。

## 已完成

|项目|实施及依据|
|---|---|
|训练验证20路|`learning.validation`接入`parallel_evaluation.evaluate_requests`，各方法读取`training.validation_parallel_envs=20`|
|前沿质量20路|复用现有`ParallelEpisodeRunner.evaluate_records`，集中批量推理、完成通道立即续填；每条轨迹独立固定采样种子|
|worker池复用|验证复用训练池；前沿按模型复用池。训练采集通道与验证通道分别受限，池大小取最大值|
|独立请求计时|四偏好串行、独占实测；前沿池先关闭，再执行时延实验。并行墙钟、通道用时和独立时延分列|
|减少重复求解|Full前沿sample1复用best-of-3的repeat 0，附母结果job/hash引用；独立请求仍分别实测三种解码|
|减少数据和I/O重复|固定数据每阶段加载一次；每Epoch合并累计文件导出；未更新的验证表不重写，逐Epoch采集证据在验证前保存|
|完成结果复用|已完成评测重启直接跳过；只缺sample1派生文件时，从母结果重建，省去模型加载和worker启动|

主要代码：[并行请求与种子](../agent/comparators/parallel_evaluation.py)、[验证及训练池](../agent/comparators/learning.py)、[最终评测编排](../scripts/experiment_suite.py)、[20路清单](../configs/experiment_suite_v1.json)。

验证仍保留全部原始动作、目标和物理审计；训练的算例序列、预算及更新规则按既定合同执行。Full前沿复用减少8910次轨迹求解，最终评测实际求解189540条，连同验证共856440条。

## 验证证据

- 全量测试：**135 passed**，覆盖七种学习配置的串行/并行轨迹、异长实例续填、1路/2路种子一致性、冻结权重、DDQN随机源不被评测消耗、共享池验证、首样本复用和训练通道上限。[JUnit](../result/verification/evaluation_parallel_20261024/tests.xml)、[控制台](../result/verification/evaluation_parallel_20261024/tests_console.txt)。
- 在调整前沿池释放时机后，入口编排及PowerShell启动测试复核：**15 passed**；这些测试与全量集合重叠。[JUnit](../result/verification/evaluation_parallel_20261024/workflow_tests.xml)。
- CUDA预检通过：7种配置、数据及源码快照；身份`31a46411a427f9054ef0e7875d24378a0a2b807169cd157d3c9c44ce018ac48a`。[预检结果](../result/verification/evaluation_parallel_20261024/preflight/preflight.json)。

```powershell
.\.venv\Scripts\python.exe -m pytest test -q --basetemp=.pv24 --junitxml=result/verification/evaluation_parallel_20261024/tests.xml
.\.venv\Scripts\python.exe -m pytest test/test_experiment_evaluation_execution.py test/test_experiment_launchers.py -q --basetemp=.px24
```

## 本机吞吐探针

Windows 11、24逻辑CPU、RTX 4060 Laptop GPU、Python 3.12.5、Torch 2.14.1+cu132，主进程Torch线程数1。使用未训练网络及标定集C01/C14/C27，各截取两个订单，分别执行三个端点偏好。保留原机器/工人/模块规模。此处只验证实现与局部吞吐，不能据此推断正式算法质量或完整实验耗时。

计时前预热模型和worker。串行每条轨迹新建环境；并行的环境对象在池启动时创建，计时包含worker内reset、环境推进、图观察及终态校验，并捕获完整轨迹。比较每条动作序列、调度日志、完成状态、选中repeat及目标，相对及绝对数值容差1e-12，全部通过。模型权重未变化。

|方法|请求/轨迹|环境步|串行秒|20路秒|热池加速|池启动秒|
|---|---:|---:|---:|---:|---:|---:|
|Full|9 / 27|1293|35.67|6.24|5.72×|18.06|
|DCAN|9 / 27|1308|32.69|33.12|0.99×|16.61|
|Rule-DDQN|9 / 9|288|2.30|0.58|3.97×|17.32|

证据：[Full](../result/verification/evaluation_parallel_20261024/benchmark_full.json)、[DCAN与DDQN](../result/verification/evaluation_parallel_20261024/benchmark_comparators.json)。复现入口：

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_experiment_evaluation --arms full --workers 20 --max-orders 2 --output result/verification/evaluation_parallel_20261024/benchmark_full.json
.\.venv\Scripts\python.exe -m scripts.benchmark_experiment_evaluation --arms dcan,rule_ddqn --workers 20 --max-orders 2 --output result/verification/evaluation_parallel_20261024/benchmark_comparators.json
```

上表的加速比不包含启动。Full计入此次启动后为24.30秒；短批次的启动开销显著。完整模板的早期探针记录过88.72秒启动，该探针未完成整个串行基线，不作为吞吐结果。正式验证复用训练池，因此不会在20次验证时反复启动。此次Full/DCAN平均推理批次约12.3/12.1，存在尾段不足20条的情况；DDQN探针仅9条轨迹，最多同时9条。完整长批次的收益待远端日志确认。

## 仍存在的效率点

1. **DCAN尚未真正批量计算图网络。** [DCAN.forward_batch](../agent/comparators/dcan.py)内部逐观察调用`_forward`后拼接输出。此次策略推理通路耗31.59/33.12秒，约95.4%，抵消了环境并行的收益。已恢复它的20路环境，但内部图计算仍需单独改造、验证输出及梯度一致性；本轮未改写该网络结构。
2. Full此次策略推理通路耗4.76/6.24秒，约76.3%；该计时含观察打包、网络及取回动作。下一步应细分这里的成本，不能继续用增加worker数推断线性提速。
3. [进化搜索](../agent/comparators/evolution.py)的候选评估仍串行。正式独占计时协议下，225小时正式搜索上限及5小时开发上限独立计算。候选并行会改变600秒内的搜索进程，需另行实现并重做开发检查。
4. [累计训练表](../result/training_history.py)仍按Epoch全量重建。此次已去除重复导出和未变化验证表重写；是否继续改为增量导出，应根据正式I/O计时决定。

## 正式启动

三个现有PowerShell入口默认采用20路验证/前沿，无需增加参数。该修订改变源码和运行身份；已有旧预检/结果时，三个入口统一指定新目录，例如`-OutputDir result/experiment_suite_v1_parallel20`，重新预检。完整命令与计时字段见[实验体系合同](experiment_suite_v1.md)。
