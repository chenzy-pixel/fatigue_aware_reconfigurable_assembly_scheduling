# 20路评测实现复核

2026-10-24。范围：本轮并行评测、训练池复用、派生结果缓存、计时和报告审计。已修复三类问题；本轮未启动正式实验。

|问题|触发与影响|修复|
|---|---|---|
|派生sample1缓存未核对母结果|Full结果在同一job下重新生成或被恢复替换后，旧sample1文件仍通过自身哈希检查；报告此前也不验证保存的母结果引用|重跑时比较母job与内容哈希，过期派生结果自动重建；报告按种子和实例检查母引用及首条原始轨迹，文件读取顺序不影响检查|
|已使用训练观察保留过久|更新完成后整批图观察仍由`rollout.buffer`引用，验证和下一次采集期间可能同时驻留旧观察与新数据，增加峰值主机内存|更新及样本计数完成后立即清空buffer，逐Episode指标和轨迹摘要继续保存；DDQN replay按原合同保留|
|复用worker池缺少完整入口检查|请求20路但只提供2路池时被静默降为2路；调用配置与池环境不一致时可能按错误配置解释指标|不足容量或物理/目标配置不一致直接报错，并校验调用网络配置；汇总前检查返回轨迹索引完整且有序，派生sample1要求有序的三次采样母结果|

代码位置：[训练缓存释放](../agent/comparators/learning.py)、[评测入口检查](../agent/comparators/parallel_evaluation.py)、[派生缓存刷新](../scripts/experiment_suite.py)、[报告跨文件审计](../analysis/experiment_suite.py)。

## 验证

本轮 **61 passed，0 failures，0 errors**，耗时144.58秒。[JUnit证据](../result/verification/evaluation_parallel_review_20261024/tests.xml)。测试集合包括：

- 七种学习配置的串行/并行轨迹与结果核对、异长实例续填和独立采样种子。
- 同job母结果更新后，sample1刷新且不重载模型、不启动worker；派生文件缺失后重建。
- 报告拒绝过期母哈希及与母结果不一致的首轨迹，倒序读取仍可验证。
- 验证开始前及下一批采集前，上一批观察缓存已释放；2000次算例尝试、100Epoch和20次验证计数继续通过。
- 评测池容量、环境配置不匹配以及轨迹漏回的拒绝检查。
- 报告的合成数据完整流水线与原有训练/规则/搜索合同回归。

```powershell
.\.venv\Scripts\python.exe -m pytest test/test_experiment_evaluation_execution.py test/test_episode_training.py test/test_experiment_suite.py test/test_parallel_experiment_evaluation.py -q --basetemp=.review24 --junitxml=result/verification/evaluation_parallel_review_20261024/tests.xml
```

当前CUDA预检通过，身份为`983172a95fe6f70fd3f583ac8882d2d8ea7b749821bf4e2d4e46ff08322cbc2a`，包含7种网络探针、数据身份与源码快照。[预检结果](../result/verification/evaluation_parallel_review_20261024/preflight/preflight.json)。前一份效率核查中的预检身份属于修复前版本；正式运行以此次源码重新预检。

## 结论边界

本轮没有观察到串并行结果不一致，修订保持既定采样、动作合法性、目标、训练预算和更新规则。检查覆盖当前改动及其调用链，不代表整个项目或完整27例正式实验已获得无缺陷证明。

DCAN的`forward_batch`内部仍逐图计算，是已确认的效率限制；本轮未改写该网络。先前裁剪实例吞吐结果见[效率核查](evaluation_parallel_efficiency_20261024.md)，不能直接换算完整实验耗时。两种进化搜索继续按独占资源计费。
