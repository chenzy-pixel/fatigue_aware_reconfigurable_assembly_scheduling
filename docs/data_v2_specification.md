# 算例生成与训练分布 v2 实验规格

实施日期：2026-09-30。默认配置为 `configs/default.json`，生成器和数据 schema 均为 `2.0.0`，结果 schema 为 `8.0.0`。

## 生成与诊断

保留七类场景和现有结构：8 台机器、6 名工人、3 类模块、240 分钟时域。候选接受条件只有数据合法性、既定场景结构约束及 `necessary_conditions_v1` 必要可行性预检查，包括资源资格、关键路径与容量下界。每个实例最多尝试 100 次，失败抛出含静态拒绝原因的 `GenerationError`，构建保持指定数量及场景配额。

启发式完成情况、makespan、配置缺口、重构比例、工人竞争、疲劳屏蔽和波次重叠仅作为诊断。训练默认关闭启发式及反事实诊断；validation、test、OOD、stress 运行启发式诊断，反事实诊断默认关闭。诊断截断保留实例；执行异常直接终止构建。调度验证失败保留诊断结果，并标记可行性未知。stress 发布没有启发式截断比例门槛。

| 元数据 | 含义 |
|---|---|
| `severity` | 本次采样压力强度上限，取值 `(0,1]` |
| `feasibility_status` | `observed_feasible` 或 `unknown` |
| `diagnostic_status` | `not_run`、`completed` 或 `truncated` |
| `diagnostic_terminal_reason` | 诊断结束原因；未运行时为 `null` |

只有诊断完整完成、调度验证通过、疲劳安全且 makespan 在时域内时，标记 `observed_feasible`。预检查通过、算法失败均不能证明不可行。训练默认标签为 `unknown`。

manifest 保存生成配置、环境配置、分布契约三个 SHA256，以及场景数量、诊断状态、可行性状态和静态拒绝原因汇总。实例保存同样哈希。载入、缓存和断点续建校验哈希，文件内容另以 SHA256 校验。

生成配置哈希继续排除历史课程字段和窗口长度，保持既有固定评测池身份。在线缓存对完整生成配置、环境、模板、训练总轮数及预检查版本取哈希；当前配置产生独立缓存指纹。算法种子不进入实例缓存指纹。

## 训练采样

| 场景（固定顺序） | 比例 | 每100个episode数量 |
|---|---:|---:|
| easy | 5% | 5 |
| balanced | 35% | 35 |
| machine_bottleneck | 15% | 15 |
| reconfiguration_bottleneck | 15% | 15 |
| worker_bottleneck | 10% | 10 |
| fatigue_bottleneck | 10% | 10 |
| high_arrival_pressure | 10% | 10 |

2026-10-21起，默认训练直接读取 `generator.dataset_pressure_weights`，全程使用完整参数范围（`severity=1.0`）。每100个episode使用最大余数法分配配额，余数平局按表中顺序决定，再沿用独立实例随机流洗牌。尾窗按实际长度分配。实例种子仍为训练种子起点加episode索引；洗牌计划依赖总训练长度。训练预算、模型及固定评测身份沿用现有契约。

生成器保留强度参数用于诊断；在线训练固定为1。历史保存配置仍可用于评测和分析，其训练复现使用原代码版本。

场景计划、成本配置和偏好使用独立随机流。成本配置类别由生成器版本、split、实例种子及成本流标记派生，不依赖场景标签或强度；协议标记为 `independent_pressure_cost_preference_v2`。每 20 个 episode 包含三个端点各两次及十四个 scrambled-Sobol 混合偏好。实例流不依赖算法种子。默认 2000-episode 计划审计七类场景与三个端点、混合偏好的交叉覆盖。

`train_log.csv` 保存场景、强度、实际订单/工序/重构时间/疲劳参数、偏好、完成状态及原因；`training_distribution.json` 汇总场景数量、参数范围、交叉覆盖、分场景完成率及生成/环境耗时。

## 固定验证与结果口径

开发验证池 500 个实例。选模 50 个，按场景顺序配额为 `3、18、7、7、5、5、5`；诊断 49 个，每场景 7 个。场景内以 manifest 哈希、实例种子、子集角色派生稳定排序，先选选模子集，再排除已选实例选诊断子集。选样不读取策略成绩，跨算法种子与 checkpoint 固定。

`data/manifests/v2/validation/subsets.json` 保存显式索引、种子、文件 SHA256、子集 SHA256。子集哈希包含数据 manifest、索引及文件内容哈希，角色作为说明字段。训练保存同一索引及哈希到 `validation_subsets.json` 和 checkpoint。

评测 API 和 CLI 支持 `instance_indices` / `--instance-indices`，与 offset/limit 互斥。主验证沿用 13 个偏好、三次采样、原验证频率及 completion-first 规则。诊断子集每五次主验证及训练结束运行，结果不传给 checkpoint 或学习率控制器。

完成率以所有选中实例为分母，包括 `unknown`，报告每个偏好、场景及可行性标签的完成情况。只在策略和参考启发式都完整、安全、时域内完成时计算 heuristic gap；否则 JSON 为 `null`，CSV 留空。成功轨迹质量与失败进度、失败原因分别汇总。聚合和分析校验数据、子集、生成/环境/分布协议、冻结尺度和结果版本。

## 数据和开发对照

v2 目录为 `data/instances/v2`、`data/manifests/v2`、`data/instances/train_cache/v2`。开发数据 validation 500，test、OOD、stress 各 20。v1 默认配置原始字节保存于 `configs/archive/data_v1.json`，历史文件 SHA256 保存于 `configs/archive/data_v1_hashes.json`；历史数据可用对应配置读取。发布前后校验 565 个历史文件。新旧数据协议分别汇总。

历史开发对照（不作为当前训练入口）使用算法种子 11 的两组各 400 episode：原课程＋完整参数范围，新课程＋渐进强度。两组使用相同 v2 静态生成规则、固定验证池及偏好协议。每 200 episode 验证一次，目标比例选 20 个实例、13 个偏好、一次采样；结束诊断每场景一个实例。固定 best 在全部 20 个测试实例上进行 66 点偏好、一次采样。保存分场景完成率、原始目标、失败原因及生成/训练耗时。首轮判断协议正确、数值稳定及可追溯性，性能结论需完整多种子实验。

## 可运行入口与验收

在项目根目录运行；`python` 应指向已安装项目依赖的解释器：

```powershell
python -m scripts.prepare_data_v2
python -m pytest test/test_data_v2.py test/test_dataset.py test/test_dataset_splits.py test/test_feasibility_precheck.py test/test_generator_distributions.py
python -m scripts.run_00_smoke
python -m eval --dataset validation --policy heuristic --instance-indices 4 11 18
```

数据准备入口默认采用 4 个生成进程，可用 `--generation-workers` 调整；支持断点续建，已发布数据默认只校验，显式 `--overwrite` 才重建。串行与并行生成的实例、manifest 字节保持一致。审计摘要为 `data/manifests/v2/audit.json`；短对照摘要为 `result/audits/data_v2/comparison.json`，详细轨迹和 checkpoint 在各 run 目录。历史短对照源码保存在清理快照，原始比较结果见 `docs/experiments/data_v2/`。当前回归覆盖训练计划、PPO数值、固定子集、checkpoint、逐轨迹评测矩阵及无效gap留空规则。

验收覆盖诊断截断不重采样、诊断异常传播、必要条件拒绝、状态标签、固定场景配额、尾窗和全程完整强度、随机流独立、缓存/断点/worker 一致、旧文件保护、50/49 配额与无重叠、实际验证索引、未知实例分母、无有效参考 gap 为空、诊断不参与选模，以及训练—验证—最终评测链路。
