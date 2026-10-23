# 四类对照的远端启动与结果核验

2026-10-23：当前schema13三因子27例实验使用[实验体系v1](experiment_suite_v1.md)及`configs/experiment_suite_v1.json`，包括35次按2000算例尝试训练、五规则和两类搜索。以下保留历史协议记录。

2026-10-21：已返回600秒结果后的NSGA-II＋LS / MOEA/D扩展采用独立的[效率预检与1800秒启动入口](heuristic_search_runtime.md)，复用原批次保存配置、7例、4偏好及种群16/32；先测远端并行吞吐，再运行每请求1800秒。下文完整suite入口及其历史匹配训练协议另行保留。

本套件实现NSGA-II-LS、MOEA/D、DCAN适配版和规则DDQN，并重新训练PPO作为匹配参照。源码说明及实验合同见[实施契约](comparator_implementation_spec.md)；文献和改造边界见[候选方法审查](baseline_candidates_20261017.md)。用户在远端手动启动，本机smoke只是链路检查，远端实验完成前没有正式性能结论。全项目矩阵与交付协议见[实验清单](experiment_runlist.md)和[三次取优说明](default_best_of_three.md)。

## 同步和预检

将本轮全部修改/新增源码与配置同步到远端，包含`agent/comparators`、`analysis/comparator_study.py`、`scripts/comparison_suite.py`、PowerShell包装器、`configs/comparators`及`configs/runtime.py`。有新增未提交文件时，仅git pull不能同步它们。保持现有environment/data/agent/mo_alns及其他依赖源码同版。

远端需保留项目固定V2验证集、实例模板、归一化清单及原Python环境。使用已激活的CUDA PyTorch环境，项目根目录执行：

```powershell
python -u -m scripts.comparison_suite --stage preflight --output-dir result/analysis/comparison_suite_remote_v1
```

也可以使用PowerShell包装器；默认只预检：

```powershell
.\scripts\run_comparison_suite.ps1 -Python python -OutputDir result/analysis/comparison_suite_remote_v1
```

预检检查500例validation及21调参/21选模/7确认互斥划分、三个模型的构建和合法前向，并保存代码/配置/数据/硬件身份与任务规模。每方法的配置可从`configs/comparators`检查。套件只继承当前环境和训练参数，不依赖旧PPO权重。

## 启动全部本阶段工作

远端预检成功后：

```powershell
python -u -m scripts.comparison_suite --stage all --remote-run --output-dir result/analysis/comparison_suite_remote_v1 --workers 8 --training-workers 20
```

或：

```powershell
.\scripts\run_comparison_suite.ps1 -Stage all -RemoteRun -Python python -OutputDir result/analysis/comparison_suite_remote_v1 -Workers 8 -TrainingWorkers 20
```

顺序为搜索种群调参→学习率调参→三方法三种子匹配训练→搜索预算曲线及必要扩展→独立确认→报告→正式清单。`all`到清单生成结束，完整test/OOD/stress须另行启动。

这是一套长批次，建议在远端持久终端运行。预检给出的600秒曲线时间上限为126 CPU小时（含三种搜索、756请求），条件1800秒扩展最多再378 CPU小时；8个进程可并行，但这不是可靠墙钟ETA。另有搜索调参168请求×600秒、ALNS机制验收、9个学习调参请求、9个正式训练请求和3003个独立确认交付。确认阶段串行计时，可能比并行曲线耗时更长。

学习正式预算为350411次env.step/方法/训练种子；调参为35041次。原生PPO和DCAN报告greedy及best-of-3，DDQN报告greedy。三次重复计时必须产生相同候选轨迹，中位时延不额外增加选优候选数。

确认计时前加载模型并在调参实例上预热；请求时延包括环境构建/重置、决策、仿真和完整轨迹检查，三次取优累计三条轨迹。模型加载、预热、数据读取和结果文件写盘不计入该请求时限；GPU计时前后同步。它描述已加载服务的交付时延，冷启动成本需另行计量。

## 分阶段执行和恢复

`--stage`支持`preflight`、`smoke`、`tune-search`、`tune-learning`、`train`、`curves`、`compare`、`report`、`formal`、`formal-evaluate`、`front-search`、`all`。重计算阶段在远端加`--remote-run`，并要求同一身份的预检已通过。单独执行时保持同一输出目录和并行宽度，例如：

```powershell
python -u -m scripts.comparison_suite --stage train --remote-run --output-dir result/analysis/comparison_suite_remote_v1 --workers 8 --training-workers 20
```

相同命令重跑会跳过身份匹配的完整搜索/训练请求。搜索以完整请求恢复；中断的训练从零重跑该请求。源码、配置、数据、硬件、搜索或训练worker数改变时使用新输出目录，不拼接不同身份的结果。旧本机MO-ALNS目录与新suite输出分别保留；新套件不要求复制历史PPO权重。

新增challenge功能修改了`agent/ppo/parallel.py`和`data/generate_orders.py`，这两个文件也纳入comparison_suite源码身份。同步这批修改后，旧源码下的预检/缓存不能沿用；应在新输出目录重新预检。若继续已有远端批次，使用该批次冻结的原代码版本。独立[challenge评估](challenge_suite_protocol.md)另有数据、权重和入口，当前仅接PPO/启发式，不自动扩展本套件的正式split。

## 结果与门槛

- `preflight.json`、`identity.json`、`subsets.json`、`execution_settings.json`：执行前的输入/硬件/子集依据。
- `search_selection.json`、`learning_selection.json`：冻结参数，调参与正式成本分开。
- `training/<method>/<seed>/`：config、5档validations及原始采样、best/last checkpoint、training_log、episodes和summary。核对训练步数350411、更新/样本重用及独立验证耗时。确认集不参与选模。
- `curve_<method>.csv/svg/pdf/png`、`budget_<method>.json`、`improvement_<method>.json`：早期到充分搜索的曲线、初始化改善及预算证据。`alns_reference`保留独立机制和预算产物。
- `confirmation/<arm>/`：7例×13偏好×3种子，包含原始候选、动作、选择、完整请求时延。充分预算与`_ppo_deadline`分别保存。短时限是三个新PPO模型各自三次取优中位时延的跨模型中位数，各模型原始时延同时保留。
- `paired_deliveries.csv`、`delivery_hv.csv`、`paid_archive_hv.csv`、`comparison_summary.json`、`comparison.svg/pdf/png`、`report.md`：独立请求端点、13点交付HV、全部付费档案HV与按实例聚类的配对统计。搜索种子不当作训练种子。
- `training_extension_manifest.json`：末档仍有明显收益时提示训练充分性未确认，列出统一2B扩展；不自动执行。

未完成的确认矩阵不能生成完整报告。预算未充分或确认集不支持正改善时，正式清单标明需要复核，正式扩展入口拒绝自动放行。独立改善按搜索相对自身初始化最优解的实例平均Q增益计算，95% bootstrap区间下界须大于0；不要求超过PPO。失败不从覆盖率或全覆盖损失中删除。7例确认仅为小规模证据；实现与smoke的实际检查记录见[验证记录](comparator_verification_20261017.md)。

## 独立的整前沿搜索

默认主表每个偏好从零开始。若要一次种群运行输出整条前沿，另行执行：

```powershell
python -u -m scripts.comparison_suite --stage front-search --front-seconds 1800 --remote-run --output-dir result/analysis/comparison_suite_remote_v1
```

此处1800是**每实例整次运行的总预算**，不是每偏好预算。产物放在`whole_front`，禁止将共享档案端点混入独立请求主表。与学习策略比较时，使用同实例全部13个偏好请求的累计成本；`paid_archive_hv.csv`已保留该总成本。

## 正式批次入口

完成报告并检查正式manifest的预算/独立确认门槛后，可另外执行：

```powershell
python -u -m scripts.comparison_suite --stage formal-evaluate --remote-run --output-dir result/analysis/comparison_suite_remote_v1
```

它读取已冻结的checkpoint、参数和正式manifest，在test/OOD/stress上运行66偏好评测。默认`all`不会进入此阶段。
