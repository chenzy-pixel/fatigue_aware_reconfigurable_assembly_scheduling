# 三因子27例实验体系 v1

确认日期：2026-10-23。机器清单：`configs/experiment_suite_v1.json`。入口：`python -m scripts.experiment_suite`，也可使用`python -m scripts.comparison_suite --suite`。正式性能和预算充分性需要远端实验产物。

## 实验矩阵与训练合同

|类别|配置|
|---|---|
|完整方法与消融|Full、w/o Graph、w/o Direct、w/o Residual|
|学习对照|HGNN-A2C、Rule-DDQN、DCAN-adapted|
|固定规则|EFT-LF、SRW-LF、MC-MC、MWL-MWL、Existing-Heuristic|
|进化搜索|NSGA-II-LS、MOEA/D|
|Full推理方式|greedy、repeat 0单次采样、best-of-3；共用Full模型|

正式训练35次：7配置×5种子11/23/37/53/71。Full统一复用。每次2000次算例尝试，每批20例，共100个Epoch；每5个Epoch在27例×13偏好验证，共20次。所有尝试均计数，工程截断保留价值自举。按安全性、最低偏好完成率和公共Q选模。分别记录实际训练/验证步数、更新次数、样本使用次数和耗时；比较采用相同算例尝试次数口径。

固定标定54例、验证27例、测试27例；数据schema3、观察schema13、预测语义`order_chain_remaining_context_v3_1`。目标为Excess Flow、重构费用、geometric OWA(q=.5)，沿用冻结尺度和公共Q。测试按N分三规模组各9例，另作K、M描述性分层；M同时影响机器、工人和模块数。

首轮参数冻结。A2C共享γ=1、GAE λ=.95、Adam学习率1e-4、价值系数.5、熵系数.01、梯度裁剪.5。每批20例只遍历一次，以64条转移为微批、按样本数加权累积梯度后更新一次。PPO、DCAN、DDQN沿用现有效参数。35次训练包含700次验证，约666900条验证轨迹，资源估算须计入。

## 组件与规则

新增`network.direct_features_enabled`和`network.residual_enabled`，默认true，写入网络规格与检查点校验。旧Full检查点缺失字段时按true解释；干预身份必须匹配。

- w/o Graph：节点MLP与池化替代图消息传播。
- w/o Direct：生产、工人和WAIT评分头的直接目标特征置零；图与候选编码保留预测信息。
- w/o Residual：输出base logits；残差模块与gate冻结并从优化器排除，记录总参数量和可训练参数量。
- 单条参考路径与批量路径使用相同干预，各变体重新训练。

|规则|生产排序|工人排序|
|---|---|---|
|EFT-LF|预测完工时刻|投影疲劳|
|SRW-LF|订单剩余工作量，随后预测完工时刻|投影疲劳|
|MC-MC|预计重构费用增量|当前拆装费用增量|
|MWL-MWL|预计劳动OWA增量|当前劳动OWA增量|
|Existing-Heuristic|保留当前HeuristicPolicy实现|保留当前决胜逻辑|

前四项从合法pair中选择，有效且有限投影优先；全部无效时按稳定实体ID回退。分数平局按操作/机器或机器/工人ID决胜。无pair时使用合法WAIT，无合法动作时报错。确定性规则每实例生成一次解，对全部偏好重评分；质量样本单位为实例。规则请求时间标注为共享确定性求解耗时加重评分耗时的估计。

Rule-DDQN是状态、规则动作及更新机制共同改变的整体方法对照。DCAN-adapted使用本项目环境和奖励，仍采用PPO；[原文依据](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c411eef5b6ee7d9895ad129a4bfc6cc-Abstract-Conference.html)。HGNN-A2C承担主要更新机制对照。疲劳中性与目标替换沿用各自历史研究清单。

## 搜索与计费

|任务|偏好|正式运行量|上限|预算|
|---|---|---:|---|---:|
|整条前沿|66点网格|2×27×5=270|每实例600秒|45小时|
|独立请求|三个端点＋精确等权|2×27×5×4=1080|每请求600秒|180小时|

前沿入口`solve_frontier`只初始化一次种群，内部辅助偏好为等权，连续维护一个档案，结束后从档案为66点交付。独立请求各自初始化，各请求单独保存。66点网格不含精确等权点，学习方法额外执行四偏好评价。

学习前沿主报66偏好全部付费可行样本的HV，辅报交付前沿HV；DDQN采用greedy。Full额外比较greedy和sample1；sample1与best-of-3的repeat 0使用同一采样种子。图构建、候选生成、环境推进、合法性校验均计费，加载和预热另列。保存60/180/600秒快照，只纳入截止前完成验证的候选，初始化、局部搜索和失败尝试均记录成本。

开发检查使用标定集C01/C14/C27第一份实例、种子11，共30次600秒请求，额外5小时。每请求须完成初始化、至少一轮完整迭代和至少两个不同可行轨迹。`freeze-budget`检查完整覆盖后生成含证据哈希的`search_budget.json`；未通过则保留记录并分析，预算调整需要修订协议。600秒不代表充分收敛。

## 结果与统计

预检保存有效配置、源码哈希、`source.zip`、数据清单、硬件、参数量及完整矩阵。身份绑定源码、各方法配置、预算、偏好和数据。同目录使用操作系统锁串行执行；训练中断从零重启，完整身份匹配请求可复用，评测要求完整训练。

```
<output>/preflight.json、source.zip
<output>/training/<arm>/<seed>/
  config.json、epochs/、training_log.json、episodes.json
  validations.json、validation_requests/epoch_*.json
  best_checkpoint.pt、last_checkpoint.pt、summary.json
<output>/rules/<rule>/<instance>.json
<output>/development/<task>/<method>/<seed>/*.json
<output>/search_budget.json
<output>/evaluation/<task>/<arm>/<seed>/*.json
<output>/report/
```

验证原始请求按Epoch单独保存，累计验证文件保留摘要、逐实例行及原始文件路径。报告逐文件读取，仅在内存保留统计与绘图信息。

- 请求指标：完成率、公共Q、原始三目标、交付时间；失败Q=1，共同成功质量与目标差值另列：先在每个实例/偏好内分别平均各方法的成功重复，双方均有成功解才进入该条件统计；随后按实例汇总。
- 前沿指标：付费解HV、交付HV、覆盖率；统一`x/(scale+x)`变换、参考点(1,1,1)，只纳入合法完成解。
- 每实例内先汇总种子/偏好，再对27实例配对。bootstrap为10000次、随机源20261023、按三规模组分层，输出95%区间。双侧Wilcoxon，差值全0时p=1。Holm在消融、规则、学习、进化、解码组内按任务及指标分别校正。
- 输出交付明细、逐实例表、规模/K/M汇总、配对统计、共同成功差值、独立种子波动、训练耗时与各实验组表格。
- 学习曲线采用各种子共同覆盖的实际步数区间；搜索曲线保存各截止点。另输出27例逐实例质量图。代表性前沿预先固定C01/C14/C27、种子11。图输出SVG/PDF/300dpi PNG。
- 缺失、重复、身份冲突、Q与目标不符、保存HV与付费解不符均阻止正式报告生成。评测文件带内容哈希。

## 三个PowerShell启动入口

三个入口共用默认结果目录`result/experiment_suite_v1`、实验清单和预检身份；默认各使用五种子11/23/37/53/71，自动选择仓库内的`.venv` Python。可以用`-Python`指定解释器。

|启动文件|内容|默认流程|
|---|---|---|
|`scripts/run_main_experiment.ps1`|Full、Rule-DDQN、DCAN，共15次训练|预检→训练→两类评测；包含Full推理策略对照|
|`scripts/run_ablations_a2c.ps1`|w/o Graph、w/o Direct、w/o Residual、HGNN-A2C，共20次训练|预检→训练→两类评测|
|`scripts/run_rules_search.ps1`|五规则、NSGA-II-LS与MOEA/D的前沿和独立请求|预检→五规则→开发搜索→冻结预算→正式搜索|

在仓库根目录分别手动执行，每条命令运行对应组的完整流程：

```powershell
.\scripts\run_main_experiment.ps1 -RemoteRun
.\scripts\run_ablations_a2c.ps1 -RemoteRun
.\scripts\run_rules_search.ps1 -RemoteRun
```

全部组的结果齐备后，由第三个入口生成完整汇总：

```powershell
.\scripts\run_rules_search.ps1 -Stage report
```

`-Stage`默认`all`。主实验和消融入口也可单独指定`preflight`、`train`、`evaluate`；规则搜索入口支持`preflight`、`rules`、`develop-search`、`freeze-budget`、`search`、`report`。默认完整规则搜索流程止于搜索，报告作为明确的收尾选项，因此该组可独立执行。

仅查看启动流程时加`-DryRun`，只做预检时加`-Stage preflight`。正式阶段通过`-RemoteRun`传递既有执行开关；任一步骤失败立即停止该入口的后续步骤。三个入口共用底层`run_experiment_suite.ps1`及`ExperimentLaunch.psm1`，算法和预算不因入口分组而改变。

共同参数包括`-Device`（默认cuda）、`-OutputDir`、`-Manifest`、`-TrainingWorkers`（默认20）、`-Seeds 11,23`和`-Task frontier/independent_request/both`（默认both）。所有组保持相同的设备、worker数、清单和结果目录；修改这些选项后重新预检。`-Task`控制正式评测/搜索，开发搜索始终覆盖两类任务以完成预算检查。主实验与消融两组覆盖完整7×5训练矩阵，Full模型沿用同一份结果。

例如，仅运行一个种子的消融训练：

```powershell
.\scripts\run_ablations_a2c.ps1 -Stage preflight -Device cuda -Seeds 11 -TrainingWorkers 8
.\scripts\run_ablations_a2c.ps1 -Stage train -Device cuda -Seeds 11 -TrainingWorkers 8 -RemoteRun
```

启动脚本及共享模块纳入源码快照和身份校验。默认单作业、单Torch线程，计时评测使用独占资源。保留通用Python入口用于Linux或其他编排：`python -m scripts.experiment_suite --stage <stage> --device cuda --output-dir result/experiment_suite_v1`，正式执行阶段增加`--remote-run`，学习子组通过`--arms`指定。

本地小实例验收：

```powershell
& $py -m scripts.experiment_suite --stage smoke --device cpu --output-dir result/verification/experiment_suite_local
& $py -m pytest test/test_experiment_suite.py test/test_episode_training.py test/test_factorial27.py -q --basetemp=.ep
```
