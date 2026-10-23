> 历史 schema12 说明。2026-10-23 起的默认主线采用[三因子批次调度协议](factorial27_migration.md)；本文以下内容保留用于解释历史实现和产物。

# 当前主线完整实验执行清单

独立测试补充：[瓶颈与重构倍率外推套件](challenge_suite_protocol.md)已物化100例test_hard及20组四档倍率外推，共180例，完成数据审计及9例smoke。当前仅支持冻结PPO和启发式，完整66偏好评价单独执行，不计入训练矩阵，也不由comparison_suite的all/formal-evaluate自动展开。

更新日期：2026-10-17。2026-10-16的数据与历史产物核对结果保留；新增对照按当前源码同步。本文件是执行与验收清单，列出的矩阵不代表已完成。用户在远端手动启动实验；本机已取消的MO-ALNS批次不再续跑，远端实时状态未读取。

当前三种子启动脚本：[run_three_seeds.ps1](../scripts/run_three_seeds.ps1)。[逐实验命令清单](experiment_three_seeds.md)逐行列出26个实验；默认只运行Universal，支持指定多个实验、预检和dry-run。

后续训练统一采用默认固定七场景配比与完整参数范围，原有结果按各自保存配置解释。训练配比与随机流规则见[主线协议](experiment_protocol.md)。

## 1．冻结条件与顺序

PPO主线：schema12/log2、HGNN shared-head、excess Flow、OWA(q=.5)、gamma=1、GAE lambda=.95、best_of_3_v1。默认尺度为 `(368.3143,353.27,21.03145448655653)`。当前先执行 **11、23、37** 三个训练种子；配置允许的53、71留待后续扩展，以下运行数按三个种子展开；三次采样属于同一模型的解码预算，不是三个独立训练种子。第6节的新对照套件另按实际环境交互匹配，不能把其训练预算写成这里的2000 episodes。

执行顺序：数据/协议预检 → 每种配置小规模链路检查 → 核心矩阵 → 单因素对照 → 固定best的独立比较与泛化 → 多种子汇总。可先用seed11检查工程，再完成同一预先声明矩阵的其余种子；不能因seed11的测试表现删选不利种子或条件。

| 优先级 | 当前配置数 | 三种子训练数 | 总episodes | 用途 |
|---|---:|---:|---:|---|
| P1 核心 | 8 | 24 | 30,000 | 主方法、单目标参照、图消融、疲劳对照 |
| P2 单因素 | 7 | 21 | 42,000 | 支持raw/excess、OWA、编码、lambda及q相关实证主张 |
| P3 条件扩展 | 11 | 33 | 45,000 | 研究交互作用或单目标OWA机制时再执行 |
| 合计 | 26 | 78 | 117,000 | 现有全部PPO训练配置的完全展开 |

**核心加单因素是45次训练；78次表示把11个条件扩展也全部跑完。** P3不是当前主线结论的先决条件。新对照套件的9次匹配训练、9次学习率试验、搜索、独立评价和统计另计。单目标是经验参照，未经精确求解不能称为理论上界或最优解。

## 2．数据与启动门槛

2026-10-16已逐条载入并核对V2数据：validation=500，test=20，OOD=20，stress=20。具体manifest哈希及26配置展开表见 `result/analysis/proof_selection_review_20261016/experiment_inventory.json`。PPO主线验证固定分层50实例；49个独立诊断实例按既有频率使用，不参与选模。新对照套件采用第6节单独冻结的验证划分。

当前20实例test/OOD/stress属于开发规模。配置中的publication档为test=1000、OOD=500、stress=500，但不能仅凭配置项宣称已完成大规模评价。需要更大样本时，先指定独立版本目录、物化固定文件并冻结manifest，再让全部比较方法使用同一数据版本；不能覆盖已有正式数据后混用旧结果。

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiments --suite all --seed 11 --preflight-only
.\.venv\Scripts\python.exe -m scripts.run_00_smoke
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.t
```

旧V1四份清单按 `.gitattributes` 保持CRLF换行，565个历史文件通过原归档哈希校验。既有V2文件可直接读取，本轮实验不需要重新生成数据；`scripts.prepare_data_v2`仍保留归档校验。

## 3．P1：核心训练矩阵

每次训练保存原始轨迹统计、固定验证子集身份、选模事件、best/last及最终best测试。Universal/no_graph每100 episodes验证，其他单目标每40 episodes验证；均每20 episodes更新。

| 角色 | 配置 | 每run episodes | seeds | runs |
|---|---|---:|---|---:|
| Universal主方法 | [configs/default.json](../configs/default.json) | 2000 | 11/23/37 | 3 |
| Flow单目标参照 | [configs/e1/single_flow.json](../configs/e1/single_flow.json) | 1000 | 11/23/37 | 3 |
| Cost单目标参照 | [configs/e1/single_cost.json](../configs/e1/single_cost.json) | 1000 | 11/23/37 | 3 |
| Workload单目标参照 | [configs/e1/single_workload.json](../configs/e1/single_workload.json) | 1000 | 11/23/37 | 3 |
| 取消图消息传播：node MLP pooling | [configs/ablations/no_graph.json](../configs/ablations/no_graph.json) | 2000 | 11/23/37 | 3 |
| Flow单目标疲劳中性 | [configs/ablations/neutral_flow.json](../configs/ablations/neutral_flow.json) | 1000 | 11/23/37 | 3 |
| Cost单目标疲劳中性 | [configs/ablations/neutral_cost.json](../configs/ablations/neutral_cost.json) | 1000 | 11/23/37 | 3 |
| Workload单目标疲劳中性 | [configs/ablations/neutral_workload.json](../configs/ablations/neutral_workload.json) | 1000 | 11/23/37 | 3 |

图配对为Universal ↔ no_graph；疲劳配对为三个full单目标 ↔ 相应neutral。其余变量由 [ablation_protocol](../configs/ablation_protocol.py) 校验。

按现有通用入口串行执行三个种子，任一失败停止：

```powershell
foreach ($experimentSeed in @(11,23,37)) {
    & .\.venv\Scripts\python.exe -m scripts.run_experiments --suite all --seed $experimentSeed
    if ($LASTEXITCODE -ne 0) { throw "Core experiment failed: seed=$experimentSeed" }
}
```

该命令包含全部8个角色，不再另启动相同Universal/full单目标。批次记录在 `result/runs/experiment_seed*_startup/launch.json`，可从每个job的run_name定位实际模型。

## 4．P2：主张对应的单因素实验

以P1 Universal为共同基线，新增以下7个Universal配置。每个2000 episodes、三种子，共21次训练。

| 对照问题 | 配置 | 需要报告的边界 |
|---|---|---|
| bounded全局目标编码 | [configs/controls/bounded.json](../configs/controls/bounded.json) | 仅变全局目标编码，目标、尺度、奖励保持匹配 |
| GAE lambda=.97 | [configs/controls/lambda097.json](../configs/controls/lambda097.json) | 与lambda=.95/.99同预算；诊断MC对齐不能替代训练结果 |
| GAE lambda=.99 | [configs/controls/lambda099.json](../configs/controls/lambda099.json) | 与lambda=.95/.97同预算；报告所有训练种子与失败 |
| OWA q=.25及其冻结尺度 | [configs/controls/owa_q025.json](../configs/controls/owa_q025.json) | 使用自身冻结尺度；公共q=.5坐标与原生坐标分别报告 |
| OWA q=.75及其冻结尺度 | [configs/controls/owa_q075.json](../configs/controls/owa_q075.json) | 使用自身冻结尺度；比较总劳动、最大个人劳动与集中程度 |
| raw Flow表示与其冻结尺度 | [configs/controls/raw.json](../configs/controls/raw.json) | 同时采用对应冻结Flow尺度，测试的是既定目标表示协议；不能称为仅改归一化公式 |
| Variance劳动目标及其冻结尺度 | [configs/controls/variance.json](../configs/controls/variance.json) | 保留实际个体负荷，公共坐标重算OWA/Variance；两种原生Q不能直接比较 |

```powershell
$controlConfigs = @(
    'configs/controls/raw.json', 'configs/controls/variance.json',
    'configs/controls/bounded.json', 'configs/controls/lambda097.json',
    'configs/controls/lambda099.json', 'configs/controls/owa_q025.json',
    'configs/controls/owa_q075.json'
)
foreach ($experimentSeed in @(11,23,37)) {
    & .\.venv\Scripts\python.exe -m scripts.run_experiments --config $controlConfigs --seed $experimentSeed
    if ($LASTEXITCODE -ne 0) { throw "Control experiment failed: seed=$experimentSeed" }
}
```

用验证协议比较候选超参数，测试用于确认预先声明的比较。不同q/目标的交付先按各自原生Q选出，再在公共目标空间评分；若重新挑候选须另起实验标签。

## 5．P3：条件扩展的全部现有配置

以下覆盖其余11个PPO配置；仅为需要相应交互/机制结论时的扩展清单，当前并无证据要求全部启动。

| 目的 | 配置 | 每run episodes | 三种子runs |
|---|---|---:|---:|
| raw × bounded | [configs/controls/raw_bounded.json](../configs/controls/raw_bounded.json) | 2000 | 3 |
| raw × OWA q=.25 | [configs/controls/raw_owa_q025.json](../configs/controls/raw_owa_q025.json) | 2000 | 3 |
| raw × OWA q=.75 | [configs/controls/raw_owa_q075.json](../configs/controls/raw_owa_q075.json) | 2000 | 3 |
| raw × 单目标OWA q=.25 | [configs/controls/raw_single_owa_q025.json](../configs/controls/raw_single_owa_q025.json) | 1000 | 3 |
| raw × 单目标OWA q=.5 | [configs/controls/raw_single_owa_q05.json](../configs/controls/raw_single_owa_q05.json) | 1000 | 3 |
| raw × 单目标OWA q=.75 | [configs/controls/raw_single_owa_q075.json](../configs/controls/raw_single_owa_q075.json) | 1000 | 3 |
| raw × 单目标Variance | [configs/controls/raw_single_variance.json](../configs/controls/raw_single_variance.json) | 1000 | 3 |
| 单目标OWA q=.25 | [configs/controls/single_owa_q025.json](../configs/controls/single_owa_q025.json) | 1000 | 3 |
| 单目标OWA q=.75 | [configs/controls/single_owa_q075.json](../configs/controls/single_owa_q075.json) | 1000 | 3 |
| 单目标Variance | [configs/controls/single_variance.json](../configs/controls/single_variance.json) | 1000 | 3 |
| Variance × bounded | [configs/controls/variance_bounded.json](../configs/controls/variance_bounded.json) | 2000 | 3 |

与P2相同，通过 `scripts.run_experiments --config <所选配置...> --seed <种子>` 执行。原始raw单目标OWA(q=.5)与主线excess单目标不是重复配置；差异需要明确标为Flow表示协议。

## 6．求解器比较、交付与原始池指标

### 6.1 新对照套件：先验证，再冻结正式预算

统一入口为 [comparison_suite.py](../scripts/comparison_suite.py)，方法身份为`comparison_suite_v1`；合同、远端操作和本地验证分别见[实施契约](comparator_implementation_spec.md)、[启动说明](comparator_remote_startup.md)、[验证记录](comparator_verification_20261017.md)。四种新方法均为本项目适配，DCAN不称为原论文完整复现。

| 实验 | 运行矩阵 | 当前支持/产物 |
|---|---|---|
| 匹配PPO / DCAN适配版 / Rule-DDQN | 各3训练种子11/23/37，共9次；每次350411次实际env.step | 从头训练；PPO/DCAN主交付三次取优，并列greedy诊断；DDQN贪心单次 |
| 学习率试验 | 每方法3个学习率，seed11，共9次；每次35041步 | 在调参集选择学习率，正式训练重新开始 |
| NSGA-II-LS / MOEA/D | 种群16/32/64，在7例×4偏好×seed11调参，共168个600秒请求 | 冻结种群后，各跑21例×4偏好×3搜索种子曲线 |
| MO-ALNS v2 | 套件内另建alns_reference，重新机制验收及21例×4偏好×3搜索种子曲线 | 使用当前修复版与原生环境；读取实际选中预算，不默认认定600秒充分 |
| 三种搜索的合理预算 | 每方法先600秒，必要时完整矩阵扩展至1800秒 | 同时保存初始化、10/30/60/180/600及可用的1800秒端点；平台和机制门槛共同决定预算 |
| 独立确认 | 7例×13偏好×3训练/搜索种子，每臂273份交付；11臂共3003份 | 5个学习交付臂，3个充分搜索臂，3个PPO交付时限搜索臂 |
| 正式test/OOD/stress | 参数、权重与预算冻结后，各实例66偏好 | `all`只生成formal_manifest；满足预算和独立改善门槛后另行formal-evaluate |

500例validation中，按固定哈希和七类压力分出21调参、21选模、7确认实例，三者互斥。调参使用三端点和等权共4偏好；正式训练在20/40/60/80/100%实际步数档位，用21选模实例×13偏好选best。确认集不参与选模、学习率或搜索参数选择。这里的21例选模不是P1的50例主验证；两套协议分别保存身份。

350411步来自旧PPO实际交互日志的预算标定；旧权重不充当新匹配PPO。所有实际env.step（含WAIT）计费，关闭forced compression；档位和总预算边界截断保留自举。等交互不等于等优化或等算力，另报梯度更新、样本重用、训练/验证墙钟和设备。末档仍明显改善时生成统一2B扩展清单，不自动执行。

搜索从请求开始计费，包含初始化、代理、修复、解码与档案更新。完整候选须在截止前完成，短时限无可行解时明确记无交付。NSGA-II-LS至少完成5代并实际执行局部搜索，MOEA/D至少5次完整扫描，ALNS至少20个完整邻域及4次权重更新；各压力组至少90%请求达到对应机制门槛。平台要求无新增成功、各压力组平均Q增益≤.002、总体90分位增益≤.01；按[具体选择器](../scripts/comparison_suite.py)结合尾段平台取最早合格档位。1800秒仍未达标时保留`not_established`，不把最大预算改名为充分预算。

远端先预检：

```powershell
python -u -m scripts.comparison_suite --stage preflight --output-dir result/analysis/comparison_suite_remote_v1
```

预检成功后，在同一远端、同一源码和输出目录执行：

```powershell
python -u -m scripts.comparison_suite --stage all --remote-run --output-dir result/analysis/comparison_suite_remote_v1 --workers 8 --training-workers 20
```

默认8个搜索进程、20个训练环境worker，学习任务串行用GPU，确认时延串行测量。重跑只跳过身份匹配的完整请求；中断训练从零开始。更换源码、配置、数据、硬件或并行宽度时另建输出目录。完整test/OOD/stress入口与恢复规则见[远端说明](comparator_remote_startup.md)。

PPO短时限取同实例、偏好的三个新PPO模型各自三次取优交付中位时延，再取跨模型中位数。每个模型重复计时三次，保持同一组三条轨迹，不能额外做九次选优。短时限搜索若只返回初始化解，就报告这一结果；充分搜索质量仍按其独立预算判断。无需战胜PPO才能成为有效对照。

主表按实例—偏好交付对齐，另报13点交付HV和全部付费档案HV。各臂先在同一实例—偏好内平均训练/搜索重复，再配对；两类种子不解释为相同实验单位。7例确认按整个实例pooled bootstrap；21例调参允许分层实例bootstrap。独立改善门槛针对搜索相对自身初始化最优解的Q增益区间，不要求优于PPO；原始启发式候选及初始化成本也保留供核查。

### 6.2 原有入口与其他参照

| 实验 | 运行范围 | 当前边界 |
|---|---|---|
| P1 PPO Universal | 3个best；最终test每实例66偏好×3次 | 与新匹配PPO分别标记；其OOD/stress包装见第8节 |
| 启发式 | test/OOD/stress各一次确定性物理轨迹 | 后续按公共66偏好评分；完整网格包装待补 |
| 随机策略 | 按预先声明的算法种子和采样预算评价 | 单偏好CLI可用；完整网格包装待补，不冒用DDQN单次或PPO三次协议 |
| 单目标参照 | P1三个单目标×3训练种子 | 自动测试各自one-hot端点；与Universal对应端点配对 |

单独的[MO-ALNS研究脚本](mo_alns_v2_implementation_20261017.md)保留历史λ=.95/seed11固定PPO参照，包含固定概率LNS诊断；新套件则重新训练三个PPO模型。两种研究的权重、训练重复和耗时不混用。`scripts.mo_alns_benchmark`仍可读取预算已冻结的manifest批量求解；仓库默认600秒配置是验证起点，旧“每偏好300次评价”不再是正式预算。评价次数上限仅用于明确声明的回归或诊断。

原有`analysis.mo_alns_analysis`分析原始候选池；新`analysis.comparator_study`已提供套件确认集的交付配对、完成覆盖、共同成功Q/目标、HV及报告。前者不能替代后者，后者也不自动覆盖所有PPO消融和正式split的汇总。

## 7．机制、协议与效率实验

| 编号 | 实验 | 数据/执行方式 | 验收和解释 |
|---|---|---|---|
| M1 | 偏好响应 | 每个P1/P2 Universal best的66点×3次完整原始行与selected行 | 各目标端点变化、固定比例路径的单调比例、完成覆盖、共同成功集；不把非单调直接判为实现错误 |
| M2 | OWA机制 | P1 OWA与P2 Variance/q条件的实际worker_loads | 均值劳动、最大劳动、方差、OWA及集中项；非支配检查按完整目标 |
| M3 | 三次取优增益 | 固定同一checkpoint，使用已有三次原始请求 | 对比每次单采样及三次交付，累计全部求解成本；按实例聚类，不能把三次作为训练重复 |
| M4 | 选模口径敏感性 | 同一训练路径的完整validation行，离线分别应用per_sample/best_of_3 | 报告重选episode；要测试该候选必须有其权重。与改变学习率控制后的新训练区分 |
| M5 | 新旧训练控制协议因果比较（条件扩展） | 需另建per_sample完整训练配置，与P1匹配3种子/预算 | 新增3次Universal训练；当前26配置未包含此臂。只有要声称新选模改善训练时才需要 |
| M6 | critic/GAE诊断 | 对同一冻结策略完整轨迹重放lambda与MC | 解释估计误差/方差，保持原始PPO轨迹；不能据此推出哪个训练lambda更优 |
| M7 | 批量/投影性能 | 相同实例、状态、动作、设备及精度；reference_v8与phase_batched_v1、投影等价检查 | 先验输出等价，再多次计时；报告环境、观测、推理、IPC、端到端耗时 |
| M8 | 最佳与最后权重稳定性 | 固定验证选出的best，以及预先声明的last诊断 | last结果不能反向替换按验证选择的主结果；记录失败模式和成本尾部 |

M3/M4可首先在现有旧协议记录上复算，作为历史诊断。当前 [verify_best_of_three_protocol](../analysis/verify_best_of_three_protocol.py) 固定读取三条lambda旧记录；它不自动扫描新实验，也不补出缺失中间权重。

## 8．目前需补齐的执行/统计环节

以下区分原有入口的缺口与新套件已支持的范围：

1. **P1及全部消融的OOD/stress完整网格**：`eval.py --preference-set final_test`当前要求`--dataset test`，并且只接受PPO；这些历史/主线run仍需正式CLI包装。新套件的`formal-evaluate`已支持其冻结学习/搜索臂在test/OOD/stress的66偏好评测，但不自动接收任意P1/P2模型，也不在`all`中运行。
2. **启发式/随机的公共偏好网格**：普通CLI可以跑单偏好；Universal的`--preference-set`不接受它们。启发式可重用同一物理轨迹在各偏好评分；随机要声明每偏好采样与种子预算，需对应包装和公共汇总。
3. **全部消融及正式split的交付汇总**：`analysis.ablation_analysis`仍按原始sampling_repeat配对，旧Pareto/ALNS分析仍针对原始池；新套件已实现3003份独立确认交付的完整性和配对统计。将它推广到全部消融、不同目标条件及正式split仍需对应适配，不能直接把确认报告当作正式测试报告。
4. **公共坐标与跨条件配对**：raw、Variance、不同q的原生分数不同，应从实际Flow/Cost/worker_loads重算公共指标，保留首次training_*及原生selected标记。公共评分不应重写训练身份或偷偷重选。
5. **未来中间权重复核**：当前通常保留best/last。若要反事实测试其他验证episode，须预先保存所需候选权重；只有验证日志不足以重建权重。
6. **独立压力与规模泛化**：challenge_v1已提供100例压力测试和80例重构倍率配对实例，但全量模型评价尚待执行，其他学习/搜索方法仍需challenge适配。现有OOD/stress及这些倍率档位不自动构成跨订单/机器/工人规模证据；相应规模实验需独立定义并冻结，目前未计入训练矩阵。

新套件已完成实现及小实例回归；余下缺口和远端正式性能分别验收，不以预检或smoke替代。

## 9．已有产物与新协议的完成状态

2026-10-16本地产物核对发现以下4条schema12/shared-head正式训练：均2000 episodes，best/last与最终原始测试CSV存在，均为per_sample_v1。

| 历史实验 | 旧选模best episode | 新口径离线重选episode | 当前用途 |
|---|---:|---:|---|
| OWA/log2/lambda=.95/seed11 | 1900 | 1800 | 历史基线与固定权重三次取优复算 |
| OWA/log2/lambda=.97/seed11 | 400 | 1900 | 同上 |
| OWA/log2/lambda=.99/seed11 | 700 | 1400 | 同上 |
| Variance/log2/lambda=.95/seed11 | 1800 | 本轮未重选 | 旧协议目标对照 |

前三行复算记录来自 `result/analysis/best_of_three_protocol_verification/`。其中原best的三次取优测试Q分别为.262400/.324509/.313682；这些测试值属于原best权重，不属于表中重选episode。记录已注明重选候选未测试。新协议还可能改变学习率衰减路径，因此P1/P2仍需用新协议完整训练；现有数据不能把这些run标记为已完成。

截至2026-10-17，本机的新对照证据限于预检、定向测试与小实例smoke，详见[验证记录](comparator_verification_20261017.md)。尚无新套件正式训练、充分搜索或独立确认成绩；合成3003行报告测试只验证统计链路。旧两套2-episode smoke同样仅用于集成。远端队列需另核实，不能据本地缺少记录断言远端没运行。

## 10．产物与统计验收清单

每个训练run核对：保存配置及runtime身份、全部训练预算、validation_subsets及manifest哈希、train/update/validation日志、best/last权重哈希、checkpoint选择事件、原始和selected验证/测试行。PPO主线50实例验证应有1950请求/650交付；20实例Universal测试应有3960请求/1320交付；单目标对应150/50及60/20。新套件每模型每次选模为21×13=273交付，PPO/DCAN原始819请求，DDQN原始273请求；独立确认每臂跨三个种子共273交付。文件格式按各自入口解释，suite请求JSON不冒称主线CSV。

每个实验主表至少报告：完成率（Universal最差偏好及整体分别列）、成功条件Q、Flow/Cost/OWA/Variance、个体总量/最大值、物理违规、失败原因、总求解和推理耗时。跨目标比较加公共坐标；疲劳中性加原始疲劳暴露。

每个训练种子是独立实验单位。先报告每seed结果，再均值/标准差及配对效应；实例/偏好/三次请求相关，不能把3960行当作3960个独立训练重复。配对检验须保留共同实例、区分失败覆盖，并说明多重比较；三个seed的精确双侧Wilcoxon即使全同方向，其最小p为2/2^3=.25，不能承诺三seed必定得到p<.05。

现有主线支持经验比较；没有精确最优值时报告相对基线差距，不能标为最优性gap。数学证明、集成测试和训练性能分别给出证据。
