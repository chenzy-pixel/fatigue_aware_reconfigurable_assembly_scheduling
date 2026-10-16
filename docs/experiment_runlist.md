# 当前主线完整实验执行清单

核对日期：2026-10-16。以当前配置、训练/评价入口、固定V2数据及本地正式产物为依据。本文件是执行与验收清单；这里列出的正式实验本轮尚未启动。远端状态未读取，历史运行不自动算作新协议完成项。

当前三种子启动脚本：[run_three_seeds.ps1](../scripts/run_three_seeds.ps1)。[逐实验命令清单](experiment_three_seeds.md)逐行列出26个实验；默认只运行Universal，支持指定多个实验、预检和dry-run。

## 1．冻结条件与顺序

统一主线：schema12/log2、HGNN shared-head、excess Flow、OWA(q=.5)、gamma=1、GAE lambda=.95、best_of_3_v1。默认尺度为 `(368.3143,353.27,21.03145448655653)`。当前先执行 **11、23、37** 三个训练种子；配置允许的53、71留待后续扩展，以下运行数按三个种子展开；三次采样属于同一模型的解码预算，不是三个独立训练种子。

执行顺序：数据/协议预检 → 每种配置小规模链路检查 → 核心矩阵 → 单因素对照 → 固定best的独立比较与泛化 → 多种子汇总。可先用seed11检查工程，再完成同一预先声明矩阵的其余种子；不能因seed11的测试表现删选不利种子或条件。

| 优先级 | 当前配置数 | 三种子训练数 | 总episodes | 用途 |
|---|---:|---:|---:|---|
| P1 核心 | 8 | 24 | 30,000 | 主方法、单目标参照、图消融、疲劳对照 |
| P2 单因素 | 7 | 21 | 42,000 | 支持raw/excess、OWA、编码、lambda及q相关实证主张 |
| P3 条件扩展 | 11 | 33 | 45,000 | 研究交互作用或单目标OWA机制时再执行 |
| 合计 | 26 | 78 | 117,000 | 现有全部PPO训练配置的完全展开 |

**核心加单因素是45次训练；78次表示把11个条件扩展也全部跑完。** P3不是当前主线结论的先决条件。MO-ALNS、独立评价、诊断和离线统计另计。单目标是经验参照，未经精确求解不能称为理论上界或最优解。

## 2．数据与启动门槛

本轮已逐条载入并核对当前V2数据：validation=500，test=20，OOD=20，stress=20。具体manifest哈希及26配置展开表见 `result/analysis/proof_selection_review_20261016/experiment_inventory.json`。主验证固定分层50实例；49个独立诊断实例按既有频率使用，不参与选模。

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

| 实验 | 运行矩阵 | 当前支持/产物 |
|---|---|---|
| PPO Universal | P1的3个best；test/OOD/stress；每实例66偏好×3次 | test在训练末尾自动完成；OOD/stress全网格入口见第8节缺口 |
| MO-ALNS | 3种子×3数据集=9个数据集作业；每实例66偏好 | 已有manifest/批次入口；每偏好最多300次候选评价，输出最佳候选及档案 |
| 启发式 | 3个固定数据集各运行一次确定性轨迹 | 无训练；物理轨迹不随偏好变，后续在相同66偏好下评分 |
| 随机策略 | 5算法种子×3数据集，按明确偏好和3次采样 | 单偏好CLI已支持；完整66偏好比较需要网格包装 |
| 单目标参照 | P1三个单目标×3种子 | 自动测试各自one-hot端点；与Universal对应端点配对 |

MO-ALNS正式批次：

```powershell
.\.venv\Scripts\python.exe -m scripts.mo_alns_benchmark --manifest configs/baselines/mo_alns_manifest.json --output-dir result/analysis/mainline_mo_alns
```

产出 `candidates.csv`、`pareto_archive.csv`、`search_log.csv`、`operator_statistics.csv`、schedule/reconfigurations与summary。manifest的ppo_candidates字段只是来源记录，不会自动执行PPO或完成比较。

三次PPO和300次MO-ALNS候选评价是不同预算，应同时报告端到端求解时间、全部候选/解码次数及交付质量。不能称为相同计算预算。若论文需要同预算结论，另固定双方可执行的预算网格并统计质量—耗时曲线；当前MO-ALNS有8条初始化规则，最小候选预算为8，不能直接设为3并当作可运行对照。PPO的其他预算也须定义独立协议，不能冒用best_of_3_v1。

现有 `analysis.mo_alns_analysis` 用于原始候选池，输入PPO原始final_sampled/instance_metrics和MO-ALNS candidates。主结果还需“每实例—偏好一份交付”的完成率/Q/目标/交付HV；不以原始池指标替代。

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

这些限制来自当前源码核对，清单不把尚未支持的命令当作可运行入口：

1. **Universal OOD/stress完整网格**：`eval.py --preference-set final_test`当前明确要求`--dataset test`，并且只接受PPO。底层 `evaluate_preference_grid_parallel`可接收数据集，但尚缺完整正式CLI包装。应补齐split、网格、三次种子、原始/selected输出与provenance后执行。不要直接把`test`命令的split改成`ood`并假定成功。
2. **启发式/随机的公共偏好网格**：普通CLI可以跑单偏好；Universal的`--preference-set`不接受它们。启发式可重用同一物理轨迹在各偏好评分；随机要声明每偏好采样与种子预算，需对应包装和公共汇总。
3. **交付层比较统计**：`analysis.ablation_analysis`按原始sampling_repeat配对；`analysis.pareto_analysis`和`analysis.mo_alns_analysis`以全部原始候选池计算完成率、Q和HV。新主表需要按(instance_id, preference_key)配对selected交付，两个模型允许选中不同repeat；原始请求仍承担安全和预算核查。该统计出口需补齐，现有训练summary可验证各run自身交付口径。
4. **公共坐标与跨条件配对**：raw、Variance、不同q的原生分数不同，应从实际Flow/Cost/worker_loads重算公共指标，保留首次training_*及原生selected标记。公共评分不应重写训练身份或偷偷重选。
5. **未来中间权重复核**：当前通常保留best/last。若要反事实测试其他验证episode，须预先保存所需候选权重；只有验证日志不足以重建权重。
6. **大规模与真正规模泛化**：现有OOD/stress是其固定分布条件，不能自动命名为跨规模泛化。要提出跨订单/机器/工人规模主张，需独立定义、物化并冻结相应算例；目前未计入78次训练或9个求解器作业。

这几项是后续执行完整论文比较的工作项。本轮完成证明、文档与清单核对，不将它们记为已实现或已跑完。

## 9．已有产物与新协议的完成状态

本地发现以下4条schema12/shared-head正式训练：均2000 episodes，best/last与最终原始测试CSV存在，均为per_sample_v1。

| 历史实验 | 旧选模best episode | 新口径离线重选episode | 当前用途 |
|---|---:|---:|---|
| OWA/log2/lambda=.95/seed11 | 1900 | 1800 | 历史基线与固定权重三次取优复算 |
| OWA/log2/lambda=.97/seed11 | 400 | 1900 | 同上 |
| OWA/log2/lambda=.99/seed11 | 700 | 1400 | 同上 |
| Variance/log2/lambda=.95/seed11 | 1800 | 本轮未重选 | 旧协议目标对照 |

前三行复算记录来自 `result/analysis/best_of_three_protocol_verification/`。其中原best的三次取优测试Q分别为.262400/.324509/.313682；这些测试值属于原best权重，不属于表中重选episode。记录已注明重选候选未测试。新协议还可能改变学习率衰减路径，因此P1/P2仍需用新协议完整训练；现有数据不能把这些run标记为已完成。

本轮本地未发现满足新协议的完整正式训练记录；前次两套2-episode smoke用于集成验证。远端队列需另核实，不能据本地缺少记录断言远端没运行。

## 10．产物与统计验收清单

每个训练run核对：保存配置及runtime身份、全部训练预算、validation_subsets及manifest哈希、train/update/validation日志、best/last权重哈希、checkpoint选择事件、原始和selected验证/测试行。50实例验证应有1950请求/650交付；20实例Universal测试应有3960请求/1320交付；单目标对应150/50及60/20。

每个实验主表至少报告：完成率（Universal最差偏好及整体分别列）、成功条件Q、Flow/Cost/OWA/Variance、个体总量/最大值、物理违规、失败原因、总求解和推理耗时。跨目标比较加公共坐标；疲劳中性加原始疲劳暴露。

每个训练种子是独立实验单位。先报告每seed结果，再均值/标准差及配对效应；实例/偏好/三次请求相关，不能把3960行当作3960个独立训练重复。配对检验须保留共同实例、区分失败覆盖，并说明多重比较；三个seed的精确双侧Wilcoxon即使全同方向，其最小p为2/2^3=.25，不能承诺三seed必定得到p<.05。

现有主线支持经验比较；没有精确最优值时报告相对基线差距，不能标为最优性gap。数学证明、集成测试和训练性能分别给出证据。
