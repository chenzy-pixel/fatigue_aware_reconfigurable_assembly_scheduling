# 瓶颈与重构时间外推评估协议

本协议依据2026-10-17确认的修改计划实施。两个新增集合仅用于冻结模型后的独立评估。模型、超参数和搜索预算不得根据该集合调整；若用于修改方法，应将其转为开发数据并另留独立确认数据。

## 数据与证书

`test_hard`按机器、重构、工人、疲劳和高到达压力分层，每类20个实例，共100个。每类成本结构配额为balanced_cost 8、downtime_dominant 4、labor_dominant 4、fixed_cost_dominant 4。沿用既有生成范围，severity=1。按固定候选顺序接纳当前确定性启发式已完整、安全、时域内完成且动作证书重放通过的实例；每个“压力×成本结构”子层最多200个候选。配额不足明确失败，不放宽准入。由此得到的是固定参考启发式可解的条件子集。

`reconfiguration_extrapolation`使用重构瓶颈底座，20组，成本配额8/4/4/4；基础实例只需在1.4档通过结构和必要条件检查，选择不依赖启发式是否成功。每组倍率1.4、1.6、1.8、2.0，共80例。1.6是训练范围边界，1.8与2.0属于倍率因子的外推。局部扰动和量化意味着个别最终时长仍可能落在训练时长范围内。

每组只采样一次未量化拆装参数，各档独立乘倍率并量化；其他物理字段一致。全部档位保留，分别记录预检查、诊断和证书状态。未获证书标为unknown，不将启发式失败解释为不可行。外推统计同时给出全部固定组和四档共同获证书子集。

候选实例、未获证书的完整诊断动作、参考轨迹和接受原因保存在各集合`candidates/`目录；成功证书保存动作、调度和重构日志。证书按当前每个状态的合法动作重放，检查完整完成、合法调度、疲劳安全、时域及日志一致性。实际压力指标是诊断，不额外按策略成绩筛选。

套件使用独立loader和manifest，保留既有V1/V2数据契约。实例、证书、候选台账、配置、模板、生成与物理协议均有哈希绑定。新数据与已发布数据按排除实例身份的物理内容去重。冻结manifest不可覆盖；修改生成源码或套件配置后使用新的版本目录。构建中断只恢复身份及内容一致的候选。

## 入口

在项目根目录使用已安装依赖的Python运行。默认配置[challenge_suite.json](../configs/challenge_suite.json)，数据为`data/instances/challenge_v1`，冻结入口为[suite.json](../data/manifests/challenge_v1/suite.json)。已有冻结数据可直接审计和评价；`build`用于初次构建，同身份重跑时审计既有数据，不覆盖冻结集合。

```powershell
.\.venv\Scripts\python.exe -m scripts.challenge_suite --stage build --generation-workers 4
if ($LASTEXITCODE -ne 0) { throw 'Challenge build failed' }
.\.venv\Scripts\python.exe -m scripts.challenge_suite --stage audit
if ($LASTEXITCODE -ne 0) { throw 'Challenge audit failed' }
.\.venv\Scripts\python.exe -m scripts.challenge_suite --stage smoke --output-dir result/audits/challenge_v1_smoke
```

smoke固定选test_hard每类首例及外推首组四档，共9例；每例三个端点与等权偏好、每偏好三次，共108条PPO轨迹，并运行9条独立启发式轨迹后重评分为36个请求。默认加载计划中指定的schema12/log2、OWA、lambda=.95、seed11历史best及原训练配置。保留其历史per_sample选模身份，新增交付使用best_of_3_v1。它是链路验收，不是泛化性能证明。

正式全量评估命令（本次仅交付入口，不自动执行）：

```powershell
.\.venv\Scripts\python.exe -m scripts.challenge_suite --stage evaluate --policy ppo --output-dir result/evaluations/challenge_v1_seed11
if ($LASTEXITCODE -ne 0) { throw 'PPO challenge evaluation failed' }
.\.venv\Scripts\python.exe -m scripts.challenge_suite --stage evaluate --policy heuristic --output-dir result/evaluations/challenge_v1_seed11
if ($LASTEXITCODE -ne 0) { throw 'Heuristic challenge evaluation failed' }
.\.venv\Scripts\python.exe -m scripts.challenge_suite --stage report --output-dir result/evaluations/challenge_v1_seed11
```

更换模型时同时指定`--training-config <原训练配置>`及`--checkpoint <权重>`，不同模型或训练种子使用独立输出目录。支持`--suite-manifest <suite.json或单集合manifest.json>`与`--device cpu|cuda`。每个请求原子保存，缺失或损坏请求在相同评价身份下重算；模型/配置/数据/偏好/采样种子/源码或执行硬件身份改变时拒绝混用。

当前CLI的`--policy`仅支持`ppo`和`heuristic`。`challenge_suite`使用独立loader，不把新名称注册为`eval.py --dataset`选项；`comparison_suite --stage formal-evaluate`也不会自动包含这两个集合。DCAN、DDQN与搜索方法的challenge适配仍需单独接入。远端运行需同步冻结实例、manifest、证书及候选台账，并另行提供所选PPO权重和原训练配置；本套件默认历史seed11模型与对照套件的新匹配PPO身份不同。

## 统计与产物

PPO正式评价固定66偏好，每请求三次采样。全部样本使用既有final_test种子派生，并复用项目原生成功优先、原生Q最小、repeat最小的选解规则。每个原始样本保存动作、调度、重构和完整指标。失败Q为1，失败原因及真实目标另列；原始请求和交付统计分别保留。启发式每实例只执行一次物理轨迹，再按同一目标尺度重评分。

PPO执行复用原有ParallelEpisodeRunner，按原训练配置的validation_parallel_envs聚批独立请求，并应用原配置torch_num_threads。新增capture_trajectories参数仅在明确启用时返回完整动作/调度/重构日志，默认调用保持原行为。批次规模、执行方式和源码哈希写入评价身份；每个实例—偏好的随机数流仍独立派生。并行请求耗时用于记录本次执行，不直接解释为单请求独占硬件延迟。

评价身份记录原始训练配置、checkpoint哈希和完整checkpoint元数据；不重新标定测试尺度。`raw_samples.csv`、`selected_samples.csv`及`report.json`包含压力、成本、倍率、可行性、逐偏好及低cost区域分组，三目标、Q、完成率、失败原因和机制指标。成对比较仅在共同成功请求上计算质量差，另列完整分母的完成率及失败填1的Q。

bootstrap固定统计seed20261017、重复10000次，以实例为单位；外推以基础组为单位，将四档和全部偏好绑定重采样。区间条件于给定训练模型，各训练种子单独报告。更多偏好或三次采样不增加独立实例数。20个基础组与100个压力实例提供首批证据，不自动构成统计充分性保证。

## 实施验证

实现回归覆盖旧路径实例字节、证书重放/非法证据拒绝、倍率配对、unknown保留、配额不足、台账及产物篡改、恢复身份、原生三次取优和分组统计单位。完整数据发布前逐一重放全部已获证书实例、检查目标配额和数据重叠、校验V1/V2既有文件哈希。实际运行结果见`data/manifests/challenge_v1/audit.json`及`result/audits/challenge_v1_smoke/report.json`，以实际产物为准。

### 已保存的首批证据（2026-10-17）

以下数字来自已保存的[构建审计](../data/manifests/challenge_v1/audit.json)和[smoke报告](../result/audits/challenge_v1_smoke/report.json)，不是全量66偏好性能评价。

| 集合 | 已物化 | 构建审计 |
|---|---:|---|
| test_hard | 100例 | 107个候选中接纳100例，全部有可重放可行证书 |
| reconfiguration_extrapolation | 20组×4档=80例 | 34例observed_feasible、46例unknown；全部80例保留 |

构建审计记录全部已获证书重放通过，1129个既有V1/V2保护文件未变。unknown表示参考启发式没有给出可行证书，不是不可行判定；压力标签也不保证每例都触发疲劳屏蔽或工人竞争。

9例、4偏好的smoke包含108条PPO物理轨迹，以及9条启发式物理轨迹重评分得到的36行；报表共144条原始样本行、72份交付。PPO为36/36交付成功，启发式为24/36，后者12份失败均为外推请求的horizon失败。全部原始行通过活动约束检查且无工程采样截断；“无采样截断”不等于“任务全部完成”。这些是单个历史模型的链路证据，不能据此宣称稳定泛化优势。

保存的`result/audits/challenge_v1/challenge_tests.xml`记录65项通过、1项跳过。后续模型比较仍需固定模型与预算、完成全量评价，并按训练种子和独立实例/基础组报告。
