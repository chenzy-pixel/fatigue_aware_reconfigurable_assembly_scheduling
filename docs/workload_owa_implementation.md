# 几何 OWA 实际劳动目标

主线配置默认使用 `workload_objective.mode="geometric_owa"`、`q=0.5`。
目标计算全体工人的 DIS/INS 劳动分钟，包括零负荷工人；按负荷降序排列并使用
归一化几何权重。H=6、q=0.5 时权重为 `(32,16,8,4,2,1)/63`。

## 运行与指标契约

新训练使用 observation schema 11 和 `flow/cost/workload` 偏好。raw Flow 对应
workload v5 奖励，Flow excess 对应 workload v6 奖励；累计训练回报仍为
`Delta progress - Delta quality - failure penalty`，gamma=1。

`workload_objective_value` 是终局实际负荷的活动目标，
`committed_workload_objective_value` 是已承诺阶段时长的活动目标。
中途的实际负荷沿用阶段完成或物理失败结算后计入的口径；分派时立即计入承诺
负荷。成功终局两个向量和目标值一致。`worker_load_variance` 始终是真实方差。
结果另保存带工人身份的 `worker_loads`、`worker_load_total`、`worker_load_max`。

`variance` 模式作为显式对照保留，使用原方差尺度。当前 schema 10 checkpoint
通过保存配置及原契约运行；缺少目标模式只能在经过验证的 schema 10 快照中
解释为 variance。schema 11 与 schema 10、不同模式、q、尺度均不能相互加载或续训。
网络参数维度相同不代表目标或观测兼容。

## 冻结尺度

本次标定使用既有固定分层50个验证实例和确定性 `HeuristicPolicy`，实际参考配置
保存于 `result/audits/workload_owa/calibration/reference_config.json`。
其中49条成功且物理合法，全部成功/失败结果及逐工人负荷已保存。
三个 q 均从同一批参考负荷计算，未使用测试集。

| q | workload 尺度，成功轨迹均值 |
|---|---:|
| 0.25 | 21.655756896165062 |
| 0.50 | 21.03145448655653 |
| 0.75 | 20.20910214991848 |

raw/excess 使用各自的原 Flow 尺度1089.15/368.3143，Cost均为353.27。
新清单包含完整参考行、成功数量、目标定义、原尺度来源、配置和数据哈希。
读取时重新计算 OWA 及均值，并检查清单文件和内容哈希。

重新标定使用 `scripts/build_workload_normalization.py`。默认参考为当前显式方差
对照配置；需要通过 `--manifest-tag` 选择新的不可覆盖清单名称，不能改写本次
已冻结清单。新参考证据写入输出目录的同名tag子目录，开始计算前检查清单和
证据文件冲突。若要复用本次参考，指定保存的参考配置、`--reuse-reference` 和
`--reference-directory result/audits/workload_owa/calibration`，并选择新的清单名称。

## 正式实验命令

以下 PowerShell 命令在仓库根目录执行。先运行启动检查：

```powershell
& .\.venv\Scripts\python.exe scripts/run_workload_training.py --preflight-only --device cuda
```

默认 q=0.5、Flow excess，顺序训练 workload 单目标、Universal 和方差对照：

```powershell
& .\.venv\Scripts\python.exe scripts/run_workload_training.py --device cuda --seed 11
```

单目标 q 敏感性实验：

```powershell
& .\.venv\Scripts\python.exe scripts/run_workload_training.py --experiment workload --q 0.25 --device cuda --seed 11; if ($LASTEXITCODE -eq 0) { & .\.venv\Scripts\python.exe scripts/run_workload_training.py --experiment workload --q 0.5 --device cuda --seed 11 }; if ($LASTEXITCODE -eq 0) { & .\.venv\Scripts\python.exe scripts/run_workload_training.py --experiment workload --q 0.75 --device cuda --seed 11 }
```

`--experiment universal` 运行指定q的Universal；`--flow-mode raw` 选择原始Flow。
两种 Flow 启动路径的 OWA 与方差对照均采用 shared head；原 `e1/single_variance`
配置仍保留其原网络头，raw 启动器使用新增的 shared head 方差对照配置。
`--seed` 支持既有算法种子11/23/37/53/71。启动器在子实验失败时停止并保存状态。
workload单目标及方差对照沿用1000 episode预算，Universal沿用其原配置预算。

五种子正式实验（q=0.5，每个种子均运行三个实验，任一失败即停止）：

```powershell
foreach ($experimentSeed in 11,23,37,53,71) { & .\.venv\Scripts\python.exe scripts/run_workload_training.py --device cuda --seed $experimentSeed; if ($LASTEXITCODE -ne 0) { throw "workload experiment failed: seed $experimentSeed" } }
```

## 统一评价与支配诊断

策略执行采用训练配置和观测契约。跨目标质量与HV比较使用显式统一评价配置，
默认几何OWA q=0.5及同一冻结尺度；各q自身目标下的指标另行报告。
旧方差策略按旧观测执行，再由完整实际负荷重算OWA。
历史行缺少负荷向量时，只有完整终止重构日志和工人名单才允许重建。

`analysis/workload_dominance.py` 提供统一重算和逐工人支配诊断：比较同实例、
同物理条件下成功合法的候选，保存跨方法/同方法的比例及替代见证；失败保留
在完成率分母内。合并分析要求相同偏好及采样预算，保持工人身份对应。
严格单调性是目标函数性质；PPO的近似输出仍需实际检查支配率。

## 回归与冒烟入口

`scripts/check_workload_legacy.py --verify` 验证真实schema10 checkpoint的旧轨迹。
`scripts/audit_workload_history.py` 使用生产目标模块重建174条成功轨迹及5536次阶段增量，
重现82对支配见证，保存源文件哈希及完整见证标识。
`scripts/run_workload_smoke.py --device cuda` 运行五组40 episode集成实验，并校验
每组两次PPO更新、固定验证子集、checkpoint重载和回报恒等式。
冒烟产物为 `result/audits/workload_owa/smoke/runs.json` 及对应运行目录。
冒烟的失败和完成率作为覆盖证据保存，不能用于策略收敛或参数优劣结论。

本轮五组均完成40 episode、两次PPO更新和checkpoint重载。五组训练回报恒等式
最大误差均为0，采样中的物理约束检查通过；采样失败均记录为horizon。

| 冒烟实验 | 训练成功/40 | 两轮固定子集验证成功/候选数 |
|---|---:|---:|
| 方差对照 | 32 | 8/10 |
| OWA q=0.25 | 32 | 8/10 |
| OWA q=0.5 | 29 | 9/10 |
| OWA q=0.75 | 28 | 10/10 |
| Universal q=0.5 | 34 | 33/40 |

checkpoint确定性重载校验在首个验证实例上五组均以horizon终止，回报及物理检查
仍通过。这说明本轮只证明集成链路可运行，不能将采样完成率解释为收敛效果。
四个单目标采用相同5实例、1偏好、1次采样，20候选统一重算结果及支配报告保存于
`result/audits/workload_owa/smoke/single_objective_pool.json`；Universal的15候选单独保存于
`universal_pool.json`。两份报告均使用冻结raw Flow、q=0.5坐标及真实工人名单。

MO-ALNS使用一个固定验证实例及8次评价预算完成小规模接入验证，成功且物理合法，
活动OWA值为7.866666666666666，真实方差为3.882222222222223；产物位于
`result/runs/workload_mo_alns_smoke_20261011`。

2026-10-12全量回归为830 passed、2 skipped、1 failed；收尾改动的补充回归为
219 passed、1 skipped（与全量测试有重叠，不作去重总数相加）。
唯一失败是`test_legacy_data_is_preserved_and_loadable`：现有
`data/manifests/ood/manifest.json`与HEAD字节相同，其SHA256为
`e1dd636df2fb71be56bff50392366351eb5850a77ff041c0a3450d24c5c62689`，
原归档预期为`dda85d56609cf6dd69087ad9e79251b854cb3193cff106b9a1e481b7272209b6`。
这是HEAD已有的数据清单不一致，本轮未修改数据或归档哈希表。
测试XML、标定、冒烟与代码状态汇总位于`result/audits/workload_owa/verification.json`。

真实schema10checkpoint在两实例上的110/158次决策、每步观测/掩码/奖励哈希、
动作及终局三指标与修改前记录完全一致。独立评价产物还校验保存的
`evaluation_config.json`、评价尺度和配置哈希，保留原训练身份及质量；重复评分
仍保留首次`training_*`，旧/新模型可在同一冻结OWA坐标下分析。
