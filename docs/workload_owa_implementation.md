# 几何 OWA 实际劳动目标

主线配置默认使用 `workload_objective.mode="geometric_owa"`、`q=0.5`。
目标计算全体工人的 DIS/INS 劳动分钟，包括零负荷工人；按负荷降序排列并使用
归一化几何权重。H=6、q=0.5 时权重为 `(32,16,8,4,2,1)/63`。

## 运行与指标契约

新训练使用 observation schema 12 和 `flow/cost/workload` 偏好。raw Flow 对应
workload v5 奖励，Flow excess 对应 workload v6 奖励；累计训练回报仍为
`Delta progress - Delta quality - failure penalty`，gamma=1。

`workload_objective_value` 是终局实际负荷的活动目标，
`committed_workload_objective_value` 是已承诺阶段时长的活动目标。
中途的实际负荷沿用阶段完成或物理失败结算后计入的口径；分派时立即计入承诺
负荷。成功终局两个向量和目标值一致。`worker_load_variance` 始终是真实方差。
结果另保存带工人身份的 `worker_loads`、`worker_load_total`、`worker_load_max`。

`variance` 模式作为 schema12 下的显式对照保留，使用冻结方差尺度。运行时要求明确的 workload 定义，配置和 checkpoint 只能使用 schema12。不同模式、q、尺度和全局编码不能交叉加载或续训；网络参数维度相同不代表目标或观测兼容。

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

默认 q=.5、excess Flow，配置和验证协议见 [主线实验](experiment_protocol.md)。

```powershell
python -m scripts.run_experiments --config configs/e1/single_workload.json configs/default.json configs/controls/single_variance.json --seed 11 --preflight-only
python -m scripts.run_experiments --config configs/e1/single_workload.json configs/default.json configs/controls/single_variance.json --seed 11
python -m scripts.run_experiments --config configs/controls/single_owa_q025.json configs/e1/single_workload.json configs/controls/single_owa_q075.json --seed 11
```

多种子逐个启动并在失败时停止：

```powershell
foreach ($experimentSeed in 11,23,37,53,71) { python -m scripts.run_experiments --config configs/default.json --seed $experimentSeed; if ($LASTEXITCODE -ne 0) { throw "experiment failed: seed $experimentSeed" } }
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

`scripts/audit_workload_history.py` 使用生产目标模块重建174条成功轨迹及5536次阶段增量，
重现82对支配见证，保存源文件哈希及完整见证标识。
历史 OWA 集成检查运行了五组40 episode实验（原源码见[整理快照](project_cleanup.md)），并校验
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
旧V1四份清单的换行已恢复为归档约定的CRLF，565个历史文件通过原哈希校验。归档数据读取与当前模型契约分别检查；当前运行时明确拒绝非schema12配置。
