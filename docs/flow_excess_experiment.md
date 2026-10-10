# Flow 比例下界抵扣独立实验

本实验按 2026-10-08 确认方案实现。默认配置使用 `raw_v1`，新增实验使用
`excess_proportional_lb_v1`。Universal、单目标 Flow 和 MO-ALNS 分别继承
现有网络、固定数据、正式训练与搜索预算。完整多种子训练由以下入口单独运行。

## 运行入口

修改后的 Flow/Universal 在远端启动时，使用[远端启动与最新性能结果](flow_excess_remote_startup.md)
中的独立入口。下面的完整实验入口用于 raw/excess 配对对照。

```powershell
python scripts/run_flow_excess_smoke.py
python scripts/benchmark_flow_excess.py
python scripts/summarize_flow_excess_acceptance.py
python scripts/run_flow_normalization_experiment.py
python scripts/run_flow_single_flow_experiment.py
python scripts/run_flow_excess_mo_alns.py
```

前三条为实现验收和证据汇总。后三条分别执行 seed11 的 Universal、单目标 Flow、MO-ALNS
匹配 raw/excess 实验，写入各自带时间戳的运行目录。正式种子可通过
`--seed` 指定；Universal 入口默认 2000 轮，单目标默认 1000 轮。
本次交付只运行短程 PPO 和小预算 MO-ALNS。

汇总入口读取已保存的测试 XML。以下命令生成本次两份测试报告（Windows 使用短
临时路径避免超长文件名），随后可运行 `summarize_flow_excess_acceptance.py`：

```powershell
python -m pytest -q --tb=short --basetemp .p --junitxml result/analysis/flow_excess_integration_20261008/pytest.xml
python -m pytest -q test/test_flow_excess.py test/test_flow_excess_analysis.py test/test_flow_excess_replay.py test/test_mo_alns.py test/test_v8_protocol.py test/test_repository_contract.py test/test_evaluation_agent_contract.py test/test_checkpoint_evaluation.py test/test_pareto_analysis.py test/test_v8_pareto_analysis.py test/test_runtime_performance.py test/test_evaluation_refill.py --basetemp .f --junitxml result/analysis/flow_excess_integration_20261008/related_tests.xml
```

冒烟的 `--phase audit/train/replay/decisions` 可分开执行；`--raw-run` 可复用已完成
的 raw 冒烟。`replay` 使用本工作区已有的三组 seed11 历史运行和
`result/analysis/flow_excess_replay_20261008/traces/`，复核全部 180 条轨迹。
冒烟配置为 CPU、hidden_dim=32、1 层、2 个 episode、每段 16 步、2 个训练 worker；
验证为 1 实例 × 13 偏好 × 1 repeat，最终冒烟为同一验证实例 × 66 偏好 × 1 repeat。
它验证完整评估入口，不构成独立测试集成绩。

## 状态与数学契约

reset 使用 `estimate_processing_ticks(op,machine)` 对所有兼容机器取最小整数 tick，
保存只读数组及整实例整数和。在 `_start_processing` 固化
`OperationRuntime.planned_duration_ticks`；初始完成事件和日志使用该计划工期。
失败日志剪裁保留计划工期和 `planned_end`。

每次读取状态重新计算：

\[
D(t)=\sum_{op\in DONE}p_{min}(op)+
\sum_{op\in PROCESSING}p_{min}(op)\frac{t-S_{op}}{p_{op}},\qquad
F_{ex}(t)=F(t)-D(t)+\text{flow penalty}.
\]

实现先求 DONE 的整数 tick 和，再求加工中的分数项，最后换算分钟；不维护
抵扣账本。遍历所有工序，分数项只来自正在加工的工序。分母只读取固化计划工期，
不读被剪裁的结束时间，不重新估计加工速度。成功终局检查抵扣整数和等于
整实例下界。浮点非负检查只允许 1e-8 的数值误差。

订单内工序串行且 `p_min <= p`，因此罚项前 excess 连续且单调不减。
最快加工不增加 excess，慢机器按额外加工损失增加，等待按完整时间增加。
成功时 `F_ex = F − Σp_min = 总等待时间 + Σ(p − p_min)`。
真实失败保留部分加工信用和原 Flow 罚项；失败罚项造成终止跳变。
工程截断保留物理状态和部分信用，没有失败罚项，PPO 使用最终观测自举。

在**同尺度、γ=1、成功轨迹**上，完成时一次抵扣与比例抵扣总回报相同。
失败轨迹及含失败概率的随机策略排序可能改变。excess 与 raw 的有界归一化
使用不同尺度时，标量化偏好的含义也会改变。完成事件的奖励集中不代表
累计目标重复计数。

## WAIT 契约

`project_wait_state(...,settle_terminal=True)` 在隔离副本上调用实际
`_execute_wait` 和 `_resolve_terminal_or_deadlock`，覆盖零时间阶段交接、事件、
疲劳恢复、成功、horizon 及下一状态 mask 死锁。投影不调用 `observe`，避免递归；
运行态、日志、计数、缓存和负荷账本均独立。测试用投影前后完整序列化状态
验证原环境不变。

excess 的 WAIT Flow 特征为 `(目标 after − 目标 before)/368.3143`，保留完整罚项。
同一投影复用于订单裕量；非法或终止 WAIT 的两项裕量特征使用 0。
生产候选时间特征和输入维度保持当前 schema 10。raw 特征沿用原语义。

## 冻结尺度与身份

| 模式 | Flow | Cost | Variance | 奖励身份 |
|---|---:|---:|---:|---|
| raw_v1 | 1089.15 | 353.27 | 2.2629 | single_stage_progress_quality_failure_v3 |
| excess_proportional_lb_v1 | 368.3143 | 353.27 | 2.2629 | single_stage_progress_quality_failure_v4 |

excess 清单为 `configs/manifests/flow_excess_scales_20261008.json`，SHA256：
`17367f71e7a529ee504feb2b531a6a44fdf7458f323797ca9be33508723ce61b`。
它包含固定验证集的 50 条基线记录，其中 49 条成功；未舍入均值为
`368.31428571428575`。验证指标来源 SHA256 为
`41f02e4274bf3727923c11925e595e2f2e09c7b07be8b6e950e3e89c9d9f3fa1`，
子集文件 SHA256 为 `f16d9036615349b9b43b026f2d777697957b813e3e1220ff65d17d3bb0a56270`。
加载器校验内容哈希、数量、唯一实例、均值及尺度；Cost/Variance 继承原出处。

`flow_mode` 缺失按 raw 解析，未知值拒绝。模式、清单哈希和奖励身份进入
runtime manifest、network spec、checkpoint 契约和结果出处。
缺少 Flow 身份的兼容旧 checkpoint 只解释为 raw；raw/excess 交叉加载拒绝。
已有旧观测版本限制继续执行。

## 训练、报告与评估

`flow_time_objective`、`total_flow_time`、`censored_flow_time` 保持原始 Flow。
新增 `flow_excess_objective`、`flow_processing_lower_bound`、`flow_lower_bound_credit`、
`reward_objective_flow`、`flow_mode`。偏好质量、奖励重建、checkpoint 选优和
MO-ALNS 候选/档案选解读取配置指定目标；`quality_score` 使用原始目标的固定参考。

| 评估 | Flow 字段 | 固定尺度 | 参考点 |
|---|---|---|---|
| 主 HV | flow_time_objective | (1089.15,353.27,2.2629) | (1,1,1) |
| 补充 HV | flow_excess_objective | (368.3143,353.27,2.2629) | (1,1,1) |

两者均使用 `J/(s+J)`。同一实例完整可行调度的 Pareto 集不因减去常数下界改变；
有界变换及尺度改变后 HV 数值一般不同，两个口径各自汇总。
`analyze_dual_flow_runs` 验证各训练身份，再比较相同实例、物理环境、66 偏好、
repeat 和采样种子。`analyze_candidate_flow_runs` 接受单目标或 solver 结果，校验
共同实例/物理口径及各方法内部的采样或搜索预算；PPO repeats 与 solver 搜索预算
分别记录。历史成功结果缺少 excess 时，仅从哈希校验后的实例整数下界推导；
已有 excess 字段必须与该下界一致。工程截断不进入正式 HV。

## 验收数据与适用范围

证据目录：`result/analysis/flow_excess_integration_20261008/`。

| 核验 | 结果 |
|---|---|
| 固定验证集启发式 | 50 实例，49 成功；两模式动作、物理调度一致 |
| 历史状态复核 | 180 轨迹、20,522 状态；最大误差 4.55e-13 |
| 完整合法 WAIT 投影 | 12,652 状态；失败终止比例 0 |
| WAIT 特征 p50/p95/p99/max | 0.003891 / 0.033293 / 0.060187 / 0.146410 |
| WAIT 投影耗时 p50/p95/p99/max | 4.19 / 7.21 / 10.57 / 42.10 ms |
| 短程 PPO | 两模式训练、保存/加载、13 偏好验证、66 偏好评估与双 HV 均完成 |
| 真实 PPO 决策 | 两模式各 64 步；GAE、自举和 4 次更新的数值有限 |
| MO-ALNS 冒烟 | 两模式各 1 实例/Flow 端点/8 次搜索评估，重放与双口径一致 |
| 环境耗时 | 单实例三次配对，raw 28.90 ms/步，excess 37.50 ms/步，约增加 30% |

环境耗时包括 `step` 与观测构建，受同时运行的测试和主机负载影响；完整 WAIT
副本投影有可测成本，不能据此预测 GPU 训练吞吐。PPO 前 32 步/episode 的 WAIT
审计另存 `ppo_legal_wait_projections.csv`，两模式各 64 个合法状态；本批未出现
终止投影。horizon、死锁和成功终止由边界测试覆盖。

离线事件奖励方差见 [离线回放](flow_excess_replay_findings.md) 及
[归一化采用判断](flow_normalization_decision_20261008.md)。它与实际 PPO 决策网格不同。
真实奖励、GAE 优势和更新统计分别记录于 `ppo_decision_steps.csv`、
`ppo_decision_updates.csv`；其中梯度方差是**固定更新前参数下，各真实 transition 的
PPO loss 梯度协方差 trace（总体分母 n）**，包含策略、价值和熵项，优势按 PPO
方式标准化。它描述该短段样本，不等于完整训练的随机 minibatch 梯度方差。
这些运行不证明收敛后改善 Flow、HV、优势方差或梯度方差。

本次真实决策审计的四段结果分别列出如下，样本是未收敛策略的 32 步前缀，
不用于显著性或收敛收益比较。

| 模式/段 | 决策奖励方差 | GAE 优势方差 | 更新前每 transition 梯度协方差 trace |
|---|---:|---:|---:|
| raw/0 | 7.0633e-5 | 3.8953e-4 | 0.13292 |
| raw/1 | 3.5358e-5 | 8.3543e-5 | 0.07415 |
| excess/0 | 8.7541e-5 | 8.0342e-4 | 0.11941 |
| excess/1 | 4.3710e-5 | 2.4146e-4 | 0.09744 |

相关回归组 176 项通过；随后补充的无证书 WAIT 隔离测试所在文件 20 项通过。
全仓测试为 635 通过、2 跳过、1 失败：
`test_data_v2::test_legacy_data_is_preserved_and_loadable` 检测到现有
`data/manifests/ood/manifest.json` 与归档哈希不一致。本次实现没有修改该数据。
现有 raw golden 观测/动作/奖励测试、运行优化和评估 lane refill 回归通过。

主要证据文件还有：`fixed_validation_metrics.csv`、`legal_wait_projections.csv`、
`native_historical_replay.csv`、`smoke_runs.json`、`matched_hv/`、`mo_alns_matched/`、
`environment_timing.csv` 和 `pytest.xml`。训练运行目录保存实际 rollout 决策、
验证明细、checkpoint、config、runtime manifest 和来源哈希。
`acceptance.json` 汇总验收结果、证据 SHA256 和当前核验代码 SHA256；训练执行时的
代码出处以各运行自带 provenance 为准。
