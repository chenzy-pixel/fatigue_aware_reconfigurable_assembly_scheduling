# E1 / E2–E2.7 完整失败审查与重构报告

审查日期：2026-08-22
审查分支：`codex/e2-7-e1-warmstart-safe-gate`（基线 HEAD `b1491a2`）
审查性质：E1–E2.6 代码、配置、提交历史、训练日志、评估产物和 checkpoint 协议审查；在审查结论之上实现 E2.7 的 E1 warm-start 与安全 commit/defer 重构。未把 smoke 或单 seed 开发结果表述为论文结论。

## 一、结论先行

E1 成功、E2 旧版“有 accepted 但 Pareto 坍缩”、E2.1–E2.6 全部未正式通过，并不是一个单一 PPO bug，而是一条连续的设计退化链：

1. **E1 的成功是真实但范围有限的成功。** Seed 11 的 accepted checkpoint 在 20 个固定 validation 实例上完成率 100%，按 `canonical_bounded_quality_v1` 相对启发式的平均质量 gap 为 **-12.67%**，确实达到“优于启发式约 10%”的量级。但它仍是单 seed、20 个 validation 实例的开发证据，不等于五 seed、test/OOD/stress 的论文结论。
2. **旧 E2 的 `accepted_checkpoint.pt` 只证明 canonical 单点通过旧门禁，不证明 Pareto 通过。** 旧 E2 继续使用 `balanced_guarded_v7`，验证和 checkpoint promotion 固定看 `w=(0.5,0.3,0.2)`；它没有检查 22 个偏好下的多样性、非支配点数或偏好响应。因此“有 accepted”与“Pareto 好”本来就是两个不同命题。
3. **旧 E2 的 Pareto 坍缩有完整实证。** 在同为 22 个候选的公平预算下，E2 在 OOD/stress/test 上的平均 hypervolume 分别比 E1 低 `0.0256/0.0214/0.0231`，每个数据集均为 1 胜 4 负；每实例唯一轨迹从 E1 的约 `21.9` 降到 E2 的 `3.57/4.06/4.82`，三个目标的前沿跨度也全面缩小。
4. **E2.1/E2.2 失败于可行性，不是 Pareto 门禁太苛刻。** 两者 2000 episode 后固定 validation 的最佳完成率都只有 95%，始终没有进入 quality phase。Replay 明确定位为工人直接偏好项破坏全匹配：`w=(0,0.2,0.8)` 时在 worker bottleneck 实例上进入 advance/defer 循环；关闭 worker 直接偏好或允许 worker recovery 后恢复 100% 完成。E2.2 只改生产层级 decoder，所以没有触及真正故障点。
5. **E2.3 修复了主验证可行性，却未通过全偏好安全。** Canonical validation 为 100%，最终 full-grid 440 个候选中只有 430 个完成，10 个都因 `horizon` 截断，集中在 `w_flow<=0.2`。其多样性反而是后续版本里最好的一档：`12.65` 条轨迹、`12.5` 个目标向量、`4.9` 个非支配点；它失败在安全/活性，不是不可控。
6. **E2.4 为安全切断了最关键的偏好通道。** 它把 commit/defer 改成 `preference_conditioning=false` 的 state-only gate，full-grid 440/440 全部安全，但多样性降到 `6.85/6.35/3.5`，低于 `8/8/4` 门槛。它达到了“安全”，代价是基本失去 commit/defer 层面的偏好可控性。
7. **E2.5 的单调 flow boost 作用在已经饱和的 gate 上。** 三个 kappa 虽把平均 commit boost 从约 `0.195` 提到 `0.394/0.587`，但所有实验的 `base-defer -> final-commit` greedy flip 都是 **0**。在最终候选中，“存在合法生产 pair 的状态数”与“gate 选择 commit 的状态数”完全相等，说明 gate 只在没有合法 pair 时 defer；正向 commit boost 无法改变任何离散决策。三个 kappa 因而全部停在 controllability/diversity 门禁前。
8. **E2.6 是实现级的“零有效训练信号”。** 它只对“commit/defer 都合法且冻结 base gate 会 defer”的状态施加 counterfactual loss；实际三个 lambda 运行中 eligible count 全程为 0、counterfactual loss 全程为 0、flip count 为 0。三个配置的最终网络权重 SHA256 完全相同，所有 full-grid 结果也完全相同。`lambda=0.05/0.10/0.20` 实际上跑的是同一个普通 PPO 实验，校准变量从未进入有效梯度。
9. **E2.7 已按“从 E1 重建、以 E2.3 诊断”的路线实现，但尚未获得开发验收。** 新配置只允许从 E1 seed11 accepted checkpoint 严格加载 116 个共享参数；E2.3 last checkpoint 仅参与失败状态池 provenance，不加载权重。canonical `w0=(0.5,0.3,0.2)` 下 raw logits、概率和 value 已通过逐位 identity 测试。开发 checkpoint 只有在连续两次 440/440 full-grid、本地全部门禁以及 validation/test/OOD/stress 等预算 HV 全部严格超过 E1 时才会生成，文件名固定为 `development_accepted_pareto_checkpoint.pt`。

一句话总因果：**旧 E2 的偏好信号太弱而被基础策略忽略；E2.1 把偏好直接加到动作排序后又破坏 worker 活性；E2.3 只在 production 侧保留偏好后恢复可行性，但低 flow 偏好会过度 defer；E2.4 用“有 pair 就 commit”的安全 gate 消灭了过度 defer，也同时消灭了主要 Pareto 决策自由度；E2.5/E2.6 都在这个饱和 gate 上继续加正向 commit 补丁，因此没有可翻转的状态，也没有可学习的 counterfactual 样本。**

## 二、审查范围与证据

本次审查覆盖：

- Git 标签 `E1` 以后至 HEAD 的 11 个 E2.1–E2.6 相关提交；
- `configs/v7/` 中 E1、旧 E2、E2.1–E2.6 及校准配置；
- `environment/env.py` 的稠密增量 reward、可行性和 worker matching 逻辑；
- `agent/ppo/network.py` 的 preference encoder、direct ranker、production gate 和 E2.6 counterfactual head；
- `agent/ppo/agent.py` 的 PPO/GAE、counterfactual loss 和诊断汇总；
- `train.py` 的两阶段训练、Pareto full-grid、promotion 与 checkpoint 语义；
- `result/runs/` 的 summary、validation/update/train log、Pareto candidates/log 和 replay；
- `result/analysis/e1_e2_full/` 的五 seed、公平候选预算 Pareto 分析；
- E2.6 只读 calibration auditor；
- 原审查阶段聚焦 E2 的 9 个测试文件：**78 passed in 71.08s**；
- E2.7 新增专用回归覆盖 warm-start、canonical identity、层级等价、三目标单调、阶段冻结、shield 边界、reward identity、开发 checkpoint 语义与旧 checkpoint 严格加载。

测试全过说明接口、schema、符号和门禁实现没有明显单元级崩坏；它不能证明策略能学到有效 Pareto 响应。当前主要失败属于**机制与训练分布不匹配**，E2.6 另有明确的空 eligible 集设计失效。

## 三、E1 到底成功了什么

### 3.1 可以确认的事实

`result/audits/20260815_182548_c0_e1_validation/audit.json` 的 protocol-v2 审计全部通过：

| 方法 | validation 完成率 | canonical quality | 相对启发式 quality gap | fatigue CVaR |
|---|---:|---:|---:|---:|
| C0 | 100% | 0.33465 | -9.59% | 0.5953 |
| E1 seed 11 | 100% | 0.32293 | **-12.67%** | 0.4948 |

E1 的 bounded residual 机制是克制的：单调候选 ranker 保留主导地位，context 只通过受限 residual 做例外修正。因此它没有重写安全行为，只是在固定权重下改善候选排序。

### 3.2 不能过度解读的部分

- 上表只有一个算法 seed 和 20 个 validation 实例；
- 五 seed E1 summary 中 seed 71 的 `mean_relative_heuristic_gap_percent` 仍为正；
- 当前仓库没有一份 E1 五 seed、publication test/OOD/stress、统计显著性的最终论文表。

因此准确表述应是：**“E1 seed 11 在固定开发验证集上取得 accepted checkpoint，并按 canonical bounded quality 比启发式好 12.67%。”** 暂不能写成“E1 稳定普遍优于启发式 10%”。

## 四、旧 E2：为什么有 accepted 仍然坍缩

### 4.1 accepted 的语义被混淆了

旧 E2 配置虽然启用了 preference conditioning，但 checkpoint promotion 仍继承 `balanced_guarded_v7`。代码允许非 Pareto 模式在 feasibility→quality 的 `transition` 时直接写 `accepted_checkpoint.pt`；例如旧 E2 seed 23 的 accepted episode 是 30、accepted quality updates 为 0、原因是 `transition_anchor`。

因此：

- `accepted_checkpoint.pt` 存在：说明至少进入了 quality 阶段并留下了 canonical 安全锚点；
- Pareto accepted：应同时通过 440/440、安全、coverage、8/8/4、多目标响应和 canonical guard；
- 旧 E2 从未执行第二种语义。

后续 E2.3–E2.6 则明确禁止 transition 直接成为 accepted，必须先建立 full-grid Pareto baseline。所以直接比较“旧版有 accepted 文件、后续没有”也不完全公平：门禁定义已经变了。

### 4.2 坍缩的量化证据

五 seed、三个固定 split、每实例每 arm 都是 22 个候选：

| split | E1 HV | E2 HV | E2-E1 | E1/E2 唯一轨迹 | E1/E2 唯一目标向量 |
|---|---:|---:|---:|---:|---:|
| OOD | 0.4436 | 0.4180 | -0.0256 | 21.90 / 3.57 | 21.90 / 3.45 |
| Stress | 0.4298 | 0.4084 | -0.0214 | 21.94 / 4.06 | 21.94 / 3.94 |
| Test | 0.3564 | 0.3333 | -0.0231 | 21.89 / 4.82 | 21.89 / 4.63 |

三个 split 的 HV 都是 1 seed 胜、4 seed 负。Test 上 E2 的 normalized front span 只有 E1 的约 18%（flow）、17%（cost）、34%（variance）。这不是轻微退化，而是明显的策略条件坍缩。

### 4.3 原因

1. 旧 E2 从 E1 **完全重新训练**，不加载 E1 trunk/head，主动放弃了已验证的 canonical 能力锚点。
2. preference 只通过 embedding 进入 contextual scorer；E1 的主 monotone ranker 不随偏好改变，而 contextual residual 又被小 gate 约束。最终不同偏好的信号很容易被主 ranker 淹没。
3. checkpoint 只按 canonical preference 选择，会系统性偏向一个“平均上还行、对偏好不敏感”的策略。
4. scalarized dense reward 本身实现为严格 telescoping 增量，没有发现 reward identity 错误；问题不在奖励符号，而在偏好信号到离散动作的控制权太弱。

## 五、逐版本失败链

| 版本 | 实际状态 | 关键证据 | 直接原因 |
|---|---|---|---|
| E2（旧） | 有 accepted，Pareto 坍缩 | HV 三 split 全负；E2 唯一轨迹仅 3.57–4.82/22 | canonical-only promotion + 偏好只走弱 residual |
| E2.1 | 2000 ep，feasibility 未达到 | validation 最佳 95%，始终 feasibility；无 accepted | direct preference 同时作用 worker，破坏 matching/liveness |
| E2.2 | 2000 ep，feasibility 未达到 | 最佳仍 95%；gate-first 与 joint argmax replay 无分歧 | 只改 production hierarchy，没改 worker 根因 |
| E2.3 | 进入 quality，无 accepted | canonical 100%；full-grid 430/440 完成；12.65/12.5/4.9 | 低 flow 偏好过度 defer，10 个 horizon truncation |
| E2.4 | 安全，无 accepted | 440/440；6.85/6.35/3.5 | state-only、无偏好 gate 把安全和可控性一起锁死 |
| E2.5 k=1 | 校准停止 | 最终 8.45/7.60/3.40 | 目标向量和非支配数不过 |
| E2.5 k=2 | 校准停止 | 最终 9.20/8.55/3.40，flow Spearman +0.0175 | 非支配数不过且 flow 响应方向错误 |
| E2.5 k=3 | 校准停止 | 最终 6.40/5.95/3.25 | 多样性全面不足 |
| E2.6 λ=0.05/0.10/0.20 | 审计 `stopped` | eligible=0、loss=0、flip=0；三者权重 hash 相同 | counterfactual 训练集合为空 |

### 5.1 E2.1/E2.2 的 replay 因果证据

两版在同一个 `validation_worker_bottleneck_2000003` 上表现一致：

- `w=(0,0.2,0.8)`：82 步后进入 stall，0/15 orders 完成，约 30 次 matching deficit；
- `w=(0,0,1)`：正常完成 15/15；
- 改成 joint argmax：仍失败，说明不是 hierarchical decoder 选择方式；
- 放松 `require_full_matching`：仍失败；
- 允许 worker recovery：恢复 15/15；
- 关闭 worker direct preference、只保留 production direct：恢复 15/15；
- 只保留 worker direct preference：继续失败。

所以 E2.2 的“production hierarchical”修补方向从一开始就偏离故障域。

### 5.2 E2.3 的 10 个失败候选

最终 full-grid 的 10 个失败全部是 `termination_reason=horizon`，偏好点为：

- `0_0_1`：2 个；
- `0_0.2_0.8`：2 个；
- `0_0.6_0.4`：1 个；
- `0.2_0_0.8`、`0.2_0.2_0.6`、`0.2_0.4_0.4`、`0.2_0.6_0.2`、`0.2_0.8_0`：各 1 个。

所有失败都满足 `w_flow<=0.2`，且 fatigue margin 非负、schedule violation 为 0。故障是低 flow 权重下的活性/完工问题，而不是疲劳安全或非法调度。

### 5.3 E2.4–E2.6 的 gate 饱和证据

在最终候选集合中：

| 运行 | 有合法 production pair 的决策数 | gate commit 数 | base-defer→commit flip |
|---|---:|---:|---:|
| E2.4 seed 11 | 31,020 | 31,020 | 未启用 residual |
| E2.5 k=1 | 31,020 | 31,020 | 0 |
| E2.5 k=2 | 31,020 | 31,020 | 0 |
| E2.5 k=3 | 31,020 | 31,020 | 0 |
| E2.6 λ=0.05 | 31,020 | 31,020 | 0 |

这说明所有 defer 都是“无合法 pair 时被迫 defer”，而不是“commit/defer 都合法时主动等待”。E2.5 只允许高 flow 增加 commit logit，面对已经“可 commit 必 commit”的 base gate，天然不可能产生 greedy flip。

E2.6 又把 eligible 定义为 `commit_legal && defer_legal && base_margin < 0`。实际训练和评估中这个集合为空，所以：

- 50 个 update 的 counterfactual eligible 总和均为 0；
- counterfactual loss 非零 update 数均为 0；
- λ 三个候选的最终网络权重 SHA256 均为 `cdbcae5beff70c1e99a5da5ac6e12320d205ac2a657c17e590de0f04171c3c34`；
- 三个候选的 HV、canonical quality、diversity、Spearman 完全一致。

这是本轮审查里最明确的实现/实验设计失败：**超参数确实写进 config，但乘到的 loss 永远为 0。**

## 六、为什么“改了七千多行”没有形成有效进展

从 E2.1 到 E2.6 及其小修复共新增约 **7,422 行、删除 286 行**。复杂度主要增加在：

- 新动作语义和 gate；
- 逐版本 schema/provenance；
- full-grid promotion 与安全回滚；
- 大量诊断字段和测试；
- 校准协议与 auditor。

这些工作让失败“可观测、可审计”，但没有先验证最关键的机制前提：

1. 偏好改变是否能在相同状态上改变合法动作排序；
2. commit/defer 是否存在可控、又不破坏活性的自由度；
3. counterfactual eligible 集是否非空；
4. 新 loss 是否产生非零梯度并实际改变参数；
5. 同预算下多偏好训练是否仍有足够实例多样性。

结果是代码围绕门禁越来越完整，而策略自由度越来越少。E2.4 之后的工作本质上是在一个离散决策已经饱和的 gate 上调概率，没有改变 argmax 行为。

## 七、实验与工程层面的次要但重要问题

### 7.1 训练实例多样性下降 5 倍

E1 和旧 E2 的 2000 trajectories 对应 2000 个 unique online instances；E2.1 起固定 5 个偏好共享同一实例，2000 trajectories 只对应 400 个 unique instances。配对训练便于学偏好差异，但在不增加总 trajectories 的情况下，实例覆盖被压缩到 1/5。500-episode 校准更只有 100 个 unique instances。

这不是 E2.1 deadlock 的直接原因，但会降低泛化并放大对固定 validation 的适配。若保持 5 偏好配对，至少应把总 trajectory 预算扩到约 10,000，才能匹配 E1 的 2,000 个独立训练实例覆盖。

### 7.2 所有正式 E2 产物都是 dirty provenance

旧 E2、E2.1–E2.6 的 summary 均记录 `git.dirty=true`。虽然保存了 source hash 和 effective config hash，但没有随每个 run 保存可恢复的 source snapshot/patch，因此仅靠 commit 不能重建精确运行代码。这使版本间细小差异的论文级归因不够可靠。

### 7.3 旧 E2 schema 元数据不一致

五个旧 E2 run 的 `config.json` 都写 evaluation schema `4.2.0`，但 summary provenance 记录 `4.1.0`。分析脚本最终读到了可用结果，但元数据内部不一致，说明 run/checkpoint/evaluator 的版本边界没有完全封闭。

### 7.4 产物目录有重复嵌套

`v7_2000_e2_pareto_seed37/` 内又嵌套了一份 `v7_2000_e2_1_pareto_seed11/`；其 summary 与顶层副本 SHA256 完全相同。这不改变数值结论，但会给自动聚合带来重复计数风险。

### 7.5 `promotion_decision_reason` 只显示第一个失败门禁

代码按 safety→coverage→controllability→preference-response→counterfactual 的顺序给单一 reason。E2.6 summary 显示 `controllability_failed`，容易遮蔽它同时存在的 counterfactual coverage=0、flip=0、loss=0。完整审计必须看所有 pass 字段，不能只看 reason。

### 7.6 Pareto snapshot 在计算 HV/多样性时未先排除未完成候选

`_pareto_snapshot` 会先把所有候选的目标向量放进 HV、unique 和 nondominated 计算，再用 `all_safe` 阻止不安全候选 promotion。因此 E2.3 不会因无效解被错误 accepted，但其 reported HV/diversity 可能被未完成解污染。正式统计应与离线分析一致，先过滤未完成、截断和违反约束的候选。

## 八、建议的恢复路线

### P0：停止沿 E2.6 继续堆补丁

不应继续在 E2.6 上调 lambda、阈值或最大 scale。eligible 集为空时，任何 lambda 搜索都没有意义；E2.5 已按预注册规则停止，E2.6 auditor 也已返回 `stopped`，formal seed training 未授权。当前 E2.7 不是 E2.6 的增量校准，而是从 E1 accepted 权重重新建立的独立开发协议。

### P1：把 E1 固化为可交付主线

1. 冻结 E1 seed 11 checkpoint、effective config、数据 manifest、源码快照和 protocol-v2 audit；
2. 补齐 E1 五 seed 的 validation/test/OOD/stress；
3. 报告 canonical quality、完成率、fatigue CVaR、推断时间、均值±标准差与配对 Wilcoxon；
4. 在论文中把 E1 表述为固定权重主方法，把 E2 暂列为扩展实验而不是已完成贡献。

### P2：重新定义 E2 的最小机制，不从 E2.6 续补

建议从 E1 canonical 策略出发设计一个可 warm-start 的 preference adapter，而不是从头训练整网：

1. E1 trunk/head 在 canonical `0.5/0.3/0.2` 保持行为锚定；
2. preference adapter 只在**经过安全/活性 mask 后的合法动作集合**内改变排序；
3. worker 侧禁止 flow/cost 直接破坏 matching，继续采用 E2.3/E2.4 的 safe worker variance 原则；
4. production commit/defer 不再使用“无偏好、可 commit 必 commit”的饱和 gate，而应建立一个带 liveness shield 的可控等待动作；
5. 用 KL/distillation 或 canonical behavior cloning 约束 canonical anchor，避免每个版本从零重新寻找可行策略。

### P3：先做机制单元实验，再跑 PPO

在任何 2000-episode 正式训练前，必须有一个 20 实例固定 state bank，检查：

- 每个目标至少存在一定比例的同状态 preference-induced action flip；
- `w_flow` 增大时 flow proxy 的选择方向单调；
- worker safe mask 与 matching invariant 永不被偏好绕过；
- commit/defer eligible coverage 至少 20/20，而不是训练结束才发现为 0；
- auxiliary loss 非零、对应 head 梯度非零、不同 λ 在第一个 update 后参数 hash 必须分叉；
- canonical preference 的动作与 E1 anchor 一致率达到预设下限。

只有这些通过，才值得进入 PPO。

### P4：恢复公平的训练预算与验收顺序

建议门禁顺序：

1. Reward identity 与 mask/invariant；
2. 440/440 completion、0 violation、fatigue safe；
3. 两次连续 full-grid 通过 8/8/4 和三个 Spearman 方向；
4. canonical quality 相对 E1 不退化超过 1%；
5. 五 seed、held-out test/OOD/stress 下 HV 相对 E1 equal-budget 有稳定提升；
6. 通过后才生成真正名为 `accepted_pareto_checkpoint.pt` 的文件。

同时将 checkpoint 命名拆开：

- `feasibility_anchor_checkpoint.pt`
- `canonical_accepted_checkpoint.pt`
- `full_grid_safe_checkpoint.pt`
- `accepted_pareto_checkpoint.pt`

避免再把“存在 accepted 文件”误解为“Pareto 通过”。

## 九、E2.7 重构落地状态

### 9.1 初始化与能力锚点

- 新实验名：`v7_e2_7_e1_warmstart_safe_gate_v1`，不修改 E2.3–E2.6 配置和历史产物；
- `--warm-start-checkpoint` 与严格续训 `--initial-checkpoint` 互斥；
- 唯一允许的权重来源为 `result/runs/v7_2000_e1_seed11/accepted_checkpoint.pt`；
- 加载时要求 E1 116 个 state key 全部存在且形状一致，任何缺失、多余映射或形状不一致立即失败；optimizer 始终重新创建；
- 运行目录写出 `warm_start_mapping.json`，记录源路径、checkpoint SHA256、network SHA256、加载数量、新参数清单、optimizer 未恢复和 canonical identity 协议；
- E2.3 update200 last checkpoint 只记录 SHA256 和十个失败单元来源，明确标记 `e2_3_weights_loaded=false`。

### 9.2 安全且可控的 commit/defer

E2.7 保留 E1 flat pair/defer head，不再把 preference embedding 拼入 E1 scorer。对合法 pair logits `l_i`，基础 commit log-mass 使用 `logsumexp(l_i)`；偏好适配器只给全部合法 pair 共同增加中心化有界残差。因此在 canonical `w0` 下残差严格为 0，raw flat logits 和 softmax 均与 E1 相同；非 canonical 下只改变 commit 总概率，不破坏 pair 条件分布。

三个 gate 系数采用非负参数化，flow 偏离 canonical 只能增加 commit margin，cost/variance 偏离只能降低 margin，残差由 `tanh` 限制在配置上界内。反事实辅助损失固定在同一批 safe dual-legal 状态上同时计算 flow/cost/variance 三个极端 anchor；状态池不足 64、eligible 不足、loss 为零或 gate 梯度为零时在 PPO 前停止。

环境 shield 对每个生产 defer 生成证书，记录 `wait_ticks`、`remaining_work_lower_bound_ticks`、deadline slack、risk 和 reason。规则为：无 commit 时保留唯一 defer；零时间 worker handoff 允许；dual-legal 正等待必须指向确定的下一状态事件；若 `wait + lower_bound + 1 > remaining_horizon` 则屏蔽；仍合法且接近边界的等待加入单独 `defer_risk_shaping`。训练/并行汇总在 reward identity 中同时剥离 feasibility shaping 与 defer-risk shaping，误差门限保持 `1e-8`。

### 9.3 训练、诊断和 checkpoint 门禁

- 2000 trajectories 对应 2000 个独立 online instances，每实例只采样一个 preference；
- 200 update 分为 `40 gate / 80 production-pair / 80 safe-worker-variance`，GNN trunk 全程冻结；
- production 阶段只开放中心化 pair 残差与 production 排序头，worker 阶段只让 variance 进入 worker 排序；flow/cost 不进入 worker 决策；
- E1 validation 与 E2.3 十失败单元共同建立固定状态池，smoke 实测可收集 256 个 safe dual-legal 状态，其中每类最多 128；
- 每 5 update 回放十失败单元，每 20 update 运行 20×22 full-grid；HV、多样性与非支配统计先过滤未完成、截断、疲劳/调度违规候选；
- full-grid 本地门禁通过后，才额外运行 test/OOD/stress 各 20×22，并与 E1 seed11 等预算 HV 逐 split 严格比较；
- 任一门禁失败不生成通用 `accepted_checkpoint.pt`。连续两次 full-grid 和全部 held-out 门禁通过后，只生成 `development_accepted_pareto_checkpoint.pt`，元数据固定 `development_scope=single_seed_development`、`formal_eligible=false`。

### 9.4 新的审计产物

每个 E2.7 run 会生成或更新：

- `warm_start_mapping.json`：E1 参数映射、hash、新参数和 fresh optimizer；
- `safe_dual_legal_state_pool.json`：E1 validation / E2.3 失败单元状态池与双方 checkpoint provenance；
- `e2_3_failure_replay.csv`：十失败单元逐 update 回放；
- `e2_7_gate_flip_report.json`：dual-legal coverage、flow-cost / flow-variance 各自的 flip rate、单调违规；两组都必须达到 5%；
- `e2_7_defer_shield_report.json`：候选数、屏蔽数、原因、最大 risk；
- `e2_7_full_grid_report.json`：最近一次 20×22 全门禁结果；
- `e2_7_heldout_comparison.json` 与 `e2_7_heldout_candidates.csv`：validation/test/OOD/stress 等预算 HV 比较与候选明细。

正式协议会在相应诊断触发后生成这些产物；若本地 full-grid 门禁未就绪，held-out 报告仍会落盘并明确记录未执行原因。Smoke 只生成其实际触发的 warm-start、状态池和 shield 报告。这些是协议与实现产物，不是成功结论。只有真实 2000-trajectory seed11 运行完成并生成开发 accepted 文件，才能报告 E2.7 开发验收通过。

### 9.5 实现验证结果

- 全量回归：`311 passed, 1 skipped`，覆盖 E1/E2 历史 checkpoint 严格加载及旧 defer/reward schema 兼容；
- 最新 smoke：`result/runs/v7_e2_7_e1_warmstart_safe_gate_v1_smoke_r3/`，完成 10 个独立 online instances 和 1 次 PPO update；
- warm-start：116/116 共享参数加载、10 个新参数、optimizer state 为空，checkpoint hash 为 `084184afa53cebfb04d691f5f70a9abb630a1aa0d7f4ecfdb28caeca87299533`；
- canonical identity：raw logits、softmax probability、value 的最大绝对误差均为 0；
- 固定状态池：256 个 safe dual-legal 状态，SHA256 为 `ecb92074192c6fe07f325210ad73f1b6d4f3ba7b1c937e9cbaa305e30bddebd0`，其中 E1 validation 与 E2.3 失败单元各贡献最多 128 个状态，`e2_3_weights_loaded=false`；
- PPO 前置诊断：eligible=64、counterfactual loss=`0.0011604604340391233`、gate gradient 非零、单调违规为 0；
- reward identity：10 条 trajectory 的最大绝对误差为 `5.551115123125783e-17`，低于 `1e-8`；
- smoke 元数据正确标记 `development_accepted=false`、`formal_eligible=false`，且没有生成 `accepted_checkpoint.pt` 或开发 accepted 文件。

## 十、最终判定

- **E1：保留。** 它是目前唯一有清晰正收益与 accepted 证据的主线，但需补五 seed/held-out 论文级验证。
- **旧 E2：保留为负结果。** 它证明 preference embedding + canonical checkpoint selection 会发生条件坍缩；其 accepted 不应称为 Pareto accepted。
- **E2.1/E2.2：判定设计失败。** worker direct preference 破坏 matching，E2.2 修错层级。
- **E2.3：保留为最有价值的中间版本。** 它兼具较好 diversity 和 canonical 可行性，是重做 E2 时最值得复用的诊断基线，但必须解决低-flow horizon truncation。
- **E2.4：保留为 safety control。** 它证明 state-only gate 能保安全，也证明该 gate 会压缩 Pareto 自由度。
- **E2.5：按预注册规则停止。** 不应追加 kappa 或跑 formal seeds。
- **E2.6：停止并作废 lambda 校准解释。** 三个 λ 没有任何有效 counterfactual 梯度，不能把结果解释为“λ 都不好”，只能解释为“训练目标未被激活”。
- **E2.7：实现完成、开发结果待跑。** 它正确执行 E1 warm-start、E2.3 诊断复用和安全可控 commit/defer 协议；在正式 seed11 2000 trajectories 通过全部门禁前，不得称为 accepted 或优于 E1。

## 十一、主要证据索引

- E1 protocol-v2 audit：`result/audits/20260815_182548_c0_e1_validation/audit.json`
- 旧 E2 公平预算报告：`result/analysis/e1_e2_full/report.md`
- 旧 E2 完整统计：`result/analysis/e1_e2_full/summary.json`
- E2.1 replay：`result/runs/v7_2000_e2_1_pareto_seed11/e2_1_matching_replay.json`
- E2.2 replay：`result/runs/v7_2000_e2_2_hierarchical_seed11/e2_2_matching_deadlock_replay.json`
- E2.3–E2.6 full-grid：各 run 的 `pareto_validation_log.csv` 与 `pareto_validation_candidates.csv`
- E2.6 update 级零 loss：三个校准 run 的 `update_log.csv`
- E2.6 auditor：`e2_6_calibration_audit.py`
- Checkpoint/Pareto promotion：`train.py:87-100,1122-1265,1644-1655`
- E2.5/E2.6 gate：`agent/ppo/network.py:2377-2555`
- E2.6 loss：`agent/ppo/agent.py:549-630,771-787`
- Direct preference ranker：`agent/ppo/network.py:3578-3639`
- E2.7 配置：`configs/v7/e2_7_e1_warmstart_safe_gate_v1.json`
- E2.7 专用测试：`test/test_e2_7.py`
- E2.7 最新 smoke 映射与状态池：`result/runs/v7_e2_7_e1_warmstart_safe_gate_v1_smoke_r3/`
