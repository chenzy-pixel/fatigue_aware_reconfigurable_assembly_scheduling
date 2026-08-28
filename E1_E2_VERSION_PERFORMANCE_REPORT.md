# E1 与 E2 系列版本性能、缺陷与论文可报告性审计

审计日期：2026-08-25  
审计范围：E1、旧 E2、E2.1–E2.6、E2.7 v1/v2/v2.1 的配置、训练 summary、validation/full-grid 日志、离线 equal-budget 分析、checkpoint 与失败诊断。  
报告性质：当前仓库开发证据审计，不把 smoke、单 seed、开发集或未完成运行表述为论文最终结论。

## 1. 结论摘要

1. **E1 仍是当前唯一具备 accepted checkpoint、稳定完成率和跨五 seed 等预算 Pareto 证据的可交付主线。** E1 五个训练 seed 均有 accepted checkpoint；seed 11 的 protocol-v2 固定 validation 审计为 100% 完成，canonical quality `0.32293`，相对启发式 `-12.67%`。在 test/OOD/stress 的五 seed、每实例 22 候选分析中，E1 的平均 HV 分别为 `0.3564/0.4436/0.4298`。
2. **旧 E2 的 canonical checkpoint 可以训练出来，但 preference-conditioned Pareto 能力坍缩。** 五个 seed 均存在 accepted checkpoint，但该文件只通过 canonical `w0=(0.5,0.3,0.2)` 的旧门禁。在同为 22 候选的公平预算下，旧 E2 在 test/OOD/stress 的 HV 比 E1 低 `0.0231/0.0256/0.0214`，每个 split 都是 1 胜 4 负；唯一轨迹只有 `3.57–4.82/22`。
3. **E2.1–E2.2 是可行性失败。** canonical validation 最佳完成率均为 95%，full-grid 完成率均为 96.36%。直接把偏好作用到 worker 排序破坏了 matching/liveness；E2.2 只修改 production hierarchy，没有处理根因。
4. **E2.3 是旧系列里最有价值的中间版本。** canonical validation 恢复 100%，多样性达到 `12.65` 条轨迹、`12.50` 个目标向量、`4.90` 个非支配候选；但 full-grid 仅 `430/440` 完成，10 个失败全部是低 flow 权重下的 horizon truncation。
5. **E2.4–E2.6 把安全做稳，却把 Pareto 决策自由度压缩掉。** E2.4–E2.6 均达到 440/440，但多样性和非支配数不过门禁。E2.5 的 flow boost 没有产生任何 greedy gate flip；E2.6 的 counterfactual eligible 集为 0，三个 lambda 的 loss、flip 和最终网络 hash 完全一致。
6. **E2.7 首次显示出“warm-start 后兼顾安全和高 HV”的潜力，但当前仍未验收。** v1 的 validation HV 达 `0.4503`，比 E1 seed11 validation 参考值高 18.67%，但 439/440、variance Spearman 不过、canonical 质量未过。v2.1 的最近完整 full-grid 为 440/440、HV `0.4186`、多样性 `14.45/14.10/5.25`，但 variance Spearman 为错误方向 `+0.0523`，canonical quality 比 E1 差 5.23%，训练最终在 update 120 因 `preference_stage_failed` 停止，heldout 从未执行。
7. **E2 的主要问题不是参数量不足。** checkpoint state-dict 张量元素数从 E1 的 `0.919M` 增加到旧 E2 的 `1.018M`、E2.7 的 `1.037M`；后两者分别比 E1 大约 10.8% 和 12.8%。失败来自条件信号、离散决策自由度、活性约束和训练/验收机制，而不是模型比 E1 小。

当前论文级判定：**E1 可作为主方法候选；旧 E2 可作为负结果；E2.3 可作为机制消融；E2.4 可作为 safety control；E2.5/E2.6 应停止；E2.7 仍是开发中的扩展，不能报告为 accepted 或优于 E1。**

## 2. 指标口径与可比性

### 2.1 统一目标与方向

- 三个最小化目标：flow time、reconfiguration cost、worker-load variance。
- canonical 权重：`w0=(0.5,0.3,0.2)`。
- canonical bounded quality 越低越好：

  ```text
  Q = 0.5 * F/(1200+F) + 0.3 * C/(1000+C) + 0.2 * V/(50+V)
  ```

- Hypervolume（HV）越高越好。
- 唯一动作轨迹、唯一目标向量、非支配候选数越高，通常表示覆盖更丰富。
- preference-response Spearman 期望为负：某目标权重增大时，对应的最小化目标应下降。当前门槛通常要求每个目标 `rho <= -0.05`。

### 2.2 E1 与旧 E2 的公平预算

- 每个实例每个方法都是 22 个候选。
- E1：1 次 greedy + sampled seeds `100001–100021`。
- 旧 E2：21 个 denominator-five simplex 偏好点 + canonical 点。
- 五个算法 seed：`11/23/37/53/71`。
- 三个固定 split：test/OOD/stress，各 20 个 dev 实例。
- 总候选行 `13,200`；离线分析过滤 99 个无效候选，保留 `13,101` 行。
- E1 无效候选 `27/6600=0.41%`；旧 E2 无效候选 `72/6600=1.09%`。

### 2.3 重要限制

1. 当前前沿是固定 rollout 预算下的**经验前沿**，没有精确 Pareto set 或最优解证明。
2. 五 seed equal-budget 使用的是 dev profile：每 split 只有 20 个固定实例；尚未运行 publication profile 的 validation 500、test 1000、OOD 500、stress 500。
3. E1–旧 E2 的 HV seed 级 Wilcoxon p 值为 `0.125–0.1875`，未达到常用 `p<0.05`；方向一致但统计功效不足。
4. E2.1–E2.6 在线 `_pareto_snapshot` 历史实现先计算 HV/多样性、再做安全门禁，未完成候选可能污染其报告值。因此这些版本的 HV 只用于开发诊断，不应进入正式横向论文表。
5. E2.7 使用 `safe_completed_candidates_v1` 先过滤无效候选，口径更可靠，但目前只有 seed 11 validation 开发值。
6. equal-budget 的 `front_size` 和 `union contribution` 是候选行计数，重复的非支配目标向量可能被多次计数。旧 E2 每实例只有约 3.45–4.63 个唯一目标向量，却可出现 11.79–15.07 个“front rows”，所以这两个字段不能解释为不同 Pareto 解数量。HV 对精确重复点不敏感，HV 主结论仍可保留。

## 3. E1：固定偏好策略与 rollout Pareto filtering

### 3.1 机制

E1 在固定 canonical 权重下训练一个 bounded-residual policy head：单调 production/worker ranker 保持主导，context 只通过有界 residual 修正例外。E1 不接收偏好向量；Pareto 候选来自同一个随机策略的多次推理采样，再做非支配筛选。

### 3.2 seed 11 protocol-v2 审计

| 方法 | 完成率 | canonical quality | 相对启发式 quality gap | fatigue CVaR90 |
|---|---:|---:|---:|---:|
| C0 | 100% | 0.33465 | -9.59% | 0.5953 |
| E1 seed 11 | 100% | **0.32293** | **-12.67%** | **0.4948** |

该结果来自 20 个固定 validation 实例，serial/parallel、sampling RNG、manifest、metric hash 和 action trace 审计均通过。它证明 seed 11 的 E1 accepted checkpoint 在开发验证集上有效，但不是五 seed publication 结论。

### 3.3 五 seed 训练结果

下表来自各 run 内嵌 validation/final evaluation。所有 seed 都训练 2000 episodes、200 updates、2000 个独立 online instances，并产生 accepted checkpoint。

| seed | accepted episode | accepted quality updates | validation completion | validation quality | 内嵌 quality gap vs heuristic | fatigue CVaR90 |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 1820 | 2 | 100% | 0.32409 | -11.52% | 0.4948 |
| 23 | 180 | 1 | 100% | 0.33602 | -8.31% | 0.4770 |
| 37 | 80 | 1 | 100% | 0.36574 | -1.22% | 0.7498 |
| 53 | 30 | 0 | 100% | 0.34962 | -5.10% | 0.6268 |
| 71 | 320 | 1 | 100% | 0.33864 | -8.07% | 0.4298 |
| mean ± sd | — | — | 100% | **0.34282 ± 0.01570** | **-6.84 ± 3.88%** | — |

注意：上表是历史 run 内嵌 evaluator 的值；seed 11 的最新 protocol-v2 独立审计值是 `0.32293/-12.67%`。其余四个 seed 尚未按完全相同的 protocol-v2 publication 流程重评，不能把两套数值拼成最终论文均值。

accepted episode 从 30 到 1820，seed 间收敛时点差异很大；seed 53 的 `accepted_quality_updates=0` 说明它的 accepted 更接近 phase-transition anchor，而不是充分的质量阶段改进。这是 E1 checkpoint 语义仍需细分的地方。

### 3.4 五 seed、22 候选 equal-budget Pareto 性能

| split | E1 HV mean ± sd | canonical quality | unique traces / 22 | unique objectives / 22 | normalized span: flow / cost / variance |
|---|---:|---:|---:|---:|---:|
| OOD | **0.4436 ± 0.0090** | 0.2977 | 21.90 | 21.90 | 0.0231 / 0.0939 / 0.1272 |
| Stress | **0.4298 ± 0.0124** | 0.2999 | 21.94 | 21.94 | 0.0175 / 0.0851 / 0.1312 |
| Test | **0.3564 ± 0.0174** | 0.3499 | 21.89 | 21.89 | 0.0237 / 0.1130 / 0.1256 |

去重前后需要区分：E1 的 22 个候选几乎都是不同轨迹，但每实例真正的非支配候选平均只有 OOD `3.92`、Stress `4.30`、Test `4.45`，范围包含 1。E1 的强项是稳定产生广泛候选；并非每次采样都提高 Pareto 集。

### 3.5 E1 的优点

- 训练机制简单，固定偏好下安全性和完成率最好验证。
- 2000 episodes 对应 2000 个独立训练实例，实例覆盖充足。
- 采样多样性很高；同预算下 HV、front span 和有效率优于旧 E2。
- 不需要偏好条件网络，因此没有跨偏好梯度干扰和 conditioning collapse。
- checkpoint、sampling seed 和 action trace 的 protocol-v2 可复现链较完整。

### 3.6 E1 的缺陷

- 不是 preference-conditioned Pareto learner；不能输入 `w` 后定向输出相应权衡。
- 推理时需要 22 次 rollout 加后处理，优势依赖推理预算。
- 极端目标覆盖是随机采样的副产品，没有机制保证。
- 单实例非支配点数可能只有 1；高 trajectory diversity 不等于高 Pareto efficiency。
- 五 seed publication-size 数据、统计检验、推理时间和强传统求解器对比尚未完成。
- accepted checkpoint 的语义仍混有 transition anchor，部分 seed 的质量阶段更新很少。

## 4. 旧 E2：from-scratch preference-conditioned policy

### 4.1 机制

旧 E2 从头训练，不加载 E1。每个 episode 输入一个 `w=(w_flow,w_cost,w_variance)`；70% 来自 `Dirichlet(1,1,1)`，30% 来自三个顶点、等权点和 canonical 点。偏好 embedding 进入 contextual scorer、defer/advance head 和 critic，但 E1 主 monotone ranker 本身不随偏好改变。

### 4.2 canonical 训练表面结果

五个 seed 都训练 2000 episodes、2000 个独立实例并存在 `accepted_checkpoint.pt`；其 validation completion 都是 100%。但 accepted 由 `balanced_guarded_v7` 在 canonical 点选择，不检查 22 偏好的安全、HV、唯一轨迹、非支配数或偏好响应。

| seed | accepted episode | quality updates | validation quality | 相对启发式 reward gap |
|---:|---:|---:|---:|---:|
| 11 | 80 | 2 | 0.35754 | -0.17% |
| 23 | 30 | 0 | 0.34923 | -0.47% |
| 37 | 1420 | 10 | 0.36114 | +1.66% |
| 53 | 1650 | 2 | 0.35375 | +1.82% |
| 71 | 120 | 3 | 0.33802 | +0.25% |

seed 23 在 quality update 为 0 时就有 accepted，直接说明“存在 accepted 文件”不能解释为 Pareto 已通过。

### 4.3 与 E1 的 equal-budget 对比

| split | E1 HV | 旧 E2 HV | E2-E1 | seed 胜/平/负 | Wilcoxon p | E1 / E2 unique traces |
|---|---:|---:|---:|---:|---:|---:|
| OOD | 0.4436 | 0.4180 | **-0.0256** | 1/0/4 | 0.1250 | 21.90 / 3.57 |
| Stress | 0.4298 | 0.4084 | **-0.0214** | 1/0/4 | 0.1875 | 21.94 / 4.06 |
| Test | 0.3564 | 0.3333 | **-0.0231** | 1/0/4 | 0.1875 | 21.89 / 4.82 |

Test 的 E2 front span 只有 E1 的约 18%（flow）、17%（cost）、34%（variance）。平均 preference-response 在多个 split 接近 0，且 Stress 的 variance response 为 `+0.0569`，表明提高 variance 权重反而没有稳定降低 variance。

### 4.4 旧 E2 的缺陷

- checkpoint 选择只看 canonical，训练目标和验收目标不一致。
- 从 E1 完全重训，丢弃已验证的 canonical 能力锚点。
- preference 只控制弱 contextual residual，容易被固定主 ranker 淹没。
- 多个偏好映射到少数相同轨迹；这是典型条件坍缩。
- 无效候选率 `1.09%`，高于 E1 的 `0.41%`。
- 全部正式 run 记录 `git.dirty=true`，且 run 内没有完整 source snapshot/patch；精确源码复现不足。
- evaluation schema 在 config 与 summary provenance 之间有 `4.2.0/4.1.0` 不一致。

## 5. E2.1–E2.7 性能总表

表中 E2.1–E2.6 的 HV 可能受未完成候选污染，仅用于版本内诊断；E2.7 已先过滤无效候选。

| 版本 | seed / 预算 | canonical validation | full-grid 安全/完成 | HV | traces / objectives / ND | rho flow / cost / variance | 最终状态 |
|---|---|---:|---:|---:|---:|---:|---|
| E2.1 | 11 / 2000 ep | 最佳 95% | 424/440, 96.36% | 0.3232* | — / 12.35 / 6.40 | 未记录 | feasibility 未达到 |
| E2.2 | 11 / 2000 ep | 最佳 95% | 424/440, 96.36% | 0.3199* | — / 12.10 / 5.90 | 未记录 | feasibility 未达到 |
| E2.3 | 11 / 2000 ep | 100% | 430/440, 97.73% | 0.4020* | 12.65 / 12.50 / 4.90 | -0.203 / -0.208 / +0.001 | safety failed |
| E2.4 | 11 / 2000 ep | 100% | **440/440** | 0.3214* | 6.85 / 6.35 / 3.50 | -0.106 / -0.064 / -0.105 | diversity failed |
| E2.5 k=1 | 101 / 500 ep | 100% | **440/440** | 0.3226* | 8.45 / 7.60 / 3.40 | -0.098 / -0.070 / -0.503 | calibration stopped |
| E2.5 k=2 | 101 / 500 ep | 100% | **440/440** | 0.3284* | 9.20 / 8.55 / 3.40 | +0.018 / -0.086 / -0.345 | calibration stopped |
| E2.5 k=3 | 101 / 500 ep | 100% | **440/440** | 0.3222* | 6.40 / 5.95 / 3.25 | -0.051 / -0.107 / -0.553 | calibration stopped |
| E2.6 λ=.05/.10/.20 | 101 / 500 ep | 100% | **440/440** | 0.3253* | 6.25 / 5.65 / 2.80 | +0.011 / -0.106 / -0.451 | zero-signal, stopped |
| E2.7 v1 | 11 / 2000 ep | 100% | 439/440, 99.77% | **0.4503** | 12.45 / 12.40 / 6.10 | -0.550 / -0.438 / -0.035 | safety/variance/quality failed |
| E2.7 v2 | 11 / 260 ep | 100% | **440/440** | 0.3837 | 7.60 / 7.50 / 3.55 | -0.076 / -0.216 / +0.027 | 运行不完整，无 summary |
| E2.7 v2.1 | 11 / 1200 ep | 100% | **440/440** | **0.4186** | **14.45 / 14.10 / 5.25** | -0.167 / -0.268 / +0.052 | update120 stage failed |

`*`：历史 snapshot 未先排除全部无效候选，不能与 E1/E2.7 的过滤后 HV 作正式论文比较。

## 6. E2.1：direct preference ranker

### 改动

E2.1 把偏好更直接地加入 production 和 worker 候选排序，并使用每实例 5 个偏好的 paired training。2000 trajectories 因此只覆盖 400 个独立实例。

### 性能

- canonical validation 最佳完成率 95%，到 episode 2000 仍是 95%。
- full-grid 完成率 `424/440=96.36%`。
- 历史 snapshot：HV `0.3232`、唯一目标向量 `12.35`、非支配候选 `6.40`。
- 没有 accepted checkpoint，训练一直停在 feasibility phase。

### 缺陷与因果证据

在 `validation_worker_bottleneck_2000003`、`w=(0,0.2,0.8)` 下，策略在约 82 步后进入 advance/defer 循环，0/15 orders 完成。关闭 worker direct preference 或允许 worker recovery 后恢复 15/15；只保留 worker direct preference 仍失败。因此根因不是 Pareto 门禁，而是 worker 偏好项破坏 full matching 和系统活性。

## 7. E2.2：hierarchical production decoder

### 改动

先选择 production gate，再在 commit 集合中选择 pair，试图避免 joint argmax 的动作语义冲突。

### 性能

- canonical validation 最佳完成率仍为 95%。
- full-grid 仍为 `424/440=96.36%`。
- HV `0.3199`，比 E2.1 更低；唯一目标向量 `12.10`、非支配候选 `5.90`。
- 无 accepted checkpoint，仍停在 feasibility phase。

### 缺陷

replay 中 gate-first 与 joint argmax 对故障轨迹没有实质差异；真正破坏活性的是 worker direct preference。E2.2 修改了 production 层，没有修改 worker 故障域，所以属于修错层级。

## 8. E2.3：safe production preference

### 改动

移除 worker 侧危险的 flow/cost 直接偏好，只在 production 侧保留主要偏好控制，并对 worker 采用安全的 variance 处理。

### 性能

- canonical validation 恢复到 100%。
- full-grid `430/440=97.73%`。
- 多样性 `12.65/12.50/4.90`，是 E2.3–E2.6 中最好的一档。
- flow/cost Spearman 分别 `-0.203/-0.208`；variance `+0.001`，基本无响应。
- 历史 HV `0.4020`，但可能被 10 个未完成候选污染。

### 缺陷

10 个失败全部是 `termination_reason=horizon`，集中在 `w_flow<=0.2`；无调度违规，fatigue margin 非负。说明低 flow 偏好诱导 production 过度 defer，属于活性/完工失败，不是疲劳安全失败。E2.3 很适合作为“偏好自由度与 liveness 冲突”的中间消融，但不能作为可部署模型。

## 9. E2.4：neutral state-only safety gate

### 改动

commit/defer gate 改为不看偏好的 state-only gate，偏好只在更安全的下游路径作用。

### 性能

- canonical 和 full-grid 均 100% 完成，440/440、0 schedule violation。
- HV `0.3214`。
- 多样性降至 `6.85/6.35/3.50`，低于 `8/8/4` 门槛。
- 三个 Spearman 都是负方向，但整体 coverage 不足。

### 缺陷

安全来自“有合法 pair 就 commit”的近饱和决策。最终候选中有合法 production pair 的决策数与 gate commit 数都为 `31,020`。因此 E2.4 通过消除主动等待自由度获得安全，也消除了 commit/defer 层面的主要偏好可控性。它适合作为 safety control，不适合作为 Pareto 主方法。

## 10. E2.5：monotone flow commit boost

### 改动

在 E2.4 冻结 gate 上增加 `kappa * max(w_flow-0.2,0)` 的正向 commit residual，预注册 `kappa=1/2/3`，每个运行 500 episodes、100 个独立实例。

### 两次最终 full-grid 审计

| kappa | traces | objectives | ND | flow rho | 结果 |
|---:|---:|---:|---:|---:|---|
| 1 | 8.00 / 8.45 | 7.45 / 7.60 | 3.40 / 3.40 | 最终 -0.098 | objectives、ND 不过 |
| 2 | 9.55 / 9.20 | 8.95 / 8.55 | 3.90 / 3.40 | +0.061 / +0.018 | ND 不过且 flow 方向错误 |
| 3 | 6.50 / 6.40 | 6.00 / 5.95 | 2.85 / 3.25 | 最终 -0.051 | 多样性全面不足 |

### 缺陷

mean commit boost 随 kappa 大致从 `0.195 -> 0.394 -> 0.587` 增加，commit probability 也略增，但所有实验的 `base-defer -> final-commit` greedy flip 都是 0。基础 gate 已经在所有可 commit 状态选择 commit，正向 boost 只能改变概率，不能改变 argmax 行为。按预注册规则应停止，不能继续插入新 kappa 后挑最好结果。

## 11. E2.6：counterfactual preference consistency

### 改动

学习 state-dependent high-flow commit scale，并仅在 `commit/defer` 均合法且冻结 base gate 会 defer 的状态上施加 counterfactual loss；预注册 `lambda=0.05/0.10/0.20`。

### 性能

三个 lambda 的最终值完全一致：

- 440/440 完成；
- HV `0.3253165`；
- `6.25/5.65/2.80`；
- Spearman `+0.01098/-0.10580/-0.45073`；
- counterfactual eligible `0`、loss `0`、greedy flip `0`。

三个最终 network SHA256 同为 `cdbcae5beff70c1e99a5da5ac6e12320d205ac2a657c17e590de0f04171c3c34`。

### 缺陷

这是零有效训练信号，不是“lambda 搜索后发现三个 lambda 都不好”。训练目标的 eligible 集在真实策略分布中为空，所以 lambda 没有进入有效梯度。E2.6 的校准解释应作废，只能报告机制未激活。

## 12. E2.7：E1 warm-start 与 centered preference adapters

### 12.1 共同机制

- 从 E1 seed11 accepted checkpoint 严格加载 116/116 共享 state keys；E2.3 只用于失败状态池，不加载权重。
- canonical `w0` 下 raw logits、softmax probability 和 value 与 E1 保持 identity。
- GNN trunk 冻结；偏好适配器分阶段训练 gate、production pair、worker variance。
- worker flow/cost 直接偏好为零；所有偏好动作必须经过 matching、安全和 liveness mask。
- 训练恢复为 2000 trajectories 对应 2000 个独立实例，不再把实例覆盖压缩到 1/5。
- E2.7 的 checkpoint state-dict 为 `1.0366M` 张量元素，比 E1 大约 12.8%。

### 12.2 E2.7 v1

性能：

- 2000 episodes、200 updates、2000 个独立实例。
- full-grid `439/440=99.77%`，唯一失败位于低-flow 半区。
- 过滤无效候选后 HV `0.45025`，比 E1 seed11 validation reference `0.37940` 高 18.67%。
- 多样性 `12.45/12.40/6.10`。
- gate extreme flip rate `0.6075`，0 monotonicity violation。
- Spearman flow/cost 很强：`-0.550/-0.438`；variance `-0.035`，未达到 `-0.05`。
- canonical quality `0.34058`，相对启发式 gap `-8.49%`，未达到开发门禁要求的 `-10%`，且比 E1 validation quality 差约 5%。

缺陷：

- 仍有 1 个未完成候选，低-flow safety gate 失败。
- worker variance 通道响应不足。
- canonical anchor 虽保持初始化 identity，训练后质量保护不足。
- validation 本地门禁未过，test/OOD/stress heldout 被正确锁住，未执行。
- 没有 `accepted_checkpoint.pt` 或 `development_accepted_pareto_checkpoint.pt`。

### 12.3 E2.7 v2

该目录只保存到约 260 episodes/update 26 的中间产物，没有 summary、train/update/validation logs，不是完整实验。

中间 full-grid 为 440/440，HV `0.38366`，但多样性 `7.60/7.50/3.55` 未过，variance Spearman `+0.027` 为错误方向，canonical gap 仍为 `-8.37%`。只能作为中间调试记录，不能与完整版本排名。

### 12.4 E2.7 v2.1

最近一次完整 full-grid（update 106）：

- 440/440、0 schedule violation，low-flow 220/220；
- HV `0.41863`，比 E1 seed11 validation reference 高 10.34%；
- `14.45` 条唯一轨迹、`14.10` 个唯一目标向量、`5.25` 个非支配候选；
- gate flow-cost / flow-variance flip rate 为 `0.309/0.355`，0 monotonicity violation；
- Spearman flow/cost 为 `-0.167/-0.268`，variance 为错误方向 `+0.052`；
- canonical quality `0.34103`，相对启发式 `-8.37%`，比 E1 reference `0.32409` 差 5.23%；
- heldout 报告为 `local_development_gate_not_ready`，没有 test/OOD/stress 结果。

训练最终在 update 120、episode 1200 停止：

- `training_status=preference_stage_failed`；
- stage 仍为 `production_pair`，从 update 10 进入后未能过渡到 worker variance；
- production-pair control correct rate 已到 `0.511`，但 `constraint_status=constraint_active`、loss 约 `0.205`，连续通过数仍为 0；
- worker variance scale 仍为 0，因此 variance preference 没有被真正训练；
- safety replay 通过、canonical identity error 为 0，失败不是数值爆炸或安全事故，而是阶段训练/约束未达标。

这解释了 v2.1 的表面矛盾：HV 和轨迹多样性已经较好，但第三目标响应方向错误，因为训练在进入 worker-variance 阶段之前就结束。

### 12.5 E2.7 的额外复现缺陷

- 当前工作区的 `configs/v7/e2_7_e1_warmstart_safe_gate_v1.json` 和 `v2_1.json` 已继续演化为 monitored-v4 配置，而 v2/v2.1 运行目录内嵌的是 metric-gated-v2/v3。必须以 run 内 `config.json` 解释历史结果。
- v2.1 失败时 `summary.json` 被写成仅含失败现场的最小 summary，丢失正常 run summary 的完整 provenance、episodes、validation 汇总等字段；虽然原始 CSV 仍在，但自动聚合会误判。
- `warm_start_mapping.json` 和状态池 provenance 含远端机器绝对路径；hash 可核对，但路径本身不可移植。
- 未生成完整 source snapshot/patch 时，dirty 或持续演化的代码不能仅靠当前 HEAD 复现历史行为。

## 13. 模型规模比较

以下是 checkpoint `network` state-dict 中所有张量元素数，不等同于当前阶段实际可训练参数数；冻结策略不同，不能只按总量推断优化难度。

| 版本 | state entries | tensor elements | 相对 E1 |
|---|---:|---:|---:|
| E1 | 116 | 919,188 | 1.000× |
| 旧 E2 | 120 | 1,018,132 | 1.108× |
| E2.1 / E2.3 | 121 | 1,018,133 | 1.108× |
| E2.4 | 125 | 1,019,671 | 1.109× |
| E2.5 | 126 | 1,019,672 | 1.109× |
| E2.6 | 129 | 1,021,080 | 1.111× |
| E2.7 v1/v2.1 | 126 | 1,036,570 | 1.128× |

旧 E2 和 E2.7 都比 E1 大。E2 失败不是“参数量不够”，而是一个共享网络必须学习多个冲突的偏好子任务；增加少量 conditioner 参数不能自动保证 preference-to-action 可控性。

## 14. 跨版本共同缺陷

### 14.1 训练预算不等价

E1 和旧 E2 的 2000 trajectories 对应 2000 个独立实例；E2.1–E2.4 的 5 偏好 paired training 只对应 400 个独立实例，E2.5/E2.6 的 500 trajectories 只有 100 个独立实例。若比较泛化，应同时报告 trajectory budget 和 unique-instance budget。

### 14.2 checkpoint 名称曾混淆能力层级

旧 E2 的 `accepted_checkpoint.pt` 只代表 canonical 单点通过；E2.3 以后要求 full-grid Pareto 门禁。不同版本的 accepted 文件不是同一统计事件。正式论文应拆分：

- feasibility anchor；
- canonical accepted；
- full-grid safe；
- Pareto accepted。

### 14.3 选择集与报告集未完全隔离

旧 E2–E2.6 多次使用固定 20-instance validation 做调参、门禁和版本选择，容易对小型开发集过拟合。E2.7 的 heldout 锁定设计是正确方向，但截至当前没有任何 E2.7 版本触发 heldout。

### 14.4 统计功效不足

E1/旧 E2 equal-budget 只有 5 个 algorithm-seed pair；即便三个 split 都是 1/0/4，Wilcoxon 仍不显著。E2.1–E2.7 大多只有一个 seed，更不能形成论文一般性结论。

### 14.5 wall-clock 不可直接比较

运行跨不同日期、机器用户目录、代码 dirty 状态和 checkpoint optimizer 内容；当前文件中的 inference/training wall time 不构成公平速度对比。若论文报告速度，必须在同一硬件、同一 parallelism、同一候选预算下重新评测。

## 15. 论文可报告性分级

| 版本 | 当前角色 | 可以报告 | 不能报告 |
|---|---|---|---|
| E1 | 主方法候选 | seed11 开发审计；五 seed dev equal-budget 基线 | publication-size 普遍优越性 |
| 旧 E2 | 负结果 | canonical accepted 与 Pareto collapse 的分离 | “E2 已学会 Pareto” |
| E2.1 | 失败消融 | worker direct preference 破坏 liveness | Pareto 性能排名 |
| E2.2 | 失败消融 | production hierarchy 未修复 worker 根因 | 层级 decoder 有效 |
| E2.3 | 机制中间点 | 多样性恢复与低-flow truncation 冲突 | 安全可部署 |
| E2.4 | safety control | state-only gate 达到 440/440 | 高质量 Pareto 方法 |
| E2.5 | 预注册停止 | 饱和 gate 上正向 boost 无 greedy flip | 追加 kappa 后择优 |
| E2.6 | 零信号诊断 | eligible 集为空、loss 未激活 | lambda 效果比较 |
| E2.7 v1 | 开发失败 | 高 HV/强 flow-cost 响应与剩余失败门禁 | accepted / heldout 优于 E1 |
| E2.7 v2 | 不完整运行 | 中间调试现象 | 任何最终性能结论 |
| E2.7 v2.1 | 最有希望的未通过版本 | 440/440、高多样性、stage 失败诊断 | accepted / test/OOD/stress 优越性 |

## 16. 当前建议

1. **论文主线暂定 E1。** 先用相同 protocol-v2 evaluator 对五个 E1 accepted checkpoint 跑 publication profile，并报告 5 seed 均值±标准差、配对检验、有效率、推理预算和传统强基线。
2. **统一正式 Pareto evaluator。** 所有版本先过滤未完成/截断/违规/疲劳越界候选，再对目标向量去重；HV、distinct nondominated count、trajectory diversity 分开报告。
3. **E2.7 若继续，应优先解决阶段训练而不是扩大网络。** v2.1 已证明 gate 和 production pair 能产生多样性；当前最直接缺口是 worker-variance 阶段未进入、canonical 质量保护不足和 final acceptance summary 不完整。
4. **在 validation 本地门禁全部通过前保持 heldout 封闭。** 通过后才运行五 seed test/OOD/stress equal-budget，避免继续对 heldout 调参。
5. **冻结每个正式 run 的可恢复 provenance。** 保存 effective config、source snapshot/patch、git diff、环境版本、checkpoint hash 和 evaluator hash，不能只保存 dirty flag。

## 17. 主要证据索引

- E1 protocol-v2 审计：`result/audits/20260815_182548_c0_e1_validation/audit.json`
- E1 policy head：`E1_POLICY_HEAD.md`
- E1/旧 E2 equal-budget 报告：`result/analysis/e1_e2_full/report.md`
- E1/旧 E2 完整统计：`result/analysis/e1_e2_full/summary.json`
- E1/旧 E2 seed 明细：`result/analysis/e1_e2_full/seed_summary.csv`
- E2 总失败审查：`E2_FAILURE_AUDIT_REPORT.md`
- E2.1 replay：`result/runs/v7_2000_e2_1_pareto_seed11/e2_1_matching_replay.json`
- E2.2 replay：`result/runs/v7_2000_e2_2_hierarchical_seed11/e2_2_matching_deadlock_replay.json`
- E2.3–E2.6：对应 run 的 `pareto_validation_log.csv`、`pareto_validation_candidates.csv`、`update_log.csv`
- E2.7 v1：`result/runs/20260823_215521_train_parallel/`
- E2.7 v2：`result/runs/e2_7_v2/`
- E2.7 v2.1：`result/runs/e2_7_v2_1_seed11/`

