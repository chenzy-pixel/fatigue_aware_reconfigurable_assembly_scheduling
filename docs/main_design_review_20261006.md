# Main 设计复核 · 2026-10-06

后续实施：R4 的消息配对表达限制已采用聚合前联合 ReLU 处理；实现与验收见
[联合消息实施记录](joint_messages_20261006.md)。下文保留 `43add12` 审查时的版本身份。

审查对象：工作区 `codex/schema10-merged-main`，HEAD 与本次仓库记录的 `origin/main` 同为 `43add121372327ae61b0d25749e4aca51890306d`。本地名为 `main` 的分支仍指向较早提交，未把它作为审查对象。未执行远端 fetch，也未修改生产代码、配置、固定算例或已有测试。

结论：当前调度、schema-10 投影、PPO 与工程截断自举的主干设计成立，但仍有七项可验证的功能缺陷或设计缺口。正式结果分析尤其需要统一物理可行性与比较身份检查。以下构造审计仅证明相应缺口存在，不代表实际训练性能或失败频率。

复现脚本：`result/audits/main_review_20261006/reproduce_findings.py`。证据：同目录 `findings.json`、构造实例及合成分析输入。审计目录由仓库现有规则忽略；本报告为持久化审查记录。

```powershell
.\.venv\Scripts\python.exe result/audits/main_review_20261006/reproduce_findings.py
```

## R1 · P1：通用 Pareto 分析会接纳原问题中不安全的 neutral 调度

位置：`analysis/pareto_analysis.py:148`、`:161`；`data/distribution.py:21`。

`valid_candidate()` 只用模拟器的 `maximum_worker_fatigue` 判断安全，没有使用独立原模型审计的 `fatigue_monitor_peak`；协议检查也没有限制 `fatigue_mode`。用于共享基准数据的 environment hash 有意排除了 fatigue mode，因而该 hash 不能承担运行物理模型相同的校验。

复现使用通过 `validate_instance()` 的一订单实例，原始工人初始疲劳为 0.74；neutral 环境依法在 M6 上连续派 H5 拆卸、安装并完成加工。得到：

| 字段 | 值 |
|---|---:|
| task_succeeded | true |
| neutral 模拟器 maximum_worker_fatigue | 0 |
| 原模型 fatigue_monitor_peak | 1.0 |
| safe_fatigue_limit | 0.75 |
| 原模型超限面积 | 3.4977485714 |

将该实际调度的目标与安全字段放入完整的合成 13×3 分析单元，默认 full 配置的 `analyze_rows()` 仍接受全部 39 个候选，完成率 1.0、HV 0.1735836245。实例与分析 hash 为明确标注的构造输入，未冒称发布数据上的真实评测。

影响：通用 CSV 分析及 PPO/MO-ALNS 比较可能把改变物理约束的结果当作同一问题的安全前沿。V8 专用 `analyze_runs()` 已拒绝 neutral；匹配疲劳消融也有独立语义，不能把本项扩大为所有分析入口都失效。

建议：区分数据集身份与执行物理模型身份；正式原问题前沿只接纳 full，安全判断统一使用原模型审计及调度检查。neutral 结果继续进入其专门的反事实消融报告。

## R2 · P2：前沿贡献统计把重复采样数量当作方法优劣

位置：`analysis/pareto_analysis.py:279`；`analysis/mo_alns_analysis.py:47`。

非支配排序保留相同点，union contribution 随后按原始候选行数计数，比较脚本又将它作为越大越好的性能指标。PPO 每偏好三次采样，MO-ALNS 每偏好一个选解，预算差异由此直接进入胜负判定。

完整 13 偏好构造矩阵中，两方法所有候选的三目标完全一致：HV 均为 0.1212194644，贡献却分别为 PPO=39、MO-ALNS=13；胜平负报告给 MO-ALNS 记一次 loss。正式 66 点矩阵同样会出现 198 对 66 的机械差异。

建议：先按目标向量或调度轨迹去重，再定义贡献；共享同一点时按共同覆盖、平分归属等明确规则处理，或从优劣检验中移除原始候选贡献数。HV 本身不受这项重复计数问题影响。

## R3 · P2：README 的 V8 Pareto 命令与公开 CLI 脱节

位置：`README.md:259`；`analysis/pareto_analysis.py:327`。

README 承诺 `python -m analysis.pareto_analysis --run-dir ...` 支持评测目录和多个训练种子；当前 parser 只定义 `--candidate-csv`。原样运行文档命令，尚未访问数据即退出，returncode=2，提示缺少 `--candidate-csv`。

`analysis/v8_pareto_analysis.py` 提供所需 Python API，但没有 CLI main；匹配消融汇总通过 API 调用可以执行，不能代替 README 中独立主方法的公开入口。

建议：恢复同一 CLI 对目录模式和 CSV 模式的分派，或提供可执行的新入口并同步文档；对 README 命令做参数层与微型文件层的入口验证。

## R4 · P2：HGNN 消息聚合仍会丢失邻居与边属性的对应关系

位置：`agent/ppo/network.py:390`、`:420`、`:428`。

消息为对 `[h_neighbor, edge_attributes]` 做一次 Linear，随后聚合，非线性在聚合之后才执行。这等价于分别累加邻居项和边属性项；保持各端点属性和的交叉交换无法被图编码识别。

对当前 schema 10 重跑两份合法构造实例及网络种子 0/1/11/23/37：完整观测的边属性不同，两类相关边在每个端点的属性和差值为 0；float64 图上下文最大差不超过 8.89e−16，critic 差不超过 1.12e−16。同一固定派发规则的 Flow 分别为 270 与 300，对应回报为 0.8013464298 与 0.7840406004。

这是已验证的表达能力限制。构造工时不属于当前常用生成范围，且此证据没有证明两状态的最优价值不同；actor 直接读取候选边，不能据此断言 actor 也无法区分或正式训练必然失败。

建议：有属性关系改用聚合前的非线性联合消息或由边属性调制邻居表示的门控；作为模型变更更新 checkpoint 契约，重新训练并做匹配实验。

## R5 · P2：runner 步数截止仍漏标采样截断

位置：`agent/ppo/parallel.py:1331`、`:1540`、`:1619`。

当前 `collect_training_batch(step_limit=2)` 两步后停止采样并重置实例，最后 transition 的 done=False，自举正常；返回指标却同时为 terminated=False、truncated=False、sampling_truncated=False、task_done=False，terminal_reason=None。

影响：采样尝试无法归入成功、失败、外部截断三类，日志与覆盖率可能误读。正式训练当前没有传这个 step_limit；默认受影响入口是 smoke 及直接调用 runner 的用户。环境本身的 decision/zero-time 工程保护已正确标记截断。

建议：runner cutoff 显式记录 sampling_truncated 和 rollout_step_limit，保留物理末状态、done=False 与现有自举，并覆盖 reset 强制动作段。

## R6 · P2：长训练缺少持续断点和完整 resume

位置：`train.py:694`、`:865`、`:903`。

best 只在验证改善时保存，last 只在训练循环完成后保存；`--initial-checkpoint` 使用 load_optimizer=False，仅作网络初始化。检查点没有恢复 episode/课程游标、随机源、选模状态与学习率 plateau 状态的训练入口。

影响：2000-episode Universal 等长实验中断后不能从最近一次更新准确续训；best 可能已落后很多更新，重新初始化不等于继续同一次实验。这是此前报告已指出、当前仍存在的工程缺口。

建议：每次或按配置间隔在更新边界原子保存完整训练状态；把权重初始化与 resume 明确区分，并验证中断恢复与连续执行的一致性。

## R7 · P2：独立 V8 方法比较未核对问题和奖励控制变量

位置：`analysis/v8_pareto_analysis.py:67`、`:155`。

每个 run 的有效配置 hash 校验成立，不代表不同 run 的比较条件相同。shared_identity 只比较 dataset、manifest、实例集合、偏好、重复预算、尺度及归一化 hash；未比较 environment、reward、生成协议、执行精度等，也未逐行核对这些身份。

完整构造网格中，两个 run 都有按各自配置正确构造的 provenance；其中一个把 max_decisions 从 5000 改为 5001、失败惩罚从 2 改为 7，独立 `analyze_runs()` 仍接受共 792 个候选并生成方法比较。

这说明入口不能保证所声称的 matched 比较，不证明这些参数变更本身一定造成该构造数据的性能差。匹配消融汇总上游已有 `training_contract` 校验，其防护应保留并复用。

建议：定义方法比较合同，明确哪些算法/结构字段允许不同；物理模型、数据协议、执行预算及受控训练条件逐项校验，并核对行、配置和 provenance 的一致性。研究者有意比较不同训练协议时应明确标注该实验设计。

## 验证范围和修复顺序

全仓常规 pytest：**525 passed、2 skipped、1 failed，730.82 秒**。两个 slow 测试按既有规则跳过；失败是 `test/test_data_v2.py::test_legacy_data_is_preserved_and_loadable`。命令如下，最终输出保存在审计目录 `pytest_final_output.txt`。

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp/review1006
```

对历史清单全部 565 个文件另做只读哈希核查：无缺失，561 个直接匹配，4 个旧版 `data/manifests/{ood,stress,test,validation}/manifest.json` 不匹配。四个文件本地为 LF，冻结哈希对应 CRLF；仅在内存中将 LF 转为 CRLF 后，四个 SHA256 **全部准确匹配**。`.gitattributes` 已指定这些路径 `eol=crlf`，本地工作树文件尚为 LF，Git 的规范化比较仍显示 clean。这项失败属于当前工作区字节换行状态，不是新发现的数据内容损坏，也未计为 R1–R7。`prepare_data_v2.py` 在此状态下同样会被历史保护检查阻止。证据为 `legacy_hashes.json` 与 `legacy_line_endings.json`；本轮保留原文件没有改写。

未运行完整正式长训练、publication 档扩展、慢速测试或训练种子矩阵；本轮不能证明收敛、泛化与论文性能结论。

建议先修 R1/R2/R3/R5，防止结果解释错误和公开入口失败；同步补 R6 保障长实验；R4 单独升级网络并重新训练；R7 统一正式比较合同。当前多种子配置存在，seed11 消融编排也明确为描述性实验，不能把推理重复数当作独立训练次数或独立实例数。
