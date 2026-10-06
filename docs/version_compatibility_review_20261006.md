# 版本兼容性复核 · 2026-10-06

审查远端 main：`0970917452f0bee63dcad16fb530a28f1d8ab0ab`，已 fetch 核对。当前工作分支 HEAD 为 `c74668b941ad7ef1dd06818359ba3d6b50bc5b70`，源码树与远端 main 相同。范围覆盖近期结构消融、schema 10 与 HGNN 联合 ReLU 的版本边界。

发现三处可复现的兼容性遗漏，集中在评估复用与结果分析入口。正常 checkpoint 加载、配置快照加载和训练 run 发现流程已经执行新身份校验；这些保护尚未贯穿所有入口。

后续实施：C2 已统一接入内存 agent 网络与 checkpoint 疲劳身份校验，并覆盖 prepared policy 及并行入口。下文保留修复前的审查记录；当前接口见 [架构中的评估契约](ARCHITECTURE.md#7-可复现评测)。

C2 验收：新增 63 个回归用例，相关回归合并复测后为 139 passed、1 skipped，覆盖实际并行 worker、串并行轨迹与 RNG 一致性、checkpoint CLI、微型 PPO 更新—验证—重载—评估。拒绝在推理或 worker 执行前发生，目标网络、Adam、训练模式及 RNG 保持原状态；新建 neutral agent 和身份一致的已加载 agent 仍可评估。证据位于 `result/audits/evaluation_agent_fix_20261006/`。初跑的 5 个 Windows 长路径失败使用短临时路径复测通过；另 1 个测试的 16 维模型/128 维配置已对齐，原模式恢复断言保留。

## C1 · P2：消融汇总仍接受缺失或错误消息身份的来源

位置：`analysis/ablation_analysis.py:120`、`:133`、`:138`；`scripts/ablation_protocol.py:48`。

`validate_role_evaluation()` 直接读取训练和评估的 config JSON，通过旧 `protocol_profile()` 比较。这一 profile 不含 runtime manifest，也没有生成 normalized network config 的计算身份。checkpoint 只手动检查 encoder/head、观测 schema 与 fatigue mode，没有调用 `infer_checkpoint_network_spec()`。因此，同为 schema 10 的旧线性消息文件能绕过新身份边界。

复现构造完整 full_flow 的 20 实例 × 3 repeat 来源，使用实际当前网络的 state-dict 和完整 spec，然后分别移除两项消息身份，或把身份改为错误值；相应快照和文件 hash 保持自洽。两组 60 行均通过 `validate_role_evaluation()`，而相同 checkpoint 被正式 spec 校验拒绝，相同快照被 `load_config()` 拒绝。此实验是旧格式校验反例，指标来源为测试 fixture，不是实际长训练效果。

影响：已有完整消融 batch manifest 的汇总可以纳入应要求重新训练的模型结果。`discover_training_run()` 已调用严格的 `validate_training_run()`，新训练发现与正常评估链路不受这一旁路影响。

建议：汇总复用 `load_config()`、`validate_training_run()` 与统一的 network-spec 校验，再核对实际评估来源；移除重复的旧式契约检查。回归必须覆盖消息身份缺失、错误及旧 runtime manifest，不能只验证 schema 与结构选择。

## C2 · P2：内存 agent 评估缺少校验，并可能错误标注 encoder

位置：`eval.py:165`、`:184`、`:505`、`:826`；准备好的 policy 复用入口也需要一致检查。

通过 checkpoint 路径建立 `EvaluationPolicy` 时会加载并检查网络与 fatigue 身份。直接传入 `ppo_agent` 则跳过这些检查；输出行的 `encoder_variant`、`actor_head_variant` 来自调用者配置，而非实际执行的网络。

复现使用合法的一订单一工序实例，给 `evaluate_dataset()` 提供实际 `node_mlp_pool` 网络和 HGNN 配置。评估正常完成，但行记录为 `encoder_variant=hetero_gnn`。直接运行已有的 `assert_network_config_matches_spec()` 则会拒绝这组组合。

另一个构造 checkpoint 在 metadata 中声明 neutral；加载为内存 agent 后可接受 full 评估，而同一文件走 checkpoint 路径会因 fatigue mode 不匹配被拒绝。该例验证入口行为差异，不声称权重经过 neutral 长训练。

影响范围：公共 Python API、复用内存 agent/prepared policy 的调用。正常训练构造相同配置的 agent 与评估，以及使用 checkpoint 路径的 CLI，不由这个反例推出有问题。

建议：评估开始前统一比较 agent 的实际 network spec 与评估配置；已加载 agent 同时检查 checkpoint metadata 的 fatigue mode。输出身份应从通过校验的实际执行对象生成，串行、并行和 prepared-policy 复用应遵循同一规则。

## C3 · P2：独立 CSV 分析没有消息版本身份，能混合旧、新模型结果

位置：`eval.py:505`；`analysis/pareto_analysis.py:161`；`result/metrics.py:278`。

当前评估行没有 `message_function` 和 `message_aggregation`。结果 schema 8.0.0 与 observation schema 10 是不同契约；它们不能区分 schema 10 下的旧线性消息与新联合 ReLU。配置和源码 hash 虽然提供间接来源，CSV 分析并未按它们验证该计算边界。

复现：完整 66 偏好 × 3 repeat 矩阵、相同 PPO arm 中，交替标记 `linear_v1` 和 `attributed_joint_relu_v1`，`analyze_rows()` 接受全部 198 行；两条明确带不同消息身份的评估行也能通过 `aggregate_evaluation_rows()`。这是合成协议输入，不代表真实训练质量。

影响：直接 CSV 分析和聚合无法发现旧、新计算语义混用。专用 V8 `analyze_runs()` 会调用严格的 `load_config()`，可拒绝带旧 runtime manifest 的目录；不能把此问题推广为所有目录分析都接受旧快照。

建议：将实际模型计算身份写入 PPO 评估行和评估汇总；同一模型 arm 的候选必须拥有兼容身份和完整来源。图消融允许 HGNN 与 node_mlp_pool 按 encoder 派生不同身份，但各自仍需验证。启发式与求解器应记录自己的适用身份，历史 CSV 缺字段时应给出明确处理边界。

## 已验证的兼容边界

| 边界 | 结果 |
|---|---|
| 旧 observation schema、缺失/错误消息身份 checkpoint | 明确拒绝，迁移开关不能绕过 |
| 拒绝后的目标网络与 Adam 状态 | 现有回归验证保持原状态 |
| 当前 checkpoint 网络与 optimizer 往返 | 通过 |
| 当前 schema-10 真实 CLI 评估、执行权重 hash 和来源 | 通过 |
| runtime manifest 旧快照 | `load_config()` 拒绝并要求重新训练 |
| 图消融身份派生、其他消融控制变量 | 正式配置合同回归通过；C1 是单独汇总旁路 |
| 结构变体参考/批处理输出与梯度、CPU/CUDA | 当前回归通过 |

旧 schema 模型与同 schema 的旧线性消息模型都需要重新训练。这是已约定的执行语义边界，不能因 state-dict 形状相同而放行。原始实例数据与模型观测不是同一版本契约。

## 验证与证据

本次针对性回归：**118 passed，77.41 秒**。

```powershell
.\.venv\Scripts\python.exe result/audits/version_compatibility_20261006/reproduce.py
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
.\.venv\Scripts\python.exe -m pytest test/test_joint_messages.py test/test_checkpoint_evaluation.py test/test_repository_contract.py test/test_structural_ablations.py test/test_ablation_protocol.py test/test_pareto_analysis.py -q --basetemp=result/audits/version_compatibility_20261006/pytest_tmp --junitxml=result/audits/version_compatibility_20261006/regression.xml
```

审查证据位于 `result/audits/version_compatibility_20261006/`：`reproduce.py`、`findings.json`、`regression.xml`、`regression_output.txt`。findings 保存源码指纹与构造来源路径，审计目录按现有仓库规则忽略。

源码与上一轮完整回归相同，本次未重复完整回归。上一轮为 545 passed、2 skipped、1 个历史 manifest 换行失败；详见 [联合消息验收](joint_messages_20261006.md)。该 LF/CRLF 原始字节问题与以上三个模型版本入口缺口分开记录。

此前报告的 README `--run-dir` 命令与 Pareto CLI 参数不匹配仍存在；本次作为已知文档/API问题记录，不重复列为新发现。版本说明旧页中有历史 schema 名称，应通过当前 schema-10 与联合消息文档判断运行边界。
