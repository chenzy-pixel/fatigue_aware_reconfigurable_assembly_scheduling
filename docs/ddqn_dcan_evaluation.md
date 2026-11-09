# Rule-DDQN 与 DCAN 正式测试

`scripts/run_ddqn_dcan_evaluation.py` 使用已训练模型，依次完成 Rule-DDQN 和 DCAN 的正式评测。每个方法沿用自己的 `source.zip`、`preflight.json`、有效配置、五个种子和固定测试数据，由 `scripts/resume_full_evaluation.py` 恢复执行版本。

## 远端准备与启动

在远端项目根目录、已激活的 Python / PyTorch CUDA 环境执行：

```powershell
git pull --ff-only origin codex/experiment-suite-v1
python scripts/run_ddqn_dcan_evaluation.py
```

项目需保留 Git 中的固定数据和归一化文件，以及两份原结果目录：

```text
result/
  experiment_suite_v1_parallel20/
    preflight.json
    source.zip
    training/rule_ddqn/{11,23,37,53,71}/
  experiment_suite_v1_dcan_optimized/
    preflight.json
    source.zip
    training/dcan/{11,23,37,53,71}/
```

迁移到另一台电脑时，完整复制这两个结果文件夹到新项目的 `result/` 下。脚本使用启动它的 Python 解释器；若两个模型需要不同环境，可以在各自环境中使用下面的单方法命令。

## 执行范围与结果

- 先测试 Rule-DDQN，成功后测试 DCAN；同一时间执行一个模型、一个种子。任一步失败立即停止。
- 每个方法使用保存清单里的全部种子 `11,23,37,53,71`，覆盖 27 个固定测试算例。
- `frontier` 使用 66 个偏好、20 路环境和集中推理；`independent_request` 使用四个偏好，串行独占测时。
- 正式入口执行 `evaluate`。每个模型先通过原执行身份和训练完成校验；完整且身份匹配的测试单位会复用。
- 控制台实时输出执行进度，测试内容和运行环境记录写回各自结果目录。

```text
result/experiment_suite_v1_parallel20/
  evaluation/{frontier,independent_request}/rule_ddqn/<seed>/*.json
  evaluation_runtime/rule_ddqn_*.json
result/experiment_suite_v1_dcan_optimized/
  evaluation/{frontier,independent_request}/dcan/<seed>/*.json
  evaluation_runtime/dcan_*.json
```

每个方法完整覆盖应为 270 个测试 JSON（5 种子 × 27 算例 × 2 任务），两个方法合计 540 个。收回结果时，分别保留它们所在的原结果根目录关系。

## 单独执行与文件检查

单独测试其中一个方法：

```powershell
python scripts/run_ddqn_dcan_evaluation.py --methods rule_ddqn
python scripts/run_ddqn_dcan_evaluation.py --methods dcan
```

仅核对归档、训练完成记录与固定数据：

```powershell
python scripts/run_ddqn_dcan_evaluation.py --prepare-only
```

文件检查通过不代表模型已经完成加载或推理。CUDA 推理和完整测试的完成状态以实验机器实际生成的结果为准。两次运行的环境和硬件记录分别保存在 `evaluation_runtime/`，跨机器耗时比较需结合这些记录解释。

需要自定义结果位置时，可使用 `--rule-ddqn-output-dir` 和 `--dcan-output-dir`；相对路径以项目根目录为准。`--project-root` 可指定另一个完整项目位置。选中的结果目录必须各自包含原 `preflight.json`、`source.zip` 和相应训练模型。
