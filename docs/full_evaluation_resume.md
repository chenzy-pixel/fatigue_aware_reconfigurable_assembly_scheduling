# 补齐已训练 Full 的正式评测

`scripts/resume_full_evaluation.py` 从原实验目录的 `source.zip`、`preflight.json` 和固定数据准备隔离的执行副本，使用已保存的 Full 最佳模型完成正式测试。入口只调用 `evaluate`，覆盖原清单的全部种子、66 偏好前沿、四偏好独立请求以及 Full 的三种解码方式。

在远端项目根目录、已激活的原实验 Python 环境执行：

```powershell
git pull --ff-only origin codex/experiment-suite-v1
python scripts/resume_full_evaluation.py
```

如果原环境通过 `$dcanPython` 指定解释器，也可执行：

```powershell
& $dcanPython .\scripts\resume_full_evaluation.py
```

脚本在 `result/` 下查找 `training/full/<seed>/best_checkpoint.pt`。找到唯一训练根目录后，先核对各训练的完成记录、归档源码哈希、有效配置和固定数据，再在 `.cache/full_evaluation_*` 中准备执行副本。运行前必须通过原执行身份校验；正式评测入口在加载模型时继续核对检查点的训练协议和来源。

原训练的代码、预检、源码归档与模型保持原样。已有且身份匹配的完整测试单位由原评测器复用，缺失的单位继续求解。运行环境和硬件记录另存于原结果根目录的 `evaluation_runtime/`，便于解释本次测得的耗时。

## 明确选择结果目录

存在多份 Full 结果时，脚本会列出候选并停止。用 `--output-dir` 指定其中实际需要补测的原训练目录，例如：

```powershell
python scripts/resume_full_evaluation.py --output-dir result/experiment_suite_v1_parallel20
```

该路径应直接包含 `preflight.json`、`source.zip` 和 `training/full/`。参数也接受绝对路径。固定数据从当前项目的原路径读取，因此需保留完整的固定数据和冻结归一化清单。

仅检查归档、训练完成记录和固定文件准备时可执行：

```powershell
python scripts/resume_full_evaluation.py --prepare-only
```

此模式不加载深度学习运行时，也不证明模型可推理。实际运行时的有效身份或模型加载失败，会保留具体错误并停止；应解决原始输入或环境差异后再继续。

## 输出位置

控制台首先显示 `Original Full output:`，所有结果写回该目录：

```text
<原结果根目录>/
  evaluation/
    frontier/{full,full_greedy,full_sample1}/<seed>/*.json
    independent_request/{full,full_greedy,full_sample1}/<seed>/*.json
  evaluation_runtime/full_*.json
```

归档恢复与模型来源检查用于保持质量评测口径。硬件、Python 或 Torch 版本改变仍可能影响采样数值及求解时间，跨机器耗时应结合新保存的运行记录解释。

## 本地检查

辅助脚本及其文件准备测试仅依赖 Python 标准库：

```powershell
python -m unittest discover -s test -p test_resume_full_evaluation.py -v
```

单元检查覆盖真实的文件准备和错误处理，不执行正式训练或模型评测。CUDA 模型加载与完整测试以实验机器的实际输出为准。
