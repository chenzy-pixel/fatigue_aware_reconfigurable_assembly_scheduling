# DQN 与 A2C 补充训练

使用远端原训练 Python 环境，在项目根目录运行：

```powershell
python scripts/run_dqn_a2c_training.py
```

依次训练普通 DQN 的 11、23、37、71 四个种子，再训练 A2C 的 71。
每个种子 2000 episodes、100 epochs，每 5 epochs 验证一次，共 20 次验证。
训练与验证均保留原快照的并行配置（现有正式快照为 20 workers）；作业之间串行。
此入口只执行训练，正式测试需在训练完成后另行安排。

| 方法 | 配置和数据来源 | 结果目录（相对项目根目录） |
| --- | --- | --- |
| DQN | `result/experiment_suite_v1_parallel20` 中的原 Rule-DDQN 快照，加本次 DQN 实现 | `result/experiment_suite_v1_dqn/training/dqn/{11,23,37,71}/` |
| A2C | `result/experiment_suite_v1` 中的原 A2C 快照 | `result/experiment_suite_v1/training/a2c/71/` |

普通 DQN 使用相同的池化状态、两层 256 单元 MLP、去重调度规则动作、合法动作掩码、经验回放、epsilon 探索及目标网络同步配置。
其 TD 目标由目标网络直接选取合法动作最大值：`r + max_a Q_target(s_next, a)`，终止状态不自举。
方法标识为 `dqn`，模型族为 `rule_dqn_v1`；检查点与 DDQN 的模型族不同。
配置中的 `comparison.ddqn` 字段继续保存这组共享回放和同步参数。
DQN 作为补充实验单独保存身份与源码快照，后续统计需明确采用四个种子的样本量。

原始结果根目录必须保留 `preflight.json` 和 `source.zip`，项目的原固定数据文件必须齐全。
启动器先验证两个原始快照与固定数据，再验证实际运行身份。
DQN 在隔离副本上应用本次算法实现，并写入自己的 `preflight.json`、`source.zip` 和检查点身份；A2C 使用原始冻结代码。
因此，拉取当前代码不会改变 A2C 的原训练身份。

已有同身份的完成结果会复用。未完成请求按原训练器规则从该种子的第 0 个 episode 重训，已有权重不构成精确续训状态。
A2C 此前发生过系统内存分配失败，本入口保留原并行参数；启动脚本本身不代表该问题已经解决。
主要产物为 `training_log.csv`、`validation_log.csv`、`summary.json`、`best_checkpoint.pt` 和 `last_checkpoint.pt`。

仅检查归档、配置和数据文件（不启动 Torch 运行检查或训练）：

```powershell
python scripts/run_dqn_a2c_training.py --prepare-only
```

在远端训练环境验证运行身份并建立 DQN 预检记录，随后退出：

```powershell
python scripts/run_dqn_a2c_training.py --check-only
```

分别启动：

```powershell
python scripts/run_dqn_a2c_training.py --methods dqn
python scripts/run_dqn_a2c_training.py --methods a2c
```

原始结果位于其他目录时，使用 `--dqn-base-output-dir`、`--a2c-output-dir` 指向实际原始根目录；
使用 `--dqn-output-dir` 指定独立的 DQN 结果根目录。相对路径均相对 `--project-root`（默认脚本所在项目）。
身份不同的 DQN 结果需要使用新的结果根目录。
