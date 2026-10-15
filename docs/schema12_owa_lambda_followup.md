# OWA/log2：lambda 0.97 与 0.95 多种子补充实验

按以下顺序串行运行，均从头训练 2000 episodes，每 20 episodes 更新一次：

| 顺序 | 预设 | gamma | lambda | seed |
|---|---|---:|---:|---:|
| 1 | owa_log2_lambda097 | 1.0 | 0.97 | 11 |
| 2 | owa_log2 | 1.0 | 0.95 | 23 |
| 3 | owa_log2 | 1.0 | 0.95 | 37 |

参考已完成的 `schema12_screen_seed11_20261012_201659_owa_log2`。配置展开后仅实验名称、lambda 或训练种子不同；shared-head、schema12、log2、excess Flow、OWA(q=.5)、failure-v6、冻结尺度、学习率、验证与测试协议、completion-first checkpoint 规则均保持一致。新的训练种子会改变随机训练流；固定验证/测试协议保持不变。

## 远端串行启动

将 `scripts/run_schema12_lambda_followup.ps1` 复制到远端仓库的同名位置，然后在仓库根目录、原训练 Python 环境运行：

```powershell
.\scripts\run_schema12_lambda_followup.ps1
```

若没有激活虚拟环境，可显式指定：

```powershell
.\scripts\run_schema12_lambda_followup.ps1 -Python .\.venv\Scripts\python.exe
```

只检查不训练：

```powershell
.\scripts\run_schema12_lambda_followup.ps1 -Python .\.venv\Scripts\python.exe -PreflightOnly
```

脚本只依赖远端已有的 `run_flow_excess_training.py`，不要求远端先更新新的 `--experiment` 选项。它核对展开后的基线配置 SHA256，必要时写入 0.97 预设；如同名预设内容不符则停止。先对三组全部预检（含网络前向）再依次训练，每组训练包含原有 best 完整测试；任意一步返回非零退出码就停止队列。

预检不是训练完成。配置、检查报告、launch 状态与训练结果由原启动器保存在带时间戳的 `result/analysis` 或 `result/runs` 目录，重复执行会开始新实验，不会续训。

已同步新版 Python 启动器时，也可分别执行：

```powershell
python scripts/run_schema12_screening.py --experiment owa_log2_lambda097 --seed 11
python scripts/run_schema12_screening.py --experiment owa_log2 --seed 23
python scripts/run_schema12_screening.py --experiment owa_log2 --seed 37
```

分别执行时，应确认上一组成功后再启动下一组；自动串行运行优先使用上面的 PowerShell 脚本。

## 判读

0.97/11 与已有 0.95/11、0.99/11 作同种子筛选。0.95 的 11/23/37 三个训练种子用于衡量基线训练随机性。不能将单种子的 0.97 与三种子均值直接当作充分的配对优劣证明。

使用固定验证协议选模，再报告测试完成率、共同成功覆盖、共同 OWA 坐标下 Q/HV、Flow/Cost/OWA 与偏好响应；训练曲线同时检查优势标准差、explained variance、value loss、KL 和 clip fraction。无需为本实验修改原 checkpoint 规则。
