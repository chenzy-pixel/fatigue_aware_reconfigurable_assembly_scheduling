# Flow 新实验：性能优化与远端启动

2026-10-08 的本机工作只包括优化、回归和启动验证。正式实验在远端运行。

当前 Cost 完成后的 seed11 清单与顺序入口见
[七项后续实验](remaining_seed11_experiments.md)，包含主线 Variance、原归一化 Universal、excess Flow
和 shared_head 的三个单目标及 Universal。下文保留各独立入口的说明。

## excess 的含义

成功调度的 `Flow excess = 原始 Flow − 全部工序的最快加工时间之和`。
例如原始 Flow 为 1000 分钟、最快加工下界为 600 分钟，excess 就是 400 分钟。
这 400 分钟由订单等待和使用慢机器增加的加工时间构成。加工中采用比例抵扣，
真实失败继续加入现有罚项。

训练用 excess 及尺度 368.3143，使策略关注可改变的损失部分。单目标 Flow 在
成功调度上的排序仍与原始 Flow 一致；多目标偏好下的标量化权衡和训练信号
会改变。Cost 与工人负荷方差保持现有定义。报告中的原始 Flow 和主 HV 保持
统一口径，excess HV 为补充。完整契约见 [实验说明](flow_excess_experiment.md)。

## 本次优化

热点是完整 WAIT 副本的重复深拷贝。原实现先复制所有容器，又单独复制运行态、
日志和账本，还复制了随后立即清空的缓存。

现在运行态和账本各复制一次；静态 reset 数组共享只读使用；失效缓存直接清空；
诊断集合复制容器；执行日志按标量记录复制，遇到嵌套可变扩展时保留深拷贝。
事件的 payload 字典独立。实际 WAIT、事件处理、阶段切换、horizon/死锁终止
仍调用同一仿真内核。

原工作区的候选特征复用与批量计算、评估 lane 完成后立即补位、WAIT 证书缓存、
疲劳恢复二分和网络批量诊断继续使用。原候选参考函数与当前实现的 13 个状态
全部观测数组对照再次通过。

| 测量 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| 13 状态完整 WAIT 的中位耗时之和 | 153.32 ms | 28.38 ms | 减少 81.5% |
| 同状态完整冷观测中位耗时之和 | 771.65 ms | 669.77 ms | 减少 13.2% |
| 优化后单实例每步（raw/excess） | raw 25.38 ms | excess 26.54 ms | excess 约多 4.6% |

前两项在同一进程交替执行保留的旧函数与新函数，逐项核对投影内核状态及完整
观测数组；第三项两种模式各预热一个完整 episode，再交替测量五次、动作相同。
原先约 30% 是另一轮单实例测量，不能看作固定开销。以上局部测量不等于远端
完整训练的提速比例。基准与源码哈希保存在
`result/audits/flow_excess_optimization_20261008/`，预热明细在 `warm_timing/`。

## 远端启动

同步当前代码、`configs/flow_excess/`、新尺度清单和 V2 固定数据，使用远端自己的
Python/PyTorch 环境。在项目根目录先执行启动预检：

```powershell
python scripts/check_flow_excess_startup.py
```

它校验数据清单、excess/v4 身份、尺度和 CUDA，并用完整的 128 维、2 层网络
对两个配置实际前向；只产生预检报告。然后分别运行：

```powershell
python scripts/run_flow_excess_single_flow.py
python scripts/run_flow_excess_universal.py
```

也可用一个入口顺序跑完两个新实验：

```powershell
python scripts/run_flow_excess_training.py
```

这些入口只启动修改后的配置；需要 raw/excess 匹配对照时，使用原有
`run_flow_normalization_experiment.py` / `run_flow_single_flow_experiment.py`。

| 实验 | seed | 训练 episode | 每次 PPO 更新 episode | 验证间隔 | 正式验证 | 最终评估 |
|---|---:|---:|---:|---:|---|---|
| Flow 单目标 | 11 | 1000 | 20 | 40 | 50 实例 × 1 偏好 × 3 repeat | 20 实例 × 1 偏好 × 3 repeat |
| Universal | 11 | 2000 | 20 | 100 | 50 实例 × 13 偏好 × 3 repeat | 20 实例 × 66 偏好 × 3 repeat |

两者默认 CUDA、训练/验证各 20 worker，保留原网络、奖励失败惩罚、诊断和数据
预算，从新配置开始训练。启动入口支持 `--seed`、`--device`、`--parallel-envs` 和
`--validation-parallel-envs`；修改 worker 数仍保持每次更新 20 个 episode。

输出使用带时间戳的独立目录：`result/runs/flow_excess_seed11_*_flow/` 与
`*_universal/`。启动目录另存有效配置、`preflight.json` 和 `launch.json`；各训练
目录保存 terminal.log、训练/验证日志、checkpoint 和最终评估结果。顺序入口
在第一项失败时停止，保留另一项 pending 状态及失败日志。

本机已通过两个配置的 CUDA 前向预检。V2 validation/test 清单 SHA256 分别为
`5e973c436d14f079fdf8992c8ac772dc4e9efebf1e0477a38e13e7240f457c35` 与
`d42ade3bd2ea6cc328dc4a185e8787d4aa00fc11d7fce67319fd10684c4eccf3`。
本机正式训练没有启动。

本轮相关回归 **82 项通过**，覆盖比例抵扣、horizon/死锁/成功 WAIT、嵌套日志
隔离、时间上下文、资源投影、raw golden、原运行优化和远端启动入口。
`final_regression.xml` 与启动预检报告保存在本机验收目录。
