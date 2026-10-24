# 实验训练控制台进度

2026-10-24。主实验及消融/A2C入口复用原`TrainingConsoleReporter`，恢复`[RUN]`、每Epoch的`[train]`、`[val]`和`[DONE]`输出。

## 显示内容

- 运行开始：方法/消融标签、种子、设备、采集并行数、训练次数与Epoch数、验证频率、验证轨迹数量及并行数。
- 每Epoch：完成率、平均奖励、工序进度、平均Q；PPO/A2C/DCAN的策略损失、价值损失、熵和KL；DDQN的TD损失、探索率。
- 阶段切换：worker启动、采集、更新和验证在开始与结束时各打印一次。结束行显示已耗时；采集/验证的完成数量、活动环境、实际步数来自执行回调。
- 验证结束：交付完成率、偏好平衡Q、安全状态、检查点事件、最佳episode，另列原始轨迹和交付数量。
- 模型保存：文件写入完成后打印`[checkpoint]`事件、Epoch和已保存的检查点路径。
- 运行结束：耗时、最佳指标、检查点及结果目录；复用已完成训练时明确打印复用状态。

日志由执行线程在活动节点打印；阶段内部的回调只更新计数，阶段退出时统一展示最终值。逐步worker磁盘日志仍由原调试开关控制。

## 真实小规模联调

命令：

```powershell
.\.venv\Scripts\python.exe -m scripts.check_console_progress --output result/verification/console_progress_events_20261024/live
```

本地CPU、2个worker、缩小HGNN网络，2次训练尝试、1个Epoch、1次单实例单偏好验证；训练使用既有在线生成器，验证实例截取一个订单。该探针用于界面与调用链核验。

训练615步、验证69步、原始验证轨迹3条；完成一次训练批次和一次验证。完整输出见[实际控制台记录](../result/verification/console_progress_events_20261024/live_console.txt)，计数见[联调摘要](../result/verification/console_progress_events_20261024/live/console_probe.json)。其中片段为实际输出：

```text
[collect] start | epoch 1/1 | episodes 0/2 | active=0 | steps=0 | elapsed=0s
[collect] done | epoch 1/1 | episodes 2/2 | active=0 | steps=615 | elapsed=7s
[update] start | epoch 1/1 | elapsed=0s
[update] done | epoch 1/1 | elapsed=1s
[train] ep 2/2 | upd 1 | complete 100.0% | reward 0.759 | progress 1.000 | quality 0.241
        p_loss -0.0032 | v_loss 0.002 | ent 0.57 | kl 0.0001 | lr=1.0e-04
[val] start | epoch 1/1 | rollouts 0/3 | active=0 | steps=0 | elapsed=0s
[val] done | epoch 1/1 | rollouts 3/3 | active=0 | steps=69 | elapsed=0s
[val] ep 2 | best_of_3_v1 complete 100.0% | quality 0.003293 | progress 1.000
      safety=PASS | event=best_initialized | best_ep=2
      requests=3 | deliveries=1 | raw complete=100.0% | raw Q=0.006263
```

## 回归证据

- [活动节点显示与训练回归](../result/verification/console_progress_events_20261024/tests.xml)：30 passed，包括阶段开始/结束、真实计数、异常退出、检查点路径、DDQN预算，以及2000次尝试、100Epoch、20次验证计数。
- 显示测试通过模拟阶段持续60秒及多次计数更新，核对整个阶段只在进入及退出时打印；正常退出与异常退出分别核验。

控制台记录保留本次实际捕获的输出。源码变化后，应按当前源码重新预检。

源码见[控制台报告器](../result/console_reporter.py)、[训练接入](../agent/comparators/learning.py)、[采集回调](../agent/comparators/collection.py)、[评测回调](../agent/ppo/parallel.py)。

## 运行版本

运行中的进程使用启动时加载的版本。此修订纳入源码身份，下一次运行时在新输出目录重新预检。启动方式见[实验体系](experiment_suite_v1.md)。
