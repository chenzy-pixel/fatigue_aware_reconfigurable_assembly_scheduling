# V2 单目标 seed11 追加训练

2026-10-01。此阶段沿用 V2 的环境、奖励、网络、冻结尺度、固定验证数据和 completion-first 选模协议，
对应三个已完成的 `*_v2_seed11_1000` 实验。追加训练记录为独立阶段。

2026-10-02 主配置已更新为 `(1089.15,353.27,2.2629)`。本文和 `continue_v2.py` 描述的是已完成的原尺度追加阶段，入口继承源 run 的有效配置；新尺度训练需单独记录，旧 checkpoint 的归一化哈希不能直接替换。

## 阶段设置

| 目标 | 起始模型 | 起始模型轮次 | 新阶段 episode | 初始学习率 |
|---|---|---:|---:|---:|
| Flow | 原 run 的 best_checkpoint.pt | 840 | 500 | 5e-5 |
| Cost | 原 run 的 best_checkpoint.pt | 960 | 500 | 5e-5 |
| Variance | 原 run 的 best_checkpoint.pt | 1000 | 500 | 5e-5 |

新阶段 500 个 episode、每 20 个 episode 一次 PPO 更新，共 25 次更新。PPO epochs、clip、entropy 等参数复用原实验；
沿用原学习率 plateau 控制规则，新阶段初始化控制器。severity 全程为 1.0，压力分布固定为原课程末端：
easy 5%、balanced 35%、machine/reconfiguration bottleneck 各 15%，worker/fatigue/high-arrival 各 10%。

训练实例索引从原 run 最后一个已使用的实例索引之后开始，当前为 1000–1499，实例种子为 1001000–1001499。
`continue_v2.py` 的 runner adapter 将阶段内索引映射到这些新索引，阶段日志仍为 episode 1–500。
固定数据的 split 种子范围保持原值；直接修改 `dataset.splits.train.seed_start` 会改变 V2 分布契约哈希，
使原验证 manifest 不再兼容。

固定验证仍使用原 50 个分层算例 × 3 次 sampled 解码，根种子 100011、100012、100013，间隔 40 episode。
诊断子集 49 个算例，每五次正式验证及阶段结束运行。新阶段完成后，现有训练引擎用阶段 best 做最终测试。
跨阶段比较只使用固定验证指标：先完成率、后偏好质量，完全平局保留原 best。旧 run 和 checkpoint 保留。

## 在跑完原实验的电脑上启动

使用原来跑这三个 V2 实验的项目目录与 Python 环境。把 `scripts/continue_v2.py` 放到这个 V2 项目的 `scripts` 目录。
它复用现有训练引擎，不需要复制新的算例文件。原 run 应位于：

```text
result/runs/flow_v2_seed11_1000/
result/runs/cost_v2_seed11_1000/
result/runs/variance_v2_seed11_1000/
```

在该项目根目录执行：

```powershell
# 只准备和检查配置；不启动训练。
python scripts/continue_v2.py --check

# 三个目标依次追加 500 个 episode。
python scripts/continue_v2.py --run
```

仅训练一个目标：

```powershell
python scripts/continue_v2.py --run --objective cost
```

如果使用原项目的虚拟环境，将上述 `python` 替换为 `& .\.venv\Scripts\python.exe`。
若原结果保存在别处，加 `--run-root '原结果根目录'`。
脚本要求 V2 数据模块 `data/distribution.py`，并检查模型兼容性、完整权重、数据协议、冻结尺度和原验证子集。

## 输出和后续判定

```text
result/continuation/v2_seed11_best_plus500/
  flow_plus500.json
  cost_plus500.json
  variance_plus500.json
  continuation_plan.json
  runs/
    flow_v2_seed11_best_plus500/
    cost_v2_seed11_best_plus500/
    variance_v2_seed11_best_plus500/
```

每个新 run 保存常规训练/验证日志、best/last 模型、`continuation_lineage.json` 和
`continuation_comparison.json`。后者给出原 best 与新阶段 best 的完成率、质量比较，以及
`recommended_checkpoint`：只有新阶段按 completion-first 规则改善时才推荐新模型。

当前入口从已学到的网络权重继续学习，优化器、随机状态与训练控制器重新初始化。旧 checkpoint 缺少完整随机状态和控制器状态，
这个阶段应称为“从 best 开始的微调”，不能宣称为严格断点恢复。原 best 已经经历的训练轮数分别为 840、960、1000，
因此模型训练路径长度分别为 1340、1460、1500；计算预算则是在原 1000 episode 运行之后额外花费 500 episode。

追加 500 轮后，检查固定验证完成率、各压力类型的反复失败、严重未完成工序和订单重构等待。仍改善才进行下一阶段；
若尾部失败持续，应优先调整调度紧迫度/工人分配行为。跨算法种子的稳定性仍需独立实验。

## 实验变更记录

2026-10-01：在既有单目标实验上增加独立微调阶段。明确改变 episode 预算、初始学习率、课程锚点和训练实例索引；
保持问题定义、数据协议、奖励、网络、尺度与验证矩阵。每个阶段通过配置和 lineage 记录这些差异。

本机已检查三个目标的配置及完整权重，短运行用于验证入口；正式 500-episode 微调需在运行电脑上执行。
