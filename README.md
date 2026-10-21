# Fatigue-aware reconfigurable assembly scheduling

**疲劳感知的可重构装配调度**

本项目研究多个模块化装配工位共享有限换型工人的多目标调度问题。机器通过更换工艺模块执行不同工序，普通加工由机器完成，工人负责模块拆卸与安装。调度需要综合考虑工序先后关系、设备兼容性、人员资质及疲劳积累与恢复，协调生产安排、机器重构、工人分配和等待时机，在订单流经时间、重构费用与人员劳动负担之间进行权衡。

主方法采用支持目标偏好输入的异质图 PPO，表示工序、机器、工人、订单和模块之间的关系，并依据仿真环境给出的合法动作生成调度方案。同一模型可在不同偏好下产生候选方案。仓库同时提供事件驱动仿真、启发式与强化学习对照、固定评测数据，以及从训练、选模到质量和耗时比较的实验流程。

## 方法与目标

| 内容 | 说明 |
|---|---|
| 主方法 | 偏好条件异质图 PPO，联合处理工序—机器配对、重构机器—工人配对及等待决策 |
| 优化目标 | Excess Flow、重构费用、基于实际拆装劳动时间的几何 OWA；劳动目标兼顾总投入与负荷集中程度 |
| 搜索对照 | MO-ALNS、NSGA-II＋局部搜索、MOEA/D |
| 学习对照 | DCAN 适配版、规则选择型 DDQN |
| 评价方式 | 多种子、固定实例与偏好网格，分别报告交付质量、完成覆盖和计算成本 |

当前[默认配置](configs/default.json)采用 schema12、log2 观测编码、HGNN shared-head 和 OWA(q=.5)，训练全程按固定七场景配比采样完整参数范围。目标定义、冻结尺度及训练参数见[主线实验协议](docs/experiment_protocol.md)。新增对照已完成本地集成检查，正式性能待远端实验验证，详见[验证记录](docs/comparator_verification_20261017.md)。

## 快速开始

以下为 PowerShell 示例，在项目根目录运行，并使用已激活的 Python 环境。默认配置使用 CUDA；正式训练在远端实验机器执行。

**1. 安装依赖**

```powershell
python -m pip install -r requirements.txt
```

**2. 预检配置、数据与运行环境**

```powershell
.\scripts\run_three_seeds.ps1 -Experiment universal -Python python -PreflightOnly
```

**3. 预检通过后，启动 Universal 三种子训练**

```powershell
.\scripts\run_three_seeds.ps1 -Experiment universal -Python python
```

脚本依次运行种子 **11、23、37**，任一失败即停止。`-DryRun` 可预览命令，`-List` 可查看实验配置。单目标、消融及其他启动方式见[逐实验运行清单](docs/experiment_three_seeds.md)。

## 实验与评估

当前固定 V2 数据包含 **500 个验证实例**，test、OOD、stress 各 **20 个实例**。另已物化[独立挑战集](docs/challenge_suite_protocol.md)：100 个瓶颈实例及 20 组四档重构倍率配对实例，供冻结模型后评估，当前完成数据审计与小规模链路检查。PPO 主线与对照套件分别冻结验证划分和预算；完整实验矩阵见[实验执行清单](docs/experiment_runlist.md)。

PPO 主线训练结束后自动测试验证选出的 best 模型。PPO/DCAN 主交付采用同实例、同偏好三次采样取优，DDQN 采用单次贪心；搜索方法先通过验证曲线确定合理预算，再另测 PPO 实际交付时限下的表现。选模和交付规则见[评估协议](docs/default_best_of_three.md)，独立测试命令见[运行清单](docs/experiment_three_seeds.md#独立评估)。

新增对照通过 `python -m scripts.comparison_suite` 统一运行，默认仅预检。匹配训练、搜索预算选择、独立确认及正式测试的分阶段操作见[远端启动说明](docs/comparator_remote_startup.md)。

结果保留配置、源码/数据/权重身份、训练与验证日志、原始候选及选中交付；交付质量与全部付费候选池的 Pareto/HV 分别报告。

## 文档导航

- **理解问题**：[工业背景](docs/research_background_cesa3r.md) · [数学定义与证明](data/docs/README.md)
- **了解实现**：[当前架构](docs/ARCHITECTURE.md) · [对照方法实施契约](docs/comparator_implementation_spec.md)
- **运行实验**：[完整实验清单](docs/experiment_runlist.md) · [三种子逐项命令](docs/experiment_three_seeds.md) · [对照远端启动](docs/comparator_remote_startup.md)
- **独立压力评估**：[瓶颈测试与重构倍率外推套件](docs/challenge_suite_protocol.md)
- **解释结果**：[主线协议](docs/experiment_protocol.md) · [三次取优与选模](docs/default_best_of_three.md) · [选模证明](data/docs/SELECTION.md) · [消融协议](docs/ablation_protocol.md)
- **维护与研究**：[路径迁移](docs/project_cleanup.md) · [Fluid 搜索研究设计（尚未实现）](docs/fluid_bottleneck_search_ppo_research_20261014.md)
