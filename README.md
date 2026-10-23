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

当前[默认配置](configs/default.json)采用图观测 schema13、log2 观测编码、HGNN shared-head 和 OWA(q=.5)，训练在线生成覆盖全部27种参数组合的算例。全部订单在时刻0可用，全部工序完成即成功，疲劳安全保持硬约束。目标定义、冻结尺度及参数见[当前实验规格](docs/factorial27_spec.md)。主线与对照已完成本地集成和一致性检查，正式性能仍待实验验证，详见[环境、图观测与数据审计](docs/batch_migration_audit_20261023.md)。

## 快速开始

以下为 PowerShell 示例，在项目根目录运行，并使用已激活的 Python 环境。默认配置使用 CUDA；正式训练在远端实验机器执行。本机也可将 `python` 替换为 `.venv/Scripts/python.exe` 使用项目环境。

**1. 安装依赖**

```powershell
python -m pip install -r requirements.txt
```

**2. 预检配置、数据与运行环境**

```powershell
python -m scripts.audit_factorial27
python -m scripts.run_experiments --stage preflight --device cpu
```

审计会重放当前固定实例的调度证书，并核对已提供的历史材料。首次克隆未包含的本地历史缓存会单独计数；当前固定数据或现存历史文件的哈希不一致会报错。正式训练前可用 `--device cuda` 检查实验机器。

**3. 启动三种子训练**

PPO、DCAN、Rule-DDQN统一按每方法、每种子2000次调度尝试训练，每批20例，共100个Epoch；在第5、10、……、100个Epoch验证，共20次。成功、物理失败和工程保护截断均计入尝试次数。以下命令在远端执行：

```powershell
foreach ($seed in 11, 23, 37) {
    python -m scripts.comparison_suite --stage train --method ppo --seed $seed --remote-run --output-dir "result/formal_episodes_seed$seed"
    if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $seed" }
}
```

以上依次运行种子 **11、23、37**，任一失败即停止。通过 `--config` 指定配置、`--method` 选择方法；当前[消融与对照配置表](configs/factorial27_ablations.json)列出可用变体。训练协议及验证间隔纳入运行与checkpoint身份；`--method dcan`、`--method rule_ddqn`使用相同训练例数。

## 实验与评估

当前采用三因子全组合设计：订单数 **12/24/48**、每单工序数 **3/4/5**、机器数 **4/8/12**，共 **27组**。小、中、大规模按订单数划分，每种规模包含9组工序数与机器数组合；工人数为机器数的3/4，模块数为机器数的3/2。其余参数共享统一范围，详见[27例参数表](data/tables/factorial27_parameters.csv)和[公共参数](data/tables/factorial27_common_parameters.csv)。

固定数据包含 **54个标定实例、27个验证实例、27个正式测试实例**，保存在 `data/instances/v3`，三个集合独立生成。训练算例在线构造，每27次算例启动覆盖全部组合。标定集用于开发与搜索预算研究，验证集用于选模，测试集用于最终评价；参考调度与可重放证书用于数据审计。

训练保存验证选出的 best 模型；正式测试通过独立 `evaluate` 阶段加载模型执行。PPO/DCAN 主交付采用同实例、同偏好三次采样取优，DDQN 采用单次贪心；搜索方法通过标定集研究计算预算。所有方法均记录完成覆盖、原始目标和计算耗时。当前选模、交付与独立测试命令见[实验运行说明](docs/factorial27_migration.md#正式实验入口)。

主线与对照通过 `python -m scripts.comparison_suite` 统一运行，默认仅预检；`calibrate-budget`、`train` 和 `evaluate` 分别执行预算标定、训练与评价，正式实验要求显式指定 `--remote-run`。当前数据使用 schema3，预测语义为 `order_chain_remaining_context_v3_1`；学习模型检查训练协议身份，搜索预算清单检查源码与数据身份。

结果保留配置、源码/数据/权重身份、训练与验证日志、原始候选及选中交付。每个采样与学习批次记为一个 epoch，逐算例记录三个目标、决策次数、环境步数、奖励、耗时、完成状态、疲劳与工人负荷，并导出分组 CSV。完整解目标与截断阶段值分别记录；固定验证在每5个Epoch学习更新后执行，并在训练末尾保证执行一次。训练环境交互、全部验证候选的环境交互及其耗时分别记录；完成计划仅表示训练额度执行完毕，收敛需依据曲线判断。论文曲线字段与统计口径见[Epoch与三目标训练记录](docs/training_history.md)。

历史 schema12 实验及独立挑战集按原配置和代码解释，归档位置与复现方法见[历史恢复说明](docs/factorial27_migration.md#历史恢复)。

## 文档导航

- **理解问题**：[工业背景](docs/research_background_cesa3r.md) · [数学定义与证明索引](data/docs/README.md) · [当前证明与边界](data/docs/BATCH_PROTOCOL.md)
- **了解实现**：[当前实现与迁移说明](docs/factorial27_migration.md) · [环境、图观测与数据审计](docs/batch_migration_audit_20261023.md)
- **运行实验**：[当前实验规格](docs/factorial27_spec.md) · [预算、训练与评价命令](docs/factorial27_migration.md#正式实验入口) · [消融与对照配置表](configs/factorial27_ablations.json)
- **解释结果**：[目标与参数定义](docs/factorial27_spec.md) · [Epoch与三目标训练记录](docs/training_history.md) · [参考调度诊断](data/tables/factorial27_diagnostics.csv)
- **历史实验资料**：[schema12架构](docs/ARCHITECTURE.md) · [对照方法实施契约](docs/comparator_implementation_spec.md) · [完整实验清单](docs/experiment_runlist.md) · [三种子运行清单](docs/experiment_three_seeds.md) · [对照远端启动](docs/comparator_remote_startup.md) · [瓶颈测试与重构倍率外推](docs/challenge_suite_protocol.md) · [历史主线协议](docs/experiment_protocol.md) · [三次取优与选模](docs/default_best_of_three.md) · [选模证明](data/docs/SELECTION.md) · [历史消融协议](docs/ablation_protocol.md)
