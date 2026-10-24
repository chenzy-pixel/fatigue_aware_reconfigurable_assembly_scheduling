# Fatigue-aware reconfigurable assembly scheduling

**疲劳感知的可重构装配调度**

本项目研究多个模块化装配工位共享有限换型工人的多目标调度问题。机器通过更换工艺模块执行工序，工人负责模块拆卸与安装。调度同时考虑工序先后、设备兼容、人员资质及疲劳积累与恢复，优化Excess Flow、重构费用和实际劳动时间的geometric OWA(q=.5)。

主方法采用偏好条件异质图PPO。当前维护七个训练配置、五种子：Full、移除图传播、移除直接特征、移除残差、A2C、Rule-DDQN、DCAN；另有五条规则、NSGA-II-LS及MOEA/D。固定数据为54例标定、27例验证、27例测试；训练在线覆盖全部27种参数组合。数据schema3、图观测schema13，全部订单在时刻0可用，疲劳安全为硬约束。

## 环境与本地检查

在项目根目录、已激活的Python环境中执行；也可把python替换为`.venv/Scripts/python.exe`。

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.experiment_suite --stage smoke --device cpu
```

pytest验证当前模块和小实例机制；smoke以小网络、少量完整episode覆盖七配置的采集、更新、验证、模型保存与加载，以及五规则和两搜索算法。测试与smoke输出统一进入`.cache/`，不代表正式性能结论。

完整固定数据证书重放按需执行：

```powershell
python -m scripts.audit_factorial27
```

已发布固定数据位于`data/instances/v3`，清单位于`data/manifests/v3`。`python -m scripts.prepare_factorial27`在数据齐备时校验现有数据；仅在首次准备时生成固定实例和冻结尺度。

## 正式实验

默认结果目录为`result/experiments/`。先在实验机器做预检，再分别运行三个实验组：

```powershell
python -m scripts.experiment_suite --stage preflight --device cuda
.\scripts\run_main_experiment.ps1 -RemoteRun
.\scripts\run_ablations_a2c.ps1 -RemoteRun
.\scripts\run_rules_search.ps1 -RemoteRun
.\scripts\run_rules_search.ps1 -Stage report
```

| 实验组 | 内容 |
|---|---|
| 主实验 | Full、Rule-DDQN、DCAN，各五种子 |
| 消融与A2C | 三个结构消融和A2C，各五种子 |
| 规则与搜索 | 五条规则；NSGA-II-LS和MOEA/D的预算开发、冻结与正式评价 |

三个快捷入口调用同一个Python编排器，`-DryRun`查看任务顺序；`-Stage`选择单阶段。设备、worker和输出目录选项需在各组保持一致。Linux等环境直接调用`python -m scripts.experiment_suite`的相同阶段；具体命令见[实验运行说明](docs/experiment_suite_v1.md)。

每方法每种子训练2000次调度尝试，每批20例，共100个Epoch，每5个Epoch验证，共20次。默认训练与验证各20路环境，集中批量推理。PPO/A2C/DCAN交付使用同实例、同偏好三次采样取优，DDQN单次贪心。正式评价包含66偏好前沿和四偏好独立请求，统计以实例为独立单位。

控制台显示启动摘要、每批两行训练指标、验证摘要及最终模型位置；训练记录逐例三目标、完成状态、真实交互步数和耗时，保存CSV/JSON及best/last模型。报告输出逐实例结果、配对统计和SVG/PDF/PNG图。算法性能与预算充分性由正式实验结果判断。

## 文档与历史恢复

- [实验规格与问题定义](docs/factorial27_spec.md)
- [当前架构](docs/ARCHITECTURE.md)
- [正式实验清单与命令](docs/experiment_suite_v1.md)
- [Epoch与训练记录](docs/training_history.md)
- [数学定义与证明](data/docs/README.md)
- [工业背景](docs/research_background_cesa3r.md)
- [清理范围与历史恢复](docs/project_cleanup.md)

已有模型、数据和研究产物按生成它们的提交解释。当前执行身份为`experiment_execution_v2`；旧结果使用原提交读取，新运行使用新目录。
