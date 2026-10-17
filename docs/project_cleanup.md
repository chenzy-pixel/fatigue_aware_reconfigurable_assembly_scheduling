# 主线整理与路径迁移（2026-10-16）

2026-10-17补充当前对照入口；原整理快照、迁移证据和历史结果位置保留。

默认训练入口为 `train.py --config configs/default.json`；批次与预检使用 `python -m scripts.run_experiments --config <配置...> --seed 11 [--preflight-only]`。单目标位于 `configs/e1/`，对照位于 `configs/controls/`。

| 旧入口或配置族 | 当前入口 |
|---|---|
| run_main、run_flow_excess、run_shared_head、run_workload、run_schema12 等训练包装 | scripts.run_experiments，显式指定配置 |
| check_flow_excess_startup、check_remaining_seed11 | scripts.run_experiments --preflight-only |
| run_11、run_12 | scripts.run_experiments --suite structural / fatigue |
| run_13、run_14 | scripts.ablation_protocol / analysis.ablation_analysis |
| observation/screening/owa_log2、workload/q05/universal、flow_excess/shared_head/universal | configs/default.json |
| q、编码、variance、lambda 变体 | configs/controls/ 中的显式配置 |
| 日期化实验分析与诊断 | analysis/studies/ |
| 旧观测与资源审计的回归辅助函数 | test/support/ |
| NSGA-II-LS、MOEA/D、DCAN适配版、规则DDQN及新匹配PPO | scripts.comparison_suite；configs/comparators/；scripts/run_comparison_suite.ps1 |
| MO-ALNS分层预算与历史固定PPO比较 | scripts.mo_alns_study；新套件内ALNS校准另存alns_reference |

正式结果、模型、固定数据、尺度清单和 outputs 研究产物保持原位置。旧续训、固定批次和过程探针的原源码保存在忽略目录 `result/analysis/project_cleanup_20261016/source_before.zip`，其原实验设置须按保存配置读取。

完整逐路径迁移表、清理前工作区 patch、产物哈希清单和检查日志位于 `result/analysis/project_cleanup_20261016/`。旧协议的远端队列须在其原代码与配置下完成；此次整理只影响本地项目。


原有`train.py/eval.py`及PPO消融主线的actor统一为shared-head，旧参数化分支已从这些入口和矩阵中移除；对应主线加载schema12 shared-head checkpoint。已有shared-head模型的旧元数据字段在读取时转为通用字段，权重不变。新增suite使用独立DCAN和DDQN家族，按方法分派加载，不能用“主线只保留shared-head”推断项目没有其他对照网络。

本次删减前源码、历史消融清单及验证记录保存在忽略目录 `result/analysis/shared_head_cleanup_20261016/`，清单可在原实验代码中复核。

## 新对照的当前路径与执行身份

- 方法实现：`agent/comparators/`；精确交互训练与交付评价在`collection.py`、`learning.py`，进化搜索在`evolution.py`，模型在`dcan.py`、`ddqn.py`。MO-ALNS v2仍位于`agent/mo_alns/`。
- 批次与统计：`scripts/comparison_suite.py`及[PowerShell包装器](../scripts/run_comparison_suite.ps1)；`analysis/comparator_study.py`输出预算曲线、确认集交付配对、实例bootstrap和两类HV。
- 合同与说明：[实施契约](comparator_implementation_spec.md)、[远端启动](comparator_remote_startup.md)、[实现验证](comparator_verification_20261017.md)。原有[三种子脚本](experiment_three_seeds.md)只对应26个PPO配置；套件9次匹配训练与9次学习率试验另计。

用户在远端手动启动新批次，本机已取消的MO-ALNS长实验不续跑。套件先预检，再显式`--stage all --remote-run`；`all`停在正式清单，test/OOD/stress的`formal-evaluate`须满足预算和确认门槛后另行执行。完整请求可按同一身份跳过，中断训练从零重跑；源代码、配置、数据、硬件或并行宽度改变时使用新目录。

新套件主结果使用350411次实际交互训练的三个PPO模型；旧MO-ALNS研究固定历史seed11权重。两种研究及原有2000-episode主线分别保存身份，不迁移或重标旧成绩。现有本机测试与smoke仅证明链路，充分搜索和方法性能仍待远端结果。
