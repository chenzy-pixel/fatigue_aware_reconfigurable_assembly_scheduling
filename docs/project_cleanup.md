# 主线整理与路径迁移（2026-10-16）

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

正式结果、模型、固定数据、尺度清单和 outputs 研究产物保持原位置。旧续训、固定批次和过程探针的原源码保存在忽略目录 `result/analysis/project_cleanup_20261016/source_before.zip`，其原实验设置须按保存配置读取。

完整逐路径迁移表、清理前工作区 patch、产物哈希清单和检查日志位于 `result/analysis/project_cleanup_20261016/`。旧协议的远端队列须在其原代码与配置下完成；此次整理只影响本地项目。


当前 actor 统一为 shared-head，旧参数化分支已从训练、评估和消融矩阵中移除。历史模型与原始结果继续保留；当前运行时只加载 shared-head checkpoint。已有 shared-head 模型的旧元数据字段在读取时转为通用字段，权重不变。

本次删减前源码、历史消融清单及验证记录保存在忽略目录 `result/analysis/shared_head_cleanup_20261016/`，清单可在原实验代码中复核。
