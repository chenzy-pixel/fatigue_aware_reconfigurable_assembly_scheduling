# 四项模块缺陷修复 · 2026-10-04

本次实现 schema 7、显式旧模型迁移、MO-ALNS 完整入口修复、PPO Dropout 限制及真实完成重构密度。结果表继续使用 7.0.0，策略头继续使用 V8。

## 当前行为

| 问题 | 修复行为 |
|---|---|
| 隐藏累计目标 | 全局观测由 11 维扩为 16 维：追加累计 Flow/尺度、累计成本/尺度、承诺负荷方差/尺度、剩余决策预算比例及剩余连续零时间动作预算比例。奖励与观测读取相同目标向量。 |
| checkpoint 兼容 | schema 5/6 默认拒绝；显式迁移执行 5→6→7 或 6→7，新输入权重及 Adam 动量补零，步数保留，检查参数映射。保存原文件哈希、源权重哈希、执行权重哈希及迁移链。 |
| 基线入口 | 搜索协议独立为 `mo_alns.protocol`，方法身份为 MO_ALNS_v1；修正回报重建、完整配置传递和聚合签名，两个直接脚本支持项目根目录执行。 |
| Dropout | 配置、直接网络构造、PPO 初始化及 checkpoint 推断均拒绝非零 Dropout。单实例评估在 eval 模式执行，并在异常时恢复原模式。 |
| 重构密度 | 由环境中的 DONE 重构统一生成 per-minute/per-operation 密度，分母分别为当前时间和实际已完成工序数。裁短 INS 继续参与实际工时与疲劳暴露，完成计数为零。零分母为 None。 |

全局特征名称及顺序集中于 `environment/observation_schema.py`，写入 network spec 并在构造、forward、value 与加载时检查。旧网络的低层构造须明确指定历史观测版本；正常工厂只接受 schema 7。

## 迁移接口

Python：`PPOAgent.load(path, allow_observation_migration=True, load_optimizer=False)`；训练初始化：`train(..., initial_checkpoint=path, allow_observation_migration=True)`。

CLI：训练初始化与评估均支持 `--allow-observation-migration`，配置快照重载使用同一个开关。`load_config(path, allow_observation_migration=True)` 只允许已知观测身份升级，其他运行身份保持严格校验。

新增全局编码器输入列以零初始化，原 schema 5 的时间上下文扩展仍按原规则补零。Adam 的 exp_avg/exp_avg_sq（及存在时的 max_exp_avg_sq）同步扩展；参数名或状态形状不匹配时拒绝加载。已迁移 schema 7 文件可直接重载，不重复补列。训练运行保存初始化文件、源权重、执行权重和迁移身份。

补零保持旧网络输出；它提供历史复评和初始化能力。正式新训练采用 schema 7，迁移初始化的实验须保留来源。

## 基线口径

MO-ALNS 保留原 22 点搜索网格、预算和搜索标量器。结果中的 search_scalarizer 明确记录尺度 1200/1000/50、augmentation=1e-4、epsilon=1e-6；公共评测质量采用当前冻结尺度。搜索分数与公共评测分数分别报告。完整比较 V8 与 MO-ALNS 仍需独立设计匹配预算。

真实入口：

```powershell
python scripts/mo_alns.py --config configs/baselines/mo_alns_smoke.json --dataset test --canonical-only --smoke
python scripts/mo_alns_benchmark.py --manifest configs/baselines/mo_alns_manifest.json --output-dir result/analysis/mo_alns
```

## 验收记录

- 合法历史回归：A/B 两种排序在 t=50 原有节点、边、11 维全局和动作特征一致；新增 Flow 特征分别等于 50/scale、70/scale。
- 动态状态检查覆盖工作分配、时间推进、任务失败结算、承诺负荷、预算消耗/恢复，以及复制、buffer、序列化和批量图打包。
- schema 5 的三种网络变体及 schema 6 的迁移保持输出一致，校验 Adam 动量、step、哈希、保存重载和错误映射拒绝；真实 schema-5 评估 CLI 覆盖普通 sampled 与 66×3 网格。
- 基线正式和 smoke 配置：微型固定实例、一偏好、8 次候选预算真实 CLI 求解并保存公共格式；manifest 入口完成一实例的全部 22 点。
- schema-7 微型训练：在线采集 2 episode、一次 PPO 更新，保存并重载 best/last，1 个固定实例的 13 点验证和 66 点最终网格，检查无调度违规。训练模板保留生成器要求的三波结构，验证/评估使用微型固定实例。
- 密度覆盖零时长截断、部分安装后截断、边界完成及零分母；独立核算三测试实例各两条轨迹（含失败）中的三目标。
- 黄金更新前已确认原 mask、211 步动作、奖励轨迹及成功终止保持一致；仅更新观测哈希和全局宽度，保留 schema-6 哈希。

最终全量回归：**348 passed、3 skipped，451.77 秒**。跳过项为两个默认 slow 审计及一个历史来源文件缺失的检查。补充接口/迁移/独立核算回归：**45 passed，78.64 秒**。`git diff --check` 通过。

测试文件：`test/test_module_design_repairs.py`、`test/test_module_entrypoints.py`，以及更新后的旧模型迁移/真实评估/黄金契约测试。完整机器可读记录为 `result/audits/module_repairs_pytest.xml`，补充记录为 `result/audits/module_repairs_targeted.xml`。

最终真实入口产物保存在 `result/audits/m7f/`：`test_real_schema7_training_upd0/runs/ppo/` 包含 best/last、更新日志、13 点验证和 66 点评估；两项基线配置测试目录内的 `runs/baseline/` 包含指标和搜索输出；manifest 测试目录内的 `benchmark/` 保存全部 22 点结果。各测试目录的 CLI 日志与微型算例保留，可按来源配置重跑。

这次验收验证实现与管线，不替代正式多种子性能实验。完整续训与运行启动时源码快照属于后续工程工作。
