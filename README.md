# 疲劳感知可重构装配批次调度

当前主线采用三因子27例设计：订单数12/24/48，每单工序3/4/5，机器4/8/12；工人与模块随资源规模派生。全部订单初始可用，全部工序完成即成功。优化Excess Flow、重构费用与geometric OWA劳动负荷，疲劳安全保持硬约束。

小、中、大规模按订单数12、24、48划分，每种规模包含9组工序数与机器数组合。工人数为机器数的3/4，模块数为机器数的3/2；加工、拆装、费用与疲劳参数采用统一范围。

- [完整规格](docs/factorial27_spec.md)
- [迁移、数据、启动与历史恢复](docs/factorial27_migration.md)
- [当前证明与边界](data/docs/BATCH_PROTOCOL.md)
- [27例参数表](data/tables/factorial27_parameters.csv)
- [公共参数](data/tables/factorial27_common_parameters.csv)
- [实际参考调度诊断](data/tables/factorial27_diagnostics.csv)
- [Epoch记录与三目标训练曲线](docs/training_history.md)
- [环境、图观测与数据生成审计](docs/batch_migration_audit_20261023.md)

使用工程的PyTorch环境：

```powershell
.venv/Scripts/python.exe -m scripts.audit_factorial27
.venv/Scripts/python.exe -m scripts.run_experiments --stage preflight --device cpu
.venv/Scripts/python.exe -m pytest test -q --basetemp=.pytest_batch13
.venv/Scripts/python.exe -m scripts.check_training_history --output result/smoke/epoch_history_v3_1
```

数据已冻结于`data/instances/v3`：标定54、验证27、正式测试27。当前使用数据schema3、图观测schema13及预测语义`order_chain_remaining_context_v3_1`。配置、缓存、标定来源与模型身份均进行校验；不同预测语义的模型不能直接混用。正式训练通过`scripts.run_experiments`或`scripts.comparison_suite`核验独立预算清单，具体命令见迁移文档。

每个epoch对应一次采样与学习批次，PPO内部优化轮数单独记录。逐算例保存三个目标值、决策次数、环境步数、奖励、耗时、停止原因、疲劳和工人负荷等；`episodes.csv`提供明细，`epoch_objectives.csv`提供按规模、参数组合与偏好分组的曲线数据。完整合法解与截断阶段值分开记录，完整目标缺失时留空。固定验证沿用交互预算20%、40%、60%、80%、100%的节点，结果保存到`validation_instances.csv`和`validation_objectives.csv`。

2026-10-23审计通过64项回归测试、108个固定实例证书重放，以及27个随机调度轨迹的25,761次交互与408个图快照检查。上述本地检查验证实现一致性，不表示训练收敛或方法优势；正式训练和搜索预算仍需标定。

旧代码、配置及3836个历史产物的身份见`result/migration/factorial27_20261023/before.json`与`source_before.zip`，旧测试和退休入口位于`archive/schema12`。复现历史实验需使用归档代码和原配置。

审计入口严格检查当前固定算例和证书；历史文件逐项核对已保存哈希，并报告已验证与未提供的数量。首次克隆未包含的本地历史缓存计入`protected_unavailable`，不计为已验证；任何现存文件哈希变化都会报错。
