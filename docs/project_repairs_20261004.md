# 项目审查缺陷修复（2026-10-04）

本次落实审查中六项缺陷及 V8 Pareto/HV 分析适配。修改沿用当前工作区的冻结尺度、失败惩罚 2、奖励公式、13 点验证和 66 点最终评测协议。

## 变更

| 项目 | 当前行为 | 主要位置 |
|---|---|---|
| 完成/决策上限 | 最后一道工序恰好在决策上限那一步完成时判成功；仍未完成时保持失败惩罚 | `environment/env.py` |
| 配置快照 | `load_config()` 重载有效配置，检查保存的 runtime identity；归一化清单用项目相对路径，旧绝对路径缺失时按名称定位并核验固定 SHA256 | `configs/config.py`、`configs/normalization.py` |
| checkpoint 迁移来源 | 加载前检查源文件权重哈希；记录源 checkpoint 和实际执行的 schema-6 网络哈希；迁移 provenance 校验两者并记录迁移信息 | `agent/ppo/agent.py`、`result/provenance.py`、`eval.py` |
| 固定实例测试 | 从已提交 YAML 读取，fixture 和黄金契约测试可在无 pickle 缓存的检出中执行 | `test/conftest.py`、`test/test_repository_contract.py` |
| 直接运行冒烟 | 根目录导入引导支持脚本形式和模块形式 | `scripts/run_00_smoke.py` |
| 匹配疲劳消融 | 清单统一管理九个 run、五个比较；三组 full 配置继承 neutral 的全部训练预算，仅开启完整疲劳动力学；训练、评测及汇总校验实际 checkpoint、预算、配置、数据及采样单元 | `configs/manifests/ablation_seed11.json`、`configs/ablation_protocol.py`、`scripts/run_11_*.py`–`run_14_*.py` |
| V8 Pareto/HV | 直接读取训练 final CSV 或独立 eval 输出；检查 66 点、采样重复、数据和尺度；逐实例过滤失败/违规轨迹并计算前沿、HV、覆盖率和训练种子均值/标准差 | `analysis/v8_pareto_analysis.py`、`analysis/pareto_analysis.py` |

provenance schema 升级为 1.1.0。普通 sampled CLI 在各重复之间复用同一加载完成的 agent，并与网格 CLI 一致记录迁移后执行网络的来源。

HV 采用 `J/(scale+J)` 和参考点 `(1,1,1)`。计算核改为沿 x 扫描并积分 y/z 矩形并集，适配每实例 66 × 3 个候选。原始三目标保留在明细中，前沿按实例和训练种子分别计算。报告同时保存每次重复的 HV 均值与合并重复的候选集 HV；失败/违规轨迹计入完成统计，前沿只包含成功且物理安全的轨迹。

方法比较要求同一数据 manifest、相同实例、偏好、重复预算、目标尺度及匹配训练种子；同训练种子的评测采样种子也须匹配。不同实例的目标向量分别求前沿后汇总。无疲劳场景的暴露数据继续用于情景对照，结构方法的 Pareto 分析使用完整疲劳环境。

## 使用

原有消融命令顺序保持可用。`run_12_train_neutral.py` 现在生成三组 full/neutral 配对，共六个 1000-episode 运行。`run_13` 在启动第一项评测前检查整个训练矩阵；重复运行时核验已完成评测的 checkpoint、配置及全部测试单元。`run_14` 在配对差值表外生成三个结构方法的 Pareto/HV 报告。

```powershell
python scripts/run_10_ablation_smoke.py
python scripts/run_11_train_structural.py
python scripts/run_12_train_neutral.py
python scripts/run_13_evaluate_ablations.py
python scripts/run_14_summarize_ablations.py

python -m analysis.pareto_analysis --run-dir result/runs/universal_seed11_ep2000 --output-dir result/analysis/v8_pareto
python -m analysis.pareto_analysis --run-dir main=result/runs/ablation_eval_universal_seed11 --run-dir no_graph=result/runs/ablation_eval_no_graph_seed11 --output-dir result/analysis/v8_comparison
```

同一方法的多个训练种子可重复使用同一标签，例如多次指定 `--run-dir main=...`。输出包含 `candidates.csv`、`pareto_front.csv`、`instance_summary.csv`、`seed_summary.csv`、`paired_instances.csv`、`summary.json` 和 `report.md`。

## 验收

- 首组针对性测试：44 项通过。
- 扩展回归：85 项通过、1 项历史来源校验跳过。
- V8 分析测试覆盖完整 66 × 3 单元、重复/缺失单元、失败/超限过滤、数据/尺度/配置身份、多个训练种子及已知 HV 值；扫掠计算与独立小网格矩形并集算法一致。
- 真实 CLI 集成：合成 schema-5 checkpoint 在一个合法微型固定实例上完成 198 条网格评测和 3 条普通 sampled 评测，均保存源权重、实际执行网络和迁移身份，零调度违规；输出随后进入 V8 分析。
- 现有 66 点真实冒烟评测输出成功通过新分析入口，生成逐实例前沿及 HV；该运行使用 validation 实例，报告标明 `independent_test=false`。
- 隔离检出常规全量测试：**319 passed, 3 skipped，460.80 秒**。两个 slow 测试按默认规则跳过，另一个历史来源校验缺少已归档文件。该副本启动时的 pickle/cache/checkpoint 数均为 0，依赖既有本机缓存的问题已消除。
- 补充最终控制变量及 checkpoint 检查：**18 项通过**；完成预算、配置、权重哈希、验证集和实际评测单元均有正例与拒绝用例。
- 最终 V8 分析回归：**17 项通过**，覆盖率按弱支配定义计入相等前沿点。
- 直接运行 `python scripts/run_00_smoke.py`：**通过**，耗时 302.19 秒，产物为 `result/runs/universal_protocol_smoke_20261004_174735/`。13 条验证和 66 条最终网格记录均零调度违规，best/last 文件存在，保存的配置通过标准入口重载。

验收日志和报告位于 `result/audits/project_repairs_20261004/`。隔离检出开始时不存在 pickle 缓存、checkpoint 或历史结果。微型数据和冒烟模型用于验证实现流程；正式配对训练及其性能结果需由上面的运行入口生成。
