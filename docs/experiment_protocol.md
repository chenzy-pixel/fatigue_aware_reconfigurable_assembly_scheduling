# 当前实验协议

更新日期：2026-09-30。`configs/default.json` 是当前 Universal 协议的主配置，`configs/v8/universal.json` 继承它。单目标和 MO-ALNS 配置复用相同环境、奖励与冻结尺度。

## 固定尺度和质量

| 目标 | 冻结尺度 |
|---|---:|
| Flow | 1152.2093959731544 |
| Cost | 386.674652792805 |
| 工人负荷方差 | 4.937746913580247 |

清单为 `configs/manifests/e1_tail5_scales_20260928.json`，文件 SHA256 为 `23d85a70311feed66d679dd7e5a51b055a76984584350dd9fbbf0fc29554557b`。加载配置时校验清单及其哈希，训练期间保持固定。

每个尺度来自对应最近单目标实验第 840、880、920、960、1000 轮验证中成功轨迹目标均值的中位数。Flow 来源为相对工人时间实验；Cost 和 Variance 来源为各自 1000 轮实验。清单保留来源 run、五个均值与日志哈希；来源运行内容可由 Git 历史查阅。

偏好质量采用 `q_i = J_i/(s_i+J_i)`，再计算 `(max_i(w_i*q_i) + rho*sum_i(w_i*q_i))/(1+rho)`，其中 `rho=0.05`。该公式同时用于 PPO 偏好奖励、MO-ALNS 选解和结果分析。

`quality_score` 保留固定参考指标 `canonical_bounded_quality_v1`：尺度 `(1200,1000,50)`、权重 `(0.5,0.3,0.2)` 的有界加权和。`preference_quality_score` 使用本次冻结尺度和每条轨迹自身偏好；主实验解释使用原始三目标及偏好质量。

## 网络、奖励和预算

网络为 V8 HGNN actor-critic，schema-5 图包含六类节点和十二类关系。生产与工人采用 pair-plus-WAIT；WAIT 由精确进展证书控制。工人 Flow 专家采用 `candidate_zscore_v1`，标准差下限 `0.001`。正式执行模式为 `phase_batched_v1`，精度为 `float32`。

所有训练入口使用 `single_stage_progress_quality_failure_v2`：`r_t = delta_progress + Q_t - Q_(t+1) - failure_penalty`。任务失败仅在终止步扣 1，实际质量保持可重建；`gamma=1`、feasibility shaping 关闭。rollout cutoff 使用 critic 自举。

| 配置 | 训练轮数 | 验证间隔 | 训练 worker | 验证 worker | 每次 PPO 更新 episode |
|---|---:|---:|---:|---:|---:|
| Universal | 2000 | 100 | 20 | 20 | 20 |
| Flow | 1000 | 40 | 20 | 20 | 20 |
| Cost | 1000 | 40 | 20 | 20 | 20 |
| Variance | 1000 | 40 | 20 | 20 | 20 |

独立算法种子为 `11,23,37,53,71`。Universal 训练采用 20-episode 偏好块：三个单目标端点各两次，加十四个 scrambled-Sobol simplex 偏好。单目标训练、验证与最终评估使用对应 one-hot 偏好。

## 训练验证

Universal 每次验证为新版 500 实例验证池中固定分层选取的 50 个实例 × 13 个偏好 × 3 次 sampled 解码，共 1950 条轨迹。温度为 1.0，有序偏好坐标依次为 `(Flow,Cost,Variance)`：

| 编号 | 偏好 |
|---|---|
| 1 | (1, 0, 0) |
| 2 | (0, 1, 0) |
| 3 | (0, 0, 1) |
| 4 | (0.5, 0.5, 0) |
| 5 | (0.5, 0, 0.5) |
| 6 | (0, 0.5, 0.5) |
| 7 | (0.333333333333, 0.333333333333, 0.333333333333) |
| 8 | (0.6, 0.3, 0.1) |
| 9 | (0.6, 0.1, 0.3) |
| 10 | (0.3, 0.6, 0.1) |
| 11 | (0.3, 0.1, 0.6) |
| 12 | (0.1, 0.6, 0.3) |
| 13 | (0.1, 0.3, 0.6) |

checkpoint 采用 completion-first：先最大化所有验证偏好中的最低成功率；完成率在 `1e-12` 内平局时，最小化偏好等权质量。质量先在每个偏好的成功轨迹内求均值，再对偏好等权平均。任一偏好没有成功轨迹时取 `+inf`。完全平局保留已有 best；每个 run 保存 last。

验证采样根种子为 `algorithm_seed+100000+repeat`，每个 instance-preference 单元另通过 SHA256 派生独立 Torch RNG。

## 最终评估和分析

最终评估重载固定 best checkpoint，采用 `{(i/10,j/10,k/10): i+j+k=10}` 的 66 点网格，每实例每偏好采样三次，温度为 1.0。根种子为 `algorithm_seed+300000+repeat`。测试集、OOD 和 stress 均只读加载已物化数据，实例数量由实际评测命令确定并写入 provenance。

MO-ALNS 使用同阶段的偏好集合和冻结尺度，默认每偏好最多 300 次环境评估，输出每偏好一个档案选解。它使用求解器预算；PPO 使用每偏好三次采样预算。报告保存两种预算和求解时间。

结果 schema 为 `8.0.0`。逐轨迹 CSV 记录原始目标、偏好、成功/失败、安全检查、算法种子、重复索引、执行模式、尺度及清单哈希。分析校验每个实例、方法、种子的完整偏好/重复矩阵，拒绝重复单元、缺失单元和不匹配的协议数据。

Pareto/HV 使用安全成功候选、有界归一化和参考点 `(1,1,1)`。统计先按实例计算，再按算法种子聚合；方法配对以匹配的种子均值为单位。

```powershell
python -m analysis.pareto_analysis --candidate-csv result/runs/v8_universal_seed11/final_sampled_instance_metrics.csv --output-dir result/analysis/universal
python -m scripts.mo_alns --config configs/baselines/mo_alns.json --dataset test --algorithm-seed 11
python -m analysis.mo_alns_analysis --ppo-candidate-csv result/analysis/universal/candidates.csv --mo-alns-candidate-csv result/runs/mo_alns_test/instance_metrics.csv --output-dir result/analysis/ppo_mo_alns
```

训练、评估和分析从配置读取偏好集合、重复次数及尺度。固定数据和尺度清单的原始字节保持不变。

重新建立尺度清单时，`scripts.build_normalization_manifest` 要求显式传入 `--flow-run`、`--cost-run`、`--variance-run` 和新 `--output` 路径，以来源记录生成不可覆盖的清单。

## 算例协议 v2

详细参数、采样规则与验收见 [算例生成与训练分布 v2 实验规格](data_v2_specification.md)。默认数据目录为 `data/instances/v2` 和 `data/manifests/v2`，在线缓存为 `data/instances/train_cache/v2`。生成器与数据 schema 为 `2.0.0`；旧配置及数据哈希见 `configs/archive`。

候选只依据合法性、场景结构和必要可行性预检查接受。启发式诊断不筛选实例；截断实例标记 `unknown` 并保留，训练默认关闭诊断。主验证子集按目标场景比例分配为 3/18/7/7/5/5/5；49 个诊断实例每场景 7 个，与主验证互不重叠。诊断每五次主验证及训练结束运行，不参与选模。所有子集保存实际索引和实例、子集哈希。

训练采用 100-episode 窗口配额和固定线性课程，压力上限从 0.4 逐步增到 1.0。完成能力评测包含全部选中实例及未知实例；heuristic gap 仅在策略和参考均完整、安全完成时有效。成功质量、失败进度及原因分别报告；结果聚合校验数据协议、子集和尺度哈希。
