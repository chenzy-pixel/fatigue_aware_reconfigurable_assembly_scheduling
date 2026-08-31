# 疲劳感知可重构装配调度（PyTorch + PPO）

本项目是一个可直接运行、面向论文复现的 PyTorch + PPO 调度系统，覆盖固定算例与主动压力数据生成、离散事件仿真、疲劳约束、双资源匹配准入、异质图策略网络、两阶段完成约束训练、固定数据集评估、可复现审计和结果持久化。

`data/instances/fixed_instance.yaml` 是固定标准算例的数值唯一事实源。运行时代码不依赖 `docs/dev_context/` 中的开发上下文或项目根目录下的 PDF；二者仅用于开发和论文背景参考，可独立删除。

当前默认配置 `configs/default.json` 是 `M1_candidate_graph_v6` 稳定基线。主线同时提供通过 `extends` 继承默认配置的 v7 C0/E1/E2 系列实验：`configs/v7/c0_v6_control.json` 保持 v6 策略头，`configs/v7/e1_context_exception.json` 只加入 E1 有界上下文残差，E2 系列逐步验证偏好条件、Pareto 审计和匹配安全控制。C0/E1 的正式协议与结果 schema 仍为 `v7_e1_protocol_v2`/`4.1.0`；E2、E2.2、E2.3 分别使用 schema `4.2.0`、`4.3.0`、`4.4.0`，不跨实验迁移 checkpoint。

## 数据依赖

- `data/instances/fixed_instance.yaml`：固定算例与随机实例生成模板，由配置项 `paths.fixed_instance` 指定，属于必需输入。
- `data/instances/fixed_15x4_v1.pkl`：由固定算例生成的本地缓存，可删除并重新生成，不是事实源。
- `data/manifests/` 与对应的 `data/instances/{validation,test,ood,stress}/`：固定数据集评估所需；仅运行固定算例或在线生成训练实例时不依赖这些集合。
- `docs/dev_context/` 和 `可重构人机协同装配系统.pdf`：非运行时依赖，不参与配置加载、实例读取、训练、评估或测试。

## 建模边界

- 8 台机器与工位一一对应，支持三种模块中的两种。
- 6 名工人只参与模块拆卸和安装，普通加工不占用工人。
- 15 个订单各有 4 道串行工序，按三个波次释放。
- 环境内部使用 0.1 分钟整数 tick，所有时长向上量化。
- 策略 1 选择“工序–机器”或 `ADVANCE`；策略 2 选择“重构事件–工人”或 `ADVANCE`。
- 安装工人不在拆卸开始时预留；拆卸完成后重新决策。
- 随机实例会同步重建波次订单清单和实际释放区间，实例校验拒绝波次引用不一致的数据。
- 主动压力生成器支持 easy、balanced、机器瓶颈、重构瓶颈、工人瓶颈、疲劳瓶颈和高到达压力；ID 实例保持固定机器适用性和工人资格矩阵。
- 生成实例使用规范化 JSON 与 SHA-256 manifest；验证、测试、OOD 和 stress 集固定存盘，训练可按课程分布在线生成。
- 每个生成候选都会检查波次与释放区间、资源可行性、主导模块比例、串行前序无环、数值有限性以及启发式调度可行性；train/validation/test/OOD 必须在 240 分钟内完成，stress 允许最多 20% 截断。
- 到达计划期时结算正在进行的拆装负荷与疲劳；决策数/零时间动作保护触发时在当前时刻截断，不跳过未来事件并推进到计划期末。
- 评估保留总流经时间、重构成本和工人负荷方差三个原始最小化目标；正式模型选择先约束完成率，再使用不可变的有界质量指标。经验 Pareto 分析仍直接使用三个原始目标，不把加权分数解释为真实 Pareto 最优。
- 环境返回轻量 NumPy 异质图观测，包含工序、机器、工人三类节点和五类关系；能力边的 `EST` 表示计入已有机器承诺与必要乐观重构后的最早加工开始时刻。
- 默认策略网络是纯 PyTorch 两层异质关系 GNN；原类型感知 MLP 以 `typed_mlp` 编码器保留，供消融和旧 checkpoint 复评。

## 目录

```text
configs/                 默认配置与 JSON 继承加载器
  v7/                    C0、E1 有界残差与 E2 偏好条件配置
agent/
  ppo/                   类型编码器、双策略头、Critic、GAE、PPO
  baselines/             启发式与掩码随机策略
data/
  generate_orders.py     固定缓存、压力实例和固定数据集生成
  dataset.py             规范化记录、manifest 与哈希校验加载器
  manifests/             validation/test/ood/stress 固定集合清单
  instances/             标准算例、固定缓存及按集合划分的规范化 JSON 实例
docs/
  dev_context/           可选且可删除的建模与图结构开发上下文
environment/             实体、状态、动作掩码与离散事件环境
test/                    数据、环境、端到端与 PPO 测试
result/
  runs/                  训练/评估运行目录
train.py                 PPO 训练入口
eval.py                  heuristic/random/PPO 评估入口
e1_reproducibility_audit.py  C0/E1 串行/并行可复现性审计
pareto_analysis.py       C0/E1 经验 Pareto 前沿与 Hypervolume 分析
e2_preference_analysis.py E1/E2 等预算偏好响应与经验 Pareto 分析
benchmark_parallel.py    并行 rollout 吞吐基准
```

## 安装

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install setuptools==80.9.0
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --no-build-isolation
```

如果系统没有 `python` 命令，可用本机 Python 3.12 的完整路径创建 `.venv`。
Visdom 0.2.4 的源码包仍依赖 `pkg_resources`，因此先固定 setuptools 并关闭
构建隔离，避免新版 setuptools 的隔离构建环境缺少该模块。

## 策略网络

默认网络配置为：

```json
"network": {
  "encoder_type": "hetero_gnn",
  "hidden_dim": 128,
  "message_passing_layers": 2,
  "dropout": 0.0
}
```

三类节点先独立投影到统一维度。每层为五类概念关系分别学习
`Linear([源节点嵌入, 边特征])`；能力边和锁定边展开正反向消息并共享该
关系的权重。到达同一节点的消息按入度取均值，随后执行
`Dropout(ReLU(LayerNorm(残差 + 均值消息)))`。实现只依赖 PyTorch，
不需要 PyTorch Geometric。

生产头按工序优先顺序评分 `(工序, 机器)`；工人头通过机器的唯一锁定边
恢复工序后，按机器优先顺序评分 `(工序, 机器, 工人)`。Critic 分别均值
池化工序、机器和工人节点，再与全局特征拼接。

### 策略头 v6（默认基线）

默认 `M1_candidate_graph_v6` 使用独立、带符号的 softplus 权重进行候选内
多目标排序：

\[
z_i=\sum_j s_j\operatorname{softplus}(\theta_j)\tilde x_{ij}
+\sigma(g_c)\overline c+\sigma(g_r)\widetilde c_i.
\]

生产候选的 processing time、reconfiguration time、拆装固定成本、预计劳动
成本和预计停机成本均为惩罚项，初始有效权重依次为
`-0.30/-0.30/-0.20/-0.20/-0.20`；horizon slack 为收益项，初始权重
`+0.30`。工人候选的阶段时长、预计疲劳比例、增量劳动成本、增量停机成本
和增量负荷方差均为惩罚项，初始有效权重依次为
`-0.30/-0.30/-0.20/-0.20/-0.20`。每个 raw 参数彼此独立，softplus 与固定
符号保证训练不能把成本学成奖励，或把 slack 学成惩罚。

标准化只使用当前合法 pair；少于两个合法 pair 或某维零方差时，该维贡献
为零，masked 候选不会影响合法候选排序。公共上下文和逐候选中心化残差分别
使用 sigmoid 门控，两个生产门控和两个工人门控均以 logit `-4.0` 初始化；
contextual scorer 的输出层零初始化，使训练初期由显式单调特征主导。
生产动作语义仍为 `pair_plus_defer_v1`，环境边特征未变，因此
`observation_schema_version` 保持为 3。

v6 checkpoint 的 `network_spec` 保存完整特征顺序、参数化方式、上下文模式
和门控初始化。v5 checkpoint 不支持自动转换，会在加载 state dict 前报告
必须重训；不得迁移网络权重或优化器。历史 `credit_assignment_*.json` 和已有
结果保持 v5 语义，可检出提交 `002f13d` 复现。每次 PPO 更新记录 11 个
`policy_head_weight_*` 和 4 个 `policy_head_gate_*` 指标；这些指标同时写入
`update_log.csv`、`summary.json` 和 checkpoint metadata。

### 策略头 v7：E1 有界上下文残差

`configs/v7/e1_context_exception.json` 在 v6 单调 ranker 上只增加有界残差：

```text
sigmoid(gate) * residual_scale_ratio
* max(std(relative_logits), 1e-3)
* tanh(raw_residual)
```

E1 使用 `gate = -2`、`residual_scale_ratio = 2`，并关闭不能改变 worker
pair-vs-`ADVANCE` 竞争的公共偏置。残差的量级由当前候选 ranker logits 的
标准差约束，保留显式单调排序作为主信号。对应的 C0 配置继续使用 v6 策略头，
但共享相同训练与评估协议；两个配置均保持
`training.forced_action_compression = false` 和
`non_delay_worker_dispatch = true`。

v7 checkpoint 使用 `observation_schema_version = 4`。加载器严格比对
`network_spec`、策略头版本、动作语义及节点/边特征维度；v6 与 v7 权重不会
相互转换，也不会部分加载。完整设计边界见 `E1_POLICY_HEAD.md`。

做 MLP 消融时复制实验配置并把 `network.encoder_type` 改为
`typed_mlp`；`message_passing_layers` 和 `dropout` 对该基线不生效。
缺少 `encoder_type` 的旧配置按 `typed_mlp` 解释。v6 checkpoint 的
`network_spec` 同时保存编码器结构、节点/边特征维度和
`observation_schema_version = 3`。加载器会从旧 state dict 推断 schema v1
维度；若与当前观测不一致，会报告明确的 observation schema incompatibility，
不会部分加载。旧最佳 checkpoint 只复用既有 E0 指标作基线，新训练从头初始化。

### E2：偏好条件化多目标策略

`configs/v7/e2_preference_conditioned.json` 继承 E1，并把 episode 级偏好
`(w_flow,w_cost,w_variance)` 通过独立两层编码器接入 production/worker
Actor、defer/advance head 和 Critic。训练偏好采用 70% 均匀单纯形
`Dirichlet(1,1,1)` 与 30% 顶点/中心/canonical 锚点混合采样；采样种子只由
算法 seed 和 episode index 派生，不受并行环境数影响。

E2 checkpoint 使用 observation schema 5，不能与 E1 部分加载。质量阶段按
episode 偏好标量化，但正式 `quality_score`、周期验证和 checkpoint 晋升仍固定
使用 `0.5/0.3/0.2`。完整契约见 `E2_PREFERENCE_CONDITIONING.md`。

### E2.2：分层 commit 与偏好候选选择

`configs/v7/e2_2_hierarchical_preference.json` 保留 E2.1 的五锚点训练、
22 点验证网格、Tchebycheff 标量化和 Pareto 晋升门槛，但把 production Actor
改为两层分布。第一层独立决定 `commit/defer`，第二层只在 commit 后对合法
production pair 做条件 softmax；`direct_main_rank_v1` 直接偏好分数只进入
第二层，因此候选数量、候选 logit 公共偏移和偏好尺度不会改变 commit 总概率。
sampled PPO 仍使用相同的扁平动作 ID 和联合 log-prob，greedy 则先解码门控、
再选候选 top-1。E2.2 使用 `v7_e2_2_pareto_protocol_v1` 和结果 schema
`4.3.0`，checkpoint 与 E2.1 明确不兼容。

E2.2 结果持久化 `preference_override_count`、
`preference_override_rate` 和 `mean_preference_logit_std`；跨实例汇总按
`ranker_top_decision_count` 加权。本机只执行 pytest 契约验证，正式 seed 11
训练应在远端运行：

```powershell
.\.venv\Scripts\python.exe train.py `
  --config configs\v7\e2_2_hierarchical_preference.json `
  --algorithm-seed 11 `
  --run-name v7_2000_e2_2_hierarchical_seed11
```

远端验收要求 canonical validation 连续三次 100%、生成
`accepted_checkpoint.pt`、最终 22 点 validation 440/440 完成且无截断、
无调度违规并满足疲劳约束；在此之前不进行正式 test Pareto 分析。

### E2.3：安全生产偏好与可恢复匹配

`configs/v7/e2_3_safe_production_preference.json` 是独立实验，不改写 E2、
E2.1、E2.2 的配置或历史结论。它回到 E2.1 的扁平
`pair_plus_defer_v1` production 动作语义；直接三目标偏好只进入 production
候选排序，worker scorer 仍可学习偏好条件，但不再叠加直接偏好 logit。

worker 动作使用 `matching_admission_recovery_v2`：零 matching deficit 时只
保留仍为零的动作，正 deficit 时只保留严格减小 deficit 的恢复动作；有恢复
pair 时屏蔽普通等待，没有即时恢复 pair 但存在未来事件或疲劳恢复时才允许
推进。production commit 同时对当前拆装任务和所有未来安装任务做联合安全
匹配，避免先选低成本重构、后续却无法形成完整工人匹配。

canonical validation 连续三次 20/20 只保存 `phase1_checkpoint.pt` 并进入
quality，不再直接接受。`accepted_checkpoint.pt` 只能由每 20 个 quality
update 的完整 22 点审计产生，且必须精确覆盖 20×22=440 个候选、全部完成、
零截断、零调度违规、满足疲劳线，并通过平均唯一动作轨迹数 8、唯一目标
向量数 8、非支配点数 4 三项最低可控性门槛。后续晋升继续沿用 E2.1 的
Hypervolume 改善与 canonical quality 容差。结果 schema 为 `4.4.0`，记录
production/worker 分头偏好覆盖、matching deficit 与未来安装联合准入诊断；
worker direct-preference override 必须为 0。

本机只运行静态检查和 pytest 契约测试，不运行 smoke、训练或 validation
rollout。远端 seed 11 启动命令为：

```powershell
python train.py `
  --config configs\v7\e2_3_safe_production_preference.json `
  --algorithm-seed 11 `
  --run-name v7_2000_e2_3_safe_production_seed11
```

正式验收还要求 `validation_worker_bottleneck_2000003` 的 22 个偏好全部完成、
matching deficit 不再形成不可恢复等待链，且三项反坍缩门槛全部通过。在这些
条件满足前不开展正式 test Pareto 分析。

## 运行

```powershell
.\.venv\Scripts\python.exe data\generate_orders.py --config configs\default.json
.\.venv\Scripts\python.exe eval.py --config configs\default.json --dataset validation --policy heuristic
.\.venv\Scripts\python.exe train.py --config configs\default.json --smoke
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest --runslow -m slow -q
```

### v7 C0/E1 训练与评估协议

先运行 v7/E1 契约测试和最小训练：

```powershell
.\.venv\Scripts\python.exe -m pytest test\test_v7_policy.py test\test_v7_experiments.py test\test_e1_protocol_v2.py -q
.\.venv\Scripts\python.exe train.py --config configs\v7\e1_context_exception.json --smoke --run-name e1_mainline_smoke
```

正式的同协议 seed 11 C0/E1 训练使用各自配置；两者都从头训练，不能互相加载
checkpoint：

```powershell
.\.venv\Scripts\python.exe train.py --config configs\v7\c0_v6_control.json --algorithm-seed 11 --run-name v7_2000_c0_seed11
.\.venv\Scripts\python.exe train.py --config configs\v7\e1_context_exception.json --algorithm-seed 11 --run-name v7_2000_e1_seed11
```

正式论文质量指标固定为：

```text
Q_paper = 0.5 * F / (1200 + F)
        + 0.3 * C / (1000 + C)
        + 0.2 * V / (50 + V)
```

`Q_paper` 越小越好。`canonical_bounded_quality_v1` 的尺度和权重不可由实验配置
改写；`quality_metric_sha256` 防止混合聚合不同指标。训练奖励中的
`reward_quality_score` 作为诊断保留，正式报告使用 `quality_score`。评估结果
使用 schema `4.1.0`。

sampled 评估采用 `per_instance_sha256_v1`：每个实例根据稳定
`instance_id` 和指定 sampling seed 获得独立 `torch.Generator`，因此
`parallel_envs = 1/2/10` 消耗完全相同的随机流；逐实例结果同时写入
`action_trace_sha256`。checkpoint、训练摘要和评估指标携带源码、有效配置、
数据 manifest、固定模板、Git 状态、协议、评估器和 checkpoint 哈希。

### C0/E1 可复现性审计与经验 Pareto

`e1_reproducibility_audit.py` 默认读取上述两个 run 的
`accepted_checkpoint.pt`，在固定 validation 集上核对 greedy 串行结果、
三个固定 sampled seed（100011/100012/100013）以及 1/10 worker 并行复现，
并确认 sampled 评估不修改全局 Python/NumPy/Torch RNG：

```powershell
.\.venv\Scripts\python.exe e1_reproducibility_audit.py
```

审计通过后，可按固定实例提取经验非支配前沿，并使用规范化目标与固定参考点
计算三维 Hypervolume：

```powershell
.\.venv\Scripts\python.exe pareto_analysis.py `
  --audit-dir result\audits\<timestamp>_c0_e1_validation `
  --output-dir result\analysis\c0_e1_pareto_seed11
```

分析只读取 `greedy_serial` 和 `sampled_serial`，并用动作轨迹哈希核对后排除
并行复现副本。三个目标均按最小化处理；主结果按 `instance_id` 分别构建
前沿，跨实例均值前沿仅作为辅助诊断。输出包括逐候选/逐前沿/逐实例 CSV、
`summary.json`、英文报告，以及 PDF 和 300 dpi PNG 图。该结果是单训练 seed
的 validation 工程预检，不证明真实 Pareto 最优，也不支持多 seed 显著性
结论。实现测试可单独运行：

```powershell
.\.venv\Scripts\python.exe -m pytest test\test_pareto_analysis.py -q
```

### E2 训练、指定偏好推断与等预算 Pareto

```powershell
.\.venv\Scripts\python.exe -m pytest test\test_e2_preference.py -q
.\.venv\Scripts\python.exe train.py --config configs\v7\e2_preference_conditioned.json --smoke --run-name e2_preference_smoke
.\.venv\Scripts\python.exe eval.py --config configs\v7\e2_preference_conditioned.json --dataset test --policy ppo --checkpoint result\runs\v7_2000_e2_seed11\accepted_checkpoint.pt --preference 0.5 0.3 0.2
```

正式 E1/E2 对照为每实例 22 次等预算推断：E2 使用 21 个分母为 5 的单纯形
格点加 canonical 点，E1 使用一次 greedy 加 sampled seeds 100001--100021。
五个算法 seed 及 test/OOD/stress 的示例 manifest 位于
`configs/v7/e2_analysis_manifest.example.json`：

```powershell
.\.venv\Scripts\python.exe e2_preference_analysis.py --manifest configs\v7\e2_analysis_manifest.example.json --output-dir result\analysis\e1_e2_equal_budget
```

输出包括逐候选/逐前沿/逐实例/逐 seed CSV、三维 Hypervolume、偏好响应
Spearman、单目标顶点相对 canonical 的变化、exact Wilcoxon、PDF/PNG 图和报告。
这些是经验 rollout 前沿，不作真实 Pareto 最优声明；固定权重单点成绩单独报告。

构建并使用主动压力数据集：

```powershell
.\.venv\Scripts\python.exe data\generate_orders.py --config configs\default.json --build-all --profile dev
.\.venv\Scripts\python.exe train.py --config configs\default.json --smoke
.\.venv\Scripts\python.exe eval.py --config configs\default.json --dataset validation --policy heuristic
```

`dev` 生成 validation/test/OOD/stress 各 20 个实例；`publication` 生成
500/1000/500/500 个实例。OOD 是启发式必须在 240 分钟内完成的严格
分布外集合；stress 复用相同扰动因素并保留最多 20% 的困难截断尾部。
可用 `--build-split validation` 单独构建集合，
或用 `--count` 临时覆盖规模。重建已发布划分时增加 `--overwrite`；
未发布完成的构建默认断点续建，`--no-resume` 可放弃临时进度。生成过程
只接受满足对应压力硬条件的实例；达到最大尝试次数后会报告拒绝原因，
不会静默降级。

训练默认按课程在线生成实例，每个 episode 使用
`train.seed_start + episode` 的新实例，不向 `data/instances/` 写训练集；
`--fixed-instance` 仅允许与 `--smoke` 一起调试。实例种子固定使用训练区间
`[1000000, 2000000)`，与 PPO 算法种子 `[11, 23, 37, 53, 71]`
相互独立。使用 `--algorithm-seed` 选择算法种子，因此五次重复实验共享
同一实例序列，只改变网络初始化和策略采样随机性。

正式训练默认使用 20 个 Windows `spawn` worker 并行生成和推进完整 episode，
主进程将变长图按节点类型拼接、偏移关系边索引，只对最终动作 logits 做
padding 后批量推理。策略在一批
episode 完成前保持冻结，随后合并计算 GAE 并执行一次 PPO 更新；因此
2000 个 episode 对应 100 次 on-policy 更新（1000 个 episode 对应 50 次）。
可用 `--parallel-envs 1`
回退串行训练，或用 `--parallel-envs N` 覆盖并行度。正式训练要求
`validation_interval_episodes` 能被并行度整除。

默认在线课程使用线性 anchors，而不是突变的阶段采样：

- `0.00–0.20`：easy 30%、balanced 50%，五类瓶颈各 4%；
- `0.20–0.55`：线性过渡至 easy 10%、balanced 40%，五类瓶颈各 10%；
- `0.55–0.80`：线性过渡至固定验证分布；
- `0.80–1.00`：easy 5%、balanced 35%、machine/reconfiguration 各
  15%，worker/fatigue/high-arrival 各 10%。

旧的 `until_fraction` 阶梯课程仍可读取。给定 episode 的实例种子和课程
进度后，压力类型选择保持确定性。

### 双资源匹配准入

默认 `environment.worker_resource_control.mode` 为
`matching_admission_v1`。环境在同一状态内复用一个资源可行性快照：它汇总
所有 `WAIT_DIS/WAIT_INS` 任务、空闲且资格/阶段末疲劳均安全的工人边、最大
匹配、matching deficit 和每个任务的备选工人数。

- 模块已匹配的直接加工保持原行为，不要求工人担保。
- 需要重构的 production 候选会把新拆卸任务加入当前任务集；只有完整安全
  匹配仍存在，且乐观拆卸、安装和加工能在 horizon 前完成时才允许锁机。
- worker pair 被选走后，剩余任务也必须仍有完整匹配；存在这种安全 pair 时
  `ADVANCE` 被屏蔽，避免策略故意空等。
- 尚无 pending task、但候选仅因疲劳暂时不可行时，下一事件包含候选工人的
  最早安全恢复时刻，不会误判 deadlock。

`legacy_postcheck` 仅供 E0 回归。能力边保留 machine-only EST，并追加
resource-ready time、乐观 finish、拆/装安全工人比例、commit 后 matching
deficit 和 horizon slack；全局向量追加安全空闲工人比例、当前匹配缺口、
最小工人备选比例和最小候选 slack。除 slack 外的时间特征除以 horizon 并
裁剪到 `[0, 2]`，slack 裁剪到 `[-1, 1]`。

### 两阶段完成约束奖励

默认 `reward.mode` 为 `hierarchical_constrained_v1`。令

```text
Q_reward = 0.5 * F / (1200 + F)
         + 0.3 * C / (1000 + C)
         + 0.2 * V / (50 + V)
```

其中 `F` 是包含既有截断罚值的 `flow_time_objective`，`C` 是重构成本，
`V` 是工人负荷方差，因此 `0 <= Q_reward < 1`。默认训练尺度与正式
`Q_paper` 一致，但两者在记录中保持独立：训练消融可以产生不同的
`reward_quality_score`，不能改变正式 `quality_score`。再令 `c` 为订单完成比例、
`u = 1 - c`、`T` 表示 horizon、deadlock 或 decision-limit 截断，则
可行性阶段的整轨迹代理回报为

```text
R_feasibility = c + complete - 0.5 * T - 1.0 * u
R_quality     = R_feasibility - 0.5 * Q_reward
```

终局截断和未完成罚项只施加一次；串行 rollout、并行 rollout、验证中的
`feasibility_proxy_return` 使用相同公式。完整轨迹仍返回 `2.0`；12–18 单
规模下，只差一个订单的截断可行性回报最高约为 `0.33–0.39`。实例级
`unfinished_order_penalty = 240` 分钟仍只用于流经时间目标，数据 schema
保持不变。固定 validation 连续 3 次达到 100% greedy 完成率后才进入质量
阶段。

PPO 使用额外的训练专用 potential shaping：

```text
r_training = r_base + 0.25 * (Phi(next) - Phi(current))
Phi = 0.50 * operation_progress
    + 0.25 * minimum_safe_worker_alternative_ratio
    + 0.25 * minimum_nonnegative_horizon_slack
```

无 pending task 时资源余量为 1，terminal 的 `Phi` 为 0。因此同一实例所有
终局轨迹只相差相同常数，不改变正式终局目标排序。`reward_base` 继续与
`proxy_return_from_metrics` 校验恒等式；`reward_shaping` 和
`reward_training` 只描述 PPO 实际收到的信用信号。legacy reward 不叠加
shaping。

训练在可行性阶段每隔 `validation_interval_episodes` 个 episode 以及最后
一个 episode 读取固定 validation manifest。质量阶段每次 PPO 更新后都
立即验证。默认 `aligned_quality` 只晋升 greedy 完成率 100% 且平均
`quality_score` 严格下降的候选；未晋升候选继续记录，但不会因一次波动立即
覆盖正式 checkpoint。初始学习率为 `1e-4`；greedy 验证连续 15 次未刷新
历史最优时乘 `0.5`，最低为 `2.5e-5`。阶段切换重置停滞计数，checkpoint
回滚后重新应用当前学习率，不会被 checkpoint 中的旧学习率覆盖。

v7 C0/E1 使用 `balanced_guarded_v7`：greedy 候选必须保持 100% 完成率、
`quality_score` 至少改善 `1e-4`，且负荷方差相对 accepted anchor 的退化不
超过 5%。通过 greedy 条件后立即运行 100011/100012/100013 三个 sampled
seed；相对 anchor 的最差重复完成率下降不得超过 2%，疲劳 CVaR90 退化不得
超过 5%，且必须通过疲劳安全线，候选才会成为新的 accepted checkpoint。
阶段切换 anchor 同样要经过该 sampled 安全检查。

可行性阶段按 greedy 的“完成率优先、正式质量分数次之”选择键另存
`best_feasibility_checkpoint.pt`，包含网络、optimizer、学习率、验证指标和
来源 episode。稳定性控制器只有在连续 2 次 greedy 完成率均不高于 90% 时，
才将网络和 optimizer 回滚到最近一次 100% 完成的 `safe_checkpoint.pt`；
单次退化和 5 个百分点波动均不回滚。回滚后 3 次 greedy 验证处于 cooldown，
并清空退化和 plateau 计数。该临时模型不替代正式的 accepted checkpoint。

默认 `aligned_quality` 每第 5 次 greedy 验证以及训练结束时，在固定验证集
使用三个固定且独立的 `torch.Generator` 做 3 次 sampled 解码，只诊断“采样
训练成功但 greedy 推断失败”的差异，不触发普通稳定性回滚或学习率调整。
`balanced_guarded_v7` 将这一周期诊断替换为候选触发的官方三 seed sampled
guard，并在正式 checkpoint 落盘后再次复评；该 guard 明确参与阶段切换和
候选晋升。独立 `eval.py` 仍默认 greedy 串行评估，也可显式传入
`--decode-mode sampled --sampling-seed <seed>`。

`phase1_checkpoint.pt` 保存进入质量阶段时的模型；`safe_checkpoint.pt` 保存
最近一次 100% greedy 完成的状态；`accepted_checkpoint.pt` 保存当前通过
晋升规则的 shadow best；`last_checkpoint.pt` 保存训练结束时最后一个在线
候选。正式训练完成时，`checkpoint.pt` 和 `best_checkpoint.pt` 都复制自
`accepted_checkpoint.pt`，三者 SHA-256 必须一致，并从磁盘重新执行一次
greedy 与三 seed sampled 验证。若训练预算内未满足阶段切换门槛，
`summary.json` 标记
`formal_training_status = feasibility_not_reached`，该运行不得作为正式模型。
此时仅保存 `last_candidate_checkpoint.pt`，正式的 `checkpoint.pt` 和
`best_checkpoint.pt` 均不生成。缺少 `reward.mode` 的历史配置仍按原三项
归一化加权和解释。

消融筛选使用同一入口，命令会固定 seed 11、600 episode、10 个并行 worker
和每 10 episode greedy 验证；E0 不允许重训：

```powershell
.\.venv\Scripts\python.exe train.py --config configs\default.json --ablation E1 --run-name matching_e1
.\.venv\Scripts\python.exe train.py --config configs\default.json --ablation E2 --run-name shaping_e2
.\.venv\Scripts\python.exe train.py --config configs\default.json --ablation E3 --run-name controller_e3
```

这里的历史筛选 E1 仅启用 matching admission/schema v2，E2 再启用
shaping，E3 再启用新回滚/LR 控制器；这组编号与 v7 的 E1 有界残差策略头
不是同一实验命名空间。`summary.json.ablation_gate` 会先从全部训练记录筛出
`reconfiguration_bottleneck`，再核对最后最多 200 个重构实例的完成率，同时
报告请求窗口、全部可用样本数和实际样本数；另核对已知失败实例、最后 10 次
验证、回滚率、LR floor、约束违反和基础奖励恒等式。历史配置项
`training_window_episodes` 仍兼容，但同样按重构实例数解释。gate 只报告是否
具备正式多 seed 训练资格，不会自动启动 2000-episode 训练。

同一入口还保留 `R11/S11/L11/Q11/Q12/Q13` 协议比较变体；它们同样固定
seed 11、600 episode 和 10-episode greedy 验证间隔。具体覆盖项由
`train.py::_apply_ablation_variant` 集中定义，E0 始终只复用历史基线。

```powershell
.\.venv\Scripts\python.exe train.py --config configs\default.json --parallel-envs 20 --run-name ppo_2000_parallel
.\.venv\Scripts\python.exe benchmark_parallel.py --episodes 20 --steps 64 --workers 20
```

固定数据集评估：

```powershell
.\.venv\Scripts\python.exe eval.py --config configs\default.json --dataset validation --policy heuristic
.\.venv\Scripts\python.exe eval.py --config configs\default.json --dataset test --policy ppo --checkpoint result\runs\<run>\checkpoint.pt
.\.venv\Scripts\python.exe eval.py --config configs\default.json --dataset ood --policy ppo --checkpoint result\runs\<run>\checkpoint.pt
.\.venv\Scripts\python.exe eval.py --config configs\default.json --dataset validation --policy ppo --checkpoint result\runs\<run>\checkpoint.pt --decode-mode sampled --sampling-seed 100011
```

评估配置必须与 checkpoint 的 `network_spec` 匹配；v7 checkpoint 应改用对应
的 `configs\v7\*.json`，不能用 `configs\default.json` 强行加载。

## Visdom 实时科研监控

Visdom 默认关闭，可通过配置中的 `logging.visdom.enabled` 或训练参数
`--visdom` 开启。先在一个 PowerShell 窗口启动仅本机可访问的服务：

```powershell
.\.venv\Scripts\python.exe -m visdom.server -port 8097 -bind_local
```

Visdom 0.2.4 首次启动会下载前端 JavaScript/CSS 资源，需保证该次启动能够访问
网络；资源缓存后本地启动不再重复下载。

再启动训练：

```powershell
.\.venv\Scripts\python.exe train.py --config configs\default.json --smoke --visdom --run-name visdom_smoke
```

浏览器访问 `http://localhost:8097`。每次运行使用独立环境
`fatigue_assembly_<run>_seed<seed>`，固定窗口覆盖训练完成率与目标、
奖励分量、greedy/sampled 完成率、截断数、未完成订单、终局代理回报及
启发式 gap、疲劳与重构压力、PPO 的 KL/clip/梯度/explained variance、
吞吐计时、压力类型分解，以及固定验证实例的资源甘特图和逐工人疲劳轨迹。
不同环境使用相同窗口标题，可在 Visdom 中进行多种子对比。

若服务没有启动，训练默认不会失败，而是将所有 Visdom 事件写入运行目录的
`visdom_events.log`。服务启动后可回放：

```powershell
.\.venv\Scripts\python.exe -m result.visdom_replay --run-dir result\runs\<run>
```

代表实例诊断默认每 5 次验证，以及发生阶段切换、候选回滚或产生新最优模型
时更新；原始 schedule、reconfiguration 和 fatigue trace 同时保存在
`diagnostics/validation_<episode>.json`。Visdom 是只读观察层，不通过界面
修改超参数，也不替代 CSV/JSON 正式实验记录。

评估只读取 validation/test/OOD/stress 的 manifest 与已校验实例，不在评估时
重新生成。目录包含生效配置、聚合统计 `metrics.json`、逐实例
`instance_metrics.csv`、`schedule.csv` 和 `reconfigurations.csv`。
聚合统计使用样本标准差，并分别报告完成实例的 makespan/真实流经时间、
全部实例的惩罚后目标/成本/负荷方差、截断数、推理与求解计时，以及相对
启发式 gap（负值表示优于启发式）。`metrics.json` 和逐实例行还记录评估
schema、正式质量指标及哈希、数据 manifest 哈希、解码模式、sampling seed、
动作轨迹哈希和完整 provenance。

训练目录始终包含 `train_log.csv`、`update_log.csv`、`validation_log.csv`、
`summary.json` 和 `last_checkpoint.pt`。验证出现 100% 完成状态后生成
`safe_checkpoint.pt`；可行性历史最优写入 `best_feasibility_checkpoint.pt`；
达到阶段门槛后生成 `phase1_checkpoint.pt` 和 `accepted_checkpoint.pt`。
满足正式资格时还生成字节一致的 `checkpoint.pt` 和 `best_checkpoint.pt`；
未达到门槛时改为生成 `last_candidate_checkpoint.pt`。`update_log.csv` 记录
每批 transition 数、
采样/推理/更新耗时、实际吞吐量、奖励阶段和候选接受/回滚状态。
此外还记录 approximate KL、clip fraction、梯度范数、梯度裁剪比例、
更新前 explained variance、return/advantage/value 统计、当前学习率，以及
v6/v7 策略头的 11 个命名有效权重和 4 个上下文门控；
`train_log.csv` 记录 `reward_base`、`reward_shaping`、`reward_training`、
`reward_truncation`、`reward_unfinished`，以及 matching deficit、资源准入
屏蔽率、最小工人备选数、matching-preserving 动作、候选恢复推进和机台等待
工人时间；
`validation_log.csv` 保留原 greedy 列并增加 `sampled_*`、
sampled-minus-greedy、平均未完成订单数和 feasibility proxy return。
`summary.json` 记录最佳可行验证及来源 episode、可行性回滚/cooldown、学习率
衰减、sampled 验证、后 500 episode 诊断均值、消融 gate、正式训练状态、
checkpoint 哈希和最终从磁盘复评结果。

## E2.4：偏好中立生产门控与安全方差偏好

E2.4 是独立于 E2、E2.1、E2.2 和 E2.3 的 Pareto 实验。E2.3 已修复 worker
bottleneck 的不可恢复 matching deficit；E2.4 进一步处理低 flow 偏好在扁平
`pair_plus_defer_v1` 分布中诱发的 production defer 链。E2.2 虽使用分层
commit/pair，但其 defer head 仍读取 preference embedding，因此不是偏好中立
门控。

E2.4 的 `hierarchical_state_only_gate_then_pair_v3` 先用 action-set 状态特征
输出 `P(commit)` 与 `P(defer)`；该 gate 不读取 preference、pair logits 或
直接偏好项。仅在 commit 后，条件 pair softmax 接收 production 偏好。PPO
继续使用相同的扁平动作 ID 和联合 log-prob。worker 保留 E2.3 的
`matching_admission_recovery_v2` 安全 mask，只在安全候选内加入直接的负载方差
偏好；worker 的直接 flow/cost 项恒为零。

### E2.4–E2.7 分层训练门禁

这些配置显式启用 `training.gate_policy.version =
"tiered_training_gates_v1"`（schema `5.0.0`）。训练期的硬门禁只包括数值
非有限值、非法动作、固定验证清单损坏、canonical identity 漂移，以及调度/疲劳
物理违规。调度或疲劳违规会先保存 `latest_rejected_candidate.pt`，恢复最近安全
checkpoint，并将学习率减半后继续；完成率、KL、梯度、pair loss、偏好响应和
Pareto 指标只记录或驱动 plateau 学习率控制，不再触发完成率回滚或提前停训。

每次固定审计分别记录 `physical_safety_pass`、`completion_pass` 和
`evaluation_integrity_pass`；`all_safe` 保留为历史兼容字段。通过物理安全和清单
完整性的状态写入 `last_safe_checkpoint.pt`，完整网格的最佳候选按完成率、
Hypervolume、update id 排序写入 `best_safe_candidate_checkpoint.pt`。

训练结束后才分别冻结并验收这两个候选，输出
`final_acceptance_best_safe.json`、`final_acceptance_last_safe.json` 和
`final_acceptance.json`。任一候选通过即可接受，优先选择 best-safe；此时才生成
兼容的 `accepted_checkpoint.pt`。E2.7 仅在 validation 前置条件通过后执行
validation/test/OOD/stress heldout；E2.4–E2.6 将 heldout 标为 `not_configured`。
旧配置缺少该版本字段时保持原有门禁语义。

远端运行 seed11：

```powershell
python train.py `
  --config configs\v7\e2_4_neutral_gate_safe_variance.json `
  --algorithm-seed 11 `
  --run-name v7_2000_e2_4_neutral_gate_seed11
```

E2.4 使用 result schema `5.0.0`。逐实例、validation、Pareto 和 summary 同时
持久化 matching recovery、production/worker preference、state-only gate 和
safety guard 字段；provenance 的 schema 版本来自生效配置，不再硬编码为 `4.1.0`。

### MO-ALNS 元启发式强基线

`agent/mo_alns/` 提供环境解码的多目标 ALNS：解编码包含工序优先级、机器与
拆卸/安装工人的完整偏好排序以及三类等待基因。任何候选均通过
`AssemblySchedulingEnv` 的合法动作掩码逐步重放，因此保持拆装分阶段、疲劳
恢复和 matching admission 的原始语义。搜索使用 8 个初始化规则、6 个破坏算子、
6 个修复算子、Pareto archive、增广 Tchebycheff 标量化与自校准模拟退火。

默认配置每实例–偏好点使用 300 次完整环境评价；全网格是 denominator-five 的
21 点加 canonical `(0.5, 0.3, 0.2)`，共 22 个端点。先用小预算做 smoke：

```powershell
.\.venv\Scripts\python.exe mo_alns.py `
  --config configs\baselines\mo_alns_smoke.json `
  --dataset test --smoke --instance-limit 1 --parallel-envs 1
```

正式单种子运行：

```powershell
.\.venv\Scripts\python.exe mo_alns.py `
  --config configs\baselines\mo_alns.json `
  --dataset test --algorithm-seed 11 --parallel-envs 20
```

`mo_alns_benchmark.py` 可执行 `configs/baselines/mo_alns_manifest.json`
中的五种子、三数据集协议。它输出 22 个端点、实例级 archive、完整排程日志、
算子统计和 provenance。用 `mo_alns_analysis.py` 将这些 `candidates.csv` 与既有
E1/E2 分析结果合并，得到三方法经验 Pareto、Hypervolume、贡献、canonical 质量和
成对检验。报告明确标注 MO-ALNS 是 solver-budget arm，不把其内部搜索评价次数伪装
成 E1/E2 的 equal-rollout budget。

```powershell
.\.venv\Scripts\python.exe mo_alns_analysis.py `
  --e1-e2-candidate-csv result\analysis\e1_e2_full\candidates.csv `
  --mo-alns-candidate-csv result\runs\mo_alns_formal\candidates.csv `
  --output-dir result\analysis\e1_e2_mo_alns_solver_budget_v1
```

### E1 三个单目标策略

`e1_single_flow.json`、`e1_single_cost.json` 和
`e1_single_variance.json` 继承同一个 E1 单目标公共配置。三者仅将
`reward.quality_weights` 分别设为 `(1,0,0)`、`(0,1,0)` 和 `(0,0,1)`。
公共配置使用 `temporal_matching_admission_recovery_v3`、
`deadline_progress_viability_shield_v2` 和
`single_objective_guarded_v1`；soft risk shaping 系数固定为零。v3 保留静态
快速路径，并在静态检查失败时使用有限节点的确定性时序 oracle；历史 E1/E2
配置仍保留原动作域。

正式 validation 固定为同一份、同一顺序的 200 个 publication 实例。日常 validation
只读取该 manifest 的前 50 个；改善候选触发完整 200-instance 审计。已有 500-instance
manifest 也可直接复用其前 200 个。首次在训练电脑生成后，三个策略共用该 manifest：

```powershell
.\.venv\Scripts\python.exe data\generate_orders.py `
  --config configs\v7\e1_single_flow.json `
  --build-split validation --profile publication --count 200 --overwrite
```

阶段一进入质量阶段仍要求连续 3 次 `completion_rate=1.0`。质量阶段采用最近 5 次
合格日常验证的原始目标中位数：`completion_rate>=0.95`、零 schedule violation 和物理/疲劳
安全。窗口中位数相对 candidate anchor 严格下降 `1e-9` 时触发一次完整 200-instance 审计；
连续两次严格低于 0.95 回滚并清空当前窗口。审计以 `failed_count = 200 - completed_count`
统一计入 incomplete/truncated 实例，最多允许 4 个失败（完成率至少 98%），且 schedule
violation 和物理/疲劳安全必须为零。只有审计通过且字典序 `(failed_count, window_median)`
改善时，才原子替换唯一的 `accepted_checkpoint.pt`。该 accepted 是 98% 实验候选，
`formal_eligible` 保持为 false；项目正式完成标准仍是 100%。
训练收尾会从磁盘隔离重载 accepted checkpoint，并另启短生命周期验证进程池，按
`training.validation_parallel_envs` 并行复核 200 个实例；`summary.json` 的
`final_checkpoint_evaluation.evaluation_config` 记录实际执行模式与并行度。

本机 smoke 命令如下。Smoke 只验证配置、rollout/PPO 更新、安全机制诊断字段和
reward identity，不用于判断收敛，也不进入后续 payoff matrix：

```powershell
.\.venv\Scripts\python.exe train.py --config configs\v7\e1_single_flow.json --smoke --algorithm-seed 11 --run-name e1_single_flow_smoke_seed11
.\.venv\Scripts\python.exe train.py --config configs\v7\e1_single_cost.json --smoke --algorithm-seed 11 --run-name e1_single_cost_smoke_seed11
.\.venv\Scripts\python.exe train.py --config configs\v7\e1_single_variance.json --smoke --algorithm-seed 11 --run-name e1_single_variance_smoke_seed11
```

另一台训练电脑运行 seed 11、2000 episodes 的正式实验。配置已固定
`training.episodes=2000`：

```powershell
.\.venv\Scripts\python.exe train.py --config configs\v7\e1_single_flow.json --algorithm-seed 11 --run-name e1_single_flow_seed11_2000
.\.venv\Scripts\python.exe train.py --config configs\v7\e1_single_cost.json --algorithm-seed 11 --run-name e1_single_cost_seed11_2000
.\.venv\Scripts\python.exe train.py --config configs\v7\e1_single_variance.json --algorithm-seed 11 --run-name e1_single_variance_seed11_2000
```

`train.py` 会自动同时显示并保存 stdout/stderr；无需在命令外添加
`Tee-Object`。正常完成或训练阶段异常退出后，日志位于本次 run 目录内的
`terminal.log`，并包含实际命令、UTC 起止时间与退出码。因此原训练命令可直接使用：

```powershell
$train = ".\.venv\Scripts\python.exe"
& $train train.py --config configs\v7\e1_single_flow.json --algorithm-seed 11 --parallel-envs 20 --run-name e1_single_flow_seed11_2000_diagnostic
```

训练完成后，将三个正式 run 目录传给收敛分析入口：

```powershell
.\.venv\Scripts\python.exe single_objective_analysis.py `
  --flow-run result\runs\e1_single_flow_seed11_2000 `
  --cost-run result\runs\e1_single_cost_seed11_2000 `
  --variance-run result\runs\e1_single_variance_seed11_2000 `
  --output-dir result\analysis\e1_single_objective_seed11
```

每个策略输出一张五面板 PDF、300-dpi PNG 和对应的原始 50-instance validation 数据 CSV，
并标记 feasibility→quality 切换点、200-instance 审计触发点及 accepted checkpoint episode。
completion 面板同时绘制 1.0 和 0.95 参考线。汇总文件
`convergence_diagnostics.json` 和 `convergence_report.md` 报告 95% 合格点数、审计数量、
accepted 数量、98% 审计结果以及另外两个目标的原始变化。审计逐实例失败保存在
`single_objective_audit_failures.csv`；审计摘要保存在 `single_objective_audit_log.csv`。
正式 accepted checkpoint 之后再在同一 calibration set 上计算 payoff matrix。

`configs/baselines/mo_alns_temporal_v3.json` 提供与三个单目标策略相同的 v3
动作域，用于 MO-ALNS 对照；原 `mo_alns.json` 保持历史 v2/静态动作域语义。
