# 疲劳感知可重构装配调度最小系统

本项目是一个可直接运行的 PyTorch + PPO 最小系统，覆盖固定算例加载、离散事件仿真、疲劳约束、两阶段动作屏蔽、启发式/随机基线、轻量 PPO 训练与结果持久化。

`数据/fixed_instance.yaml` 是数值唯一事实源；`数据/参数定义.md` 和 `数据/异质动态图参数定义.md` 定义环境语义；PDF 只作为论文方案背景。生成的 `data/instances/fixed_15x4_v1.pkl` 是可删除并重新生成的缓存。

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
- 评估采用 `(总流经时间, 重构成本, 工人负荷方差)` 严格词典序；PPO 使用下述带完成约束的两阶段代理目标，后三项仍是有界加权近似而不是精确词典序 PPO。
- 环境返回轻量 NumPy 异质图观测，包含工序、机器、工人三类节点和五类关系；能力边的 `EST` 表示计入已有机器承诺与必要乐观重构后的最早加工开始时刻。
- 默认策略网络是纯 PyTorch 两层异质关系 GNN；原类型感知 MLP 以 `typed_mlp` 编码器保留，供消融和旧 checkpoint 复评。

## 目录

```text
configs/                 单一实验配置 default.json
agent/
  ppo/                   类型编码器、双策略头、Critic、GAE、PPO
  baselines/             启发式与掩码随机策略
data/
  generate_orders.py     固定缓存、压力实例和固定数据集生成
  dataset.py             规范化记录、manifest 与哈希校验加载器
  manifests/             validation/test/ood/stress 固定集合清单
  instances/             固定缓存及按集合划分的规范化 JSON 实例
environment/             实体、状态、动作掩码与离散事件环境
test/                    数据、环境、端到端与 PPO 测试
result/
  runs/                  训练/评估运行目录
train.py                 PPO 训练入口
eval.py                  heuristic/random/PPO 评估入口
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

### 策略头 v6

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

做 MLP 消融时复制实验配置并把 `network.encoder_type` 改为
`typed_mlp`；`message_passing_layers` 和 `dropout` 对该基线不生效。
缺少 `encoder_type` 的旧配置按 `typed_mlp` 解释。新 checkpoint 的
`network_spec` 同时保存编码器结构、节点/边特征维度和
`observation_schema_version = 3`。加载器会从旧 state dict 推断 schema v1
维度；若与当前观测不一致，会报告明确的 observation schema incompatibility，
不会部分加载。旧最佳 checkpoint 只复用既有 E0 指标作基线，新训练从头初始化。

## 运行

```powershell
.\.venv\Scripts\python.exe data\generate_orders.py --config configs\default.json
.\.venv\Scripts\python.exe eval.py --config configs\default.json --dataset validation --policy heuristic
.\.venv\Scripts\python.exe train.py --config configs\default.json --smoke
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest --runslow -m slow -q
```

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

正式训练默认使用 10 个 Windows `spawn` worker 并行生成和推进完整 episode，
主进程将变长图按节点类型拼接、偏移关系边索引，只对最终动作 logits 做
padding 后批量推理。策略在一批
episode 完成前保持冻结，随后合并计算 GAE 并执行一次 PPO 更新；因此
1000 个 episode 对应 100 次 on-policy 更新。可用 `--parallel-envs 1`
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
Q = (
    1.0  * F / (3600 + F)
  + 0.1  * C / (1000 + C)
  + 0.01 * V / (100 + V)
) / 1.11
```

其中 `F` 是包含既有截断罚值的 `flow_time_objective`，`C` 是重构成本，
`V` 是工人负荷方差，因此 `0 <= Q < 1`。再令 `c` 为订单完成比例、
`u = 1 - c`、`T` 表示 horizon、deadlock 或 decision-limit 截断，则
可行性阶段的整轨迹代理回报为

```text
R_feasibility = c + complete - 0.5 * T - 1.0 * u
R_quality     = R_feasibility - 0.5 * Q
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
立即验证；完成率低于 100% 的候选会同时回滚网络和 optimizer。所有阶段
控制均由 greedy 验证决定。初始学习率为 `1e-4`；greedy 验证连续 15 次未
刷新历史最优时乘 `0.5`，最低为 `2.5e-5`。阶段切换重置停滞计数，checkpoint
回滚后重新应用当前学习率，不会被 checkpoint 中的旧学习率覆盖。

可行性阶段按相同的 greedy 词典序另存
`best_feasibility_checkpoint.pt`，包含网络、optimizer、学习率、验证指标和
来源 episode。只有连续 2 次 greedy 验证均相对历史最佳下降至少 10 个百分点
才回滚；单次退化和 5 个百分点波动均不回滚。回滚后 3 次 greedy 验证处于
cooldown，并清空退化和 plateau 计数。该临时模型不替代正式的
`best_checkpoint.pt`。

每第 5 次 greedy 验证以及训练结束时，固定验证集使用三个固定且独立的
`torch.Generator` 做 3 次 sampled 解码。sampled 只诊断“采样训练成功但
greedy 推断失败”的差异，不触发阶段切换、checkpoint、回滚或学习率调整。
独立 `eval.py` 仍默认 greedy 串行评估，也可显式传入
`--decode-mode sampled --sampling-seed <seed>`。

`phase1_checkpoint.pt` 保存进入质量阶段时的模型，`checkpoint.pt` 保存
训练结束时最后一个已接受状态，`best_checkpoint.pt` 按完成率、流经时间
目标、重构成本和工人负荷方差的词典序保存最优验证模型。若训练预算内未
满足阶段切换门槛，`summary.json` 标记
`formal_training_status = feasibility_not_reached`，该运行不得作为正式模型。
此时仅保存 `last_candidate_checkpoint.pt`，正式的 `checkpoint.pt` 和
`best_checkpoint.pt` 均不生成。缺少 `reward.mode` 的历史配置仍按原三项
归一化加权和解释。

消融筛选使用同一入口，命令会固定 seed 11、600 episode 和每 10 episode
greedy 验证；E0 不允许重训：

```powershell
.\.venv\Scripts\python.exe train.py --config configs\default.json --ablation E1 --run-name matching_e1
.\.venv\Scripts\python.exe train.py --config configs\default.json --ablation E2 --run-name shaping_e2
.\.venv\Scripts\python.exe train.py --config configs\default.json --ablation E3 --run-name controller_e3
```

E1 仅启用 matching admission/schema v2，E2 再启用 shaping，E3 再启用新
回滚/LR 控制器。`summary.json.ablation_gate` 会先从全部训练记录筛出
`reconfiguration_bottleneck`，再核对最后最多 200 个重构实例的完成率，同时
报告请求窗口、全部可用样本数和实际样本数；另核对已知失败实例、最后 10 次
验证、回滚率、LR floor、约束违反和基础奖励恒等式。历史配置项
`training_window_episodes` 仍兼容，但同样按重构实例数解释。gate 只报告是否
具备正式多 seed 训练资格，不会自动启动 2000-episode 训练。

```powershell
.\.venv\Scripts\python.exe train.py --config configs\default.json --parallel-envs 10 --run-name ppo_1000_parallel
.\.venv\Scripts\python.exe benchmark_parallel.py --episodes 10 --steps 64 --workers 10
```

固定数据集评估：

```powershell
.\.venv\Scripts\python.exe eval.py --config configs\default.json --dataset validation --policy heuristic
.\.venv\Scripts\python.exe eval.py --config configs\default.json --dataset test --policy ppo --checkpoint result\runs\<run>\best_checkpoint.pt
.\.venv\Scripts\python.exe eval.py --config configs\default.json --dataset ood --policy ppo --checkpoint result\runs\<run>\best_checkpoint.pt
```

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
启发式 gap（负值表示优于启发式）。训练目录另含 `train_log.csv`、
`update_log.csv`、`validation_log.csv`、`summary.json`、最后模型
`checkpoint.pt`、验证最优模型 `best_checkpoint.pt`、最佳可行模型
`best_feasibility_checkpoint.pt`，以及达到阶段门槛时生成的
`phase1_checkpoint.pt`。`update_log.csv` 记录每批 transition 数、
采样/推理/更新耗时、实际吞吐量、奖励阶段和候选接受/回滚状态。
此外还记录 approximate KL、clip fraction、梯度范数、梯度裁剪比例、
更新前 explained variance、return/advantage/value 统计、当前学习率，以及
v6 策略头的 11 个命名有效权重和 4 个上下文门控；
`train_log.csv` 记录 `reward_base`、`reward_shaping`、`reward_training`、
`reward_truncation`、`reward_unfinished`，以及 matching deficit、资源准入
屏蔽率、最小工人备选数、matching-preserving 动作、候选恢复推进和机台等待
工人时间；
`validation_log.csv` 保留原 greedy 列并增加 `sampled_*`、
sampled-minus-greedy、平均未完成订单数和 feasibility proxy return。
`summary.json` 记录最佳可行验证及来源 episode、可行性回滚/cooldown、学习率
衰减、sampled 验证、后 500 episode 诊断均值和消融 gate。
