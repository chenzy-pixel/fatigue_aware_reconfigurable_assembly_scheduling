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

做 MLP 消融时复制实验配置并把 `network.encoder_type` 改为
`typed_mlp`；`message_passing_layers` 和 `dropout` 对该基线不生效。
缺少 `encoder_type` 的旧配置按 `typed_mlp` 解释。新 checkpoint 会保存
`network_spec`；旧 MLP checkpoint 可继续加载，但评估配置的编码器类型和
隐藏维度必须与 checkpoint 一致，避免误用实验架构。

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
`V` 是工人负荷方差，因此 `0 <= Q < 1`。可行性阶段的整轨迹代理回报为
`completed_orders / total_orders + complete`；固定 validation 连续 3 次
达到 100% 完成率后进入质量阶段，代理回报变为
`completed_orders / total_orders + complete - 0.5 * Q`。完整轨迹回报
恒大于 1.5，任意未完成轨迹回报恒小于 1。

训练在可行性阶段每隔 `validation_interval_episodes` 个 episode 以及最后
一个 episode 读取固定 validation manifest。质量阶段每次 PPO 更新后都
立即验证；完成率低于 100% 的候选会同时回滚网络和 optimizer。独立
`eval.py` 仍保持串行，确保单实例计时可比。

`phase1_checkpoint.pt` 保存进入质量阶段时的模型，`checkpoint.pt` 保存
训练结束时最后一个已接受状态，`best_checkpoint.pt` 按完成率、流经时间
目标、重构成本和工人负荷方差的词典序保存最优验证模型。若训练预算内未
满足阶段切换门槛，`summary.json` 标记
`formal_training_status = feasibility_not_reached`，该运行不得作为正式模型。
此时仅保存 `last_candidate_checkpoint.pt`，正式的 `checkpoint.pt` 和
`best_checkpoint.pt` 均不生成。缺少 `reward.mode` 的历史配置仍按原三项
归一化加权和解释。

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
奖励分量、验证集及启发式 gap、疲劳与重构压力、PPO 的 KL/clip/梯度/
explained variance、吞吐计时、压力类型分解，以及固定验证实例的资源甘特图
和逐工人疲劳轨迹。不同环境使用相同窗口标题，可在 Visdom 中进行多种子对比。

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
`checkpoint.pt`、验证最优模型 `best_checkpoint.pt`，以及达到阶段门槛时
生成的 `phase1_checkpoint.pt`。`update_log.csv` 记录每批 transition 数、
采样/推理/更新耗时、实际吞吐量、奖励阶段和候选接受/回滚状态。
此外还记录 approximate KL、clip fraction、梯度范数、梯度裁剪比例、
更新前 explained variance、return/advantage/value 统计和当前学习率；
`validation_log.csv` 包含四类启发式 gap 以及疲劳、重构和工人竞争统计。
