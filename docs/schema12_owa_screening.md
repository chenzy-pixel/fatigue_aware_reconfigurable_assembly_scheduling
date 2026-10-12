# Schema12 与 OWA 最小筛选实验

首轮新增两次 Universal 训练，读取结果后再选择一次编码对照。首次预算为 4000 episodes，完整筛选最多 6000 episodes；四份配置中的两个 bounded 配置为备选，第三次训练只选择其中一份。

## 实验清单

| 顺序 | 配置（configs/observation/screening/） | 第三目标 | 全局编码 | 轮数 | 执行条件 |
|---|---|---|---|---|---|
| A | variance_log2.json | Variance | log2_v1 | 2000 | 首轮 |
| B | owa_log2.json | geometric OWA，q=0.5 | log2_v1 | 2000 | 首轮 |
| C | owa_bounded.json 或 variance_bounded.json | 与保留的 A/B 相同 | bounded_v1 | 2000 | 检查 A/B 后，二选一 |

所有新增实验固定 shared_head（代码值 shared_preference）、excess Flow、schema12、10 维全局观测和全局完工裕量、seed11、20 episodes/PPO update、同一训练实例流及 Universal 偏好采样。每组从头训练。每组 2000 episodes 用于观察后期完成率与 best/last 差距。

冻结尺度为 Flow=368.3143、Cost=353.27；Variance=2.2629，OWA(q=0.5)=21.03145448655653。A/B 的第三目标及对应冻结尺度、相关观测、奖励和模型身份随目标定义配套切换；这是目标定义的完整对照。A/B 的 PPO 超参数、数据、训练预算、偏好、编码及其他奖励参数相同。C 与它对应的 log2 配置只改变编码及实验名称（及自动派生的身份字段）。

## 启动

在仓库根目录使用项目虚拟环境。启动器按顺序执行，任一训练子进程失败即停止，并保存每项的最终配置、启动检查和运行状态。

先检查 A/B 的数据、CUDA 与网络前向：

```powershell
.\.venv\Scripts\python.exe scripts/run_schema12_screening.py --preflight-only
```

启动首轮 A、B 两组：

```powershell
.\.venv\Scripts\python.exe scripts/run_schema12_screening.py
```

单独启动或在 A 已完成后只运行 B：

```powershell
.\.venv\Scripts\python.exe scripts/run_schema12_screening.py --experiment owa_log2
```

根据验证集结果选择第三组：保留 OWA 时运行第一条；保留 Variance 时运行第二条。

```powershell
.\.venv\Scripts\python.exe scripts/run_schema12_screening.py --experiment owa_bounded
.\.venv\Scripts\python.exe scripts/run_schema12_screening.py --experiment variance_bounded
```

也可对任一单组追加 `--preflight-only`。默认沿用 CUDA 和 20 个并行环境；`--parallel-envs` 仅改变执行并发，已固定的 PPO 更新批次仍为 20。`--seed` 可在后续重复实验时显式指定。

## 对照解释

1. **B 对 A：OWA 是否值得保留。** 两者共享 schema12/log2/裕量，检验第三目标定义变化。OWA 的目标同时包含负担大小与均衡偏好，Variance 优化离散程度，不能通过两者各自的训练回报或归一化质量分数直接判胜。
2. **C 对对应的 A 或 B：单独验证输入编码。** 相同目标、尺度、裕量、模型结构和预算下，比较 log2 与 bounded。
3. **A 对已有 schema10 shared_head/excess/Variance Universal：整体更新参考。** 复用 `remaining_flow_experiments_seed11_20261008_225956_shared_head_universal` 的保存配置、best/last 和结果。历史 run 已训练 2000 episodes，当前同名配置已继承 OWA，复核历史结果必须读取 run 内的 config.json。此项跨 schema，只能作为整体版本变化参考，不能拆分裕量和非线性输入各自的贡献。

首轮能对照 OWA 与两种新编码。裕量单独消融、线性与非线性输入单独消融需要额外匹配的观测协议；现有 schema10/11 为九维线性，不能用它们直接声称分离出了其中一项的效果。

## 评估与继续条件

每 100 episodes 沿用 50 验证实例 × 13 偏好 × 3 次采样；维持既有 completion-first 选模。训练结束自动评估 best：20 测试实例 × 66 偏好 × 3 次采样（3960 条）。A/B 各一次现有标准评估即可提供对照。分配第三次训练预算的判断使用验证集；测试结果作为最终描述。

检查整个验证曲线，以及最后五个验证点的整体完成率、最差偏好完成率、纯 Cost 端点完成率、失败原因和进度，结合 best 与训练末期的差距判断是否稳定。验证汇总必须同时保留实例、偏好与采样种子，以便做配对比较。

对相同实例/偏好/采样种子、双方均成功的轨迹报告 raw Flow、Cost、Variance、OWA(q=0.5)、最大工人负担和总劳动量，并报告这部分样本覆盖率。新评估 CSV 已保存 worker_loads，可重算两种负担指标；失败轨迹的少加工、低成本不能计作质量改善。如计算质量/HV，所有候选必须在同一套目标、尺度和参考点下重算；Variance 与 OWA 的原始训练质量/HV 不直接混比。

完成率优先。如果 B 出现持续的整体或 Cost 端点完成率退步，先诊断其原因；OWA 平均值降低本身不足以扩大实验。如果完成率稳定，并在共同成功样本上获得值得保留的负担改善与 Flow/Cost 折中，补 OWA/bounded；若保留 Variance，补 Variance/bounded。三组仍无稳定改善时先分析，不扩大矩阵。

训练器保存 last，最后一次验证覆盖 last 的状态；首轮先利用该验证结果。仅对保留候选补 last 的完整测试，届时必须读取各自保存配置。best-of-three 可从同一批三次采样结果离线择优，并统一成功优先与目标评价口径；单次采样与三次择优分别报告预算。OWA/Variance 跨目标择优时，在共同评价口径下重算选择结果。

有初步收益后，优先为保留方案与其配对基线增加相同种子的重复实验，再按问题补专家头、raw Flow、q=0.25/0.75、单目标上界、OOD 或独立裕量消融。当前单种子筛选只用于决定后续预算。
