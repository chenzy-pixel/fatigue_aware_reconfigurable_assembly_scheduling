# 工人 Flow 候选相对时间实验

2026-09-27：增加一个独立消融配置 `configs/e1/single_flow_relative_time.json`。
基线为 `configs/e1/single_flow.json`，固定偏好 (1,0,0)，seed 11，1000 轮。
已完成的两次 Flow 运行均使用 20 个并行环境；当前相对时间配置后来调整为
40 个训练 worker、40 个验证 worker、每次 PPO 更新 40 个 episode。

实验只替换工人 Flow 专家的直连时间输入。令 d 为阶段工期/horizon，
在当前合法 machine-worker pair 内计算 z=(d-mean(d))/max(std(d),0.001)，
使用总体标准差。z 直接进入已有负号 tanh ranker，得分为 -tanh(z)。
只有零个或一个合法 pair 时输入为 0；工期全相同也为 0。非法 pair 不参与统计。
尺度下限 0.001 是显式实验参数：horizon=240 分钟时相当于 0.24 分钟。
它抑制微小时间差，不宣称已经调优。绝对工期仍进入 action/context 编码器。

生产头、Cost/Variance 直连特征、WAIT 特征、奖励、GAE、熵系数、mask、
Sampled temperature=1、每 40 轮验证和每实例 3 次采样均沿用基线。
WAIT 不参与 pair 工期标准化。该变换可能改变 pair 相对 WAIT 的偏好；这是
实验需要观察的影响，不将候选分数拉开本身视为成功。

新字段写入 checkpoint network_spec；旧 V8 checkpoint 缺字段时解释为
absolute_v1。跨归一化模式或尺度下限的加载被拒绝，避免把旧权重误当新实验。
新实验从随机初始化训练。独立 run-name 保留原有 1000 轮基线产物。

PowerShell 启动（项目根目录）：

```powershell
python -u train.py --config configs/e1/single_flow_relative_time.json --episodes 1000 --algorithm-seed 11 --parallel-envs 20 --validation-parallel-envs 20 --episodes-per-update 20 --run-name e1_flow_relative_time_seed11_ep1000
```

这条命令记录历史并行预算；已完成运行的完整生效配置保存在其运行目录的
`config.json`。当前同名配置默认使用 40/40/40，重新运行请使用新的 run-name。

评估：同一固定验证集和采样种子下比较完成率，成功集合交集上的配对 Flow，
以及每实例失败情况；同时查看 worker/pair/WAIT 的选择概率和等待次数、等待时间。
采用原有 completion-first checkpoint 规则，比较 best 与最终轮次，避免仅凭
成功子集均值判断收益。

## 正式训练结果与尺度记录（2026-09-28）

`e1_flow_relative_time_seed11_ep1000` 已完成 1000 轮，best 位于第 960 轮。
固定验证集 50 个实例各采样 3 次，成功 149/150，成功轨迹 Flow 均值为
1150.7771812080537；best 的最终测试成功数为 58/60。
原绝对时间基线 best 为第 520 轮，验证成功数同为 149/150，Flow 均值为
1160.5570469798658。此处为单个训练种子的结果比较。

用户已确认主实验固定 Flow 尺度为 **1152.2093959731544**，取本实验第 840、880、920、
960、1000 轮验证中成功轨迹 Flow 均值的中位数。
主实验的另外两个尺度为 Cost=386.674652792805、工人负荷方差=4.937746913580247，
同样取各自单目标实验最后五次验证成功轨迹目标均值的中位数。Universal 训练验证
采用 13 个固定偏好、每 100 轮一次，最终评估采用 66 点。详细协议见
[主实验协议](experiment_protocol.md)。Universal 主模型也采用
`candidate_zscore_v1` 和 `0.001` 标准差下限，但并行环境数由其自身配置决定。

## 实现验证

- `test_flow_relative_time.py`、`test_v8_policy.py`、`test_ppo_batching.py`：29 项通过。
- 旧 V8 Flow best checkpoint 可按 `absolute_v1` 加载。
- `smoke_flow_relative_time_20260927`：2 轮、2 个训练环境，一次 PPO 更新，
  验证、best/last 保存、best 重新加载与最终 Sampled 评测全部完成，进程退出码 0。
  最终使用验证集 2 个实例各采样 3 次，完成 5/6，调度约束违规 0。
  此运行仅验证流程；1000 轮正式实验结果见上节。
