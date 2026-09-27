# 工人 Flow 候选相对时间实验

2026-09-27：增加一个独立消融配置 `configs/e1/single_flow_relative_time.json`。
基线为 `configs/e1/single_flow.json`，固定偏好 (1,0,0)，seed 11，1000 轮，20 个并行环境。

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
python -u train.py --config configs/e1/single_flow_relative_time.json --episodes 1000 --algorithm-seed 11 --parallel-envs 20 --run-name e1_flow_relative_time_seed11_ep1000
```

评估：同一固定验证集和采样种子下比较完成率，成功集合交集上的配对 Flow，
以及每实例失败情况；同时查看 worker/pair/WAIT 的选择概率和等待次数、等待时间。
采用原有 completion-first checkpoint 规则，比较 best 与最终轮次，避免仅凭
成功子集均值判断收益。新训练结束前不报告性能改进。

## 实现验证

- `test_flow_relative_time.py`、`test_v8_policy.py`、`test_ppo_batching.py`：29 项通过。
- 旧 V8 Flow best checkpoint 可按 `absolute_v1` 加载。
- `smoke_flow_relative_time_20260927`：2 轮、2 个训练环境，一次 PPO 更新，
  验证、best/last 保存、best 重新加载与最终 Sampled 评测全部完成，进程退出码 0。
  最终使用验证集 2 个实例各采样 3 次，完成 5/6，调度约束违规 0。
  此运行仅验证流程，不代表 1000 轮性能；正式实验尚未启动。
