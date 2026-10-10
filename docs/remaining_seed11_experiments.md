# Cost 完成后的七项 seed11 实验

本机只做配置和运行验证，正式训练在远端执行。等当前主线 Cost 结束后，同步当前
代码、新配置和尺度清单，再在远端项目根目录依次执行：

```powershell
python scripts/check_remaining_seed11_experiments.py
python scripts/run_remaining_seed11_experiments.py
```

第一条检查全部七项的数据、模式、尺度、网络身份和实际 CUDA 前向，输出预检报告。
第二条按下表顺序训练，所有算法种子为 **11**。

| 顺序 | 网络 | 训练目标 | Flow 口径 | episode |
|---:|---|---|---|---:|
| 1 | 主线 objective_experts | Variance | raw_v1 | 1000 |
| 2 | 主线 objective_experts | Universal 三目标偏好 | raw_v1 | 2000 |
| 3 | 主线 objective_experts | Flow | excess_proportional_lb_v1 | 1000 |
| 4 | shared_head | Flow | excess_proportional_lb_v1 | 1000 |
| 5 | shared_head | Cost | excess_proportional_lb_v1 | 1000 |
| 6 | shared_head | Variance | excess_proportional_lb_v1 | 1000 |
| 7 | shared_head | Universal 三目标偏好 | excess_proportional_lb_v1 | 2000 |

`shared_head` 对应代码中的 `network.actor_head_variant="shared_preference"`。
三种单目标分别固定偏好 `(1,0,0)`、`(0,1,0)`、`(0,0,1)`。
所有 shared_head 新实验都使用比例下界抵扣、Flow 尺度 **368.3143** 和 v4 奖励
身份；Cost/Variance 的目标定义与尺度 **353.27 / 2.2629** 保持原协议。
主线第三项 Variance 继承原 raw/v3 协议，补齐正在运行的主线单目标组。
紧随其后的主线 Universal 使用同一 raw/v3 协议和原 Flow 尺度 1089.15，训练 2000 episode。

配置位于 `configs/flow_excess/shared_head/`，分别继承原单目标/Universal 的
训练、数据、网络大小与验证预算。默认 hidden_dim=128、2 层、CUDA、训练/验证
各 20 worker、每次 PPO 更新 20 个 episode；修改 worker 并行度仍保留该更新预算。
单目标每 40 episode 验证 50 实例 × 1 偏好 × 3 repeat；两项 Universal
每 100 episode 验证 50 实例 × 13 偏好 × 3 repeat，最终为 20 实例 × 66 偏好 × 3 repeat。

每项使用独立时间戳目录和新初始化；保存有效配置、checkpoint、终端输出、验证及
最终评估。启动目录的 `launch.json` 记录七项顺序与状态。任何一项返回失败时
顺序入口停止，后面的项目保持 pending，便于定位问题。

若需要分开启动，对应命令如下：

```powershell
python scripts/run_main_single_variance.py
python scripts/run_main_universal.py
python scripts/run_flow_excess_single_flow.py
python scripts/run_shared_head_flow_excess.py
```

第四条按 Flow → Cost → Variance → Universal 顺序执行 shared_head 四项。
可通过 `--experiment flow/cost/variance/universal` 单独选择其中一项。
通用参数为 `--seed`、`--device`、`--parallel-envs`、`--validation-parallel-envs`、
`--preflight-only`。Cost 与 Variance 的新 shared_head 实验也会使用 excess Flow
观测，因此这组结果反映共享头与新观测/归一化配置的整体效果。

本机七项完整网络的 CUDA 前向预检已通过；四项 shared_head 的真实小步 rollout、
GAE/PPO 更新、奖励重建以及 checkpoint 保存/重载已验证。配置、顺序启动、
只读预检和原 checkpoint/观测兼容检查的日志位于
`result/audits/flow_excess_optimization_20261008/shared_head_launch.xml` 与
`shared_head_integration.xml`。本机没有启动正式训练。
