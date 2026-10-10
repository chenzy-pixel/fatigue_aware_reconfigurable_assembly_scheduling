# Flow 比例下界抵扣的离线回放协议

2026-10-08。仅验证用户提出的状态闭式抵扣方案及奖励分配，不修改环境、
训练配置或 checkpoint 契约。沿用 `experiment_protocol.md` 的固定实例与
已执行物理调度，结果与此前诊断独立落盘。

- 输入：`ablation_eval_full_{flow,cost,variance}_seed11_20261007_232505_102441`
  的 `schedule.csv`、`reconfigurations.csv`、`instance_metrics.csv` 与测试实例。
  每臂 20 实例 × 3 采样，180 条轨迹，包括所有失败。
- 下界：在原生环境 reset 后取兼容机器 `estimate_processing_ticks` 的最小值；
  实际加工计划时长也通过同一函数核对。失败截断使用计划时长，保留
  `planned_end`，不得用已执行的 `end-start` 当比例分母。
- 主比较：完成时一次抵扣与按加工比例抵扣。另计算前 p_min ticks 全速抵扣
  作为补充。每个时点从状态闭式计算，原始 Flow 也由订单释放/完成重建。
- 网格：全日志事件时间的并集，以及原生 resolution 的每个 tick。时间推进
  后统一处理该时刻完成事件；失败罚项另作为同 tick 的终止跳变。
- 奖励：固定 Flow-only 偏好，`r=delta_progress+q_before-q_after-2*I_failed`。
  各臂为调度来源，不代表 Cost/Variance 模型原本的奖励。分别使用旧尺度
  1089.15 与前次仅由验证集标定的诊断尺度 368.31428571428575；两种扣法的
  比较始终保持相同尺度。报告质量奖励与总奖励方差，注明网格与样本数量。
- `schedule.csv` 无法恢复全部 WAIT 恢复时点、零时间动作或重构 lock 时点。
  本回放不宣称还原原 PPO 决策步、混合偏好奖励、GAE/优势估计或其方差。
- 核验：比例/前置抵扣在罚项前非负且单调，比例在事件处连续；成功终值
  相同；失败保留部分信用；进度、Flow 与报告匹配；所有方法奖励望远镜
  恒等式成立。保留失败结果与逐轨迹配对统计，不把重复当独立训练 seed。

入口：`python scripts/replay_flow_excess.py`。
输出：`result/analysis/flow_excess_replay_20261008/`。
