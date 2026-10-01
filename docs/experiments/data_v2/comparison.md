# 算例协议 v2 开发对照

[机器可读统计](comparison.json) · [运行验收记录](run_acceptance.json)

算法种子 11；每组 400 episode。每 200 episode 验证 20 个固定实例 × 13 偏好 × 1 次采样；结束诊断 7 个实例；best checkpoint 在 20 个测试实例 × 66 偏好 × 1 次采样上评测。

开发数据生成及首次审计耗时：339.2 秒；历史文件校验：565 个。

| 课程 | 总耗时（秒） | 训练实例生成/载入（累计秒） | 测试轨迹完成率 | 失败原因 |
|---|---:|---:|---:|---|
| original_course | 9975.5 | 21.4 | 98.79% | {"horizon": 16} |
| progressive_course | 11609.8 | 20.2 | 98.26% | {"horizon": 23} |

分场景测试轨迹统计；原始目标均值只统计成功轨迹。

| 课程 | 场景 | 轨迹数 | 完成率 | Flow | Cost | Variance |
|---|---|---:|---:|---:|---:|---:|
| original_course | balanced | 462 | 100.00% | 769.551 | 478.393 | 13.544 |
| original_course | easy | 66 | 100.00% | 501.868 | 291.790 | 8.075 |
| original_course | fatigue_bottleneck | 132 | 100.00% | 1702.781 | 1060.282 | 14.579 |
| original_course | high_arrival_pressure | 132 | 100.00% | 1545.403 | 855.463 | 19.983 |
| original_course | machine_bottleneck | 198 | 100.00% | 1459.578 | 852.734 | 24.651 |
| original_course | reconfiguration_bottleneck | 198 | 91.92% | 1280.414 | 1167.271 | 24.987 |
| original_course | worker_bottleneck | 132 | 100.00% | 1662.045 | 922.985 | 15.969 |
| progressive_course | balanced | 462 | 99.57% | 769.398 | 539.467 | 27.049 |
| progressive_course | easy | 66 | 100.00% | 514.317 | 318.424 | 13.991 |
| progressive_course | fatigue_bottleneck | 132 | 98.48% | 1732.415 | 1117.553 | 22.501 |
| progressive_course | high_arrival_pressure | 132 | 100.00% | 1565.039 | 914.363 | 36.763 |
| progressive_course | machine_bottleneck | 198 | 99.49% | 1463.047 | 923.867 | 41.964 |
| progressive_course | reconfiguration_bottleneck | 198 | 90.91% | 1306.018 | 1303.902 | 38.139 |
| progressive_course | worker_bottleneck | 132 | 100.00% | 1672.205 | 961.085 | 34.571 |

本轮用于验收协议、数值稳定性和可追溯性；性能结论需完整多种子实验。完整验证/诊断统计、失败进度、配置及轨迹位于 comparison.json 和各 run 目录。

耗时分解（秒）。训练采样与 PPO 更新为墙钟耗时；实例生成/载入另按各 episode 累计统计。

| 课程 | 训练采样与更新 | 主验证 | 结束诊断 | 最终测试 |
|---|---:|---:|---:|---:|
| original_course | 2233.8 | 2317.1 | 404.5 | 4959.8 |
| progressive_course | 1850.6 | 2254.0 | 518.8 | 6923.8 |
