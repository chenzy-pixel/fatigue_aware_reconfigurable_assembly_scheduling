# 四类对照实现验证记录

2026-10-17。本轮完成本地实现、定向测试及小实例链路检查，用户选择在远端手动启动正式批次。尚无本套件的正式训练、充分搜索或独立确认成绩。

已实现的接口与源文件：`agent/comparators`包含种群算法、DCAN、规则DDQN、原子产物/身份、精确交互采集和学习训练/评价；`scripts/comparison_suite.py`及PowerShell包装器提供分阶段执行；`analysis/comparator_study.py`提供曲线、配对统计和报告。配置位于`configs/comparators`，远端操作见`comparator_remote_startup.md`。

## 真实检查产物

- `result/analysis/comparators_full_regression.xml`：97项通过，覆盖新算法、旧PPO保存加载/批处理、MO-ALNS预算及评价agent契约。
- `result/analysis/comparators_release_tests.xml`：81项通过、10项失败。10项均是已有配置的`flow_mode disagrees with normalization manifest`；测试名称和异常已与既有MO-ALNS审计逐项比较一致，见`result/analysis/comparators_preexisting_failures.json`。未改写这些配置，也未将失败隐藏成通过。
- 最后补测`result/analysis/comparators_last_checks.xml`：5项通过，包括DCAN原生工序重编号等变、三个方法checkpoint重载和DDQN真实更新/目标同步。测试集合有重叠，不将数字相加作总测试数。
- Python编译检查和`git diff --check`通过。
- `result/analysis/comparison_suite_ready_v2/preflight.json`：通过PowerShell正式包装器调用的最终预检，核实固定数据/子集、模型构建及合法前向；身份见同目录`identity.json`。

## 小实例smoke

产物：`result/analysis/comparison_suite_smoke_release/smoke.json`及其`smoke/<method>`子目录。

| 方法 | 实际交互/评价预算 | 机制结果 |
|---|---:|---|
| PPO | 精确20次env.step | 5个选模档位，10次梯度更新，best/last均保存 |
| DCAN | 精确20次env.step | 5个选模档位，10次梯度更新，best/last均保存 |
| Rule-DDQN | 精确20次env.step | 5个greedy验证档位；未达1024步预热，不进行训练更新 |
| NSGA-II-LS | 32次完整评价 | 完成7代、9次局部搜索评价；交付可行候选 |
| MOEA/D | 32次完整评价 | 完成6次扫描；交付可行候选；本小实例未发生替换 |

DDQN的梯度更新、replay容量和Double-Q目标通过单独小型单元测试验证；20步smoke不被表述为已经有效训练DDQN。MOEA/D没有替换说明该小实例没有提供改进证据，不能因此宣称其搜索有收益。所有本机smoke的目标值和时延均不用于方法胜负结论。

合成完整确认矩阵另用于验证3003条交付的覆盖检查、CSV、13点交付HV、付费档案HV、按实例bootstrap及SVG/PDF/PNG；该测试数据标记synthetic，不能作为真实确认结果。删去一条请求后报告器正确拒绝完整性不合格的矩阵。

## 实现边界

- DCAN是条件注意力结构的同环境适配，对工人服务和辅助节点做了扩展，并使用项目标量奖励/PPO；不称为原论文完整复现。数值测试覆盖条件注意力公式及合法动作/阶段/重编号行为。
- 原PPO训练/评价默认入口与历史checkpoint解释保持独立；suite通过新的分阶段训练器复用其优化与worker传输。完整GPU吞吐尚待远端实测。
- 请求级恢复只跳过完整且身份匹配的产物；中断训练从零重跑。身份包括源码、配置、数据、硬件，并锁定并行宽度。
- 正式预算350411次env.step，三训练种子11/23/37。远端训练、曲线、独立确认及正式manifest的实际产物是后续性能判断依据。
