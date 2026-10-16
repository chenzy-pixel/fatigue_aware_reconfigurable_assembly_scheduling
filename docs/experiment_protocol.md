# 主线实验协议

## 模型与冻结条件

默认 [配置](../configs/default.json) 为 schema12、log2、HGNN shared-head、excess Flow、OWA(q=.5)，PPO `gamma=1`、`gae_lambda=.95`、dropout=0。Universal 训练 2000 episodes，单目标训练 1000；每次 PPO 更新采集 20 episodes。

尺度取自 [冻结清单](../configs/manifests/workload_excess_q50_20261011.json)：Flow=368.3143、Cost=353.27、Workload=21.03145448655653。实验内冻结尺度；不同目标模式、OWA q、尺度和观测编码属于不同身份，评估模型时使用其保存配置。

全局累计目标输入为 `log2(1+J/s)`。质量坐标为 `q_i=J_i/(s_i+J_i)`，使用配置中的 augmented Tchebycheff 标量化，rho=.05。奖励为 `ΔP-ΔQ-2I_failure`，附加势函数项关闭。

## 数据、验证与选模

训练实例按固定算法种子在线生成；验证与测试使用提交的 V2 固定数据及对应哈希。改变训练种子不改变固定验证/测试协议。

Universal 每 100 episodes 验证：固定分层 50 实例 × 13 偏好 × 3 次采样，共 1950 请求、650 交付。偏好包括三端点、三边中点、等权中心和 `(0.6,0.3,0.1)` 的六个排列。

单目标每 40 episodes 验证，仅使用相应 one-hot 偏好。正式解码使用 sampled、temperature=1。同实例、同偏好的三次请求按成功优先、Q 更低优先、采样序号更小优先取优。

Universal checkpoint 按取优后的最差偏好完成率优先，再比较偏好平衡 Q；单目标按取优后的整体完成率优先。安全且完整的首次验证初始化 best，同分保留原 best。每次训练保存 last，最终测试重载 best；无有效 best 时明确报告。

## 测试与公平比较

Universal 最终测试使用 step=.1 的 66 点 simplex 网格，每实例、偏好三次采样。原始请求、被选交付、完整性、安全性和全部推理预算同时报告；详见 [聚合契约](default_best_of_three.md)。

比较固定实例、训练种子、目标模式、尺度、更新预算、选模、解码与求解预算。raw Flow、variance、不同 q、编码或 lambda 使用 [显式对照](ablation_protocol.md)。新默认协议与历史逐采样协议分别解释，不重写历史结果。

结果目录保存有效 config、provenance、checkpoint、训练/验证日志、完整原始评测行和被选行。预检、启动和完成是不同状态；smoke 用于检查集成，不说明训练质量。

## 理论与执行清单

[数学证明目录](../data/docs/README.md)区分目标/物理逐轨迹结论与[三次取优选模](../data/docs/SELECTION.md)。训练的期望回报、候选取优和checkpoint的完成率优先规则具有不同的排序依据。真实失败报表Q=1，训练质量差分仍使用其有界状态Q。

[完整实验清单](experiment_runlist.md)列明当前三个种子对应的24个核心训练、21个单因素训练、33个条件扩展训练，以及求解器、泛化、协议与统计实验。数量表示按当前11、23、37三个种子展开的矩阵，不表示已启动或完成；扩展实验按研究主张选择。
