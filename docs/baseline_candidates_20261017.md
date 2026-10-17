# 本项目新增对照方法筛选（2026-10-17）

本文记录实施前的文献、作者代码及项目接口审查；该审查阶段没有移植候选算法或启动实验。用 Exa 沿进化搜索、双资源/疲劳、图强化学习、多目标偏好学习四条线完成16组检索，共92条结果、77个不同URL（包含同一论文的不同页面，不等于77篇论文）。最终结论只依据当前代码、论文原文/出版方页面及作者仓库；第三方代码售卖和仅有聚合摘要的结果未作为复现依据。

同日实施状态补充：用户确认后，NSGA-II-LS、MOEA/D、DCAN适配版和Rule-DDQN已接入独立套件，完成本地定向测试与小实例smoke；性能尚待远端实验。当前算法合同、远端手动命令和实际验证分别见[实施契约](comparator_implementation_spec.md)、[启动说明](comparator_remote_startup.md)、[验证记录](comparator_verification_20261017.md)。下文保留筛选时的判断与文献边界，不把候选价值判断当作强基线成绩。

建议优先增加两个对照：**项目适配的 NSGA-II＋局部搜索**，以及 **DCAN 同环境适配版**。前者提供与 MO-ALNS 不同的种群搜索机制；后者提供有作者代码的现代多目标学习模型。MOEA/D 可作为第二种进化搜索，PGRL 可作为另一种偏好条件学习路线。这里的“优先”是适配价值判断，算法在本项目中的强度均待实验验证。

**筛选依据：本项目并非标准 FJSP。**

- 生产动作是 `(operation, machine)+WAIT`；重构阶段动作是 `(machine, worker)+WAIT`。同模块可直接加工，换模块先锁定机器，再调度 DIS/INS 工人。普通加工不持续占用工人。依据：`environment/env.py:2820`、`:2875`、`:2947`；动作合法性由 `get_action_mask` 在当前状态重新计算，见 `environment/env.py:1734` 附近。
- 工人资质、疲劳积累/恢复、安全约束及 DIS/INS 分阶段资源竞争不能用固定 setup 常数代替。很多 DRCFJSP 论文要求工人持续参与加工，与此不同。
- 主线目标为 proportional excess Flow、重构费用、实际劳动 geometric OWA(q=.5)，并非 makespan、最大疲劳或机器负荷。当前定义见 `docs/ARCHITECTURE.md`、`environment/workload.py` 与 `configs/default.json:445`。
- 设 `z_k=J_k/(s_k+J_k)`，当前公共质量为 `Q_lambda=(max_k lambda_k*z_k+rho*sum_k lambda_k*z_k)/(1+rho)`，`rho=.05`，尺度固定。实现见 `environment/types.py:252`。这与部分论文的线性加权和不同。

**推荐候选及适配边界。**

| 候选 | 可核实依据 | 本项目用途 | 必须改造 | 优先级/工作量判断 |
|---|---|---|---|---|
| NSGA-II＋局部搜索，项目适配版 | Tan等疲劳DRCFJSP的ENSGA（2021）；另有DCAN作者公开的NSGA-II实现 | 与单轨迹ALNS形成种群搜索对照，覆盖三目标折中 | 拓扑顺序、机器、DIS工人、INS工人、WAIT/动态模式编码及算子；原生环境解码；本项目目标；局部搜索 | 首选启发式；中等 |
| MOEA/D；充分验证后可加VNS | DCAN作者公开实现；RL-MOMA/D文献提供双资源与负荷背景 | 检验分解、邻域协作搜索，与偏好条件PPO对应 | 同上；分解坐标与尺度冻结；预算按整个种群计费 | 第二种启发式；中等 |
| DCAN，ICLR 2026 | ICLR论文原文、作者仓库及条件注意力/PPO源码 | 单模型支持不同偏好，直接检验偏好进入注意力的价值 | 加入工人、模块、阶段、WAIT；阶段候选评分；改用当前奖励与环境；重训 | 首选学习对照；中高 |
| PGRL，Swarm and Evolutionary Computation 2024 | 作者仓库、超网络actor和ES训练入口已读取 | 偏好生成actor参数的路线，与直接条件评分形成区别 | 工人/阶段动作、当前目标和交付协议；若保留ES需单独设计训练计费 | 有价值的第二学习对照；较高 |
| DANIEL，TNNLS 2023在线发表/2024卷期 | 作者FJSP-DRL仓库与论文 | 成熟的工序/机器双注意力单目标调度骨架 | 增加工人/阶段/偏好；当前目标与奖励；重训 | 经典参考；DCAN已在其基础上扩展，二者可先选一个 |
| DQN/DDQN派工规则选择；如DTPDQN路线 | 作者DTPDQN仓库及Liu等IJPR 2022作者仓库 | 引入不同学习算法，比较学习选择规则与端到端候选评分 | 当前状态含偏好/疲劳/重构；生产与工人规则库；合法WAIT；当前奖励 | 补充对照；低至中等，强度受规则库限制 |

“工作量”是基于接口差异的工程判断，不是实测工期。NSGA-II与MOEA/D是算法框架，不是可以直接加载的预训练模型；学习方法的现有FJSP权重也不能直接用于本项目。

**NSGA-II＋局部搜索：最值得补的启发式。**

Tan等的 [A fatigue-conscious dual resource constrained flexible job shop scheduling problem by enhanced NSGA-II: An application from casting workshop](https://www.sciencedirect.com/science/article/abs/pii/S0360835221004617)，发表于 Computers & Industrial Engineering 160 (2021), 107557，明确同时优化 makespan 与最大工人疲劳，采用派工规则初始化、关键路径移动局部搜索。本轮核实了出版方公开内容，未取得全部算法细节，也未核实到作者公开代码。因此不能将一个自行添加局部搜索的版本称为该文ENSGA的忠实复现。

建议实现身份为 `NSGA-II-LS adapted to this environment`：保留非支配排序、拥挤距离、精英保留及交叉变异；针对本项目设计顺序插入/交换、机器排名变异、DIS/INS工人变异、合法等待和资源模式变异。局部搜索既要处理机器排序，也要处理工人竞争和模块复用。只用机器关键路径可能漏掉实际重构瓶颈。

当前 `MOALNSSolution` 已有顺序、机器排名、DIS/INS工人排名及等待位，`decode_solution` 能在原环境中执行，可复用为第一版公共候选接口：`agent/mo_alns/types.py:35`、`agent/mo_alns/solver.py:315`。这只是降低工程成本；需要检查该编码和解码器是否过度限制新基线的搜索空间，不能据此声称覆盖所有可行调度。精确初值的 `replay_actions` 是已保存轨迹，变异/交叉后必须解除旧轨迹绑定并重新解码。

DCAN的 [作者仓库进化算法目录](https://github.com/ai-for-decision-making-tue/Neural_Multi-Objective_CO_for_FJSP/tree/caf60d8da721382c79009539a9c8d9ccbc39e734/evolutionary_algorithms) 已核实包含 `run_genetic_algorithm.py`、`run_moead.py`、配置及调度环境。这比来源不明的复现代码更适合审查遗传/分解框架，但其FJSP解码器不能替代当前环境。

**MOEA/D与混合方法：分清“搜索中用RL”和“训练后直接调度”。**

[A reinforcement learning enhanced memetic algorithm for multi-objective flexible job shop scheduling toward Industry 5.0](https://doi.org/10.1080/00207543.2024.2357740)（2024在线，IJPR 63(1), 2025）提出RL-MOMA/D：双资源、工人学习效应、模糊加工时间，优化 makespan、机器总负荷、最大工人负荷；Q-learning用于选择/引导变邻域搜索。本轮核实了摘要，未核实到作者代码。它可以支持MOEA/D＋VNS的设计方向，但当前项目没有其模糊工时和学习效应，不应将这些问题假设搬入评测环境。

该类方法仍然要在新实例上反复评价候选。不能仅因带有Q-learning就把其交付延迟当作纯学习策略的一次推理延迟。

**DCAN：本轮最优先的学习模型候选。**

[Neural Multi-Objective Combinatorial Optimization for Flexible Job Shop Scheduling Problems](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c411eef5b6ee7d9895ad129a4bfc6cc-Abstract-Conference.html)，ICLR 2026。[原文](https://proceedings.iclr.cc/paper_files/paper/2026/file/5c411eef5b6ee7d9895ad129a4bfc6cc-Paper-Conference.pdf) 的核心是把偏好嵌入引入工序与机器注意力，并采用分解式PPO；一个模型处理不同偏好。考察的目标含 makespan、tardiness、earliness、平均flowtime和加工成本，但没有本项目的DIS/INS疲劳安全及劳动OWA。其成本按加工时间构造，也不等于本项目的重构费用。

作者 [代码](https://github.com/ai-for-decision-making-tue/Neural_Multi-Objective_CO_for_FJSP) 已公开，本轮读取到 `main` commit `caf60d8da721382c79009539a9c8d9ccbc39e734`。已检查的关键文件：

- [条件注意力实现](https://github.com/ai-for-decision-making-tue/Neural_Multi-Objective_CO_for_FJSP/blob/caf60d8da721382c79009539a9c8d9ccbc39e734/model/mo_attention_layer.py)：偏好参与注意力与嵌入更新。
- [PPO/GAE实现](https://github.com/ai-for-decision-making-tue/Neural_Multi-Objective_CO_for_FJSP/blob/caf60d8da721382c79009539a9c8d9ccbc39e734/model/PPO.py#L132)：多目标分支按偏好线性加权奖励和值估计。

建议先做“DCAN注意力结构＋公共环境/奖励/训练器”的模型对照，保留条件注意力这一核心区别，并添加能观察工人疲劳、资质和模块状态的输入及相应阶段候选评分。若只更换名称而仍完全使用本项目HGNN，就不能称为DCAN对照；若不提供工人状态，则可能人为削弱对照。

**奖励差异需要明确处理。** 原文§4.2采用加权和分解，不能直接照搬其向量critic与加权优势到当前切比雪夫Q。最干净的第一版是保留当前标量奖励与critic，对照注意力模型，命名为同环境适配版。若另做完整算法适配，再单独声明训练目标差异。

原文讨论“切比雪夫不具有时间可加性”，并不否定本项目对**累计Q直接做差**的奖励。`sum_t [Q(s_t)-Q(s_{t+1})]=Q(s_0)-Q(s_T)` 对非线性Q同样成立；项目的进度项、失败项及 `gamma=1` 仍按现有契约处理。不能将“逐目标增量先max后累加”与“累计质量先max再差分”混为一谈。

**PGRL：有代码，但训练方法不能靠文件名判断。**

[论文](https://doi.org/10.1016/j.swevo.2024.101605)：Fast Pareto set approximation for multi-objective flexible job shop scheduling via parallel preference-conditioned graph reinforcement learning，Swarm and Evolutionary Computation 88 (2024), 101605。[作者仓库](https://github.com/Chupeng24/PGRL)，本轮读取到 `master` commit `e63c66fbe9be26e634a2d0790aaee701ad671fa3`。

已核实的源码事实：

- [PPO_Actor1.py:116](https://github.com/Chupeng24/PGRL/blob/e63c66fbe9be26e634a2d0790aaee701ad671fa3/multi_opt_models/PPO_Actor1.py#L116) 的 `assign(pref)` 用超网络生成actor参数。它不能简单描述为“每个偏好都独立训练一个完整模型”。
- [FJSP_Env.py:141](https://github.com/Chupeng24/PGRL/blob/e63c66fbe9be26e634a2d0790aaee701ad671fa3/FJSP_Env.py#L141) 的三个奖励分量对应makespan下界、机器总负荷、最大机器负荷的增量，与本项目三目标不同。
- [ES_parallel_training.py:118](https://github.com/Chupeng24/PGRL/blob/e63c66fbe9be26e634a2d0790aaee701ad671fa3/ES_parallel_training.py#L118) 对扰动参数的正负回报进行归一化、加权噪声更新；是演化策略入口，不是PPO梯度更新。该文件评价函数也按偏好加权三项终局指标。

所以可选择保留PGRL的超网络和ES，构成与PPO更不同的学习对照，但需要对扰动引发的所有完整仿真计费。若将优化器换成项目PPO，应明确命名“PGRL-inspired hypernetwork＋PPO”，不能冒称作者原版。当前只完成源码审查，没有运行其训练/推理或验证现代依赖兼容性。

**其他候选的定位。**

- [DANIEL作者仓库](https://github.com/wrqccc/FJSP-DRL) 与 [论文](https://arxiv.org/html/2305.05119v2)：经典工序/机器双注意力；不是“双资源工人调度”的dual。原论文以makespan为核心，移植仍需工人、偏好、阶段与等待动作。DCAN已沿其架构扩展多目标，预算有限时优先DCAN即可。
- [DTPDQN作者代码](https://github.com/kpzhang/DTPDQN) 提供动态多目标双Q学习、工序/机器派工规则与扰动事件处理；[Liu等IJPR 2022作者代码](https://github.com/RK0731/Deep-reinforcement-learning-for-dynamic-scheduling-of-a-flexible-job-shop) 提供sequencing/routing学习和规则。这些可以支持“学习选择规则”的对照，但本轮仅审查仓库说明，尚未逐行验证其Q目标/训练门槛。移植时工人阶段不能固定为较弱规则；规则输出必须再通过当前合法动作集合选择。
- [A Flexible Job Shop Scheduling Problem Involving Reconfigurable Machine Tools Under Industry 5.0](https://arxiv.org/abs/2410.23302)（2024预印本，2025修订）同时研究可重构工具、工人、配置相关setup、移动和休息，并提出三层编码memetic算法。问题背景比普通FJSP更接近，但本轮未核实到作者代码或与当前三目标相同的模型，适合作编码/邻域参考，暂不作为最快落地的首选。
- [A Memetic Algorithm with Reinforcement Learning for Sociotechnical Production Scheduling](https://arxiv.org/abs/2212.10936)（IEEE Access 2023）涉及可选机器/工人、资质、setup、物料到达、部分自动化、DES与DRL，实验优化makespan和总拖期。本轮已读原文部分；文中提到公开代码，但尚未定位并核实仓库，故暂列备选，不能标为已验证可复现。

**对照设计与预算建议。**

1. 所有方案共用当前实例、物理环境、合法动作、三目标、尺度和终局验证。适配过程中只替换求解策略/网络，不给某个基线换一个简化问题。外部文献已有机制、当前工程改造和待验证表现分开记录。
2. 对搜索方法，沿用初始化/早期/600秒/必要时1800秒的验证路线作为共同试验框架。600/1800秒是候选观察点，不预先宣称对NSGA-II或MOEA/D足够；应检查完整代数/邻域数、有效不同解数和持续改进。种群大小需结合单次原生解码成本验证，不能照搬论文100种群、1000代，然后让本项目整个时限只够初始化。
3. 先固定共同初值池并计入预算，再允许各方法用剩余预算扩展。若研究专门的初始化增强，另列机制实验。统计每次完整/中断评价、代理调用、初始化耗时和截止时交付，避免将大量重复轨迹当作搜索强度。
4. **单偏好请求**：从零开始，预算内选当前偏好Q最优可行解。NSGA-II/MOEA/D即使内部维护多目标种群，其他方向的计算也计入当前请求；不能免费复用其他请求的档案。
5. **整条Pareto前沿**：NSGA-II/MOEA/D天然一次搜索输出多个点；应另设每实例总预算，与PPO全部K个偏好的总交付成本、ALNS全部K次请求的总成本比较。这种结果与独立请求端点主表分开，不能把一次种群运行的耗时摊除后声称每个偏好都独立用这么短时间完成。
6. 学习方法匹配实际环境交互预算、训练种子、数据与验证选模机会，同时单列优化步数、样本重用、训练墙钟和硬件。PGRL的正负扰动评估也属于交互成本；每偏好独立训练的模型族要累计全部训练成本。同episodes数本身不足以定义相同预算。
7. 新增随机学习策略按项目best-of-3报告，保留原始单次与选中交付，整个三元组计时；greedy单次结果另列。搜索方法同时给充分搜索曲线和远端PPO实际交付短时限结果，不把短时限结论推广为整体强弱。
8. 分层验证先行，按实例汇总偏好/搜索种子再做配对统计。搜索seed不是训练seed，三次采样不是三个独立训练模型。所有正式实验在远端执行；本轮不新增本机实验。

建议最小有效组合是：**当前PPO＋原启发式/规则组＋MO-ALNS＋NSGA-II-LS＋DCAN适配版**。若还有资源，再加MOEA/D及规则DDQN；若关注偏好生成机制或训练算法差异，再做PGRL。固定概率LNS仍用于检验ALNS自适应贡献，但与ALNS共享邻域，不能替代一个独立的种群搜索对照。

没有实验依据支持这些新增候选一定强于当前PPO、MO-ALNS或彼此。公开论文和代码支持的是“值得比较、能够设计适配”，在当前问题上的质量、完成率、延迟和训练稳定性仍须远端验证。
