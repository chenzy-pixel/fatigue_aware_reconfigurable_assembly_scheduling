# 三因子批次调度迁移与复现

2026-10-23 环境、图观测与数据生成复核及修复见[迁移审计](batch_migration_audit_20261023.md)。预测语义现为 `order_chain_remaining_context_v3_1`，加载模型时检查该身份。

训练 epoch、逐算例指标与论文曲线的数据口径见[训练记录契约](training_history.md)。

实施日期：2026-10-23。当前默认协议为数据3、观察13、结果10。实验规格见 `factorial27_spec.md`；数字以 `configs/default.json` 和固定实例文件为准。

## 已锁定的行为

全部订单在零时刻可用，完成全部工序即成功。机器兼容、工人资格、疲劳安全仍为硬约束。三个目标是Excess Flow、重构费用、geometric OWA(q=.5)。正常完整轨迹满足 raw Flow=ΣCj、Excess Flow=raw Flow−固定加工下界、累计奖励=1−终局Q。

资源因子M同时扩展机器、工人和模块，不能将比较结果解释为仅机器数量的因果效应。每组合一份正式实例；多偏好和多随机运行不增加独立实例数量。训练覆盖全部27组，因此正式大规模结果属于同设计空间评价。

## 接口与依赖

`FactorCell(N,K,M)`定义生成单元。`AssemblyInstance`删除时域和波次，订单release_time只允许0。环境保留reset/step接口；真实任务终止与工程truncated分别处理。工程截断保留物理状态和价值自举。

图含工序、机器、工人、订单、模块五类节点和十二类关系。模块身份由requires/supports/qualified关系及当前/目标安装标记表达；节点和锁定边不再随模块总数增加特征列。schema13节点维数为9/10/7/7/7，全局11维，偏好3维。无效投影使用None及显式有效位；其填零值不得当作真实的零耗时路径。

时间缩放τ从实例最快加工下界计算，加载后固定，时间超过τ不影响合法性。WAIT依据完成、恢复可行和恢复工期改善事件推进。恢复搜索最多到相关工人恢复为零，工程保护没有物理期限含义。生产pair+WAIT和工人pair+WAIT的动作槽位约定继续使用。

DDQN的释放优先规则改为最短剩余订单工作量。MO-ALNS的波次相关算子改为模块相关算子。策略直接评分、候选投影、训练元数据、选模和结果汇总均使用新契约。新旧消融与模型结果分别解释。

## 数据与尺度

`data/instances/v3`保存标定54、验证27、测试27份实例及证书；`data/manifests/v3`保存显式索引和文件哈希。标定、验证和测试种子空间分离。训练每轮27组合各一次，独立洗牌，延长训练长度不改变已有前缀。

`data/tables/factorial27_parameters.csv`是正式27例主参数表，`factorial27_common_parameters.csv`是公共范围，`factorial27_diagnostics.csv`记录108例的实际参考结果。参考轨迹只证明已有完整合法解，不是最优性证明，也不作为正式算法免费初始解。

`configs/factorial27_normalization.json`冻结54例完整参考目标均值。三个尺度分别为Flow 4479.251851851852、Cost 1393.7120629561937、OWA 36.93053642901436。标定时使用临时单位尺度，参考策略不读取Q排序；正式模型使用冻结尺度。

已发布的实例不自动重抽。再次调用准备脚本会校验冻结集。修改生成协议应使用新数据身份和新路径。

## 本地命令

在工程根目录，使用包含PyTorch的项目环境：

```powershell
.venv/Scripts/python.exe -m scripts.prepare_factorial27
.venv/Scripts/python.exe -m scripts.audit_factorial27
.venv/Scripts/python.exe -m scripts.run_experiments --stage preflight --device cpu
.venv/Scripts/python.exe -m pytest test -q --basetemp=.pytest_batch13
.venv/Scripts/python.exe -m scripts.run_00_smoke
```

本地smoke对PPO/DCAN/DDQN各采24个真实交互步，并以一个独立验证实例检查完整交付；三个搜索方法各探测8次完整评价额度。DDQN另有小型回放更新回归。上述结果只证明集成，不证明收敛或预算充分。

## 正式实验入口

长实验必须显式设置`--remote-run`。预算研究使用标定集，完整训练的选模使用验证集，正式评价使用测试集。例如下列数值仅是调用语法示例，不是经确认的正式预算：

```powershell
.venv/Scripts/python.exe -m scripts.comparison_suite --stage calibrate-budget --method ppo --steps 10000 --remote-run --output-dir result/budget_pilot
.venv/Scripts/python.exe -m scripts.comparison_suite --stage calibrate-budget --method nsga2_ls --seconds 60 --remote-run --output-dir result/search_budget_pilot
```

冻结预算清单需要`identity`（preflight对应的study_identity）、`status="frozen"`、正整数`training_steps`及`search_seconds`字典，三个搜索键为mo_alns/nsga2_ls/moead。根据开发曲线形成清单后，再运行：

```powershell
.venv/Scripts/python.exe -m scripts.comparison_suite --stage train --method ppo --budget-manifest path/to/budget.json --remote-run --output-dir result/formal_seed11
.venv/Scripts/python.exe -m scripts.comparison_suite --stage evaluate --method ppo --checkpoint result/formal_seed11/ppo/best_checkpoint.pt --budget-manifest path/to/budget.json --remote-run --output-dir result/formal_test
```

PPO/DCAN使用三次采样取优，DDQN单次贪心；全部候选计费。算法种子用`--seed`指定。正式训练预算和搜索充分性尚待开发实验确定，本次实施不自动启动这些实验。

## 历史恢复

审计输出中的 `protected_verified` 是实际核对成功的历史文件数，`protected_unavailable` 是本机未提供的历史文件数；缺少材料时 `history_status=partial_archive`，不表示完整历史校验。首次克隆可独立验证当前固定实例和证书。核验完整历史归档时可调用 `audit(config, require_history=True)`，此时任何缺失文件都会报错；所有模式均拒绝现存文件的哈希变化。

`result/migration/factorial27_20261023/before.json`记录原提交3dbe6073bc1352f8d69b3c1700001d7e07805d66、源文件和3836个历史产物哈希；`source_before.zip`保存293份源文件，`dependencies.txt`记录运行依赖。旧测试和退休入口保存在`archive/schema12`。历史实验使用原代码版本和原配置解释，不让旧checkpoint继承新默认值。

未跟踪的原始三因子DOCX草稿和其他并发文件未覆盖。当前规格与机器可读配置替代其中的波次设定。

当前消融入口清单为`configs/factorial27_ablations.json`，包括节点MLP、三种单目标的疲劳中性控制以及具有独立冻结尺度的variance控制。历史参数扫描配置继续按原协议复现。原`train.py`仍可用于小型集成采样；正式匹配步数训练使用comparison_suite入口。正式结果目录可用`python -m scripts.summarize_factorial27 <目录>`导出逐请求及逐实例CSV。
