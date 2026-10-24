# 实验体系v1实施验收

日期：2026-10-23。规格和启动命令见[experiment_suite_v1.md](experiment_suite_v1.md)。以下为本地代码、接口与机制证据。

## 当前实现

已接入三个训练消融、独立A2C更新、五种固定规则、两种进化算法的单次前沿与独立请求入口、35次训练矩阵、逐请求计费和实例级统计。默认训练合同为每种子2000次算例尝试、100个Epoch、20次验证。

检查点包含消融与算法身份；A2C在修改权重前拒绝其他算法检查点。检查点及结果原子写入。原始验证请求按Epoch保存；报告逐文件审计完整性和内容哈希。正式长阶段要求同身份预检和`--remote-run`，同目录通过操作系统锁保证单作业执行。

## 自动测试

命令：

```powershell
.\.venv\Scripts\python.exe -m pytest test -q --basetemp=.ep6 --junitxml=result/verification/experiment_suite_20261023_final_tests.xml
```

结果：**111 passed，0 failures，0 errors**。证据：[JUnit XML](../result/verification/experiment_suite_20261023_final_tests.xml)。覆盖默认Full保持一致、生产/工人/WAIT直接通路、单条与批量消融一致、残差冻结、A2C一次累积更新、终止/截断自举、2000次尝试与20次验证计数、规则合法性/回退/WAIT、前沿预算、截止快照、完整矩阵与条件成功统计、预算冻结及锁释放。

报告完整流水线使用合成结果验证CSV、配对统计、全部图表生成，并查看代表性前沿和学习曲线的版式；合成数据只作为软件测试夹具。

## 真实小规模联调

命令：

```powershell
.\.venv\Scripts\python.exe -m scripts.experiment_suite --stage smoke --device cpu --output-dir result/verification/experiment_suite_20261023_final
```

证据：[smoke/verification.json](../result/verification/experiment_suite_20261023_final/smoke/verification.json)。每配置2次算例尝试、1个Epoch、1次单偏好验证，使用缩小的HGNN网络；DDQN预热参数也为验收临时缩小。验证与测试推理候选按各算法交付方式执行。

|配置|训练实际env.step|优化器更新数|主评测候选数|重载与评测|
|---|---:|---:|---:|---|
|Full|615|20|3|通过|
|w/o Graph|482|16|3|通过|
|w/o Direct|669|22|3|通过|
|w/o Residual|615|20|3|通过|
|HGNN-A2C|615|1|3|通过|
|Rule-DDQN|493|492|1|通过|
|DCAN-adapted|692|22|3|通过|

七种配置保存的实例哈希和偏好序列一致；评测保持模型权重不变；Full的sample1与best-of-3的repeat 0动作轨迹一致；逐Epoch原始验证动作数与验证env.step计数一致。五规则在小实例合法完成。

NSGA-II-LS和MOEA/D各自完成前沿及独立请求联调，每次搜索至多3秒/40次原生解码，种群8；前沿只创建一个搜索，独立请求分别创建新搜索，候选及档案成功序列化。输出为同一smoke目录的`nsga2_ls_*.json`和`moead_*.json`。

联调曾发现前沿结果转换误用了不存在的偏好`.key`接口；已改用统一`preference_key`函数，补充入口级回归并重新完整联调通过。

## 预检与源码身份

历史PowerShell包装入口完成七种配置的CPU预检，本地记录位于`result/verification/experiment_suite_20261023_final/preflight.json`。当次身份：

`15f15f2d991873fe2f8428003cce0d3bff0f0bd8c459a4cbb3741012487d2a6b`

该身份与当次真实联调记录相同。历史预检和源码快照保留在本地，按当时源码解释。发布版本的预检、源码集合及完整检出验证见[发布验收](experiment_publication_verification_20261023.md)。正式远端按实际设备和worker数重新预检，保存其自己的身份。

## 后续远端实验

本地验收确认实现与机制；35次正式训练、30次600秒开发检查及1350次正式搜索尚未启动。600秒搜索预算清单需开发检查全部通过后由`freeze-budget`生成。训练充分性、搜索充分性及方法性能均待正式产物验证。
