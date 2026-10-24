# 三个PowerShell启动入口验收

日期：2026-10-23。启动命令及共同参数见[实验体系v1](experiment_suite_v1.md)。

## 分组与执行

|文件|方法|默认执行顺序|
|---|---|---|
|`scripts/run_main_experiment.ps1`|Full、Rule-DDQN、DCAN，五种子共15次训练|preflight → train → evaluate|
|`scripts/run_ablations_a2c.ps1`|三个单因素消融、HGNN-A2C，五种子共20次训练|preflight → train → evaluate|
|`scripts/run_rules_search.ps1`|五规则、两种进化算法，前沿和独立请求分别评价|preflight → rules → develop-search → freeze-budget → search|

三组共用`result/experiment_suite_v1`及同一清单、设备、worker和源码身份。第三个入口的`-Stage report`在所有结果齐备后生成完整汇总。公共编排位于`ExperimentLaunch.psm1`，经既有`run_experiment_suite.ps1`调用同一Python入口。

默认`-Stage all`执行本组流程，正式阶段要求`-RemoteRun`；`-DryRun`只返回阶段及参数。默认解释器优先使用仓库`.venv`。任一步失败立即停止后续阶段。

## 测试

`test/test_experiment_launchers.py`通过真实PowerShell进程执行三个脚本，使用参数捕获器替代Python实验进程。检查阶段顺序、7×5训练矩阵完整且无重复、工作目录恢复、带空格和单引号路径、种子/任务/worker传递、正式开关及失败停止行为；这些测试不启动训练或搜索。

全量回归结果：**123 passed，0 failures，0 errors**，无测试警告。命令与结果见[JUnit XML](../result/verification/experiment_launchers_20261023_final_tests.xml)：

```powershell
.\.venv\Scripts\python.exe -m pytest test -q --basetemp=.lp2 --junitxml=result/verification/experiment_launchers_20261023_final_tests.xml
```

## 真实预检

已通过新主实验脚本运行七种配置的真实CPU预检，使用自动发现的项目虚拟环境：

```powershell
.\scripts\run_main_experiment.ps1 -Stage preflight -Device cpu -OutputDir result/verification/experiment_launchers_20261023
```

上述历史预检记录和源码快照保留在本地`result/verification/experiment_launchers_20261023/`。当次身份为：

`661020a07b2aff322a6d7f4474f4dbce9335bd57f3bd13619731d4b509e26ccb`

启动脚本和共享模块纳入源码哈希、快照及预检身份。发布版本的预检与完整检出验证见[发布验收](experiment_publication_verification_20261023.md)。七种算法的小规模机制证据见[实现验收记录](experiment_suite_verification_20261023.md)，各记录按当时源码身份解释。

本轮只执行参数捕获测试、自动回归与真实预检。35次正式训练及长搜索等待用户在远端通过三个入口手动启动。
