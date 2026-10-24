# 实验体系发布验收

日期：2026-10-23。实验合同见[experiment_suite_contract_v1.md](experiment_suite_contract_v1.md)，三个启动入口及远端命令见[experiment_suite_v1.md](experiment_suite_v1.md)。

## 源码身份与预检

源码集合复用项目`result.provenance.source_files`：根目录Python模块、各运行包Python模块、`result/`直属Python模块，以及PowerShell启动脚本与共享模块。根目录`utils.py`等公共依赖纳入身份。回归测试验证运行模块变化会改变指纹，结果缓存目录变化不改变指纹。

已通过主实验脚本完成七种配置的CPU预检：

```powershell
.\scripts\run_main_experiment.ps1 -Stage preflight -Device cpu -OutputDir result/verification/experiment_publication_20261023
```

证据：[preflight.json](../result/verification/experiment_publication_20261023/preflight.json)、[source.zip](../result/verification/experiment_publication_20261023/source.zip)。当次身份：

`8645f739a5aed8cea4bb0e95c65d202a89f781bc221322c3519e6b46463e857b`

快照共151个源文件，全部来自本次Git交付范围。逐文件SHA256及ZIP完整性通过检查。正式远端按实际设备和worker数生成独立预检身份。

## 完整检出验证

从暂存Git树独立导出全部跟踪文件，使用项目虚拟环境运行完整测试。导出目录的`configs`和`utils`导入路径均已核对，测试直接读取该目录的代码、配置和固定数据。

结果：**124 passed，0 failures，0 errors**。证据：[tests.xml](../result/verification/experiment_publication_20261023/tests.xml)、[staged_snapshot.json](../result/verification/experiment_publication_20261023/staged_snapshot.json)。测试树哈希在后者中记录。

此前的小规模训练、重载和搜索机制证据按原源码身份保留，见[实施验收](experiment_suite_verification_20261023.md)。35次正式训练与长搜索仍由用户在远端手动启动。
