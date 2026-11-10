# 本轮实验结果

本发布收录 Full、三个结构消融、Rule-DDQN、DCAN、Full 两种解码对照、五条规则及两个搜索方法。学习方法使用五种子（11、23、37、53、71），固定测试算例 27 个；正式任务为 66 偏好前沿与四偏好独立请求。A2C 的实现和实验配置保留，其未完成运行不进入结果表。

本轮收录 30 次完成训练、3,780 个正式评测文件和 103,950 条交付记录。完整附件按方法和任务分成 85 个独立 ZIP；仓库保存压缩后的训练与评测明细，以及可直接阅读的汇总表图。

## 阅读与复算

- `tables/summary.csv`：按全部实例、规模及因素分组的指标。
- `tables/per_instance.csv`：先在每个固定实例内汇总种子和偏好，供实例配对分析。
- `tables/paired_statistics.csv`：Full 与各方法的配对检验、分层 bootstrap 区间和组内 Holm 校正；差值为对照方法减 Full，统计独立单位为固定实例。
- `details/*.jsonl.gz`：逐评测单元、逐偏好的交付记录、候选目标值、成功状态、搜索快照、耗时与来源身份。删除轨迹后保留复算指标所需字段；每条有独立内容校验值。
- `training/`：30 次完成训练的逐例、逐 Epoch 和验证 CSV（gzip 压缩）、有效配置和摘要。
- `origins.json`：四个来源批次、硬件、原配置和固定数据清单；`audit.json` 为本发布核验结果。

在项目根目录安装依赖后执行：

```shell
python -m scripts.publish_results recompute
```

此命令仅使用本目录明细和仓库固定数据重建 `tables/` 与 `figures/`。重新训练或运行模型评价需要自行准备模型权重及对应运行环境。

## 原始记录

[本轮 Release](https://github.com/chenzy-pixel/fatigue_aware_reconfigurable_assembly_scheduling/releases/tag/experiments-20261110) 提供分组 ZIP：正式评测、详细验证、搜索预算开发记录，以及按执行身份去重的源码快照。每个 ZIP 独立解压，目录关系与原实验一致。

`assets.json` 列出资产字节数与 SHA-256。自动下载并校验，或对已有附件复核：

```shell
python -m scripts.download_results --assets /path/to/downloads
python -m scripts.publish_results verify-assets --assets /path/to/downloads
```

`files.csv` 对照原始相对路径、所属 ZIP、原文件与发布文件 SHA-256。发布副本将电脑绝对路径替换为 `<WORKSPACE>` 或 `<LOCAL>`，数值结果保持一致。原 JSON 内的 `content_sha256` 与 sample1 父引用继续标识原始证据；发布副本完整性由 `files.csv` 中的 `published_sha256` 核验。

不同批次保留各自执行身份和硬件记录，运行时间须结合测时口径解释。Full 的前沿 sample1 复用已付费的 repeat 0；规则复用确定性求解后重新评分。600 秒搜索结果表示指定预算下的表现，训练与搜索预算充分性尚未建立。
