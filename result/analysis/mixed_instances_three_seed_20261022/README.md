# HGNN-PPO：三 seed 实验结果

更新：2026-10-22（UTC+8）。训练种子为 11、23、37；使用固定混合实例训练 2000 轮，schema12、log2、shared-head、Excess Flow、OWA(q=.5)、γ=1、λ=.95。各模型先在同一实例和偏好内三次采样取优，再跨训练 seed 等权汇总。

## 正式测试结果

每个模型在固定 test 的 20 实例 × 66 偏好上产生 3960 条原始轨迹、1320 份交付。Q 为成功交付的偏好均衡质量，越低越好；完成率单独报告。

| Seed | 训练起止（UTC+8） | Best 轮次 | 测试 Q ↓ | 交付完成率 | 原始完成率 |
|---|---|---:|---:|---:|---:|
| 11 | 10-19 20:00 → 10-20 17:19 | 2000 | 0.228057 | 100% | 99.369% |
| 23 | 10-20 17:20 → 10-21 14:17 | 1500 | 0.236580 | 100% | 98.636% |
| 37 | 10-21 14:18 → 10-22 11:14 | 1900 | 0.233998 | 100% | 97.045% |
| 均值 | | | **0.232879** | **100%** | **98.350%** |

测试 Q 的跨 seed 样本标准差为 0.004370，全部 3960 份交付成功。原始训练、验证、诊断和测试核验记录见 [audit.json](audit.json)，原始汇总见 [run_summary.csv](run_summary.csv)。

## 偏好响应

按每实例计算目标权重与目标值的 Spearman 相关系数，再对实例和三个训练 seed 平均。越负表示整体响应方向越明确。

| 目标 | HGNN |
|---|---:|
| Flow | −0.805 |
| Cost | −0.445 |
| OWA | −0.422 |

数值来自 [preference_correlations_matched.csv](preference_correlations_matched.csv)，沿用历史对照的共同成功组合：seed 23 为 1316 格，另两个 seed 各 1320 格。平均方向不代表每个实例都严格单调。

## 与启发式比较

7 实例、Flow/Cost/OWA/等权四偏好。HGNN-PPO 为三个 best 模型各自三次取优后的均值；NSGA-II＋LS 与 MOEA/D 使用冻结的搜索 seed 11、每请求 600 秒结果。总体是四偏好等权平均。

| 偏好 | HGNN-PPO | NSGA-II＋LS | MOEA/D |
|---|---:|---:|---:|
| Flow | 0.479841 | **0.458413** | 0.468982 |
| Cost | **0.298329** | 0.544813 | 0.567689 |
| OWA | **0.229230** | 0.391858 | 0.443629 |
| 等权 | **0.193082** | 0.200164 | 0.212662 |
| 总体 | **0.300121** | 0.398812 | 0.423240 |

该比较沿用搜索调参的 7 例，完整方法、数据范围和比较边界见 [report.md](report.md)。逐请求证据见 [unified_raw_samples.csv](unified_raw_samples.csv)、[unified_selected_samples.csv](unified_selected_samples.csv) 和 [unified_verification.json](unified_verification.json)。

## 图表

![训练 Reward 与训练成功率](mix_figures/01_mix_reward.png)

![验证集收敛与独立诊断](mix_figures/02_mix_convergence.png)

![Best 模型偏好响应](mix_figures/03_mix_preference_response.png)

![HGNN-PPO 与启发式比较](mix_figures/04_hgnnppo_heuristic_comparison.png)

高清 PNG、矢量 PDF/SVG 及重绘脚本见 [图表目录](mix_figures/README.md)。

## 归档与核验

- `archives/seed11_logs.zip`、`seed23_logs.zip`、`seed37_logs.zip`：各 18 份原始日志、指标、保存配置与训练摘要，保留完整原始字节。
- `archives/heuristic_evidence.zip`：seed 11/23 的补评请求和冻结搜索请求；seed 37 的补评请求直接保存在 [seed37](seed37/) 下。
- [publication_manifest.json](publication_manifest.json)：每份文件和压缩归档的大小、SHA256、训练源码身份及 checkpoint 哈希。本次提交不包含 best/last 模型权重。
- [.gitattributes](.gitattributes) 固定本目录的原始字节，避免换行转换改变归档证据。

在项目根目录核验全部归档和哈希（仅使用 Python 标准库）：

```powershell
python result/analysis/mixed_instances_three_seed_20261022/archive_results.py --verify
```

需要恢复原始目录布局时：

```powershell
python result/analysis/mixed_instances_three_seed_20261022/archive_results.py --restore
```

恢复时保留内容相同的现有文件；同路径内容不一致会停止。归档不携带模型，重新运行策略评估仍需对应本地 checkpoint。

训练基点是 `58d5845`，归档源码状态 SHA256 为 `48f9b5e4be66ccab4169dedf7927fe97773ac1de7eb4b9679ea9c11e81af02ef`。历史训练时含未提交源码改动，具体身份保存在原配置、摘要及核验记录中。完整对照报告及研究分析复算还使用此前的渐进训练产物；本次训练日志归档对应上述三个固定混合 run。
