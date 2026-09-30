# 算例协议 v2 数据审计

开发数据于 2026-09-30 在独立 v2 目录物化。生成及首次审计耗时 339.24 秒。565 个历史文件的字节哈希均通过保护检查。

| 集合 | 数量 | observed_feasible | unknown | completed 诊断 | truncated 诊断 |
|---|---:|---:|---:|---:|---:|
| validation | 500 | 484 | 16 | 484 | 16 |
| test | 20 | 19 | 1 | 19 | 1 |
| ood | 20 | 20 | 0 | 20 | 0 |
| stress | 20 | 20 | 0 | 20 | 0 |

未知实例的诊断原因：

| 场景 | 结束原因 | 数量 |
|---|---|---:|
| reconfiguration_bottleneck | horizon | 16 |

验证池场景数量（固定顺序）：

| easy | balanced | machine | reconfiguration | worker | fatigue | arrival |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 175 | 75 | 75 | 50 | 50 | 50 |

选模子集配额 3/18/7/7/5/5/5；诊断子集每场景 7 个。两者无重叠。显式索引和实例哈希见 [subsets.json](../data/manifests/v2/validation/subsets.json)。

- diagnostic 子集 SHA256：93a35beb6b188fe421cf240e31f58ff679785456c513fea784256c3318f1694c
- target 子集 SHA256：b54cbf99baa94debc48ec245a163da8269dd2dc2d0fab8094c051b5eb65d57f5

默认 2000-episode、种子 11 计划的每类场景均覆盖 Flow、Cost、Variance 三个端点及混合偏好。窗口配额和强度规则见 [实验规格](data_v2_specification.md)。

完整静态拒绝统计、交叉覆盖和子集契约见 [audit.json](../data/manifests/v2/audit.json)。未知标签表示尚无完整、安全的可行调度证据。
