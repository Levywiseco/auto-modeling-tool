# Report 对象

所有 Report 对象共享导出方法：`to_markdown(max_rows=50)` 与
`save(path)`。表均为 Polars DataFrame。

| 对象 | 稳定性 | 产出于 |
|------|--------|--------|
| `DataProfileReport` | Stable | `profile_data` |
| `BinningReport` | Stable | `profile_risk`（作为 `.report`） |
| `RiskProfile` | Stable | `profile_risk` |
| `MonitoringReport` | Experimental | `Monitor.monitor` |

## DataProfileReport

| 字段 | 类型 | 内容 |
|------|------|------|
| `overview_table` | DataFrame | 数据集级事实（行数、坏率、重复行…） |
| `dq_table` | DataFrame | 每特征：缺失/特殊/唯一值/旗标 |
| `stats_table` | DataFrame | 数值特征描述统计（排除特殊值） |
| `trend_tables` | dict | `row_count` / `missing_rate`（有 group_col 时） |
| `metadata` | dict | 运行口径 |

## BinningReport

| 字段 | 类型 | 内容 |
|------|------|------|
| `summary_table` | DataFrame | 每特征：iv / iv_strength / ks / missing_rate / n_bins / monotonic_woe / psi_max |
| `detail_table` | DataFrame | 每 (feature, bin)：bin_label / count / bad_rate / woe / iv |
| `trend_tables` | dict | `psi` / `missing_rate` / `bad_rate`（有 group_col 时） |
| `metadata` | dict | 分箱参数与运行口径 |

## RiskProfile

| 字段 | 类型 | 内容 |
|------|------|------|
| `report` | BinningReport | 评估结果 |
| `binner` | WoeBinner | 拟合好的分箱规则，可直接 `transform` 新数据 |
| `features` | list[str] | 实际完成分箱的特征 |
| `target` | str | 目标列 |
| `metadata` | dict | 运行口径 |
| `summary_table` / `detail_table` | property | `report` 同名表的快捷方式 |

## MonitoringReport

| 字段 | 类型 | 内容 |
|------|------|------|
| `summary_table` | DataFrame | 每特征：psi_latest / psi_max / missing_delta / status |
| `detail_table` | DataFrame | 每 (feature, group, bin) 分布 vs 基准 |
| `trend_tables` | dict | `psi` / `missing_rate`（+`bad_rate` / `score_mean`） |
| `features` | list[str] | 监控的特征 |
| `target` | str \| None | 目标列 |
| `binner` | WoeBinner | 监控所用分箱规则 |
| `metadata` | dict | 基准定义、期数、口径 |

## 分箱索引协议

| bin_idx | 含义 |
|---------|------|
| `0, 1, 2, …` | 正常箱，从低到高 |
| `-1` | 缺失（null / NaN / missing_values） |
| `-2` | 其他（未匹配的类别值） |
| `-3, -4, …` | 特殊值箱，按 `special_values` 顺序递减 |
