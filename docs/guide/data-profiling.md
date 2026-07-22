# 数据画像

建模前先回答三个问题：**数据缺不缺、脏不脏、稳不稳。**
`profile_data` 一次调用给出全部答案。

## 基本用法

```python
from src import profile_data

report = profile_data(
    df,
    target="target",           # 可选：在 overview 中给出坏率
    group_col="month",         # 可选：趋势表按月展开
    special_values=[-999, -1], # 与真实缺失分开统计
)
```

## 三张核心表

### overview_table — 数据集全貌

行数、特征数、数值/类别列数、重复行数、估算内存、坏率、期数。

### dq_table — 每特征质量

| 列 | 含义 |
|----|------|
| `dtype` | 列类型 |
| `n_missing` / `missing_rate` | 缺失（null + NaN）计数与占比 |
| `n_special` / `special_rate` | 特殊值计数与占比 |
| `n_unique` | 唯一值数 |
| `flags` | `high_missing`（缺失率 ≥ 阈值）/ `constant`（常量列） |

按缺失率降序排列，问题特征一眼可见。

### stats_table — 数值分布

排除缺失与特殊值后的 count / mean / std / min / p25 / median / p75 / max。
特殊值不参与统计，避免 `-999` 拉爆均值。

## 趋势表

给了 `group_col` 时，`trend_tables` 提供：

- `row_count` — 每期进件量
- `missing_rate` — 特征 × 期缺失率宽表（突然升高通常意味着上游取数变更）

## 导出

```python
report.to_markdown()          # Markdown 字符串
report.save("dq_report.md")   # 直接落盘
```

!!! tip "何时使用"
    每次拿到新数据源、每次上游宣称"没有变更"、每次模型效果莫名下降时。
