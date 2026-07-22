# 报告与导出

所有工作流返回结构化 Report 对象；导出是对象上的方法，
而不是工作流的副作用。

## 统一结构

| 字段 | 说明 |
|------|------|
| `summary_table` | 特征级汇总（Polars DataFrame） |
| `detail_table` | 分箱级明细 |
| `trend_tables` | dict：指标 → 按期展开的宽表 |
| `metadata` | 本次运行口径（参数、基准、行数） |

## Markdown 导出

```python
md = report.to_markdown()        # 字符串，含 metadata + 所有表
report.save("reports/dq.md")     # 落盘，自动建目录
```

## 与下游系统对接

表都是 Polars DataFrame，按需转换：

```python
report.summary_table.to_pandas()             # Pandas
report.summary_table.write_csv("psi.csv")    # CSV
report.summary_table.write_parquet("x.pq")   # Parquet
report.summary_table.to_dicts()              # list[dict]，直接进 JSON
```

## metadata 是审计线索

每个 report 的 `metadata` 记录了本次运行的完整口径：

```python
report.metadata
# {'workflow': 'monitor', 'benchmark': '2026-01', 'group_col': 'score_month',
#  'n_groups': 6, 'psi_include_missing': False, ...}
```

复盘"这个 PSI 当时是怎么算的"时，答案就在这里。
