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

## 流水线的 Excel 审计报告

走[配置驱动主流程](config-pipeline.md)时，除了 Report 对象，还会在输出目录
落一份多页的 `Model_Report_N.xlsx`。固定包含三页：

| Sheet | 内容 |
|-------|------|
| `Overview_Performance` | 主指标（Dev / OOT） |
| `Report_Index` | 本册所有 sheet 的用途索引 |
| `Artifact_Metadata` | 复现与部署元信息 |

按配置追加的页：

| Sheet | 触发条件 |
|-------|---------|
| `Dev_Score_Bins` / `OOT_Score_Bins` | 分类任务默认 |
| `Score_PSI` | 有 OOT 时的分数 PSI |
| `Stability_Summary` | 特征稳定性汇总 |
| `Segment_Summary` | 配了 `evaluation.segment_cols` |
| `Temporal_Stability` | 配了 `evaluation.temporal_col` |
| `Benchmark_Performance` | 配了 `evaluation.benchmark_cols` |

外加分箱、WOE、IV、变量审计和筛选过程页。

`export_excel: false` 可以关掉整份 Excel（分类和回归都生效）。
`Overview_Performance` 和 `Artifact_Metadata` 两页是[发布门禁](release-gate.md)
的检查项，关掉 Excel 后门禁的 `report_contract` 会失败。

## metadata 是审计线索

每个 report 的 `metadata` 记录了本次运行的完整口径：

```python
report.metadata
# {'workflow': 'monitor', 'benchmark': '2026-01', 'group_col': 'score_month',
#  'n_groups': 6, 'psi_include_missing': False, ...}
```

复盘"这个 PSI 当时是怎么算的"时，答案就在这里。
