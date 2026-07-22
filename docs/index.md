# Auto Modeling Tool

**接收 Pandas 或 Polars 数据，返回结构化 Report 对象。**

Auto Modeling Tool 是一个 Polars 优先的信贷风控建模工具库，覆盖
数据画像、分箱评估、特征筛选、建模、监控与评分卡交付的完整工作流。

```python
import polars as pl
from src import profile_risk

df = pl.DataFrame({
    "month": ["2026-01"] * 4 + ["2026-02"] * 4 + ["2026-03"] * 4,
    "income": [3200, 3600, -999, None, 3300, 4200, -999, 5800,
               3400, 4300, None, 6100],
    "utilization": [0.12, 0.18, 0.52, 0.61, 0.14, 0.29, 0.54, 0.58,
                    0.16, 0.31, 0.56, 0.63],
    "target": [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
})

profile = profile_risk(
    df,
    target="target",
    features=["income", "utilization"],
    group_col="month",
    n_bins=4,
    missing_values=[-999],
    special_values=[-999],
)

profile.summary_table          # 特征级 iv / ks / psi_max
profile.detail_table           # 分箱明细
profile.binner                 # 可复用的分箱规则
```

## 从任务开始

| 任务 | 入口 | 文档 |
|------|------|------|
| 🔍 数据质量检查 | `profile_data(df, ...)` | [数据画像](guide/data-profiling.md) |
| 🎯 特征风险评估 | `profile_risk(df, target=...)` | [分箱与风险评估](guide/binning-risk-evaluation.md) |
| 🧪 候选特征筛选 | `FeatureSelector` | [特征筛选](guide/feature-selection.md) |
| 📡 特征 / 模型监控 | `Monitor().monitor(df, ...)` | [监控](guide/monitoring.md) |
| 📋 报告交付 | `report.to_markdown()` | [报告与导出](guide/reports.md) |

## 入口怎么选

| 目标 | 推荐 API | 输出 |
|------|----------|------|
| 拿到一张宽表，先看质量 | `profile_data` | `DataProfileReport` |
| 评估特征区分度与跨期稳定性 | `profile_risk` | `RiskProfile`（report + binner） |
| 上线后按月监控 | `Monitor().monitor` | `MonitoringReport` |
| 把监控结果推送到 IM | `generate_monitoring_alert` | 中文告警文本 |
| 细粒度控制分箱 | `WoeBinner` | sklearn 风格转换器 |

## 版本稳定性

生产环境请固定版本号。各模块稳定等级见[稳定性与兼容性](project/stability.md)：
`binning` / `features` / `evaluation` / `analysis` / `reports` 为 **Stable**，
`monitoring` / `pipelines` 为 **Experimental**。
