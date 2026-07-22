# 特征与模型监控

模型上线只是开始。`Monitor` 用与开发期**同一套分箱规则**跟踪
特征分布、缺失率、坏率与模型分的漂移，并生成中文告警。

## 两种基准模式

=== "按期对比（group_col）"

    ```python
    from src import Monitor

    report = Monitor(binner_params={"n_bins": 10}).monitor(
        df,
        features=["income", "utilization", "model_score"],
        target="target",
        score_col="model_score",
        group_col="score_month",   # 首期自动作为基准
    )
    ```

=== "对照开发样本（benchmark_df）"

    ```python
    report = Monitor().monitor(
        prod_df,                    # 生产数据
        features=model_features,
        benchmark_df=dev_df,        # 开发样本作基准
        binner=profile.binner,      # 开发期分箱规则
    )
    ```

## summary_table 与 status 判定

| 列 | 含义 |
|----|------|
| `psi_latest` / `psi_max` | 最新期 / 历史最大 PSI（相对基准） |
| `missing_rate_benchmark` / `missing_rate_latest` / `missing_delta` | 缺失率漂移 |
| `status` | `正常`（PSI<0.1）/ `警告`（0.1-0.25）/ `严重`（≥0.25） |

阈值可在构造函数调整：`Monitor(psi_warn=0.1, psi_critical=0.25)`。

## 趋势表

| 表 | 触发条件 | 内容 |
|----|----------|------|
| `trend_tables["psi"]` | 总是 | 特征 × 期 PSI |
| `trend_tables["missing_rate"]` | 总是 | 特征 × 期缺失率 |
| `trend_tables["bad_rate"]` | 给了 `target` | 每期样本量与坏率 |
| `trend_tables["score_mean"]` | 给了 `score_col` | 每期分均值与标准差 |

## 中文告警

```python
from src import generate_monitoring_alert, AlertConfig

text = generate_monitoring_alert(
    report,
    score_col="model_score",
    model_features=["income", "utilization"],  # 入模特征优先级更高
    config=AlertConfig(psi_warn=0.1, psi_critical=0.25,
                       missing_delta_warn=0.05),
)
print(text)   # 直接推飞书 / 企微 / 邮件
```

输出按 🔴 严重 → 🟡 警告排序，入模特征与模型分排在前面；
一切正常时输出"✅ 各项指标均在阈值内，无需处理"。

!!! warning "PSI 口径"
    PSI 默认不含缺失箱和特殊值箱（它们的漂移由 `missing_delta`
    单独衡量）。如需并入，构造 `Monitor(psi_include_missing=True,
    psi_include_special=True)`。
