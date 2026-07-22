# 10 分钟 Quickstart

用一份带月份、缺失值和特殊值的信贷样例数据，走完
**画像 → 风险评估 → 监控 → 告警** 四步。

## 0. 准备数据

```python
import numpy as np
import polars as pl

np.random.seed(7)
n = 1200
months = ["2026-01", "2026-02", "2026-03"]

utilization = np.random.beta(2, 5, n)
income = np.random.lognormal(8, 0.5, n)
income[:60] = -999                       # 上游给的特殊值
target = np.random.binomial(1, np.clip(0.05 + 0.5 * utilization, 0, 1))

df = pl.DataFrame({
    "month": [months[i % 3] for i in range(n)],
    "income": income,
    "utilization": utilization,
    "target": target,
})
```

## 1. 数据画像

```python
from src import profile_data

dq = profile_data(df, target="target", group_col="month",
                  special_values=[-999])

dq.overview_table                  # 行数 / 列数 / 坏率 / 重复行
dq.dq_table                        # 每个特征的缺失率、特殊值率、唯一值数
dq.trend_tables["missing_rate"]    # 缺失率按月趋势
```

## 2. 特征风险评估

```python
from src import profile_risk

profile = profile_risk(
    df,
    target="target",
    features=["income", "utilization"],
    group_col="month",             # 首月自动作为 PSI 基准
    n_bins=4,
    missing_values=[-999],
    special_values=[-999],
)

profile.summary_table
# ┌─────────────┬────────┬─────────────┬────────┬──────────────┬────────┐
# │ feature     ┆ iv     ┆ iv_strength ┆ ks     ┆ missing_rate ┆ psi_max│
# ╞═════════════╪════════╪═════════════╪════════╪══════════════╪════════╡
# │ utilization ┆ 0.31   ┆ strong      ┆ 0.28   ┆ 0.0          ┆ 0.01   │
# │ income      ┆ 0.02   ┆ weak        ┆ 0.05   ┆ 0.0          ┆ 0.02   │
# └─────────────┴────────┴─────────────┴────────┴──────────────┴────────┘

profile.detail_table               # 每个箱的 count / bad_rate / woe / iv
profile.report.trend_tables["psi"] # 特征 × 月份 PSI 矩阵
```

## 3. 上线监控

```python
from src import Monitor

report = Monitor().monitor(
    df_prod,                       # 生产期数据
    features=profile.features,
    target="target",
    group_col="score_month",
    binner=profile.binner,         # 复用开发期分箱规则，口径一致
)

report.summary_table               # psi_max / missing_delta / status
```

## 4. 中文告警

```python
from src import generate_monitoring_alert

print(generate_monitoring_alert(report, model_features=profile.features))
```

```
📡 监控报警摘要

基准：2026-01 ｜ 周期列：score_month ｜ 期数：3 ｜ 特征数：2

🟡 警告（建议关注）
  - 入模特征 `utilization` PSI 达 0.1523（最新期 0.1523），分布相对基准发生一定偏移
```

## 下一步

- [核心 API 约定](api-conventions.md) — 参数放哪里、返回什么
- [特征与模型监控](../guide/monitoring.md) — 基准选择、告警阈值
- [Report 对象](../reference/report-objects.md) — 所有表结构速查
