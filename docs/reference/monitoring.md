# Monitoring API

## Monitor

```python
Monitor(
    *,
    binner_params: dict | None = None,   # 默认 WoeBinner 的构造参数
    psi_include_missing: bool = False,
    psi_include_special: bool = False,
    psi_warn: float = 0.1,               # status 判定阈值
    psi_critical: float = 0.25,
)
```

构造函数只放跨运行复用的策略；一个实例可服务多个数据集。

### Monitor.monitor

```python
monitor(
    df,
    *,
    features: list[str],
    target: str | None = None,        # 启用坏率趋势
    score_col: str | None = None,     # 启用分均值趋势
    group_col: str | None = None,     # 周期列；缺省需给 benchmark_df
    binner: WoeBinner | None = None,  # 复用开发期分箱规则
    benchmark_df=None,                # 基准数据；缺省 = group_col 首期
) -> MonitoringReport
```

基准选择规则：`benchmark_df` 优先；否则取 `group_col`
排序后的第一期。分箱规则未提供时在基准切片上拟合。

返回 [`MonitoringReport`](report-objects.md#monitoringreport)。

---

## AlertConfig

```python
AlertConfig(
    psi_warn: float = 0.1,
    psi_critical: float = 0.25,
    missing_delta_warn: float = 0.05,
    missing_delta_critical: float = 0.10,
    score_mean_relative_delta_warn: float = 0.05,
    score_mean_relative_delta_critical: float = 0.10,
    max_items_per_priority: int = 10,
)
```

指标 ≥ critical 为 🔴 严重，≥ warn 为 🟡 警告，否则不出现在告警中。

---

## generate_monitoring_alert

```python
generate_monitoring_alert(
    report: MonitoringReport,
    *,
    score_col: str | None = None,       # 模型分列名，漂移按模型级报
    model_features: list[str] | None = None,  # 入模特征优先展示
    config: AlertConfig | None = None,
) -> str
```

返回按优先级排序的中文告警文本（严重在前、入模特征在前），
可直接推送 IM / 邮件。

```python
>>> print(generate_monitoring_alert(report, score_col="score"))
📡 监控报警摘要

基准：2026-01 ｜ 周期列：month ｜ 期数：3 ｜ 特征数：2

🔴 严重（需立即处理）
  - 模型分 `score` PSI 达 0.3120（最新期 0.3120），分布相对基准发生显著偏移
```
