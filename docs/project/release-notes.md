# Release Notes

## 2.1.0（2026-07-23）

以任务为中心的一次大版本迭代：新增统一工作流入口、结构化 Report
对象体系与监控告警模块。

### 新增

- **`src.analysis`** — 任务式入口
    - `profile_data(df, ...)`：数据质量画像（overview / dq / stats 三表 + 按期趋势）
    - `profile_risk(df, target=...)`：一次调用完成分箱 + IV/KS + 跨期 PSI，
      返回 `RiskProfile`（report + 可复用 binner）
- **`src.monitoring`** — 监控与告警
    - `Monitor`：PSI / 缺失率 / 坏率 / 分均值漂移监控，支持
      `group_col` 按期对比与 `benchmark_df` 对照开发样本两种基准模式
    - `generate_monitoring_alert` + `AlertConfig`：按优先级排序的中文告警文本
- **`src.reports`** — 结构化 Report 对象
    - `DataProfileReport` / `BinningReport` / `RiskProfile` / `MonitoringReport`
    - 统一 `summary_table` / `detail_table` / `trend_tables` / `metadata` 结构
    - `to_markdown()` / `save()` 导出
- **`src.evaluation.stability`** — 分箱分布 PSI 计算原语
  （`bin_distribution` / `psi_from_distributions` / `psi_level`）
- **文档站** — MkDocs Material，含 Quickstart、API 约定、使用指南与 Reference

### 变更

- 顶层包直接导出工作流 API：`from src import profile_risk, Monitor, ...`
- README 重写为"从任务开始"结构，新增稳定性分级说明

### 兼容性

- 既有低层 API（`WoeBinner` / `FeatureSelector` / `calculate_*`）完全不变
- 本版本无破坏性变更

## 2.0.0

- Polars 优先架构、WOE 分箱（quantile / uniform / cart）、
  特征筛选、模型训练与评估、自动化流水线
