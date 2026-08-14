# Release Notes

## 2.2.0（2026-08-14）

对齐 `USER_GUIDE 0.75` 的生产化加固：防泄漏的 Dev/OOT 建模、配置驱动、
可部署的打分 Artifact、多页审计报告与发布门禁。

### 新增

- **配置驱动主流程** — `python -m src.main --config configs/pipeline_config.yaml`
    - `src.config`：YAML schema、相对路径按配置文件解析、旧键别名兼容
    - 29 个新 CLI 参数，每项配置都能临时覆盖
    - 见[配置驱动主流程](../guide/config-pipeline.md)
- **打分 Artifact** — `src.modeling.artifact`
    - `scoring_artifact.pkl` 打包预处理器、分箱、筛选器与模型
    - `score_with_artifact` 拿原始字段直接打分，带列契约校验
    - 见[独立打分与 Artifact](../guide/scoring-artifact.md)
- **回归任务** — `src.pipelines.regression_pipeline`
    - RMSE / MAE / R²，可选 `log1p` 目标变换（打分自动逆变换）
    - 见[回归任务](../guide/regression.md)
- **发布门禁** — `src.evaluation.quality_gate`
    - Artifact 契约 → 字段契约 → 报告 sheet 契约 → 可选 AUC/PSI 阈值
    - 见[发布门禁](../guide/release-gate.md)
- **Excel 审计报告** — `src.reports.excel`，分箱 / WOE / IV / 变量审计 /
  筛选 / Dev-OOT 指标 / 分数分箱 / Score PSI / 稳定性 / 分群 / 跨期 / 基准
- **信用分转换** — 可配 `base_score` / `pdo` / 上下限
- **独立脚本** — `score_new_samples` / `evaluate_model_performance` /
  `explore_dataset` / `validate_release`，见 [CLI 与脚本](../reference/cli.md)
- **样本权重**贯穿分箱统计 → IV → 筛选 → 训练 → 评估 → PSI → 监控
- **模型选择** — logistic / tree / random forest / XGBoost / LightGBM /
  CatBoost，树模型支持 `dev_holdout` / `oot` 早停
- **数据格式** — CSV / XLSX / XLS / Parquet / PKL
- **CI** — GitHub Actions，Python 3.10 / 3.11 / 3.12，103 个测试

### 变更

- Dev/OOT 切分支持样本标签、日期边界与随机兜底；预处理、WOE 与特征筛选
  **只在 Dev 上 fit**，避免 OOT 信息泄漏
- WOE 支持类别变量、未见/稀有类别、平滑与数值变量单调校准
- Model Report 固定包含 `Overview_Performance` / `Report_Index` /
  `Artifact_Metadata`
- 监控支持数值与类别特征、分组趋势、加权 PSI
- Pandas 与 CatBoost 变为显式运行时依赖
- 日志在 Windows GBK 控制台下不再乱码

### 修复

- WOE 筛选结果算完了但没真正进入建模
- `ScorecardBuilder` 按 bin index 而非 bin 值取分
- 重复分数值导致 lift 表和报告生成失败
- `export_excel: false` 未被分类/回归流水线遵守
- 打分输出列统一为 `pred_proba`，保留 `prediction` 兼容别名

### 兼容性

- v2.1.0 的 9 个 CLI 参数全部保留
- `--test-size` 语义变为"没有样本列和日期边界时，随机切出的 OOT 比例"
- v2.1.0 的低层 API 不变

!!! warning "升级后 OOT 指标可能变差"
    2.2.0 之前预处理和分箱是在全量数据上 fit 的，OOT 指标偏乐观。
    修好泄漏之后，同一份数据的 OOT AUC 通常会低一些 —— 那才是真实水平。

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
