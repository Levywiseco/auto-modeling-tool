# 核心 API 约定

一句话：**稳定策略放构造函数，本次运行的数据和列名放方法参数。**
一个对象可以服务多个数据切片，没有隐式状态依赖。

## 参数层级

| 放在哪 | 放什么 | 例子 |
|--------|--------|------|
| 构造函数 | 跨运行复用的策略、阈值 | `Monitor(binner_params={"n_bins": 10}, psi_warn=0.1)` |
| 方法参数 | 本次运行的数据、列名、输出 | `monitor(df, features=[...], group_col="month")` |

## 两层 API

| 层级 | 输入形态 | 例子 |
|------|----------|------|
| 高层工作流 | 整张业务表 `df, target=...` | `profile_risk(df, target="target")` |
| 底层工具 | sklearn 风格 `X, y` | `WoeBinner().fit(X, y)` |

高层工作流负责编排（转换、分箱、评估、汇总），
底层工具负责单一职责，可独立使用、可被工作流复用。

## 结构化输出优先

工作流返回 Report 对象，而不是直接写文件：

| 字段 | 内容 |
|------|------|
| `summary_table` | 特征级汇总，一行一个特征，按重要性排序 |
| `detail_table` | 分箱级明细，一行一个 (feature, bin) |
| `trend_tables` | dict：指标名 → 按 group/时间展开的宽表 |
| `metadata` | 本次运行的口径：参数、行数、基准定义 |

落盘是导出方法的事：`report.to_markdown()` / `report.save(path)`。

## 输入兼容

所有高层入口同时接受 Pandas 与 Polars DataFrame；
内部统一转为 Polars 计算。底层转换器（`WoeBinner` 等）
的输出格式跟随输入格式（Pandas 进 Pandas 出）。

## 缺失值与特殊值

- `missing_values` / `special_values` 是显式参数（如 `[-999, -1]`）
- 缺失单独成箱（bin_idx = -1），特殊值各自成箱（bin_idx ≤ -3）
- PSI 默认**不含**缺失箱与特殊值箱，可用
  `psi_include_missing` / `psi_include_special` 打开

## 关键入口速查

| 场景 | 调用 |
|------|------|
| 自动分箱 + 评估 | `profile_risk(df, target=..., group_col=...)` |
| 复用已有分箱规则 | `profile_risk(df, target=..., binner=fitted_binner)` |
| 监控 | `Monitor().monitor(df, features=..., binner=..., group_col=...)` |
| 告警文本 | `generate_monitoring_alert(report, score_col=..., model_features=...)` |
