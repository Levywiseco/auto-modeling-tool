# 稳定性与兼容性

## 模块稳定等级

| 模块 | 等级 | 承诺 |
|------|------|------|
| `src.binning` | **Stable** | 接口不做破坏性变更 |
| `src.features` | **Stable** | 同上 |
| `src.evaluation` | **Stable** | 同上 |
| `src.analysis`（`profile_data` / `profile_risk`） | **Stable** | 2.1.0 起提供 |
| `src.reports` | **Stable** | 表**新增列**不视为破坏性变更 |
| `src.monitoring` | **Experimental** | 告警阈值、文案、summary 列可能调整 |
| `src.pipelines` | **Experimental** | 接口可能调整 |
| `src.modeling.tuning` / `calibration` | **Experimental** | 接口可能调整 |
| `src.config`（YAML schema） | **Experimental** | 2.2.0 起提供；字段名可能调整，旧键保留别名 |
| `src.modeling.artifact` | **Experimental** | 2.2.0 起提供；artifact 格式带 `artifact_version` |
| `src.evaluation.quality_gate` | **Experimental** | 2.2.0 起提供；检查项可能增加 |

## 兼容性约定

- **Stable 模块**：只在主版本号变更时才可能有破坏性变更
- **Report 表结构**：新增列随时可能发生；下游按列名取数，不要按位置
- **生产环境**：固定版本号使用，升级前跑一遍自己的回归

## 已知行为口径

- PSI 默认不含缺失箱与特殊值箱
- `group_col` 的第一组（升序）自动作为 PSI 基准
- 分箱索引协议：正常箱 ≥ 0，缺失 -1，其他 -2，特殊值 ≤ -3
- 预处理 / WOE / 特征筛选**只在 Dev 上 fit**（2.2.0 起），OOT 只做 transform
- Artifact 格式版本独立于包版本，见 `artifact["artifact_version"]`
- 打分输出列：分类 `pred_proba`（`prediction` 为兼容别名），回归 `pred_value`

## Python 版本支持

| Python | 状态 |
|--------|------|
| 3.10 / 3.11 / 3.12 | CI 每次提交自动验证 |
| 3.9 | 支持（`requires-python = ">=3.9"`），2.2.0 人工验证通过，未进 CI 矩阵 |
