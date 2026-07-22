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

## 兼容性约定

- **Stable 模块**：只在主版本号变更时才可能有破坏性变更
- **Report 表结构**：新增列随时可能发生；下游按列名取数，不要按位置
- **生产环境**：固定版本号使用，升级前跑一遍自己的回归

## 已知行为口径

- PSI 默认不含缺失箱与特殊值箱
- `group_col` 的第一组（升序）自动作为 PSI 基准
- 分箱索引协议：正常箱 ≥ 0，缺失 -1，其他 -2，特殊值 ≤ -3
