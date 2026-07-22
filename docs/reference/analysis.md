# Analysis API

## profile_data

```python
profile_data(
    df,                              # Pandas / Polars DataFrame
    *,
    features: list[str] | None = None,
    target: str | None = None,
    group_col: str | None = None,
    special_values: list | None = None,
    high_missing_thr: float = 0.9,
) -> DataProfileReport
```

| 参数 | 说明 |
|------|------|
| `df` | 完整业务表 |
| `features` | 要画像的列，缺省 = 除角色列外全部 |
| `target` | 目标列，在 overview 中给出坏率 |
| `group_col` | 分组列（如申请月份），启用趋势表 |
| `special_values` | 特殊值哨兵（如 `[-999, -1]`），与缺失分开统计 |
| `high_missing_thr` | 缺失率超过此值打 `high_missing` 旗标 |

返回 [`DataProfileReport`](report-objects.md#dataprofilereport)。

---

## profile_risk

```python
profile_risk(
    df,
    *,
    target: str,
    features: list[str] | None = None,     # 缺省 = 全部数值列
    group_col: str | None = None,
    method: str = "quantile",              # quantile / uniform / cart
    n_bins: int = 5,
    missing_values: list | None = None,
    special_values: list | None = None,
    psi_include_missing: bool = False,
    psi_include_special: bool = False,
    monotonic: bool = False,               # 仅 cart 生效
    binner: WoeBinner | None = None,       # 复用已有分箱规则
) -> RiskProfile
```

| 参数 | 说明 |
|------|------|
| `target` | 二分类目标列（1 = 坏） |
| `group_col` | 分组列；首组为 PSI 基准，启用 trend_tables |
| `method` / `n_bins` | 分箱策略（传入 `binner` 时忽略） |
| `missing_values` / `special_values` | 独立成箱的哨兵值 |
| `psi_include_missing` / `psi_include_special` | PSI 是否计入缺失/特殊箱 |
| `binner` | 预拟合的 `WoeBinner`；未拟合则用传入数据拟合 |

返回 [`RiskProfile`](report-objects.md#riskprofile)：
`.report`（summary/detail/trends）、`.binner`（可复用规则）、
`.features`、`.metadata`。

### 典型调用

```python
# 开发期：全量评估
profile = profile_risk(df, target="target", group_col="month",
                       missing_values=[-999], special_values=[-999])

# 复用规则评估新样本
profile_oos = profile_risk(df_oos, target="target",
                           binner=profile.binner)
```
