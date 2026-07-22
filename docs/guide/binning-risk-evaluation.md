# 分箱与风险评估

`profile_risk` 是核心工作流：一次调用完成
**分箱 → WOE/IV → KS → 跨期 PSI → 汇总排序**，
返回 `RiskProfile`（结构化报告 + 可复用分箱器）。

## 基本用法

```python
from src import profile_risk

profile = profile_risk(
    df,
    target="target",
    features=["income", "utilization"],  # 缺省 = 全部数值列
    group_col="month",                   # 首月为 PSI 基准
    method="quantile",                   # quantile / uniform / cart
    n_bins=5,
    missing_values=[-999],
    special_values=[-999],
)
```

## summary_table 怎么读

| 列 | 含义 | 经验参考 |
|----|------|----------|
| `iv` | 信息值 | <0.02 无效 · 0.02-0.1 弱 · 0.1-0.3 中 · 0.3-0.5 强 · >0.5 疑似穿越 |
| `iv_strength` | IV 等级标签 | `suspicious` 的特征先查泄漏 |
| `ks` | 分箱级 KS | 单特征 >0.3 已属很强 |
| `missing_rate` | 缺失率 | 过高的特征谨慎入模 |
| `psi_max` | 各期 vs 首期 PSI 最大值 | >0.1 关注，>0.25 慎用 |
| `monotonic_woe` | WOE 是否单调 | 评分卡通常要求单调 |

**选特征的次序建议：先看 psi_max 剔除不稳定的，再按 iv 从高到低选。**

## detail_table — 分箱明细

每行一个 (feature, bin)：`bin_label`、`count`、`bad_rate`、`woe`、`iv` 贡献。
缺失箱 `bin_idx = -1`，特殊值箱 `bin_idx ≤ -3`。

## 跨期趋势

```python
profile.report.trend_tables["psi"]           # 特征 × 月份 PSI
profile.report.trend_tables["missing_rate"]  # 特征 × 月份缺失率
profile.report.trend_tables["bad_rate"]      # 每月样本量与坏率
```

## 复用分箱规则

`profile.binner` 是拟合好的 `WoeBinner`，两种典型复用：

```python
# 1. 给新数据打分（WOE 变换）
X_woe = profile.binner.transform(new_df, return_type="woe")

# 2. 传给监控，保证开发期与监控期口径一致
from src import Monitor
Monitor().monitor(prod_df, features=profile.features,
                  binner=profile.binner, group_col="score_month")
```

## 三种分箱方法

| method | 策略 | 适用 |
|--------|------|------|
| `quantile` | 等频 | 默认；对偏态分布稳健 |
| `uniform` | 等距 | 分布均匀、需要可解释边界时 |
| `cart` | 决策树最优分箱 | 追求区分度；配合 `monotonic=True` 保证单调 |
