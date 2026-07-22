# 特征筛选

`FeatureSelector` 提供多种筛选方法，sklearn 风格接口。

## 基本用法

```python
from src.features import FeatureSelector

selector = FeatureSelector(method="iv", iv_threshold=0.02)
df_selected = selector.fit_transform(df_woe, target_col="target")
selector.get_selected_features()
```

## 方法一览

| method | 原理 | 适用 |
|--------|------|------|
| `iv` | 按信息值阈值筛选 | 风控标配，先跑这个 |
| `correlation` | 剔除高相关特征对中 IV 较低者 | 去共线性 |
| `variance` | 剔除低方差特征 | 快速粗筛 |
| `rfe` | 递归特征消除 | 配合具体模型精筛 |
| `mutual_info` | 互信息 | 捕捉非线性关系 |

## 推荐流程

```python
# 1. profile_risk 先看全景（IV + PSI 稳定性）
profile = profile_risk(df, target="target", group_col="month")
stable = profile.summary_table.filter(
    (pl.col("psi_max") < 0.1) & (pl.col("iv") > 0.02)
)["feature"].to_list()

# 2. 在稳定特征里去共线性
from src.features import remove_multicollinearity
final_feats = remove_multicollinearity(df.select(stable), threshold=0.8)
```

先稳定性后区分度：一个 IV 高但按月漂移的特征，上线三个月就会反噬。
