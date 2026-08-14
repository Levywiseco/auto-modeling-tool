# 回归任务

2.2.0 起，同一套流水线也能做回归 —— 预测额度、预测损失金额、预测还款
天数这类连续目标，不再需要硬套二分类。

## 跑一个回归

配置里把 `target_mode` 改掉就行：

```yaml
shared:
  target_mode: regression
  bad_col: loss_amount        # 连续目标
  sample_col: sample
  target_transform: log1p     # 可选，长尾金额建议开

modeling:
  algorithm: xgboost          # 回归默认就是 xgboost
```

```bash
python -m auto_modeling_tool.main --config configs/pipeline_config.yaml
```

或者纯命令行：

```bash
python -m auto_modeling_tool.main --input data.csv --target loss_amount \
  --target-mode regression --model xgboost --target-transform log1p
```

## 和分类的差别

| | 分类 | 回归 |
|---|------|------|
| WOE 分箱 | ✅ 走完整链路 | ❌ 不做（没有好坏标签） |
| 特征筛选 | IV / 相关性 / RFE / 方差 | 不走 IV |
| 指标 | AUC / KS / PSI | **RMSE / MAE / R²** |
| 默认算法 | `logistic` | `xgboost` |
| 打分输出列 | `pred_proba` | `pred_value` |
| Artifact | preprocessor + binner + selector + model | preprocessor + model + target_transform |

支持的回归算法：`linear`（`linear_regression`）、`tree`、`random_forest`、
`xgboost`、`lightgbm`、`catboost`。

## log1p 目标变换

金额类目标通常是长尾的，直接回归会被大额样本拽着走。`target_transform: log1p`
在训练前做 `log1p`，打分时自动 `expm1` 逆变换回原始量纲 —— 你拿到的
`pred_value` 始终是可直接用的金额，不用自己记得反变换。

!!! warning "目标不能有负数"
    `log1p` 要求目标非负，有负值会直接抛 `ValidationError`。

## Python 里用

```python
from auto_modeling_tool.pipelines.regression_pipeline import RegressionPipeline

pipeline = RegressionPipeline(
    target_col="loss_amount",
    model_type="xgboost",
    sample_col="sample",
    target_transform="log1p",
    early_stopping_eval="oot",
    early_stopping_rounds=50,
)
pipeline.fit("data.csv")

metrics = pipeline.evaluate()        # {"rmse": ..., "mae": ..., "r2": ...}
pipeline.save("output")              # 落 pipeline.pkl + scoring_artifact.pkl

preds = pipeline.predict(new_df)     # 已逆变换的预测值
```

一步到位的函数式入口：

```python
from auto_modeling_tool.pipelines.regression_pipeline import run_regression_pipeline

result = run_regression_pipeline(
    "data.csv",
    target_col="loss_amount",
    output_dir="output",
    sample_col="sample",
    target_transform="log1p",
)
result["metrics"]
```

## 样本权重与 Early Stopping

两者回归同样支持：

- `use_sample_weight: true` + `weight_col` → 训练和 RMSE/MAE/R² 都加权
- `early_stopping_eval: oot` → 用 OOT 集做早停（`dev_holdout` 则从 Dev 里切）

## 打分与门禁

回归 artifact 和分类一样能独立打分、一样过发布门禁：

```bash
python scripts/score_new_samples.py --model output/scoring_artifact.pkl \
  --input new.csv --output scored.csv          # 输出 pred_value

python scripts/evaluate_model_performance.py --input scored.csv \
  --target loss_amount --score-column pred_value --task regression

python scripts/validate_release.py --model-dir output
```

回归 artifact 不支持 `return_proba=True`，调了会抛异常。
