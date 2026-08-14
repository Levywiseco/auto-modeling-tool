# 独立打分与 Artifact

训练完的模型要能在另一台机器、另一个进程里，**拿原始字段**直接打分 ——
不需要把预处理、分箱、筛选的代码再抄一遍。这就是 `scoring_artifact.pkl`。

## Artifact 里装了什么

流水线跑完会在 `output/` 落一个 `scoring_artifact.pkl`，它把打分需要的
每一环都序列化在一起：

| Key | 内容 |
|-----|------|
| `artifact_version` | Artifact 格式版本（当前 `1.0`） |
| `task` | `classification` 或 `regression` |
| `feature_columns` | **原始入模字段**契约（打分数据必须包含这些列） |
| `preprocessor` | 拟合好的预处理器（只在 Dev 上 fit） |
| `binner` / `selector` | WOE 分箱规则与筛选器（分类任务） |
| `woe_feature_columns` / `selected_features` | 中间列名契约 |
| `model` | 训练好的模型 |
| `target_transform` | 回归的目标变换（如 `log1p`，打分时自动逆变换） |
| `metadata` | 复现信息与训练期指标 |

## 命令行打分

```bash
python scripts/score_new_samples.py \
  --model output/scoring_artifact.pkl \
  --input new_applications.csv \
  --output output/new_scores.csv
```

`--model` 也可以直接给输出目录或 `pipeline.pkl`，脚本会自己找。

输出列：

| 任务 | 列 | 含义 |
|------|-----|------|
| 分类 | `pred_proba` | 正类概率（**主列**） |
| 分类 | `prediction` | 兼容别名，与 `pred_proba` 同值 |
| 分类 | `pred_score` | 信用分，加 `--convert-to-credit-score` 才有 |
| 回归 | `pred_value` | 连续预测值（已做逆变换） |

!!! note "为什么有两个概率列"
    2.2.0 之前输出列叫 `prediction`。改名为 `pred_proba` 是为了和回归的
    `pred_value` 区分开，`prediction` 保留为别名，老的下游脚本不用改。

## 转成信用分

```bash
python scripts/score_new_samples.py \
  --model output/scoring_artifact.pkl \
  --input new_applications.csv \
  --output output/new_scores.csv \
  --convert-to-credit-score --base-score 600 --pdo 40 \
  --min-score 300 --max-score 950
```

用的是标准的 base score / PDO（Points to Double the Odds）换算，结果按
`min_score` / `max_score` 截断。也可以在配置里开 `scoring.convert_to_credit_score`，
让主流程直接输出这一列。

## Python 里用

```python
import polars as pl
from auto_modeling_tool.modeling.artifact import load_scoring_artifact, score_with_artifact

artifact = load_scoring_artifact("output/scoring_artifact.pkl")
df = pl.read_csv("new_applications.csv")

proba = score_with_artifact(artifact, df, return_proba=True)   # 概率
label = score_with_artifact(artifact, df)                      # 0/1
```

`score_with_artifact` 会先校验列契约：缺了 `feature_columns` 里的字段会
直接抛 `ValidationError`，而不是算出一个错的分数。

回归 artifact 不支持 `return_proba=True`，会抛异常。

## 自己构造 Artifact

如果你的流程不走 `auto_pipeline`，也可以手工组装：

```python
from auto_modeling_tool.modeling.artifact import build_scoring_artifact, save_scoring_artifact

artifact = build_scoring_artifact(
    target_col="target",
    feature_columns=raw_features,
    woe_feature_columns=woe_cols,
    selected_features=selected,
    preprocessor=preprocessor,
    binner=binner,
    selector=selector,
    model=model,
    metadata={"metrics": {"auc_roc": 0.78}},
)
save_scoring_artifact(artifact, "output/scoring_artifact.pkl")
```

回归用 `build_regression_artifact`（不需要 binner / selector，多一个
`target_transform`）。

组装完建议跑一遍 [发布门禁](release-gate.md) 验证契约完整。
