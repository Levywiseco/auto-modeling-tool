# 配置驱动主流程

2.2.0 起，常规建模不需要写 Python：改一份 YAML，跑一条命令。

```bash
python -m auto_modeling_tool.main --config configs/pipeline_config.yaml
```

配置里的相对路径按**配置文件所在目录**解析，不是按当前工作目录 ——
把配置和数据放一起，在任何目录下跑结果都一样。

## 一份完整配置

```yaml
shared:
  target_mode: classification   # classification | regression
  bad_col: target
  sample_col: sample            # 标记 Dev/OOT 的列
  dev_label: dev
  oot_label: oot
  random_state: 42
  use_sample_weight: false
  weight_col: weight
  target_transform: null        # 回归可用 log1p

data:
  path: data/dataset.csv        # 支持 CSV / XLSX / XLS / Parquet / PKL
  encoding: utf-8

output:
  dir: output

preprocess:
  clean_strategy: median        # 缺失填充策略
  normalize_method: zscore      # minmax | zscore | robust

binning:
  n_bins: 10
  method: quantile              # quantile | uniform | cart
  min_samples_bin: 50
  smoothing: 0.5                # WOE 平滑，防稀有类别爆炸
  monotonic: false              # 数值变量单调校准

feature_screening:
  method: iv                    # iv | correlation | rfe | variance
  n_features: 20

modeling:
  algorithm: logistic           # logistic | tree | random_forest
                                # | xgboost | lightgbm | catboost
  model_params: {}
  early_stopping_eval: none     # none | dev_holdout | oot
  early_stopping_rounds: 30
  early_stopping_metric: logloss

evaluation:
  segment_cols: []              # 分群对比，如 [channel, city]
  temporal_col: null            # 跨期稳定性，如 apply_month
  benchmark_cols: []            # 与外部评分对比，如 [bureau_score]
  export_excel: true

scoring:
  convert_to_credit_score: false
  credit_score_col: pred_score
  base_score: 500
  pdo: 50                       # Points to Double the Odds
  min_score: 300
  max_score: 900
```

## Dev/OOT 怎么切

三种方式，按优先级：

| 方式 | 配置 | 适用 |
|------|------|------|
| 样本标签 | `shared.sample_col` + `dev_label` / `oot_label` | 上游已经标好 |
| 日期边界 | `sample_split.date_col` + `oot_start` | 按时间切，`oot_start` 当天算 OOT |
| 随机兜底 | `--test-size`（默认 0.2） | 前两者都没给 |

!!! warning "这是防泄漏的关键"
    预处理、WOE 分箱、特征筛选**只在 Dev 上 fit**，再 transform 到 OOT。
    2.2.0 之前是在全量上 fit 的，OOT 指标会偏乐观。升级后同一份数据的
    OOT AUC 通常会低一些 —— 那才是真实水平。

## CLI 覆盖

每个配置项都能用命令行临时盖掉，适合做对比实验而不改文件：

```bash
python -m auto_modeling_tool.main --config configs/pipeline_config.yaml \
  --model xgboost --early-stopping-eval oot --n-features 30
```

常用覆盖参数：

| 参数 | 覆盖的配置 |
|------|-----------|
| `--input` / `--target` / `--output` | `data.path` / `shared.bad_col` / `output.dir` |
| `--target-mode` | `shared.target_mode` |
| `--model` | `modeling.algorithm` |
| `--early-stopping-eval` / `--early-stopping-rounds` | `modeling.*` |
| `--use-sample-weight` / `--weight-column` | `shared.use_sample_weight` / `weight_col` |
| `--sample-column` / `--date-column` / `--oot-start` | Dev/OOT 切分 |
| `--segment-column` / `--temporal-column` / `--benchmark-column` | `evaluation.*`（可重复传） |
| `--convert-to-credit-score` / `--base-score` / `--pdo` | `scoring.*` |

完整清单见 [CLI 与脚本参考](../reference/cli.md)。

## 不用配置文件也行

老的纯 CLI 用法完全保留：

```bash
python -m auto_modeling_tool.main --input data.csv --target target --output output
```

## 输出物

```
output/
├── scoring_artifact.pkl     # 独立打分用，含全部拟合好的变换
├── pipeline.pkl             # 完整流水线对象
└── Model_Report_1.xlsx      # 多页审计报告
```

- `scoring_artifact.pkl` → [独立打分与 Artifact](scoring-artifact.md)
- `Model_Report_N.xlsx` → [报告与导出](reports.md)
- 上线前的检查 → [发布门禁](release-gate.md)

## 样本权重

`use_sample_weight: true` 之后，权重会贯穿**整条链路**，不是只在训练时用：

分箱统计 → IV → 特征筛选 → 模型训练 → 评估指标 → PSI → 监控趋势。

回归任务的训练、OOT early stopping 和评估同样支持。拒绝推断、抽样回补
这类场景下，这一条决定了指标是不是可信。
