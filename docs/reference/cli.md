# CLI 与脚本参考

## `python -m auto_modeling_tool.main`

主流水线入口。也可以用安装后的 `automodel` 命令。

### 输入输出

| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置路径，见[配置驱动主流程](../guide/config-pipeline.md) |
| `--input` / `-i` | 输入数据（CSV / XLSX / XLS / Parquet / PKL） |
| `--target` / `-t` | 目标列名 |
| `--output` / `-o` | 输出目录 |
| `--encoding` | 读 CSV 的编码，默认 `utf-8` |
| `--exclude-column` | 排除列，可重复传 |

### 任务与模型

| 参数 | 可选值 |
|------|--------|
| `--target-mode` | `classification` \| `regression` |
| `--model` | `logistic` \| `tree` \| `random_forest` \| `xgboost` \| `lightgbm` \| `catboost`（回归另支持 `linear`） |
| `--target-transform` | `log1p`（仅回归） |
| `--early-stopping-eval` | `none` \| `dev_holdout` \| `oot` |
| `--early-stopping-rounds` | 整数 |
| `--early-stopping-metric` | 如 `logloss` |
| `--seed` | 随机种子 |

### Dev / OOT 切分

| 参数 | 说明 |
|------|------|
| `--sample-column` | 标记 Dev/OOT 的列 |
| `--dev-label` / `--oot-label` | 该列里代表 Dev / OOT 的值 |
| `--date-column` | 按时间切分用的日期列 |
| `--oot-start` | OOT 起始日期/值（含当天） |
| `--test-size` | 前面都没给时，随机切出的 OOT 比例，默认 0.2 |

### 预处理与分箱

| 参数 | 可选值 |
|------|--------|
| `--clean-strategy` | 缺失填充策略，默认 `median` |
| `--normalize-method` | `minmax` \| `zscore` \| `robust` |
| `--n-bins` | 分箱数 |
| `--method` | `quantile` \| `uniform` \| `cart` |
| `--min-samples-bin` | 每箱最小样本数 |
| `--smoothing` | WOE 平滑系数 |
| `--monotonic` | 开启数值变量单调校准 |

### 筛选与权重

| 参数 | 说明 |
|------|------|
| `--selection` | `iv` \| `correlation` \| `rfe` \| `variance` |
| `--n-features` | 保留特征数 |
| `--use-sample-weight` | 开启样本权重 |
| `--weight-column` | 权重列名 |

### 评估与评分

| 参数 | 说明 |
|------|------|
| `--segment-column` | 分群对比列，可重复传 |
| `--temporal-column` | 跨期稳定性列 |
| `--benchmark-column` | 基准分对比列，可重复传 |
| `--convert-to-credit-score` | 输出信用分列 |
| `--credit-score-col` | 信用分列名，默认 `pred_score` |
| `--base-score` / `--pdo` | 基准分与翻倍分数，默认 500 / 50 |
| `--min-score` / `--max-score` | 信用分上下限，默认 300 / 900 |

---

## `scripts/score_new_samples.py`

拿 artifact 给新数据打分。

```bash
python scripts/score_new_samples.py \
  --model output/scoring_artifact.pkl \
  --input new.csv --output scored.csv
```

| 参数 | 说明 |
|------|------|
| `--model` | **必填**，artifact 文件、`pipeline.pkl` 或输出目录 |
| `--input` | **必填**，原始字段数据 |
| `--output` | **必填**，CSV 或 Parquet 输出路径 |
| `--convert-to-credit-score` | 追加信用分列（仅分类） |
| `--credit-score-col` / `--base-score` / `--pdo` / `--min-score` / `--max-score` | 信用分参数 |

输出列：分类 `pred_proba`（+ 兼容别名 `prediction`），回归 `pred_value`。

---

## `scripts/evaluate_model_performance.py`

对已经打好分的文件算指标。

```bash
python scripts/evaluate_model_performance.py \
  --input scored.csv --target target --score-column pred_proba
```

| 参数 | 说明 |
|------|------|
| `--input` / `--target` | **必填**，打分文件与真实标签列 |
| `--score-column` | 分数列，默认 `prediction`（打分脚本的主列是 `pred_proba`，建议显式传） |
| `--task` | `classification`（默认）\| `regression` |
| `--threshold` | 分类阈值，默认 0.5 |
| `--sample-column` / `--dev-label` / `--oot-label` | 分 Dev/OOT 出指标 |
| `--weight-column` / `--use-sample-weight` | 加权指标 |
| `--encoding` | 默认 `utf-8` |
| `--output` | 结果 JSON，默认 `performance_report.json` |

---

## `scripts/explore_dataset.py`

独立 EDA，不依赖训练流程。

```bash
python scripts/explore_dataset.py \
  --input data.csv --target target --sample-column sample --output eda.xlsx
```

| 参数 | 说明 |
|------|------|
| `--input` | **必填** |
| `--target` | 有则算 IV / PSI |
| `--output` | 默认 `eda_report.json`，给 `.xlsx` 则出 Excel |
| `--sample-column`（别名 `--split-column`）/ `--dev-label` / `--oot-label` | Dev/OOT 拆分 |
| `--weight-column` / `--use-sample-weight` | 加权统计 |
| `--exclude-column` | 排除列，可重复传 |
| `--n-bins` | 分箱数，默认 20 |
| `--export-woe-detail` | 额外导出 WOE 明细 |
| `--encoding` | 默认 `utf-8` |

---

## `scripts/validate_release.py`

发布门禁，详见[发布门禁](../guide/release-gate.md)。

| 参数 | 说明 |
|------|------|
| `--model-dir` | **必填**，输出目录或 artifact 路径 |
| `--report` | 报告路径，不传则自动找最新 `Model_Report_*.xlsx` |
| `--min-auc` / `--max-psi` | 可选阈值门禁 |
| `--json` | 把逐项结论写成 JSON |

失败时返回非零退出码。
