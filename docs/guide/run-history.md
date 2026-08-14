# 运行历史

跑第二次实验，第一次的结果就没了——这是 3.1.0 之前的状态。现在每次运行都会
自动归档，可以随时回看和对比。

```bash
python -m auto_modeling_tool.main --config config.yaml
```

跑完日志里多一行：

```
📁 Run archived to runs/20260815-004143-logistic-f2b06f
```

## 归档了什么

```
runs/20260815-004143-logistic-f2b06f/
├── run.json              # 轻量索引：指标、配置、入模特征
├── config.yaml           # 最终生效的配置（YAML + CLI 覆盖合并后）
├── scoring_artifact.pkl
├── pipeline.pkl
└── Model_Report_1.xlsx
```

`output/` 目录**行为完全不变**，仍然是最新一次的产物。所有既有用法
（`--model output/scoring_artifact.pkl` 之类）不受影响，归档是额外多存一份。

!!! tip "config.yaml 存的是最终配置"
    包含 CLI 覆盖合并后的真实取值。三个月后想复现某个模型，直接
    `--config runs/<run_id>/config.yaml` 就行，不用回忆当时敲了什么参数。

## run id 怎么读

```
20260815-004143-logistic-f2b06f
└─ 时间戳 ──┘ └─ 算法 ─┘ └ 短哈希
```

时间戳保证按时间排序，算法名便于肉眼识别，短哈希让同一秒内的多次运行不冲突。

## 看历史

```bash
python -m auto_modeling_tool.runs list
```

```
RUN ID                           CREATED              TASK            MODEL     AUC_ROC  KS_STATISTIC  GINI    SCORE_PSI
-------------------------------  -------------------  --------------  --------  -------  ------------  ------  ---------
20260815-004159-lightgbm-b9c062  2026-08-15 00:41:59  classification  lightgbm  0.6150   0.1937        0.2301  0.0032
20260815-004143-logistic-f2b06f  2026-08-15 00:41:43  classification  logistic  0.6534   0.2451        0.3068  0.0075
```

加 `--json` 输出结构化数据，方便接自己的看板。

## 对比两次运行

这是真正有价值的部分——**指标、配置、入模特征三个维度一起看**：

```bash
python -m auto_modeling_tool.runs compare 20260815-004143 20260815-004159
```

```
20260815-004143-logistic-f2b06f  ->  20260815-004159-lightgbm-b9c062

Metrics
  auc_roc              0.6534 ->     0.6150  (-0.0384)
  ks_statistic         0.2451 ->     0.1937  (-0.0514)
  score_psi            0.0075 ->     0.0032  (-0.0043)

Config changes
  model_type               'logistic' -> 'lightgbm'
  n_features               8 -> 2

Features  (2 shared)
  removed: multi_loan_bin, util_bin
```

run id 可以只写前缀，只要不产生歧义。也可以直接传目录路径。

Config changes 只显示**真正不同**的键，输出目录这类每次必然不同的路径会被
自动忽略，不构成噪音。

## 看单次运行的全部细节

```bash
python -m auto_modeling_tool.runs show 20260815-004143
```

输出 JSON，含完整配置、全部指标和入模特征列表。

## 配置

```yaml
output:
  dir: output
  runs_dir: runs        # 归档目录，相对配置文件解析
  archive_run: true     # 设 false 可关闭
```

命令行等价开关：

| 参数 | 说明 |
|------|------|
| `--runs-dir` | 指定归档目录 |
| `--no-archive-run` | 本次不归档，只写 `output/` |

## 归档失败不会弄丢模型

跑了二十分钟的模型不该因为磁盘满或目录只读而白跑。归档出问题时会打一条
warning 然后正常返回，`output/` 里的产物一切照旧。

## Python 里用

```python
from auto_modeling_tool.runs import list_runs, resolve_run, compare_runs

for record in list_runs("runs"):
    print(record.run_id, record.headline())

left = resolve_run("20260815-004143")
right = resolve_run("20260815-004159")
diff = compare_runs(left, right)
diff["metrics"]["auc_roc"]["delta"]
diff["config_changes"]
```

`resolve_run` 接受完整 id、唯一前缀或目录路径；前缀有歧义会明确报错而不是
随便挑一个。
