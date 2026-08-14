# 发布门禁

模型要上线之前，先让机器检查一遍：**Artifact 是不是完整的、报告是不是齐的、
指标是不是达标的**。不通过就别推。

```bash
python scripts/validate_release.py --model-dir output
```

```
Release gate: artifact contract -> driver/feature contract
           -> report sheet contract -> optional AUC/PSI thresholds
           -> merge/release decision
```

## 检查项

| 检查 | 通过条件 |
|------|---------|
| `artifact_loadable` | `scoring_artifact.pkl` 能被反序列化，且是 dict |
| `artifact_contract` | 必备 key 齐全：`artifact_version`、`task`、`feature_columns`、`model`、`preprocessor`；分类任务再加 `binner`、`selector`、`woe_feature_columns`、`selected_features` |
| `feature_contract` | 入模字段列表和筛选后特征列表都非空 |
| `report_contract` | 报告里有 `Overview_Performance` 和 `Artifact_Metadata` 两页 |
| `min_auc` | 传了 `--min-auc` 才检查，比对 `metadata.metrics.auc_roc` |
| `max_psi` | 传了 `--max-psi` 才检查，比对 `score_psi`（回落到 `psi`） |

前四项永远跑，后两项是可选的阈值门禁。任何一项失败，整体判定失败。

## 加上指标阈值

```bash
python scripts/validate_release.py \
  --model-dir output \
  --min-auc 0.70 --max-psi 0.10 \
  --json output/release_check.json
```

`--json` 会把逐项结论落成文件，方便挂到 CI 或审批流里当证据。

不传 `--report` 时，脚本会自己在 `--model-dir` 里找最新的
`Model_Report_*.xlsx`。

## Python 里用

```python
from auto_modeling_tool.evaluation.quality_gate import validate_release

result = validate_release(
    "output",
    min_auc=0.70,
    max_psi=0.10,
)

result.passed                      # True / False
for check in result.checks:
    print(check.name, check.passed, check.detail)

result.as_dict()                   # 可直接 json.dumps
```

`source` 可以传输出目录、artifact 文件路径，或者已经加载好的 artifact dict。

## 挂进 CI

```yaml
- name: Release gate
  run: |
    python -m auto_modeling_tool.main --config configs/pipeline_config.yaml
    python scripts/validate_release.py --model-dir output \
      --min-auc 0.70 --max-psi 0.10 --json output/release_check.json
```

脚本失败时返回非零退出码，CI 会直接红。

## 门禁挡不住的事

这一层只验证**契约和阈值**，不替你判断模型好不好：

- 特征逻辑是否合理、有没有用到穿越变量 —— 要人看
- 分箱是否单调、业务上讲不讲得通 —— 看报告的分箱页
- 样本是否有代表性 —— 看 `Segment_Summary` 和 `Temporal_Stability`

门禁的价值是**挡住低级事故**：artifact 少存了一个 selector、报告没生成、
PSI 已经飘到 0.3 还想上线。
