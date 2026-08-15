# Release Notes

## 3.1.3（2026-08-15）

对抗性审查的第二批。其中 PSI 那条使得特征漂移监控对几乎所有真实特征都报"稳定"。

### 修复

- **基准里只要有一个缺失值，PSI 就恒等于 0.0。** 分位数边界直接对原始数组取
  `np.quantile`，一个 NaN 会让所有边界变成 NaN、所有分箱判定为空，两个population
  同时塌缩到相同的 epsilon 占比。实测：一对 PSI 为 10.39 的分布，把基准中**一个**
  值设成 NaN 后 PSI 变成 0.0。信贷特征几乎都有缺失，因此
  `calculate_feature_psi` 对绝大多数特征都报 "Stable"，而发布门禁的 `--max-psi`
  读的正是这个数。现在边界只用有限值计算；整列缺失会明确报错。

- **`clean_strategy` 对 mean/median 以外的值静默填 0。** CLI 提供了 `forward` 和
  `backward`，但有状态的预处理器两个都没实现，一路落到填字面量 0——对收入是最差
  情形，对 DPD 是最好情形，且训练与保存的 artifact 中行为一致地错。依赖行序的
  填充在"拟合一次、逐行打分"的变换里本就无法成立，现在会明确报错并从 CLI 选项中
  移除；`zero` 从兜底分支变成显式选项。

- **`--selection rfe` 会崩溃。** CLI 和两份指南都宣传 `rfe`，而实现只认
  `recursive`。现在 `rfe` 被接受为等价写法，CLI 还补上了本来能用却被挡住的
  `recursive` 与 `mutual_info`。

- **运行归档里混入了别的运行的报告。** `archive_run` 复制了输出目录下所有
  `Model_Report_*.xlsx`，于是第三次运行的归档里躺着报告 1、2、3，并且会无限膨胀。
  现在只归档本次运行刚写出的那一份。

- **归档的运行无法用 `--config` 重放。** 快照存的是扁平的流水线参数，而加载器只
  认嵌套 schema，于是 `--config runs/<id>/config.yaml` 直接报 "Config must define
  data.path"——与该功能自己文档里的承诺相矛盾。加载器现在接受归档格式，重放能精确
  复现原始指标。

- **文档里的示例会报错。** README 与特征筛选指南都写着
  `fit_transform(df, target_col=...)`，而这不是真实签名；指南还从一个从未导出过的
  路径导入 `remove_multicollinearity`，并把返回值描述成特征列表（实际是
  `(DataFrame, 被剔除列)`）。所有示例现在都由 `tests/test_documented_api.py`
  真实执行。

## 3.1.2（2026-08-15）

对抗性审查发现的两个缺陷，**都会让模型看起来比实际更好**。如果你已经用更早的
版本训过模型，建议重训并对比。

### 修复

- **评估用的辅助列被当成模型特征训练了。** `segment_cols`、`temporal_col`、
  `benchmark_cols` 没有被任何机制排除——它们留在特征池里，一路进了分箱和 IV
  筛选，只要有信号就会被选进模型。

    用时间列做特征是时间泄漏：模型学到"3 月坏率高"，上线遇到没见过的月份就
    毫无意义。用基准分做特征则直接毁掉了它本来的用途——对比变成了自己跟自己
    比，而且让一个外部评分变成生产打分的硬依赖。

    实测：在两个辅助列都带信号的样例上，排除它们之后 **OOT AUC 从 0.9987 掉到
    0.5657**——后者才是这个模型真实的区分能力。

    现在它们和目标列、样本标记列一样属于角色列。跨期稳定性、基准对比、分群
    汇总三张报告页不受影响，打分契约也不再要求模型根本不用的列。

- **KS 被并列分数抬高了。** `calculate_ks` 在每一行上取累积 TPR/FPR 差的最大值，
  而不是只在阈值真正可能落下的位置取，于是同分样本被当成可区分的。极端情况下，
  一个全部同分（零区分度）的模型，只要输入恰好按标签排序，就会报出 KS 0.924。

    连续概率不受影响，这也是它一直没暴露的原因——但它恰好咬在信贷最常见的地方：
    评分卡的整数分、粗粒度分箱分、规则引擎输出的少数几档。现在 KS 只在每个同分组
    的末尾读取，在所有测试场景（含加权）下都与 sklearn 的 ROC 口径完全一致。

## 3.1.1（2026-08-15）

### 修复

- **运行历史会静默丢失运行。** `new_run_id` 的哈希只取了秒级时间戳、模型名，
  以及一个由 `archive_run` 填入 output 目录的 seed——而 output 目录在一次参数
  sweep 中是恒定的。于是同一秒内启动的每次运行都得到完全相同的 id，
  `archive_run` 又用 `mkdir(exist_ok=True)` 加覆盖式复制写进已存在的目录。
  没有报错，没有警告。实测：跑 6 次，只归档 1 次。

    这正是该功能自己的旗舰用法——扫一个参数再对比归档结果——而它恰恰是会
    毁掉数据的那条路径。

    现在 id 后缀改用每次调用的随机熵，`archive_run` 遇到已存在目录会加后缀而
    不是覆盖。id 与 `created_at` 也改为同一次时钟读数，不会再跨秒对不上。

    `test_same_second_runs_do_not_collide` 一直是绿的，因为它手工喂了两个不同
    的 seed，而真实调用方从不这么做。现在它断言"相同输入生成的 50 个 id 互不
    相同"，并新增一个测试：在固定时间戳下归档 5 次，必须能取回 5 条。

## 3.1.0（2026-08-15）

### 新增

- **运行历史。** 每次运行自动归档到 `runs/<run_id>/`，含最终生效配置、
  达成的指标与可部署 artifact，模型在很久之后仍可追溯。`output/` 行为不变，
  仍是最新一次产物，既有路径全部照常工作。
  - `python -m auto_modeling_tool.runs list` —— 历史列表，最新在前
  - `python -m auto_modeling_tool.runs compare <a> <b>` —— 指标增减、配置差异、
    入模特征进出，三个维度并排看
  - `python -m auto_modeling_tool.runs show <id>` —— 单次运行全部细节
  - run id 形如 `<时间戳>-<算法>-<哈希>`：可排序、可读、同秒不冲突，
    支持唯一前缀检索
  - 配置项 `output.runs_dir` / `output.archive_run`；命令行 `--runs-dir` /
    `--no-archive-run`
  - 归档失败只记 warning 不中断——跑了二十分钟的模型不会因为磁盘满而白跑
  - 详见[运行历史](../guide/run-history.md)

### 修复

- **`n_features` 在默认筛选方法下完全无效。** `_select_by_iv` 从未收到这个
  参数，IV 筛选只受 `iv_threshold` 支配，配 `n_features: 8` 和 `n_features: 3`
  得到的入模变量一模一样。现在它会对 IV 排序后的列表做截断，与其他筛选方法
  行为一致。**这是对比两次归档运行时发现的——运行历史交付的第一个战果。**

    !!! warning "IV 路径的结果会变"
        此前"凡是 IV 超过阈值就全留"的运行，现在最多保留 `n_features` 个
        （默认 20），按 IV 从高到低取。

## 3.0.1（2026-08-14）

### 修复

- Dev/OOT 分数 PSI 只进了报告表格，没有写入 `metrics_`，因而没能进入
  `scoring_artifact.pkl`。发布门禁的 `--max-psi` 读到的永远是 `psi=None` 并
  判定失败——这个阈值实际上是失效的。现在 PSI 与其他指标一样落进 artifact，
  流水线契约测试会断言它不丢。

## 3.0.0（2026-08-14）

打包规范化与代码质量修复。**唯一的破坏性变更是导入名。**

### 破坏性变更

- **包名 `src` → `auto_modeling_tool`**。2.x 安装后会往 site-packages 写入
  五个通用顶层名（`src` / `tests` / `scripts` / `examples` / `build`），任何
  同样打包的项目都会与之冲突。现在 `packages.find` 限定为
  `auto_modeling_tool*`，全新安装只贡献一个顶层名。

  迁移是查找替换，API、参数、输出零变化：

  ```python
  from src import profile_risk                    # 2.x
  from auto_modeling_tool import profile_risk     # 3.0
  ```

  ```bash
  python -m src.main --config ...                 # 2.x
  python -m auto_modeling_tool.main --config ...  # 3.0
  ```

  `automodel` 命令不受影响。

### 修复

- **`ProbabilityCalibrator` 从来就跑不起来。** 它自 v2.0 就作为公开 API 导出，
  但覆盖率仅 20%，没有任何测试真正调用过。两个独立缺陷叠加：内部
  `DummyClassifier` 的基类顺序写成 `(BaseEstimator, ClassifierMixin)`，
  sklearn 按 MRO 解析 tags，`is_classifier()` 因而判定成回归器；修好后暴露
  更深的问题——该 dummy 的 `predict_proba` 用闭包捕获了完整概率向量、无视
  传入的 `X`，导致交叉验证每折都长度不匹配。根因是用错工具：
  `CalibratedClassifierCV` 用于包装分类器，无法消费已算好的分数。改用标准
  做法（Platt 用 `LogisticRegression` 拟合原始分，isotonic 用
  `IsotonicRegression`）。600 行过度自信分数上 Brier 0.2294 →
  0.1781（sigmoid）/ 0.1669（isotonic）。
- `ProbabilityCalibrator` 现在接受 `pl.Series` 作为 `y_prob`，长度不一致会
  给出清晰报错。
- 移除 `woe_binning` 中一处死变量赋值。

### 新增

- **LICENSE** —— README 与 `pyproject` 都声明 MIT，但文件从未提交，链接是死的。
- `tests/test_standalone_tools.py` —— 覆盖 `calibration` / `cross_validation` /
  `tuning` 这三个"导出为公开 API 但流水线不调用"的模块。校准器正是因为
  没有测试才带病发布。

### 变更

- `ruff check` 归零（原 1511 errors）。约 1140 处空白/导入排序/f-string，
  364 处 typing 现代化（`List[str]` → `list[str]`；PEP 585 在声明下界 3.9
  上可用），81 处未使用导入。无行为变更。

### 已知缺口

- `calibration` / `tuning` / `cross_validation` 仍只能直接 import 使用，
  `auto_pipeline` 不调用；`ModelTrainer` 另有一套私有调参实现。
- 无拒绝推断、无模型层单调约束、无冠军挑战者对比。

## 2.2.0（2026-08-14）

对齐 `USER_GUIDE 0.75` 的生产化加固：防泄漏的 Dev/OOT 建模、配置驱动、
可部署的打分 Artifact、多页审计报告与发布门禁。

### 新增

- **配置驱动主流程** — `python -m auto_modeling_tool.main --config configs/pipeline_config.yaml`
    - `auto_modeling_tool.config`：YAML schema、相对路径按配置文件解析、旧键别名兼容
    - 29 个新 CLI 参数，每项配置都能临时覆盖
    - 见[配置驱动主流程](../guide/config-pipeline.md)
- **打分 Artifact** — `auto_modeling_tool.modeling.artifact`
    - `scoring_artifact.pkl` 打包预处理器、分箱、筛选器与模型
    - `score_with_artifact` 拿原始字段直接打分，带列契约校验
    - 见[独立打分与 Artifact](../guide/scoring-artifact.md)
- **回归任务** — `auto_modeling_tool.pipelines.regression_pipeline`
    - RMSE / MAE / R²，可选 `log1p` 目标变换（打分自动逆变换）
    - 见[回归任务](../guide/regression.md)
- **发布门禁** — `auto_modeling_tool.evaluation.quality_gate`
    - Artifact 契约 → 字段契约 → 报告 sheet 契约 → 可选 AUC/PSI 阈值
    - 见[发布门禁](../guide/release-gate.md)
- **Excel 审计报告** — `auto_modeling_tool.reports.excel`，分箱 / WOE / IV / 变量审计 /
  筛选 / Dev-OOT 指标 / 分数分箱 / Score PSI / 稳定性 / 分群 / 跨期 / 基准
- **信用分转换** — 可配 `base_score` / `pdo` / 上下限
- **独立脚本** — `score_new_samples` / `evaluate_model_performance` /
  `explore_dataset` / `validate_release`，见 [CLI 与脚本](../reference/cli.md)
- **样本权重**贯穿分箱统计 → IV → 筛选 → 训练 → 评估 → PSI → 监控
- **模型选择** — logistic / tree / random forest / XGBoost / LightGBM /
  CatBoost，树模型支持 `dev_holdout` / `oot` 早停
- **数据格式** — CSV / XLSX / XLS / Parquet / PKL
- **CI** — GitHub Actions，Python 3.9 / 3.10 / 3.11 / 3.12，103 个测试

### 变更

- Dev/OOT 切分支持样本标签、日期边界与随机兜底；预处理、WOE 与特征筛选
  **只在 Dev 上 fit**，避免 OOT 信息泄漏
- WOE 支持类别变量、未见/稀有类别、平滑与数值变量单调校准
- Model Report 固定包含 `Overview_Performance` / `Report_Index` /
  `Artifact_Metadata`
- 监控支持数值与类别特征、分组趋势、加权 PSI
- Pandas 与 CatBoost 变为显式运行时依赖
- 日志在 Windows GBK 控制台下不再乱码

### 修复

- WOE 筛选结果算完了但没真正进入建模
- `ScorecardBuilder` 按 bin index 而非 bin 值取分
- 重复分数值导致 lift 表和报告生成失败
- `export_excel: false` 未被分类/回归流水线遵守
- 打分输出列统一为 `pred_proba`，保留 `prediction` 兼容别名

### 兼容性

- v2.1.0 的 9 个 CLI 参数全部保留
- `--test-size` 语义变为"没有样本列和日期边界时，随机切出的 OOT 比例"
- v2.1.0 的低层 API 不变

!!! warning "升级后 OOT 指标可能变差"
    2.2.0 之前预处理和分箱是在全量数据上 fit 的，OOT 指标偏乐观。
    修好泄漏之后，同一份数据的 OOT AUC 通常会低一些 —— 那才是真实水平。

## 2.1.0（2026-07-23）

以任务为中心的一次大版本迭代：新增统一工作流入口、结构化 Report
对象体系与监控告警模块。

### 新增

- **`auto_modeling_tool.analysis`** — 任务式入口
    - `profile_data(df, ...)`：数据质量画像（overview / dq / stats 三表 + 按期趋势）
    - `profile_risk(df, target=...)`：一次调用完成分箱 + IV/KS + 跨期 PSI，
      返回 `RiskProfile`（report + 可复用 binner）
- **`auto_modeling_tool.monitoring`** — 监控与告警
    - `Monitor`：PSI / 缺失率 / 坏率 / 分均值漂移监控，支持
      `group_col` 按期对比与 `benchmark_df` 对照开发样本两种基准模式
    - `generate_monitoring_alert` + `AlertConfig`：按优先级排序的中文告警文本
- **`auto_modeling_tool.reports`** — 结构化 Report 对象
    - `DataProfileReport` / `BinningReport` / `RiskProfile` / `MonitoringReport`
    - 统一 `summary_table` / `detail_table` / `trend_tables` / `metadata` 结构
    - `to_markdown()` / `save()` 导出
- **`auto_modeling_tool.evaluation.stability`** — 分箱分布 PSI 计算原语
  （`bin_distribution` / `psi_from_distributions` / `psi_level`）
- **文档站** — MkDocs Material，含 Quickstart、API 约定、使用指南与 Reference

### 变更

- 顶层包直接导出工作流 API：`from auto_modeling_tool import profile_risk, Monitor, ...`
- README 重写为"从任务开始"结构，新增稳定性分级说明

### 兼容性

- 既有低层 API（`WoeBinner` / `FeatureSelector` / `calculate_*`）完全不变
- 本版本无破坏性变更

## 2.0.0

- Polars 优先架构、WOE 分箱（quantile / uniform / cart）、
  特征筛选、模型训练与评估、自动化流水线
