# Changelog

## [3.3.1] - 2026-08-15

### Added

- **Tests for three behaviours mutation testing found unprotected.** Deliberately
  breaking each of these left all 190 tests green:
  - the pipeline fitting preprocessing on Dev+OOT instead of Dev alone — the
    leakage-safety the project is built around had no coverage at the pipeline
    level, only on `DataPreprocessor` in isolation
  - `ScorecardBuilder` hardcoding its factor and ignoring the configured `PDO`
  - `probability_to_credit_score` ignoring its `pdo` argument

  All three mutants are now killed. 193 tests.

## [3.3.0] - 2026-08-15

### Added

- **`clean_strategy: "keep"` — let missingness reach WOE.** `AutoPipeline` always
  imputes before binning, so `WoeBinner`'s Missing bin was unreachable for model
  features and the missing-vs-observed contrast was silently discarded. In credit
  work that contrast is frequently the strongest signal a feature carries: "no
  bureau record" is not a median applicant.

  Measured on a fixture where the 25% missing group defaults at 0.60 against 0.14
  for the rest: imputing gave IV 0.653 and no Missing bin; `keep` gave IV 1.020
  with the Missing bin at WOE 1.44, and OOT AUC rose 0.676 to 0.716.

  `keep` is safe through the classification pipeline because WOE maps every bin —
  including Missing — to a numeric value before the model sees it.

## [3.2.0] - 2026-08-15

Final batch from the adversarial audit. Six defects, five of which share one
theme: sample weights did not mean the same thing at every stage. This matters
precisely where credit modelling lives — undersampled goods, reject inference —
so anything trained with weights before this release should be refitted.

### Fixed

- **Binning cut points ignored sample weights entirely.** Quantile, CART and the
  sequential CART fallback all fitted their cuts on raw rows. With goods
  undersampled 1:50, the "equal-frequency" bins held 39%/24%/18%/13%/7% of the
  weighted population instead of 20% each, and CART split the sample rather than
  the population. Weighted quantiles now drive the cuts, and the tree receives
  `sample_weight` — its splits match a direct sklearn fit exactly. The unweighted
  Polars fast path is untouched.

- **WOE/IV smoothing was scale-dependent.** The fixed 0.5 Laplace constant was
  added to weighted sums, so multiplying every weight by a constant — which
  changes nothing about their relative meaning — moved IV by a factor of 1.6.
  Smoothing is now expressed in mean-weight units, making IV invariant to weight
  scale and exactly unchanged when unweighted.

- **PSI bin edges came from the unweighted benchmark.** Under undersampling the
  "deciles" held 1%-19% of the weighted population, so PSI concentrated its
  resolution on the oversampled tail and was blind where the population actually
  is. Edges now split the weighted benchmark; measured spread fell from 0.177
  to 0.0006.

- **`RegressionPipeline` ignored `use_sample_weight`.** It keyed weighting off
  the presence of `weight_col` alone, so the shipped config — which carries
  `use_sample_weight: false` next to `weight_col: weight` — trained a weighted
  regression and an unweighted classification from the same file. On a fixture
  where weighting flips the relationship, the coefficient went -0.17 to +4.82
  with the flag nominally off.

- **`log1p` regression with `early_stopping_eval='oot'` validated on the wrong
  scale.** The model fits `log1p(y)` but the OOT eval set received raw targets,
  so early stopping compared log-space predictions to raw labels: the validation
  curve ran 14.215 → 13.921 (flat noise) instead of 0.567 → 0.270, and the tree
  count was chosen at random. `docs/guide/regression.md` demonstrates exactly
  this combination.

- **`ScorecardBuilder.score()` returned a constant for NumPy input.** The array
  branch labelled raw driver columns with the model's `_bin` WOE names, so the
  binner matched nothing and every row collapsed to `offset_` — 1 distinct score
  instead of 137, with no error raised. The array path now uses the binner's
  fitted driver names and rejects a mismatched column count.

### Added

- `auto_modeling_tool.core.stats` — `weighted_quantile` / `weighted_mean`, shared
  by binning and PSI. Uniform weights delegate to `np.quantile` so unweighted
  behaviour is bit-for-bit unchanged.
- `tests/test_weight_contract.py` — 12 tests pinning each of the above, including
  a direct comparison of CART cuts against sklearn with weights applied.

## [3.1.3] - 2026-08-15

Second batch from the adversarial audit. The PSI defect made drift monitoring
report "Stable" for essentially every real feature.

### Fixed

- **PSI returned exactly 0.0 whenever the benchmark contained a null.** Quantile
  edges came from `np.quantile` over the raw array, so one NaN made every edge
  NaN, every bin membership test False, and both populations collapse to the
  same epsilon share. A benchmark/actual pair with PSI 10.39 reported 0.0 after
  a single value was set to NaN. Credit features are almost always partly null,
  so `calculate_feature_psi` reported "Stable" for nearly everything, and the
  release gate's `--max-psi` was reading that number. Edges are now computed
  over finite values only; an all-missing population raises instead.

- **`clean_strategy` silently imputed 0 for anything but mean/median.** The CLI
  offered `forward` and `backward`; the stateful preprocessor implemented
  neither and fell through to a literal 0 — worst-case for income, best-case for
  DPD, applied identically in training and in the saved artifact. Order-dependent
  fills cannot work in a fitted transform that scores row-wise, so they are now
  rejected with an explanatory error and removed from the CLI choices. `zero` is
  explicit rather than a fallthrough.

- **`--selection rfe` crashed.** The CLI and both guides advertised `rfe`; the
  implementation only accepted `recursive`. `rfe` is now an accepted spelling,
  and the CLI additionally exposes `recursive` and `mutual_info`, which worked
  but were unreachable.

- **Run archives contained other runs' reports.** `archive_run` copied every
  `Model_Report_*.xlsx` in the output directory, so the third run's archive held
  reports 1, 2 and 3 and grew without bound. Only the report the run just wrote
  is archived.

- **An archived run could not be replayed through `--config`.** The snapshot is
  flat pipeline kwargs; the loader only understood the nested schema, so
  `--config runs/<id>/config.yaml` failed with "Config must define data.path" —
  contradicting the feature's documented promise. The loader now accepts the
  archived form, and replaying reproduces the original metrics exactly.

- **Documented examples raised.** README and the feature-selection guide both
  showed `fit_transform(df, target_col=...)`, which is not the signature, and the
  guide imported `remove_multicollinearity` from a path that never exported it
  while describing its return value as a feature list rather than
  `(DataFrame, dropped)`. All examples are now executed by
  `tests/test_documented_api.py`.

## [3.1.2] - 2026-08-15

Two defects found by an adversarial audit, both of which inflated reported model
quality. If you have models trained with an earlier version, retrain and compare.

### Fixed

- **Evaluation columns were trained on as model features.** `segment_cols`,
  `temporal_col` and `benchmark_cols` were excluded from nothing — they stayed in
  the feature pool, went through binning and IV screening, and were selected into
  the model whenever they carried signal.

  Training on a temporal column is time leakage: the model learns "March is bad"
  and is meaningless on an unseen month. Training on a benchmark column destroys
  the very comparison it was configured for, and makes an external score a hard
  dependency of production scoring. Measured on a fixture where both carried
  signal, OOT AUC fell from 0.9987 to 0.5657 once they were excluded — the second
  number is the model's real discrimination.

  These columns are now role columns, like the target and the sample marker. The
  Temporal_Stability, Benchmark_Performance and Segment_Summary report sheets are
  unaffected, and the scoring artifact no longer demands columns the model never
  uses.

- **KS was inflated by tied scores.** `calculate_ks` took the maximum cumulative
  TPR/FPR gap over every row rather than only where a threshold could fall, so
  samples sharing a score were treated as separable. A constant score — zero
  discrimination — reported KS 0.924 when the input arrived sorted by label.

  Continuous probabilities were unaffected, which is why this survived: it bites
  exactly where credit work lives, on scorecard integer points, coarse binned
  scores and rule-engine outputs. KS is now read only at the end of each tied
  group and matches the sklearn ROC definition on every case tested, weighted and
  unweighted.

## [3.1.1] - 2026-08-15

### Fixed

- **Run history silently lost runs.** `new_run_id` hashed only the second-resolution
  timestamp, the model name, and a seed that `archive_run` filled with the output
  directory — constant across a parameter sweep. Every run starting in the same
  second therefore produced an identical id, and `archive_run` wrote into the
  existing directory with `mkdir(exist_ok=True)` plus overwriting copies. No error,
  no warning. Measured: six runs launched, one archived.

  This was the feature's own flagship workflow — sweeping a parameter and comparing
  the archived runs — and it was the workflow that destroyed the data.

  The id suffix is now per-call entropy, and `archive_run` refuses to reuse an
  existing directory, suffixing instead. The id and `created_at` also derive from a
  single clock reading, so they can no longer disagree across a second boundary.

  `test_same_second_runs_do_not_collide` passed throughout because it hand-fed two
  different seeds, which the production caller never does. It now asserts that 50
  ids generated from identical inputs are all distinct, and a new test archives five
  runs at one fixed timestamp and requires all five back.

## [3.1.0] - 2026-08-15

### Added

- **Run history.** Every pipeline run is archived to `runs/<run_id>/` with the
  resolved config, the metrics it reached, and the deployable artifact, so a
  model can be traced back long after the fact. `output/` is untouched — it
  still holds the latest run and every existing path keeps working.
  - `python -m auto_modeling_tool.runs list` — history, newest first
  - `python -m auto_modeling_tool.runs compare <a> <b>` — metric deltas, config
    diff, and added/removed model features side by side
  - `python -m auto_modeling_tool.runs show <id>` — one run in full
  - Run ids are `<timestamp>-<algorithm>-<hash>`: sortable, readable, and safe
    for two runs landing in the same second. Any unique prefix resolves.
    (The same-second guarantee did not actually hold in 3.1.0 — fixed in 3.1.1.)
  - `output.runs_dir` / `output.archive_run` in config; `--runs-dir` /
    `--no-archive-run` on the CLI
  - Archiving failures are logged and swallowed — a twenty-minute run is never
    lost to a full disk

### Fixed

- **`n_features` was ignored by the default selection method.** `_select_by_iv`
  never received it, so IV selection was governed purely by `iv_threshold` and
  configuring `n_features: 8` or `n_features: 3` produced identical feature
  sets. It now caps the IV-ranked list, matching the behaviour every other
  method already had. Found by comparing two archived runs — the first thing
  run history paid for.

  Note: on the IV path this changes results. Runs that previously kept every
  feature above the IV threshold will now keep at most `n_features` of them
  (default 20), strongest first.

## [3.0.1] - 2026-08-14

### Fixed

- Dev/OOT score PSI was computed for the report but never added to `metrics_`,
  so it never reached `scoring_artifact.pkl`. `validate_release(..., max_psi=)`
  and `scripts/validate_release.py --max-psi` therefore always read `psi=None`
  and failed the check — the threshold was inert. PSI is now a metric like any
  other, and the pipeline contract test asserts it survives into the artifact.

## [3.0.0] - 2026-08-14

Packaging and code-quality pass. The only breaking change is the import name.

### Breaking

- **The package is now `auto_modeling_tool`, not `src`.** Installing 2.x put
  five generic top-level names into site-packages — `src`, `tests`, `scripts`,
  `examples` and `build` — so `import src` resolved to this project and any
  other package doing the same would collide. `packages.find` is now scoped to
  `auto_modeling_tool*`; a fresh install contributes exactly one top-level name.

  Migration is a find-and-replace:

  ```python
  from src import profile_risk          # 2.x
  from auto_modeling_tool import profile_risk   # 3.0
  ```

  ```bash
  python -m src.main --config ...                  # 2.x
  python -m auto_modeling_tool.main --config ...   # 3.0
  ```

  The `automodel` console script is unaffected. No API, argument or output
  changed — only the import path. Module references in older changelog entries
  below are written with the new name for readability; they lived under `src.`
  at the time.

### Fixed

- `ProbabilityCalibrator` could never run. Two independent defects:
  its internal dummy classifier listed `BaseEstimator` before `ClassifierMixin`,
  so sklearn's MRO-based tag resolution reported a regressor and
  `CalibratedClassifierCV` refused to fit; and the dummy's `predict_proba`
  closed over the full probability vector while ignoring its `X` argument, so
  every cross-validation fold received a length mismatch. Reimplemented with the
  standard formulation — `LogisticRegression` on the raw score for Platt
  scaling, `IsotonicRegression` for isotonic. On a 600-row over-confident score,
  Brier improves 0.2294 → 0.1781 (sigmoid) / 0.1669 (isotonic).
- `ProbabilityCalibrator` now accepts `pl.Series` for `y_prob` (previously only
  `pl.DataFrame`, awkward in a Polars-first codebase) and raises a clear error
  on length mismatch instead of failing deep inside sklearn.
- Removed a dead `bin_expr` assignment in `woe_binning`.

### Added

- `LICENSE` — the MIT text that README and `pyproject` both pointed at but which
  was never committed.
- `tests/test_standalone_tools.py` — direct coverage for `calibration`,
  `cross_validation` and `tuning`, the three modules that are exported as public
  API but are not called by any pipeline. The calibrator shipped broken
  precisely because nothing tested it.

### Changed

- `ruff check` is clean (was 1511 errors). Roughly 1140 whitespace/import/
  f-string fixes, 364 typing modernizations (`List[str]` → `list[str]`; PEP 585
  generics are available on the declared 3.9 floor), and 81 unused imports.
  No behaviour change; verified with 114 passing tests on 3.9.6.

### Known gaps

- `calibration`, `tuning` and `cross_validation` remain reachable only by direct
  import — `auto_pipeline` does not call them, and `ModelTrainer` carries its own
  private tuning implementation separate from `modeling.tuning`.
- No reject inference, no model-level monotonic constraints, no
  champion/challenger comparison.

## [2.2.0] - 2026-08-14

Production-hardening iteration aligned with `USER_GUIDE 0.75`: leakage-safe
Dev/OOT modeling, configuration-driven runs, deployable scoring artifacts,
a multi-sheet audit report, and a pre-release quality gate.

### Added

- `src.config` — canonical YAML configuration
  - `configs/pipeline_config.yaml` as the single entry point:
    `python -m src.main --config configs/pipeline_config.yaml`
  - `load_pipeline_config` / `config_to_pipeline_kwargs`, relative paths
    resolved against the config file, legacy `default_config` wrapper and
    key aliases supported, every field overridable from the CLI
- `src.modeling.artifact` — serializable scoring artifacts
  - `build_scoring_artifact` / `build_regression_artifact` bundle the fitted
    preprocessor, binner, selector and model plus the driver contract
  - `score_with_artifact` scores raw driver data end to end;
    `save_scoring_artifact` / `load_scoring_artifact` round-trip
    `output/scoring_artifact.pkl`
- `src.pipelines.regression_pipeline` — regression task support with
  RMSE / MAE / R², optional `log1p` target transform and artifact round-trip
- `src.evaluation.quality_gate` — pre-release validation
  - `validate_release` checks the artifact contract, driver/feature contract,
    required report sheets and optional AUC / PSI thresholds
- `src.reports.excel` — multi-sheet audit workbook (binning, WOE, IV,
  variable audit, screening, Dev/OOT metrics, score bins, score PSI,
  stability, segment, temporal and benchmark views)
- Credit-score conversion with configurable `base_score`, `pdo` and score
  bounds, written to `pred_score` by default
- Standalone scripts
  - `scripts/score_new_samples.py` — score new data from an artifact file or
    an output directory
  - `scripts/evaluate_model_performance.py` — evaluate an already-scored file
  - `scripts/explore_dataset.py` — standalone EDA with Dev/OOT and Excel output
  - `scripts/validate_release.py` — run the release gate from the command line
- Sample weighting (`use_sample_weight` / `weight_col`) threaded through
  binning statistics, IV, feature screening, training, evaluation, PSI and
  monitoring trends, plus regression training and OOT early stopping
- Configurable classification algorithms (logistic, tree, random forest,
  XGBoost, LightGBM, CatBoost) with `dev_holdout` / `oot` early stopping
- Data loaders for CSV, XLSX, XLS, Parquet and PKL
- GitHub Actions CI running the suite on Python 3.9 / 3.10 / 3.11 / 3.12
- Tests: `test_config.py`, `test_regression.py`, `test_reports.py`,
  `test_exploration.py`, `test_pipeline_contract.py` (103 tests total)

### Changed

- Dev/OOT splitting now supports sample labels, date boundaries and a random
  fallback; preprocessing, WOE and feature screening are fitted on Dev only
  to prevent OOT leakage
- WOE binning handles categorical variables, unseen and rare categories,
  smoothing, and monotonic calibration for numeric variables
- Model Report always contains `Overview_Performance`, `Report_Index` and
  `Artifact_Metadata`
- Monitoring supports numeric and categorical features, group trends,
  weighted PSI, missing rate, bad rate and score trends
- Pandas and CatBoost are now declared runtime dependencies
- Logger writes UTF-8 safely under Windows GBK consoles

### Fixed

- WOE screening results were computed but never reached the model
- `ScorecardBuilder` assigned points by bin index instead of bin value
- Duplicate score values broke lift-table and report generation
- `export_excel: false` was ignored by the classification and regression
  pipelines
- Standalone scoring output now uses `pred_proba` (with `prediction` kept as
  a compatibility alias); regression outputs `pred_value`

### Compatibility

- All nine v2.1.0 CLI flags still work; 29 new flags were added alongside them
- `--test-size` now means "proportion held out as OOT when no sample column or
  date boundary is given", rather than a plain random train/test split
- Low-level APIs from v2.1.0 are untouched
- `requires-python` stays at `>=3.9` and is now enforced: all four versions run
  the full suite in CI, 103 passed on each

### Upgrade note

- Preprocessing and binning used to be fitted on the full dataset, which made
  OOT metrics optimistic. With the leakage fixed, OOT AUC on the same data is
  usually lower — that is the honest number.

## [2.1.0] - 2026-07-23

MARS-inspired iteration: task-oriented workflow entry points, structured
report objects, and a monitoring & alerting module.

### Added

- `src.analysis` — task-oriented entry points
  - `profile_data(df, ...)` → `DataProfileReport` (overview / dq / stats
    tables + per-period trends)
  - `profile_risk(df, target=...)` → `RiskProfile` (binning + IV/KS +
    cross-period PSI in one call; carries a reusable fitted binner)
- `src.monitoring` — feature/model monitoring
  - `Monitor` with two benchmark modes (`group_col` first period, or an
    explicit `benchmark_df`), PSI / missing-rate / bad-rate / score-mean
    trends, and per-feature `status` verdicts
  - `generate_monitoring_alert` + `AlertConfig` — priority-sorted Chinese
    alert digest for IM/email push
- `src.reports` — structured report objects with a shared
  `summary_table` / `detail_table` / `trend_tables` / `metadata` layout
  and `to_markdown()` / `save()` exports
- `src.evaluation.stability` — binned-distribution PSI primitives
- MkDocs Material documentation site (`docs/`, `mkdocs.yml`)
- Tests: `tests/test_analysis.py`, `tests/test_monitoring.py`

### Changed

- Top-level package now exports the workflow API directly
  (`from src import profile_risk, Monitor, generate_monitoring_alert`)
- README restructured around tasks ("start from your task", entry-point
  decision table, stability levels); removed trailing artifacts
- Version bumped to 2.1.0

### Compatibility

- No breaking changes; all existing low-level APIs untouched

## [2.0.0]

- Initial Polars-first architecture: WOE binning (quantile / uniform /
  CART), feature selection, model training & evaluation, auto pipeline
