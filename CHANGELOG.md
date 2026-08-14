# Changelog

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
