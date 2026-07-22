# Changelog

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
