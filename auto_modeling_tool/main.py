"""CLI entry point for the leakage-safe auto-modeling pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from .config import config_to_pipeline_kwargs, load_pipeline_config
from .core.logger import logger
from .pipelines.auto_pipeline import AutoPipeline
from .runs import archive_run as _archive_run


def _archive_if_enabled(
    output_dir: str,
    *,
    runs_dir: Optional[str],
    archive: bool,
    config: dict[str, Any],
    metrics: dict[str, Any],
    selected_features: Optional[list[str]],
    task: str,
    model_type: str,
) -> Optional[Path]:
    """Archive the finished run, never failing the pipeline over it.

    A run that took twenty minutes must not be lost to a full disk or a
    read-only path, so archiving problems are reported and swallowed.
    """
    if not archive or not runs_dir:
        return None
    try:
        run_path = _archive_run(
            output_dir,
            runs_dir=runs_dir,
            config=config,
            metrics=metrics,
            selected_features=selected_features,
            task=task,
            model_type=model_type,
        )
        logger.info(f"📁 Run archived to {run_path}")
        return run_path
    except Exception as exc:
        logger.warning(f"Could not archive run (outputs are intact): {exc}")
        return None


def run_modeling_pipeline(
    data_path: str,
    target_col: str,
    output_dir: str = "output",
    *,
    target_mode: str = "classification",
    model_type: str = "logistic",
    model_params: Optional[dict[str, Any]] = None,
    target_transform: Optional[str] = None,
    early_stopping_eval: str = "none",
    early_stopping_rounds: Optional[int] = None,
    early_stopping_metric: Optional[str] = None,
    test_size: float = 0.2,
    n_bins: int = 10,
    binning_method: str = "quantile",
    selection_method: str = "iv",
    n_features: int = 20,
    random_state: int = 42,
    sample_col: Optional[str] = None,
    date_column: Optional[str] = None,
    oot_start: Optional[Any] = None,
    dev_label: Any = "dev",
    oot_label: Any = "oot",
    clean_strategy: str = "median",
    normalize_method: Optional[str] = "zscore",
    min_samples_bin: int = 50,
    monotonic: bool = False,
    smoothing: float = 0.5,
    exclude_columns: Optional[list[str]] = None,
    use_sample_weight: bool = False,
    weight_col: Optional[str] = None,
    segment_cols: Optional[list[str]] = None,
    temporal_col: Optional[str] = None,
    benchmark_cols: Optional[list[str]] = None,
    export_excel: bool = True,
    convert_to_credit_score: bool = False,
    credit_score_col: str = "pred_score",
    base_score: float = 500.0,
    pdo: float = 50.0,
    min_score: float = 300.0,
    max_score: float = 900.0,
    data_encoding: str = "utf-8",
    runs_dir: Optional[str] = "runs",
    archive_run: bool = True,
) -> dict[str, Any]:
    """Run the classification or regression pipeline selected by config."""
    # Captured before any local is introduced, so this is exactly the resolved
    # configuration (YAML plus CLI overrides) that produced the run.
    resolved_config = dict(locals())

    common = {
        "target_col": target_col,
        "test_size": test_size,
        "random_state": random_state,
        "sample_col": sample_col,
        "date_column": date_column,
        "oot_start": oot_start,
        "dev_label": dev_label,
        "oot_label": oot_label,
    }
    fit_kwargs = {
        "sample_col": sample_col,
        "date_column": date_column,
        "oot_start": oot_start,
        "dev_label": dev_label,
        "oot_label": oot_label,
        "clean_strategy": clean_strategy,
        "normalize_method": normalize_method,
        "min_samples_bin": min_samples_bin,
        "monotonic": monotonic,
        "smoothing": smoothing,
        "exclude_columns": exclude_columns or [],
        "use_sample_weight": use_sample_weight,
        "weight_col": weight_col,
        "segment_cols": segment_cols or [],
        "temporal_col": temporal_col,
        "benchmark_cols": benchmark_cols or [],
        "export_excel": export_excel,
        "encoding": data_encoding,
    }

    if target_mode == "regression":
        if convert_to_credit_score:
            raise ValueError("convert_to_credit_score is only supported for classification")
        from .pipelines.regression_pipeline import RegressionPipeline

        pipeline = RegressionPipeline(
            model_type=model_type,
            model_params=model_params,
            target_transform=target_transform,
            early_stopping_eval=early_stopping_eval,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_metric=early_stopping_metric,
            clean_strategy=clean_strategy,
            normalize_method=normalize_method,
            **common,
        )
        pipeline.fit(data_path, **fit_kwargs)
        metrics = pipeline.evaluate()
        pipeline.save(output_dir)
        run_path = _archive_if_enabled(
            output_dir,
            runs_dir=runs_dir,
            archive=archive_run,
            config=resolved_config,
            metrics=metrics,
            selected_features=pipeline.feature_columns_,
            task="regression",
            model_type=model_type,
        )
        return {
            "model": pipeline.model_,
            "metrics": metrics,
            "feature_importance": None,
            "selected_features": pipeline.feature_columns_,
            "output_path": Path(output_dir),
            "run_path": run_path,
            "pipeline": pipeline,
        }

    if target_mode != "classification":
        raise ValueError("target_mode must be classification or regression")
    pipeline = AutoPipeline(
        model_type=model_type,
        model_params=model_params,
        early_stopping_eval=early_stopping_eval,
        early_stopping_rounds=early_stopping_rounds,
        early_stopping_metric=early_stopping_metric,
        convert_to_credit_score=convert_to_credit_score,
        credit_score_col=credit_score_col,
        base_score=base_score,
        pdo=pdo,
        min_score=min_score,
        max_score=max_score,
        n_bins=n_bins,
        binning_method=binning_method,
        selection_method=selection_method,
        n_features=n_features,
        **common,
    )
    pipeline.fit(data_path, **fit_kwargs)
    metrics = pipeline.evaluate()
    pipeline.save(output_dir)
    run_path = _archive_if_enabled(
        output_dir,
        runs_dir=runs_dir,
        archive=archive_run,
        config=resolved_config,
        metrics=metrics,
        selected_features=pipeline.selected_features_,
        task="classification",
        model_type=model_type,
    )
    return {
        "model": pipeline.model_,
        "metrics": metrics,
        "feature_importance": pipeline.feature_importance_,
        "selected_features": pipeline.selected_features_,
        "output_path": Path(output_dir),
        "run_path": run_path,
        "pipeline": pipeline,
    }


def run_configured_pipeline(
    config_path: str,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run from YAML, applying only explicitly supplied CLI overrides."""
    config = load_pipeline_config(config_path)
    kwargs = config_to_pipeline_kwargs(config)
    kwargs.update(overrides or {})
    return run_modeling_pipeline(**kwargs)


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "data_path": args.input,
        "target_col": args.target,
        "output_dir": args.output,
        "target_mode": args.target_mode,
        "model_type": args.model,
        "target_transform": args.target_transform,
        "early_stopping_eval": args.early_stopping_eval,
        "early_stopping_rounds": args.early_stopping_rounds,
        "early_stopping_metric": args.early_stopping_metric,
        "test_size": args.test_size,
        "n_bins": args.n_bins,
        "binning_method": args.method,
        "selection_method": args.selection,
        "n_features": args.n_features,
        "random_state": args.seed,
        "sample_col": args.sample_column,
        "date_column": args.date_column,
        "oot_start": args.oot_start,
        "dev_label": args.dev_label,
        "oot_label": args.oot_label,
        "clean_strategy": args.clean_strategy,
        "normalize_method": args.normalize_method,
        "min_samples_bin": args.min_samples_bin,
        "monotonic": args.monotonic,
        "smoothing": args.smoothing,
        "exclude_columns": args.exclude_column,
        "use_sample_weight": args.use_sample_weight,
        "weight_col": args.weight_column,
        "segment_cols": args.segment_column,
        "temporal_col": args.temporal_column,
        "benchmark_cols": args.benchmark_column,
        "data_encoding": args.encoding,
        "convert_to_credit_score": args.convert_to_credit_score,
        "credit_score_col": args.credit_score_col,
        "base_score": args.base_score,
        "pdo": args.pdo,
        "min_score": args.min_score,
        "max_score": args.max_score,
        "runs_dir": args.runs_dir,
        "archive_run": False if args.no_archive_run else None,
    }
    return {key: value for key, value in values.items() if value is not None}

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AutoModelTool - configuration-driven Dev/OOT pipeline"
    )
    parser.add_argument("--config", help="Canonical YAML pipeline configuration")
    parser.add_argument("--input", "-i", help="Input data file path")
    parser.add_argument("--target", "-t", help="Target column name")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--target-mode", choices=["classification", "regression"])
    parser.add_argument("--model", choices=[
        "logistic", "linear", "linear_regression", "tree",
        "random_forest", "xgboost", "lightgbm", "catboost",
    ])
    parser.add_argument("--target-transform", choices=["log1p"])
    parser.add_argument(
        "--early-stopping-eval",
        choices=["none", "dev_holdout", "oot"],
    )
    parser.add_argument("--early-stopping-rounds", type=int)
    parser.add_argument("--early-stopping-metric")
    parser.add_argument("--test-size", type=float, help="Random fallback OOT proportion")
    parser.add_argument("--n-bins", type=int)
    parser.add_argument("--method", choices=["quantile", "uniform", "cart"])
    parser.add_argument(
        "--selection",
        choices=["iv", "correlation", "variance", "rfe", "recursive", "mutual_info"],
    )
    parser.add_argument("--n-features", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-column", help="Column containing Dev/OOT labels")
    parser.add_argument("--date-column", help="Date column for chronological split")
    parser.add_argument("--oot-start", help="First date/value included in OOT")
    parser.add_argument("--dev-label")
    parser.add_argument("--oot-label")
    parser.add_argument(
        "--clean-strategy",
        choices=["mean", "median", "zero", "keep"],
    )
    parser.add_argument("--normalize-method", choices=["minmax", "zscore", "robust"])
    parser.add_argument("--min-samples-bin", type=int)
    parser.add_argument("--monotonic", action="store_true", default=None)
    parser.add_argument("--smoothing", type=float)
    parser.add_argument("--use-sample-weight", action="store_true", default=None)
    parser.add_argument("--weight-column")
    parser.add_argument("--segment-column", action="append")
    parser.add_argument("--temporal-column")
    parser.add_argument("--benchmark-column", action="append")
    parser.add_argument("--encoding")
    parser.add_argument("--convert-to-credit-score", action="store_true", default=None)
    parser.add_argument("--credit-score-col")
    parser.add_argument("--base-score", type=float)
    parser.add_argument("--pdo", type=float)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--max-score", type=float)
    parser.add_argument(
        "--runs-dir",
        help="Where to archive this run (default: runs)",
    )
    parser.add_argument(
        "--no-archive-run",
        action="store_true",
        default=None,
        help="Skip run archiving; only write the output directory",
    )
    parser.add_argument(
        "--exclude-column",
        action="append",
        help="Column excluded from modeling; repeat for multiple columns",
    )
    return parser

def main() -> int:
    args = _parser().parse_args()
    overrides = _cli_overrides(args)

    try:
        if args.config:
            results = run_configured_pipeline(args.config, overrides)
        else:
            if not args.input or not args.target:
                raise ValueError("--input and --target are required unless --config is used")
            defaults = {
                "data_path": args.input,
                "target_col": args.target,
                "output_dir": args.output or "output",
                "target_mode": args.target_mode or "classification",
                "model_type": args.model or "logistic",
                "target_transform": args.target_transform,
                "early_stopping_eval": args.early_stopping_eval or "none",
                "early_stopping_rounds": args.early_stopping_rounds,
                "early_stopping_metric": args.early_stopping_metric,
                "test_size": args.test_size if args.test_size is not None else 0.2,
                "n_bins": args.n_bins if args.n_bins is not None else 10,
                "binning_method": args.method or "quantile",
                "selection_method": args.selection or "iv",
                "n_features": args.n_features if args.n_features is not None else 20,
                "random_state": args.seed if args.seed is not None else 42,
            }
            defaults.update(overrides)
            results = run_modeling_pipeline(**defaults)

        metrics = results["metrics"]
        metric = metrics.get(
            "auc_roc",
            metrics.get("rmse", metrics.get("accuracy", float("nan"))),
        )
        metric_name = "AUC" if "auc_roc" in metrics else (
            "RMSE" if "rmse" in metrics else "accuracy"
        )
        print(f"\nSuccess. Primary metric ({metric_name}): {metric:.4f}")
        return 0
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
