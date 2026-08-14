# -*- coding: utf-8 -*-
"""CLI entry point for the leakage-safe auto-modeling pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .config import config_to_pipeline_kwargs, load_pipeline_config
from .pipelines.auto_pipeline import AutoPipeline


def run_modeling_pipeline(
    data_path: str,
    target_col: str,
    output_dir: str = "output",
    *,
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
    exclude_columns: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Run the shared AutoPipeline implementation."""
    pipeline = AutoPipeline(
        target_col=target_col,
        test_size=test_size,
        n_bins=n_bins,
        binning_method=binning_method,
        selection_method=selection_method,
        n_features=n_features,
        random_state=random_state,
        sample_col=sample_col,
        date_column=date_column,
        oot_start=oot_start,
        dev_label=dev_label,
        oot_label=oot_label,
    )
    pipeline.fit(
        data_path,
        sample_col=sample_col,
        date_column=date_column,
        oot_start=oot_start,
        dev_label=dev_label,
        oot_label=oot_label,
        clean_strategy=clean_strategy,
        normalize_method=normalize_method,
        min_samples_bin=min_samples_bin,
        monotonic=monotonic,
        exclude_columns=exclude_columns or [],
    )
    metrics = pipeline.evaluate()
    pipeline.save(output_dir)
    return {
        "model": pipeline.model_,
        "metrics": metrics,
        "feature_importance": pipeline.feature_importance_,
        "selected_features": pipeline.selected_features_,
        "output_path": Path(output_dir),
        "pipeline": pipeline,
    }


def run_configured_pipeline(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run from YAML, applying only explicitly supplied CLI overrides."""
    config = load_pipeline_config(config_path)
    kwargs = config_to_pipeline_kwargs(config)
    kwargs.update(overrides or {})
    return run_modeling_pipeline(**kwargs)


def _cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    values = {
        "data_path": args.input,
        "target_col": args.target,
        "output_dir": args.output,
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
        "exclude_columns": args.exclude_column,
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
    parser.add_argument("--test-size", type=float, help="Random fallback OOT proportion")
    parser.add_argument("--n-bins", type=int)
    parser.add_argument("--method", choices=["quantile", "uniform", "cart"])
    parser.add_argument("--selection", choices=["iv", "correlation", "rfe", "variance"])
    parser.add_argument("--n-features", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-column", help="Column containing Dev/OOT labels")
    parser.add_argument("--date-column", help="Date column for chronological split")
    parser.add_argument("--oot-start", help="First date/value included in OOT")
    parser.add_argument("--dev-label")
    parser.add_argument("--oot-label")
    parser.add_argument(
        "--clean-strategy",
        choices=["mean", "median", "zero", "forward", "backward"],
    )
    parser.add_argument("--normalize-method", choices=["minmax", "zscore", "robust"])
    parser.add_argument("--min-samples-bin", type=int)
    parser.add_argument("--monotonic", action="store_true", default=None)
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
        metric = metrics.get("auc_roc", metrics.get("accuracy", float("nan")))
        print(f"\nSuccess. Primary metric: {metric:.4f}")
        return 0
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
