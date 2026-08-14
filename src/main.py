# -*- coding: utf-8 -*-
"""CLI entry point for the leakage-safe auto-modeling pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AutoModelTool - leakage-safe Dev/OOT modeling pipeline"
    )
    parser.add_argument("--input", "-i", required=True, help="Input data file path")
    parser.add_argument("--target", "-t", required=True, help="Target column name")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Random fallback OOT proportion",
    )
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--method",
        default="quantile",
        choices=["quantile", "uniform", "cart"],
    )
    parser.add_argument(
        "--selection",
        default="iv",
        choices=["iv", "correlation", "rfe", "variance"],
    )
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-column", help="Column containing Dev/OOT labels")
    parser.add_argument("--date-column", help="Date column for chronological split")
    parser.add_argument("--oot-start", help="First date/value included in OOT")
    parser.add_argument("--dev-label", default="dev")
    parser.add_argument("--oot-label", default="oot")
    args = parser.parse_args()

    try:
        results = run_modeling_pipeline(
            data_path=args.input,
            target_col=args.target,
            output_dir=args.output,
            test_size=args.test_size,
            n_bins=args.n_bins,
            binning_method=args.method,
            selection_method=args.selection,
            n_features=args.n_features,
            random_state=args.seed,
            sample_col=args.sample_column,
            date_column=args.date_column,
            oot_start=args.oot_start,
            dev_label=args.dev_label,
            oot_label=args.oot_label,
        )
        metrics = results["metrics"]
        metric = metrics.get("auc_roc", metrics.get("accuracy", float("nan")))
        print(f"\nSuccess. Primary metric: {metric:.4f}")
        return 0
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
