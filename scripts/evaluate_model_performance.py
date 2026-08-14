#!/usr/bin/env python3
"""Evaluate an existing score file without rerunning model training."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_data
from src.evaluation.metrics import (
    calculate_all_metrics,
    calculate_regression_metrics,
)
from src.reports.excel import write_model_report


def _evaluate_frame(
    frame: pl.DataFrame,
    *,
    task: str,
    target_col: str,
    score_col: str,
    threshold: float,
    weight_col: Optional[str],
    use_sample_weight: bool,
) -> dict[str, float]:
    y_true = frame.get_column(target_col).to_numpy()
    score = frame.get_column(score_col).to_numpy()
    weights = (
        frame.get_column(weight_col)
        if use_sample_weight and weight_col
        else None
    )
    if task == "regression":
        return calculate_regression_metrics(
            y_true,
            score,
            sample_weight=weights,
        )

    prediction = (score >= threshold).astype(int)
    return calculate_all_metrics(
        y_true,
        prediction,
        score,
        sample_weight=weights,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Score file")
    parser.add_argument("--target", required=True, help="Target column")
    parser.add_argument("--score-column", default="prediction")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sample-column")
    parser.add_argument("--dev-label", default="dev")
    parser.add_argument("--oot-label", default="oot")
    parser.add_argument("--weight-column")
    parser.add_argument(
        "--use-sample-weight",
        action="store_true",
        help="Use the weight column in metrics; otherwise use unit weights",
    )
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--output", default="performance_report.json")
    args = parser.parse_args()

    path = Path(args.input)
    load_kwargs = {"encoding": args.encoding} if path.suffix.lower() == ".csv" else {}
    frame = load_data(path, **load_kwargs)
    required = [args.target, args.score_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Score file is missing required columns: {missing}")

    groups: dict[str, pl.DataFrame] = {"all": frame}
    if args.sample_column:
        if args.sample_column not in frame.columns:
            raise ValueError(f"Sample column '{args.sample_column}' not found")
        groups = {
            "dev": frame.filter(pl.col(args.sample_column) == args.dev_label),
            "oot": frame.filter(pl.col(args.sample_column) == args.oot_label),
        }
        if any(len(group) == 0 for group in groups.values()):
            raise ValueError("Both Dev and OOT groups must contain rows")

    metrics: dict[str, Any] = {}
    for name, group in groups.items():
        values = _evaluate_frame(
            group,
            task=args.task,
            target_col=args.target,
            score_col=args.score_column,
            threshold=args.threshold,
            weight_col=args.weight_column,
            use_sample_weight=args.use_sample_weight,
        )
        metrics.update({f"{name}_{key}": float(value) for key, value in values.items()})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": args.task,
        "input": str(path),
        "target_col": args.target,
        "score_col": args.score_column,
        "threshold": args.threshold,
        "rows": len(frame),
        "use_sample_weight": args.use_sample_weight,
        "weight_col": args.weight_column,
    }
    if output.suffix.lower() == ".xlsx":
        write_model_report(
            output.parent,
            metrics,
            metadata=metadata,
            filename=output.name,
        )
    else:
        output.write_text(
            json.dumps(
                {"metrics": metrics, "metadata": metadata},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"Saved performance report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
