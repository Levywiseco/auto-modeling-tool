#!/usr/bin/env python3
"""Score raw driver data with a saved scoring artifact."""

import argparse
from pathlib import Path
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_data
from src.modeling.artifact import load_scoring_artifact, score_with_artifact
from src.modeling.scorecard import probability_to_credit_score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="scoring_artifact.pkl or pipeline.pkl")
    parser.add_argument("--input", required=True, help="Raw driver data")
    parser.add_argument("--output", required=True, help="CSV or Parquet output path")
    parser.add_argument(
        "--convert-to-credit-score",
        action="store_true",
        default=None,
        help="Add a clipped credit-score column for classification artifacts",
    )
    parser.add_argument("--credit-score-col")
    parser.add_argument("--base-score", type=float)
    parser.add_argument("--pdo", type=float)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--max-score", type=float)
    args = parser.parse_args()

    model_path = Path(args.model)
    if model_path.is_dir():
        model_path = model_path / "scoring_artifact.pkl"
    artifact = load_scoring_artifact(model_path)
    raw = load_data(args.input)
    is_regression = artifact.get("task") == "regression"
    predictions = score_with_artifact(artifact, raw, return_proba=not is_regression)
    output_column = "prediction" if not is_regression else "pred_value"
    result = raw.with_columns(pl.Series(output_column, predictions))

    scoring = artifact.get("metadata", {}).get("scoring", {})
    convert_to_credit_score = (
        bool(args.convert_to_credit_score)
        if args.convert_to_credit_score is not None
        else bool(scoring.get("convert_to_credit_score", False))
    )
    if convert_to_credit_score:
        if is_regression:
            raise ValueError("Credit-score conversion is only supported for classification")
        result = result.with_columns(
            pl.Series(
                args.credit_score_col or scoring.get("credit_score_col", "pred_score"),
                probability_to_credit_score(
                    predictions,
                    base_score=(
                        args.base_score
                        if args.base_score is not None
                        else scoring.get("base_score", 500.0)
                    ),
                    pdo=(
                        args.pdo
                        if args.pdo is not None
                        else scoring.get("pdo", 50.0)
                    ),
                    min_score=(
                        args.min_score
                        if args.min_score is not None
                        else scoring.get("min_score", 300.0)
                    ),
                    max_score=(
                        args.max_score
                        if args.max_score is not None
                        else scoring.get("max_score", 900.0)
                    ),
                ),
            )
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        result.write_parquet(output)
    else:
        result.write_csv(output)
    print(f"Saved {len(result)} {output_column} values to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
