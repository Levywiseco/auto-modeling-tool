#!/usr/bin/env python3
"""Score raw driver data with a saved scoring artifact."""

import argparse
from pathlib import Path
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_data
from src.modeling.artifact import load_scoring_artifact, score_with_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="scoring_artifact.pkl or pipeline.pkl")
    parser.add_argument("--input", required=True, help="Raw driver data")
    parser.add_argument("--output", required=True, help="CSV or Parquet output path")
    args = parser.parse_args()

    artifact = load_scoring_artifact(args.model)
    raw = load_data(args.input)
    probabilities = score_with_artifact(artifact, raw, return_proba=True)
    result = raw.with_columns(pl.Series("prediction", probabilities))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        result.write_parquet(output)
    else:
        result.write_csv(output)
    print(f"Saved {len(result)} scores to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
