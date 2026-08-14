#!/usr/bin/env python3
"""Run lightweight independent EDA without fitting a model."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_data
from src.reports.excel import write_model_report


NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input dataset")
    parser.add_argument("--output", default="eda_report.json")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    path = Path(args.input)
    load_kwargs = {"encoding": args.encoding} if path.suffix.lower() == ".csv" else {}
    frame = load_data(path, **load_kwargs)

    missing = [
        {
            "column": column,
            "dtype": str(frame[column].dtype),
            "rows": len(frame),
            "null_count": frame[column].null_count(),
            "missing_rate": frame[column].null_count() / max(len(frame), 1),
            "n_unique": frame[column].n_unique(),
        }
        for column in frame.columns
    ]
    numeric = []
    categorical = []
    for column in frame.columns:
        series = frame[column]
        if series.dtype in NUMERIC_DTYPES:
            numeric.append(
                {
                    "column": column,
                    "min": series.min(),
                    "max": series.max(),
                    "mean": series.mean(),
                    "median": series.median(),
                    "std": series.std(),
                }
            )
        else:
            top = series.value_counts().sort("count", descending=True).head(10)
            categorical.extend(
                {
                    "column": column,
                    "value": row.get(column),
                    "count": row["count"],
                }
                for row in top.to_dicts()
            )

    overview: Dict[str, Any] = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": frame.columns,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".xlsx":
        write_model_report(
            output.parent,
            {},
            metadata=overview,
            tables={
                "Missing_Rate": missing,
                "Numeric_Stats": numeric,
                "Categorical_Top": categorical,
            },
            filename=output.name,
        )
    else:
        output.write_text(
            json.dumps(
                {
                    "overview": overview,
                    "missing_rate": missing,
                    "numeric_stats": numeric,
                    "categorical_top": categorical,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    print(f"Saved EDA report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
