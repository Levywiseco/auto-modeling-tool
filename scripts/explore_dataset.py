#!/usr/bin/env python3
"""Explore a dataset and optionally produce Dev/OOT IV and PSI audit tables."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_modeling_tool.binning.woe_binning import WoeBinner
from auto_modeling_tool.data.loaders import load_data
from auto_modeling_tool.data.split import split_dev_oot
from auto_modeling_tool.evaluation.stability import psi_from_distributions
from auto_modeling_tool.reports.excel import write_model_report

NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}


def _missing_rate(series: pl.Series) -> float:
    missing = series.is_null()
    if series.dtype in {pl.Float32, pl.Float64}:
        missing = missing | series.is_nan()
    return float(missing.sum() / max(len(series), 1))


def _weighted_mean(
    series: pl.Series,
    weight: Optional[pl.Series] = None,
) -> float:
    """Return an unweighted or sample-weighted mean for a numeric series."""
    values = series.cast(pl.Float64).to_numpy()
    if len(values) == 0:
        return float("nan")
    if weight is None:
        return float(np.mean(values))
    weights = weight.cast(pl.Float64).to_numpy()
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        return float("nan")
    return float(np.average(values, weights=weights))


def _weighted_distribution(
    series: pl.Series,
    weight: Optional[pl.Series] = None,
) -> dict[int, float]:
    weights = (
        np.ones(len(series), dtype=float)
        if weight is None
        else weight.to_numpy().astype(float)
    )
    total = float(weights.sum())
    result: dict[int, float] = {}
    for value, current_weight in zip(series.to_list(), weights):
        if value is None:
            continue
        key = int(value)
        result[key] = result.get(key, 0.0) + float(current_weight)
    return {key: value / total for key, value in result.items()} if total else {}


def _numeric_stats(frame: pl.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for column in columns:
        series = frame[column]
        rows.append({
            "column": column,
            "n": len(series),
            "n_valid": len(series) - series.null_count(),
            "missing_rate": _missing_rate(series),
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "p01": series.quantile(0.01),
            "p25": series.quantile(0.25),
            "p50": series.quantile(0.50),
            "p75": series.quantile(0.75),
            "p99": series.quantile(0.99),
        })
    return rows


def _categorical_stats(frame: pl.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for column in columns:
        counts = (
            frame.select(pl.col(column))
            .with_columns(
                pl.col(column).cast(pl.String).fill_null("(missing)").alias(column)
            )
            .to_series()
            .value_counts()
            .sort("count", descending=True)
            .head(20)
        )
        rows.extend({
            "column": column,
            "value": row[column],
            "count": row["count"],
        } for row in counts.to_dicts())
    return rows


def _feature_tables(
    dev: pl.DataFrame,
    oot: pl.DataFrame,
    *,
    target_col: str,
    feature_columns: list[str],
    weight_col: Optional[str],
    use_sample_weight: bool,
    n_bins: int,
    export_woe_detail: bool,
) -> dict[str, Any]:
    dev_weight = (
        dev[weight_col].cast(pl.Float64)
        if use_sample_weight and weight_col
        else None
    )
    oot_weight = (
        oot[weight_col].cast(pl.Float64)
        if use_sample_weight and weight_col
        else None
    )
    binner = WoeBinner(n_bins=n_bins, min_samples_bin=1)
    binner.fit(
        dev.select(feature_columns),
        dev[target_col],
        sample_weight=dev_weight,
    )
    dev_index = binner.transform(dev.select(feature_columns), return_type="index")
    oot_index = binner.transform(oot.select(feature_columns), return_type="index")

    iv_dev = binner.get_iv_report()
    oot_stats = binner.compute_bin_stats(
        oot.select(feature_columns),
        oot[target_col],
        sample_weight=oot_weight,
    )
    iv_oot = (
        oot_stats.group_by("feature")
        .agg(pl.col("iv").sum().alias("total_iv"))
        .sort("total_iv", descending=True)
        if len(oot_stats)
        else pl.DataFrame({"feature": [], "total_iv": []})
    )

    psi_rows = []
    for feature in feature_columns:
        if f"{feature}_bin" not in dev_index.columns:
            continue
        expected = _weighted_distribution(
            dev_index[f"{feature}_bin"], dev_weight
        )
        actual = _weighted_distribution(
            oot_index[f"{feature}_bin"], oot_weight
        )
        psi_rows.append({
            "feature": feature,
            "psi": psi_from_distributions(expected, actual),
            "dtype": str(dev[feature].dtype),
        })

    tables: dict[str, Any] = {
        "7_IV_DEV": iv_dev,
        "8_IV_OOT": iv_oot,
        "9_PSI_Numeric": [
            row for row in psi_rows if dev[row["feature"]].dtype in NUMERIC_DTYPES
        ],
        "10_PSI_Categorical": [
            row for row in psi_rows if dev[row["feature"]].dtype not in NUMERIC_DTYPES
        ],
    }
    if export_woe_detail:
        tables["WOE_Detail"] = [
            {
                "feature": feature,
                "bin_idx": bin_idx,
                "label": binner.bin_mappings_.get(feature, {}).get(bin_idx),
                "woe": woe,
                "iv": binner.bin_ivs_.get(feature, {}).get(bin_idx),
            }
            for feature, mapping in binner.bin_woes_.items()
            for bin_idx, woe in mapping.items()
        ]
    return tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input dataset")
    parser.add_argument("--target", help="Binary target for IV/PSI")
    parser.add_argument("--output", default="eda_report.json")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--sample-column", "--split-column")
    parser.add_argument("--dev-label", default="dev")
    parser.add_argument("--oot-label", default="oot")
    parser.add_argument("--weight-column")
    parser.add_argument("--use-sample-weight", action="store_true")
    parser.add_argument("--exclude-column", action="append", default=[])
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--export-woe-detail", action="store_true")
    args = parser.parse_args()

    path = Path(args.input)
    load_kwargs = {"encoding": args.encoding} if path.suffix.lower() == ".csv" else {}
    frame = load_data(path, **load_kwargs)
    if args.target and args.target not in frame.columns:
        raise ValueError(f"Target column '{args.target}' not found")
    if args.use_sample_weight:
        if not args.weight_column or args.weight_column not in frame.columns:
            raise ValueError("--use-sample-weight requires an existing --weight-column")
        weights = frame[args.weight_column].cast(pl.Float64).to_numpy()
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Weights must be finite and strictly positive")

    role_columns = set(args.exclude_column)
    if args.target:
        role_columns.add(args.target)
    if args.sample_column:
        role_columns.add(args.sample_column)
    if args.weight_column:
        role_columns.add(args.weight_column)
    feature_columns = [column for column in frame.columns if column not in role_columns]
    numeric_columns = [
        column for column in feature_columns if frame[column].dtype in NUMERIC_DTYPES
    ]
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]

    missing = [
        {
            "column": column,
            "dtype": str(frame[column].dtype),
            "rows": len(frame),
            "null_count": frame[column].null_count(),
            "missing_rate": _missing_rate(frame[column]),
            "n_unique": frame[column].n_unique(),
        }
        for column in frame.columns
    ]
    overview: dict[str, Any] = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "numeric_features": len(numeric_columns),
        "categorical_features": len(categorical_columns),
        "column_names": frame.columns,
    }
    tables: dict[str, Any] = {
        "1_Overview": overview,
        "2_Missing_Rate": missing,
        "3_Num_Dist_DEV": _numeric_stats(frame, numeric_columns),
        "5_Char_Dist_DEV": _categorical_stats(frame, categorical_columns),
    }

    if args.target and args.sample_column:
        split = split_dev_oot(
            frame,
            args.target,
            sample_column=args.sample_column,
            dev_label=args.dev_label,
            oot_label=args.oot_label,
        )
        dev, oot = split.dev, split.oot
        tables["4_Num_Dist_OOT"] = _numeric_stats(oot, numeric_columns)
        tables["6_Char_Dist_OOT"] = _categorical_stats(oot, categorical_columns)
        tables["2_Missing_Rate"] = [
            {
                "column": column,
                "dev_missing_rate": _missing_rate(dev[column]),
                "oot_missing_rate": _missing_rate(oot[column]),
                "delta": _missing_rate(oot[column]) - _missing_rate(dev[column]),
            }
            for column in feature_columns
        ]
        tables.update(
            _feature_tables(
                dev,
                oot,
                target_col=args.target,
                feature_columns=feature_columns,
                weight_col=args.weight_column,
                use_sample_weight=args.use_sample_weight,
                n_bins=args.n_bins,
                export_woe_detail=args.export_woe_detail,
            )
        )
        dev_weight = (
            dev[args.weight_column].cast(pl.Float64)
            if args.use_sample_weight and args.weight_column
            else None
        )
        oot_weight = (
            oot[args.weight_column].cast(pl.Float64)
            if args.use_sample_weight and args.weight_column
            else None
        )
        overview.update({
            "dev_rows": len(dev),
            "oot_rows": len(oot),
            "dev_bad_rate": _weighted_mean(dev[args.target], dev_weight),
            "oot_bad_rate": _weighted_mean(oot[args.target], oot_weight),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".xlsx":
        write_model_report(
            output.parent,
            {},
            metadata=overview,
            tables=tables,
            filename=output.name,
        )
    else:
        serializable = {
            key: value.to_dicts() if isinstance(value, pl.DataFrame) else value
            for key, value in tables.items()
        }
        output.write_text(
            json.dumps(
                {"overview": overview, "tables": serializable},
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

