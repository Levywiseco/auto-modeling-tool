# -*- coding: utf-8 -*-
"""Frame-level helpers shared by workflow entry points."""

from typing import Any, List, Optional

import polars as pl

from ..core.exceptions import DataTypeError, ValidationError

NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}


def ensure_polars(df: Any) -> pl.DataFrame:
    """Accept a Pandas or Polars frame and return a Polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
    except ImportError:
        pass
    if isinstance(df, dict):
        return pl.DataFrame(df)
    raise DataTypeError(
        f"Unsupported input type: {type(df)}. "
        "Expected pandas.DataFrame, polars.DataFrame/LazyFrame, or dict."
    )


def resolve_features(
    df: pl.DataFrame,
    features: Optional[List[str]],
    exclude: List[Optional[str]],
    numeric_only: bool = False,
) -> List[str]:
    """Resolve the feature list, defaulting to all (numeric) non-role columns."""
    excluded = {c for c in exclude if c}
    if features is None:
        cols = [c for c in df.columns if c not in excluded]
        if numeric_only:
            cols = [c for c in cols if df.schema[c] in NUMERIC_DTYPES]
        return cols

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValidationError(f"Features not found in dataframe: {missing}")
    return [c for c in features if c not in excluded]


def group_keys_sorted(df: pl.DataFrame, group_col: str) -> List[Any]:
    """Distinct group values in ascending order (time-like groups sort naturally)."""
    return df[group_col].unique().sort().to_list()
