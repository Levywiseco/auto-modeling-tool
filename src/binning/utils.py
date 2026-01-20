# -*- coding: utf-8 -*-
"""
Binning utility functions using Polars for high-performance operations.

This module provides standalone functions for binning operations that can be
used independently or as building blocks for more complex binning strategies.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl

from ..core.logger import logger
from ..core.decorators import time_it


@time_it
def calculate_bins(
    data: Union[pl.DataFrame, pl.LazyFrame],
    column: str,
    num_bins: int,
    *,
    method: str = "uniform",
    exclude_values: Optional[List[Any]] = None,
) -> List[float]:
    """
    Calculate bin edges for a given column using various methods.
    
    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Input data.
    column : str
        Column name to compute bins for.
    num_bins : int
        Number of bins to create.
    method : str, default "uniform"
        Binning method: "uniform" (equal-width) or "quantile" (equal-frequency).
    exclude_values : list, optional
        Values to exclude from bin calculation (e.g., missing indicators).
        
    Returns
    -------
    list of float
        Bin edges including -inf and inf: [-inf, cut1, cut2, ..., inf]
        
    Example
    -------
    >>> edges = calculate_bins(df, "age", 5, method="quantile")
    >>> edges
    [-inf, 25.0, 35.0, 45.0, 55.0, inf]
    """
    # Materialize if LazyFrame
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    
    # Build filter expression
    filter_expr = pl.col(column).is_not_null()
    
    # Check for float columns and handle NaN
    if data[column].dtype in [pl.Float32, pl.Float64]:
        filter_expr = filter_expr & (~pl.col(column).is_nan())
    
    # Exclude special values
    if exclude_values:
        # Filter compatible types
        dtype = data[column].dtype
        safe_exclude = []
        for v in exclude_values:
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64] and isinstance(v, (int, float)):
                safe_exclude.append(v)
            elif dtype in [pl.Float32, pl.Float64] and isinstance(v, (int, float)):
                safe_exclude.append(v)
            elif dtype in [pl.Utf8, pl.String] and isinstance(v, str):
                safe_exclude.append(v)
        
        if safe_exclude:
            filter_expr = filter_expr & (~pl.col(column).is_in(safe_exclude))
    
    # Get filtered data
    filtered = data.filter(filter_expr)
    
    if len(filtered) == 0:
        logger.warning(f"Column '{column}' has no valid values after filtering")
        return [float('-inf'), float('inf')]
    
    if method == "uniform":
        # Equal-width binning
        min_val = filtered.select(pl.col(column).min()).item()
        max_val = filtered.select(pl.col(column).max()).item()
        
        if min_val == max_val:
            return [float('-inf'), float('inf')]
        
        step = (max_val - min_val) / num_bins
        cuts = [min_val + step * i for i in range(1, num_bins)]
        
    elif method == "quantile":
        # Equal-frequency binning
        quantiles = np.linspace(0, 1, num_bins + 1)[1:-1].tolist()
        
        q_exprs = [
            pl.col(column).quantile(q).alias(f"q{i}")
            for i, q in enumerate(quantiles)
        ]
        
        q_result = filtered.select(q_exprs).row(0)
        cuts = [float(v) for v in q_result if v is not None]
        cuts = sorted(set(cuts))  # Remove duplicates
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'uniform' or 'quantile'.")
    
    return [float('-inf')] + cuts + [float('inf')]


@time_it
def apply_binning(
    data: Union[pl.DataFrame, pl.LazyFrame],
    column: str,
    bin_edges: List[float],
    *,
    missing_bin: int = -1,
    special_values: Optional[Dict[Any, int]] = None,
) -> pl.Series:
    """
    Apply binning to a column using provided bin edges.
    
    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Input data.
    column : str
        Column name to bin.
    bin_edges : list of float
        Bin edges including -inf and inf.
    missing_bin : int, default -1
        Bin index for missing values.
    special_values : dict, optional
        Mapping of special values to bin indices, e.g., {-999: -3, -1: -4}.
        
    Returns
    -------
    pl.Series
        Bin indices as Int16 series.
        
    Example
    -------
    >>> bins = apply_binning(df, "age", [-inf, 25, 50, inf])
    >>> bins
    Series: 'age_bin' [i16]
    [0, 1, 0, 2, ...]
    """
    # Materialize if LazyFrame
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    
    # Build binning expression
    result_expr = pl.when(pl.col(column).is_null()).then(pl.lit(missing_bin))
    
    # Handle NaN for float columns
    if data[column].dtype in [pl.Float32, pl.Float64]:
        result_expr = result_expr.when(pl.col(column).is_nan()).then(pl.lit(missing_bin))
    
    # Handle special values
    if special_values:
        for val, bin_idx in special_values.items():
            result_expr = result_expr.when(pl.col(column) == val).then(pl.lit(bin_idx))
    
    # Apply bin edges (excluding -inf and inf)
    inner_edges = bin_edges[1:-1]
    for i, edge in enumerate(inner_edges):
        result_expr = result_expr.when(pl.col(column) < edge).then(pl.lit(i))
    
    # Last bin
    result_expr = result_expr.otherwise(pl.lit(len(inner_edges)))
    
    return data.select(result_expr.cast(pl.Int16).alias(f"{column}_bin")).get_column(f"{column}_bin")


@time_it
def calculate_woe(
    data: Union[pl.DataFrame, pl.LazyFrame],
    target: str,
    bin_column: str,
    *,
    smooth: float = 0.5,
) -> pl.DataFrame:
    """
    Calculate WOE (Weight of Evidence) values for a binned column.
    
    Uses Polars vectorized operations for optimal performance.
    
    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Input data containing bin column and target.
    target : str
        Name of the binary target column (1 = event, 0 = non-event).
    bin_column : str
        Name of the column containing bin indices.
    smooth : float, default 0.5
        Smoothing factor to avoid division by zero and log(0).
        
    Returns
    -------
    pl.DataFrame
        DataFrame with columns: bin_idx, count, bad, good, bad_rate, woe, iv
        
    Example
    -------
    >>> woe_df = calculate_woe(df, "target", "age_bin")
    >>> woe_df
    shape: (5, 7)
    ┌─────────┬───────┬─────┬──────┬──────────┬────────┬────────┐
    │ bin_idx ┆ count ┆ bad ┆ good ┆ bad_rate ┆ woe    ┆ iv     │
    │ ---     ┆ ---   ┆ --- ┆ ---  ┆ ---      ┆ ---    ┆ ---    │
    │ i16     ┆ u32   ┆ i64 ┆ i64  ┆ f64      ┆ f64    ┆ f64    │
    ╞═════════╪═══════╪═════╪══════╪══════════╪════════╪════════╡
    │ 0       ┆ 1000  ┆ 100 ┆ 900  ┆ 0.1      ┆ -0.693 ┆ 0.035  │
    └─────────┴───────┴─────┴──────┴──────────┴────────┴────────┘
    """
    # Materialize if LazyFrame
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    
    # Calculate totals
    total_bad = data.select(pl.col(target).sum()).item()
    total_good = len(data) - total_bad
    
    if total_bad == 0 or total_good == 0:
        logger.warning("Target has only one class, WOE calculation may be unreliable")
        total_bad = max(total_bad, 1)
        total_good = max(total_good, 1)
    
    # Aggregate by bin
    result = (
        data.group_by(bin_column)
        .agg([
            pl.count().alias("count"),
            pl.col(target).sum().alias("bad"),
        ])
        .with_columns([
            (pl.col("count") - pl.col("bad")).alias("good")
        ])
        .with_columns([
            (pl.col("bad") / pl.col("count")).alias("bad_rate"),
            # Distribution with smoothing
            ((pl.col("bad") + smooth) / (total_bad + smooth * 2)).alias("dist_bad"),
            ((pl.col("good") + smooth) / (total_good + smooth * 2)).alias("dist_good"),
        ])
        .with_columns([
            (pl.col("dist_bad") / pl.col("dist_good")).log().alias("woe")
        ])
        .with_columns([
            ((pl.col("dist_bad") - pl.col("dist_good")) * pl.col("woe")).alias("iv")
        ])
        .sort(bin_column)
        .rename({bin_column: "bin_idx"})
        .select(["bin_idx", "count", "bad", "good", "bad_rate", "woe", "iv"])
    )
    
    return result


@time_it
def binning_with_woe(
    data: Union[pl.DataFrame, pl.LazyFrame],
    target: str,
    column: str,
    num_bins: int,
    *,
    method: str = "quantile",
    exclude_values: Optional[List[Any]] = None,
    special_values: Optional[Dict[Any, int]] = None,
) -> Dict[str, Any]:
    """
    Complete binning pipeline: calculate bins, apply, and compute WOE.
    
    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Input data.
    target : str
        Binary target column name.
    column : str
        Feature column to bin.
    num_bins : int
        Number of bins.
    method : str, default "quantile"
        Binning method: "uniform" or "quantile".
    exclude_values : list, optional
        Values to exclude from bin calculation.
    special_values : dict, optional
        Mapping of special values to bin indices.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'bin_edges': List of bin edges
        - 'woe_table': DataFrame with WOE statistics
        - 'total_iv': Total Information Value
        - 'binned_column': Name of the binned column
        
    Example
    -------
    >>> result = binning_with_woe(df, "target", "age", 5)
    >>> result['total_iv']
    0.156
    >>> result['woe_table']
    shape: (5, 7)
    ...
    """
    # Materialize if LazyFrame
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    
    logger.info(f"🔧 Binning column '{column}' with {num_bins} bins")
    
    # Step 1: Calculate bin edges
    bin_edges = calculate_bins(
        data, column, num_bins, 
        method=method, 
        exclude_values=exclude_values
    )
    
    # Step 2: Apply binning
    bin_column = f"{column}_bin"
    binned_series = apply_binning(
        data, column, bin_edges, 
        special_values=special_values
    )
    
    # Add binned column to data
    data_with_bins = data.with_columns(binned_series)
    
    # Step 3: Calculate WOE
    woe_table = calculate_woe(data_with_bins, target, bin_column)
    
    # Calculate total IV
    total_iv = woe_table.select(pl.col("iv").sum()).item()
    
    logger.info(f"✅ Binning complete. Total IV: {total_iv:.4f}")
    
    return {
        'bin_edges': bin_edges,
        'woe_table': woe_table,
        'total_iv': total_iv,
        'binned_column': bin_column,
    }


def interpret_iv(iv: float) -> str:
    """
    Interpret Information Value for predictive power.
    
    Parameters
    ----------
    iv : float
        Information Value.
        
    Returns
    -------
    str
        Interpretation category.
    """
    if iv < 0.02:
        return "Not Useful"
    elif iv < 0.1:
        return "Weak Predictor"
    elif iv < 0.3:
        return "Medium Predictor"
    elif iv < 0.5:
        return "Strong Predictor"
    else:
        return "Suspicious (Too Good)"


@time_it
def calculate_psi(
    expected: pl.Series,
    actual: pl.Series,
    *,
    smooth: float = 0.0001,
) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.
    
    Parameters
    ----------
    expected : pl.Series
        Expected (baseline) binned values.
    actual : pl.Series
        Actual (current) binned values.
    smooth : float, default 0.0001
        Smoothing factor to avoid log(0).
        
    Returns
    -------
    float
        PSI value. Interpretation:
        - < 0.1: No significant change
        - 0.1 - 0.25: Moderate change
        - > 0.25: Significant change
        
    Example
    -------
    >>> psi = calculate_psi(train_bins, test_bins)
    >>> print(f"PSI: {psi:.4f}")
    PSI: 0.0823
    """
    # Get bin distributions
    expected_dist = (
        expected.value_counts()
        .with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("pct")
        )
    )
    
    actual_dist = (
        actual.value_counts()
        .with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("pct")
        )
    )
    
    # Join distributions
    col_name = expected.name
    combined = (
        expected_dist.select([pl.col(col_name), pl.col("pct").alias("expected")])
        .join(
            actual_dist.select([pl.col(col_name), pl.col("pct").alias("actual")]),
            on=col_name,
            how="outer"
        )
        .fill_null(smooth)
        .with_columns([
            pl.col("expected").clip(smooth, 1.0),
            pl.col("actual").clip(smooth, 1.0),
        ])
    )
    
    # Calculate PSI
    psi = combined.select(
        ((pl.col("actual") - pl.col("expected")) * 
         (pl.col("actual") / pl.col("expected")).log()).sum()
    ).item()
    
    return float(psi)