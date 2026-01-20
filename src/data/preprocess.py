# -*- coding: utf-8 -*-
"""
High-performance data preprocessing module using Polars.

This module provides fast data cleaning and transformation functions
that leverage Polars' vectorized operations for optimal performance.
"""

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import polars as pl

from ..core.base import MarsTransformer
from ..core.logger import logger
from ..core.decorators import time_it


# =============================================================================
# Functional API (Stateless Functions)
# =============================================================================

@time_it
def clean_data(
    df: Union[pl.DataFrame, pl.LazyFrame],
    *,
    fill_strategy: Literal["forward", "backward", "mean", "median", "zero", "drop"] = "forward",
    fill_value: Optional[Any] = None,
    custom_null_values: Optional[List[Any]] = None,
) -> pl.DataFrame:
    """
    Clean data by handling missing values.
    
    Uses Polars' optimized fill operations for high performance.
    
    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data.
    fill_strategy : str, default "forward"
        Strategy for filling missing values:
        - "forward": Forward fill (use previous value)
        - "backward": Backward fill (use next value)
        - "mean": Fill with column mean (numeric only)
        - "median": Fill with column median (numeric only)
        - "zero": Fill with 0 (numeric) or "" (string)
        - "drop": Drop rows with any missing values
    fill_value : Any, optional
        Custom value to fill. Overrides fill_strategy.
    custom_null_values : list, optional
        Additional values to treat as null (e.g., [-999, "unknown"]).
        
    Returns
    -------
    pl.DataFrame
        Cleaned data.
        
    Example
    -------
    >>> df = clean_data(df, fill_strategy="mean")
    >>> df = clean_data(df, fill_value=0)
    >>> df = clean_data(df, custom_null_values=[-999, "unknown"])
    """
    # Materialize if LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    logger.info(f"🧹 Cleaning data: {len(df)} rows × {len(df.columns)} columns")
    
    # Handle custom null values
    if custom_null_values:
        for val in custom_null_values:
            df = df.with_columns([
                pl.when(pl.col(c) == val)
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in df.columns
            ])
    
    # Apply fill strategy
    if fill_value is not None:
        df = df.fill_null(fill_value)
    elif fill_strategy == "forward":
        df = df.fill_null(strategy="forward")
    elif fill_strategy == "backward":
        df = df.fill_null(strategy="backward")
    elif fill_strategy == "mean":
        # Only fill numeric columns with mean
        numeric_cols = [
            c for c in df.columns 
            if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
        ]
        if numeric_cols:
            df = df.with_columns([
                pl.col(c).fill_null(pl.col(c).mean()) for c in numeric_cols
            ])
    elif fill_strategy == "median":
        numeric_cols = [
            c for c in df.columns 
            if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
        ]
        if numeric_cols:
            df = df.with_columns([
                pl.col(c).fill_null(pl.col(c).median()) for c in numeric_cols
            ])
    elif fill_strategy == "zero":
        df = df.fill_null(0).fill_null("")
    elif fill_strategy == "drop":
        df = df.drop_nulls()
    
    logger.info(f"✅ Cleaned: {len(df)} rows remaining")
    return df


@time_it
def normalize_data(
    df: Union[pl.DataFrame, pl.LazyFrame],
    *,
    method: Literal["minmax", "zscore", "robust"] = "minmax",
    columns: Optional[List[str]] = None,
    clip_outliers: bool = False,
    outlier_percentile: float = 0.01,
) -> pl.DataFrame:
    """
    Normalize numerical features using various scaling methods.
    
    Uses Polars expressions for vectorized computation.
    
    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data.
    method : str, default "minmax"
        Normalization method:
        - "minmax": Scale to [0, 1] range
        - "zscore": Standardize to mean=0, std=1
        - "robust": Use median and IQR (robust to outliers)
    columns : list of str, optional
        Columns to normalize. If None, all numeric columns.
    clip_outliers : bool, default False
        Whether to clip outliers before normalization.
    outlier_percentile : float, default 0.01
        Percentile for outlier clipping (e.g., 0.01 clips at 1% and 99%).
        
    Returns
    -------
    pl.DataFrame
        Normalized data.
        
    Example
    -------
    >>> df = normalize_data(df, method="zscore")
    >>> df = normalize_data(df, method="minmax", columns=["age", "income"])
    """
    # Materialize if LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    # Determine columns to normalize
    if columns is None:
        columns = [
            c for c in df.columns 
            if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
        ]
    
    if not columns:
        logger.warning("No numeric columns found to normalize")
        return df
    
    logger.info(f"📊 Normalizing {len(columns)} columns using {method} method")
    
    # Build normalization expressions
    norm_exprs = []
    
    for col in columns:
        col_expr = pl.col(col).cast(pl.Float64)
        
        # Optionally clip outliers first
        if clip_outliers:
            lower = df.select(pl.col(col).quantile(outlier_percentile)).item()
            upper = df.select(pl.col(col).quantile(1 - outlier_percentile)).item()
            col_expr = col_expr.clip(lower, upper)
        
        if method == "minmax":
            # (x - min) / (max - min)
            norm_expr = (
                (col_expr - col_expr.min()) / 
                (col_expr.max() - col_expr.min() + 1e-10)
            ).alias(col)
        elif method == "zscore":
            # (x - mean) / std
            norm_expr = (
                (col_expr - col_expr.mean()) / 
                (col_expr.std() + 1e-10)
            ).alias(col)
        elif method == "robust":
            # (x - median) / IQR
            q25 = df.select(pl.col(col).quantile(0.25)).item()
            q75 = df.select(pl.col(col).quantile(0.75)).item()
            iqr = q75 - q25 if q75 != q25 else 1.0
            median = df.select(pl.col(col).median()).item()
            norm_expr = ((col_expr - median) / iqr).alias(col)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        norm_exprs.append(norm_expr)
    
    df = df.with_columns(norm_exprs)
    logger.info(f"✅ Normalization complete")
    
    return df


@time_it
def preprocess_data(
    df: Union[pl.DataFrame, pl.LazyFrame],
    *,
    clean_strategy: str = "forward",
    normalize_method: Optional[str] = "minmax",
    custom_null_values: Optional[List[Any]] = None,
    drop_columns: Optional[List[str]] = None,
    keep_columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Applies cleaning and normalization in sequence.
    
    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data.
    clean_strategy : str, default "forward"
        Strategy for missing value handling.
    normalize_method : str or None, default "minmax"
        Normalization method. Set to None to skip normalization.
    custom_null_values : list, optional
        Values to treat as null.
    drop_columns : list of str, optional
        Columns to drop.
    keep_columns : list of str, optional
        Columns to keep (others will be dropped).
        
    Returns
    -------
    pl.DataFrame
        Preprocessed data.
        
    Example
    -------
    >>> df = preprocess_data(df, clean_strategy="mean", normalize_method="zscore")
    """
    # Materialize if LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    logger.info(f"🔧 Starting preprocessing pipeline")
    
    # Column selection
    if keep_columns is not None:
        df = df.select(keep_columns)
    if drop_columns is not None:
        df = df.drop(drop_columns)
    
    # Clean data
    df = clean_data(
        df, 
        fill_strategy=clean_strategy,
        custom_null_values=custom_null_values
    )
    
    # Normalize data
    if normalize_method is not None:
        df = normalize_data(df, method=normalize_method)
    
    logger.info(f"✅ Preprocessing complete: {len(df)} rows × {len(df.columns)} columns")
    
    return df


# =============================================================================
# Class-based API (Stateful Transformer)
# =============================================================================

class DataPreprocessor(MarsTransformer):
    """
    Stateful data preprocessor following sklearn Transformer pattern.
    
    Learns statistics from training data and applies them to new data.
    
    Parameters
    ----------
    clean_strategy : str, default "mean"
        Strategy for missing value handling.
    normalize_method : str or None, default "minmax"
        Normalization method.
    custom_null_values : list, optional
        Values to treat as null.
        
    Attributes
    ----------
    stats_ : dict
        Learned statistics from fit (mean, std, min, max per column).
        
    Example
    -------
    >>> preprocessor = DataPreprocessor(normalize_method="zscore")
    >>> preprocessor.fit(X_train)
    >>> X_train_processed = preprocessor.transform(X_train)
    >>> X_test_processed = preprocessor.transform(X_test)
    """
    
    def __init__(
        self,
        clean_strategy: str = "mean",
        normalize_method: Optional[str] = "minmax",
        custom_null_values: Optional[List[Any]] = None,
    ):
        super().__init__()
        self.clean_strategy = clean_strategy
        self.normalize_method = normalize_method
        self.custom_null_values = custom_null_values or []
        
        # Learned statistics
        self.stats_: Dict[str, Dict[str, float]] = {}
        self.numeric_columns_: List[str] = []
    
    @time_it
    def _fit_impl(self, X: pl.DataFrame, y: Optional[pl.Series] = None, **kwargs) -> None:
        """Learn statistics from training data."""
        logger.info(f"📊 Fitting preprocessor on {len(X)} rows")
        
        # Identify numeric columns
        self.numeric_columns_ = [
            c for c in X.columns 
            if X[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
        ]
        
        # Calculate statistics for each numeric column
        for col in self.numeric_columns_:
            series = X[col]
            self.stats_[col] = {
                "mean": series.mean(),
                "std": series.std() or 1.0,
                "min": series.min(),
                "max": series.max(),
                "median": series.median(),
                "q25": series.quantile(0.25),
                "q75": series.quantile(0.75),
            }
        
        logger.info(f"✅ Learned statistics for {len(self.numeric_columns_)} numeric columns")
    
    @time_it
    def _transform_impl(self, X: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """Apply learned transformation to data."""
        # Handle custom null values
        if self.custom_null_values:
            for val in self.custom_null_values:
                X = X.with_columns([
                    pl.when(pl.col(c) == val)
                    .then(None)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in X.columns
                ])
        
        # Fill missing values
        fill_exprs = []
        for col in self.numeric_columns_:
            if col not in X.columns:
                continue
            stats = self.stats_.get(col, {})
            
            if self.clean_strategy == "mean":
                fill_val = stats.get("mean", 0)
            elif self.clean_strategy == "median":
                fill_val = stats.get("median", 0)
            else:
                fill_val = 0
            
            fill_exprs.append(pl.col(col).fill_null(fill_val))
        
        if fill_exprs:
            X = X.with_columns(fill_exprs)
        
        # Normalize
        if self.normalize_method:
            norm_exprs = []
            for col in self.numeric_columns_:
                if col not in X.columns:
                    continue
                stats = self.stats_[col]
                
                if self.normalize_method == "minmax":
                    min_val, max_val = stats["min"], stats["max"]
                    norm_expr = (
                        (pl.col(col).cast(pl.Float64) - min_val) / 
                        (max_val - min_val + 1e-10)
                    ).alias(col)
                elif self.normalize_method == "zscore":
                    mean_val, std_val = stats["mean"], stats["std"]
                    norm_expr = (
                        (pl.col(col).cast(pl.Float64) - mean_val) / 
                        (std_val + 1e-10)
                    ).alias(col)
                elif self.normalize_method == "robust":
                    median_val = stats["median"]
                    iqr = stats["q75"] - stats["q25"]
                    iqr = iqr if iqr > 0 else 1.0
                    norm_expr = (
                        (pl.col(col).cast(pl.Float64) - median_val) / iqr
                    ).alias(col)
                else:
                    continue
                
                norm_exprs.append(norm_expr)
            
            if norm_exprs:
                X = X.with_columns(norm_exprs)
        
        return X