# -*- coding: utf-8 -*-
"""
High-performance feature generation module using Polars.

This module provides efficient feature generation functions that leverage
Polars' vectorized operations for common feature engineering tasks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from ..core.base import MarsTransformer
from ..core.logger import logger
from ..core.decorators import time_it


# =============================================================================
# Functional API
# =============================================================================

@time_it
def generate_polynomial_features(
    X: Union[pl.DataFrame, pl.LazyFrame, np.ndarray],
    *,
    degree: int = 2,
    include_bias: bool = False,
    interaction_only: bool = False,
    columns: Optional[List[str]] = None,
) -> Union[pl.DataFrame, np.ndarray]:
    """
    Generate polynomial features using sklearn.
    
    Parameters
    ----------
    X : pl.DataFrame, pl.LazyFrame, or np.ndarray
        Input features.
    degree : int, default 2
        Maximum polynomial degree.
    include_bias : bool, default False
        Include bias (intercept) column.
    interaction_only : bool, default False
        Only generate interaction terms.
    columns : list of str, optional
        Specific columns to use (for DataFrame input).
        
    Returns
    -------
    pl.DataFrame or np.ndarray
        Polynomial features (same type as input for numpy).
        
    Example
    -------
    >>> X_poly = generate_polynomial_features(X, degree=2)
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    logger.info(f"🔧 Generating polynomial features (degree={degree})")
    
    # Handle Polars input
    if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
        if isinstance(X, pl.LazyFrame):
            X = X.collect()
        
        # Select columns
        if columns is not None:
            X_subset = X.select(columns)
        else:
            # Only numeric columns
            numeric_cols = [
                c for c in X.columns
                if X[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
            ]
            X_subset = X.select(numeric_cols)
        
        X_np = X_subset.to_numpy()
        original_cols = X_subset.columns
        is_polars = True
    else:
        X_np = X
        is_polars = False
    
    # Generate polynomial features
    poly = PolynomialFeatures(
        degree=degree, 
        include_bias=include_bias, 
        interaction_only=interaction_only
    )
    X_poly = poly.fit_transform(X_np)
    
    if is_polars:
        # Create feature names
        feature_names = poly.get_feature_names_out(original_cols)
        result = pl.DataFrame(X_poly, schema=list(feature_names))
        logger.info(f"✅ Generated {len(feature_names)} polynomial features")
        return result
    else:
        logger.info(f"✅ Generated {X_poly.shape[1]} polynomial features")
        return X_poly


@time_it
def generate_interaction_features(
    X: Union[pl.DataFrame, pl.LazyFrame, np.ndarray],
    *,
    columns: Optional[List[str]] = None,
) -> Union[pl.DataFrame, np.ndarray]:
    """
    Generate interaction features (product of feature pairs).
    
    Parameters
    ----------
    X : pl.DataFrame, pl.LazyFrame, or np.ndarray
        Input features.
    columns : list of str, optional
        Specific columns to use.
        
    Returns
    -------
    pl.DataFrame or np.ndarray
        Interaction features.
        
    Example
    -------
    >>> X_interact = generate_interaction_features(X)
    """
    return generate_polynomial_features(
        X, 
        degree=2, 
        include_bias=False, 
        interaction_only=True,
        columns=columns
    )


@time_it
def generate_ratio_features(
    X: Union[pl.DataFrame, pl.LazyFrame],
    *,
    column_pairs: Optional[List[Tuple[str, str]]] = None,
    auto_detect: bool = True,
    epsilon: float = 1e-10,
) -> pl.DataFrame:
    """
    Generate ratio features between column pairs.
    
    Parameters
    ----------
    X : pl.DataFrame or pl.LazyFrame
        Input features.
    column_pairs : list of tuple, optional
        Specific (numerator, denominator) pairs.
    auto_detect : bool, default True
        Automatically generate ratios for all numeric column pairs.
    epsilon : float, default 1e-10
        Small value to avoid division by zero.
        
    Returns
    -------
    pl.DataFrame
        DataFrame with ratio features appended.
        
    Example
    -------
    >>> X_ratios = generate_ratio_features(X, column_pairs=[("income", "debt")])
    """
    if isinstance(X, pl.LazyFrame):
        X = X.collect()
    
    logger.info("🔧 Generating ratio features")
    
    # Get numeric columns
    numeric_cols = [
        c for c in X.columns
        if X[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
    ]
    
    # Determine column pairs
    if column_pairs is None and auto_detect:
        # Generate all pairs (limited to avoid explosion)
        if len(numeric_cols) > 10:
            logger.warning(f"Too many columns ({len(numeric_cols)}), limiting to first 10")
            numeric_cols = numeric_cols[:10]
        
        column_pairs = [
            (c1, c2) for i, c1 in enumerate(numeric_cols)
            for c2 in numeric_cols[i+1:]
        ]
    
    if not column_pairs:
        return X
    
    # Build ratio expressions
    ratio_exprs = []
    for num_col, denom_col in column_pairs:
        if num_col in X.columns and denom_col in X.columns:
            ratio_name = f"{num_col}_div_{denom_col}"
            ratio_expr = (
                pl.col(num_col).cast(pl.Float64) / 
                (pl.col(denom_col).cast(pl.Float64).abs() + epsilon)
            ).alias(ratio_name)
            ratio_exprs.append(ratio_expr)
    
    if ratio_exprs:
        X = X.with_columns(ratio_exprs)
        logger.info(f"✅ Generated {len(ratio_exprs)} ratio features")
    
    return X


@time_it
def generate_log_features(
    X: Union[pl.DataFrame, pl.LazyFrame],
    *,
    columns: Optional[List[str]] = None,
    base: str = "natural",
    handle_negative: bool = True,
) -> pl.DataFrame:
    """
    Generate logarithmic transformations of features.
    
    Parameters
    ----------
    X : pl.DataFrame or pl.LazyFrame
        Input features.
    columns : list of str, optional
        Columns to transform. If None, all positive numeric columns.
    base : str, default "natural"
        Log base: "natural", "log2", or "log10".
    handle_negative : bool, default True
        Apply log1p to handle zeros and sign-preserving log for negatives.
        
    Returns
    -------
    pl.DataFrame
        DataFrame with log features appended.
        
    Example
    -------
    >>> X_log = generate_log_features(X, columns=["income", "balance"])
    """
    if isinstance(X, pl.LazyFrame):
        X = X.collect()
    
    logger.info(f"🔧 Generating log features (base={base})")
    
    # Determine columns
    if columns is None:
        columns = [
            c for c in X.columns
            if X[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
        ]
    
    # Build log expressions
    log_exprs = []
    for col in columns:
        if col not in X.columns:
            continue
        
        if handle_negative:
            # Sign-preserving log transformation: sign(x) * log(1 + |x|)
            if base == "natural":
                log_expr = (
                    pl.col(col).sign() * 
                    (pl.col(col).abs() + 1).log()
                ).alias(f"{col}_log")
            elif base == "log2":
                log_expr = (
                    pl.col(col).sign() * 
                    (pl.col(col).abs() + 1).log() / np.log(2)
                ).alias(f"{col}_log2")
            elif base == "log10":
                log_expr = (
                    pl.col(col).sign() * 
                    (pl.col(col).abs() + 1).log() / np.log(10)
                ).alias(f"{col}_log10")
        else:
            # Standard log (only positive values)
            if base == "natural":
                log_expr = pl.col(col).log().alias(f"{col}_log")
            elif base == "log2":
                log_expr = (pl.col(col).log() / np.log(2)).alias(f"{col}_log2")
            elif base == "log10":
                log_expr = (pl.col(col).log() / np.log(10)).alias(f"{col}_log10")
        
        log_exprs.append(log_expr)
    
    if log_exprs:
        X = X.with_columns(log_exprs)
        logger.info(f"✅ Generated {len(log_exprs)} log features")
    
    return X


@time_it
def generate_binned_features(
    X: Union[pl.DataFrame, pl.LazyFrame],
    *,
    columns: Optional[List[str]] = None,
    n_bins: int = 10,
    method: str = "quantile",
) -> pl.DataFrame:
    """
    Generate binned versions of numeric features.
    
    Parameters
    ----------
    X : pl.DataFrame or pl.LazyFrame
        Input features.
    columns : list of str, optional
        Columns to bin.
    n_bins : int, default 10
        Number of bins.
    method : str, default "quantile"
        Binning method: "quantile" or "uniform".
        
    Returns
    -------
    pl.DataFrame
        DataFrame with binned features appended.
    """
    from ..binning.utils import calculate_bins, apply_binning
    
    if isinstance(X, pl.LazyFrame):
        X = X.collect()
    
    logger.info(f"🔧 Generating binned features (n_bins={n_bins})")
    
    # Determine columns
    if columns is None:
        columns = [
            c for c in X.columns
            if X[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
        ]
    
    for col in columns:
        if col not in X.columns:
            continue
        
        try:
            bin_edges = calculate_bins(X, col, n_bins, method=method)
            binned = apply_binning(X, col, bin_edges)
            X = X.with_columns(binned.alias(f"{col}_binned"))
        except Exception as e:
            logger.warning(f"Failed to bin {col}: {e}")
    
    return X


def generate_features(
    X: Union[pl.DataFrame, pl.LazyFrame, np.ndarray],
    *,
    polynomial: bool = True,
    polynomial_degree: int = 2,
    interactions: bool = False,
    ratios: bool = False,
    logs: bool = False,
) -> Union[pl.DataFrame, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate multiple types of features in one call.
    
    Parameters
    ----------
    X : pl.DataFrame, pl.LazyFrame, or np.ndarray
        Input features.
    polynomial : bool, default True
        Generate polynomial features.
    polynomial_degree : int, default 2
        Polynomial degree.
    interactions : bool, default False
        Generate interaction-only features.
    ratios : bool, default False
        Generate ratio features.
    logs : bool, default False
        Generate log features.
        
    Returns
    -------
    pl.DataFrame or tuple of np.ndarray
        Generated features.
        
    Example
    -------
    >>> X_gen = generate_features(X, polynomial=True, logs=True)
    """
    logger.info("🔧 Starting feature generation pipeline")
    
    results = []
    
    if polynomial:
        poly_features = generate_polynomial_features(
            X, degree=polynomial_degree, interaction_only=False
        )
        results.append(poly_features)
    
    if interactions and not polynomial:
        # Only if not already generating polynomial features
        interact_features = generate_interaction_features(X)
        results.append(interact_features)
    
    if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
        if isinstance(X, pl.LazyFrame):
            X = X.collect()
        
        if ratios:
            X = generate_ratio_features(X)
        
        if logs:
            X = generate_log_features(X)
        
        if results:
            # Merge polynomial features
            poly_df = results[0]
            # Keep only new columns
            new_cols = [c for c in poly_df.columns if c not in X.columns]
            if new_cols:
                X = pl.concat([X, poly_df.select(new_cols)], how="horizontal")
        
        return X
    else:
        # Return numpy arrays
        if len(results) == 2:
            return results[0], results[1]
        elif len(results) == 1:
            return results[0], None
        else:
            return X, None


# =============================================================================
# Class-based API
# =============================================================================

class FeatureGenerator(MarsTransformer):
    """
    Feature generator following sklearn Transformer pattern.
    
    Parameters
    ----------
    polynomial_degree : int, default 0
        Polynomial degree (0 = disabled).
    include_ratios : bool, default False
        Generate ratio features.
    include_logs : bool, default False
        Generate log features.
    include_bins : bool, default False
        Generate binned features.
        
    Example
    -------
    >>> generator = FeatureGenerator(polynomial_degree=2, include_logs=True)
    >>> generator.fit(X_train)
    >>> X_train_gen = generator.transform(X_train)
    """
    
    def __init__(
        self,
        polynomial_degree: int = 0,
        include_ratios: bool = False,
        include_logs: bool = False,
        include_bins: bool = False,
        n_bins: int = 10,
    ):
        super().__init__()
        self.polynomial_degree = polynomial_degree
        self.include_ratios = include_ratios
        self.include_logs = include_logs
        self.include_bins = include_bins
        self.n_bins = n_bins
        
        # Fitted attributes
        self.numeric_columns_: List[str] = []
        self._poly_transformer = None
    
    @time_it
    def _fit_impl(self, X: pl.DataFrame, y: Optional[pl.Series] = None, **kwargs) -> None:
        """Learn feature generation parameters."""
        self.numeric_columns_ = [
            c for c in X.columns
            if X[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
        ]
        
        if self.polynomial_degree > 1:
            from sklearn.preprocessing import PolynomialFeatures
            self._poly_transformer = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=False
            )
            X_np = X.select(self.numeric_columns_).to_numpy()
            self._poly_transformer.fit(X_np)
    
    def _transform_impl(self, X: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """Apply feature generation."""
        result = X
        
        if self.polynomial_degree > 1 and self._poly_transformer is not None:
            X_np = X.select(self.numeric_columns_).to_numpy()
            X_poly = self._poly_transformer.transform(X_np)
            feature_names = self._poly_transformer.get_feature_names_out(self.numeric_columns_)
            
            # Only add new columns
            new_cols = [n for n in feature_names if n not in X.columns]
            if new_cols:
                poly_df = pl.DataFrame(X_poly, schema=list(feature_names))
                result = pl.concat([result, poly_df.select(new_cols)], how="horizontal")
        
        if self.include_ratios:
            result = generate_ratio_features(result, columns=self.numeric_columns_)
        
        if self.include_logs:
            result = generate_log_features(result, columns=self.numeric_columns_)
        
        if self.include_bins:
            result = generate_binned_features(result, columns=self.numeric_columns_, n_bins=self.n_bins)
        
        return result