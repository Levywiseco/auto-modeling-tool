# -*- coding: utf-8 -*-
"""
High-performance feature selection module using Polars.

This module provides efficient feature selection methods that leverage
Polars' vectorized operations for correlation analysis and integrate
with sklearn for model-based selection.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import polars as pl

from ..core.base import MarsTransformer
from ..core.logger import logger
from ..core.decorators import time_it
from ..core.exceptions import ValidationError


# =============================================================================
# Functional API
# =============================================================================

@time_it
def select_features(
    X: Union[pl.DataFrame, pl.LazyFrame],
    y: Union[pl.Series, np.ndarray],
    *,
    method: Literal["recursive", "correlation", "variance", "iv", "mutual_info"] = "recursive",
    n_features: Optional[int] = None,
    threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    iv_threshold: float = 0.02,
) -> List[str]:
    """
    Select features based on the specified method.
    
    Parameters
    ----------
    X : pl.DataFrame or pl.LazyFrame
        Feature dataset.
    y : pl.Series or np.ndarray
        Target variable.
    method : str, default "recursive"
        Feature selection method:
        - "recursive": Recursive Feature Elimination (RFE) with LogisticRegression
        - "correlation": Remove highly correlated features
        - "variance": Remove low-variance features
        - "iv": Select by Information Value (requires binary target)
        - "mutual_info": Mutual information based selection
    n_features : int, optional
        Number of features to select (for RFE and mutual_info).
        If None, uses threshold-based selection.
    threshold : float, default 0.01
        Threshold for variance-based selection.
    correlation_threshold : float, default 0.95
        Maximum allowed correlation between features.
    iv_threshold : float, default 0.02
        Minimum IV for feature selection.
        
    Returns
    -------
    list of str
        Names of selected features.
        
    Example
    -------
    >>> selected = select_features(X, y, method="correlation")
    >>> X_selected = X.select(selected)
    """
    # Materialize if LazyFrame
    if isinstance(X, pl.LazyFrame):
        X = X.collect()
    
    # Convert y to numpy if needed
    if isinstance(y, pl.Series):
        y = y.to_numpy()
    
    logger.info(f"🔍 Selecting features using '{method}' method")
    logger.info(f"   Input: {len(X.columns)} features, {len(X)} samples")
    
    if method == "recursive":
        selected = _select_rfe(X, y, n_features or 10)
    elif method == "correlation":
        selected = _select_by_correlation(X, correlation_threshold)
    elif method == "variance":
        selected = _select_by_variance(X, threshold)
    elif method == "iv":
        selected = _select_by_iv(X, y, iv_threshold)
    elif method == "mutual_info":
        selected = _select_mutual_info(X, y, n_features or 10)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"✅ Selected {len(selected)} features")
    
    return selected


def _select_rfe(
    X: pl.DataFrame, 
    y: np.ndarray, 
    n_features: int
) -> List[str]:
    """Recursive Feature Elimination using LogisticRegression."""
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    
    # Get numeric columns only
    numeric_cols = [
        c for c in X.columns
        if X[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
    ]
    
    if not numeric_cols:
        logger.warning("No numeric columns found for RFE")
        return []
    
    X_np = X.select(numeric_cols).to_numpy()
    
    # Handle NaN values
    X_np = np.nan_to_num(X_np, nan=0.0)
    
    model = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
    rfe = RFE(model, n_features_to_select=min(n_features, len(numeric_cols)))
    
    try:
        fit = rfe.fit(X_np, y)
        selected = [col for col, selected in zip(numeric_cols, fit.support_) if selected]
    except Exception as e:
        logger.warning(f"RFE failed: {e}. Returning all numeric features.")
        selected = numeric_cols
    
    return selected


@time_it
def _select_by_correlation(
    X: pl.DataFrame, 
    threshold: float
) -> List[str]:
    """
    Remove highly correlated features using Polars vectorized operations.
    
    Optimization: Uses Polars correlation matrix computation instead of
    pandas, which is faster for large datasets.
    """
    # Get numeric columns
    numeric_cols = [
        c for c in X.columns
        if X[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
    ]
    
    if len(numeric_cols) <= 1:
        return numeric_cols
    
    logger.info(f"   Computing correlation matrix for {len(numeric_cols)} features...")
    
    # Compute correlation matrix using Polars
    # For each pair, compute Pearson correlation
    corr_data = {}
    
    for col in numeric_cols:
        corr_data[col] = []
        for other_col in numeric_cols:
            if col == other_col:
                corr_data[col].append(1.0)
            else:
                corr = X.select(
                    pl.corr(col, other_col)
                ).item()
                corr_data[col].append(corr if corr is not None else 0.0)
    
    # Build correlation matrix as numpy array for easier processing
    corr_matrix = np.array([corr_data[c] for c in numeric_cols])
    
    # Find features to drop (upper triangle, excluding diagonal)
    to_drop = set()
    n = len(numeric_cols)
    
    for i in range(n):
        if numeric_cols[i] in to_drop:
            continue
        for j in range(i + 1, n):
            if numeric_cols[j] in to_drop:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                # Drop the feature with lower variance
                var_i = X.select(pl.col(numeric_cols[i]).var()).item() or 0
                var_j = X.select(pl.col(numeric_cols[j]).var()).item() or 0
                
                if var_i < var_j:
                    to_drop.add(numeric_cols[i])
                else:
                    to_drop.add(numeric_cols[j])
    
    selected = [c for c in numeric_cols if c not in to_drop]
    
    logger.info(f"   Removed {len(to_drop)} correlated features (threshold={threshold})")
    
    return selected


@time_it
def _select_by_variance(
    X: pl.DataFrame, 
    threshold: float
) -> List[str]:
    """Select features with variance above threshold."""
    # Get numeric columns
    numeric_cols = [
        c for c in X.columns
        if X[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
    ]
    
    if not numeric_cols:
        return []
    
    # Compute variance for all columns in single pass
    var_exprs = [pl.col(c).var().alias(c) for c in numeric_cols]
    variances = X.select(var_exprs).row(0)
    
    selected = [
        col for col, var in zip(numeric_cols, variances)
        if var is not None and var > threshold
    ]
    
    logger.info(f"   Removed {len(numeric_cols) - len(selected)} low-variance features")
    
    return selected


def _select_by_iv(
    X: pl.DataFrame, 
    y: np.ndarray, 
    threshold: float
) -> List[str]:
    """Select features by Information Value."""
    from ..binning.utils import binning_with_woe
    
    # Get numeric columns
    numeric_cols = [
        c for c in X.columns
        if X[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
    ]
    
    # Create temporary dataframe with target
    temp_df = X.select(numeric_cols).with_columns(pl.Series("target", y))
    
    iv_results = {}
    for col in numeric_cols:
        try:
            result = binning_with_woe(temp_df, "target", col, num_bins=10, method="quantile")
            iv_results[col] = result['total_iv']
        except Exception as e:
            logger.warning(f"IV calculation failed for {col}: {e}")
            iv_results[col] = 0.0
    
    selected = [col for col, iv in iv_results.items() if iv >= threshold]
    
    # Sort by IV descending
    selected.sort(key=lambda x: iv_results[x], reverse=True)
    
    logger.info(f"   Selected {len(selected)} features with IV >= {threshold}")
    
    return selected


def _select_mutual_info(
    X: pl.DataFrame, 
    y: np.ndarray, 
    n_features: int
) -> List[str]:
    """Select features by mutual information."""
    from sklearn.feature_selection import mutual_info_classif, SelectKBest
    
    # Get numeric columns
    numeric_cols = [
        c for c in X.columns
        if X[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
    ]
    
    if not numeric_cols:
        return []
    
    X_np = X.select(numeric_cols).to_numpy()
    X_np = np.nan_to_num(X_np, nan=0.0)
    
    selector = SelectKBest(mutual_info_classif, k=min(n_features, len(numeric_cols)))
    selector.fit(X_np, y)
    
    selected = [col for col, selected in zip(numeric_cols, selector.get_support()) if selected]
    
    return selected


# =============================================================================
# Class-based API
# =============================================================================

class FeatureSelector(MarsTransformer):
    """
    Feature selector following sklearn Transformer pattern.
    
    Learns which features to keep from training data and applies
    the same selection to new data.
    
    Parameters
    ----------
    method : str, default "correlation"
        Feature selection method.
    n_features : int, optional
        Number of features to select.
    correlation_threshold : float, default 0.95
        Maximum correlation between features.
    variance_threshold : float, default 0.01
        Minimum variance for features.
    iv_threshold : float, default 0.02
        Minimum IV for features.
        
    Attributes
    ----------
    selected_features_ : list of str
        Names of selected features after fit.
    feature_scores_ : dict
        Scores for each feature (if applicable).
        
    Example
    -------
    >>> selector = FeatureSelector(method="correlation")
    >>> selector.fit(X_train, y_train)
    >>> X_train_selected = selector.transform(X_train)
    >>> X_test_selected = selector.transform(X_test)
    """
    
    def __init__(
        self,
        method: Literal["recursive", "correlation", "variance", "iv", "mutual_info"] = "correlation",
        n_features: Optional[int] = None,
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.01,
        iv_threshold: float = 0.02,
    ):
        super().__init__()
        self.method = method
        self.n_features = n_features
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.iv_threshold = iv_threshold
        
        # Fitted attributes
        self.selected_features_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}
    
    @time_it
    def _fit_impl(self, X: pl.DataFrame, y: Optional[pl.Series] = None, **kwargs) -> None:
        """Learn which features to select."""
        if y is None and self.method in ["recursive", "iv", "mutual_info"]:
            raise ValidationError(f"Target 'y' is required for method '{self.method}'")
        
        y_np = y.to_numpy() if y is not None else None
        
        self.selected_features_ = select_features(
            X, y_np,
            method=self.method,
            n_features=self.n_features,
            threshold=self.variance_threshold,
            correlation_threshold=self.correlation_threshold,
            iv_threshold=self.iv_threshold,
        )
    
    def _transform_impl(self, X: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """Apply feature selection."""
        # Only select features that exist in X
        available = [f for f in self.selected_features_ if f in X.columns]
        
        if len(available) < len(self.selected_features_):
            missing = set(self.selected_features_) - set(available)
            logger.warning(f"Missing features in transform: {missing}")
        
        return X.select(available)
    
    def get_selected_features(self) -> List[str]:
        """Return list of selected feature names."""
        return self.selected_features_.copy()


@time_it
def remove_multicollinearity(
    X: Union[pl.DataFrame, pl.LazyFrame],
    *,
    threshold: float = 0.95,
    prefer_high_variance: bool = True,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Remove multicollinear features while preserving the most informative ones.
    
    Parameters
    ----------
    X : pl.DataFrame or pl.LazyFrame
        Feature dataset.
    threshold : float, default 0.95
        Maximum allowed correlation.
    prefer_high_variance : bool, default True
        When dropping correlated pairs, keep the one with higher variance.
        
    Returns
    -------
    tuple of (filtered_df, dropped_columns)
        Filtered DataFrame and list of dropped column names.
        
    Example
    -------
    >>> X_clean, dropped = remove_multicollinearity(X, threshold=0.9)
    >>> print(f"Dropped {len(dropped)} features")
    """
    if isinstance(X, pl.LazyFrame):
        X = X.collect()
    
    selected = _select_by_correlation(X, threshold)
    dropped = [c for c in X.columns if c not in selected]
    
    return X.select(selected), dropped