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


NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64
}


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
    if isinstance(X, pl.LazyFrame):
        X = X.collect()
    
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


def _get_numeric_columns(X: pl.DataFrame) -> List[str]:
    """Get numeric column names."""
    return [c for c in X.columns if X[c].dtype in NUMERIC_DTYPES]


def _select_rfe(
    X: pl.DataFrame, 
    y: np.ndarray, 
    n_features: int
) -> List[str]:
    """Recursive Feature Elimination using LogisticRegression."""
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    
    numeric_cols = _get_numeric_columns(X)
    
    if not numeric_cols:
        logger.warning("No numeric columns found for RFE")
        return []
    
    X_np = X.select(numeric_cols).to_numpy()
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
    Remove highly correlated features using optimized numpy computation.
    
    Optimization: Uses numpy's corrcoef for O(n) matrix computation
    instead of O(n²) pairwise correlation calculations.
    """
    numeric_cols = _get_numeric_columns(X)
    
    if len(numeric_cols) <= 1:
        return numeric_cols
    
    logger.info(f"   Computing correlation matrix for {len(numeric_cols)} features...")
    
    X_np = X.select(numeric_cols).to_numpy()
    X_np = np.nan_to_num(X_np, nan=0.0)
    
    corr_matrix = np.corrcoef(X_np.T)
    
    variances = np.var(X_np, axis=0)
    
    to_drop = set()
    n = len(numeric_cols)
    
    for i in range(n):
        if numeric_cols[i] in to_drop:
            continue
        for j in range(i + 1, n):
            if numeric_cols[j] in to_drop:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                if variances[i] < variances[j]:
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
    numeric_cols = _get_numeric_columns(X)
    
    if not numeric_cols:
        return []
    
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
    from ..binning.woe_binning import WoeBinner
    
    numeric_cols = _get_numeric_columns(X)
    
    if not numeric_cols:
        return []
    
    y_series = pl.Series("target", y)
    
    binner = WoeBinner(n_bins=10, method="quantile")
    binner.fit(X.select(numeric_cols), y_series)
    
    iv_results = binner.total_iv_
    
    selected = [col for col in numeric_cols if iv_results.get(col, 0) >= threshold]
    selected.sort(key=lambda x: iv_results.get(x, 0), reverse=True)
    
    logger.info(f"   Selected {len(selected)} features with IV >= {threshold}")
    
    return selected


def _select_mutual_info(
    X: pl.DataFrame, 
    y: np.ndarray, 
    n_features: int
) -> List[str]:
    """Select features by mutual information."""
    from sklearn.feature_selection import mutual_info_classif, SelectKBest
    
    numeric_cols = _get_numeric_columns(X)
    
    if not numeric_cols:
        return []
    
    X_np = X.select(numeric_cols).to_numpy()
    X_np = np.nan_to_num(X_np, nan=0.0)
    
    selector = SelectKBest(mutual_info_classif, k=min(n_features, len(numeric_cols)))
    selector.fit(X_np, y)
    
    selected = [col for col, selected in zip(numeric_cols, selector.get_support()) if selected]
    
    return selected


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
