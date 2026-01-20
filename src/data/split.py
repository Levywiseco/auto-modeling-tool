# -*- coding: utf-8 -*-
"""
High-performance data splitting module using Polars.

This module provides fast train/test splitting functions that work
natively with Polars DataFrames while maintaining sklearn compatibility.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split as sk_train_test_split

from ..core.logger import logger
from ..core.decorators import time_it


@time_it
def train_test_split(
    data: Union[pl.DataFrame, pl.LazyFrame],
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """
    Split dataset into training and testing sets.
    
    Optimized for Polars with zero-copy operations where possible.
    
    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Complete dataset including features and target.
    target_column : str
        Name of the target column.
    test_size : float, default 0.2
        Proportion of dataset to include in test split.
    random_state : int, optional
        Random seed for reproducibility.
    shuffle : bool, default True
        Whether to shuffle data before splitting.
        
    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
        - X_train: Training features (pl.DataFrame)
        - X_test: Test features (pl.DataFrame)
        - y_train: Training target (pl.Series)
        - y_test: Test target (pl.Series)
        
    Example
    -------
    >>> X_train, X_test, y_train, y_test = train_test_split(df, "target")
    """
    # Materialize if LazyFrame
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    
    logger.info(f"🔀 Splitting data: {len(data)} rows, test_size={test_size}")
    
    n_samples = len(data)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    # Generate indices
    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Split using Polars native operations
    # Add row index for splitting
    data_with_idx = data.with_row_index("__idx__")
    
    train_data = data_with_idx.filter(pl.col("__idx__").is_in(train_indices)).drop("__idx__")
    test_data = data_with_idx.filter(pl.col("__idx__").is_in(test_indices)).drop("__idx__")
    
    # Separate features and target
    feature_columns = [c for c in data.columns if c != target_column]
    
    X_train = train_data.select(feature_columns)
    X_test = test_data.select(feature_columns)
    y_train = train_data.get_column(target_column)
    y_test = test_data.get_column(target_column)
    
    logger.info(f"✅ Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    
    return X_train, X_test, y_train, y_test


@time_it
def stratified_train_test_split(
    data: Union[pl.DataFrame, pl.LazyFrame],
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """
    Stratified split preserving class distribution in target.
    
    Uses sklearn's stratified splitting under the hood while
    maintaining Polars DataFrames as output.
    
    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Complete dataset including features and target.
    target_column : str
        Name of the target column.
    test_size : float, default 0.2
        Proportion of dataset to include in test split.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
        All outputs are Polars DataFrames/Series.
        
    Example
    -------
    >>> X_train, X_test, y_train, y_test = stratified_train_test_split(df, "target")
    """
    # Materialize if LazyFrame
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    
    logger.info(f"🔀 Stratified split: {len(data)} rows, test_size={test_size}")
    
    # Separate features and target
    feature_columns = [c for c in data.columns if c != target_column]
    
    # Use row indices for splitting
    indices = np.arange(len(data))
    y = data.get_column(target_column).to_numpy()
    
    # Stratified split of indices
    train_idx, test_idx = sk_train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Apply splits using Polars
    data_with_idx = data.with_row_index("__idx__")
    
    train_data = data_with_idx.filter(pl.col("__idx__").is_in(train_idx)).drop("__idx__")
    test_data = data_with_idx.filter(pl.col("__idx__").is_in(test_idx)).drop("__idx__")
    
    X_train = train_data.select(feature_columns)
    X_test = test_data.select(feature_columns)
    y_train = train_data.get_column(target_column)
    y_test = test_data.get_column(target_column)
    
    # Log class distribution
    train_dist = y_train.value_counts().sort("count", descending=True)
    test_dist = y_test.value_counts().sort("count", descending=True)
    logger.info(f"✅ Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    logger.debug(f"   Train distribution: {train_dist.to_dict()}")
    logger.debug(f"   Test distribution: {test_dist.to_dict()}")
    
    return X_train, X_test, y_train, y_test


@time_it
def time_series_split(
    data: Union[pl.DataFrame, pl.LazyFrame],
    target_column: str,
    date_column: str,
    *,
    test_size: float = 0.2,
    gap: int = 0,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """
    Time-based split for time series data.
    
    Ensures training data comes before test data chronologically.
    
    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Complete dataset including features, target, and date column.
    target_column : str
        Name of the target column.
    date_column : str
        Name of the date column for ordering.
    test_size : float, default 0.2
        Proportion of dataset to include in test split.
    gap : int, default 0
        Number of rows to skip between train and test (to avoid data leakage).
        
    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
        All outputs are Polars DataFrames/Series.
        
    Example
    -------
    >>> X_train, X_test, y_train, y_test = time_series_split(df, "target", "date")
    """
    # Materialize if LazyFrame
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    
    logger.info(f"📅 Time-based split: {len(data)} rows, gap={gap}")
    
    # Sort by date
    data = data.sort(date_column)
    
    n_samples = len(data)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test - gap
    
    # Split indices
    train_data = data.head(n_train)
    test_data = data.tail(n_test)
    
    # Separate features and target
    feature_columns = [c for c in data.columns if c not in [target_column, date_column]]
    
    X_train = train_data.select(feature_columns)
    X_test = test_data.select(feature_columns)
    y_train = train_data.get_column(target_column)
    y_test = test_data.get_column(target_column)
    
    logger.info(f"✅ Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    
    return X_train, X_test, y_train, y_test


@time_it
def kfold_split(
    data: Union[pl.DataFrame, pl.LazyFrame],
    target_column: str,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> List[Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]]:
    """
    K-Fold cross-validation splits.
    
    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Complete dataset.
    target_column : str
        Name of the target column.
    n_splits : int, default 5
        Number of folds.
    shuffle : bool, default True
        Whether to shuffle before splitting.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    list of tuples
        Each tuple contains (X_train, X_val, y_train, y_val) for one fold.
        
    Example
    -------
    >>> folds = kfold_split(df, "target", n_splits=5)
    >>> for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(folds):
    ...     model.fit(X_train, y_train)
    """
    from sklearn.model_selection import KFold
    
    # Materialize if LazyFrame
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    
    logger.info(f"📊 Creating {n_splits}-fold splits")
    
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    indices = np.arange(len(data))
    
    feature_columns = [c for c in data.columns if c != target_column]
    data_with_idx = data.with_row_index("__idx__")
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_data = data_with_idx.filter(pl.col("__idx__").is_in(train_idx)).drop("__idx__")
        val_data = data_with_idx.filter(pl.col("__idx__").is_in(val_idx)).drop("__idx__")
        
        X_train = train_data.select(feature_columns)
        X_val = val_data.select(feature_columns)
        y_train = train_data.get_column(target_column)
        y_val = val_data.get_column(target_column)
        
        folds.append((X_train, X_val, y_train, y_val))
        logger.debug(f"   Fold {fold_idx + 1}: Train={len(X_train)}, Val={len(X_val)}")
    
    logger.info(f"✅ Created {n_splits} folds")
    
    return folds