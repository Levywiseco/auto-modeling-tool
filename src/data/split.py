"""
High-performance data splitting module using Polars.

This module provides fast train/test splitting functions that work
natively with Polars DataFrames while maintaining sklearn compatibility.
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split as sk_train_test_split

from ..core.decorators import time_it
from ..core.logger import logger


@time_it
def train_test_split(
    data: Union[pl.DataFrame, pl.LazyFrame],
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
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
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
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
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
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
) -> list[tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]]:
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


@dataclass(frozen=True)
class DatasetSplit:
    """A validated development/OOT split used by the modeling pipeline."""

    dev: pl.DataFrame
    oot: pl.DataFrame
    strategy: str
    sample_column: Optional[str] = None
    date_column: Optional[str] = None

    def validate(self, target_column: str) -> "DatasetSplit":
        if target_column not in self.dev.columns or target_column not in self.oot.columns:
            raise ValueError(
                f"Target column '{target_column}' must exist in both dev and oot data"
            )
        if len(self.dev) == 0 or len(self.oot) == 0:
            raise ValueError("Dev and OOT samples must both contain at least one row")
        return self


@time_it
def split_dev_oot(
    data: Union[pl.DataFrame, pl.LazyFrame],
    target_column: str,
    *,
    sample_column: Optional[str] = None,
    dev_label: Any = "dev",
    oot_label: Any = "oot",
    date_column: Optional[str] = None,
    oot_start: Optional[Any] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> DatasetSplit:
    """Create an explicit Dev/OOT split, with a compatibility fallback.

    The preferred modes are an existing sample label column or a chronological
    date boundary. When neither is supplied, the legacy stratified random split
    is retained as an explicit compatibility fallback and is labeled random.
    """
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    if sample_column and date_column:
        raise ValueError("Use either sample_column or date_column, not both")
    if sample_column and sample_column not in data.columns:
        raise ValueError(f"Sample column '{sample_column}' not found in data")
    if date_column and date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' not found in data")

    if sample_column:
        available = set(data.get_column(sample_column).drop_nulls().to_list())
        if dev_label not in available or oot_label not in available:
            if {0, 1}.issubset(available) and dev_label == "dev" and oot_label == "oot":
                dev_label, oot_label = 1, 0
            else:
                raise ValueError(
                    f"Sample column '{sample_column}' must contain both "
                    f"{dev_label!r} and {oot_label!r}; found {sorted(available, key=str)!r}"
                )

        result = DatasetSplit(
            dev=data.filter(pl.col(sample_column) == dev_label),
            oot=data.filter(pl.col(sample_column) == oot_label),
            strategy="sample_column",
            sample_column=sample_column,
        )
        return result.validate(target_column)

    if date_column:
        if oot_start is None:
            raise ValueError("oot_start is required when date_column is supplied")
        result = DatasetSplit(
            dev=data.filter(pl.col(date_column) < oot_start),
            oot=data.filter(pl.col(date_column) >= oot_start),
            strategy="date",
            date_column=date_column,
        )
        return result.validate(target_column)

    logger.warning(
        "No sample/date split supplied; using stratified random split as a "
        "backward-compatible fallback. Prefer explicit Dev/OOT labels or dates."
    )
    X_dev, X_oot, y_dev, y_oot = stratified_train_test_split(
        data,
        target_column,
        test_size=test_size,
        random_state=random_state,
    )
    result = DatasetSplit(
        dev=X_dev.with_columns(y_dev.alias(target_column)),
        oot=X_oot.with_columns(y_oot.alias(target_column)),
        strategy="random",
    )
    return result.validate(target_column)
