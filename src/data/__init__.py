# -*- coding: utf-8 -*-
"""Data loading, preprocessing, and splitting module."""

from .loaders import (
    load_csv,
    load_excel,
    load_parquet,
    load_sql,
    load_data,
)
from .preprocess import (
    clean_data,
    normalize_data,
    preprocess_data,
    DataPreprocessor,
)
from .split import (
    train_test_split,
    stratified_train_test_split,
    time_series_split,
    kfold_split,
)

__all__ = [
    # Loaders
    "load_csv",
    "load_excel",
    "load_parquet",
    "load_sql",
    "load_data",
    # Preprocessing
    "clean_data",
    "normalize_data",
    "preprocess_data",
    "DataPreprocessor",
    # Splitting
    "train_test_split",
    "stratified_train_test_split",
    "time_series_split",
    "kfold_split",
]
