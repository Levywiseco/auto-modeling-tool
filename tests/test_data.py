# -*- coding: utf-8 -*-
"""
Tests for data loading and preprocessing modules.
"""

import pytest
import polars as pl
import numpy as np
import tempfile
from pathlib import Path

from src.data.preprocess import DataPreprocessor, clean_data, normalize_data
from src.data.split import train_test_split, stratified_train_test_split


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 100
        
        df = pl.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n) * 2 + 5,
            "feature3": np.random.exponential(1, n),
            "category": ["A", "B", "C"] * 33 + ["A"],
        })
        
        df = df.with_columns(
            pl.when(pl.col("feature1") > 1)
            .then(None)
            .otherwise(pl.col("feature1"))
            .alias("feature1")
        )
        
        return df
    
    def test_init(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor(
            clean_strategy="median",
            normalize_method="zscore",
        )
        
        assert preprocessor.clean_strategy == "median"
        assert preprocessor.normalize_method == "zscore"
    
    def test_fit(self, sample_data):
        """Test fitting preprocessor."""
        preprocessor = DataPreprocessor(
            clean_strategy="mean",
            normalize_method="zscore",
        )
        
        preprocessor.fit(sample_data)
        
        assert preprocessor._is_fitted
        assert len(preprocessor.stats_) == 3
        assert len(preprocessor.numeric_columns_) == 3
    
    def test_transform(self, sample_data):
        """Test transforming data."""
        preprocessor = DataPreprocessor(
            clean_strategy="mean",
            normalize_method="zscore",
        )
        
        preprocessor.fit(sample_data)
        result = preprocessor.transform(sample_data)
        
        assert isinstance(result, pl.DataFrame)
        assert result.shape == sample_data.shape
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        preprocessor = DataPreprocessor(
            clean_strategy="median",
            normalize_method="minmax",
        )
        
        result = preprocessor.fit_transform(sample_data)
        
        assert preprocessor._is_fitted
        assert isinstance(result, pl.DataFrame)
    
    def test_normalize_minmax(self, sample_data):
        """Test min-max normalization."""
        preprocessor = DataPreprocessor(
            clean_strategy="mean",
            normalize_method="minmax",
        )
        
        result = preprocessor.fit_transform(sample_data)
        
        numeric_cols = ["feature1", "feature2", "feature3"]
        for col in numeric_cols:
            col_data = result[col].drop_nulls()
            assert col_data.min() >= -0.01
            assert col_data.max() <= 1.01
    
    def test_normalize_zscore(self, sample_data):
        """Test z-score normalization."""
        preprocessor = DataPreprocessor(
            clean_strategy="mean",
            normalize_method="zscore",
        )
        
        result = preprocessor.fit_transform(sample_data)
        
        numeric_cols = ["feature1", "feature2", "feature3"]
        for col in numeric_cols:
            col_data = result[col].drop_nulls()
            assert abs(col_data.mean()) < 0.1


class TestCleanData:
    """Test cases for clean_data function."""
    
    def test_clean_forward_fill(self):
        """Test forward fill strategy."""
        df = pl.DataFrame({
            "a": [1, None, 3, None, 5],
            "b": [1, 2, None, 4, None],
        })
        
        result = clean_data(df, fill_strategy="forward")
        
        assert result["a"].null_count() == 0
        assert result["b"].null_count() == 0
    
    def test_clean_mean_fill(self):
        """Test mean fill strategy."""
        df = pl.DataFrame({
            "a": [1.0, None, 3.0, None, 5.0],
        })
        
        result = clean_data(df, fill_strategy="mean")
        
        assert result["a"].null_count() == 0
        assert result["a"].mean() == 3.0
    
    def test_clean_drop(self):
        """Test drop strategy."""
        df = pl.DataFrame({
            "a": [1, None, 3],
            "b": [None, 2, None],
        })
        
        result = clean_data(df, fill_strategy="drop")
        
        assert len(result) == 0


class TestDataSplit:
    """Test cases for data splitting functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 200
        
        df = pl.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.binomial(1, 0.3, n),
        })
        
        return df
    
    def test_train_test_split(self, sample_data):
        """Test basic train/test split."""
        X_train, X_test, y_train, y_test = train_test_split(
            sample_data,
            target_column="target",
            test_size=0.2,
            random_state=42,
        )
        
        assert len(X_train) == 160
        assert len(X_test) == 40
        assert len(y_train) == 160
        assert len(y_test) == 40
    
    def test_stratified_split(self, sample_data):
        """Test stratified split."""
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            sample_data,
            target_column="target",
            test_size=0.2,
            random_state=42,
        )
        
        train_pos_rate = y_train.mean()
        test_pos_rate = y_test.mean()
        
        assert abs(train_pos_rate - test_pos_rate) < 0.05
    
    def test_split_returns_polars(self, sample_data):
        """Test that split returns Polars types."""
        X_train, X_test, y_train, y_test = train_test_split(
            sample_data,
            target_column="target",
            test_size=0.2,
        )
        
        assert isinstance(X_train, pl.DataFrame)
        assert isinstance(X_test, pl.DataFrame)
        assert isinstance(y_train, pl.Series)
        assert isinstance(y_test, pl.Series)
