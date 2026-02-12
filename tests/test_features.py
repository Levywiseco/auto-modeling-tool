# -*- coding: utf-8 -*-
"""
Tests for feature selection module.
"""

import pytest
import polars as pl
import numpy as np

from src.features.selection import (
    select_features,
    FeatureSelector,
    remove_multicollinearity,
)


class TestFeatureSelection:
    """Test cases for feature selection functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 200
        
        X = pl.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "feature3": np.random.randn(n),
            "feature4": np.random.randn(n) * 0.01,
            "feature5": np.random.randn(n),
        })
        
        X = X.with_columns(
            (pl.col("feature1") * 0.95 + pl.col("feature2") * 0.05).alias("feature_corr")
        )
        
        y = (X["feature1"] + X["feature2"] > 0).cast(pl.Int64)
        
        return X, y
    
    def test_select_by_correlation(self, sample_data):
        """Test correlation-based feature selection."""
        X, y = sample_data
        
        selected = select_features(X, y.to_numpy(), method="correlation", correlation_threshold=0.9)
        
        assert isinstance(selected, list)
        assert len(selected) > 0
    
    def test_select_by_variance(self, sample_data):
        """Test variance-based feature selection."""
        X, y = sample_data
        
        selected = select_features(X, y.to_numpy(), method="variance", threshold=0.001)
        
        assert isinstance(selected, list)
        assert "feature4" not in selected
    
    def test_select_by_iv(self, sample_data):
        """Test IV-based feature selection."""
        X, y = sample_data
        
        selected = select_features(X, y.to_numpy(), method="iv", iv_threshold=0.0)
        
        assert isinstance(selected, list)
    
    def test_select_by_mutual_info(self, sample_data):
        """Test mutual information-based feature selection."""
        X, y = sample_data
        
        selected = select_features(X, y.to_numpy(), method="mutual_info", n_features=3)
        
        assert isinstance(selected, list)
        assert len(selected) == 3


class TestFeatureSelector:
    """Test cases for FeatureSelector class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 200
        
        X = pl.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "feature3": np.random.randn(n),
            "feature4": np.random.randn(n),
        })
        
        y = pl.Series("target", (X["feature1"] + X["feature2"] > 0).cast(pl.Int64))
        
        return X, y
    
    def test_init(self):
        """Test FeatureSelector initialization."""
        selector = FeatureSelector(method="correlation")
        
        assert selector.method == "correlation"
        assert not selector._is_fitted
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        X, y = sample_data
        
        selector = FeatureSelector(method="correlation")
        result = selector.fit_transform(X, y)
        
        assert selector._is_fitted
        assert isinstance(result, pl.DataFrame)
    
    def test_transform(self, sample_data):
        """Test transform method."""
        X, y = sample_data
        
        selector = FeatureSelector(method="variance", variance_threshold=0.001)
        selector.fit(X)
        
        result = selector.transform(X)
        
        assert isinstance(result, pl.DataFrame)
    
    def test_get_selected_features(self, sample_data):
        """Test getting selected features."""
        X, y = sample_data
        
        selector = FeatureSelector(method="variance", variance_threshold=0.001)
        selector.fit(X)
        
        selected = selector.get_selected_features()
        
        assert isinstance(selected, list)


class TestRemoveMulticollinearity:
    """Test cases for remove_multicollinearity function."""
    
    def test_remove_multicollinearity(self):
        """Test removing multicollinear features."""
        np.random.seed(42)
        n = 100
        
        X = pl.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
        })
        
        X = X.with_columns(
            (pl.col("feature1") * 0.99).alias("feature1_copy")
        )
        
        result, dropped = remove_multicollinearity(X, threshold=0.95)
        
        assert len(dropped) > 0
        assert len(result.columns) < len(X.columns)
