# -*- coding: utf-8 -*-
"""
Tests for WOE binning module.
"""

import pytest
import polars as pl
import numpy as np

from src.binning.woe_binning import WoeBinner


class TestWoeBinner:
    """Test cases for WoeBinner class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 1000
        
        df = pl.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n) * 2,
            "feature3": np.random.exponential(1, n),
            "target": np.random.binomial(1, 0.2, n),
        })
        
        return df
    
    def test_init(self):
        """Test WoeBinner initialization."""
        binner = WoeBinner(n_bins=10, method="quantile")
        
        assert binner.n_bins == 10
        assert binner.method == "quantile"
        assert not binner._is_fitted
    
    def test_fit_quantile(self, sample_data):
        """Test fitting with quantile method."""
        binner = WoeBinner(n_bins=5, method="quantile")
        
        X = sample_data.drop("target")
        y = sample_data["target"]
        
        binner.fit(X, y)
        
        assert binner._is_fitted
        assert len(binner.bin_cuts_) == 3
        assert len(binner.total_iv_) == 3
    
    def test_fit_uniform(self, sample_data):
        """Test fitting with uniform method."""
        binner = WoeBinner(n_bins=5, method="uniform")
        
        X = sample_data.drop("target")
        y = sample_data["target"]
        
        binner.fit(X, y)
        
        assert binner._is_fitted
        assert len(binner.bin_cuts_) == 3
    
    def test_fit_cart(self, sample_data):
        """Test fitting with CART method."""
        binner = WoeBinner(n_bins=5, method="cart")
        
        X = sample_data.drop("target")
        y = sample_data["target"]
        
        binner.fit(X, y)
        
        assert binner._is_fitted
    
    def test_transform_index(self, sample_data):
        """Test transform with return_type='index'."""
        binner = WoeBinner(n_bins=5, method="quantile")
        
        X = sample_data.drop("target")
        y = sample_data["target"]
        
        binner.fit(X, y)
        result = binner.transform(X, return_type="index")
        
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == X.shape[0]
    
    def test_transform_woe(self, sample_data):
        """Test transform with return_type='woe'."""
        binner = WoeBinner(n_bins=5, method="quantile")
        
        X = sample_data.drop("target")
        y = sample_data["target"]
        
        binner.fit(X, y)
        result = binner.transform(X, return_type="woe")
        
        assert isinstance(result, pl.DataFrame)
    
    def test_get_iv_report(self, sample_data):
        """Test IV report generation."""
        binner = WoeBinner(n_bins=5, method="quantile")
        
        X = sample_data.drop("target")
        y = sample_data["target"]
        
        binner.fit(X, y)
        report = binner.get_iv_report()
        
        assert isinstance(report, pl.DataFrame)
        assert "feature" in report.columns
        assert "total_iv" in report.columns
        assert "interpretation" in report.columns
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        binner = WoeBinner(n_bins=5, method="quantile")
        
        X = sample_data.drop("target")
        y = sample_data["target"]
        
        result = binner.fit_transform(X, y, return_type="index")
        
        assert binner._is_fitted
        assert isinstance(result, pl.DataFrame)
    
    def test_special_values(self, sample_data):
        """Test handling of special values."""
        binner = WoeBinner(
            n_bins=5,
            method="quantile",
            special_values=[-999],
        )
        
        X = sample_data.drop("target").clone()
        X = X.with_columns(
            pl.when(pl.col("feature1") < 0)
            .then(-999)
            .otherwise(pl.col("feature1"))
            .alias("feature1")
        )
        y = sample_data["target"]
        
        binner.fit(X, y)
        result = binner.transform(X, return_type="index")
        
        assert isinstance(result, pl.DataFrame)
    
    def test_missing_values(self, sample_data):
        """Test handling of missing values."""
        binner = WoeBinner(n_bins=5, method="quantile")
        
        X = sample_data.drop("target").clone()
        X = X.with_columns(
            pl.when(pl.col("feature1") < 0)
            .then(None)
            .otherwise(pl.col("feature1"))
            .alias("feature1")
        )
        y = sample_data["target"]
        
        binner.fit(X, y)
        result = binner.transform(X, return_type="index")
        
        assert isinstance(result, pl.DataFrame)
