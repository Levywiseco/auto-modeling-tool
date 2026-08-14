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


    def test_categorical_mapping_and_unseen_values(self):
        """Categorical levels are mapped deterministically and safely."""
        X = pl.DataFrame({
            "category": ["A", "A", "A", "B", "B", "B", "C", "C", None, "RARE"],
            "numeric": list(range(10)),
        })
        y = pl.Series("target", [0, 1, 0, 0, 1, 1, 1, 1, 0, 0])

        binner = WoeBinner(n_bins=3, min_samples_bin=2)
        binner.fit(X, y)

        scored = binner.transform(
            pl.DataFrame({
                "category": ["A", "UNSEEN", None, "C"],
                "numeric": [1, 20, 3, 8],
            }),
            return_type="index",
        )

        assert "category" in binner.category_mappings_
        assert scored["category_bin"].to_list()[1] == WoeBinner.IDX_OTHER
        assert scored["category_bin"].to_list()[2] == WoeBinner.IDX_MISSING
        assert "category" in binner.total_iv_

    def test_monotonic_numeric_woe(self):
        """Monotonic=True produces ordered WOE values for numeric bins."""
        X = pl.DataFrame({"feature": list(range(200))})
        y = pl.Series("target", [int(value >= 100) for value in range(200)])

        binner = WoeBinner(
            n_bins=5,
            min_samples_bin=5,
            monotonic=True,
        )
        binner.fit(X, y)

        numeric_woes = binner.bin_woes_["feature"]
        ordered = [
            numeric_woes[index]
            for index in sorted(index for index in numeric_woes if index >= 0)
        ]
        assert ordered == sorted(ordered)


    def test_weighted_woe_and_bin_stats(self):
        """Weights affect WOE/IV and are preserved in bin diagnostics."""
        X = pl.DataFrame({"feature": [0.0, 0.1, 0.9, 1.0]})
        y = pl.Series("target", [0, 1, 0, 1])
        weights = pl.Series("weight", [1.0, 1.0, 5.0, 5.0])

        weighted = WoeBinner(n_bins=2, min_samples_bin=1)
        weighted.fit(X, y, sample_weight=weights)
        stats = weighted.compute_bin_stats(X, y, sample_weight=weights)

        assert weighted.total_iv_["feature"] >= 0
        assert stats["count"].sum() == pytest.approx(weights.sum())
