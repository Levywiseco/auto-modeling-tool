# -*- coding: utf-8 -*-
"""
Tests for evaluation metrics module.
"""

import pytest
import polars as pl
import numpy as np

from src.evaluation.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    calculate_ks,
    calculate_gini,
    calculate_psi,
    calculate_all_metrics,
)


class TestBasicMetrics:
    """Test cases for basic metrics."""
    
    def test_accuracy(self):
        """Test accuracy calculation."""
        y_true = [1, 0, 1, 1, 0, 1]
        y_pred = [1, 0, 0, 1, 0, 1]
        
        acc = accuracy(y_true, y_pred)
        
        assert acc == 5 / 6
    
    def test_precision(self):
        """Test precision calculation."""
        y_true = [1, 0, 1, 1, 0, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        
        prec = precision(y_true, y_pred)
        
        assert prec == 2 / 3
    
    def test_recall(self):
        """Test recall calculation."""
        y_true = [1, 0, 1, 1, 0, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        
        rec = recall(y_true, y_pred)
        
        assert rec == 2 / 3
    
    def test_f1_score(self):
        """Test F1 score calculation."""
        y_true = [1, 0, 1, 1, 0, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        
        f1 = f1_score(y_true, y_pred)
        
        expected = 2 * (2/3 * 2/3) / (2/3 + 2/3)
        assert abs(f1 - expected) < 0.001
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        y_true = [1, 0, 1, 1, 0, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm["TP"] == 2
        assert cm["TN"] == 2
        assert cm["FP"] == 1
        assert cm["FN"] == 1


class TestAdvancedMetrics:
    """Test cases for advanced metrics."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        n = 100
        
        y_true = np.random.binomial(1, 0.3, n)
        y_score = np.random.uniform(0, 1, n)
        
        return y_true, y_score
    
    def test_calculate_ks(self, sample_predictions):
        """Test KS calculation."""
        y_true, y_score = sample_predictions
        
        ks, threshold = calculate_ks(y_true, y_score)
        
        assert 0 <= ks <= 1
        assert 0 <= threshold <= 1
    
    def test_calculate_gini(self, sample_predictions):
        """Test Gini calculation."""
        y_true, y_score = sample_predictions
        
        gini = calculate_gini(y_true, y_score)
        
        assert -1 <= gini <= 1
    
    def test_calculate_all_metrics(self, sample_predictions):
        """Test all metrics calculation."""
        y_true, y_score = sample_predictions
        y_pred = (y_score > 0.5).astype(int)
        
        metrics = calculate_all_metrics(y_true, y_pred, y_score)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc_roc" in metrics
        assert "ks_statistic" in metrics
        assert "gini" in metrics


class TestPSI:
    """Test cases for PSI calculation."""
    
    def test_psi_no_shift(self):
        """Test PSI with no distribution shift."""
        np.random.seed(42)
        
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)
        
        psi, table = calculate_psi(expected, actual)
        
        assert psi < 0.1
        assert isinstance(table, pl.DataFrame)
    
    def test_psi_significant_shift(self):
        """Test PSI with significant distribution shift."""
        np.random.seed(42)
        
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1, 1000)
        
        psi, table = calculate_psi(expected, actual)
        
        assert psi > 0.25
    
    def test_psi_uniform_bins(self):
        """Test PSI with uniform binning."""
        np.random.seed(42)
        
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0.5, 1, 1000)
        
        psi, table = calculate_psi(expected, actual, bin_type="uniform")
        
        assert psi > 0
