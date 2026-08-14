"""
Tests for evaluation metrics module.
"""

import numpy as np
import polars as pl
import pytest

from auto_modeling_tool.evaluation.metrics import (
    accuracy,
    calculate_all_metrics,
    calculate_gini,
    calculate_ks,
    calculate_psi,
    confusion_matrix,
    f1_score,
    precision,
    recall,
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


def test_metrics_handle_one_class_evaluation_slice():
    from auto_modeling_tool.evaluation.metrics import calculate_all_metrics

    metrics = calculate_all_metrics(
        [1, 1, 1],
        [1, 1, 0],
        [0.9, 0.8, 0.2],
        sample_weight=[1.0, 2.0, 1.0],
    )
    assert metrics["auc_roc"] == 0.5
    assert metrics["ks_statistic"] == 0.0


def test_calculate_lift_handles_duplicate_scores():
    from auto_modeling_tool.evaluation.metrics import calculate_lift

    table = calculate_lift(
        [0, 1, 0, 1],
        [0.5, 0.5, 0.5, 0.5],
        n_bins=10,
    )
    assert isinstance(table, pl.DataFrame)
    assert len(table) == 1


class TestKSTieHandling:
    """KS must be read only where a threshold could fall.

    Measuring inside a group of equal scores credits the model with
    discrimination it does not have: a constant score previously reported
    KS 0.924 whenever the input arrived sorted by label.
    """

    @staticmethod
    def _reference_ks(y, s, w=None):
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y, s, sample_weight=w)
        return float(np.max(np.abs(tpr - fpr)))

    def test_constant_score_has_no_discrimination(self):
        from auto_modeling_tool.evaluation.metrics import calculate_ks

        y = np.array([0] * 200 + [1] * 200)
        ks, _ = calculate_ks(y, np.full(400, 0.3))
        assert ks == 0.0

    def test_perfect_separation_is_one(self):
        from auto_modeling_tool.evaluation.metrics import calculate_ks

        y = np.array([0] * 200 + [1] * 200)
        score = np.concatenate([np.zeros(200), np.ones(200)])
        assert calculate_ks(y, score)[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("n_levels", [2, 4, 20])
    def test_matches_roc_definition_with_heavy_ties(self, n_levels):
        """Scorecards emit few distinct values — the tie-heavy real case."""
        from auto_modeling_tool.evaluation.metrics import calculate_ks

        rng = np.random.default_rng(7)
        y = rng.binomial(1, 0.35, 800)
        levels = np.linspace(0.1, 0.9, n_levels)
        score = rng.choice(levels, 800) + y * 0.05

        ks, _ = calculate_ks(y, score)
        assert ks == pytest.approx(self._reference_ks(y, score), abs=1e-9)

    def test_matches_roc_definition_when_weighted(self):
        from auto_modeling_tool.evaluation.metrics import calculate_ks

        rng = np.random.default_rng(11)
        y = rng.binomial(1, 0.4, 600)
        score = rng.choice([0.2, 0.4, 0.6, 0.8], 600)
        weights = rng.choice([1.0, 3.0], 600)

        ks, _ = calculate_ks(y, score, sample_weight=weights)
        assert ks == pytest.approx(self._reference_ks(y, score, weights), abs=1e-9)
