"""Tests for tools that are exported but not wired into the pipelines.

`calibration`, `tuning` and `cross_validation` are public API surface that
`auto_pipeline` never calls. They previously had no direct coverage, which is
how the calibrator shipped broken. These tests pin their behaviour.
"""

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.evaluation import cross_validate_model
from src.modeling import ProbabilityCalibrator, calibrate_probabilities
from src.modeling.tuning import get_default_param_grid, tune_hyperparameters


@pytest.fixture
def scored_sample():
    """Features, labels, and a deliberately over-confident score."""
    rng = np.random.default_rng(3)
    n = 600
    X = pl.DataFrame({f"f{i}": rng.normal(size=n) for i in range(3)})
    logit = X.to_numpy() @ np.array([1.2, -0.8, 0.5])
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    raw = (
        LogisticRegression(max_iter=300)
        .fit(X.to_numpy(), y)
        .predict_proba(X.to_numpy())[:, 1]
    )
    skewed = np.clip(raw**2.5, 1e-6, 1 - 1e-6)
    return X, y, skewed


class TestProbabilityCalibrator:
    @pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
    def test_calibration_improves_brier_score(self, scored_sample, method):
        _, y, skewed = scored_sample
        calibrated = calibrate_probabilities(pl.Series("t", y), skewed, method=method)

        assert calibrated.shape == skewed.shape
        assert brier_score_loss(y, calibrated) < brier_score_loss(y, skewed)
        assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0

    @pytest.mark.parametrize("wrap", [
        lambda p: p,
        lambda p: pl.Series("pred_proba", p),
        lambda p: pl.DataFrame({"neg": 1 - p, "pos": p}),
    ], ids=["ndarray", "series", "dataframe"])
    def test_accepts_numpy_series_and_dataframe(self, scored_sample, wrap):
        _, y, skewed = scored_sample
        calibrator = ProbabilityCalibrator("sigmoid").fit(pl.Series("t", y), skewed)
        assert calibrator.transform(wrap(skewed)).shape == (len(skewed),)

    def test_length_mismatch_raises(self, scored_sample):
        _, y, skewed = scored_sample
        with pytest.raises(ValueError, match="same length"):
            ProbabilityCalibrator().fit(y, skewed[:10])

    def test_transform_before_fit_raises(self, scored_sample):
        _, _, skewed = scored_sample
        with pytest.raises(RuntimeError, match="not fitted"):
            ProbabilityCalibrator().transform(skewed)

    def test_unknown_method_raises(self, scored_sample):
        _, y, skewed = scored_sample
        with pytest.raises(ValueError, match="Unknown method"):
            ProbabilityCalibrator("bogus").fit(y, skewed)


class TestCrossValidation:
    def test_cross_validate_model_reports_fold_scores(self, scored_sample):
        X, y, _ = scored_sample
        result = cross_validate_model(
            LogisticRegression(max_iter=200), X, pl.Series("t", y), cv=3
        )

        assert result["n_splits"] == 3
        assert len(result["fold_scores"]) == 3
        assert 0.0 <= result["mean_score"] <= 1.0


class TestTuning:
    def test_tune_hyperparameters_returns_best_model(self, scored_sample):
        X, y, _ = scored_sample
        grid = get_default_param_grid("logistic")
        best_model, best_params, best_score = tune_hyperparameters(
            LogisticRegression(max_iter=200), grid, X.to_numpy(), y, cv=2
        )

        assert best_params.keys() <= grid.keys()
        assert 0.0 <= best_score <= 1.0
        assert hasattr(best_model, "predict")

    def test_accepts_polars_input(self, scored_sample):
        X, y, _ = scored_sample
        grid = get_default_param_grid("logistic")
        _, _, score = tune_hyperparameters(
            LogisticRegression(max_iter=200), grid, X, pl.Series("t", y), cv=2
        )
        assert 0.0 <= score <= 1.0
