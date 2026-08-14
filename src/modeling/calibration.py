"""
Model calibration module.

This module provides probability calibration functionality using
Platt Scaling (Sigmoid) and Isotonic Regression.
"""

from typing import Any, Optional, Union

import numpy as np
import polars as pl

from ..core.decorators import time_it
from ..core.logger import logger


def _as_proba_array(y_prob: Union[np.ndarray, pl.DataFrame, pl.Series]) -> np.ndarray:
    """Normalize probabilities to a 1-D NumPy array of positive-class scores."""
    if isinstance(y_prob, (pl.DataFrame, pl.Series)):
        y_prob = y_prob.to_numpy()
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.ravel()
    return y_prob


class ProbabilityCalibrator:
    """
    Probability calibration wrapper.

    Supports Platt Scaling (Sigmoid) and Isotonic Regression calibration.

    Parameters
    ----------
    method : str, default "sigmoid"
        Calibration method: 'sigmoid' (Platt Scaling) or 'isotonic'.
    """

    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.calibrator_: Optional[Any] = None
        self._is_fitted = False

    @time_it
    def fit(
        self,
        y_true: Union[np.ndarray, pl.Series],
        y_prob: Union[np.ndarray, pl.DataFrame, pl.Series],
    ) -> "ProbabilityCalibrator":
        """
        Fit the calibrator.

        Parameters
        ----------
        y_true : np.ndarray or pl.Series
            True labels.
        y_prob : np.ndarray, pl.DataFrame or pl.Series
            Predicted probabilities (positive class).

        Returns
        -------
        self
            Fitted calibrator.
        """
        if isinstance(y_true, pl.Series):
            y_true = y_true.to_numpy()
        y_true = np.asarray(y_true)
        y_prob = _as_proba_array(y_prob)

        if y_prob.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"y_true and y_prob must have the same length, "
                f"got {y_true.shape[0]} and {y_prob.shape[0]}"
            )

        # Calibration fits a 1-D mapping from an existing score to a corrected
        # probability. CalibratedClassifierCV is deliberately NOT used here: it
        # wraps and refits a classifier, and cannot consume precomputed scores.
        if self.method == "sigmoid":
            from sklearn.linear_model import LogisticRegression

            # Platt scaling: logistic regression on the raw score.
            self.calibrator_ = LogisticRegression(solver="lbfgs", max_iter=1000)
            self.calibrator_.fit(y_prob.reshape(-1, 1), y_true)
        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.calibrator_ = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self.calibrator_.fit(y_prob, y_true)
        else:
            raise ValueError(
                f"Unknown method: {self.method}. Use 'sigmoid' or 'isotonic'."
            )

        self._is_fitted = True
        logger.info(f"✅ {self.method.capitalize()} calibrator fitted")

        return self

    def transform(
        self,
        y_prob: Union[np.ndarray, pl.DataFrame, pl.Series],
    ) -> np.ndarray:
        """
        Transform probabilities using fitted calibrator.

        Parameters
        ----------
        y_prob : np.ndarray, pl.DataFrame or pl.Series
            Predicted probabilities.

        Returns
        -------
        np.ndarray
            Calibrated probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        y_prob = _as_proba_array(y_prob)

        if self.method == "sigmoid":
            return self.calibrator_.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        return np.asarray(self.calibrator_.predict(y_prob))

    def fit_transform(
        self,
        y_true: Union[np.ndarray, pl.Series],
        y_prob: Union[np.ndarray, pl.DataFrame, pl.Series],
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        y_true : np.ndarray or pl.Series
            True labels.
        y_prob : np.ndarray, pl.DataFrame or pl.Series
            Predicted probabilities.

        Returns
        -------
        np.ndarray
            Calibrated probabilities.
        """
        return self.fit(y_true, y_prob).transform(y_prob)


def calibrate_probabilities(
    y_true: Union[np.ndarray, pl.Series],
    y_prob: Union[np.ndarray, pl.DataFrame, pl.Series],
    method: str = "sigmoid",
) -> np.ndarray:
    """
    Calibrate probabilities using Platt Scaling or Isotonic Regression.

    Parameters
    ----------
    y_true : np.ndarray or pl.Series
        True labels.
    y_prob : np.ndarray, pl.DataFrame or pl.Series
        Predicted probabilities (positive class).
    method : str, default "sigmoid"
        Calibration method: 'sigmoid' (Platt Scaling) or 'isotonic'.

    Returns
    -------
    np.ndarray
        Calibrated probabilities.

    Example
    -------
    >>> # Original probabilities from XGBoost
    >>> y_prob_raw = model.predict_proba(X_test)[:, 1]
    >>>
    >>> # Calibrate probabilities
    >>> y_prob_calibrated = calibrate_probabilities(y_test, y_prob_raw, method="isotonic")
    """
    calibrator = ProbabilityCalibrator(method=method)
    return calibrator.fit_transform(y_true, y_prob)


def create_calibrated_model(
    model: Any,
    X: Union[np.ndarray, pl.DataFrame],
    y: Union[np.ndarray, pl.Series],
    method: str = "sigmoid",
    cv: int = 5,
) -> Any:
    """
    Create a calibrated version of a model.

    Parameters
    ----------
    model : Any
        Base model to calibrate.
    X : np.ndarray or pl.DataFrame
        Training features.
    y : np.ndarray or pl.Series
        Training labels.
    method : str, default "sigmoid"
        Calibration method.
    cv : int, default 5
        Number of cross-validation folds for calibration.

    Returns
    -------
    Any
        Calibrated model.

    Example
    -------
    >>> from xgboost import XGBClassifier
    >>> base_model = XGBClassifier()
    >>> calibrated = create_calibrated_model(base_model, X_train, y_train, method="isotonic")
    >>> y_prob = calibrated.predict_proba(X_test)[:, 1]
    """
    from sklearn.calibration import CalibratedClassifierCV

    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pl.Series):
        y = y.to_numpy()

    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv,
    )

    calibrated_model.fit(X, y)
    logger.info(f"✅ Calibrated model created using {method}")

    return calibrated_model
