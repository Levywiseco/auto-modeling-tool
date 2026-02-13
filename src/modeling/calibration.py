# -*- coding: utf-8 -*-
"""
Model calibration module.

This module provides probability calibration functionality using
Platt Scaling (Sigmoid) and Isotonic Regression.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import polars as pl

from ..core.decorators import time_it
from ..core.logger import logger


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
        y_prob: Union[np.ndarray, pl.DataFrame],
    ) -> "ProbabilityCalibrator":
        """
        Fit the calibrator.

        Parameters
        ----------
        y_true : np.ndarray or pl.Series
            True labels.
        y_prob : np.ndarray or pl.DataFrame
            Predicted probabilities (positive class).

        Returns
        -------
        self
            Fitted calibrator.
        """
        if isinstance(y_true, pl.Series):
            y_true = y_true.to_numpy()
        if isinstance(y_prob, pl.DataFrame):
            y_prob = y_prob.to_numpy()
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]

        if self.method == "sigmoid":
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression

            self.calibrator_ = CalibratedClassifierCV(
                method="sigmoid",
                estimator=LogisticRegression(solver="lbfgs", max_iter=1000),
                cv=5,
            )
        elif self.method == "isotonic":
            from sklearn.calibration import CalibratedClassifierCV

            self.calibrator_ = CalibratedClassifierCV(
                method="isotonic",
                cv=5,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Create a dummy classifier for calibration
        from sklearn.base import BaseEstimator, ClassifierMixin

        class DummyClassifier(BaseEstimator, ClassifierMixin):
            def __init__(self):
                self.classes_ = np.unique(y_true)

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

            def predict_proba(self, X):
                return np.column_stack([1 - y_prob, y_prob])

        dummy = DummyClassifier()
        self.calibrator_.estimator = dummy
        self.calibrator_.fit(y_prob.reshape(-1, 1), y_true)

        self._is_fitted = True
        logger.info(f"✅ {self.method.capitalize()} calibrator fitted")

        return self

    def transform(
        self,
        y_prob: Union[np.ndarray, pl.DataFrame],
    ) -> np.ndarray:
        """
        Transform probabilities using fitted calibrator.

        Parameters
        ----------
        y_prob : np.ndarray or pl.DataFrame
            Predicted probabilities.

        Returns
        -------
        np.ndarray
            Calibrated probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        if isinstance(y_prob, pl.DataFrame):
            y_prob = y_prob.to_numpy()
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]

        return self.calibrator_.predict_proba(y_prob.reshape(-1, 1))[:, 1]

    def fit_transform(
        self,
        y_true: Union[np.ndarray, pl.Series],
        y_prob: Union[np.ndarray, pl.DataFrame],
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        y_true : np.ndarray or pl.Series
            True labels.
        y_prob : np.ndarray or pl.DataFrame
            Predicted probabilities.

        Returns
        -------
        np.ndarray
            Calibrated probabilities.
        """
        return self.fit(y_true, y_prob).transform(y_prob)


def calibrate_probabilities(
    y_true: Union[np.ndarray, pl.Series],
    y_prob: Union[np.ndarray, pl.DataFrame],
    method: str = "sigmoid",
) -> np.ndarray:
    """
    Calibrate probabilities using Platt Scaling or Isotonic Regression.

    Parameters
    ----------
    y_true : np.ndarray or pl.Series
        True labels.
    y_prob : np.ndarray or pl.DataFrame
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
