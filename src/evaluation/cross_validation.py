# -*- coding: utf-8 -*-
"""
Cross-validation module for model evaluation.

This module provides cross-validation functionality with support for
Stratified K-Fold, Time Series Split, and custom CV strategies.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from ..core.decorators import time_it
from ..core.logger import logger


class CrossValidator:
    """
    Cross-validation wrapper with support for multiple strategies.

    Parameters
    ----------
    cv_strategy : str, default "stratified"
        CV strategy: 'stratified', 'kfold', 'timeseries', or 'custom'.
    n_splits : int, default 5
        Number of folds.
    shuffle : bool, default True
        Whether to shuffle data before splitting.
    random_state : int, default 42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        cv_strategy: str = "stratified",
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_splits(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        y: Union[np.ndarray, pl.Series],
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get train/test splits for cross-validation.

        Parameters
        ----------
        X : np.ndarray or pl.DataFrame
            Features.
        y : np.ndarray or pl.Series
            Target.

        Returns
        -------
        list
            List of (X_train, X_test, y_train, y_test) tuples.
        """
        from sklearn.model_selection import (
            StratifiedKFold,
            KFold,
            TimeSeriesSplit,
        )

        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pl.Series):
            y = y.to_numpy()

        if self.cv_strategy == "stratified":
            cv = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        elif self.cv_strategy == "kfold":
            cv = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        elif self.cv_strategy == "timeseries":
            cv = TimeSeriesSplit(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")

        splits = []
        for train_idx, test_idx in cv.split(X, y):
            splits.append(
                (X[train_idx], X[test_idx], y[train_idx], y[test_idx])
            )

        return splits

    @time_it
    def cross_validate(
        self,
        model: Any,
        X: Union[np.ndarray, pl.DataFrame],
        y: Union[np.ndarray, pl.Series],
        scoring_func: Optional[Callable] = None,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Parameters
        ----------
        model : Any
            Model to evaluate (scikit-learn compatible).
        X : np.ndarray or pl.DataFrame
            Features.
        y : np.ndarray or pl.Series
            Target.
        scoring_func : callable, optional
            Custom scoring function. Takes (y_true, y_pred, y_prob) and returns a score.
        scoring : str, default "accuracy"
            Sklearn scoring metric (used if scoring_func is None).

        Returns
        -------
        dict
            Cross-validation results including scores per fold and statistics.
        """
        from sklearn.metrics import get_scorer

        splits = self.get_splits(X, y)
        scores = []
        fold_results = []

        logger.info(
            f"🔄 Running {self.cv_strategy} cross-validation with {self.n_splits} folds..."
        )

        scorer = get_scorer(scoring) if scoring_func is None else None

        for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(splits):
            # Clone model for each fold
            from sklearn.base import clone

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)

            y_pred = fold_model.predict(X_test)

            if scoring_func is not None:
                try:
                    y_prob = fold_model.predict_proba(X_test)[:, 1]
                    fold_score = scoring_func(y_test, y_pred, y_prob)
                except Exception:
                    fold_score = scoring_func(y_test, y_pred)
            elif scorer is not None:
                fold_score = scorer(fold_model, X_test, y_test)
            else:
                from sklearn.metrics import accuracy_score

                fold_score = accuracy_score(y_test, y_pred)

            scores.append(fold_score)
            fold_results.append(
                {
                    "fold": fold_idx + 1,
                    "score": fold_score,
                }
            )

            logger.info(f"   Fold {fold_idx + 1}: {fold_score:.4f}")

        scores = np.array(scores)

        results = {
            "cv_strategy": self.cv_strategy,
            "n_splits": self.n_splits,
            "fold_scores": fold_results,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        }

        logger.info(f"   Mean: {results['mean_score']:.4f} ± {results['std_score']:.4f}")

        return results


def cross_validate_model(
    model: Any,
    X: Union[np.ndarray, pl.DataFrame],
    y: Union[np.ndarray, pl.Series],
    cv: int = 5,
    scoring: str = "accuracy",
    cv_strategy: str = "stratified",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Quick cross-validation function.

    Parameters
    ----------
    model : Any
        Model to evaluate.
    X : np.ndarray or pl.DataFrame
        Features.
    y : np.ndarray or pl.Series
        Target.
    cv : int, default 5
        Number of folds.
    scoring : str, default "accuracy"
        Scoring metric.
    cv_strategy : str, default "stratified"
        CV strategy.
    random_state : int, default 42
        Random seed.

    Returns
    -------
    dict
        Cross-validation results.

    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> results = cross_validate_model(
    ...     LogisticRegression(),
    ...     X, y,
    ...     cv=5,
    ...     scoring="roc_auc"
    ... )
    >>> print(f"CV Score: {results['mean_score']:.4f}")
    """
    validator = CrossValidator(
        cv_strategy=cv_strategy,
        n_splits=cv,
        random_state=random_state,
    )

    return validator.cross_validate(model, X, y, scoring=scoring)


def stratified_kfold_cv(
    X: Union[np.ndarray, pl.DataFrame],
    y: Union[np.ndarray, pl.Series],
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get Stratified K-Fold splits.

    Parameters
    ----------
    X : np.ndarray or pl.DataFrame
        Features.
    y : np.ndarray or pl.Series
        Target.
    n_splits : int, default 5
        Number of splits.
    random_state : int, default 42
        Random seed.

    Returns
    -------
    list
        List of (X_train, X_test, y_train, y_test) tuples.
    """
    validator = CrossValidator(
        cv_strategy="stratified",
        n_splits=n_splits,
        random_state=random_state,
    )
    return validator.get_splits(X, y)


def timeseries_cv(
    X: Union[np.ndarray, pl.DataFrame],
    y: Union[np.ndarray, pl.Series],
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get Time Series Cross-Validation splits.

    Parameters
    ----------
    X : np.ndarray or pl.DataFrame
        Features.
    y : np.ndarray or pl.Series
        Target.
    n_splits : int, default 5
        Number of splits.

    Returns
    -------
    list
        List of (X_train, X_test, y_train, y_test) tuples.
    """
    validator = CrossValidator(
        cv_strategy="timeseries",
        n_splits=n_splits,
    )
    return validator.get_splits(X, y)
