# -*- coding: utf-8 -*-
"""
High-performance hyperparameter tuning module.

This module provides functions for hyperparameter tuning using
GridSearchCV and RandomizedSearchCV.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.decorators import time_it
from ..core.logger import logger


@time_it
def tune_hyperparameters(
    model: Any,
    param_grid: Dict[str, List[Any]],
    X_train: Union[np.ndarray, Any],
    y_train: Union[np.ndarray, Any],
    scoring: str = "accuracy",
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 0,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Tune hyperparameters using GridSearchCV.

    Parameters
    ----------
    model : Any
        Scikit-learn compatible model.
    param_grid : dict
        Parameter grid for grid search.
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    scoring : str, default "accuracy"
        Scoring metric.
    cv : int, default 5
        Number of cross-validation folds.
    n_jobs : int, default -1
        Number of parallel jobs (-1 for all CPUs).
    verbose : int, default 0
        Verbosity level.

    Returns
    -------
    tuple
        (best_estimator, best_params, best_score)

    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression()
    >>> param_grid = {"C": [0.1, 1.0, 10.0]}
    >>> best_model, best_params, score = tune_hyperparameters(
    ...     model, param_grid, X_train, y_train
    ... )
    """
    from sklearn.model_selection import GridSearchCV

    logger.info(f"🔧 Running GridSearchCV with {cv}-fold cross-validation...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"   Best params: {grid_search.best_params_}")
    logger.info(f"   Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


@time_it
def random_search_hyperparameters(
    model: Any,
    param_distributions: Dict[str, Any],
    X_train: Union[np.ndarray, Any],
    y_train: Union[np.ndarray, Any],
    scoring: str = "accuracy",
    n_iter: int = 100,
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 0,
    random_state: Optional[int] = 42,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Tune hyperparameters using RandomizedSearchCV.

    Parameters
    ----------
    model : Any
        Scikit-learn compatible model.
    param_distributions : dict
        Parameter distributions for random search.
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    scoring : str, default "accuracy"
        Scoring metric.
    n_iter : int, default 100
        Number of parameter settings sampled.
    cv : int, default 5
        Number of cross-validation folds.
    n_jobs : int, default -1
        Number of parallel jobs (-1 for all CPUs).
    verbose : int, default 0
        Verbosity level.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    tuple
        (best_estimator, best_params, best_score)

    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier()
    >>> param_distributions = {
    ...     "n_estimators": [50, 100, 200],
    ...     "max_depth": [3, 5, 7, None],
    ... }
    >>> best_model, best_params, score = random_search_hyperparameters(
    ...     model, param_distributions, X_train, y_train, n_iter=50
    ... )
    """
    from sklearn.model_selection import RandomizedSearchCV

    logger.info(f"🔧 Running RandomizedSearchCV with {n_iter} iterations...")

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        return_train_score=True,
    )

    random_search.fit(X_train, y_train)

    logger.info(f"   Best params: {random_search.best_params_}")
    logger.info(f"   Best CV score: {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_


def tune_model(
    model: Any,
    X_train: Union[np.ndarray, Any],
    y_train: Union[np.ndarray, Any],
    tuning_method: str = "grid",
    param_grid: Optional[Dict[str, List[Any]]] = None,
    param_distributions: Optional[Dict[str, Any]] = None,
    scoring: str = "accuracy",
    cv: int = 5,
    n_iter: int = 100,
    n_jobs: int = -1,
    verbose: int = 0,
    random_state: Optional[int] = 42,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Unified hyperparameter tuning interface.

    Parameters
    ----------
    model : Any
        Scikit-learn compatible model.
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    tuning_method : str, default "grid"
        Tuning method: 'grid' or 'random'.
    param_grid : dict, optional
        Parameter grid for grid search.
    param_distributions : dict, optional
        Parameter distributions for random search.
    scoring : str, default "accuracy"
        Scoring metric.
    cv : int, default 5
        Number of cross-validation folds.
    n_iter : int, default 100
        Number of iterations for random search.
    n_jobs : int, default -1
        Number of parallel jobs.
    verbose : int, default 0
        Verbosity level.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    tuple
        (best_estimator, best_params, best_score)

    Example
    -------
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> model = GradientBoostingClassifier()
    >>> best_model, best_params, score = tune_model(
    ...     model, X_train, y_train,
    ...     tuning_method="random",
    ...     param_distributions={"n_estimators": [50, 100, 200]},
    ...     n_iter=20,
    ... )
    """
    if tuning_method == "grid":
        if param_grid is None:
            raise ValueError("param_grid must be provided for grid search.")
        return tune_hyperparameters(
            model, param_grid, X_train, y_train,
            scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose,
        )
    elif tuning_method == "random":
        if param_distributions is None:
            raise ValueError("param_distributions must be provided for random search.")
        return random_search_hyperparameters(
            model, param_distributions, X_train, y_train,
            scoring=scoring, n_iter=n_iter, cv=cv,
            n_jobs=n_jobs, verbose=verbose, random_state=random_state,
        )
    else:
        raise ValueError("Invalid tuning method. Choose 'grid' or 'random'.")


# Default parameter grids for common models
DEFAULT_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "logistic": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
        "penalty": ["l2"],
        "max_iter": [500, 1000],
    },
    "decision_tree": {
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    },
    "lightgbm": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "catboost": {
        "iterations": [100, 200, 300],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "l2_leaf_reg": [1, 3, 5, 7],
    },
}


def get_default_param_grid(model_type: str) -> Optional[Dict[str, List[Any]]]:
    """
    Get default parameter grid for a model type.

    Parameters
    ----------
    model_type : str
        Model type (e.g., 'logistic', 'xgboost', 'random_forest').

    Returns
    -------
    dict or None
        Default parameter grid if available.
    """
    # Normalize model type name
    type_mapping = {
        "logistic": "logistic",
        "logistic_regression": "logistic",
        "tree": "decision_tree",
        "decision_tree": "decision_tree",
        "random_forest": "random_forest",
        "rf": "random_forest",
        "xgboost": "xgboost",
        "xgb": "xgboost",
        "lightgbm": "lightgbm",
        "lgbm": "lightgbm",
        "catboost": "catboost",
    }

    normalized_type = type_mapping.get(model_type.lower(), model_type.lower())
    return DEFAULT_PARAM_GRIDS.get(normalized_type)
