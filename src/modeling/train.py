# -*- coding: utf-8 -*-
"""
High-performance model training module.

This module provides a unified interface for training various
classification models with hyperparameter tuning support.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import polars as pl

from ..core.base import MarsBaseEstimator
from ..core.logger import logger
from ..core.decorators import time_it
from ..core.exceptions import ValidationError


SUPPORTED_MODELS = {
    "logistic": "LogisticRegression",
    "tree": "DecisionTreeClassifier",
    "xgboost": "XGBClassifier",
    "random_forest": "RandomForestClassifier",
    "lightgbm": "LGBMClassifier",
}


class ModelTrainer(MarsBaseEstimator):
    """
    Unified model training with optional hyperparameter tuning.
    
    Supports multiple classification algorithms with a consistent interface.
    
    Parameters
    ----------
    model_type : str, default "logistic"
        Type of model to train. Options: 'logistic', 'tree', 'xgboost',
        'random_forest', 'lightgbm'.
    hyperparameter_tuning : bool, default False
        Whether to perform hyperparameter tuning.
    cv_folds : int, default 5
        Number of cross-validation folds for tuning.
    scoring : str, default "roc_auc"
        Scoring metric for hyperparameter tuning.
    random_state : int, default 42
        Random seed for reproducibility.
    n_jobs : int, default -1
        Number of parallel jobs.
        
    Attributes
    ----------
    model_ : Any
        Trained model instance.
    best_params_ : dict
        Best hyperparameters found (if tuning enabled).
    cv_scores_ : list
        Cross-validation scores (if tuning enabled).
        
    Example
    -------
    >>> trainer = ModelTrainer(model_type="logistic")
    >>> trainer.fit(X_train, y_train)
    >>> predictions = trainer.predict(X_test)
    >>> probabilities = trainer.predict_proba(X_test)
    """
    
    def __init__(
        self,
        model_type: str = "logistic",
        hyperparameter_tuning: bool = False,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
        random_state: int = 42,
        n_jobs: int = -1,
        **model_params,
    ):
        self.model_type = model_type
        self.hyperparameter_tuning = hyperparameter_tuning
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_params = model_params
        
        self.model_: Optional[Any] = None
        self.best_params_: Dict[str, Any] = {}
        self.cv_scores_: List[float] = []
        self._is_fitted = False
    
    def _create_model(self, **params) -> Any:
        """Create model instance based on type."""
        if self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                **params
            )
        
        elif self.model_type == "tree":
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(
                random_state=self.random_state,
                **params
            )
        
        elif self.model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=self.n_jobs,
                **params
            )
        
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **params
            )
        
        elif self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
                **params
            )
        
        else:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Supported types: {list(SUPPORTED_MODELS.keys())}"
            )
    
    def _get_param_grid(self) -> Dict[str, List]:
        """Get default parameter grid for hyperparameter tuning."""
        param_grids = {
            "logistic": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
                "penalty": ["l2"],
            },
            "tree": {
                "max_depth": [3, 5, 7, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10],
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
            },
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5, 10],
            },
            "lightgbm": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
            },
        }
        return param_grids.get(self.model_type, {})
    
    @time_it
    def fit(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        y: Union[np.ndarray, pl.Series],
        **kwargs,
    ) -> "ModelTrainer":
        """
        Fit the model to training data.
        
        Parameters
        ----------
        X : np.ndarray or pl.DataFrame
            Training features.
        y : np.ndarray or pl.Series
            Training target.
        **kwargs
            Additional parameters.
            
        Returns
        -------
        self
            Fitted trainer instance.
        """
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pl.Series):
            y = y.to_numpy()
        
        X = np.nan_to_num(X, nan=0.0)
        
        logger.info(f"🤖 Training {self.model_type} model...")
        logger.info(f"   Training samples: {len(X)}")
        
        if self.hyperparameter_tuning:
            self._tune_hyperparameters(X, y)
        else:
            self.model_ = self._create_model(**self.model_params)
            self.model_.fit(X, y)
        
        self._is_fitted = True
        logger.info(f"✅ Model trained successfully")
        
        return self
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """Perform hyperparameter tuning using GridSearchCV."""
        from sklearn.model_selection import GridSearchCV
        
        param_grid = self._get_param_grid()
        
        if not param_grid:
            logger.warning(
                f"No parameter grid defined for {self.model_type}, "
                f"using default parameters"
            )
            self.model_ = self._create_model(**self.model_params)
            self.model_.fit(X, y)
            return
        
        logger.info(f"   Performing hyperparameter tuning with {self.cv_folds}-fold CV...")
        
        base_model = self._create_model()
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=0,
        )
        
        grid_search.fit(X, y)
        
        self.model_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.cv_scores_ = grid_search.cv_results_['mean_test_score'].tolist()
        
        logger.info(f"   Best params: {self.best_params_}")
        logger.info(f"   Best CV score: {grid_search.best_score_:.4f}")
    
    def predict(self, X: Union[np.ndarray, pl.DataFrame]) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray or pl.DataFrame
            Features to predict.
            
        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        if not self._is_fitted:
            raise ValidationError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        
        X = np.nan_to_num(X, nan=0.0)
        return self.model_.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pl.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray or pl.DataFrame
            Features to predict.
            
        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        if not self._is_fitted:
            raise ValidationError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        
        X = np.nan_to_num(X, nan=0.0)
        return self.model_.predict_proba(X)
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Get feature importance from trained model.
        
        Parameters
        ----------
        feature_names : list of str, optional
            Feature names. If None, uses generic names.
            
        Returns
        -------
        pl.DataFrame
            DataFrame with Feature and Importance columns.
        """
        if not self._is_fitted:
            raise ValidationError("Model not fitted. Call fit() first.")
        
        if hasattr(self.model_, 'feature_importances_'):
            importance = self.model_.feature_importances_
        elif hasattr(self.model_, 'coef_'):
            importance = np.abs(self.model_.coef_).flatten()
            if len(importance.shape) > 1:
                importance = np.mean(importance, axis=0)
        else:
            raise ValueError(
                f"Model {self.model_type} does not support feature importance"
            )
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        return pl.DataFrame({
            "Feature": feature_names,
            "Importance": importance,
        }).sort("Importance", descending=True)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of trained model.
        
        Returns
        -------
        dict
            Model summary including type, parameters, and metrics.
        """
        if not self._is_fitted:
            raise ValidationError("Model not fitted. Call fit() first.")
        
        summary = {
            "model_type": self.model_type,
            "model_class": type(self.model_).__name__,
            "hyperparameter_tuning": self.hyperparameter_tuning,
            "best_params": self.best_params_,
            "random_state": self.random_state,
        }
        
        if hasattr(self.model_, 'n_features_in_'):
            summary["n_features"] = self.model_.n_features_in_
        
        if hasattr(self.model_, 'classes_'):
            summary["classes"] = self.model_.classes_.tolist()
        
        return summary
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        import joblib
        
        if not self._is_fitted:
            raise ValidationError("Model not fitted. Call fit() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "model": self.model_,
            "model_type": self.model_type,
            "best_params": self.best_params_,
            "random_state": self.random_state,
        }
        
        joblib.dump(save_data, path)
        logger.info(f"✅ Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelTrainer":
        """Load model from disk."""
        import joblib
        
        path = Path(path)
        data = joblib.load(path)
        
        trainer = cls(
            model_type=data["model_type"],
            random_state=data.get("random_state", 42),
        )
        trainer.model_ = data["model"]
        trainer.best_params_ = data.get("best_params", {})
        trainer._is_fitted = True
        
        return trainer


@time_it
def train_model(
    X_train: Union[np.ndarray, pl.DataFrame],
    y_train: Union[np.ndarray, pl.Series],
    model_type: str = "logistic",
    **kwargs,
) -> Any:
    """
    Quick model training function (functional API).
    
    Parameters
    ----------
    X_train : np.ndarray or pl.DataFrame
        Training features.
    y_train : np.ndarray or pl.Series
        Training target.
    model_type : str, default "logistic"
        Type of model to train.
    **kwargs
        Additional parameters passed to ModelTrainer.
        
    Returns
    -------
    Any
        Trained model instance.
        
    Example
    -------
    >>> model = train_model(X_train, y_train, model_type="xgboost")
    >>> predictions = model.predict(X_test)
    """
    trainer = ModelTrainer(model_type=model_type, **kwargs)
    trainer.fit(X_train, y_train)
    return trainer.model_


def save_model(model: Any, path: Union[str, Path]) -> None:
    """Save model to disk."""
    import joblib
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, path)
    logger.info(f"✅ Model saved to {path}")


def load_model(path: Union[str, Path]) -> Any:
    """Load model from disk."""
    import joblib
    
    path = Path(path)
    model = joblib.load(path)
    logger.info(f"✅ Model loaded from {path}")
    return model
