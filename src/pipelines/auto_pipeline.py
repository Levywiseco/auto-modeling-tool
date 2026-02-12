# -*- coding: utf-8 -*-
"""
Complete Auto-Modeling Pipeline.

This module provides an end-to-end automated modeling workflow
that integrates all components of the AutoModelTool framework.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl

from ..core.logger import logger
from ..core.decorators import time_it
from ..core.exceptions import ValidationError

from ..data.loaders import load_data
from ..data.preprocess import DataPreprocessor
from ..data.split import stratified_train_test_split

from ..binning.woe_binning import WoeBinner

from ..features.selection import FeatureSelector
from ..features.importance import calculate_feature_importance

from ..evaluation.metrics import calculate_all_metrics

from ..utils.io import save_model, save_dataframe, generate_model_report


class AutoPipeline:
    """
    Complete Auto-Modeling Pipeline.
    
    This class provides an end-to-end automated workflow for credit risk
    modeling, including data preprocessing, WOE binning, feature selection,
    model training, and evaluation.
    
    Parameters
    ----------
    target_col : str
        Name of the target column.
    test_size : float, default 0.2
        Proportion of data for testing.
    n_bins : int, default 10
        Number of bins for WOE binning.
    binning_method : str, default "quantile"
        Binning method: 'quantile', 'uniform', 'cart'.
    selection_method : str, default "iv"
        Feature selection method.
    n_features : int, default 20
        Number of features to select.
    random_state : int, default 42
        Random seed for reproducibility.
        
    Attributes
    ----------
    preprocessor_ : DataPreprocessor
        Fitted preprocessor.
    binner_ : WoeBinner
        Fitted WOE binner.
    selector_ : FeatureSelector
        Fitted feature selector.
    model_ : Any
        Trained model.
    metrics_ : dict
        Evaluation metrics.
        
    Example
    -------
    >>> pipeline = AutoPipeline(target_col="bad_flag")
    >>> pipeline.fit(data)
    >>> metrics = pipeline.evaluate(test_data)
    >>> pipeline.save("model_output/")
    """
    
    def __init__(
        self,
        target_col: str,
        test_size: float = 0.2,
        n_bins: int = 10,
        binning_method: str = "quantile",
        selection_method: str = "iv",
        n_features: int = 20,
        random_state: int = 42,
    ):
        self.target_col = target_col
        self.test_size = test_size
        self.n_bins = n_bins
        self.binning_method = binning_method
        self.selection_method = selection_method
        self.n_features = n_features
        self.random_state = random_state
        
        self.preprocessor_: Optional[DataPreprocessor] = None
        self.binner_: Optional[WoeBinner] = None
        self.selector_: Optional[FeatureSelector] = None
        self.model_: Optional[Any] = None
        self.metrics_: Optional[Dict[str, float]] = None
        self.selected_features_: List[str] = []
        self.feature_importance_: Optional[pl.DataFrame] = None
    
    @time_it
    def fit(
        self,
        data: Union[str, Path, pl.DataFrame],
        **kwargs
    ) -> "AutoPipeline":
        """
        Fit the complete pipeline.
        
        Parameters
        ----------
        data : str, Path, or pl.DataFrame
            Input data (file path or DataFrame).
        **kwargs
            Additional parameters.
            
        Returns
        -------
        self
            Fitted pipeline.
        """
        logger.info("=" * 60)
        logger.info("🚀 Starting AutoPipeline Training")
        logger.info("=" * 60)
        
        if isinstance(data, (str, Path)):
            logger.info("\n📂 Loading data...")
            df = load_data(data)
        else:
            df = data
        
        if self.target_col not in df.columns:
            raise ValidationError(f"Target column '{self.target_col}' not found in data")
        
        logger.info(f"   Data shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        logger.info("\n🧹 Preprocessing data...")
        self.preprocessor_ = DataPreprocessor(
            clean_strategy="median",
            normalize_method="zscore",
        )
        y = df.get_column(self.target_col)
        df_clean = self.preprocessor_.fit_transform(df, y)
        df_clean = df_clean.with_columns(y.alias(self.target_col))
        
        logger.info("\n✂️ Splitting data...")
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            df_clean,
            self.target_col,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        
        self._X_test = X_test
        self._y_test = y_test
        
        logger.info("\n📊 WOE Binning...")
        self.binner_ = WoeBinner(
            n_bins=self.n_bins,
            method=self.binning_method,
            min_samples_bin=50,
        )
        X_train_woe = self.binner_.fit_transform(X_train, y_train, return_type="woe")
        
        logger.info("\n🎯 Feature selection...")
        self.selector_ = FeatureSelector(
            method=self.selection_method,
            n_features=self.n_features,
            iv_threshold=0.02,
        )
        
        woe_cols = [c for c in X_train_woe.columns if c.endswith("_bin")]
        X_train_selected = self.selector_.fit_transform(X_train_woe, y_train)
        self.selected_features_ = self.selector_.get_selected_features()
        
        logger.info("\n🤖 Training model...")
        from sklearn.linear_model import LogisticRegression
        
        self.model_ = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs',
        )
        X_train_np = X_train_selected.to_numpy()
        y_train_np = y_train.to_numpy()
        self.model_.fit(X_train_np, y_train_np)
        
        self._X_train_selected = X_train_selected
        self._y_train = y_train
        
        logger.info("\n📊 Calculating feature importance...")
        self.feature_importance_ = calculate_feature_importance(
            model=self.model_,
            X=X_train_selected,
            y=y_train,
            method="model",
        )
        
        logger.info("\n✅ Pipeline training completed!")
        
        return self
    
    @time_it
    def evaluate(
        self,
        X_test: Optional[pl.DataFrame] = None,
        y_test: Optional[pl.Series] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the fitted pipeline.
        
        Parameters
        ----------
        X_test : pl.DataFrame, optional
            Test features. If None, uses held-out test set.
        y_test : pl.Series, optional
            Test target. If None, uses held-out test set.
            
        Returns
        -------
        dict
            Evaluation metrics.
        """
        if self.model_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        
        if X_test is None:
            X_test = self._X_test
        if y_test is None:
            y_test = self._y_test
        
        X_test_woe = self.binner_.transform(X_test, return_type="woe")
        X_test_selected = self.selector_.transform(X_test_woe)
        
        X_test_np = X_test_selected.to_numpy()
        y_test_np = y_test.to_numpy()
        
        y_pred = self.model_.predict(X_test_np)
        y_prob = self.model_.predict_proba(X_test_np)[:, 1]
        
        self.metrics_ = calculate_all_metrics(y_test_np, y_pred, y_prob)
        
        logger.info(f"   AUC-ROC: {self.metrics_.get('auc_roc', 0):.4f}")
        logger.info(f"   KS: {self.metrics_.get('ks_statistic', 0):.4f}")
        logger.info(f"   Accuracy: {self.metrics_.get('accuracy', 0):.4f}")
        
        return self.metrics_
    
    def predict(
        self,
        X: pl.DataFrame,
        return_proba: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : pl.DataFrame
            Input features.
        return_proba : bool, default False
            If True, return probabilities instead of class labels.
            
        Returns
        -------
        np.ndarray or tuple
            Predictions (and probabilities if return_proba=True).
        """
        if self.model_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        
        X_preprocessed = self.preprocessor_.transform(X)
        X_woe = self.binner_.transform(X_preprocessed, return_type="woe")
        X_selected = self.selector_.transform(X_woe)
        X_np = X_selected.to_numpy()
        
        if return_proba:
            return self.model_.predict_proba(X_np)[:, 1]
        return self.model_.predict(X_np)
    
    def save(self, output_dir: Union[str, Path]) -> Path:
        """
        Save the complete pipeline.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory.
            
        Returns
        -------
        Path
            Path to saved pipeline.
        """
        import joblib
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pipeline_data = {
            "target_col": self.target_col,
            "test_size": self.test_size,
            "n_bins": self.n_bins,
            "binning_method": self.binning_method,
            "selection_method": self.selection_method,
            "n_features": self.n_features,
            "random_state": self.random_state,
            "preprocessor": self.preprocessor_,
            "binner": self.binner_,
            "selector": self.selector_,
            "model": self.model_,
            "selected_features": self.selected_features_,
            "metrics": self.metrics_,
        }
        
        joblib.dump(pipeline_data, output_path / "pipeline.pkl")
        
        if self.metrics_:
            generate_model_report(
                model=self.model_,
                metrics=self.metrics_,
                feature_importance=self.feature_importance_,
                output_dir=output_path,
            )
        
        logger.info(f"✅ Pipeline saved to {output_path}")
        
        return output_path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "AutoPipeline":
        """
        Load a saved pipeline.
        
        Parameters
        ----------
        path : str or Path
            Path to saved pipeline.
            
        Returns
        -------
        AutoPipeline
            Loaded pipeline.
        """
        import joblib
        
        path = Path(path)
        pipeline_file = path / "pipeline.pkl" if path.is_dir() else path
        
        data = joblib.load(pipeline_file)
        
        pipeline = cls(
            target_col=data["target_col"],
            test_size=data["test_size"],
            n_bins=data["n_bins"],
            binning_method=data["binning_method"],
            selection_method=data["selection_method"],
            n_features=data["n_features"],
            random_state=data["random_state"],
        )
        
        pipeline.preprocessor_ = data["preprocessor"]
        pipeline.binner_ = data["binner"]
        pipeline.selector_ = data["selector"]
        pipeline.model_ = data["model"]
        pipeline.selected_features_ = data["selected_features"]
        pipeline.metrics_ = data.get("metrics")
        
        return pipeline


@time_it
def run_pipeline(
    data_path: Union[str, Path],
    target_col: str,
    output_dir: str = "output",
    **kwargs
) -> Dict[str, Any]:
    """
    Run the complete auto-modeling pipeline (functional API).
    
    Parameters
    ----------
    data_path : str or Path
        Path to input data file.
    target_col : str
        Name of target column.
    output_dir : str, default "output"
        Directory for output files.
    **kwargs
        Additional pipeline parameters.
        
    Returns
    -------
    dict
        Pipeline results.
        
    Example
    -------
    >>> results = run_pipeline("data.csv", "bad_flag", output_dir="results/")
    """
    pipeline = AutoPipeline(target_col=target_col, **kwargs)
    pipeline.fit(data_path)
    metrics = pipeline.evaluate()
    pipeline.save(output_dir)
    
    return {
        "model": pipeline.model_,
        "metrics": metrics,
        "feature_importance": pipeline.feature_importance_,
        "selected_features": pipeline.selected_features_,
        "output_path": Path(output_dir),
        "pipeline": pipeline,
    }
