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
from ..data.split import DatasetSplit, split_dev_oot

from ..binning.woe_binning import WoeBinner

from ..features.selection import FeatureSelector
from ..features.importance import calculate_feature_importance

from ..evaluation.metrics import calculate_all_metrics

from ..modeling.artifact import build_scoring_artifact, score_with_artifact
from ..utils.io import generate_model_report


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
        sample_col: Optional[str] = None,
        date_column: Optional[str] = None,
        oot_start: Optional[Any] = None,
        dev_label: Any = "dev",
        oot_label: Any = "oot",
    ):
        self.target_col = target_col
        self.test_size = test_size
        self.n_bins = n_bins
        self.binning_method = binning_method
        self.selection_method = selection_method
        self.n_features = n_features
        self.random_state = random_state
        self.sample_col = sample_col
        self.date_column = date_column
        self.oot_start = oot_start
        self.dev_label = dev_label
        self.oot_label = oot_label

        self.preprocessor_: Optional[DataPreprocessor] = None
        self.binner_: Optional[WoeBinner] = None
        self.selector_: Optional[FeatureSelector] = None
        self.model_: Optional[Any] = None
        self.metrics_: Optional[Dict[str, float]] = None
        self.selected_features_: List[str] = []
        self.woe_feature_columns_: List[str] = []
        self.feature_columns_: List[str] = []
        self.feature_importance_: Optional[pl.DataFrame] = None
        self.split_: Optional[DatasetSplit] = None
        self._X_oot_raw: Optional[pl.DataFrame] = None
        self._y_oot: Optional[pl.Series] = None
    
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
            load_kwargs = {}
            if kwargs.get("encoding") and Path(data).suffix.lower() == ".csv":
                load_kwargs["encoding"] = kwargs["encoding"]
            df = load_data(data, **load_kwargs)
        else:
            df = data
        
        if self.target_col not in df.columns:
            raise ValidationError(f"Target column '{self.target_col}' not found in data")
        
        logger.info(f"   Data shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        logger.info("\n✂️ Creating Dev/OOT split...")
        sample_col = kwargs.get("sample_col", self.sample_col)
        date_column = kwargs.get("date_column", self.date_column)
        oot_start = kwargs.get("oot_start", self.oot_start)
        dev_label = kwargs.get("dev_label", self.dev_label)
        oot_label = kwargs.get("oot_label", self.oot_label)

        self.split_ = split_dev_oot(
            df,
            self.target_col,
            sample_column=sample_col,
            dev_label=dev_label,
            oot_label=oot_label,
            date_column=date_column,
            oot_start=oot_start,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        dev = self.split_.dev
        oot = self.split_.oot

        role_columns = {
            self.target_col,
            sample_col,
            date_column,
            kwargs.get("weight_col"),
        }
        requested_features = kwargs.get("feature_columns")
        excluded = set(kwargs.get("exclude_columns", []))
        if requested_features is None:
            feature_columns = [
                column for column in dev.columns
                if column not in role_columns and column not in excluded
            ]
        else:
            feature_columns = list(requested_features)
            missing = [column for column in feature_columns if column not in dev.columns]
            if missing:
                raise ValidationError(
                    f"Requested feature columns are missing from Dev data: {missing}"
                )
        if not feature_columns:
            raise ValidationError("No model feature columns remain after role exclusions")
        if any(column not in oot.columns for column in feature_columns):
            raise ValidationError("Dev/OOT feature schemas do not match")
        self.feature_columns_ = feature_columns

        X_dev_raw = dev.select(feature_columns)
        X_oot_raw = oot.select(feature_columns)
        y_dev = dev.get_column(self.target_col)
        y_oot = oot.get_column(self.target_col)
        self._X_oot_raw = X_oot_raw
        self._y_oot = y_oot
        self.sample_col = sample_col
        self.date_column = date_column
        self.oot_start = oot_start
        self.dev_label = dev_label
        self.oot_label = oot_label
        self._X_test = X_oot_raw
        self._y_test = y_oot
        y_train = y_dev
        y_test = y_oot

        # Critical invariant: preprocessing statistics are learned on Dev only.
        self.preprocessor_ = DataPreprocessor(
            clean_strategy=kwargs.get("clean_strategy", "median"),
            normalize_method=kwargs.get("normalize_method", "zscore"),
            custom_null_values=kwargs.get("custom_null_values"),
        )
        self.preprocessor_.fit(X_dev_raw, y_dev)
        X_train = self.preprocessor_.transform(X_dev_raw)
        X_test = self.preprocessor_.transform(X_oot_raw)

        logger.info("\n📊 WOE Binning...")
        self.binner_ = WoeBinner(
            n_bins=self.n_bins,
            method=self.binning_method,
            min_samples_bin=kwargs.get("min_samples_bin", 50),
            monotonic=kwargs.get("monotonic", False),
        )
        X_train_woe = self.binner_.fit_transform(X_train, y_train, return_type="woe")
        
        logger.info("\n🎯 Feature selection...")
        self.selector_ = FeatureSelector(
            method=self.selection_method,
            n_features=self.n_features,
            iv_threshold=0.02,
        )
        
        self.woe_feature_columns_ = [
            column for column in X_train_woe.columns
            if column.endswith("_bin") and column[:-4] in self.feature_columns_
        ]
        if not self.woe_feature_columns_:
            raise ValidationError("WOE binning did not produce usable feature columns")

        X_train_woe_selected = X_train_woe.select(self.woe_feature_columns_)
        X_train_selected = self.selector_.fit_transform(
            X_train_woe_selected,
            y_train,
        )
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
        """Evaluate on OOT by default; external X must contain raw drivers."""
        if self.model_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        X_test = self._X_oot_raw if X_test is None else X_test
        y_test = self._y_oot if y_test is None else y_test
        if X_test is None or y_test is None:
            raise ValidationError(
                "No held-out OOT data is available; provide X_test and y_test explicitly"
            )

        X_test_selected = self._transform_selected(X_test)
        y_test_np = y_test.to_numpy()
        y_pred = self.model_.predict(X_test_selected.to_numpy())
        y_prob = self.model_.predict_proba(X_test_selected.to_numpy())[:, 1]
        self.metrics_ = calculate_all_metrics(y_test_np, y_pred, y_prob)
        return self.metrics_

    def _transform_selected(self, X: pl.DataFrame) -> pl.DataFrame:
        if self.model_ is None or self.preprocessor_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        missing = [column for column in self.feature_columns_ if column not in X.columns]
        if missing:
            raise ValidationError(f"Input data is missing driver columns: {missing}")
        raw = X.select(self.feature_columns_)
        transformed = self.preprocessor_.transform(raw)
        woe = self.binner_.transform(transformed, return_type="woe")
        return self.selector_.transform(woe.select(self.woe_feature_columns_))

    def predict(
        self,
        X: pl.DataFrame,
        return_proba: bool = False,
    ) -> np.ndarray:
        return score_with_artifact(
            self.get_scoring_artifact(),
            X,
            return_proba=return_proba,
        )

    def get_scoring_artifact(self) -> Dict[str, Any]:
        if self.model_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        return build_scoring_artifact(
            target_col=self.target_col,
            feature_columns=self.feature_columns_,
            woe_feature_columns=self.woe_feature_columns_,
            selected_features=self.selected_features_,
            preprocessor=self.preprocessor_,
            binner=self.binner_,
            selector=self.selector_,
            model=self.model_,
            metadata={
                "split_strategy": self.split_.strategy if self.split_ else None,
                "random_state": self.random_state,
            },
        )

    def save(self, output_dir: Union[str, Path]) -> Path:
        """Save both the compatibility pipeline and raw-driver artifact."""
        import joblib

        if self.model_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        scoring_artifact = self.get_scoring_artifact()
        pipeline_data = {
            "artifact_version": "1.0",
            "target_col": self.target_col,
            "test_size": self.test_size,
            "n_bins": self.n_bins,
            "binning_method": self.binning_method,
            "selection_method": self.selection_method,
            "n_features": self.n_features,
            "random_state": self.random_state,
            "sample_col": self.sample_col,
            "date_column": self.date_column,
            "oot_start": self.oot_start,
            "dev_label": self.dev_label,
            "oot_label": self.oot_label,
            "preprocessor": self.preprocessor_,
            "binner": self.binner_,
            "selector": self.selector_,
            "model": self.model_,
            "feature_columns": self.feature_columns_,
            "woe_feature_columns": self.woe_feature_columns_,
            "selected_features": self.selected_features_,
            "feature_importance": self.feature_importance_,
            "metrics": self.metrics_,
            "scoring_artifact": scoring_artifact,
        }
        joblib.dump(pipeline_data, output_path / "pipeline.pkl")
        joblib.dump(scoring_artifact, output_path / "scoring_artifact.pkl")

        if self.metrics_ and self.feature_importance_ is not None:
            generate_model_report(
                model=self.model_,
                metrics=self.metrics_,
                feature_importance=self.feature_importance_,
                output_dir=output_path,
            )
        logger.info(f"Pipeline saved to {output_path}")
        return output_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AutoPipeline":
        import joblib

        path = Path(path)
        pipeline_file = path / "pipeline.pkl" if path.is_dir() else path
        data = joblib.load(pipeline_file)
        pipeline = cls(
            target_col=data["target_col"],
            test_size=data.get("test_size", 0.2),
            n_bins=data.get("n_bins", 10),
            binning_method=data.get("binning_method", "quantile"),
            selection_method=data.get("selection_method", "iv"),
            n_features=data.get("n_features", 20),
            random_state=data.get("random_state", 42),
            sample_col=data.get("sample_col"),
            date_column=data.get("date_column"),
            oot_start=data.get("oot_start"),
            dev_label=data.get("dev_label", "dev"),
            oot_label=data.get("oot_label", "oot"),
        )
        pipeline.preprocessor_ = data["preprocessor"]
        pipeline.binner_ = data["binner"]
        pipeline.selector_ = data["selector"]
        pipeline.model_ = data["model"]
        pipeline.feature_columns_ = data.get(
            "feature_columns",
            getattr(pipeline.preprocessor_, "feature_columns_", []),
        )
        pipeline.woe_feature_columns_ = data.get(
            "woe_feature_columns",
            [f"{feature}_bin" for feature in pipeline.feature_columns_],
        )
        pipeline.selected_features_ = data.get("selected_features", [])
        pipeline.feature_importance_ = data.get("feature_importance")
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
