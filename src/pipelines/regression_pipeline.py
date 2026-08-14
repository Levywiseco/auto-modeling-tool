# -*- coding: utf-8 -*-
"""Leakage-safe regression pipeline for continuous targets."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from ..core.exceptions import ValidationError
from ..data.loaders import load_data
from ..data.preprocess import DataPreprocessor
from ..data.split import DatasetSplit, split_dev_oot
from ..evaluation.metrics import calculate_regression_metrics
from ..modeling.artifact import (
    build_regression_artifact,
    score_with_artifact,
)
from ..modeling.train import ModelTrainer


NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


class RegressionPipeline:
    """Dev/OOT regression pipeline with optional log1p target transformation."""

    def __init__(
        self,
        target_col: str,
        model_type: str = "xgboost",
        test_size: float = 0.2,
        random_state: int = 42,
        sample_col: Optional[str] = None,
        date_column: Optional[str] = None,
        oot_start: Optional[Any] = None,
        dev_label: Any = "dev",
        oot_label: Any = "oot",
        target_transform: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        early_stopping_eval: str = "none",
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: Optional[str] = None,
        clean_strategy: str = "median",
        normalize_method: Optional[str] = "zscore",
        export_excel: bool = True,
    ):
        self.target_col = target_col
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.sample_col = sample_col
        self.date_column = date_column
        self.oot_start = oot_start
        self.dev_label = dev_label
        self.oot_label = oot_label
        self.target_transform = target_transform
        self.model_params = dict(model_params or {})
        self.early_stopping_eval = early_stopping_eval
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_metric = early_stopping_metric
        self.clean_strategy = clean_strategy
        self.normalize_method = normalize_method
        self.export_excel = bool(export_excel)

        self.preprocessor_: Optional[DataPreprocessor] = None
        self.model_: Optional[ModelTrainer] = None
        self.metrics_: Optional[Dict[str, float]] = None
        self.feature_columns_: List[str] = []
        self.split_: Optional[DatasetSplit] = None
        self._X_oot_raw: Optional[pl.DataFrame] = None
        self._y_oot: Optional[pl.Series] = None
        self._oot_weight: Optional[np.ndarray] = None

    def fit(self, data: Union[str, Path, pl.DataFrame], **kwargs) -> "RegressionPipeline":
        self.export_excel = bool(kwargs.get("export_excel", self.export_excel))
        if isinstance(data, (str, Path)):
            load_kwargs = {}
            if kwargs.get("encoding") and Path(data).suffix.lower() == ".csv":
                load_kwargs["encoding"] = kwargs["encoding"]
            df = load_data(data, **load_kwargs)
        else:
            df = data
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        if self.target_col not in df.columns:
            raise ValidationError(f"Target column '{self.target_col}' not found in data")

        sample_col = kwargs.get("sample_col", self.sample_col)
        date_column = kwargs.get("date_column", self.date_column)
        self.split_ = split_dev_oot(
            df,
            self.target_col,
            sample_column=sample_col,
            dev_label=kwargs.get("dev_label", self.dev_label),
            oot_label=kwargs.get("oot_label", self.oot_label),
            date_column=date_column,
            oot_start=kwargs.get("oot_start", self.oot_start),
            test_size=self.test_size,
            random_state=self.random_state,
        )
        dev, oot = self.split_.dev, self.split_.oot
        weight_col = kwargs.get("weight_col")
        role_columns = {self.target_col, sample_col, date_column, weight_col}
        requested = kwargs.get("feature_columns")
        excluded = set(kwargs.get("exclude_columns", []))
        if requested is None:
            self.feature_columns_ = [
                column
                for column in dev.columns
                if column not in role_columns
                and column not in excluded
                and dev[column].dtype in NUMERIC_DTYPES
            ]
        else:
            self.feature_columns_ = list(requested)
        if not self.feature_columns_:
            raise ValidationError("Regression requires at least one numeric feature")
        non_numeric = [
            column for column in self.feature_columns_
            if dev[column].dtype not in NUMERIC_DTYPES
        ]
        if non_numeric:
            raise ValidationError(
                f"Regression features must be numeric; got {non_numeric}"
            )

        X_dev_raw = dev.select(self.feature_columns_)
        X_oot_raw = oot.select(self.feature_columns_)
        y_dev = dev.get_column(self.target_col).cast(pl.Float64)
        self._X_oot_raw = X_oot_raw
        self._y_oot = oot.get_column(self.target_col).cast(pl.Float64)

        target_transform = kwargs.get("target_transform", self.target_transform)
        if target_transform not in {None, "log1p"}:
            raise ValidationError("target_transform must be null or 'log1p'")
        y_fit = y_dev.to_numpy()
        if target_transform == "log1p":
            if np.nanmin(y_fit) < 0:
                raise ValidationError("log1p target transform requires non-negative targets")
            y_fit = np.log1p(y_fit)

        self.preprocessor_ = DataPreprocessor(
            clean_strategy=kwargs.get("clean_strategy", self.clean_strategy),
            normalize_method=kwargs.get("normalize_method", self.normalize_method),
            custom_null_values=kwargs.get("custom_null_values"),
        )
        self.preprocessor_.fit(X_dev_raw)
        X_dev = self.preprocessor_.transform(X_dev_raw)
        X_oot = self.preprocessor_.transform(X_oot_raw)

        sample_weight = None
        oot_weight = None
        if weight_col:
            if weight_col not in dev.columns or weight_col not in oot.columns:
                raise ValidationError(f"Weight column '{weight_col}' not found in Dev/OOT data")
            sample_weight = dev.get_column(weight_col).cast(pl.Float64).to_numpy()
            oot_weight = oot.get_column(weight_col).cast(pl.Float64).to_numpy()
            self._oot_weight = oot_weight
            if (
                not np.isfinite(sample_weight).all()
                or (sample_weight <= 0).any()
                or not np.isfinite(oot_weight).all()
                or (oot_weight <= 0).any()
            ):
                raise ValidationError("Sample weights must be finite and strictly positive")

        fit_x = X_dev.to_numpy()
        fit_y = y_fit
        eval_x = None
        eval_y = None
        eval_weight = None
        early_eval = kwargs.get(
            "early_stopping_eval",
            self.early_stopping_eval,
        )
        early_rounds = kwargs.get(
            "early_stopping_rounds",
            self.early_stopping_rounds,
        )
        if (
            early_rounds
            and self.model_type in {"xgboost", "lightgbm", "catboost"}
            and early_eval in {"dev_holdout", "oot"}
        ):
            if early_eval == "oot":
                eval_x = X_oot.to_numpy()
                eval_y = self._y_oot.to_numpy()
                eval_weight = oot_weight
            else:
                indices = np.arange(len(y_fit))
                fit_idx, eval_idx = train_test_split(
                    indices,
                    test_size=0.2,
                    random_state=self.random_state,
                )
                fit_x = X_dev.to_numpy()[fit_idx]
                fit_y = y_fit[fit_idx]
                eval_x = X_dev.to_numpy()[eval_idx]
                eval_y = y_fit[eval_idx]
                if sample_weight is not None:
                    sample_weight = sample_weight[fit_idx]
                    eval_weight = (
                        dev.get_column(weight_col)
                        .cast(pl.Float64)
                        .to_numpy()[eval_idx]
                    )

        model_params = dict(self.model_params)
        if kwargs.get("model_params"):
            model_params.update(kwargs["model_params"])
        if kwargs.get("early_stopping_metric"):
            model_params["eval_metric"] = kwargs["early_stopping_metric"]
        self.model_ = ModelTrainer(
            model_type=self.model_type,
            task="regression",
            random_state=self.random_state,
            **model_params,
        )
        self.model_.fit(
            fit_x,
            fit_y,
            sample_weight=sample_weight,
            eval_set=eval_x,
            eval_y=eval_y,
            eval_sample_weight=eval_weight,
            early_stopping_rounds=early_rounds,
        )
        self.target_transform = target_transform
        return self

    def _predict_raw(self, X: pl.DataFrame) -> np.ndarray:
        if self.model_ is None or self.preprocessor_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        missing = [column for column in self.feature_columns_ if column not in X.columns]
        if missing:
            raise ValidationError(f"Input data is missing driver columns: {missing}")
        transformed = self.preprocessor_.transform(X.select(self.feature_columns_))
        predictions = self.model_.predict(transformed)
        if self.target_transform == "log1p":
            predictions = np.expm1(predictions)
        return np.asarray(predictions)

    def evaluate(
        self,
        X_oot: Optional[pl.DataFrame] = None,
        y_oot: Optional[pl.Series] = None,
    ) -> Dict[str, float]:
        X_oot = self._X_oot_raw if X_oot is None else X_oot
        y_oot = self._y_oot if y_oot is None else y_oot
        if X_oot is None or y_oot is None:
            raise ValidationError("No OOT data is available for evaluation")
        predictions = self._predict_raw(X_oot)
        self.metrics_ = calculate_regression_metrics(
            y_oot,
            predictions,
            sample_weight=self._oot_weight,
        )
        return self.metrics_

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        artifact = self.get_scoring_artifact()
        return score_with_artifact(artifact, X)

    def get_scoring_artifact(self) -> Dict[str, Any]:
        if self.model_ is None or self.preprocessor_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        return build_regression_artifact(
            target_col=self.target_col,
            feature_columns=self.feature_columns_,
            preprocessor=self.preprocessor_,
            model=self.model_,
            target_transform=self.target_transform,
            metadata={
                "split_strategy": self.split_.strategy if self.split_ else None,
                "random_state": self.random_state,
                "metrics": self.metrics_ or {},
            },
        )

    def save(self, output_dir: Union[str, Path]) -> Path:
        if self.model_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        artifact = self.get_scoring_artifact()
        joblib.dump(
            {
                "artifact_version": "1.0",
                "target_col": self.target_col,
                "model_type": self.model_type,
                "model_params": self.model_params,
                "early_stopping_eval": self.early_stopping_eval,
                "early_stopping_rounds": self.early_stopping_rounds,
                "early_stopping_metric": self.early_stopping_metric,
                "feature_columns": self.feature_columns_,
                "preprocessor": self.preprocessor_,
                "model": self.model_,
                "target_transform": self.target_transform,
                "export_excel": self.export_excel,
                "metrics": self.metrics_,
                "scoring_artifact": artifact,
            },
            output / "pipeline.pkl",
        )
        joblib.dump(artifact, output / "scoring_artifact.pkl")
        if self.export_excel:
            from ..reports.excel import write_model_report
            write_model_report(
                output,
                self.metrics_ or {},
                metadata={
                    "target_col": self.target_col,
                    "task": "regression",
                    "artifact_version": "1.0",
                    "split_strategy": self.split_.strategy if self.split_ else None,
                    "target_transform": self.target_transform,
                    "model_type": self.model_type,
                    "early_stopping_eval": self.early_stopping_eval,
                    "early_stopping_rounds": self.early_stopping_rounds,
                },
            )
        return output

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RegressionPipeline":
        path = Path(path)
        data = joblib.load(path / "pipeline.pkl" if path.is_dir() else path)
        pipeline = cls(
            target_col=data["target_col"],
            model_type=data.get("model_type", "xgboost"),
            model_params=data.get("model_params"),
            early_stopping_eval=data.get("early_stopping_eval", "none"),
            early_stopping_rounds=data.get("early_stopping_rounds"),
            early_stopping_metric=data.get("early_stopping_metric"),
            target_transform=data.get("target_transform"),
            export_excel=data.get("export_excel", True),
        )
        pipeline.feature_columns_ = data["feature_columns"]
        pipeline.preprocessor_ = data["preprocessor"]
        pipeline.model_ = data["model"]
        pipeline.metrics_ = data.get("metrics")
        return pipeline


def run_regression_pipeline(
    data_path: Union[str, Path],
    target_col: str,
    output_dir: Union[str, Path] = "output",
    **kwargs,
) -> Dict[str, Any]:
    pipeline = RegressionPipeline(target_col=target_col, **kwargs)
    pipeline.fit(data_path, **kwargs)
    metrics = pipeline.evaluate()
    pipeline.save(output_dir)
    return {"pipeline": pipeline, "metrics": metrics, "output_path": Path(output_dir)}

