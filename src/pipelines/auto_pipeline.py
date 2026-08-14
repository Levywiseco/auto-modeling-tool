# -*- coding: utf-8 -*-
"""Leakage-safe, configuration-driven classification pipeline."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from ..binning.woe_binning import WoeBinner
from ..core.decorators import time_it
from ..core.exceptions import ValidationError
from ..core.logger import logger
from ..data.loaders import load_data
from ..data.preprocess import DataPreprocessor
from ..data.split import DatasetSplit, split_dev_oot
from ..evaluation.metrics import calculate_all_metrics, calculate_lift, calculate_psi
from ..features.importance import calculate_feature_importance
from ..features.selection import FeatureSelector
from ..modeling.artifact import build_scoring_artifact, score_with_artifact
from ..modeling.train import ModelTrainer
from ..utils.io import generate_model_report


def _validated_weight(
    frame: pl.DataFrame,
    weight_col: Optional[str],
    enabled: bool,
) -> Optional[pl.Series]:
    """Return a validated weight vector or None when weighting is disabled."""
    if not enabled:
        return None
    if not weight_col:
        raise ValidationError(
            "use_sample_weight=true requires shared.weight_col or --weight-column"
        )
    if weight_col not in frame.columns:
        raise ValidationError(f"Weight column '{weight_col}' not found")
    weights = frame.get_column(weight_col).cast(pl.Float64)
    values = weights.to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValidationError("Sample weights must be finite and strictly positive")
    return weights


class AutoPipeline:
    """End-to-end classification pipeline with auditable Dev/OOT artifacts."""

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
        model_type: str = "logistic",
        model_params: Optional[Dict[str, Any]] = None,
        early_stopping_eval: str = "none",
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: Optional[str] = None,
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
        self.model_type = str(model_type).lower()
        self.model_params = dict(model_params or {})
        self.early_stopping_eval = early_stopping_eval
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_metric = early_stopping_metric

        self.preprocessor_: Optional[DataPreprocessor] = None
        self.binner_: Optional[WoeBinner] = None
        self.selector_: Optional[FeatureSelector] = None
        self.model_: Optional[Any] = None
        self.metrics_: Optional[Dict[str, float]] = None
        self.dev_metrics_: Optional[Dict[str, float]] = None
        self.selected_features_: List[str] = []
        self.woe_feature_columns_: List[str] = []
        self.feature_columns_: List[str] = []
        self.feature_importance_: Optional[pl.DataFrame] = None
        self.split_: Optional[DatasetSplit] = None
        self.weight_col_: Optional[str] = None
        self.use_sample_weight_: bool = False
        self._X_dev_raw: Optional[pl.DataFrame] = None
        self._X_oot_raw: Optional[pl.DataFrame] = None
        self._X_dev_transformed: Optional[pl.DataFrame] = None
        self._X_oot_transformed: Optional[pl.DataFrame] = None
        self._X_train_selected: Optional[pl.DataFrame] = None
        self._X_oot_selected: Optional[pl.DataFrame] = None
        self._y_train: Optional[pl.Series] = None
        self._y_oot: Optional[pl.Series] = None
        self._weight_dev: Optional[pl.Series] = None
        self._weight_oot: Optional[pl.Series] = None
        self.report_tables_: Dict[str, Any] = {}
        self.segment_cols_: List[str] = []
        self.temporal_col_: Optional[str] = None
        self.benchmark_cols_: List[str] = []
        self._dev_frame: Optional[pl.DataFrame] = None
        self._oot_frame: Optional[pl.DataFrame] = None

    @time_it
    def fit(
        self,
        data: Union[str, Path, pl.DataFrame],
        **kwargs,
    ) -> "AutoPipeline":
        logger.info("=" * 60)
        logger.info("🚀 Starting AutoPipeline Training")
        logger.info("=" * 60)

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
        dev = self.split_.dev
        oot = self.split_.oot
        self._dev_frame = dev
        self._oot_frame = oot

        self.segment_cols_ = list(kwargs.get("segment_cols") or [])
        self.temporal_col_ = kwargs.get("temporal_col")
        self.benchmark_cols_ = list(kwargs.get("benchmark_cols") or [])
        requested_eval_cols = self.segment_cols_ + (
            [self.temporal_col_] if self.temporal_col_ else []
        ) + self.benchmark_cols_
        missing_eval_cols = [
            column for column in requested_eval_cols
            if column not in df.columns
        ]
        if missing_eval_cols:
            raise ValidationError(
                f"Evaluation columns are missing from input data: {missing_eval_cols}"
            )

        weight_col = kwargs.get("weight_col")
        use_sample_weight = bool(kwargs.get("use_sample_weight", False))
        dev_weight = _validated_weight(dev, weight_col, use_sample_weight)
        oot_weight = _validated_weight(oot, weight_col, use_sample_weight)

        role_columns = {self.target_col, sample_col, date_column, weight_col}
        requested_features = kwargs.get("feature_columns")
        excluded = set(kwargs.get("exclude_columns", []))
        if requested_features is None:
            feature_columns = [
                column
                for column in dev.columns
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
        self.weight_col_ = weight_col
        self.use_sample_weight_ = use_sample_weight
        self.sample_col = sample_col
        self.date_column = date_column
        self.oot_start = kwargs.get("oot_start", self.oot_start)
        self.dev_label = kwargs.get("dev_label", self.dev_label)
        self.oot_label = kwargs.get("oot_label", self.oot_label)
        self._X_dev_raw = dev.select(feature_columns)
        self._X_oot_raw = oot.select(feature_columns)
        self._y_train = dev.get_column(self.target_col)
        self._y_oot = oot.get_column(self.target_col)
        self._weight_dev = dev_weight
        self._weight_oot = oot_weight

        self.preprocessor_ = DataPreprocessor(
            clean_strategy=kwargs.get("clean_strategy", "median"),
            normalize_method=kwargs.get("normalize_method", "zscore"),
            custom_null_values=kwargs.get("custom_null_values"),
        )
        self.preprocessor_.fit(self._X_dev_raw, self._y_train)
        self._X_dev_transformed = self.preprocessor_.transform(self._X_dev_raw)
        self._X_oot_transformed = self.preprocessor_.transform(self._X_oot_raw)

        self.binner_ = WoeBinner(
            n_bins=self.n_bins,
            method=self.binning_method,
            min_samples_bin=kwargs.get("min_samples_bin", 50),
            monotonic=kwargs.get("monotonic", False),
            smoothing=kwargs.get("smoothing", 0.5),
        )
        X_train_woe = self.binner_.fit_transform(
            self._X_dev_transformed,
            self._y_train,
            return_type="woe",
            sample_weight=self._weight_dev,
        )
        X_oot_woe = self.binner_.transform(
            self._X_oot_transformed,
            return_type="woe",
        )

        self.woe_feature_columns_ = [
            column
            for column in X_train_woe.columns
            if column.endswith("_bin") and column[:-4] in self.feature_columns_
        ]
        if not self.woe_feature_columns_:
            raise ValidationError("WOE binning did not produce usable feature columns")

        self.selector_ = FeatureSelector(
            method=self.selection_method,
            n_features=self.n_features,
            iv_threshold=0.02,
        )
        X_train_selected = self.selector_.fit_transform(
            X_train_woe.select(self.woe_feature_columns_),
            self._y_train,
            sample_weight=self._weight_dev,
        )
        X_oot_selected = self.selector_.transform(
            X_oot_woe.select(self.woe_feature_columns_)
        )
        if not X_train_selected.columns:
            raise ValidationError("Feature selection returned no usable features")
        self.selected_features_ = self.selector_.get_selected_features()
        self._X_train_selected = X_train_selected
        self._X_oot_selected = X_oot_selected

        train_x = X_train_selected.to_numpy()
        train_y = self._y_train.to_numpy()
        train_weight = (
            self._weight_dev.to_numpy() if self._weight_dev is not None else None
        )
        eval_x = None
        eval_y = None
        eval_weight = None
        fit_x = train_x
        fit_y = train_y
        fit_weight = train_weight
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
                eval_x = X_oot_selected.to_numpy()
                eval_y = self._y_oot.to_numpy()
                eval_weight = (
                    self._weight_oot.to_numpy()
                    if self._weight_oot is not None
                    else None
                )
            else:
                indices = np.arange(len(train_y))
                stratify = train_y if len(np.unique(train_y)) > 1 else None
                fit_idx, eval_idx = train_test_split(
                    indices,
                    test_size=0.2,
                    random_state=self.random_state,
                    stratify=stratify,
                )
                fit_x = train_x[fit_idx]
                fit_y = train_y[fit_idx]
                eval_x = train_x[eval_idx]
                eval_y = train_y[eval_idx]
                if train_weight is not None:
                    fit_weight = train_weight[fit_idx]
                    eval_weight = train_weight[eval_idx]

        model_params = dict(self.model_params)
        if kwargs.get("model_params"):
            model_params.update(kwargs["model_params"])
        if kwargs.get("early_stopping_metric"):
            model_params["eval_metric"] = kwargs["early_stopping_metric"]
        self.model_ = ModelTrainer(
            model_type=kwargs.get("model_type", self.model_type),
            task="classification",
            random_state=self.random_state,
            **model_params,
        )
        self.model_.fit(
            fit_x,
            fit_y,
            sample_weight=fit_weight,
            eval_set=eval_x,
            eval_y=eval_y,
            eval_sample_weight=eval_weight,
            early_stopping_rounds=early_rounds,
        )

        self.feature_importance_ = calculate_feature_importance(
            model=self.model_,
            X=X_train_selected,
            y=self._y_train,
            method="model",
        )
        self.report_tables_ = self._build_report_tables()
        logger.info("✅ Pipeline training completed!")
        return self

    def _evaluate_selected(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        sample_weight: Optional[pl.Series] = None,
    ) -> Dict[str, float]:
        if self.model_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        y_pred = self.model_.predict(X.to_numpy())
        y_prob = self.model_.predict_proba(X.to_numpy())[:, 1]
        return calculate_all_metrics(
            y.to_numpy(),
            y_pred,
            y_prob,
            sample_weight=(
                sample_weight.to_numpy()
                if sample_weight is not None
                else None
            ),
        )

    @time_it
    def evaluate(
        self,
        X_test: Optional[pl.DataFrame] = None,
        y_test: Optional[pl.Series] = None,
    ) -> Dict[str, float]:
        """Evaluate on OOT by default; external X must contain raw drivers."""
        if self.model_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        if X_test is None:
            X_test_selected = self._X_oot_selected
            y_test = self._y_oot if y_test is None else y_test
            sample_weight = self._weight_oot
        else:
            X_test_selected = self._transform_selected(X_test)
            sample_weight = None
        if X_test_selected is None or y_test is None:
            raise ValidationError("No held-out OOT data is available")
        self.metrics_ = self._evaluate_selected(
            X_test_selected,
            y_test,
            sample_weight,
        )
        self.dev_metrics_ = self._evaluate_selected(
            self._X_train_selected,
            self._y_train,
            self._weight_dev,
        )
        self.report_tables_ = self._build_report_tables()
        return self.metrics_

    def _transform_selected(self, X: pl.DataFrame) -> pl.DataFrame:
        if self.model_ is None or self.preprocessor_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        missing = [column for column in self.feature_columns_ if column not in X.columns]
        if missing:
            raise ValidationError(f"Input data is missing driver columns: {missing}")
        transformed = self.preprocessor_.transform(X.select(self.feature_columns_))
        woe = self.binner_.transform(transformed, return_type="woe")
        return self.selector_.transform(woe.select(self.woe_feature_columns_))

    def _build_report_tables(self) -> Dict[str, Any]:
        """Build stable audit tables mirroring the guide's report contract."""
        tables: Dict[str, Any] = {}
        if self.binner_ is not None and self._X_dev_transformed is not None:
            try:
                tables["Binning_Summary"] = self.binner_.compute_bin_stats(
                    self._X_dev_transformed,
                    self._y_train,
                    sample_weight=self._weight_dev,
                )
                tables["IV_Summary"] = self.binner_.get_iv_report()
                tables["WOE_Detail"] = [
                    {
                        "feature": feature,
                        "bin_idx": bin_idx,
                        "woe": woe,
                        "iv": self.binner_.bin_ivs_.get(feature, {}).get(bin_idx),
                        "label": self.binner_.bin_mappings_.get(feature, {}).get(
                            bin_idx
                        ),
                    }
                    for feature, mapping in self.binner_.bin_woes_.items()
                    for bin_idx, woe in mapping.items()
                ]
            except Exception as exc:
                logger.warning(f"Could not build binning tables: {exc}")

        audit_rows = []
        iv_values = self.binner_.total_iv_ if self.binner_ is not None else {}
        importance = {}
        if self.feature_importance_ is not None:
            importance = {
                row["Feature"]: row["Importance"]
                for row in self.feature_importance_.to_dicts()
            }
        for feature in self.feature_columns_:
            audit_rows.append({
                "feature": feature,
                "selected": feature in self.selected_features_
                or f"{feature}_bin" in self.selected_features_,
                "iv": iv_values.get(feature),
                "importance": importance.get(f"{feature}_bin"),
                "dtype": (
                    str(self._X_dev_raw[feature].dtype)
                    if self._X_dev_raw is not None
                    else None
                ),
            })
        tables["Variable_Audit"] = audit_rows
        tables["Selection_Report"] = [
            {
                "feature": feature,
                "selected": feature in self.selected_features_,
                "selection_method": self.selection_method,
                "iv": iv_values.get(
                    feature[:-4] if feature.endswith("_bin") else feature
                ),
            }
            for feature in self.woe_feature_columns_
        ]

        if self.dev_metrics_ is not None:
            tables["Dev_Metrics"] = self.dev_metrics_
        if self.metrics_ is not None:
            tables["OOT_Metrics"] = self.metrics_
        if (
            self._X_train_selected is not None
            and self._X_oot_selected is not None
            and self._y_train is not None
            and self._y_oot is not None
        ):
            try:
                dev_prob = self.model_.predict_proba(
                    self._X_train_selected.to_numpy()
                )[:, 1]
                oot_prob = self.model_.predict_proba(
                    self._X_oot_selected.to_numpy()
                )[:, 1]
                tables["Dev_Score_Bins"] = calculate_lift(
                    self._y_train,
                    dev_prob,
                    sample_weight=self._weight_dev,
                )
                tables["OOT_Score_Bins"] = calculate_lift(
                    self._y_oot,
                    oot_prob,
                    sample_weight=self._weight_oot,
                )
                psi, psi_table = calculate_psi(
                    dev_prob,
                    oot_prob,
                    n_bins=10,
                    expected_weight=self._weight_dev,
                    actual_weight=self._weight_oot,
                )
                tables["Score_PSI"] = psi_table
                tables["Stability_Summary"] = {
                    "score_psi_dev_oot": psi,
                    "status": (
                        "critical" if psi >= 0.25
                        else "warning" if psi >= 0.1
                        else "stable"
                    ),
                }
            except Exception as exc:
                logger.warning(f"Could not build score/stability tables: {exc}")

        if self._oot_frame is not None and self._X_oot_selected is not None:
            segment_rows = []
            for segment_col in self.segment_cols_:
                for value in self._oot_frame[segment_col].unique().to_list():
                    mask = self._oot_frame[segment_col] == value
                    group_x = self._X_oot_selected.filter(mask)
                    group_y = self._y_oot.filter(mask)
                    group_weight = (
                        self._weight_oot.filter(mask)
                        if self._weight_oot is not None
                        else None
                    )
                    if len(group_y) == 0 or len(np.unique(group_y.to_numpy())) < 2:
                        continue
                    row = {
                        "segment_col": segment_col,
                        "segment": value,
                        "n_rows": len(group_y),
                    }
                    row.update(self._evaluate_selected(group_x, group_y, group_weight))
                    segment_rows.append(row)
            if segment_rows:
                tables["Segment_Summary"] = segment_rows

            if self.temporal_col_:
                temporal_rows = []
                for value in self._oot_frame[self.temporal_col_].unique().sort().to_list():
                    mask = self._oot_frame[self.temporal_col_] == value
                    group_x = self._X_oot_selected.filter(mask)
                    group_y = self._y_oot.filter(mask)
                    if len(group_y) == 0 or len(np.unique(group_y.to_numpy())) < 2:
                        continue
                    row = {
                        "period": value,
                        "n_rows": len(group_y),
                    }
                    row.update(
                        self._evaluate_selected(
                            group_x,
                            group_y,
                            self._weight_oot.filter(mask)
                            if self._weight_oot is not None
                            else None,
                        )
                    )
                    temporal_rows.append(row)
                if temporal_rows:
                    tables["Temporal_Stability"] = temporal_rows

            if self.benchmark_cols_:
                benchmark_rows = []
                for column in self.benchmark_cols_:
                    values = self._oot_frame[column].to_numpy()
                    valid = np.isfinite(values)
                    if valid.sum() == 0:
                        continue
                    y_values = self._y_oot.to_numpy()[valid]
                    prediction = (values[valid] >= 0.5).astype(int)
                    weights = (
                        self._weight_oot.to_numpy()[valid]
                        if self._weight_oot is not None
                        else None
                    )
                    row = {
                        "benchmark": column,
                        "n_rows": int(valid.sum()),
                    }
                    row.update(
                        calculate_all_metrics(
                            y_values,
                            prediction,
                            values[valid],
                            sample_weight=weights,
                        )
                    )
                    benchmark_rows.append(row)
                if benchmark_rows:
                    tables["Benchmark_Performance"] = benchmark_rows

        if self.model_ is not None:
            try:
                tables["Model_Estimation"] = self.model_.get_model_summary()
            except Exception:
                tables["Model_Estimation"] = {
                    "model_type": type(self.model_).__name__,
                }
        return tables

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
                "model_type": self.model_type,
                "use_sample_weight": self.use_sample_weight_,
                "weight_col": self.weight_col_,
            },
        )

    def save(self, output_dir: Union[str, Path]) -> Path:
        """Save pipeline, scoring artifact, and audit workbook."""
        import joblib

        if self.model_ is None:
            raise ValidationError("Pipeline not fitted. Call fit() first.")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        scoring_artifact = self.get_scoring_artifact()
        pipeline_data = {
            "artifact_version": "1.1",
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
            "model_type": self.model_type,
            "model_params": self.model_params,
            "early_stopping_eval": self.early_stopping_eval,
            "early_stopping_rounds": self.early_stopping_rounds,
            "early_stopping_metric": self.early_stopping_metric,
            "weight_col": self.weight_col_,
            "use_sample_weight": self.use_sample_weight_,
            "preprocessor": self.preprocessor_,
            "binner": self.binner_,
            "selector": self.selector_,
            "model": self.model_,
            "feature_columns": self.feature_columns_,
            "woe_feature_columns": self.woe_feature_columns_,
            "selected_features": self.selected_features_,
            "feature_importance": self.feature_importance_,
            "metrics": self.metrics_,
            "dev_metrics": self.dev_metrics_,
            "report_tables": self.report_tables_,
            "segment_cols": self.segment_cols_,
            "temporal_col": self.temporal_col_,
            "benchmark_cols": self.benchmark_cols_,
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
        from ..reports.excel import write_model_report
        write_model_report(
            output_path,
            self.metrics_ or {},
            feature_importance=self.feature_importance_,
            metadata={
                "target_col": self.target_col,
                "artifact_version": "1.1",
                "split_strategy": self.split_.strategy if self.split_ else None,
                "model_type": self.model_type,
                "selected_features": self.selected_features_,
                "use_sample_weight": self.use_sample_weight_,
            },
            tables=self.report_tables_,
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
            model_type=data.get("model_type", "logistic"),
            model_params=data.get("model_params"),
            early_stopping_eval=data.get("early_stopping_eval", "none"),
            early_stopping_rounds=data.get("early_stopping_rounds"),
            early_stopping_metric=data.get("early_stopping_metric"),
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
        pipeline.dev_metrics_ = data.get("dev_metrics")
        pipeline.weight_col_ = data.get("weight_col")
        pipeline.use_sample_weight_ = data.get("use_sample_weight", False)
        pipeline.report_tables_ = data.get("report_tables", {})
        pipeline.segment_cols_ = data.get("segment_cols", [])
        pipeline.temporal_col_ = data.get("temporal_col")
        pipeline.benchmark_cols_ = data.get("benchmark_cols", [])
        return pipeline


@time_it
def run_pipeline(
    data_path: Union[str, Path],
    target_col: str,
    output_dir: str = "output",
    **kwargs,
) -> Dict[str, Any]:
    """Functional API for the classification pipeline."""
    pipeline = AutoPipeline(target_col=target_col, **kwargs)
    pipeline.fit(data_path, **kwargs)
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
