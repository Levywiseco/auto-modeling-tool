# -*- coding: utf-8 -*-
"""Serializable scoring artifact helpers for raw-driver inference."""

from pathlib import Path
from typing import Any, Dict, Union

import joblib
import numpy as np
import polars as pl

from ..core.exceptions import ValidationError


ARTIFACT_VERSION = "1.0"


def build_scoring_artifact(
    *,
    target_col: str,
    feature_columns: list[str],
    woe_feature_columns: list[str],
    selected_features: list[str],
    preprocessor: Any,
    binner: Any,
    selector: Any,
    model: Any,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Bundle every fitted transform needed for independent scoring."""
    return {
        "artifact_version": ARTIFACT_VERSION,
        "task": "classification",
        "target_col": target_col,
        "feature_columns": list(feature_columns),
        "woe_feature_columns": list(woe_feature_columns),
        "selected_features": list(selected_features),
        "preprocessor": preprocessor,
        "binner": binner,
        "selector": selector,
        "model": model,
        "metadata": metadata or {},
    }


def score_with_artifact(
    artifact: Dict[str, Any],
    X: Union[pl.DataFrame, np.ndarray],
    *,
    return_proba: bool = False,
) -> np.ndarray:
    """Score raw driver data using a saved artifact."""
    required = [
        "preprocessor",
        "binner",
        "selector",
        "model",
        "feature_columns",
        "woe_feature_columns",
    ]
    missing_keys = [key for key in required if key not in artifact]
    if missing_keys:
        raise ValidationError(
            f"Scoring artifact is incomplete; missing keys: {missing_keys}"
        )

    if isinstance(X, np.ndarray):
        X = pl.DataFrame(X, schema=artifact["feature_columns"])
    if not isinstance(X, pl.DataFrame):
        raise TypeError("X must be a Polars DataFrame or NumPy array")

    missing_columns = [
        column for column in artifact["feature_columns"] if column not in X.columns
    ]
    if missing_columns:
        raise ValidationError(
            f"Input data is missing required driver columns: {missing_columns}"
        )

    raw = X.select(artifact["feature_columns"])
    transformed = artifact["preprocessor"].transform(raw)
    woe = artifact["binner"].transform(transformed, return_type="woe")
    woe = woe.select(artifact["woe_feature_columns"])
    selected = artifact["selector"].transform(woe)
    values = selected.to_numpy()

    model = artifact["model"]
    if return_proba:
        if not hasattr(model, "predict_proba"):
            raise ValidationError("The saved model does not expose predict_proba")
        return model.predict_proba(values)[:, 1]
    return model.predict(values)


def save_scoring_artifact(
    artifact: Dict[str, Any],
    path: Union[str, Path],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_scoring_artifact(path: Union[str, Path]) -> Dict[str, Any]:
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "scoring_artifact" in artifact:
        artifact = artifact["scoring_artifact"]
    if not isinstance(artifact, dict) or "artifact_version" not in artifact:
        raise ValidationError("File does not contain a valid scoring artifact")
    return artifact
