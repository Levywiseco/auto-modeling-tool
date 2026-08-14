# -*- coding: utf-8 -*-
"""
High-performance evaluation metrics module using Polars.

This module provides efficient calculation of classification metrics
using Polars' vectorized operations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from ..core.logger import logger
from ..core.decorators import time_it


def accuracy(
    y_true: Union[pl.Series, np.ndarray, List],
    y_pred: Union[pl.Series, np.ndarray, List],
    *,
    sample_weight: Optional[Union[pl.Series, np.ndarray, List]] = None,
) -> float:
    """Calculate (optionally weighted) accuracy."""
    true_values = _to_numpy(y_true)
    pred_values = _to_numpy(y_pred)
    weights = _validated_weights(sample_weight, len(true_values))
    return float(np.average(true_values == pred_values, weights=weights))



def precision(
    y_true: Union[pl.Series, np.ndarray, List],
    y_pred: Union[pl.Series, np.ndarray, List],
    *,
    pos_label: int = 1,
    sample_weight: Optional[Union[pl.Series, np.ndarray, List]] = None,
) -> float:
    """Calculate weighted precision."""
    true_values = _to_numpy(y_true)
    pred_values = _to_numpy(y_pred)
    weights = _validated_weights(sample_weight, len(true_values))
    tp = weights[(true_values == pos_label) & (pred_values == pos_label)].sum()
    fp = weights[(true_values != pos_label) & (pred_values == pos_label)].sum()
    return float(tp / (tp + fp)) if tp + fp > 0 else 0.0



def recall(
    y_true: Union[pl.Series, np.ndarray, List],
    y_pred: Union[pl.Series, np.ndarray, List],
    *,
    pos_label: int = 1,
    sample_weight: Optional[Union[pl.Series, np.ndarray, List]] = None,
) -> float:
    """Calculate weighted recall."""
    true_values = _to_numpy(y_true)
    pred_values = _to_numpy(y_pred)
    weights = _validated_weights(sample_weight, len(true_values))
    tp = weights[(true_values == pos_label) & (pred_values == pos_label)].sum()
    fn = weights[(true_values == pos_label) & (pred_values != pos_label)].sum()
    return float(tp / (tp + fn)) if tp + fn > 0 else 0.0



def f1_score(
    y_true: Union[pl.Series, np.ndarray, List],
    y_pred: Union[pl.Series, np.ndarray, List],
    *,
    pos_label: int = 1,
    sample_weight: Optional[Union[pl.Series, np.ndarray, List]] = None,
) -> float:
    """Calculate weighted F1."""
    prec = precision(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
    rec = recall(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
    return float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0



def confusion_matrix(
    y_true: Union[pl.Series, np.ndarray, List],
    y_pred: Union[pl.Series, np.ndarray, List],
    *,
    sample_weight: Optional[Union[pl.Series, np.ndarray, List]] = None,
) -> Dict[str, float]:
    """Return weighted confusion-matrix components."""
    true_values = _to_numpy(y_true)
    pred_values = _to_numpy(y_pred)
    weights = _validated_weights(sample_weight, len(true_values))
    return {
        "TP": float(weights[(true_values == 1) & (pred_values == 1)].sum()),
        "TN": float(weights[(true_values == 0) & (pred_values == 0)].sum()),
        "FP": float(weights[(true_values == 0) & (pred_values == 1)].sum()),
        "FN": float(weights[(true_values == 1) & (pred_values == 0)].sum()),
    }



@time_it
def calculate_auc_roc(
    y_true: Union[pl.Series, np.ndarray, List],
    y_score: Union[pl.Series, np.ndarray, List],
    *,
    sample_weight: Optional[Union[pl.Series, np.ndarray, List]] = None,
) -> float:
    """Calculate weighted ROC AUC."""
    from sklearn.metrics import roc_auc_score
    return float(
        roc_auc_score(
            _to_numpy(y_true),
            _to_numpy(y_score),
            sample_weight=None if sample_weight is None else _to_numpy(sample_weight),
        )
    )



@time_it
def calculate_ks(
    y_true: Union[pl.Series, np.ndarray, List],
    y_score: Union[pl.Series, np.ndarray, List],
    *,
    sample_weight: Optional[Union[pl.Series, np.ndarray, List]] = None,
) -> Tuple[float, float]:
    """Calculate weighted Kolmogorov-Smirnov statistic."""
    target = _to_numpy(y_true).astype(float)
    score = _to_numpy(y_score).astype(float)
    weights = _validated_weights(sample_weight, len(target))
    order = np.argsort(-score)
    target, score, weights = target[order], score[order], weights[order]
    pos_weight = float(np.sum(weights * target))
    neg_weight = float(np.sum(weights * (1.0 - target)))
    if pos_weight <= 0 or neg_weight <= 0:
        return 0.0, 0.0
    tpr = np.cumsum(weights * target) / pos_weight
    fpr = np.cumsum(weights * (1.0 - target)) / neg_weight
    differences = np.abs(tpr - fpr)
    idx = int(np.argmax(differences))
    return float(differences[idx]), float(score[idx])



@time_it
def calculate_gini(
    y_true: Union[pl.Series, np.ndarray, List],
    y_score: Union[pl.Series, np.ndarray, List],
    *,
    sample_weight: Optional[Union[pl.Series, np.ndarray, List]] = None,
) -> float:
    """Calculate weighted Gini."""
    auc = calculate_auc_roc(y_true, y_score, sample_weight=sample_weight)
    return float(2 * auc - 1)



@time_it
def calculate_lift(
    y_true: Union[pl.Series, np.ndarray, List],
    y_score: Union[pl.Series, np.ndarray, List],
    *,
    n_bins: int = 10,
    sample_weight: Optional[Union[pl.Series, np.ndarray, List]] = None,
) -> pl.DataFrame:
    """Calculate a weighted decile/lift table."""
    y_values = _to_numpy(y_true)
    score_values = _to_numpy(y_score)
    weights = _validated_weights(sample_weight, len(y_values))
    df = pl.DataFrame({
        "target": y_values,
        "score": score_values,
        "weight": weights,
    }).with_columns(
        pl.col("score").qcut(
            n_bins,
            labels=[str(i) for i in range(n_bins)],
        ).alias("bin")
    )
    overall_bad_rate = float(
        np.average(y_values, weights=weights)
    ) if len(y_values) else 0.0
    table = (
        df.with_columns([
            (pl.col("target") * pl.col("weight")).alias("bad_weight"),
        ])
        .group_by("bin")
        .agg([
            pl.col("weight").sum().alias("count"),
            pl.col("bad_weight").sum().alias("bad"),
            pl.col("score").mean().alias("avg_score"),
        ])
        .with_columns([
            (pl.col("bad") / pl.col("count")).alias("bad_rate"),
        ])
        .sort("avg_score", descending=True)
        .with_row_index("rank")
        .with_columns([
            (pl.col("bad").cum_sum() / pl.col("count").cum_sum()).alias("cum_bad_rate"),
            (pl.col("bad_rate") / overall_bad_rate).alias("lift"),
        ])
        .drop("avg_score")
    )
    return table



@time_it
def calculate_psi(
    expected: Union[pl.Series, np.ndarray, List],
    actual: Union[pl.Series, np.ndarray, List],
    *,
    n_bins: int = 10,
    bin_type: str = "quantile",
    epsilon: float = 1e-10,
) -> Tuple[float, pl.DataFrame]:
    """
    Calculate Population Stability Index (PSI).
    
    PSI measures the shift in distribution between two populations
    (e.g., training vs. validation data).
    
    PSI Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.25: Moderate change, investigation needed
    - PSI >= 0.25: Significant change, action required
    
    Parameters
    ----------
    expected : array-like
        Expected (baseline) distribution (e.g., training data).
    actual : array-like
        Actual distribution to compare (e.g., validation data).
    n_bins : int, default 10
        Number of bins for distribution comparison.
    bin_type : str, default "quantile"
        Binning strategy: "quantile" or "uniform".
    epsilon : float, default 1e-10
        Small value to prevent division by zero.
        
    Returns
    -------
    tuple of (psi_value, psi_table)
        PSI value and detailed PSI table by bin.
        
    Example
    -------
    >>> psi, table = calculate_psi(train_scores, test_scores)
    >>> print(f"PSI: {psi:.4f}")
    >>> if psi < 0.1:
    ...     print("No significant population shift")
    """
    expected = _to_series(expected, "expected")
    actual = _to_series(actual, "actual")
    
    n_expected = len(expected)
    n_actual = len(actual)
    
    if bin_type == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = [expected.quantile(q) for q in quantiles]
        bin_edges[0] = float('-inf')
        bin_edges[-1] = float('inf')
    else:
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        bin_edges[0] = float('-inf')
        bin_edges[-1] = float('inf')
    
    psi_total = 0.0
    psi_data = []
    
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        if i == n_bins - 1:
            exp_count = ((expected >= lower) & (expected <= upper)).sum()
            act_count = ((actual >= lower) & (actual <= upper)).sum()
        else:
            exp_count = ((expected >= lower) & (expected < upper)).sum()
            act_count = ((actual >= lower) & (actual < upper)).sum()
        
        exp_pct = (exp_count + epsilon) / n_expected
        act_pct = (act_count + epsilon) / n_actual
        
        psi_bin = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        psi_total += psi_bin
        
        psi_data.append({
            "bin": i + 1,
            "lower": lower if lower != float('-inf') else None,
            "upper": upper if upper != float('inf') else None,
            "expected_count": int(exp_count),
            "actual_count": int(act_count),
            "expected_pct": exp_pct,
            "actual_pct": act_pct,
            "psi": psi_bin,
        })
    
    psi_table = pl.DataFrame(psi_data)
    
    logger.info(f"📊 PSI: {psi_total:.4f}")
    
    return psi_total, psi_table


@time_it
def calculate_feature_psi(
    expected_df: pl.DataFrame,
    actual_df: pl.DataFrame,
    features: Optional[List[str]] = None,
    *,
    n_bins: int = 10,
) -> pl.DataFrame:
    """
    Calculate PSI for multiple features.
    
    Parameters
    ----------
    expected_df : pl.DataFrame
        Expected (baseline) DataFrame.
    actual_df : pl.DataFrame
        Actual DataFrame to compare.
    features : list of str, optional
        Features to calculate PSI for. If None, uses all numeric columns.
    n_bins : int, default 10
        Number of bins for PSI calculation.
        
    Returns
    -------
    pl.DataFrame
        PSI table with columns: feature, psi, interpretation
        
    Example
    -------
    >>> psi_df = calculate_feature_psi(train_df, test_df)
    >>> print(psi_df.filter(pl.col("psi") >= 0.25))
    """
    if features is None:
        NUMERIC_DTYPES = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64
        }
        features = [c for c in expected_df.columns if expected_df[c].dtype in NUMERIC_DTYPES]
    
    def interpret_psi(psi: float) -> str:
        if psi < 0.1:
            return "Stable"
        elif psi < 0.25:
            return "Moderate Shift"
        else:
            return "Significant Shift"
    
    results = []
    
    for feature in features:
        if feature not in expected_df.columns or feature not in actual_df.columns:
            continue
        
        try:
            psi_val, _ = calculate_psi(
                expected_df[feature],
                actual_df[feature],
                n_bins=n_bins
            )
            results.append({
                "feature": feature,
                "psi": psi_val,
                "interpretation": interpret_psi(psi_val),
            })
        except Exception as e:
            logger.warning(f"PSI calculation failed for {feature}: {e}")
    
    return pl.DataFrame(results).sort("psi", descending=True)


@time_it
def calculate_regression_metrics(
    y_true: Union[pl.Series, np.ndarray, List],
    y_pred: Union[pl.Series, np.ndarray, List],
    *,
    sample_weight: Optional[Union[pl.Series, np.ndarray, List]] = None,
) -> Dict[str, float]:
    """Calculate RMSE, MAE and R-squared for continuous targets."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    true_values = _to_numpy(y_true).astype(float)
    pred_values = _to_numpy(y_pred).astype(float)
    weights = None if sample_weight is None else _to_numpy(sample_weight).astype(float)
    return {
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    true_values,
                    pred_values,
                    sample_weight=weights,
                )
            )
        ),
        "mae": float(
            mean_absolute_error(
                true_values,
                pred_values,
                sample_weight=weights,
            )
        ),
        "r2": float(r2_score(true_values, pred_values, sample_weight=weights)),
    }


@time_it
def calculate_all_metrics(
    y_true: Union[pl.Series, np.ndarray, List],
    y_pred: Union[pl.Series, np.ndarray, List],
    y_score: Optional[Union[pl.Series, np.ndarray, List]] = None,
    *,
    task: str = "classification",
    sample_weight: Optional[Union[pl.Series, np.ndarray, List]] = None,
) -> Dict[str, float]:
    """Calculate common weighted classification or regression metrics."""
    logger.info("📊 Calculating all metrics...")
    if task == "regression":
        return calculate_regression_metrics(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
    if task != "classification":
        raise ValueError(f"Unknown task: {task}")

    metrics = {
        "accuracy": accuracy(y_true, y_pred, sample_weight=sample_weight),
        "precision": precision(y_true, y_pred, sample_weight=sample_weight),
        "recall": recall(y_true, y_pred, sample_weight=sample_weight),
        "f1_score": f1_score(y_true, y_pred, sample_weight=sample_weight),
    }
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    metrics.update({
        "true_positive": cm["TP"],
        "true_negative": cm["TN"],
        "false_positive": cm["FP"],
        "false_negative": cm["FN"],
    })
    if y_score is not None:
        metrics["auc_roc"] = calculate_auc_roc(
            y_true, y_score, sample_weight=sample_weight
        )
        ks, ks_threshold = calculate_ks(
            y_true, y_score, sample_weight=sample_weight
        )
        metrics["ks_statistic"] = ks
        metrics["ks_threshold"] = ks_threshold
        metrics["gini"] = calculate_gini(
            y_true, y_score, sample_weight=sample_weight
        )
    logger.info(f"✅ Calculated {len(metrics)} metrics")
    return metrics



def _validated_weights(
    sample_weight: Optional[Union[pl.Series, np.ndarray, List]],
    n_rows: int,
) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_rows, dtype=float)
    values = _to_numpy(sample_weight).astype(float)
    if len(values) != n_rows:
        raise ValueError("sample_weight must have the same length as y")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("sample_weight must be finite and strictly positive")
    return values


def _to_series(data: Union[pl.Series, np.ndarray, List], name: str = "data") -> pl.Series:
    """Convert input to Polars Series."""
    if isinstance(data, pl.Series):
        return data
    return pl.Series(name, data)


def _to_numpy(data: Union[pl.Series, np.ndarray, List]) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(data, pl.Series):
        return data.to_numpy()
    elif isinstance(data, list):
        return np.array(data)
    return data


def format_metrics_table(metrics: Dict[str, float]) -> pl.DataFrame:
    """
    Format metrics dictionary as a Polars DataFrame table.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary from calculate_all_metrics.
        
    Returns
    -------
    pl.DataFrame
        Formatted table with Metric and Value columns.
    """
    return pl.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": list(metrics.values())
    })
