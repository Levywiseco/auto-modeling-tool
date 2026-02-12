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
) -> float:
    """
    Calculate accuracy score.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
        
    Returns
    -------
    float
        Accuracy score between 0 and 1.
        
    Example
    -------
    >>> acc = accuracy([1, 0, 1, 1], [1, 0, 0, 1])
    >>> print(f"Accuracy: {acc:.2%}")
    Accuracy: 75.00%
    """
    y_true = _to_series(y_true, "y_true")
    y_pred = _to_series(y_pred, "y_pred")
    
    return (y_true == y_pred).mean()


def precision(
    y_true: Union[pl.Series, np.ndarray, List],
    y_pred: Union[pl.Series, np.ndarray, List],
    *,
    pos_label: int = 1,
) -> float:
    """
    Calculate precision score.
    
    Precision = TP / (TP + FP)
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    pos_label : int, default 1
        Positive class label.
        
    Returns
    -------
    float
        Precision score.
    """
    y_true = _to_series(y_true, "y_true")
    y_pred = _to_series(y_pred, "y_pred")
    
    true_positive = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    false_positive = ((y_true != pos_label) & (y_pred == pos_label)).sum()
    
    denominator = true_positive + false_positive
    return true_positive / denominator if denominator > 0 else 0.0


def recall(
    y_true: Union[pl.Series, np.ndarray, List],
    y_pred: Union[pl.Series, np.ndarray, List],
    *,
    pos_label: int = 1,
) -> float:
    """
    Calculate recall (sensitivity) score.
    
    Recall = TP / (TP + FN)
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    pos_label : int, default 1
        Positive class label.
        
    Returns
    -------
    float
        Recall score.
    """
    y_true = _to_series(y_true, "y_true")
    y_pred = _to_series(y_pred, "y_pred")
    
    true_positive = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    false_negative = ((y_true == pos_label) & (y_pred != pos_label)).sum()
    
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator > 0 else 0.0


def f1_score(
    y_true: Union[pl.Series, np.ndarray, List],
    y_pred: Union[pl.Series, np.ndarray, List],
    *,
    pos_label: int = 1,
) -> float:
    """
    Calculate F1 score.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    pos_label : int, default 1
        Positive class label.
        
    Returns
    -------
    float
        F1 score.
    """
    prec = precision(y_true, y_pred, pos_label=pos_label)
    rec = recall(y_true, y_pred, pos_label=pos_label)
    
    denominator = prec + rec
    return 2 * (prec * rec) / denominator if denominator > 0 else 0.0


def confusion_matrix(
    y_true: Union[pl.Series, np.ndarray, List],
    y_pred: Union[pl.Series, np.ndarray, List],
) -> Dict[str, int]:
    """
    Calculate confusion matrix components.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
        
    Returns
    -------
    dict
        Dictionary with keys: TP, TN, FP, FN
        
    Example
    -------
    >>> cm = confusion_matrix([1, 0, 1, 1], [1, 0, 0, 1])
    >>> print(cm)
    {'TP': 2, 'TN': 1, 'FP': 0, 'FN': 1}
    """
    y_true = _to_series(y_true, "y_true")
    y_pred = _to_series(y_pred, "y_pred")
    
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


@time_it
def calculate_auc_roc(
    y_true: Union[pl.Series, np.ndarray, List],
    y_score: Union[pl.Series, np.ndarray, List],
) -> float:
    """
    Calculate Area Under ROC Curve.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities or scores.
        
    Returns
    -------
    float
        AUC-ROC score.
    """
    from sklearn.metrics import roc_auc_score
    
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)
    
    return roc_auc_score(y_true, y_score)


@time_it
def calculate_ks(
    y_true: Union[pl.Series, np.ndarray, List],
    y_score: Union[pl.Series, np.ndarray, List],
) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov statistic.
    
    KS measures the maximum separation between cumulative distributions
    of positive and negative classes.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities.
        
    Returns
    -------
    tuple of (ks_statistic, ks_threshold)
        KS value and the score threshold where it occurs.
        
    Example
    -------
    >>> ks, threshold = calculate_ks(y_true, y_prob)
    >>> print(f"KS: {ks:.4f} at threshold {threshold:.3f}")
    """
    y_true = _to_series(y_true, "y_true")
    y_score = _to_series(y_score, "y_score")
    
    df = pl.DataFrame({
        "target": y_true,
        "score": y_score
    }).sort("score", descending=True)
    
    n_pos = (df["target"] == 1).sum()
    n_neg = (df["target"] == 0).sum()
    
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0
    
    df = df.with_columns([
        (pl.col("target") == 1).cum_sum().alias("cum_pos"),
        (pl.col("target") == 0).cum_sum().alias("cum_neg"),
    ]).with_columns([
        (pl.col("cum_pos") / n_pos).alias("tpr"),
        (pl.col("cum_neg") / n_neg).alias("fpr"),
    ]).with_columns([
        (pl.col("tpr") - pl.col("fpr")).abs().alias("ks_diff")
    ])
    
    max_idx = df["ks_diff"].arg_max()
    ks_stat = df["ks_diff"][max_idx]
    ks_threshold = df["score"][max_idx]
    
    return float(ks_stat), float(ks_threshold)


@time_it
def calculate_gini(
    y_true: Union[pl.Series, np.ndarray, List],
    y_score: Union[pl.Series, np.ndarray, List],
) -> float:
    """
    Calculate Gini coefficient.
    
    Gini = 2 * AUC - 1
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities.
        
    Returns
    -------
    float
        Gini coefficient.
    """
    auc = calculate_auc_roc(y_true, y_score)
    return 2 * auc - 1


@time_it
def calculate_lift(
    y_true: Union[pl.Series, np.ndarray, List],
    y_score: Union[pl.Series, np.ndarray, List],
    *,
    n_bins: int = 10,
) -> pl.DataFrame:
    """
    Calculate lift chart data.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities.
    n_bins : int, default 10
        Number of bins (deciles).
        
    Returns
    -------
    pl.DataFrame
        Lift table with columns: bin, count, bad, bad_rate, cumulative_bad_rate, lift
    """
    y_true = _to_series(y_true, "y_true")
    y_score = _to_series(y_score, "y_score")
    
    df = pl.DataFrame({
        "target": y_true,
        "score": y_score
    }).with_columns([
        pl.col("score").qcut(n_bins, labels=[str(i) for i in range(n_bins)]).alias("bin")
    ])
    
    overall_bad_rate = df["target"].mean()
    
    lift_table = (
        df.group_by("bin")
        .agg([
            pl.count().alias("count"),
            pl.col("target").sum().alias("bad"),
            pl.col("target").mean().alias("bad_rate"),
            pl.col("score").mean().alias("avg_score"),
        ])
        .sort("avg_score", descending=True)
        .with_row_index("rank")
        .with_columns([
            (pl.col("bad").cum_sum() / pl.col("count").cum_sum()).alias("cum_bad_rate"),
            (pl.col("bad_rate") / overall_bad_rate).alias("lift"),
        ])
        .drop("avg_score")
    )
    
    return lift_table


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
def calculate_all_metrics(
    y_true: Union[pl.Series, np.ndarray, List],
    y_pred: Union[pl.Series, np.ndarray, List],
    y_score: Optional[Union[pl.Series, np.ndarray, List]] = None,
) -> Dict[str, float]:
    """
    Calculate all common classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_score : array-like, optional
        Predicted probabilities (for AUC, KS, Gini).
        
    Returns
    -------
    dict
        Dictionary of metric names and values.
        
    Example
    -------
    >>> metrics = calculate_all_metrics(y_true, y_pred, y_prob)
    >>> for name, value in metrics.items():
    ...     print(f"{name}: {value:.4f}")
    """
    logger.info("📊 Calculating all metrics...")
    
    metrics = {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }
    
    cm = confusion_matrix(y_true, y_pred)
    metrics.update({
        "true_positive": cm["TP"],
        "true_negative": cm["TN"],
        "false_positive": cm["FP"],
        "false_negative": cm["FN"],
    })
    
    if y_score is not None:
        metrics["auc_roc"] = calculate_auc_roc(y_true, y_score)
        
        ks, ks_threshold = calculate_ks(y_true, y_score)
        metrics["ks_statistic"] = ks
        metrics["ks_threshold"] = ks_threshold
        
        metrics["gini"] = calculate_gini(y_true, y_score)
    
    logger.info(f"✅ Calculated {len(metrics)} metrics")
    
    return metrics


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
