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


# =============================================================================
# Basic Metrics (Polars-optimized)
# =============================================================================

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


# =============================================================================
# Advanced Metrics
# =============================================================================

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
    
    # Create DataFrame for vectorized calculation
    df = pl.DataFrame({
        "target": y_true,
        "score": y_score
    }).sort("score", descending=True)
    
    n_pos = (df["target"] == 1).sum()
    n_neg = (df["target"] == 0).sum()
    
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0
    
    # Calculate cumulative distributions
    df = df.with_columns([
        (pl.col("target") == 1).cum_sum().alias("cum_pos"),
        (pl.col("target") == 0).cum_sum().alias("cum_neg"),
    ]).with_columns([
        (pl.col("cum_pos") / n_pos).alias("tpr"),
        (pl.col("cum_neg") / n_neg).alias("fpr"),
    ]).with_columns([
        (pl.col("tpr") - pl.col("fpr")).abs().alias("ks_diff")
    ])
    
    # Find max KS
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
    
    # Probability-based metrics
    if y_score is not None:
        metrics["auc_roc"] = calculate_auc_roc(y_true, y_score)
        
        ks, ks_threshold = calculate_ks(y_true, y_score)
        metrics["ks_statistic"] = ks
        metrics["ks_threshold"] = ks_threshold
        
        metrics["gini"] = calculate_gini(y_true, y_score)
    
    logger.info(f"✅ Calculated {len(metrics)} metrics")
    
    return metrics


# =============================================================================
# Helper Functions
# =============================================================================

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