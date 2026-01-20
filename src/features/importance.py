# -*- coding: utf-8 -*-
"""
High-performance feature importance module using Polars.

This module provides efficient feature importance calculation and 
visualization using Polars DataFrames.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl

from ..core.logger import logger
from ..core.decorators import time_it


@time_it
def calculate_feature_importance(
    model: Any,
    X: Union[pl.DataFrame, pl.LazyFrame],
    y: Optional[Union[pl.Series, np.ndarray]] = None,
    *,
    method: str = "model",
    n_repeats: int = 10,
) -> pl.DataFrame:
    """
    Calculate feature importance using various methods.
    
    Parameters
    ----------
    model : Any
        Trained machine learning model.
    X : pl.DataFrame or pl.LazyFrame
        Feature dataset.
    y : pl.Series or np.ndarray, optional
        Target variable (required for permutation importance).
    method : str, default "model"
        Importance calculation method:
        - "model": Use model's feature_importances_ attribute
        - "permutation": Permutation importance
        - "shap": SHAP values (requires shap library)
    n_repeats : int, default 10
        Number of repetitions for permutation importance.
        
    Returns
    -------
    pl.DataFrame
        DataFrame with columns: Feature, Importance, Rank
        
    Example
    -------
    >>> importance_df = calculate_feature_importance(model, X)
    >>> importance_df.head()
    """
    if isinstance(X, pl.LazyFrame):
        X = X.collect()
    
    logger.info(f"📊 Calculating feature importance (method={method})")
    
    if method == "model":
        importance = _get_model_importance(model, X)
    elif method == "permutation":
        if y is None:
            raise ValueError("Target 'y' is required for permutation importance")
        importance = _get_permutation_importance(model, X, y, n_repeats)
    elif method == "shap":
        importance = _get_shap_importance(model, X)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Sort by importance
    result = importance.sort("Importance", descending=True)
    
    # Add rank
    result = result.with_row_index("Rank").with_columns(
        (pl.col("Rank") + 1).alias("Rank")
    )
    
    logger.info(f"✅ Calculated importance for {len(result)} features")
    
    return result


def _get_model_importance(
    model: Any, 
    X: pl.DataFrame
) -> pl.DataFrame:
    """Extract feature importance from model."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, use absolute coefficient values
        importance = np.abs(model.coef_).flatten()
        if len(importance) != len(X.columns):
            # Handle multi-class case
            importance = np.mean(np.abs(model.coef_), axis=0)
    else:
        raise ValueError(
            "Model does not have feature_importances_ or coef_ attribute. "
            "Use method='permutation' instead."
        )
    
    return pl.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })


def _get_permutation_importance(
    model: Any,
    X: pl.DataFrame,
    y: Union[pl.Series, np.ndarray],
    n_repeats: int
) -> pl.DataFrame:
    """Calculate permutation importance."""
    from sklearn.inspection import permutation_importance
    
    X_np = X.to_numpy()
    y_np = y.to_numpy() if isinstance(y, pl.Series) else y
    
    result = permutation_importance(
        model, X_np, y_np, 
        n_repeats=n_repeats, 
        n_jobs=-1,
        random_state=42
    )
    
    return pl.DataFrame({
        "Feature": X.columns,
        "Importance": result.importances_mean,
        "Importance_Std": result.importances_std
    })


def _get_shap_importance(
    model: Any,
    X: pl.DataFrame
) -> pl.DataFrame:
    """Calculate SHAP-based importance."""
    try:
        import shap
    except ImportError:
        raise ImportError("shap library is required for SHAP importance. Install with: pip install shap")
    
    X_np = X.to_numpy()
    
    # Try tree explainer first, fall back to kernel
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        # Use sampling for kernel explainer
        background = shap.sample(X_np, min(100, len(X_np)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
    
    shap_values = explainer.shap_values(X_np[:min(1000, len(X_np))])
    
    # Handle multi-output (classification)
    if isinstance(shap_values, list):
        shap_values = np.abs(shap_values[1])  # Use positive class
    
    importance = np.abs(shap_values).mean(axis=0)
    
    return pl.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })


def plot_feature_importance(
    importance_df: pl.DataFrame,
    *,
    top_n: int = 20,
    figsize: tuple = (10, 8),
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters
    ----------
    importance_df : pl.DataFrame
        DataFrame with Feature and Importance columns.
    top_n : int, default 20
        Number of top features to show.
    figsize : tuple, default (10, 8)
        Figure size.
    title : str, default "Feature Importance"
        Plot title.
    save_path : str, optional
        Path to save the figure.
        
    Example
    -------
    >>> plot_feature_importance(importance_df, top_n=15)
    """
    import matplotlib.pyplot as plt
    
    # Get top N features
    top_features = importance_df.sort("Importance", descending=True).head(top_n)
    
    # Reverse for horizontal bar chart (top feature at top)
    top_features = top_features.reverse()
    
    features = top_features.get_column("Feature").to_list()
    importances = top_features.get_column("Importance").to_list()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
    ax.barh(features, importances, color=colors)
    
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Saved plot to {save_path}")
    
    plt.show()


@time_it
def get_importance_summary(
    model: Any,
    X: Union[pl.DataFrame, pl.LazyFrame],
    y: Optional[Union[pl.Series, np.ndarray]] = None,
    *,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Get comprehensive feature importance summary.
    
    Returns
    -------
    dict
        Summary containing:
        - top_features: List of top N feature names
        - importance_df: Full importance DataFrame
        - cumulative_importance: Cumulative importance of top N
        
    Example
    -------
    >>> summary = get_importance_summary(model, X, top_n=10)
    >>> print(summary['top_features'])
    """
    importance_df = calculate_feature_importance(model, X, y)
    
    # Get top features
    top_df = importance_df.head(top_n)
    top_features = top_df.get_column("Feature").to_list()
    
    # Calculate cumulative importance
    total_importance = importance_df.select(pl.col("Importance").sum()).item()
    top_importance = top_df.select(pl.col("Importance").sum()).item()
    cumulative_pct = (top_importance / total_importance * 100) if total_importance > 0 else 0
    
    return {
        "top_features": top_features,
        "importance_df": importance_df,
        "cumulative_importance": cumulative_pct,
        "top_n": top_n,
    }


def rank_features_by_methods(
    model: Any,
    X: Union[pl.DataFrame, pl.LazyFrame],
    y: Optional[Union[pl.Series, np.ndarray]] = None,
) -> pl.DataFrame:
    """
    Rank features using multiple importance methods.
    
    Returns
    -------
    pl.DataFrame
        DataFrame with Feature, Model_Rank, Perm_Rank, Avg_Rank
        
    Example
    -------
    >>> ranks = rank_features_by_methods(model, X, y)
    """
    if isinstance(X, pl.LazyFrame):
        X = X.collect()
    
    results = {}
    
    # Model importance
    try:
        model_imp = calculate_feature_importance(model, X, method="model")
        model_imp = model_imp.with_columns(
            (pl.col("Importance").rank(descending=True)).alias("Model_Rank")
        )
        results["model"] = model_imp
    except Exception as e:
        logger.warning(f"Model importance failed: {e}")
    
    # Permutation importance
    if y is not None:
        try:
            perm_imp = calculate_feature_importance(model, X, y, method="permutation")
            perm_imp = perm_imp.with_columns(
                (pl.col("Importance").rank(descending=True)).alias("Perm_Rank")
            )
            results["perm"] = perm_imp
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
    
    # Merge results
    if "model" in results:
        final = results["model"].select(["Feature", "Model_Rank"])
        
        if "perm" in results:
            final = final.join(
                results["perm"].select(["Feature", "Perm_Rank"]),
                on="Feature",
                how="left"
            )
            final = final.with_columns(
                ((pl.col("Model_Rank") + pl.col("Perm_Rank")) / 2).alias("Avg_Rank")
            )
        else:
            final = final.with_columns(
                pl.col("Model_Rank").alias("Avg_Rank")
            )
        
        return final.sort("Avg_Rank")
    
    return pl.DataFrame()