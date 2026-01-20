# -*- coding: utf-8 -*-
"""
High-performance I/O utilities using Polars.

This module provides efficient file I/O operations for DataFrames,
models, and configuration management.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl
import joblib

from ..core.logger import logger
from ..core.decorators import time_it


# =============================================================================
# DataFrame I/O
# =============================================================================

@time_it
def save_dataframe(
    df: pl.DataFrame,
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a Polars DataFrame to file.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to save.
    path : str or Path
        Output file path.
    format : str, optional
        Output format: 'csv', 'parquet', 'json', 'excel'.
        If None, inferred from file extension.
    **kwargs
        Additional arguments passed to the writer.
        
    Example
    -------
    >>> save_dataframe(df, "output.parquet")
    >>> save_dataframe(df, "output.csv", separator=";")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format is None:
        format = path.suffix.lstrip('.').lower()
    
    logger.info(f"💾 Saving DataFrame to {path} ({format} format)...")
    
    if format == 'csv':
        df.write_csv(path, **kwargs)
    elif format == 'parquet':
        df.write_parquet(path, **kwargs)
    elif format == 'json':
        df.write_json(path, **kwargs)
    elif format in ('xlsx', 'excel'):
        df.write_excel(path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"✅ Saved {df.shape[0]:,} rows × {df.shape[1]} columns")


@time_it
def load_dataframe(
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    lazy: bool = False,
    **kwargs,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Load a DataFrame from file.
    
    Parameters
    ----------
    path : str or Path
        Input file path.
    format : str, optional
        Input format: 'csv', 'parquet', 'json', 'excel'.
        If None, inferred from file extension.
    lazy : bool, default False
        If True, return LazyFrame for deferred execution.
    **kwargs
        Additional arguments passed to the reader.
        
    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Loaded data.
        
    Example
    -------
    >>> df = load_dataframe("data.parquet")
    >>> lf = load_dataframe("large_data.csv", lazy=True)
    """
    path = Path(path)
    
    if format is None:
        format = path.suffix.lstrip('.').lower()
    
    logger.info(f"📂 Loading DataFrame from {path}...")
    
    if format == 'csv':
        df = pl.scan_csv(path, **kwargs) if lazy else pl.read_csv(path, **kwargs)
    elif format == 'parquet':
        df = pl.scan_parquet(path, **kwargs) if lazy else pl.read_parquet(path, **kwargs)
    elif format == 'json':
        df = pl.read_json(path, **kwargs)
        if lazy:
            df = df.lazy()
    elif format in ('xlsx', 'excel'):
        df = pl.read_excel(path, **kwargs)
        if lazy:
            df = df.lazy()
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    if not lazy:
        logger.info(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df


# =============================================================================
# Model I/O
# =============================================================================

@time_it
def save_model(
    model: Any,
    path: Union[str, Path],
    *,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save a model to disk using joblib.
    
    Parameters
    ----------
    model : Any
        Model object to save.
    path : str or Path
        Output file path.
    metadata : dict, optional
        Additional metadata to save with the model.
        
    Example
    -------
    >>> save_model(model, "model.pkl", metadata={"version": "1.0"})
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 Saving model to {path}...")
    
    save_obj = {
        "model": model,
        "metadata": metadata or {}
    }
    
    joblib.dump(save_obj, path)
    
    logger.info(f"✅ Model saved successfully")


@time_it
def load_model(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a model from disk.
    
    Parameters
    ----------
    path : str or Path
        Path to the saved model.
        
    Returns
    -------
    dict
        Dictionary with 'model' and 'metadata' keys.
        If legacy format (model only), returns dict with model.
        
    Example
    -------
    >>> result = load_model("model.pkl")
    >>> model = result["model"]
    >>> metadata = result["metadata"]
    """
    path = Path(path)
    logger.info(f"📂 Loading model from {path}...")
    
    result = joblib.load(path)
    
    # Handle legacy format (model only, not wrapped in dict)
    if not isinstance(result, dict) or "model" not in result:
        result = {"model": result, "metadata": {}}
    
    logger.info(f"✅ Model loaded successfully")
    return result


# =============================================================================
# Configuration I/O
# =============================================================================

def save_config(
    config: Dict,
    path: Union[str, Path],
) -> None:
    """
    Save configuration to JSON file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    path : str or Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Config saved to {path}")


def load_config(path: Union[str, Path]) -> Dict:
    """
    Load configuration from JSON file.
    
    Parameters
    ----------
    path : str or Path
        Path to JSON config file.
        
    Returns
    -------
    dict
        Configuration dictionary.
    """
    path = Path(path)
    
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    logger.info(f"✅ Config loaded from {path}")
    return config


# =============================================================================
# Binning/Mapping I/O
# =============================================================================

def save_binning(
    bin_edges: Dict[str, List[float]],
    path: Union[str, Path],
) -> None:
    """
    Save binning edges to JSON.
    
    Parameters
    ----------
    bin_edges : dict
        Dictionary mapping feature names to bin edges.
    path : str or Path
        Output file path.
    """
    save_config(bin_edges, path)


def load_binning(path: Union[str, Path]) -> Dict[str, List[float]]:
    """
    Load binning edges from JSON.
    
    Parameters
    ----------
    path : str or Path
        Path to binning JSON file.
        
    Returns
    -------
    dict
        Dictionary mapping feature names to bin edges.
    """
    return load_config(path)


def save_woe_mapping(
    woe_mapping: Dict[str, Dict],
    path: Union[str, Path],
) -> None:
    """
    Save WOE mapping to JSON.
    
    Parameters
    ----------
    woe_mapping : dict
        WOE mapping dictionary.
    path : str or Path
        Output file path.
    """
    save_config(woe_mapping, path)


def load_woe_mapping(path: Union[str, Path]) -> Dict[str, Dict]:
    """
    Load WOE mapping from JSON.
    
    Parameters
    ----------
    path : str or Path
        Path to WOE mapping file.
        
    Returns
    -------
    dict
        WOE mapping dictionary.
    """
    return load_config(path)


# =============================================================================
# Report Generation
# =============================================================================

@time_it
def generate_model_report(
    model: Any,
    metrics: Dict[str, float],
    feature_importance: pl.DataFrame,
    output_dir: Union[str, Path],
    *,
    report_name: str = "model_report",
) -> Path:
    """
    Generate a comprehensive model report.
    
    Parameters
    ----------
    model : Any
        Trained model object.
    metrics : dict
        Model evaluation metrics.
    feature_importance : pl.DataFrame
        Feature importance table.
    output_dir : str or Path
        Output directory.
    report_name : str, default "model_report"
        Report filename (without extension).
        
    Returns
    -------
    Path
        Path to generated report directory.
    """
    output_dir = Path(output_dir) / report_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📝 Generating model report in {output_dir}...")
    
    # Save model
    save_model(model, output_dir / "model.pkl", metadata={"metrics": metrics})
    
    # Save metrics
    save_config(metrics, output_dir / "metrics.json")
    
    # Save feature importance
    save_dataframe(feature_importance, output_dir / "feature_importance.csv")
    
    # Generate summary
    summary = {
        "model_type": type(model).__name__,
        "n_features": len(feature_importance),
        "key_metrics": {
            k: v for k, v in metrics.items()
            if k in ["auc_roc", "ks_statistic", "accuracy", "f1_score"]
        }
    }
    save_config(summary, output_dir / "summary.json")
    
    logger.info(f"✅ Report generated with {len(list(output_dir.iterdir()))} files")
    
    return output_dir


def list_saved_models(directory: Union[str, Path]) -> pl.DataFrame:
    """
    List all saved models in a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search for models.
        
    Returns
    -------
    pl.DataFrame
        Table with model information.
    """
    directory = Path(directory)
    models = []
    
    for pkl_file in directory.rglob("*.pkl"):
        try:
            data = joblib.load(pkl_file)
            
            models.append({
                "path": str(pkl_file),
                "model_type": type(data.get("model")).__name__ if isinstance(data, dict) and "model" in data else type(data).__name__,
                "has_metadata": isinstance(data, dict) and bool(data.get("metadata")),
                "size_kb": pkl_file.stat().st_size / 1024,
            })
        except Exception as e:
            logger.warning(f"Could not load {pkl_file}: {e}")
    
    return pl.DataFrame(models)