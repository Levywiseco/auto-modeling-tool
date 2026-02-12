# -*- coding: utf-8 -*-
"""
Utility helper functions for AutoModelTool.

This module provides common conversion and validation functions
used across the package.
"""

from typing import Any, List, Union, Optional

import numpy as np
import polars as pl


NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64
}


def to_polars_series(
    data: Any,
    name: str = "data",
    dtype: Optional[pl.DataType] = None
) -> pl.Series:
    """
    Convert any array-like to Polars Series.
    
    Parameters
    ----------
    data : Any
        Input data (list, numpy array, pandas Series, or Polars Series).
    name : str, default "data"
        Name for the resulting Series.
    dtype : pl.DataType, optional
        Target data type.
        
    Returns
    -------
    pl.Series
        Converted Polars Series.
        
    Example
    -------
    >>> s = to_polars_series([1, 2, 3], "values")
    >>> s = to_polars_series(np.array([1, 2, 3]), "values")
    """
    if isinstance(data, pl.Series):
        return data.alias(name) if name != data.name else data
    
    if isinstance(data, np.ndarray):
        return pl.Series(name, data, dtype=dtype)
    
    if hasattr(data, '__iter__') and not isinstance(data, str):
        return pl.Series(name, list(data), dtype=dtype)
    
    raise TypeError(f"Cannot convert {type(data)} to Polars Series")


def to_numpy(data: Any) -> np.ndarray:
    """
    Convert any array-like to numpy array.
    
    Parameters
    ----------
    data : Any
        Input data (list, numpy array, pandas Series, or Polars Series).
        
    Returns
    -------
    np.ndarray
        Converted numpy array.
        
    Example
    -------
    >>> arr = to_numpy([1, 2, 3])
    >>> arr = to_numpy(pl.Series("a", [1, 2, 3]))
    """
    if isinstance(data, np.ndarray):
        return data
    
    if isinstance(data, pl.Series):
        return data.to_numpy()
    
    if isinstance(data, list):
        return np.array(data)
    
    if hasattr(data, 'to_numpy'):
        return data.to_numpy()
    
    return np.array(data)


def get_numeric_columns(df: pl.DataFrame) -> List[str]:
    """
    Get list of numeric column names from DataFrame.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
        
    Returns
    -------
    List[str]
        List of numeric column names.
    """
    return [col for col in df.columns if df[col].dtype in NUMERIC_DTYPES]


def get_categorical_columns(df: pl.DataFrame) -> List[str]:
    """
    Get list of categorical/string column names from DataFrame.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
        
    Returns
    -------
    List[str]
        List of categorical column names.
    """
    STRING_DTYPES = {pl.Utf8, pl.String, pl.Categorical}
    return [col for col in df.columns if df[col].dtype in STRING_DTYPES]


def validate_binary_target(
    y: Union[pl.Series, np.ndarray],
    name: str = "target"
) -> None:
    """
    Validate that target variable is binary.
    
    Parameters
    ----------
    y : pl.Series or np.ndarray
        Target variable.
    name : str, default "target"
        Name for error messages.
        
    Raises
    ------
    ValueError
        If target is not binary.
    """
    if isinstance(y, pl.Series):
        n_unique = y.n_unique()
        null_count = y.null_count()
    else:
        y = np.array(y)
        n_unique = len(np.unique(y[~np.isnan(y)]))
        null_count = np.isnan(y).sum()
    
    if null_count > 0:
        raise ValueError(f"{name} contains {null_count} null values")
    
    if n_unique != 2:
        raise ValueError(
            f"{name} must be binary (exactly 2 unique values), "
            f"found {n_unique} unique values"
        )


def validate_no_nulls(
    data: Union[pl.DataFrame, pl.Series],
    name: str = "data"
) -> None:
    """
    Validate that data contains no null values.
    
    Parameters
    ----------
    data : pl.DataFrame or pl.Series
        Data to validate.
    name : str, default "data"
        Name for error messages.
        
    Raises
    ------
    ValueError
        If data contains null values.
    """
    if isinstance(data, pl.DataFrame):
        null_counts = data.null_count()
        for col in data.columns:
            if null_counts[col][0] > 0:
                raise ValueError(f"Column '{col}' in {name} contains null values")
    elif isinstance(data, pl.Series):
        if data.null_count() > 0:
            raise ValueError(f"{name} contains {data.null_count()} null values")


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    fill_value: float = 0.0,
    eps: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Safe division that handles division by zero.
    
    Parameters
    ----------
    numerator : float or np.ndarray
        Numerator.
    denominator : float or np.ndarray
        Denominator.
    fill_value : float, default 0.0
        Value to use when denominator is zero.
    eps : float, default 1e-10
        Small value to prevent division by zero.
        
    Returns
    -------
    float or np.ndarray
        Result of division.
    """
    if isinstance(denominator, np.ndarray):
        result = np.where(
            np.abs(denominator) < eps,
            fill_value,
            numerator / denominator
        )
    else:
        result = numerator / denominator if abs(denominator) >= eps else fill_value
    
    return result


def ensure_dataframe(
    data: Any,
    name: str = "data"
) -> pl.DataFrame:
    """
    Ensure input is a Polars DataFrame.
    
    Parameters
    ----------
    data : Any
        Input data.
    name : str, default "data"
        Name for error messages.
        
    Returns
    -------
    pl.DataFrame
        Polars DataFrame.
        
    Raises
    ------
    TypeError
        If input cannot be converted to DataFrame.
    """
    if isinstance(data, pl.DataFrame):
        return data
    
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    
    if isinstance(data, dict):
        return pl.DataFrame(data)
    
    if hasattr(data, 'to_pandas'):
        return pl.from_pandas(data.to_pandas())
    
    if hasattr(data, 'to_dict'):
        return pl.DataFrame(data.to_dict())
    
    raise TypeError(
        f"Cannot convert {type(data)} to Polars DataFrame. "
        f"Expected pl.DataFrame, dict, or object with to_pandas() method."
    )
