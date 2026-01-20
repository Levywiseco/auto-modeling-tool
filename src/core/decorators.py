# -*- coding: utf-8 -*-
"""
Utility decorators for AutoModelTool.

Provides performance monitoring and auto-conversion decorators
for seamless Pandas/Polars interoperability.
"""

import functools
import time
from typing import Any, Callable, TypeVar, Union

import pandas as pd
import polars as pl

from .logger import logger

F = TypeVar("F", bound=Callable[..., Any])


def time_it(func: F) -> F:
    """
    Decorator to measure and log the execution time of a function.
    
    Logs the execution time at INFO level, useful for performance profiling
    during development and production monitoring.
    
    Parameters
    ----------
    func : Callable
        The function to be timed.
        
    Returns
    -------
    Callable
        Wrapped function with timing capability.
        
    Example
    -------
    >>> @time_it
    ... def slow_function():
    ...     time.sleep(1)
    >>> slow_function()
    # Logs: "[slow_function] finished in 1.00s"
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        
        # Get class name if method
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            logger.info(f"⏱️ [{class_name}.{func.__name__}] finished in {elapsed:.4f}s")
        else:
            logger.info(f"⏱️ [{func.__name__}] finished in {elapsed:.4f}s")
            
        return result
    
    return wrapper  # type: ignore


def auto_polars(func: F) -> F:
    """
    Decorator for automatic Pandas <-> Polars conversion.
    
    This decorator enables seamless interoperability between Pandas and Polars.
    It automatically:
    1. Detects input data type (Pandas or Polars)
    2. Converts Pandas input to Polars for efficient processing
    3. Converts output back to Pandas if input was Pandas
    
    This allows internal functions to be written purely with Polars
    while maintaining API compatibility with Pandas users.
    
    Parameters
    ----------
    func : Callable
        Function expecting Polars DataFrame as first non-self argument.
        
    Returns
    -------
    Callable
        Wrapped function with auto-conversion capability.
        
    Example
    -------
    >>> @auto_polars
    ... def process_data(self, X: pl.DataFrame) -> pl.DataFrame:
    ...     return X.with_columns(pl.col("a") * 2)
    >>> 
    >>> # Works with both Pandas and Polars input
    >>> result = processor.process_data(pd.DataFrame({"a": [1, 2, 3]}))
    >>> type(result)
    <class 'pandas.core.frame.DataFrame'>
    """
    @functools.wraps(func)
    def wrapper(self: Any, X: Union[pd.DataFrame, pl.DataFrame], *args: Any, **kwargs: Any) -> Any:
        # 1. Type detection: Record if input is Pandas
        is_pandas_input = isinstance(X, pd.DataFrame)
        
        # 2. Unified entry: Force conversion to Polars
        if is_pandas_input:
            # Use Arrow memory layout for fast conversion
            X_pl = pl.from_pandas(X)
        elif isinstance(X, pl.LazyFrame):
            X_pl = X
        else:
            X_pl = X
            
        # 3. Core execution: Call the decorated function
        result = func(self, X_pl, *args, **kwargs)
        
        # 4. Output formatting: Convert back to Pandas if input was Pandas
        if is_pandas_input and isinstance(result, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(result, pl.LazyFrame):
                result = result.collect()
            return result.to_pandas()
        
        return result
    
    return wrapper  # type: ignore


def validate_fitted(func: F) -> F:
    """
    Decorator to check if estimator is fitted before transform/predict.
    
    Raises NotFittedError if the estimator's `_is_fitted` attribute is False.
    
    Parameters
    ----------
    func : Callable
        Method that requires the estimator to be fitted.
        
    Returns
    -------
    Callable
        Wrapped method with fitted check.
    """
    from .exceptions import NotFittedError
    
    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not getattr(self, '_is_fitted', False):
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted. "
                f"Call fit() before calling {func.__name__}()."
            )
        return func(self, *args, **kwargs)
    
    return wrapper  # type: ignore
