# -*- coding: utf-8 -*-
"""
Base classes for AutoModelTool estimators and transformers.

This module provides the foundational classes that all AutoModelTool
components inherit from. Inspired by sklearn's design principles and
optimized for Polars-based high-performance computing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Literal

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

from .exceptions import DataTypeError, NotFittedError
from .logger import logger


class MarsBaseEstimator(BaseEstimator):
    """
    Base class for all AutoModelTool estimators.
    
    Provides automatic data type conversion, validation, and output formatting.
    All estimators in AutoModelTool should inherit from this class.
    
    Features
    --------
    - Automatic Pandas -> Polars conversion with type validation
    - Smart output format detection (returns same type as input)
    - Schema validation for safe type conversion
    
    Attributes
    ----------
    _is_fitted : bool
        Whether the estimator has been fitted.
    _output_pandas : bool
        Whether to output Pandas DataFrames (True if input was Pandas).
        
    Notes
    -----
    Child classes should implement `_fit_impl()` and optionally `_transform_impl()`.
    """
    
    # Supported numeric types in Polars
    _PL_NUMERIC_TYPES = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    }
    
    # Supported string/categorical types
    _PL_STRING_TYPES = {pl.Utf8, pl.Categorical, pl.String}
    
    def __init__(self):
        self._is_fitted = False
        self._output_pandas = False
        self._input_columns: List[str] = []
        
    def _determine_output_format(self, input_is_pandas: bool) -> None:
        """Set output format based on input type."""
        self._output_pandas = input_is_pandas
        
    def _ensure_polars_dataframe(self, X: Any) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Convert input to Polars DataFrame with type validation.
        
        Parameters
        ----------
        X : Any
            Input data (Pandas DataFrame, Polars DataFrame/LazyFrame, or dict).
            
        Returns
        -------
        Union[pl.DataFrame, pl.LazyFrame]
            Polars DataFrame or LazyFrame.
            
        Raises
        ------
        DataTypeError
            If input type is not supported.
        """
        # Case 1: Already Polars
        if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            self._determine_output_format(input_is_pandas=False)
            return X
        
        # Case 2: Pandas DataFrame
        elif isinstance(X, pd.DataFrame):
            self._determine_output_format(input_is_pandas=True)
            try:
                X_pl = pl.from_pandas(X)
            except Exception as e:
                logger.warning(f"Standard conversion failed, trying fallback: {e}")
                # Fallback: convert column by column
                X_pl = self._safe_pandas_to_polars(X)
            
            # Validate conversion
            self._validate_conversion(X, X_pl)
            return X_pl
        
        # Case 3: Dictionary
        elif isinstance(X, dict):
            self._determine_output_format(input_is_pandas=False)
            return pl.DataFrame(X)
        
        # Case 4: Numpy array
        elif isinstance(X, np.ndarray):
            self._determine_output_format(input_is_pandas=False)
            if X.ndim == 1:
                return pl.DataFrame({"feature_0": X})
            else:
                cols = {f"feature_{i}": X[:, i] for i in range(X.shape[1])}
                return pl.DataFrame(cols)
        
        else:
            raise DataTypeError(
                f"Unsupported input type: {type(X)}. "
                f"Expected: pandas.DataFrame, polars.DataFrame, dict, or numpy.ndarray"
            )
    
    def _safe_pandas_to_polars(self, df_pd: pd.DataFrame) -> pl.DataFrame:
        """
        Safely convert Pandas DataFrame to Polars, handling edge cases.
        
        Parameters
        ----------
        df_pd : pd.DataFrame
            Input Pandas DataFrame.
            
        Returns
        -------
        pl.DataFrame
            Converted Polars DataFrame.
        """
        schema_dict = {}
        for col in df_pd.columns:
            dtype = df_pd[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                schema_dict[col] = pl.Int64
            elif pd.api.types.is_float_dtype(dtype):
                schema_dict[col] = pl.Float64
            elif pd.api.types.is_bool_dtype(dtype):
                schema_dict[col] = pl.Boolean
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                schema_dict[col] = pl.Datetime
            else:
                schema_dict[col] = pl.Utf8
        
        return pl.DataFrame({
            col: df_pd[col].tolist() for col in df_pd.columns
        }, schema=schema_dict)
    
    def _validate_conversion(self, df_pd: pd.DataFrame, df_pl: pl.DataFrame) -> None:
        """
        Validate that type conversion was successful.
        
        Checks that numeric columns in Pandas remain numeric in Polars.
        Logs warnings for potential data quality issues.
        """
        for col in df_pd.columns:
            pd_dtype = df_pd[col].dtype
            pl_dtype = df_pl[col].dtype
            
            is_pd_numeric = pd.api.types.is_numeric_dtype(pd_dtype)
            is_pl_numeric = pl_dtype in self._PL_NUMERIC_TYPES
            
            if is_pd_numeric and not is_pl_numeric:
                # Allow exception: Int -> Null (all-null column)
                if pl_dtype == pl.Null:
                    continue
                
                # Check if it looks like numeric data stored as string
                sample = df_pd[col].dropna().head(5).tolist()
                if sample:
                    looks_numeric = all(
                        isinstance(s, (int, float)) or 
                        (isinstance(s, str) and s.replace('.', '').replace('-', '').isdigit())
                        for s in sample
                    )
                    if looks_numeric:
                        logger.warning(
                            f"⚠️ Column '{col}' appears numeric but converted to {pl_dtype}. "
                            f"Sample values: {sample[:3]}"
                        )
    
    def _ensure_polars_series(self, y: Any, name: str = "target") -> Optional[pl.Series]:
        """
        Convert target variable to Polars Series.
        
        Parameters
        ----------
        y : Any
            Target variable (array-like, Series, or None).
        name : str
            Name for the resulting Series.
            
        Returns
        -------
        Optional[pl.Series]
            Polars Series or None.
        """
        if y is None:
            return None
        
        if isinstance(y, pl.Series):
            return y.alias(name)
        elif isinstance(y, pd.Series):
            return pl.Series(name, y.values)
        elif isinstance(y, np.ndarray):
            return pl.Series(name, y)
        elif isinstance(y, list):
            return pl.Series(name, y)
        else:
            try:
                return pl.Series(name, list(y))
            except Exception as e:
                raise DataTypeError(f"Cannot convert target to Polars Series: {e}")
    
    def _format_output(self, data: Any) -> Any:
        """
        Format output based on original input type.
        
        Parameters
        ----------
        data : Any
            Output data to format.
            
        Returns
        -------
        Any
            Formatted output (Pandas if input was Pandas, else Polars).
        """
        if self._output_pandas:
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            if isinstance(data, pl.DataFrame):
                return data.to_pandas()
            elif isinstance(data, pl.Series):
                return data.to_pandas()
        return data
    
    def _is_numeric(self, series_or_col: Union[pl.Series, pl.Expr]) -> bool:
        """Check if a column/series is numeric type."""
        if isinstance(series_or_col, pl.Series):
            return series_or_col.dtype in self._PL_NUMERIC_TYPES
        return False
    
    def _is_categorical(self, series_or_col: Union[pl.Series, pl.Expr]) -> bool:
        """Check if a column/series is categorical/string type."""
        if isinstance(series_or_col, pl.Series):
            return series_or_col.dtype in self._PL_STRING_TYPES
        return False
    
    def _get_numeric_columns(self, df: pl.DataFrame) -> List[str]:
        """Get list of numeric column names."""
        return [col for col in df.columns if df[col].dtype in self._PL_NUMERIC_TYPES]
    
    def _get_categorical_columns(self, df: pl.DataFrame) -> List[str]:
        """Get list of categorical/string column names."""
        return [col for col in df.columns if df[col].dtype in self._PL_STRING_TYPES]


class MarsTransformer(MarsBaseEstimator, TransformerMixin, ABC):
    """
    Base class for all AutoModelTool transformers.
    
    Extends MarsBaseEstimator with fit/transform methods following
    sklearn's TransformerMixin pattern.
    
    Features
    --------
    - Template method pattern for fit/transform
    - Automatic type conversion and output formatting
    - Abstract methods for child implementation
    
    Methods
    -------
    fit(X, y=None)
        Fit the transformer to training data.
    transform(X)
        Transform data using the fitted transformer.
    fit_transform(X, y=None)
        Fit and transform in one step.
    """
    
    def fit(self, X: Any, y: Any = None, **kwargs) -> "MarsTransformer":
        """
        Fit the transformer to training data.
        
        Parameters
        ----------
        X : Any
            Training features (Pandas/Polars DataFrame).
        y : Any, optional
            Target variable.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        self
            Fitted transformer instance.
        """
        X_pl = self._ensure_polars_dataframe(X)
        y_pl = self._ensure_polars_series(y) if y is not None else None
        
        self._input_columns = list(X_pl.columns) if isinstance(X_pl, pl.DataFrame) else []
        
        self._fit_impl(X_pl, y_pl, **kwargs)
        self._is_fitted = True
        
        return self
    
    def transform(self, X: Any, **kwargs) -> Any:
        """
        Transform data using the fitted transformer.
        
        Parameters
        ----------
        X : Any
            Data to transform (Pandas/Polars DataFrame).
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        Any
            Transformed data (same type as input).
            
        Raises
        ------
        NotFittedError
            If transformer has not been fitted.
        """
        if not self._is_fitted:
            raise NotFittedError(f"{self.__class__.__name__} is not fitted.")
        
        X_pl = self._ensure_polars_dataframe(X)
        X_new = self._transform_impl(X_pl, **kwargs)
        
        return self._format_output(X_new)
    
    def fit_transform(self, X: Any, y: Any = None, **kwargs) -> Any:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : Any
            Training features.
        y : Any, optional
            Target variable.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        Any
            Transformed data.
        """
        return self.fit(X, y, **kwargs).transform(X, **kwargs)
    
    @abstractmethod
    def _fit_impl(self, X: pl.DataFrame, y: Optional[pl.Series] = None, **kwargs) -> None:
        """
        Core fitting logic to be implemented by child classes.
        
        Parameters
        ----------
        X : pl.DataFrame
            Training data (guaranteed to be Polars DataFrame).
        y : pl.Series, optional
            Target variable.
        **kwargs : dict
            Additional parameters.
        """
        pass
    
    @abstractmethod
    def _transform_impl(self, X: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Core transformation logic to be implemented by child classes.
        
        Parameters
        ----------
        X : pl.DataFrame
            Data to transform (guaranteed to be Polars DataFrame).
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        pl.DataFrame
            Transformed data.
        """
        pass
