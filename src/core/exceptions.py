# -*- coding: utf-8 -*-
"""
Custom exceptions for AutoModelTool.

Provides a clear exception hierarchy for better error handling and debugging.
"""


class AutoModelError(Exception):
    """
    Base exception class for AutoModelTool.
    
    All custom exceptions in this package should inherit from this class.
    """
    pass


class NotFittedError(AutoModelError):
    """
    Raised when a transformer or model is used before being fitted.
    
    This exception is raised when transform(), predict(), or similar methods
    are called on an estimator that has not been fitted yet.
    
    Example
    -------
    >>> binner = WoeBinner()
    >>> binner.transform(data)  # Raises NotFittedError
    """
    pass


class DataTypeError(AutoModelError):
    """
    Raised when input data type doesn't match expected type.
    
    This exception is raised when the input data cannot be converted to
    the required Polars DataFrame format.
    
    Example
    -------
    >>> loader.load(123)  # Raises DataTypeError - expects file path
    """
    pass


class ValidationError(AutoModelError):
    """
    Raised when data validation fails.
    
    This exception is raised when input data doesn't meet the required
    conditions (e.g., missing required columns, invalid values).
    
    Example
    -------
    >>> binner.fit(X, y)  # Raises ValidationError if y contains non-binary values
    """
    pass


class ConfigurationError(AutoModelError):
    """
    Raised when configuration is invalid.
    
    This exception is raised when provided configuration parameters
    are invalid or incompatible.
    """
    pass


class FeatureError(AutoModelError):
    """
    Raised for feature-related errors.
    
    This exception is raised when there are issues with feature columns,
    such as missing features or invalid feature names.
    """
    pass
