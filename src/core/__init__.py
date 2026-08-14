"""
Core module for AutoModelTool - Polars-based high-performance implementation.

This module provides base classes and utilities for building ML pipelines
with Polars as the primary data processing engine.
"""

from .base import MarsBaseEstimator, MarsTransformer
from .decorators import auto_polars, time_it
from .exceptions import (
    AutoModelError,
    DataTypeError,
    NotFittedError,
    ValidationError,
)

__all__ = [
    "MarsBaseEstimator",
    "MarsTransformer",
    "AutoModelError",
    "NotFittedError",
    "DataTypeError",
    "ValidationError",
    "time_it",
    "auto_polars",
]
