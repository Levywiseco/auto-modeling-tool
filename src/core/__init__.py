# -*- coding: utf-8 -*-
"""
Core module for AutoModelTool - Polars-based high-performance implementation.

This module provides base classes and utilities for building ML pipelines
with Polars as the primary data processing engine.
"""

from .base import MarsBaseEstimator, MarsTransformer
from .exceptions import (
    AutoModelError,
    NotFittedError,
    DataTypeError,
    ValidationError,
)
from .decorators import time_it, auto_polars

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
