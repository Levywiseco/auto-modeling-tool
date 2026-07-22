# -*- coding: utf-8 -*-
"""
AutoModelTool - High-Performance Auto-Modeling Framework

A comprehensive machine learning automation toolkit built with Polars
for high-performance data processing.

Features
--------
- High-performance data loading with Polars LazyFrame support
- WOE binning with multiple methods (quantile, uniform, CART)
- Feature selection and generation
- Model evaluation with comprehensive metrics
- Export and report generation

Task-oriented entry points (start here)
---------------------------------------
>>> from src import profile_data, profile_risk, Monitor
>>>
>>> report = profile_data(df, target="target", group_col="month")
>>> profile = profile_risk(df, target="target", group_col="month")
>>> monitor = Monitor(binner_params={"n_bins": 10})
>>> mreport = monitor.monitor(df, features=profile.features,
...                           target="target", group_col="month",
...                           binner=profile.binner)

Low-level tools (sklearn-style)
-------------------------------
>>> from src.data import load_data, DataPreprocessor
>>> from src.binning import WoeBinner
>>> from src.features import FeatureSelector
>>> from src.evaluation import calculate_all_metrics
"""

__version__ = "2.1.0"
__author__ = "AutoModelTool Team"

# Task-oriented workflow entry points
from .analysis import profile_data, profile_risk

# Monitoring
from .monitoring import AlertConfig, Monitor, generate_monitoring_alert

# Report objects
from .reports import (
    BinningReport,
    DataProfileReport,
    MonitoringReport,
    RiskProfile,
)

# Core components
from .core import (
    MarsBaseEstimator,
    MarsTransformer,
    NotFittedError,
    DataTypeError,
    ValidationError,
    time_it,
    auto_polars,
    logger,
)

# Data module
from .data import (
    load_data,
    DataPreprocessor,
    stratified_train_test_split,
)

# Binning module
from .binning import (
    WoeBinner,
    calculate_psi,
)

# Feature module
from .features import (
    FeatureSelector,
    FeatureGenerator,
    calculate_feature_importance,
)

# Evaluation module
from .evaluation import (
    calculate_all_metrics,
    calculate_ks,
    calculate_auc_roc,
)

# Utils module
from .utils import (
    save_model,
    load_model,
    save_dataframe,
    load_dataframe,
    generate_model_report,
)

__all__ = [
    # Version
    "__version__",
    # Workflows
    "profile_data",
    "profile_risk",
    # Monitoring
    "Monitor",
    "AlertConfig",
    "generate_monitoring_alert",
    # Report objects
    "DataProfileReport",
    "BinningReport",
    "RiskProfile",
    "MonitoringReport",
    # Core
    "MarsBaseEstimator",
    "MarsTransformer",
    "NotFittedError",
    "DataTypeError",
    "ValidationError",
    "time_it",
    "auto_polars",
    "logger",
    # Data
    "load_data",
    "DataPreprocessor",
    "stratified_train_test_split",
    # Binning
    "WoeBinner",
    "calculate_psi",
    # Features
    "FeatureSelector",
    "FeatureGenerator",
    "calculate_feature_importance",
    # Evaluation
    "calculate_all_metrics",
    "calculate_ks",
    "calculate_auc_roc",
    # Utils
    "save_model",
    "load_model",
    "save_dataframe",
    "load_dataframe",
    "generate_model_report",
]
