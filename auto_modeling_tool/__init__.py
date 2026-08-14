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
>>> from auto_modeling_tool import profile_data, profile_risk, Monitor
>>>
>>> report = profile_data(df, target="target", group_col="month")
>>> profile = profile_risk(df, target="target", group_col="month")
>>> monitor = Monitor(binner_params={"n_bins": 10})
>>> mreport = monitor.monitor(df, features=profile.features,
...                           target="target", group_col="month",
...                           binner=profile.binner)

Low-level tools (sklearn-style)
-------------------------------
>>> from auto_modeling_tool.data import load_data, DataPreprocessor
>>> from auto_modeling_tool.binning import WoeBinner
>>> from auto_modeling_tool.features import FeatureSelector
>>> from auto_modeling_tool.evaluation import calculate_all_metrics
"""

__version__ = "3.1.1"
__author__ = "AutoModelTool Team"

# Task-oriented workflow entry points
from .analysis import profile_data, profile_risk

# Binning module
from .binning import (
    WoeBinner,
    calculate_psi,
)

# Core components
from .core import (
    DataTypeError,
    MarsBaseEstimator,
    MarsTransformer,
    NotFittedError,
    ValidationError,
    auto_polars,
    logger,
    time_it,
)

# Data module
from .data import (
    DataPreprocessor,
    load_data,
    stratified_train_test_split,
)

# Evaluation module
from .evaluation import (
    calculate_all_metrics,
    calculate_auc_roc,
    calculate_ks,
)

# Feature module
from .features import (
    FeatureGenerator,
    FeatureSelector,
    calculate_feature_importance,
)

# Monitoring
from .monitoring import AlertConfig, Monitor, generate_monitoring_alert

# Report objects
from .reports import (
    BinningReport,
    DataProfileReport,
    MonitoringReport,
    RiskProfile,
)

# Utils module
from .utils import (
    generate_model_report,
    load_dataframe,
    load_model,
    save_dataframe,
    save_model,
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
