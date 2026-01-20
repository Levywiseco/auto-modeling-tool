# -*- coding: utf-8 -*-
"""Feature selection, generation, and importance module."""

from .selection import (
    select_features,
    FeatureSelector,
)
from .generation import (
    generate_polynomial_features,
    generate_interaction_features,
    generate_ratio_features,
    generate_log_features,
    generate_binned_features,
    generate_features,
    FeatureGenerator,
)
from .importance import (
    calculate_feature_importance,
    plot_feature_importance,
)

__all__ = [
    # Selection
    "select_features",
    "FeatureSelector",
    # Generation
    "generate_polynomial_features",
    "generate_interaction_features",
    "generate_ratio_features",
    "generate_log_features",
    "generate_binned_features",
    "generate_features",
    "FeatureGenerator",
    # Importance
    "calculate_feature_importance",
    "plot_feature_importance",
]
