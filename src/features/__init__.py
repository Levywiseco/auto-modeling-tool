"""Feature selection, generation, and importance module."""

from .generation import (
    FeatureGenerator,
    generate_binned_features,
    generate_features,
    generate_interaction_features,
    generate_log_features,
    generate_polynomial_features,
    generate_ratio_features,
)
from .importance import (
    calculate_feature_importance,
    plot_feature_importance,
)
from .selection import (
    FeatureSelector,
    select_features,
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
