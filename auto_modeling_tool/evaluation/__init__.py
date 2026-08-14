"""Model evaluation metrics module."""

from .cross_validation import (
    CrossValidator,
    cross_validate_model,
    stratified_kfold_cv,
    timeseries_cv,
)
from .metrics import (
    accuracy,
    calculate_all_metrics,
    calculate_auc_roc,
    calculate_feature_psi,
    calculate_gini,
    calculate_ks,
    calculate_lift,
    calculate_psi,
    confusion_matrix,
    f1_score,
    format_metrics_table,
    precision,
    recall,
)
from .quality_gate import (
    ReleaseCheck,
    ReleaseValidationResult,
    validate_release,
)
from .stability import (
    bin_distribution,
    psi_from_distributions,
    psi_level,
)

__all__ = [
    # Metrics
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "calculate_auc_roc",
    "calculate_ks",
    "calculate_gini",
    "calculate_lift",
    "calculate_psi",
    "calculate_feature_psi",
    "calculate_all_metrics",
    "format_metrics_table",
    # Cross-validation
    "CrossValidator",
    "cross_validate_model",
    "stratified_kfold_cv",
    "timeseries_cv",
    # Stability
    "bin_distribution",
    "psi_from_distributions",
    "psi_level",
    "ReleaseCheck",
    "ReleaseValidationResult",
    "validate_release",
]
