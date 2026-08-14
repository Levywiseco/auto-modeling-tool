"""
High-performance WOE (Weight of Evidence) binning module using Polars.

This module provides a fast, vectorized implementation of WOE binning
for credit risk modeling, inspired by the mars-risk project.
"""

from typing import Any, Literal, Optional

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier

from ..core.base import MarsTransformer
from ..core.decorators import time_it
from ..core.exceptions import NotFittedError, ValidationError
from ..core.logger import logger


class WoeBinner(MarsTransformer):
    """
    High-performance WOE (Weight of Evidence) Binner using Polars.

    This class implements efficient binning and WOE calculation using
    Polars' vectorized operations. Supports quantile, uniform, and
    decision tree (CART) based binning methods.

    Parameters
    ----------
    n_bins : int, default 5
        Number of bins for each feature.
    method : str, default "quantile"
        Binning strategy:
        - "quantile": Equal-frequency binning
        - "uniform": Equal-width binning
        - "cart": Decision tree based optimal binning
    features : list of str, optional
        Specific features to bin. If None, bins all numeric features.
    missing_values : list, optional
        Values to treat as missing (separate bin). Default: [None, np.nan]
    special_values : list, optional
        Special values to give separate bins (e.g., [-999, -1]).
    min_samples_bin : int, default 5
        Minimum samples required per bin.
    monotonic : bool, default False
        Whether to enforce monotonic WOE trend (for CART method).

    Attributes
    ----------
    bin_cuts_ : dict
        Fitted bin edges for each feature. Format: {col: [-inf, cut1, ..., inf]}
    bin_mappings_ : dict
        Bin index to label mapping. Format: {col: {0: "00_[-inf,1.5)", -1: "Missing"}}
    bin_woes_ : dict
        WOE values per bin. Format: {col: {0: 0.5, 1: -0.3, ...}}
    bin_ivs_ : dict
        Information Value per bin.
    total_iv_ : dict
        Total IV per feature.

    Example
    -------
    >>> binner = WoeBinner(n_bins=10, method="quantile")
    >>> binner.fit(X_train, y_train)
    >>> X_binned = binner.transform(X_train, return_type="woe")
    """

    # Index protocol for special values
    IDX_MISSING = -1
    IDX_OTHER = -2
    IDX_SPECIAL_START = -3

    # Numeric types supported
    NUMERIC_DTYPES = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    }

    def __init__(
        self,
        n_bins: int = 5,
        method: Literal["quantile", "uniform", "cart"] = "quantile",
        features: Optional[list[str]] = None,
        missing_values: Optional[list[Any]] = None,
        special_values: Optional[list[Any]] = None,
        min_samples_bin: int = 5,
        monotonic: bool = False,
        smoothing: float = 0.5,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.method = method
        self.features = features
        self.missing_values = missing_values or []
        self.special_values = special_values or []
        self.min_samples_bin = min_samples_bin
        self.monotonic = monotonic
        self.smoothing = float(smoothing)
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")

        # Fitted attributes
        self.bin_cuts_: dict[str, list[float]] = {}
        self.bin_mappings_: dict[str, dict[int, str]] = {}
        self.bin_woes_: dict[str, dict[int, float]] = {}
        self.bin_ivs_: dict[str, dict[int, float]] = {}
        self.total_iv_: dict[str, float] = {}
        self.fit_failures_: dict[str, str] = {}
        self.category_mappings_: dict[str, dict[Any, int]] = {}
        self.fitted_features_: list[str] = []

        # Cache for WOE computation
        self._cache_X: Optional[pl.DataFrame] = None
        self._cache_y: Optional[pl.Series] = None
        self._cache_sample_weight: Optional[np.ndarray] = None

    @time_it
    def _fit_impl(self, X: pl.DataFrame, y: Optional[pl.Series] = None, **kwargs) -> None:
        """Fit numeric and categorical WOE mappings on the supplied sample."""
        if y is None:
            raise ValidationError("Target variable 'y' is required for WOE binning")
        if len(X) != len(y):
            raise ValidationError("X and y must contain the same number of rows")

        sample_weight = self._validate_sample_weight(
            kwargs.get("sample_weight"),
            len(X),
        )
        self._cache_X = X
        self._cache_y = y
        self._cache_sample_weight = sample_weight
        target_cols = self.features if self.features else X.columns
        numeric_cols = [
            column for column in target_cols
            if column in X.columns and X[column].dtype in self.NUMERIC_DTYPES
        ]
        categorical_cols = [
            column for column in target_cols
            if column in X.columns
            and X[column].dtype not in self.NUMERIC_DTYPES
            and X[column].dtype != pl.Null
        ]

        valid_numeric: list[str] = []
        for column in numeric_cols:
            series = X[column]
            if series.dtype in [pl.Float32, pl.Float64]:
                series = series.fill_nan(None)
            min_value, max_value = series.min(), series.max()
            if min_value is None and max_value is None:
                self.fit_failures_[column] = "All-null column"
                continue
            if min_value == max_value:
                self.fit_failures_[column] = "Constant column"
                continue
            valid_numeric.append(column)

        if valid_numeric:
            if self.method == "quantile":
                self._fit_quantile(X, valid_numeric)
            elif self.method == "uniform":
                self._fit_uniform(X, valid_numeric)
            elif self.method == "cart":
                self._fit_cart(X, y, valid_numeric)
            else:
                raise ValueError(f"Unknown binning method: {self.method}")

        if categorical_cols:
            self._fit_categorical(X, categorical_cols)

        self.fitted_features_ = valid_numeric + categorical_cols
        if not self.fitted_features_:
            logger.warning("No usable numeric or categorical columns found for binning")
            return

        self._generate_mappings()
        self._calculate_woe(X, y, sample_weight)
        if self.monotonic:
            self._enforce_monotonic_woe(X, y, sample_weight)

        logger.info(
            f"✅ Binning complete. Numeric={len(valid_numeric)}, "
            f"Categorical={len(categorical_cols)}, weighted={sample_weight is not None}"
        )

    def _fit_categorical(self, X: pl.DataFrame, cols: list[str]) -> None:
        """Map frequent categories to bins and unseen/rare values to OTHER."""
        for column in cols:
            counts = (
                X.select(pl.col(column))
                .drop_nulls()
                .to_series()
                .value_counts()
                .sort("count", descending=True)
            )
            mapping: dict[Any, int] = {}
            next_bin = 0
            for row in counts.to_dicts():
                value = row[column]
                count = int(row["count"])
                if count < self.min_samples_bin:
                    mapping[value] = self.IDX_OTHER
                else:
                    mapping[value] = next_bin
                    next_bin += 1
            self.category_mappings_[column] = mapping

    def _fit_quantile(self, X: pl.DataFrame, cols: list[str]) -> None:
        """
        Ultra-fast quantile binning using Polars expressions.

        Optimization: Build a single massive expression list for all columns
        and execute in one pass through the Rust engine.
        """
        # Build quantile points
        if self.n_bins <= 1:
            quantiles = [0.5]
        else:
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1].tolist()

        raw_exclude = self.special_values + self.missing_values

        # Build flattened expression list for all columns
        q_exprs = []
        col_names = []

        for c in cols:
            # Get safe exclude values based on column type
            safe_exclude = self._get_safe_values(X.schema[c], raw_exclude)

            for i, q in enumerate(quantiles):
                # Filter expression: exclude null, nan, and special values
                filter_expr = pl.col(c).is_not_null()
                if X[c].dtype in [pl.Float32, pl.Float64]:
                    filter_expr = filter_expr & (~pl.col(c).is_nan())
                if safe_exclude:
                    filter_expr = filter_expr & (~pl.col(c).is_in(safe_exclude))

                q_expr = (
                    pl.col(c)
                    .filter(filter_expr)
                    .quantile(q)
                    .alias(f"{c}__q{i}")
                )
                q_exprs.append(q_expr)
                col_names.append((c, i))

        # Execute single aggregation for all quantiles
        if q_exprs:
            result = X.select(q_exprs).row(0)

            # Parse results back to cuts
            current_col = None
            cuts = []

            for (col, _), val in zip(col_names, result):
                if col != current_col:
                    if current_col is not None and cuts:
                        # Save previous column's cuts
                        unique_cuts = sorted(set(c for c in cuts if c is not None))
                        self.bin_cuts_[current_col] = [float('-inf')] + unique_cuts + [float('inf')]
                    current_col = col
                    cuts = []

                if val is not None:
                    cuts.append(float(val))

            # Don't forget the last column
            if current_col is not None and cuts:
                unique_cuts = sorted(set(cuts))
                self.bin_cuts_[current_col] = [float('-inf')] + unique_cuts + [float('inf')]

    def _fit_uniform(self, X: pl.DataFrame, cols: list[str]) -> None:
        """
        Equal-width binning using vectorized min/max computation.
        """
        raw_exclude = self.special_values + self.missing_values

        # Build expressions for min/max
        stats_exprs = []
        for c in cols:
            safe_exclude = self._get_safe_values(X.schema[c], raw_exclude)

            filter_expr = pl.col(c).is_not_null()
            if X[c].dtype in [pl.Float32, pl.Float64]:
                filter_expr = filter_expr & (~pl.col(c).is_nan())
            if safe_exclude:
                filter_expr = filter_expr & (~pl.col(c).is_in(safe_exclude))

            stats_exprs.append(pl.col(c).filter(filter_expr).min().alias(f"{c}_min"))
            stats_exprs.append(pl.col(c).filter(filter_expr).max().alias(f"{c}_max"))

        # Single scan for all stats
        stats_row = X.select(stats_exprs).row(0)

        for i, c in enumerate(cols):
            min_val = stats_row[i * 2]
            max_val = stats_row[i * 2 + 1]

            if min_val is None or max_val is None or min_val == max_val:
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue

            # Generate equal-width cuts
            step = (max_val - min_val) / self.n_bins
            cuts = [min_val + step * j for j in range(1, self.n_bins)]
            self.bin_cuts_[c] = [float('-inf')] + cuts + [float('inf')]

    def _fit_cart(self, X: pl.DataFrame, y: pl.Series, cols: list[str]) -> None:
        """
        Decision tree based optimal binning with parallel execution.
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            logger.warning("joblib not installed, falling back to sequential CART")
            self._fit_cart_sequential(X, y, cols)
            return

        y_np = y.to_numpy()
        raw_exclude = self.special_values + self.missing_values

        def cart_worker(col: str) -> tuple[str, list[float]]:
            """Worker function for parallel CART fitting."""
            try:
                safe_exclude = self._get_safe_values(X.schema[col], raw_exclude)

                # Filter and extract data using Polars
                filter_expr = pl.col(col).is_not_null()
                if X[col].dtype in [pl.Float32, pl.Float64]:
                    filter_expr = filter_expr & (~pl.col(col).is_nan())
                if safe_exclude:
                    filter_expr = filter_expr & (~pl.col(col).is_in(safe_exclude))

                valid_mask = X.select(filter_expr.alias("mask")).get_column("mask").to_numpy()

                if valid_mask.sum() < self.min_samples_bin * 2:
                    return col, [float('-inf'), float('inf')]

                x_clean = X[col].filter(pl.Series(valid_mask)).to_numpy().reshape(-1, 1)
                y_clean = y_np[valid_mask]

                # Fit decision tree
                tree = DecisionTreeClassifier(
                    max_leaf_nodes=self.n_bins,
                    min_samples_leaf=self.min_samples_bin,
                    random_state=42
                )
                tree.fit(x_clean, y_clean)

                # Extract thresholds
                thresholds = tree.tree_.threshold
                valid_thresholds = sorted(set(
                    t for t in thresholds
                    if t != -2.0  # -2 is leaf node marker
                ))

                if not valid_thresholds:
                    return col, [float('-inf'), float('inf')]

                return col, [float('-inf')] + valid_thresholds + [float('inf')]

            except Exception as e:
                logger.warning(f"CART failed for {col}: {e}")
                return col, [float('-inf'), float('inf')]

        # Parallel execution
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(cart_worker)(col) for col in cols
        )

        for col, cuts in results:
            self.bin_cuts_[col] = cuts

    def _fit_cart_sequential(self, X: pl.DataFrame, y: pl.Series, cols: list[str]) -> None:
        """Sequential fallback for CART binning."""
        y_np = y.to_numpy()
        raw_exclude = self.special_values + self.missing_values

        for col in cols:
            try:
                safe_exclude = self._get_safe_values(X.schema[col], raw_exclude)

                filter_expr = pl.col(col).is_not_null()
                if X[col].dtype in [pl.Float32, pl.Float64]:
                    filter_expr = filter_expr & (~pl.col(col).is_nan())
                if safe_exclude:
                    filter_expr = filter_expr & (~pl.col(col).is_in(safe_exclude))

                valid_mask = X.select(filter_expr.alias("mask")).get_column("mask").to_numpy()

                if valid_mask.sum() < self.min_samples_bin * 2:
                    self.bin_cuts_[col] = [float('-inf'), float('inf')]
                    continue

                x_clean = X[col].filter(pl.Series(valid_mask)).to_numpy().reshape(-1, 1)
                y_clean = y_np[valid_mask]

                tree = DecisionTreeClassifier(
                    max_leaf_nodes=self.n_bins,
                    min_samples_leaf=self.min_samples_bin,
                    random_state=42
                )
                tree.fit(x_clean, y_clean)

                thresholds = tree.tree_.threshold
                valid_thresholds = sorted(set(
                    t for t in thresholds if t != -2.0
                ))

                self.bin_cuts_[col] = [float('-inf')] + valid_thresholds + [float('inf')]

            except Exception as e:
                logger.warning(f"CART failed for {col}: {e}")
                self.bin_cuts_[col] = [float('-inf'), float('inf')]

    def _generate_mappings(self) -> None:
        """Generate labels for numeric and categorical bins."""
        for col, cuts in self.bin_cuts_.items():
            if not cuts:
                continue
            mapping = {self.IDX_MISSING: "Missing"}
            for i in range(len(cuts) - 1):
                lower = cuts[i]
                upper = cuts[i + 1]
                if lower == float("-inf"):
                    label = f"{i:02d}_(-inf, {upper:.4g})"
                elif upper == float("inf"):
                    label = f"{i:02d}_[{lower:.4g}, inf)"
                else:
                    label = f"{i:02d}_[{lower:.4g}, {upper:.4g})"
                mapping[i] = label
            for j, val in enumerate(self.special_values):
                mapping[self.IDX_SPECIAL_START - j] = f"Special_{val}"
            self.bin_mappings_[col] = mapping

        for col, value_mapping in self.category_mappings_.items():
            mapping = {
                self.IDX_MISSING: "Missing",
                self.IDX_OTHER: "Other",
            }
            for value, bin_idx in value_mapping.items():
                if bin_idx >= 0:
                    mapping.setdefault(bin_idx, f"{bin_idx:02d}_{value}")
            self.bin_mappings_[col] = mapping

    def _calculate_woe(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Calculate weighted WOE/IV values for every fitted feature."""
        X_binned = self._transform_impl(X, return_type="index")
        y_np = np.asarray(y.to_numpy(), dtype=float)
        weights = self._validate_sample_weight(sample_weight, len(y_np))
        total_bad = float(np.sum(weights * y_np))
        total_good = float(np.sum(weights * (1.0 - y_np)))

        if total_bad <= 0 or total_good <= 0:
            logger.warning("Target has only one class, WOE cannot be calculated")
            return

        for col in self.fitted_features_:
            bin_col = f"{col}_bin"
            if bin_col not in X_binned.columns:
                continue

            stats = (
                pl.DataFrame({
                    "bin": X_binned[bin_col],
                    "target": y,
                    "weight": weights,
                })
                .with_columns([
                    (pl.col("target") * pl.col("weight")).alias("bad_weight"),
                    ((1.0 - pl.col("target")) * pl.col("weight")).alias("good_weight"),
                ])
                .group_by("bin")
                .agg([
                    pl.col("weight").sum().alias("count"),
                    pl.col("bad_weight").sum().alias("bad"),
                    pl.col("good_weight").sum().alias("good"),
                ])
            )

            woe_dict: dict[int, float] = {}
            iv_dict: dict[int, float] = {}
            total_iv = 0.0

            for row in stats.iter_rows(named=True):
                bin_idx = row["bin"]
                bad = float(row["bad"])
                good = float(row["good"])
                dist_bad = (bad + self.smoothing) / (
                    total_bad + self.smoothing * 2
                )
                dist_good = (good + self.smoothing) / (
                    total_good + self.smoothing * 2
                )
                woe = float(np.log(dist_bad / dist_good))
                iv = float((dist_bad - dist_good) * woe)
                woe_dict[bin_idx] = woe
                iv_dict[bin_idx] = iv
                total_iv += iv

            self.bin_woes_[col] = woe_dict
            self.bin_ivs_[col] = iv_dict
            self.total_iv_[col] = total_iv

    def _enforce_monotonic_woe(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Apply an isotonic/PAVA-style monotonic calibration to numeric bins."""
        from sklearn.isotonic import IsotonicRegression

        indexed = self._transform_impl(X, return_type="index")
        y_np = np.asarray(y.to_numpy(), dtype=float)
        weights = self._validate_sample_weight(sample_weight, len(y_np))
        total_bad = float(np.sum(weights * y_np))
        total_good = float(np.sum(weights * (1.0 - y_np)))
        if total_bad <= 0 or total_good <= 0:
            return

        for column in self.bin_cuts_:
            bin_col = f"{column}_bin"
            if bin_col not in indexed.columns:
                continue
            stats = (
                pl.DataFrame({
                    "bin": indexed[bin_col],
                    "target": y,
                    "weight": weights,
                })
                .filter(pl.col("bin") >= 0)
                .with_columns(
                    (pl.col("target") * pl.col("weight")).alias("bad_weight")
                )
                .group_by("bin")
                .agg([
                    pl.col("weight").sum().alias("count"),
                    pl.col("bad_weight").sum().alias("bad"),
                ])
                .with_columns(
                    (pl.col("bad") / pl.col("count")).alias("bad_rate")
                )
                .sort("bin")
            )
            if len(stats) < 2:
                continue

            x_values = stats["bin"].to_numpy().astype(float)
            rates = stats["bad_rate"].to_numpy().astype(float)
            increasing = IsotonicRegression(
                increasing=True, out_of_bounds="clip"
            ).fit_transform(x_values, rates)
            decreasing = IsotonicRegression(
                increasing=False, out_of_bounds="clip"
            ).fit_transform(x_values, rates)
            fitted = (
                increasing
                if np.sum((increasing - rates) ** 2)
                <= np.sum((decreasing - rates) ** 2)
                else decreasing
            )
            counts = stats["count"].to_numpy().astype(float)

            updated = dict(self.bin_woes_.get(column, {}))
            iv_total = 0.0
            for bin_idx, count, bad_rate in zip(
                x_values.astype(int), counts, fitted
            ):
                clipped_rate = float(np.clip(bad_rate, 1e-7, 1 - 1e-7))
                bad = clipped_rate * count
                good = (1.0 - clipped_rate) * count
                dist_bad = (bad + self.smoothing) / (
                    total_bad + self.smoothing * 2
                )
                dist_good = (good + self.smoothing) / (
                    total_good + self.smoothing * 2
                )
                woe = float(np.log(dist_bad / dist_good))
                iv = float((dist_bad - dist_good) * woe)
                updated[bin_idx] = woe
                iv_total += iv
            self.bin_woes_[column] = updated
            self.total_iv_[column] = iv_total

    @staticmethod
    def _validate_sample_weight(
        sample_weight: Optional[Any],
        n_rows: int,
    ) -> np.ndarray:
        if sample_weight is None:
            return np.ones(n_rows, dtype=float)
        if isinstance(sample_weight, pl.Series):
            values = sample_weight.to_numpy()
        else:
            values = np.asarray(sample_weight)
        if len(values) != n_rows:
            raise ValidationError("sample_weight must have the same length as X")
        values = values.astype(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValidationError("sample_weight must be finite and strictly positive")
        return values

    def _get_safe_values(self, dtype: pl.DataType, values: list[Any]) -> list[Any]:
        """Filter values compatible with column dtype to avoid type errors."""
        if not values:
            return []

        if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            return [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        elif dtype in [pl.Float32, pl.Float64]:
            return [v for v in values if isinstance(v, (int, float))]
        elif dtype in [pl.Utf8, pl.String]:
            return [v for v in values if isinstance(v, str)]
        else:
            return values

    @time_it
    def _transform_impl(
        self,
        X: pl.DataFrame,
        return_type: Literal["index", "label", "woe"] = "index",
        **kwargs
    ) -> pl.DataFrame:
        """
        Apply binning transformation using Polars expressions.

        Optimization: Builds a flat expression list and executes in single pass.
        """
        raw_exclude = self.special_values + self.missing_values
        exprs = []

        for col, cuts in self.bin_cuts_.items():
            if col not in X.columns:
                continue

            if not cuts or len(cuts) < 2:
                # No valid cuts, assign all to bin 0
                exprs.append(pl.lit(0).cast(pl.Int16).alias(f"{col}_bin"))
                continue

            safe_exclude = self._get_safe_values(X.schema[col], raw_exclude)

            # Start with missing handling
            result_expr = (
                pl.when(pl.col(col).is_null())
                .then(pl.lit(self.IDX_MISSING))
            )

            # Handle NaN for float columns
            if X[col].dtype in [pl.Float32, pl.Float64]:
                result_expr = result_expr.when(pl.col(col).is_nan()).then(pl.lit(self.IDX_MISSING))

            # Handle special values
            for j, val in enumerate(self.special_values):
                if val in safe_exclude or (isinstance(val, (int, float)) and X[col].dtype in self.NUMERIC_DTYPES):
                    result_expr = result_expr.when(pl.col(col) == val).then(pl.lit(self.IDX_SPECIAL_START - j))

            # Normal binning using cut points
            inner_cuts = cuts[1:-1]  # Exclude -inf and inf
            if inner_cuts:
                # Use search_sorted logic via when-then chain
                for i, cut in enumerate(inner_cuts):
                    result_expr = result_expr.when(pl.col(col) < cut).then(pl.lit(i))

                # Last bin
                result_expr = result_expr.otherwise(pl.lit(len(inner_cuts)))
            else:
                # Only one bin
                result_expr = result_expr.otherwise(pl.lit(0))

            bin_col_name = f"{col}_bin"

            if return_type == "index":
                exprs.append(result_expr.cast(pl.Int16).alias(bin_col_name))
            elif return_type == "label":
                # Map indices to labels
                mapping = self.bin_mappings_.get(col, {})
                str_mapping = {str(k): v for k, v in mapping.items()}
                exprs.append(
                    result_expr.cast(pl.Utf8)
                    .replace(str_mapping)
                    .alias(bin_col_name)
                )
            elif return_type == "woe":
                # Map indices to WOE values
                woe_mapping = self.bin_woes_.get(col, {})
                if woe_mapping:
                    # Build WOE lookup expression
                    woe_expr = pl.lit(0.0)
                    for idx, woe_val in woe_mapping.items():
                        woe_expr = (
                            pl.when(result_expr == idx)
                            .then(pl.lit(woe_val))
                            .otherwise(woe_expr)
                        )
                    exprs.append(woe_expr.alias(bin_col_name))
                else:
                    exprs.append(pl.lit(0.0).alias(bin_col_name))

        for col, value_mapping in self.category_mappings_.items():
            if col not in X.columns:
                continue

            result_expr = (
                pl.when(pl.col(col).is_null())
                .then(pl.lit(self.IDX_MISSING))
            )
            for value, bin_idx in value_mapping.items():
                result_expr = result_expr.when(pl.col(col) == value).then(pl.lit(bin_idx))
            result_expr = result_expr.otherwise(pl.lit(self.IDX_OTHER))
            bin_col_name = f"{col}_bin"

            if return_type == "index":
                exprs.append(result_expr.cast(pl.Int16).alias(bin_col_name))
            elif return_type == "label":
                label_mapping = self.bin_mappings_.get(col, {})
                label_expr = pl.lit("Other")
                for bin_idx, label in label_mapping.items():
                    label_expr = (
                        pl.when(result_expr == bin_idx)
                        .then(pl.lit(label))
                        .otherwise(label_expr)
                    )
                exprs.append(label_expr.alias(bin_col_name))
            elif return_type == "woe":
                woe_mapping = self.bin_woes_.get(col, {})
                woe_expr = pl.lit(0.0)
                for bin_idx, woe_value in woe_mapping.items():
                    woe_expr = (
                        pl.when(result_expr == bin_idx)
                        .then(pl.lit(woe_value))
                        .otherwise(woe_expr)
                    )
                exprs.append(woe_expr.alias(bin_col_name))

        if exprs:
            return X.with_columns(exprs)
        return X

    def get_woe_dict(self) -> dict[str, dict[int, float]]:
        """Return WOE dictionary for all features."""
        return self.bin_woes_

    def get_iv_report(self) -> pl.DataFrame:
        """
        Generate IV (Information Value) report for all features.

        Returns
        -------
        pl.DataFrame
            Report with columns: feature, total_iv, interpretation
        """
        if not self.total_iv_:
            raise NotFittedError("Binner not fitted. Call fit() first.")

        def interpret_iv(iv: float) -> str:
            if iv < 0.02:
                return "Not Useful"
            elif iv < 0.1:
                return "Weak"
            elif iv < 0.3:
                return "Medium"
            elif iv < 0.5:
                return "Strong"
            else:
                return "Suspicious"

        data = [
            {
                "feature": col,
                "total_iv": iv,
                "interpretation": interpret_iv(iv)
            }
            for col, iv in self.total_iv_.items()
        ]

        return pl.DataFrame(data).sort("total_iv", descending=True)

    @time_it
    def compute_bin_stats(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        sample_weight: Optional[Any] = None,
    ) -> pl.DataFrame:
        """Compute weighted bin-level diagnostics for fitted features."""
        X_binned = self.transform(X, return_type="index")
        y_np = np.asarray(y.to_numpy(), dtype=float)
        weights = self._validate_sample_weight(sample_weight, len(y_np))
        total_bad = float(np.sum(weights * y_np))
        total_good = float(np.sum(weights * (1.0 - y_np)))
        if total_bad <= 0 or total_good <= 0:
            return pl.DataFrame()

        all_stats = []
        for col in self.fitted_features_:
            bin_col = f"{col}_bin"
            if bin_col not in X_binned.columns:
                continue
            stats = (
                pl.DataFrame({
                    "bin_idx": X_binned[bin_col],
                    "target": y,
                    "weight": weights,
                })
                .with_columns([
                    (pl.col("target") * pl.col("weight")).alias("bad_weight"),
                    ((1.0 - pl.col("target")) * pl.col("weight")).alias(
                        "good_weight"
                    ),
                ])
                .group_by("bin_idx")
                .agg([
                    pl.col("weight").sum().alias("count"),
                    pl.col("bad_weight").sum().alias("bad"),
                    pl.col("good_weight").sum().alias("good"),
                ])
                .with_columns([
                    pl.lit(col).alias("feature"),
                    (pl.col("bad") / pl.col("count")).alias("bad_rate"),
                    (pl.col("bad") / total_bad).alias("dist_bad"),
                    (pl.col("good") / total_good).alias("dist_good"),
                ])
                .with_columns([
                    (
                        (
                            (pl.col("bad") + self.smoothing)
                            / (total_bad + self.smoothing * 2)
                        )
                        / (
                            (pl.col("good") + self.smoothing)
                            / (total_good + self.smoothing * 2)
                        )
                    ).log().alias("woe")
                ])
                .with_columns([
                    (
                        (pl.col("dist_bad") - pl.col("dist_good"))
                        * pl.col("woe")
                    ).alias("iv")
                ])
                .sort("bin_idx")
            )
            all_stats.append(stats)

        if all_stats:
            return pl.concat(all_stats)
        return pl.DataFrame()


# Backward compatibility alias
WoeBinning = WoeBinner
