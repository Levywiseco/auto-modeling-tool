# -*- coding: utf-8 -*-
"""
Scorecard builder module.

This module provides functionality to convert trained models (especially
logistic regression) into standard credit scorecards.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from ..core.decorators import time_it
from ..core.logger import logger


class ScorecardBuilder:
    """
    Build a credit scorecard from a trained model.

    Converts WOE-encoded logistic regression coefficients into a
    standard scorecard format.

    Parameters
    ----------
    base_score : int, default 600
        Base score (score when odds = 1:1).
    PDO : int, default 20
        Points to Double the Odds.
    target_odds : int, default 20
        Target odds (e.g., 20:1 means 20 good per 1 bad).
    round_scores : bool, default True
        Whether to round scores to integers.
    """

    def __init__(
        self,
        base_score: int = 600,
        PDO: int = 20,
        target_odds: int = 20,
        round_scores: bool = True,
    ):
        self.base_score = base_score
        self.PDO = PDO
        self.target_odds = target_odds
        self.round_scores = round_scores

        self.model_: Any = None
        self.binner_: Any = None
        self.feature_names_: List[str] = []
        self.intercept_: float = 0.0
        self.coefficients_: np.ndarray = None
        self.scorecard_: pl.DataFrame = None

    @time_it
    def fit(
        self,
        model: Any,
        binner: Any,
        feature_names: Optional[List[str]] = None,
    ) -> "ScorecardBuilder":
        """
        Fit the scorecard builder.

        Parameters
        ----------
        model : Any
            Trained model (must have coef_ and intercept_ attributes).
        binner : Any
            Fitted WoeBinner object.
        feature_names : list of str, optional
            Feature names corresponding to coefficients.

        Returns
        -------
        self
            Fitted scorecard builder.
        """
        self.model_ = model
        self.binner_ = binner

        # Extract coefficients
        if hasattr(model, "coef_"):
            self.coefficients_ = model.coef_.flatten()
        else:
            raise ValueError("Model must have coef_ attribute")

        if hasattr(model, "intercept_"):
            self.intercept_ = model.intercept_[0] if model.intercept_.ndim > 0 else model.intercept_
        else:
            self.intercept_ = 0.0

        # Get feature names
        if feature_names is not None:
            self.feature_names_ = feature_names
        elif hasattr(model, "feature_names_in_"):
            self.feature_names_ = list(model.feature_names_in_)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(len(self.coefficients_))]

        if len(self.feature_names_) != len(self.coefficients_):
            raise ValueError(
                f"Feature names ({len(self.feature_names_)}) must match "
                f"coefficients ({len(self.coefficients_)})"
            )

        # Calculate scaling factors
        self.factor_ = self.PDO / np.log(2)
        self.offset_ = self.base_score - self.factor_ * np.log(self.target_odds)

        # Build scorecard
        self._build_scorecard()

        logger.info(f"✅ Scorecard built with {len(self.scorecard_)} variables")

        return self

    def _build_scorecard(self) -> None:
        """Build the scorecard table."""
        rows = []

        # Get bin info from binner
        bin_cuts = getattr(self.binner_, "bin_cuts_", {})
        bin_woes = getattr(self.binner_, "bin_woes_", {})

        for idx, feature in enumerate(self.feature_names_):
            coef = self.coefficients_[idx]

            # Get WOE values for this feature
            woe_values = bin_woes.get(feature, [])
            bin_edges = bin_cuts.get(feature, [])

            # Calculate points for each bin
            for bin_idx, woe in enumerate(woe_values):
                # Points = (coefficient * WOE) * factor
                points = round(coef * woe * self.factor_, 0)

                # Create bin label
                if bin_edges and bin_idx < len(bin_edges) - 1:
                    bin_label = f"[{bin_edges[bin_idx]:.2f}, {bin_edges[bin_idx + 1]:.2f})"
                else:
                    bin_label = f"bin_{bin_idx}"

                rows.append({
                    "Variable": feature,
                    "Bin": bin_label,
                    "WOE": round(woe, 4),
                    "Coefficient": round(coef, 4),
                    "Points": int(points),
                })

        self.scorecard_ = pl.DataFrame(rows)

    def _get_variable_points(self, variable: str, bin_idx: int) -> int:
        """Get points for a specific variable and bin."""
        if self.scorecard_ is None:
            raise RuntimeError("Scorecard not built. Call fit() first.")

        row = self.scorecard_.filter(
            (pl.col("Variable") == variable) & (pl.col("Bin") == f"bin_{bin_idx}")
        )

        if len(row) == 0:
            return 0

        return row["Points"][0]

    def score(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate scorecard scores.

        Parameters
        ----------
        X : pl.DataFrame or np.ndarray
            Input features (original values, not WOE-encoded).

        Returns
        -------
        np.ndarray
            Calculated scores.
        """
        if self.scorecard_ is None:
            raise RuntimeError("Scorecard not built. Call fit() first.")

        # Transform to WOE first
        if isinstance(X, np.ndarray):
            X = pl.DataFrame(X, schema=self.feature_names_)

        X_woe = self.binner_.transform(X, return_type="woe")

        # Select only the features used in scorecard
        woe_cols = [c for c in X_woe.columns if c.endswith("_bin")]
        X_woe_selected = X_woe.select(woe_cols)

        # Map WOE columns to feature names
        woe_feature_map = {}
        for col in X_woe_selected.columns:
            # Extract feature name from bin column (remove _bin suffix)
            feature = col.replace("_bin", "")
            woe_feature_map[col] = feature

        # Calculate base score
        scores = np.full(len(X), self.offset_)

        # Add points from each feature
        for col in X_woe_selected.columns:
            feature = woe_feature_map.get(col)
            if feature not in self.feature_names_:
                continue

            coef_idx = self.feature_names_.index(feature)
            coef = self.coefficients_[coef_idx]

            woe_values = X_woe_selected[col].to_numpy()

            # Calculate points for each bin
            bin_woes = self.binner_.bin_woes_.get(feature, [])

            feature_points = np.zeros(len(X))
            for bin_idx, woe in enumerate(bin_woes):
                mask = (woe_values == woe)
                points = coef * woe * self.factor_
                feature_points[mask] = points

            scores += feature_points

        if self.round_scores:
            scores = np.round(scores).astype(int)

        return scores

    def predict(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        threshold: Optional[int] = None,
    ) -> np.ndarray:
        """
        Make predictions using scorecard scores.

        Parameters
        ----------
        X : pl.DataFrame or np.ndarray
            Input features.
        threshold : int, optional
            Score threshold for classification. If None, uses optimal threshold.

        Returns
        -------
        np.ndarray
            Predictions (0 or 1).
        """
        scores = self.score(X)

        if threshold is None:
            # Default threshold: base_score
            threshold = self.base_score

        return (scores >= threshold).astype(int)

    def predict_proba(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict probabilities from scorecard scores.

        Parameters
        ----------
        X : pl.DataFrame or np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        scores = self.score(X)

        # Convert score to probability
        # odds = score - offset / factor
        # prob = odds / (odds + 1)
        odds = np.exp((scores - self.offset_) / self.factor_)
        prob = odds / (odds + 1)

        return np.column_stack([1 - prob, prob])

    def get_scorecard_table(self) -> pl.DataFrame:
        """
        Get the scorecard as a DataFrame.

        Returns
        -------
        pl.DataFrame
            Scorecard table.
        """
        if self.scorecard_ is None:
            raise RuntimeError("Scorecard not built. Call fit() first.")

        return self.scorecard_

    def save_scorecard(self, path: str) -> None:
        """
        Save scorecard to CSV.

        Parameters
        ----------
        path : str
            Output path.
        """
        if self.scorecard_ is None:
            raise RuntimeError("Scorecard not built. Call fit() first.")

        self.scorecard_.write_csv(path)
        logger.info(f"✅ Scorecard saved to {path}")

    def summary(self) -> Dict[str, Any]:
        """
        Get scorecard summary.

        Returns
        -------
        dict
            Summary information.
        """
        if self.scorecard_ is None:
            raise RuntimeError("Scorecard not built. Call fit() first.")

        variable_stats = self.scorecard_.group_by("Variable").agg([
            pl.col("Points").min().alias("min_points"),
            pl.col("Points").max().alias("max_points"),
            (pl.col("Points").max() - pl.col("Points").min()).alias("points_range"),
        ])

        return {
            "base_score": self.base_score,
            "PDO": self.PDO,
            "target_odds": self.target_odds,
            "factor": self.factor_,
            "offset": self.offset_,
            "n_variables": len(self.feature_names_),
            "n_bins": len(self.scorecard_),
            "variable_stats": variable_stats.to_dicts(),
        }


@time_it
def build_scorecard(
    model: Any,
    binner: Any,
    X: Optional[Union[pl.DataFrame, np.ndarray]] = None,
    feature_names: Optional[List[str]] = None,
    base_score: int = 600,
    PDO: int = 20,
    target_odds: int = 20,
) -> ScorecardBuilder:
    """
    Build a credit scorecard from a trained model.

    Parameters
    ----------
    model : Any
        Trained model (must have coef_ and intercept_ attributes).
    binner : Any
        Fitted WoeBinner object.
    X : pl.DataFrame or np.ndarray, optional
        Training data (used to extract feature names if not provided).
    feature_names : list of str, optional
        Feature names.
    base_score : int, default 600
        Base score.
    PDO : int, default 20
        Points to Double the Odds.
    target_odds : int, default 20
        Target odds.

    Returns
    -------
    ScorecardBuilder
        Fitted scorecard builder.

    Example
    -------
    >>> # Build scorecard from logistic regression
    >>> scorecard = build_scorecard(
    ...     model=lr_model,
    ...     binner=woe_binner,
    ...     base_score=600,
    ...     PDO=20,
    ... )
    >>>
    >>> # Calculate scores
    >>> scores = scorecard.score(new_data)
    >>>
    >>> # Get scorecard table
    >>> table = scorecard.get_scorecard_table()
    """
    # Extract feature names from X if provided
    if feature_names is None and X is not None:
        if isinstance(X, pl.DataFrame):
            feature_names = X.columns
        elif isinstance(X, np.ndarray):
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    builder = ScorecardBuilder(
        base_score=base_score,
        PDO=PDO,
        target_odds=target_odds,
    )

    builder.fit(model, binner, feature_names)

    return builder
