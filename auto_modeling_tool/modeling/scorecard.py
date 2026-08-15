"""
Scorecard builder module.

This module provides functionality to convert trained models (especially
logistic regression) into standard credit scorecards.
"""

from typing import Any, Optional, Union

import numpy as np
import polars as pl

from ..core.decorators import time_it
from ..core.exceptions import ValidationError
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
        self.feature_names_: list[str] = []
        self.intercept_: float = 0.0
        self.base_points_: float = 0.0
        self.coefficients_: np.ndarray = None
        self.scorecard_: pl.DataFrame = None

    @time_it
    def fit(
        self,
        model: Any,
        binner: Any,
        feature_names: Optional[list[str]] = None,
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

        # Standard scorecard scaling. WOE is ln(bad/good), so a larger
        # sum(coef * WOE) means a worse applicant; the points contribution is
        # therefore NEGATED, otherwise the scale runs backwards and a
        # "high score = low risk" cutoff approves the worst applicants.
        self.factor_ = self.PDO / np.log(2)
        self.offset_ = self.base_score - self.factor_ * np.log(self.target_odds)
        # The model intercept is a constant shift of the log-odds, so it belongs
        # in the base points. It was recorded and then never used, which made
        # predict_proba disagree with the underlying model.
        self.base_points_ = self.offset_ - self.factor_ * float(self.intercept_)

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

            model_feature = feature
            raw_feature = feature[:-4] if feature.endswith("_bin") else feature
            coef = self.coefficients_[idx]

            # WoeBinner stores {bin_index: woe}; iterating the dict itself
            # previously used bin indexes as WOE values and corrupted points.
            woe_values = bin_woes.get(raw_feature, {})
            bin_edges = bin_cuts.get(raw_feature, [])
            bin_labels = getattr(self.binner_, "bin_mappings_", {}).get(raw_feature, {})

            for bin_idx, woe in sorted(woe_values.items(), key=lambda item: item[0]):
                points = -coef * float(woe) * self.factor_
                bin_label = bin_labels.get(bin_idx)
                if bin_label is None:
                    if bin_edges and isinstance(bin_idx, int) and 0 <= bin_idx < len(bin_edges) - 1:
                        bin_label = f"[{bin_edges[bin_idx]:.2f}, {bin_edges[bin_idx + 1]:.2f})"
                    else:
                        bin_label = f"bin_{bin_idx}"

                rows.append({
                    "Variable": model_feature,
                    "RawVariable": raw_feature,
                    "BinIndex": int(bin_idx),
                    "Bin": bin_label,
                    "WOE": round(float(woe), 4),
                    "Coefficient": round(float(coef), 4),
                    "Points": int(round(points, 0)) if self.round_scores else float(points),
                })

        self.scorecard_ = pl.DataFrame(rows)

    def _get_variable_points(self, variable: str, bin_idx: int) -> int:
        """Get points for a variable/bin using the fitted bin mapping."""
        if self.scorecard_ is None:
            raise RuntimeError("Scorecard not built. Call fit() first.")

        raw_feature = variable[:-4] if variable.endswith("_bin") else variable
        label = getattr(self.binner_, "bin_mappings_", {}).get(
            raw_feature, {}
        ).get(bin_idx, f"bin_{bin_idx}")
        row = self.scorecard_.filter(
            (pl.col("Variable") == variable) & (pl.col("BinIndex") == bin_idx)
        )
        if len(row) == 0:
            row = self.scorecard_.filter(
                (pl.col("RawVariable") == raw_feature) & (pl.col("Bin") == label)
            )
        return int(row["Points"][0]) if len(row) else 0

    def score(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Calculate scores from raw driver values."""
        if self.scorecard_ is None:
            raise RuntimeError("Scorecard not built. Call fit() first.")

        if isinstance(X, np.ndarray):
            # feature_names_ are the model's WOE column names (income_bin); the
            # array holds raw drivers (income). Labelling raw columns with WOE
            # names made binner.transform match nothing, so every row collapsed
            # to offset_ — a constant score returned with no error at all.
            raw_features = list(self.binner_.fitted_features_)
            if X.ndim != 2 or X.shape[1] != len(raw_features):
                raise ValidationError(
                    f"Array must have one column per fitted driver, in fit order: "
                    f"expected {len(raw_features)} ({raw_features}), "
                    f"got shape {X.shape}. Pass a DataFrame to score by name."
                )
            X = pl.DataFrame(X, schema=raw_features)
        X_bins = self.binner_.transform(X, return_type="index")
        scores = np.full(len(X), self.base_points_, dtype=float)

        for model_feature, coef in zip(self.feature_names_, self.coefficients_):
            raw_feature = (
                model_feature[:-4]
                if model_feature.endswith("_bin")
                else model_feature
            )
            bin_col = f"{raw_feature}_bin"
            if bin_col not in X_bins.columns:
                continue
            bin_indices = X_bins[bin_col].to_numpy()
            woe_by_bin = self.binner_.bin_woes_.get(raw_feature, {})
            feature_points = np.zeros(len(X), dtype=float)
            for bin_idx, woe in woe_by_bin.items():
                feature_points[bin_indices == bin_idx] = (
                    -float(coef) * float(woe) * self.factor_
                )
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
        # score = base_points_ - factor * sum(coef * WOE), and the model's
        # log-odds of bad is intercept + sum(coef * WOE), so inverting gives:
        log_odds_bad = (self.offset_ - scores) / self.factor_
        prob = 1.0 / (1.0 + np.exp(-log_odds_bad))

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

    def summary(self) -> dict[str, Any]:
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


def probability_to_credit_score(
    probability: Union[float, np.ndarray, list[float]],
    *,
    base_score: float = 500.0,
    pdo: float = 50.0,
    min_score: float = 300.0,
    max_score: float = 900.0,
) -> np.ndarray:
    """Map default probability to a clipped credit score.

    The guide's convention is base_score at p=0.5 and pdo points
    for each doubling/halving of the good-to-bad odds.
    """
    if pdo <= 0:
        raise ValueError("pdo must be positive")
    if min_score > max_score:
        raise ValueError("min_score must not exceed max_score")
    values = np.asarray(probability, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("probability must contain only finite values")
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    scores = base_score + (pdo / np.log(2.0)) * np.log((1.0 - clipped) / clipped)
    return np.clip(scores, min_score, max_score)


@time_it
def build_scorecard(
    model: Any,
    binner: Any,
    X: Optional[Union[pl.DataFrame, np.ndarray]] = None,
    feature_names: Optional[list[str]] = None,
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
