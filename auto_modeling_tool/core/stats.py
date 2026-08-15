"""Weight-aware statistics shared by binning, PSI and reporting.

Credit modelling routinely trains on a sample that is not the population:
goods are undersampled, rejects are inferred back in with weights. Any cut
point, quantile or smoothing constant derived from raw row counts then
describes the sample rather than the population the model is meant for.
"""

from typing import Optional, Union

import numpy as np


def normalize_weights(
    weights: Optional[Union[np.ndarray, list]],
    size: int,
) -> np.ndarray:
    """Return a finite, strictly positive weight vector of ``size``."""
    if weights is None:
        return np.ones(size, dtype=float)
    array = np.asarray(weights, dtype=float).ravel()
    if array.size != size:
        raise ValueError(
            f"Sample weights must have length {size}, got {array.size}"
        )
    if not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError("Sample weights must be finite and strictly positive")
    return array


def weighted_quantile(
    values: Union[np.ndarray, list],
    quantiles: Union[np.ndarray, list, float],
    *,
    weights: Optional[Union[np.ndarray, list]] = None,
) -> np.ndarray:
    """Quantiles of the weighted distribution, ignoring non-finite values.

    With uniform weights this agrees with ``np.quantile``'s linear
    interpolation. With weights it answers the question the unweighted version
    cannot: where does the *population* — not the sample — split.
    """
    array = np.asarray(values, dtype=float).ravel()
    q = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if np.any((q < 0) | (q > 1)):
        raise ValueError("Quantiles must lie in [0, 1]")

    w = normalize_weights(weights, array.size)
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("Cannot take quantiles of an all-missing array")
    array, w = array[finite], w[finite]

    # Uniform weights delegate to numpy so the unweighted path — by far the
    # common one — keeps its exact historical behaviour rather than shifting by
    # a plotting-position convention.
    if np.allclose(w, w[0]):
        return np.quantile(array, q)

    order = np.argsort(array, kind="mergesort")
    array, w = array[order], w[order]

    # Midpoint of each value's weight interval, rescaled to [0, 1] — the
    # weighted analogue of a plotting position.
    cumulative = np.cumsum(w)
    positions = (cumulative - 0.5 * w) / cumulative[-1]

    return np.interp(q, positions, array)


def weighted_mean(
    values: Union[np.ndarray, list],
    weights: Optional[Union[np.ndarray, list]] = None,
) -> float:
    """Mean of the weighted distribution, ignoring non-finite values."""
    array = np.asarray(values, dtype=float).ravel()
    w = normalize_weights(weights, array.size)
    finite = np.isfinite(array)
    if not finite.any():
        return float("nan")
    return float(np.average(array[finite], weights=w[finite]))
