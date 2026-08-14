"""Stability metrics on binned distributions.

Shared by ``src.analysis.profile_risk`` and ``src.monitoring.Monitor``:
both bucket features with a fitted binner, then compare bin-index
distributions between a benchmark slice and later slices.

Bin index protocol (from :class:`~src.binning.WoeBinner`):
missing = -1, other = -2, special values start at -3 descending.
"""

from typing import Optional

import polars as pl

IDX_MISSING = -1
IDX_OTHER = -2


def bin_distribution(
    binned: pl.Series,
    sample_weight: Optional[pl.Series] = None,
) -> dict[int, float]:
    """Share of rows per bin index, optionally using sample weights."""
    if binned.len() == 0:
        return {}
    if sample_weight is None:
        weights = [1.0] * binned.len()
    else:
        if len(sample_weight) != len(binned):
            raise ValueError("sample_weight must have the same length as binned")
        weights = sample_weight.cast(pl.Float64).to_list()
    total = float(sum(weight for weight in weights if weight is not None))
    if total <= 0:
        return {}
    shares: dict[int, float] = {}
    for index, weight in zip(binned.to_list(), weights):
        if index is None or weight is None:
            continue
        key = int(index)
        shares[key] = shares.get(key, 0.0) + float(weight)
    return {key: value / total for key, value in shares.items()}


def psi_from_distributions(
    expected: dict[int, float],
    actual: dict[int, float],
    *,
    include_missing: bool = False,
    include_special: bool = False,
    epsilon: float = 1e-6,
) -> float:
    """PSI between two bin-share distributions.

    Parameters
    ----------
    expected, actual : dict of bin_idx -> share
        Benchmark and comparison distributions from :func:`bin_distribution`.
    include_missing : bool, default False
        Include the missing bin (idx ``-1``) in the calculation.
    include_special : bool, default False
        Include special-value bins (idx ``<= -3``) in the calculation.
    epsilon : float
        Floor applied to empty shares to avoid log(0).
    """
    import math

    keys = set(expected) | set(actual)
    if not include_missing:
        keys.discard(IDX_MISSING)
    if not include_special:
        keys = {k for k in keys if k >= IDX_OTHER}

    psi = 0.0
    for k in sorted(keys):
        e = max(expected.get(k, 0.0), epsilon)
        a = max(actual.get(k, 0.0), epsilon)
        psi += (a - e) * math.log(a / e)
    return psi


def psi_level(psi: float, warn: float = 0.1, critical: float = 0.25) -> str:
    """Standard PSI verdict labels used across reports (Chinese)."""
    if psi >= critical:
        return "严重"
    if psi >= warn:
        return "警告"
    return "正常"
