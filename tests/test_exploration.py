"""Tests for the standalone dataset exploration helpers."""

import numpy as np
import polars as pl

from scripts.explore_dataset import _weighted_mean


def test_weighted_bad_rate_helper_uses_sample_weights():
    target = pl.Series("target", [0, 0, 1])
    weight = pl.Series("weight", [1.0, 1.0, 8.0])

    assert _weighted_mean(target) == 1 / 3
    np.testing.assert_allclose(_weighted_mean(target, weight), 0.8)

