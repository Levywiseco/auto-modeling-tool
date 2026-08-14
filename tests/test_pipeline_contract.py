# -*- coding: utf-8 -*-
"""Regression tests for the Dev/OOT and scoring contracts."""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from src.binning.woe_binning import WoeBinner
from src.data.preprocess import DataPreprocessor
from src.data.split import split_dev_oot
from src.modeling.scorecard import ScorecardBuilder


def test_explicit_sample_dev_oot_split():
    data = pl.DataFrame(
        {
            "feature": list(range(8)),
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "sample": ["dev", "dev", "dev", "dev", "oot", "oot", "oot", "oot"],
        }
    )
    result = split_dev_oot(data, "target", sample_column="sample")
    assert result.strategy == "sample_column"
    assert len(result.dev) == 4
    assert len(result.oot) == 4


def test_date_dev_oot_split():
    data = pl.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            "month": ["2024-01", "2024-02", "2024-03", "2024-04"],
        }
    )
    result = split_dev_oot(
        data,
        "target",
        date_column="month",
        oot_start="2024-03",
    )
    assert result.strategy == "date"
    assert len(result.dev) == 2
    assert len(result.oot) == 2


def test_preprocessor_uses_dev_statistics_only():
    dev = pl.DataFrame({"feature": [0.0, 1.0]})
    oot = pl.DataFrame({"feature": [100.0, 101.0]})
    preprocessor = DataPreprocessor(
        clean_strategy="median",
        normalize_method="zscore",
    )
    preprocessor.fit(dev)
    transformed = preprocessor.transform(oot)
    assert transformed["feature"].mean() > 100


def test_scorecard_uses_bin_index_for_points():
    X = pl.DataFrame({"feature": np.linspace(0, 1, 20)})
    y = pl.Series("target", [0, 1] * 10)
    binner = WoeBinner(n_bins=2, min_samples_bin=1)
    woe = binner.fit_transform(X, y, return_type="woe").select(["feature_bin"])
    model = LogisticRegression(max_iter=1000).fit(woe.to_numpy(), y.to_numpy())
    scorecard = ScorecardBuilder(
        base_score=600,
        PDO=20,
        target_odds=20,
    ).fit(model, binner, feature_names=["feature_bin"])
    table = scorecard.get_scorecard_table()
    expected = sorted(binner.bin_woes_["feature"].values())
    actual = sorted(table["WOE"].to_list())
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    scores = scorecard.score(X)
    assert len(scores) == len(X)
