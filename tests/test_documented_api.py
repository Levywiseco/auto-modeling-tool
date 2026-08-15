"""Executable checks for what the documentation promises.

Docs drifted from the code repeatedly: README and the feature-selection guide
both showed `fit_transform(df, target_col=...)`, which raises; the CLI and docs
advertised a selection method the implementation did not accept; and a function
the guide told users to import was never exported. These tests fail when a
documented promise stops being true.
"""

import numpy as np
import polars as pl
import pytest

from auto_modeling_tool.binning import WoeBinner
from auto_modeling_tool.features import FeatureSelector, remove_multicollinearity
from auto_modeling_tool.features.selection import select_features


@pytest.fixture
def frame():
    rng = np.random.default_rng(4)
    n = 400
    signal = rng.normal(size=n)
    target = (signal > 0).astype(int)
    return pl.DataFrame({
        "income": rng.normal(5000, 1200, n),
        "utilization": rng.random(n),
        "signal": signal,
        "target": target,
    })


# Every value the CLI accepts and the docs advertise must actually run.
DOCUMENTED_SELECTION_METHODS = [
    "iv", "correlation", "variance", "rfe", "recursive", "mutual_info",
]


class TestDocumentedSelectionMethods:
    @pytest.mark.parametrize("method", DOCUMENTED_SELECTION_METHODS)
    def test_method_runs(self, frame, method):
        features = frame.drop("target")
        selected = select_features(
            features, frame["target"], method=method, n_features=2
        )
        assert isinstance(selected, list)

    def test_rfe_and_recursive_are_the_same_method(self, frame):
        """The docs say `rfe`; the implementation called it `recursive`."""
        features = frame.drop("target")
        by_rfe = select_features(
            features, frame["target"], method="rfe", n_features=2
        )
        by_recursive = select_features(
            features, frame["target"], method="recursive", n_features=2
        )
        assert by_rfe == by_recursive

    def test_cli_choices_are_all_supported(self):
        """Anything --selection accepts must reach a real implementation."""
        from auto_modeling_tool.main import _parser

        action = next(
            a for a in _parser()._actions if "--selection" in (a.option_strings or [])
        )
        assert set(action.choices) <= set(DOCUMENTED_SELECTION_METHODS)


class TestDocumentedExamples:
    def test_readme_low_level_example(self, frame):
        """README's sklearn-style example, run verbatim."""
        features = frame.drop("target")
        target = frame["target"]

        binner = WoeBinner(n_bins=10, method="quantile", special_values=[-999])
        df_woe = binner.fit_transform(features, target, return_type="woe")
        assert binner.get_iv_report().height > 0

        selector = FeatureSelector(method="iv", iv_threshold=0.02)
        df_selected = selector.fit_transform(df_woe, target)
        assert df_selected.height == frame.height

    def test_remove_multicollinearity_is_importable_and_returns_a_pair(self, frame):
        """The guide imports it from the package root and unpacks two values."""
        features = frame.drop("target")
        filtered, dropped = remove_multicollinearity(features, threshold=0.8)

        assert isinstance(filtered, pl.DataFrame)
        assert isinstance(dropped, list)
        assert filtered.width + len(dropped) == features.width
