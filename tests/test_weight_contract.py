"""Sample weights must mean the same thing at every stage.

Credit modelling trains on a sample that is deliberately not the population:
goods are undersampled, rejects are inferred back in with weights. Any stage
that quietly ignores the weights — or interprets them on the wrong scale —
fits the sample instead of the population the model is for.

Each test here corresponds to a defect found by the adversarial audit.
"""

import numpy as np
import polars as pl
import pytest
from sklearn.tree import DecisionTreeClassifier

from auto_modeling_tool.binning import WoeBinner
from auto_modeling_tool.evaluation.metrics import calculate_psi


class TestSmoothingIsScaleInvariant:
    """IV must not change when every weight is multiplied by a constant."""

    @staticmethod
    def _iv(unit):
        rng = np.random.default_rng(11)
        n = 1500
        x = rng.normal(size=n)
        target = (x + rng.normal(0, 0.5, n) > 0).astype(int)
        weights = np.where(target == 1, 1.0, 4.0) * unit

        binner = WoeBinner(n_bins=5, min_samples_bin=1)
        binner.fit(pl.DataFrame({"x": x}), pl.Series("t", target),
                   sample_weight=weights)
        return binner.total_iv_["x"]

    def test_iv_is_invariant_to_weight_scale(self):
        """A fixed smoothing constant added to weighted counts is scale-bound.

        Rescaling weights changes nothing about their relative meaning, but
        moved IV by a factor of 1.6 before the constant was scaled too.
        """
        ivs = [self._iv(unit) for unit in (0.01, 1.0, 100.0, 10_000.0)]
        assert max(ivs) - min(ivs) < 1e-9

    def test_unweighted_iv_is_unchanged(self):
        """The common path must keep its historical value."""
        rng = np.random.default_rng(11)
        n = 1500
        x = rng.normal(size=n)
        target = (x + rng.normal(0, 0.5, n) > 0).astype(int)

        binner = WoeBinner(n_bins=5, min_samples_bin=1)
        binner.fit(pl.DataFrame({"x": x}), pl.Series("t", target))
        assert binner.total_iv_["x"] > 0


class TestBinningCutsUseWeights:
    """Cut points describe the population, not the sample."""

    @staticmethod
    def _skewed():
        rng = np.random.default_rng(5)
        x = np.concatenate([rng.normal(-2, 0.5, 800), rng.normal(2, 0.5, 800)])
        target = (x > 0).astype(int)
        weights = np.where(x > 0, 1.0, 50.0)
        return x, target, weights

    def test_quantile_bins_hold_equal_weighted_share(self):
        x, target, weights = self._skewed()
        binner = WoeBinner(n_bins=5, method="quantile", min_samples_bin=1)
        frame = pl.DataFrame({"x": x})
        binner.fit(frame, pl.Series("t", target), sample_weight=weights)

        bins = binner.transform(frame, return_type="index")["x_bin"].to_numpy()
        shares = [weights[bins == i].sum() / weights.sum() for i in range(5)]
        assert all(abs(s - 0.2) < 0.02 for s in shares), shares

    def test_cart_cuts_match_sklearn_with_weights(self):
        """The tree must receive sample_weight, not just the rows."""
        rng = np.random.default_rng(21)
        n = 2500
        x = rng.uniform(0, 10, n)
        target = rng.binomial(1, np.clip(0.05 + 0.09 * x, 0, 1))
        weights = np.where(x > 6, 40.0, 1.0)

        reference = DecisionTreeClassifier(
            max_leaf_nodes=2, min_samples_leaf=5, random_state=42
        )
        reference.fit(x.reshape(-1, 1), target, sample_weight=weights)
        expected = float(
            reference.tree_.threshold[reference.tree_.feature == 0][0]
        )

        binner = WoeBinner(n_bins=2, method="cart", min_samples_bin=5)
        binner.fit(pl.DataFrame({"x": x}), pl.Series("t", target),
                   sample_weight=weights)
        assert binner.bin_cuts_["x"][1] == pytest.approx(expected, abs=1e-6)

    def test_unweighted_cuts_are_unchanged(self):
        x, target, _ = self._skewed()
        frame = pl.DataFrame({"x": x})
        first = WoeBinner(n_bins=4, method="quantile", min_samples_bin=1)
        first.fit(frame, pl.Series("t", target))
        second = WoeBinner(n_bins=4, method="quantile", min_samples_bin=1)
        second.fit(frame, pl.Series("t", target), sample_weight=np.ones(len(x)))
        # Uniform weights must reproduce the unweighted fast path exactly.
        assert first.bin_cuts_["x"] == pytest.approx(second.bin_cuts_["x"])


class TestPSIEdgesUseWeights:
    def test_weighted_bins_are_true_deciles_of_the_population(self):
        """Unweighted edges put 1%-19% of the weighted population per decile."""
        rng = np.random.default_rng(3)
        bads = rng.normal(1.0, 1.0, 1500)
        goods = rng.normal(-1.0, 1.0, 1500)
        expected = np.concatenate([bads, goods])
        weights = np.concatenate([np.ones(1500), np.full(1500, 20.0)])
        actual = np.concatenate([
            rng.normal(1.0, 1.0, 1500), rng.normal(-1.0, 1.0, 30_000)
        ])

        _, table = calculate_psi(
            expected, actual, n_bins=10, expected_weight=weights
        )
        shares = [row["expected_pct"] for row in table.to_dicts()]
        assert max(shares) - min(shares) < 0.02, shares


class TestRegressionHonoursTheWeightFlag:
    @staticmethod
    def _frame():
        rng = np.random.default_rng(7)
        n = 1500
        x = rng.normal(size=n)
        group = rng.integers(0, 2, n)
        # The two groups have opposite slopes, so weighting flips the fit.
        y = np.where(group == 1, 5 * x, -5 * x) + rng.normal(0, 0.5, n)
        return pl.DataFrame({
            "x": x,
            "w": np.where(group == 1, 100.0, 1.0),
            "y": y,
            "sample": ["dev"] * 1200 + ["oot"] * 300,
        })

    def _coef(self, **kwargs):
        from auto_modeling_tool.pipelines.regression_pipeline import RegressionPipeline

        pipeline = RegressionPipeline(
            target_col="y", model_type="linear", sample_col="sample"
        )
        pipeline.fit(self._frame(), sample_col="sample", **kwargs)
        model = getattr(pipeline.model_, "model_", pipeline.model_)
        return float(np.ravel(model.coef_)[0])

    def test_flag_off_means_unweighted(self):
        """A weight column present but disabled must not weight the fit.

        The shipped config carries use_sample_weight: false next to
        weight_col: weight, so keying off the column alone made that config
        train a weighted regression and an unweighted classification.
        """
        baseline = self._coef()
        disabled = self._coef(weight_col="w", use_sample_weight=False)
        assert disabled == pytest.approx(baseline, abs=0.05)

    def test_flag_on_means_weighted(self):
        baseline = self._coef()
        enabled = self._coef(weight_col="w", use_sample_weight=True)
        assert abs(enabled - baseline) > 1.0


class TestLog1pEarlyStoppingScale:
    def test_oot_early_stopping_uses_the_transformed_target(self):
        """Otherwise the metric compares log predictions to raw labels.

        The validation curve is then flat noise and best_iteration is chosen at
        random, silently truncating the boosting run.
        """
        from auto_modeling_tool.pipelines.regression_pipeline import RegressionPipeline

        rng = np.random.default_rng(3)
        n = 2000
        x = rng.normal(size=n)
        frame = pl.DataFrame({
            "x": x,
            "amt": rng.lognormal(2.0 + 0.8 * x, 0.3),
            "sample": ["dev"] * 1600 + ["oot"] * 400,
        })

        iterations = {}
        for mode in ("dev_holdout", "oot"):
            pipeline = RegressionPipeline(
                target_col="amt", model_type="xgboost", sample_col="sample",
                target_transform="log1p", early_stopping_eval=mode,
                early_stopping_rounds=20,
                model_params={"n_estimators": 200, "max_depth": 3},
            )
            pipeline.fit(frame, sample_col="sample")
            inner = getattr(pipeline.model_, "model_", pipeline.model_)
            iterations[mode] = inner.best_iteration

        # Both now measure on the log1p scale, so they stop at a comparable
        # point instead of one of them wandering on a meaningless metric.
        assert abs(iterations["oot"] - iterations["dev_holdout"]) < 25


class TestScorecardArrayPath:
    @staticmethod
    def _built():
        from sklearn.linear_model import LogisticRegression

        from auto_modeling_tool.modeling.scorecard import ScorecardBuilder

        rng = np.random.default_rng(4)
        n = 2000
        frame = pl.DataFrame({f"f{i}": rng.normal(size=n) for i in range(4)})
        target = pl.Series(
            "t",
            (frame["f0"].to_numpy() + 0.6 * frame["f1"].to_numpy() > 0).astype(int),
        )
        binner = WoeBinner(n_bins=5, min_samples_bin=30)
        woe = binner.fit_transform(frame, target, return_type="woe")
        columns = [c for c in woe.columns if c.endswith("_bin")]
        model = LogisticRegression(max_iter=500).fit(
            woe.select(columns).to_numpy(), target.to_numpy()
        )
        card = ScorecardBuilder(base_score=600, PDO=20, target_odds=20).fit(
            model, binner, feature_names=columns
        )
        return card, frame

    def test_array_and_frame_paths_agree(self):
        """The array path labelled raw columns with WOE names, matched nothing,
        and returned a constant score with no error."""
        card, frame = self._built()
        assert np.array_equal(card.score(frame), card.score(frame.to_numpy()))

    def test_array_scores_are_not_constant(self):
        card, frame = self._built()
        assert len(np.unique(card.score(frame.to_numpy()))) > 10

    def test_wrong_column_count_raises(self):
        from auto_modeling_tool.core.exceptions import ValidationError

        card, frame = self._built()
        with pytest.raises(ValidationError, match="one column per fitted driver"):
            card.score(frame.to_numpy()[:, :2])
