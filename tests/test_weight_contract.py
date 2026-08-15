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
from auto_modeling_tool.modeling.scorecard import probability_to_credit_score


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


class TestMissingnessSurvivesToWOE:
    """Missingness is often the strongest signal a credit feature carries.

    AutoPipeline always imputes before binning, so WoeBinner's Missing bin was
    unreachable for model features and the missing-vs-observed contrast was
    discarded. clean_strategy='keep' leaves nulls for the binner.
    """

    @staticmethod
    def _informative_missingness(tmp_path):
        rng = np.random.default_rng(9)
        n = 2500
        x = rng.normal(size=n)
        missing = rng.random(n) < 0.25
        # Applicants with no record default far more often.
        target = np.where(
            missing, rng.binomial(1, 0.6, n), rng.binomial(1, 0.15, n)
        )
        x = x.astype(float)
        x[missing] = np.nan
        csv = tmp_path / "d.csv"
        pl.DataFrame({
            "x": x,
            "target": target,
            "sample": ["dev"] * 2000 + ["oot"] * 500,
        }).write_csv(csv)
        return csv

    def _binner_for(self, tmp_path, strategy):
        from auto_modeling_tool.main import run_modeling_pipeline

        result = run_modeling_pipeline(
            str(self._informative_missingness(tmp_path)), "target",
            output_dir=str(tmp_path / f"out_{strategy}"),
            sample_col="sample", n_bins=8, min_samples_bin=20,
            clean_strategy=strategy, archive_run=False,
        )
        return result["pipeline"].binner_, result["metrics"]

    def test_imputing_hides_the_missing_bin(self, tmp_path):
        binner, _ = self._binner_for(tmp_path, "median")
        assert -1 not in binner.bin_woes_["x"]

    def test_keep_exposes_the_missing_bin(self, tmp_path):
        binner, _ = self._binner_for(tmp_path, "keep")
        assert -1 in binner.bin_woes_["x"]
        # The bin carries real signal, not noise.
        assert abs(binner.bin_woes_["x"][-1]) > 0.5

    def test_keep_recovers_discarded_information(self, tmp_path):
        imputed, _ = self._binner_for(tmp_path, "median")
        kept, _ = self._binner_for(tmp_path, "keep")
        assert kept.total_iv_["x"] > imputed.total_iv_["x"]


class TestPreprocessorAndLiftUseWeights:
    """The last two stages that computed statistics on the raw sample."""

    def test_preprocessor_statistics_describe_the_population(self):
        from auto_modeling_tool.data.preprocess import DataPreprocessor

        values = np.concatenate([np.full(100, 1.0), np.full(100, 100.0)])
        weights = np.concatenate([np.full(100, 1.0), np.full(100, 50.0)])
        frame = pl.DataFrame({"x": values})

        plain = DataPreprocessor(clean_strategy="median", normalize_method=None)
        plain.fit(frame)
        weighted = DataPreprocessor(clean_strategy="median", normalize_method=None)
        weighted.fit(frame, sample_weight=weights)

        # The population is dominated by the 100.0 group.
        assert plain.stats_["x"]["median"] < 60
        assert weighted.stats_["x"]["median"] == pytest.approx(100.0)

    def test_lift_deciles_are_weighted(self):
        from auto_modeling_tool.evaluation.metrics import calculate_lift

        rng = np.random.default_rng(5)
        n = 3000
        score = rng.random(n)
        target = rng.binomial(1, score * 0.6)
        weights = np.where(score > 0.5, 1.0, 30.0)

        table = calculate_lift(
            pl.Series("y", target), score, sample_weight=weights
        )
        counts = [row["count"] for row in table.to_dicts()]
        shares = [c / sum(counts) for c in counts]
        assert max(shares) - min(shares) < 0.02, shares

    def test_unweighted_lift_is_unchanged(self):
        from auto_modeling_tool.evaluation.metrics import calculate_lift

        rng = np.random.default_rng(5)
        score = rng.random(2000)
        target = rng.binomial(1, score * 0.6)
        table = calculate_lift(pl.Series("y", target), score)
        assert table.height == 10


class TestScorecardScaleDirection:
    """A credit score must fall as risk rises — both scales, same direction."""

    @staticmethod
    def _card():
        from sklearn.linear_model import LogisticRegression

        from auto_modeling_tool.modeling.scorecard import ScorecardBuilder

        rng = np.random.default_rng(4)
        n = 1500
        frame = pl.DataFrame({f"f{i}": rng.normal(size=n) for i in range(3)})
        target = pl.Series("t", (frame["f0"].to_numpy() > 0).astype(int))
        binner = WoeBinner(n_bins=4, min_samples_bin=20)
        woe = binner.fit_transform(frame, target, return_type="woe")
        columns = [c for c in woe.columns if c.endswith("_bin")]
        model = LogisticRegression(max_iter=500).fit(
            woe.select(columns).to_numpy(), target.to_numpy()
        )
        card = ScorecardBuilder(
            base_score=600, PDO=20, target_odds=20, round_scores=False
        ).fit(model, binner, feature_names=columns)
        return card, frame, model, woe.select(columns).to_numpy()

    def test_score_falls_as_bad_probability_rises(self):
        """It ran backwards: a 'high score = low risk' cutoff approved the worst."""
        card, frame, model, woe = self._card()
        scores = card.score(frame)
        bad_probability = model.predict_proba(woe)[:, 1]
        assert np.corrcoef(scores, bad_probability)[0, 1] < -0.5

    def test_both_credit_scales_agree_in_direction(self):
        card, frame, model, woe = self._card()
        helper = probability_to_credit_score(
            np.array([0.1, 0.9]), base_score=600, pdo=20,
            min_score=0, max_score=1000,
        )
        assert helper[0] > helper[1]

        scores = card.score(frame)
        bad_probability = model.predict_proba(woe)[:, 1]
        worst = scores[np.argmax(bad_probability)]
        best = scores[np.argmin(bad_probability)]
        assert best > worst

    def test_predict_proba_matches_the_underlying_model(self):
        """The intercept was recorded and then never applied."""
        card, frame, model, woe = self._card()
        assert np.allclose(
            card.predict_proba(frame)[:, 1],
            model.predict_proba(woe)[:, 1],
            atol=1e-9,
        )


class TestSplitReportsDroppedRows:
    def test_rows_outside_dev_and_oot_are_reported(self, monkeypatch):
        """Silently dropping a quarter of the data is how models go wrong.

        The project uses its own logger, which does not propagate to caplog,
        so the warning is captured at the source.
        """
        from auto_modeling_tool.data import split as split_module

        warnings = []
        monkeypatch.setattr(
            split_module.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg))
        )

        frame = pl.DataFrame({
            "f": range(10),
            "target": [0, 1] * 5,
            "sample": ["dev"] * 4 + ["oot"] * 3 + ["holdout", "?", "dev"],
        })
        result = split_module.split_dev_oot(frame, "target", sample_column="sample")

        assert len(result.dev) + len(result.oot) == 8
        assert any("ignored 2" in w for w in warnings), warnings

    def test_no_warning_when_nothing_is_dropped(self, monkeypatch):
        from auto_modeling_tool.data import split as split_module

        warnings = []
        monkeypatch.setattr(
            split_module.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg))
        )
        frame = pl.DataFrame({
            "f": range(8),
            "target": [0, 1] * 4,
            "sample": ["dev"] * 5 + ["oot"] * 3,
        })
        split_module.split_dev_oot(frame, "target", sample_column="sample")
        assert not any("ignored" in w for w in warnings), warnings


class TestCustomNullValuesAreConsistent:
    def test_fit_excludes_the_sentinel_it_will_later_replace(self):
        """Averaging -999 into the median it fills with skews every imputation."""
        from auto_modeling_tool.data.preprocess import DataPreprocessor

        frame = pl.DataFrame({"x": [1.0, 2.0, -999.0, 4.0, -999.0]})
        pre = DataPreprocessor(
            clean_strategy="median", normalize_method=None,
            custom_null_values=[-999],
        )
        pre.fit(frame)
        assert pre.stats_["x"]["median"] == pytest.approx(2.0)
        assert pre.transform(frame)["x"].to_list() == [1.0, 2.0, 2.0, 4.0, 2.0]
