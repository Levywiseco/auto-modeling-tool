# -*- coding: utf-8 -*-
"""
Tests for task-oriented workflow entry points (profile_data / profile_risk).
"""

import numpy as np
import polars as pl
import pytest

from src.analysis import profile_data, profile_risk
from src.reports import DataProfileReport, RiskProfile


@pytest.fixture
def risk_data():
    """Credit-risk-like table with months, missing and special values."""
    np.random.seed(7)
    n = 1200
    months = ["2026-01", "2026-02", "2026-03"]

    income = np.random.lognormal(8, 0.5, n)
    utilization = np.random.beta(2, 5, n)
    # Higher utilization -> higher default probability
    prob = 0.05 + 0.5 * utilization
    target = np.random.binomial(1, np.clip(prob, 0, 1))

    income[:60] = -999          # special value
    utilization_list = utilization.tolist()
    for i in range(40):
        utilization_list[i] = None  # missing

    return pl.DataFrame({
        "month": [months[i % 3] for i in range(n)],
        "income": income,
        "utilization": utilization_list,
        "segment": ["new" if i % 2 else "repeat" for i in range(n)],
        "target": target,
    })


class TestProfileData:
    def test_returns_report_object(self, risk_data):
        report = profile_data(
            risk_data,
            target="target",
            group_col="month",
            special_values=[-999],
        )
        assert isinstance(report, DataProfileReport)
        assert not report.overview_table.is_empty()
        assert not report.dq_table.is_empty()
        assert report.metadata["workflow"] == "profile_data"

    def test_dq_counts(self, risk_data):
        report = profile_data(risk_data, target="target", special_values=[-999])
        dq = {r["feature"]: r for r in report.dq_table.iter_rows(named=True)}
        assert dq["utilization"]["n_missing"] == 40
        assert dq["income"]["n_special"] == 60
        assert dq["segment"]["n_unique"] == 2

    def test_trend_tables_with_group(self, risk_data):
        report = profile_data(risk_data, target="target", group_col="month")
        assert "missing_rate" in report.trend_tables
        assert "row_count" in report.trend_tables
        assert report.trend_tables["row_count"].height == 3

    def test_accepts_pandas(self, risk_data):
        report = profile_data(risk_data.to_pandas(), target="target")
        assert isinstance(report, DataProfileReport)

    def test_markdown_export(self, risk_data, tmp_path):
        report = profile_data(risk_data, target="target")
        md = report.to_markdown()
        assert "Data Profile Report" in md
        out = report.save(tmp_path / "profile.md")
        assert out.exists()


class TestProfileRisk:
    def test_returns_risk_profile(self, risk_data):
        profile = profile_risk(
            risk_data,
            target="target",
            features=["income", "utilization"],
            group_col="month",
            n_bins=4,
            missing_values=[-999],
            special_values=[-999],
        )
        assert isinstance(profile, RiskProfile)
        assert set(profile.features) == {"income", "utilization"}
        assert profile.binner._is_fitted

    def test_summary_columns(self, risk_data):
        profile = profile_risk(
            risk_data,
            target="target",
            features=["income", "utilization"],
            group_col="month",
            n_bins=4,
        )
        summary = profile.summary_table
        for col in ["feature", "iv", "iv_strength", "ks", "missing_rate",
                    "n_bins", "monotonic_woe", "psi_max"]:
            assert col in summary.columns
        # utilization drives the target, so it should carry meaningful IV
        util_iv = summary.filter(pl.col("feature") == "utilization")["iv"][0]
        assert util_iv > 0.02

    def test_detail_table_has_labels(self, risk_data):
        profile = profile_risk(
            risk_data, target="target", features=["utilization"], n_bins=4
        )
        detail = profile.detail_table
        assert "bin_label" in detail.columns
        assert "woe" in detail.columns
        assert detail.height >= 4

    def test_trend_tables(self, risk_data):
        profile = profile_risk(
            risk_data,
            target="target",
            features=["income", "utilization"],
            group_col="month",
        )
        trends = profile.report.trend_tables
        assert set(trends) == {"psi", "missing_rate", "bad_rate"}
        # PSI vs first month should be ~0 for the benchmark month itself
        psi_row = trends["psi"].filter(pl.col("feature") == "utilization")
        assert abs(psi_row["2026-01"][0]) < 1e-6

    def test_binner_reuse(self, risk_data):
        profile = profile_risk(
            risk_data, target="target", features=["utilization"], n_bins=4
        )
        binned = profile.binner.transform(
            risk_data.select(["utilization"]), return_type="index"
        )
        assert "utilization_bin" in binned.columns

    def test_defaults_to_numeric_features(self, risk_data):
        profile = profile_risk(risk_data, target="target", group_col="month")
        # segment is categorical and must be skipped by default
        assert "segment" not in profile.features
        assert "target" not in profile.features

    def test_missing_target_raises(self, risk_data):
        from src.core.exceptions import ValidationError
        with pytest.raises(ValidationError):
            profile_risk(risk_data, target="nonexistent")
