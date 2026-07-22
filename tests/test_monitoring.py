# -*- coding: utf-8 -*-
"""
Tests for the monitoring module (Monitor / alerting).
"""

import numpy as np
import polars as pl
import pytest

from src.analysis import profile_risk
from src.monitoring import AlertConfig, Monitor, generate_monitoring_alert
from src.reports import MonitoringReport


@pytest.fixture
def drift_data():
    """Three periods where the last period's score distribution drifts."""
    np.random.seed(11)
    frames = []
    for i, month in enumerate(["2026-01", "2026-02", "2026-03"]):
        n = 400
        shift = 0.0 if i < 2 else 0.35  # drift in the last month
        score = np.clip(np.random.beta(2, 5, n) + shift, 0, 1)
        income = np.random.lognormal(8, 0.5, n)
        target = np.random.binomial(1, np.clip(score, 0, 1))
        frames.append(pl.DataFrame({
            "month": [month] * n,
            "score": score,
            "income": income,
            "target": target,
        }))
    return pl.concat(frames)


class TestMonitor:
    def test_basic_monitoring(self, drift_data):
        monitor = Monitor(binner_params={"n_bins": 5})
        report = monitor.monitor(
            drift_data,
            features=["score", "income"],
            target="target",
            group_col="month",
        )
        assert isinstance(report, MonitoringReport)
        assert set(report.features) == {"score", "income"}
        assert {"psi", "missing_rate", "bad_rate"} <= set(report.trend_tables)

    def test_detects_drift(self, drift_data):
        monitor = Monitor(binner_params={"n_bins": 5})
        report = monitor.monitor(
            drift_data, features=["score"], target="target", group_col="month"
        )
        summary = report.summary_table.filter(pl.col("feature") == "score")
        # The shifted last month must push PSI above the warning threshold
        assert summary["psi_max"][0] > 0.1
        assert summary["status"][0] in ("警告", "严重")

    def test_stable_feature_stays_normal(self, drift_data):
        monitor = Monitor(binner_params={"n_bins": 5})
        report = monitor.monitor(
            drift_data, features=["income"], target="target", group_col="month"
        )
        summary = report.summary_table.filter(pl.col("feature") == "income")
        assert summary["status"][0] == "正常"

    def test_benchmark_df_mode(self, drift_data):
        bench = drift_data.filter(pl.col("month") == "2026-01")
        latest = drift_data.filter(pl.col("month") == "2026-03")
        monitor = Monitor(binner_params={"n_bins": 5})
        report = monitor.monitor(
            latest, features=["score"], target="target", benchmark_df=bench
        )
        assert report.metadata["benchmark"] == "benchmark_df"
        assert report.summary_table["psi_max"][0] > 0.1

    def test_reuse_binner_from_profile_risk(self, drift_data):
        profile = profile_risk(
            drift_data.filter(pl.col("month") == "2026-01"),
            target="target",
            features=["score", "income"],
            n_bins=5,
        )
        monitor = Monitor()
        report = monitor.monitor(
            drift_data,
            features=["score", "income"],
            target="target",
            group_col="month",
            binner=profile.binner,
        )
        assert report.binner is profile.binner

    def test_score_trend(self, drift_data):
        monitor = Monitor(binner_params={"n_bins": 5})
        report = monitor.monitor(
            drift_data,
            features=["score"],
            target="target",
            score_col="score",
            group_col="month",
        )
        assert "score_mean" in report.trend_tables
        assert report.metadata["score_mean_relative_delta"] is not None

    def test_requires_group_or_benchmark(self, drift_data):
        from src.core.exceptions import ValidationError
        with pytest.raises(ValidationError):
            Monitor().monitor(drift_data, features=["score"])


class TestAlerting:
    def test_alert_text_on_drift(self, drift_data):
        monitor = Monitor(binner_params={"n_bins": 5})
        report = monitor.monitor(
            drift_data,
            features=["score", "income"],
            target="target",
            score_col="score",
            group_col="month",
        )
        text = generate_monitoring_alert(
            report, score_col="score", model_features=["income"]
        )
        assert "监控报警摘要" in text
        assert "score" in text

    def test_quiet_when_stable(self, drift_data):
        monitor = Monitor(binner_params={"n_bins": 5})
        report = monitor.monitor(
            drift_data, features=["income"], target="target", group_col="month"
        )
        text = generate_monitoring_alert(report)
        assert "无需处理" in text

    def test_custom_thresholds(self, drift_data):
        monitor = Monitor(binner_params={"n_bins": 5})
        report = monitor.monitor(
            drift_data, features=["income"], target="target", group_col="month"
        )
        # Absurdly low thresholds should turn even stable features into alerts
        text = generate_monitoring_alert(
            report, config=AlertConfig(psi_warn=0.0000001, psi_critical=0.5)
        )
        assert "警告" in text

    def test_markdown_export(self, drift_data, tmp_path):
        monitor = Monitor(binner_params={"n_bins": 5})
        report = monitor.monitor(
            drift_data, features=["score"], target="target", group_col="month"
        )
        out = report.save(tmp_path / "monitoring.md")
        assert out.exists()
        assert "Monitoring Report" in out.read_text()
