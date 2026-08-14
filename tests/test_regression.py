"""Regression task and raw-driver scoring tests."""

from pathlib import Path

import numpy as np
import polars as pl

from auto_modeling_tool.evaluation.metrics import calculate_regression_metrics
from auto_modeling_tool.modeling.artifact import load_scoring_artifact, score_with_artifact
from auto_modeling_tool.modeling.train import ModelTrainer
from auto_modeling_tool.pipelines.regression_pipeline import RegressionPipeline


def test_regression_metrics():
    metrics = calculate_regression_metrics(
        [1.0, 2.0, 3.0],
        [1.0, 2.5, 2.5],
    )
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert metrics["r2"] < 1


def test_model_trainer_supports_regression():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 2 * X[:, 0] + 1
    trainer = ModelTrainer(model_type="linear", task="regression")
    trainer.fit(X, y)
    assert np.allclose(trainer.predict(X), y)
    assert trainer.get_model_summary()["task"] == "regression"


def test_regression_pipeline_log1p_artifact_roundtrip(tmp_path: Path):
    x = np.arange(40, dtype=float)
    target = np.exp(0.1 * x) - 1
    sample = np.array(["dev"] * 30 + ["oot"] * 10)
    data = pl.DataFrame({"x": x, "target": target, "sample": sample})

    pipeline = RegressionPipeline(
        target_col="target",
        model_type="linear",
        target_transform="log1p",
        normalize_method=None,
    )
    pipeline.fit(data, sample_col="sample")
    metrics = pipeline.evaluate()
    assert metrics["rmse"] >= 0

    with_artifact = pipeline.predict(data.select(["x"]))
    path = pipeline.save(tmp_path)
    artifact = load_scoring_artifact(path / "scoring_artifact.pkl")
    from_artifact = score_with_artifact(artifact, data.select(["x"]))
    assert np.allclose(with_artifact, from_artifact)


def test_regression_release_gate_discovers_report(tmp_path: Path):
    x = np.arange(30, dtype=float)
    target = 1.5 + 0.2 * x
    sample = np.array(["dev"] * 20 + ["oot"] * 10)
    data = pl.DataFrame({"x": x, "target": target, "sample": sample})

    pipeline = RegressionPipeline(
        target_col="target",
        model_type="linear",
        normalize_method=None,
    )
    pipeline.fit(data, sample_col="sample")
    pipeline.evaluate()
    pipeline.save(tmp_path)

    from auto_modeling_tool.evaluation.quality_gate import validate_release

    result = validate_release(tmp_path)
    assert result.passed
