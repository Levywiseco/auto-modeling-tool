# -*- coding: utf-8 -*-
"""Tests for canonical configuration loading."""

from pathlib import Path

import pytest

from src.config import config_to_pipeline_kwargs, load_pipeline_config


def test_canonical_config_resolves_relative_paths(tmp_path: Path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
shared:
  bad_col: target
  sample_col: sample
data:
  path: data/input.csv
output:
  dir: output
binning:
  n_bins: 7
feature_screening:
  method: iv
  n_features: 5
""",
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)
    kwargs = config_to_pipeline_kwargs(config)

    assert kwargs["target_col"] == "target"
    assert kwargs["sample_col"] == "sample"
    assert kwargs["n_bins"] == 7
    assert kwargs["n_features"] == 5
    assert kwargs["data_path"] == str(tmp_path / "data" / "input.csv")
    assert kwargs["output_dir"] == str(tmp_path / "output")


def test_legacy_default_config_wrapper_is_supported(tmp_path: Path):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        """
default_config:
  data:
    path: data.csv
    target_column: bad_flag
    test_size: 0.3
  modeling:
    model_type: logistic
""",
        encoding="utf-8",
    )

    kwargs = config_to_pipeline_kwargs(load_pipeline_config(config_path))
    assert kwargs["target_col"] == "bad_flag"
    assert kwargs["test_size"] == pytest.approx(0.3)


def test_supported_classification_algorithms_are_configurable():
    kwargs = config_to_pipeline_kwargs(
        {
            "data": {"path": "data.csv"},
            "shared": {"bad_col": "target"},
            "modeling": {
                "algorithm": "xgboost",
                "early_stopping_eval": "dev_holdout",
                "early_stopping_rounds": 20,
            },
        }
    )
    assert kwargs["model_type"] == "xgboost"
    assert kwargs["early_stopping_eval"] == "dev_holdout"
    assert kwargs["early_stopping_rounds"] == 20


def test_unsupported_algorithm_is_rejected():
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        config_to_pipeline_kwargs(
            {
                "data": {"path": "data.csv"},
                "shared": {"bad_col": "target"},
                "modeling": {"algorithm": "unsupported_model"},
            }
        )


def test_regression_config_maps_task_controls(tmp_path: Path):
    config_path = tmp_path / "regression.yaml"
    config_path.write_text(
        """
shared:
  bad_col: loss
  target_mode: regression
data:
  path: data/regression.csv
modeling:
  algorithm: linear
  target_transform: log1p
""",
        encoding="utf-8",
    )

    kwargs = config_to_pipeline_kwargs(load_pipeline_config(config_path))
    assert kwargs["target_mode"] == "regression"
    assert kwargs["model_type"] == "linear"
    assert kwargs["target_transform"] == "log1p"
