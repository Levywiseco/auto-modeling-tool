# -*- coding: utf-8 -*-
"""Canonical YAML configuration loading for the modeling pipeline."""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union


def _first(mapping: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _resolve_path(value: Any, config_path: Optional[Path]) -> Any:
    if value is None or config_path is None:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((config_path.parent / path).resolve())


def load_pipeline_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config and normalize the legacy default_config wrapper."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for --config. Install it with 'pip install pyyaml'."
        ) from exc

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Pipeline config root must be a mapping")
    if isinstance(config.get("default_config"), dict):
        config = config["default_config"]
    config["_config_path"] = str(config_path)
    return config


def config_to_pipeline_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Map the canonical schema and supported legacy aliases to CLI kwargs."""
    config_path = (
        Path(config["_config_path"])
        if config.get("_config_path")
        else None
    )
    data = config.get("data", {}) or {}
    shared = config.get("shared", {}) or {}
    split = config.get("sample_split", {}) or {}
    preprocess = config.get("preprocess", config.get("preprocessing", {})) or {}
    binning = config.get("binning", {}) or {}
    features = config.get("features", {}) or {}
    screening = config.get(
        "feature_screening",
        config.get("feature_selection", features.get("selection", {})),
    ) or {}
    modeling = config.get("modeling", {}) or {}
    output = config.get("output", {}) or {}
    variables = config.get("variables", {}) or {}

    algorithm = _first(modeling, ["algorithm", "model_type"], "logistic")
    if isinstance(modeling.get("algorithms"), list) and "algorithm" not in modeling:
        first_algorithm = modeling["algorithms"][0] if modeling["algorithms"] else {}
        algorithm = (
            first_algorithm.get("name", "logistic")
            if isinstance(first_algorithm, dict)
            else "logistic"
        )
    if str(algorithm).lower().startswith("logistic"):
        algorithm = "logistic"

    train_ratio = data.get("train_test_split_ratio")
    test_size = data.get("test_size")
    if test_size is None and train_ratio is not None:
        test_size = 1 - float(train_ratio)

    data_path = _first(data, ["path", "input_path"])
    target_col = _first(shared, ["bad_col", "target_col"])
    target_col = target_col or _first(data, ["target_column", "target"])
    sample_col = _first(
        split,
        ["sample_col", "sample_column"],
        _first(shared, ["sample_col", "sample_column"]),
    )
    date_column = _first(
        split,
        ["date_col", "date_column"],
        _first(shared, ["date_col", "date_column"]),
    )
    oot_start = _first(split, ["oot_start", "oot_begin"])
    output_dir = _first(output, ["dir", "path"], data.get("output_path", "output"))

    kwargs = {
        "data_path": _resolve_path(data_path, config_path),
        "target_col": target_col,
        "output_dir": _resolve_path(output_dir, config_path),
        "test_size": float(test_size if test_size is not None else 0.2),
        "n_bins": int(_first(binning, ["n_bins"], 10)),
        "binning_method": _first(binning, ["method", "binning_method"], "quantile"),
        "selection_method": _first(screening, ["method"], "iv"),
        "n_features": int(
            _first(screening, ["n_features"], _first(modeling, ["max_features"], 20))
        ),
        "random_state": int(
            _first(shared, ["random_state"], data.get("random_state", 42))
        ),
        "sample_col": sample_col,
        "date_column": date_column,
        "oot_start": oot_start,
        "dev_label": _first(split, ["dev_label"], _first(shared, ["dev_label"], "dev")),
        "oot_label": _first(split, ["oot_label"], _first(shared, ["oot_label"], "oot")),
        "clean_strategy": _first(preprocess, ["clean_strategy"], "median"),
        "normalize_method": _first(preprocess, ["normalize_method"], "zscore"),
        "min_samples_bin": int(_first(binning, ["min_samples_bin"], 50)),
        "monotonic": bool(_first(binning, ["monotonic"], False)),
        "exclude_columns": list(_first(variables, ["exclude_columns"], []) or []),
    }
    if algorithm not in {"logistic", "xgboost", "lightgbm"}:
        raise ValueError(
            f"Unsupported algorithm in current CLI: {algorithm!r}. "
            "P1 currently supports the classification pipeline entry point."
        )
    if not kwargs["data_path"]:
        raise ValueError("Config must define data.path or data.input_path")
    if not kwargs["target_col"]:
        raise ValueError("Config must define shared.bad_col or data.target_column")
    return kwargs
