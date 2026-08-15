"""Run history: archive each pipeline run and compare runs after the fact.

A run directory is self-contained — the config that produced it, the metrics it
reached, and the deployable artifact — so a model can be traced back months
later without re-running anything:

    runs/20260815-093012-logistic-a3f21c/
        run.json              # lightweight index: metrics, config, features
        config.yaml           # the fully resolved config, CLI overrides applied
        scoring_artifact.pkl
        pipeline.pkl
        Model_Report_1.xlsx

`run.json` holds everything `list` and `compare` need, so browsing history never
loads a pickle.
"""

import argparse
import json
import math
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

RUN_JSON = "run.json"
RUN_CONFIG = "config.yaml"
RUN_ID_TIME_FORMAT = "%Y%m%d-%H%M%S"

# Copied into the run directory when present in the pipeline output.
ARCHIVED_ARTIFACTS = ("scoring_artifact.pkl", "pipeline.pkl")
ARCHIVED_GLOBS = ("Model_Report_*.xlsx",)

# Ranked most-informative-first so comparisons lead with what matters.
HEADLINE_METRICS = (
    "auc_roc",
    "ks_statistic",
    "gini",
    "score_psi",
    "rmse",
    "mae",
    "r2",
)

# Noise in a config diff — these differ on every run by construction.
_DIFF_IGNORED_CONFIG_KEYS = frozenset({"output_dir", "runs_dir", "_config_path"})


@dataclass
class RunRecord:
    """One archived run, as read back from ``run.json``."""

    run_id: str
    path: Path
    created_at: str = ""
    task: str = "classification"
    model_type: str = ""
    data_path: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)

    @property
    def artifact_path(self) -> Optional[Path]:
        candidate = self.path / "scoring_artifact.pkl"
        return candidate if candidate.exists() else None

    def headline(self) -> dict[str, float]:
        """Metrics worth showing in a one-line summary."""
        return {
            name: self.metrics[name]
            for name in HEADLINE_METRICS
            if name in self.metrics and self.metrics[name] is not None
        }


def _short_hash(payload: str) -> str:
    """Six hex chars, enough to separate runs landing in the same second."""
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:6]


def new_run_id(
    model_type: str = "model",
    *,
    timestamp: Optional[datetime] = None,
    seed: Optional[str] = None,
) -> str:
    """Build a sortable, human-readable run id.

    ``seed`` exists so tests can pin the hash. Callers must leave it unset: the
    suffix has to vary per call, or a parameter sweep whose runs share a second
    produces one id for every run and each archive overwrites the last.
    """
    moment = timestamp or datetime.now()
    stamp = moment.strftime(RUN_ID_TIME_FORMAT)
    safe_model = "".join(c if c.isalnum() else "-" for c in str(model_type)) or "model"
    entropy = seed if seed is not None else uuid.uuid4().hex
    return f"{stamp}-{safe_model}-{_short_hash(stamp + safe_model + entropy)}"


def _unique_destination(runs_dir: Path, run_id: str) -> Path:
    """Never reuse an existing run directory; suffix instead of overwriting."""
    destination = runs_dir / run_id
    if not destination.exists():
        return destination
    for suffix in range(2, 1000):
        candidate = runs_dir / f"{run_id}-{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Cannot find a free run directory for {run_id}")


def _jsonable(value: Any) -> Any:
    """Coerce values into something a strict JSON reader accepts.

    json.dump writes NaN/Infinity by default, which Python reads back but which
    is not valid JSON — any other tool reading run.json would choke. Non-finite
    numbers become null instead.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def archive_run(
    output_dir: Union[str, Path],
    *,
    runs_dir: Union[str, Path] = "runs",
    config: Optional[dict[str, Any]] = None,
    metrics: Optional[dict[str, Any]] = None,
    selected_features: Optional[list[str]] = None,
    task: str = "classification",
    model_type: str = "",
    run_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Path:
    """Copy a completed run's outputs into ``runs/<run_id>/`` and index it.

    ``output_dir`` keeps its contents untouched, so every existing path such as
    ``output/scoring_artifact.pkl`` keeps working exactly as before.
    """
    source = Path(output_dir)
    if not source.exists():
        raise FileNotFoundError(f"Pipeline output directory not found: {source}")

    config = dict(config or {})
    metrics = dict(metrics or {})
    # One clock reading for both the id and created_at, so they cannot disagree
    # when the call straddles a second boundary.
    moment = timestamp or datetime.now()
    run_id = run_id or new_run_id(
        model_type or config.get("model_type", "model"),
        timestamp=moment,
    )

    destination = _unique_destination(Path(runs_dir), run_id)
    run_id = destination.name
    destination.mkdir(parents=True, exist_ok=True)

    for name in ARCHIVED_ARTIFACTS:
        candidate = source / name
        if candidate.exists():
            shutil.copy2(candidate, destination / name)

    # The pipeline writes an incrementing Model_Report_N.xlsx into a shared
    # output directory, so copying the whole glob would drag every earlier
    # run's report into this archive and grow without bound. Only the report
    # this run just wrote — the most recently modified — belongs here.
    for pattern in ARCHIVED_GLOBS:
        reports = sorted(source.glob(pattern), key=lambda p: p.stat().st_mtime)
        if reports:
            newest = reports[-1]
            shutil.copy2(newest, destination / newest.name)

    if config:
        _write_config_snapshot(destination / RUN_CONFIG, config)

    record = {
        "run_id": run_id,
        "created_at": moment.isoformat(timespec="seconds"),
        "task": task,
        "model_type": str(model_type or config.get("model_type", "")),
        "data_path": str(config.get("data_path", "")),
        "metrics": _jsonable(metrics),
        "config": _jsonable(config),
        "selected_features": list(selected_features or []),
    }
    (destination / RUN_JSON).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def _write_config_snapshot(path: Path, config: dict[str, Any]) -> None:
    """Persist the resolved config, preferring YAML and falling back to JSON."""
    payload = _jsonable(config)
    try:
        import yaml

        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
    except ImportError:
        path.with_suffix(".json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def load_run(run_dir: Union[str, Path]) -> RunRecord:
    """Read a single run directory back into a record."""
    path = Path(run_dir)
    index = path / RUN_JSON
    if not index.exists():
        raise FileNotFoundError(f"Not a run directory (no {RUN_JSON}): {path}")
    payload = json.loads(index.read_text(encoding="utf-8"))
    return RunRecord(
        run_id=payload.get("run_id", path.name),
        path=path,
        created_at=payload.get("created_at", ""),
        task=payload.get("task", "classification"),
        model_type=payload.get("model_type", ""),
        data_path=payload.get("data_path", ""),
        metrics=payload.get("metrics", {}) or {},
        config=payload.get("config", {}) or {},
        selected_features=payload.get("selected_features", []) or [],
    )


def list_runs(runs_dir: Union[str, Path] = "runs") -> list[RunRecord]:
    """All archived runs, newest first. Unreadable directories are skipped."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    records = []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            records.append(load_run(candidate))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    records.sort(key=lambda r: (r.created_at, r.run_id), reverse=True)
    return records


def resolve_run(reference: str, runs_dir: Union[str, Path] = "runs") -> RunRecord:
    """Look up a run by exact id, unique id prefix, or directory path."""
    # The archive is the authority. A bare run id that happens to match a
    # directory in the current working directory must not shadow the real run:
    # resolving "20260815-..." to some unrelated local folder silently compares
    # the wrong data.
    candidates = list_runs(runs_dir)
    for record in candidates:
        if record.run_id == reference:
            return record

    direct = Path(reference)
    if direct.is_dir() and (direct / RUN_JSON).exists():
        return load_run(direct)

    prefixed = [r for r in candidates if r.run_id.startswith(reference)]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        raise ValueError(
            f"Run reference {reference!r} is ambiguous; matches: "
            + ", ".join(r.run_id for r in prefixed[:5])
        )
    raise FileNotFoundError(f"No run matching {reference!r} under {runs_dir}")


def compare_runs(left: RunRecord, right: RunRecord) -> dict[str, Any]:
    """Diff two runs across metrics, config and the selected feature set."""
    metric_names = [
        name for name in HEADLINE_METRICS
        if name in left.metrics or name in right.metrics
    ]
    metric_names += sorted(
        (set(left.metrics) | set(right.metrics)) - set(HEADLINE_METRICS)
    )

    metrics = {}
    for name in metric_names:
        before, after = left.metrics.get(name), right.metrics.get(name)
        delta = None
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            delta = after - before
        metrics[name] = {"left": before, "right": after, "delta": delta}

    config = {}
    for key in sorted(set(left.config) | set(right.config)):
        if key in _DIFF_IGNORED_CONFIG_KEYS:
            continue
        before, after = left.config.get(key), right.config.get(key)
        if before != after:
            config[key] = {"left": before, "right": after}

    left_features, right_features = set(left.selected_features), set(right.selected_features)
    return {
        "left": left.run_id,
        "right": right.run_id,
        "metrics": metrics,
        "config_changes": config,
        "features": {
            "added": sorted(right_features - left_features),
            "removed": sorted(left_features - right_features),
            "shared": len(left_features & right_features),
        },
    }


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def format_run_table(records: list[RunRecord]) -> str:
    """Render `runs list` output as an aligned table."""
    if not records:
        return "No runs archived yet."

    metric_names = [
        name for name in HEADLINE_METRICS
        if any(name in r.metrics for r in records)
    ]
    headers = ["RUN ID", "CREATED", "TASK", "MODEL"] + [m.upper() for m in metric_names]
    rows = [
        [
            r.run_id,
            r.created_at.replace("T", " "),
            r.task,
            r.model_type or "-",
        ]
        + [_format_metric(r.metrics.get(name)) for name in metric_names]
        for r in records
    ]

    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    lines = ["  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def format_comparison(diff: dict[str, Any]) -> str:
    """Render `runs compare` output for a terminal."""
    lines = [f"{diff['left']}  ->  {diff['right']}", ""]

    lines.append("Metrics")
    any_metric = False
    for name, values in diff["metrics"].items():
        delta = values["delta"]
        if delta is None:
            arrow = ""
        elif isinstance(delta, float) and math.isnan(delta):
            # A metric that turned NaN is a failure, not "no change".
            arrow = "  (became NaN)"
        elif delta > 0:
            arrow = f"  (+{delta:.4f})"
        elif delta < 0:
            arrow = f"  ({delta:.4f})"
        else:
            arrow = "  (=)"
        lines.append(
            f"  {name:<16} {_format_metric(values['left']):>10}"
            f" -> {_format_metric(values['right']):>10}{arrow}"
        )
        any_metric = True
    if not any_metric:
        lines.append("  (no metrics recorded)")

    lines.append("")
    lines.append("Config changes")
    if diff["config_changes"]:
        for key, values in diff["config_changes"].items():
            lines.append(f"  {key:<24} {values['left']!r} -> {values['right']!r}")
    else:
        lines.append("  (identical)")

    features = diff["features"]
    lines.append("")
    lines.append(f"Features  ({features['shared']} shared)")
    if features["added"]:
        lines.append(f"  added:   {', '.join(features['added'])}")
    if features["removed"]:
        lines.append(f"  removed: {', '.join(features['removed'])}")
    if not features["added"] and not features["removed"]:
        lines.append("  (identical)")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m auto_modeling_tool.runs",
        description="Browse and compare archived modeling runs.",
    )
    parser.add_argument(
        "--runs-dir", default="runs", help="Run archive directory (default: runs)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="List archived runs, newest first")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--json", action="store_true", help="Emit JSON")

    show = sub.add_parser("show", help="Show one run in detail")
    show.add_argument("run", help="Run id, unique prefix, or directory path")

    compare = sub.add_parser("compare", help="Compare two runs")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("--json", action="store_true", help="Emit JSON")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "list":
        records = list_runs(args.runs_dir)[: args.limit]
        if args.json:
            print(json.dumps(
                [
                    {
                        "run_id": r.run_id,
                        "created_at": r.created_at,
                        "task": r.task,
                        "model_type": r.model_type,
                        "metrics": r.metrics,
                    }
                    for r in records
                ],
                indent=2,
                ensure_ascii=False,
            ))
        else:
            print(format_run_table(records))
        return 0

    try:
        if args.command == "show":
            record = resolve_run(args.run, args.runs_dir)
            print(json.dumps(
                {
                    "run_id": record.run_id,
                    "path": str(record.path),
                    "created_at": record.created_at,
                    "task": record.task,
                    "model_type": record.model_type,
                    "data_path": record.data_path,
                    "metrics": record.metrics,
                    "selected_features": record.selected_features,
                    "config": record.config,
                },
                indent=2,
                ensure_ascii=False,
            ))
            return 0

        if args.command == "compare":
            left = resolve_run(args.left, args.runs_dir)
            right = resolve_run(args.right, args.runs_dir)
            diff = compare_runs(left, right)
            print(json.dumps(diff, indent=2, ensure_ascii=False) if args.json
                  else format_comparison(diff))
            return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
