# -*- coding: utf-8 -*-
"""Report dataclasses shared by analysis / monitoring workflows.

Conventions (mirrored across every report object):

- ``summary_table``   feature-level aggregation, one row per feature
- ``detail_table``    bin-level records, one row per (feature, bin)
- ``trend_tables``    dict of metric name -> wide table expanded over groups/time
- ``metadata``        run context: parameters, column roles, row counts

All tables are Polars DataFrames. Use ``to_pandas()`` on any table if a
Pandas view is needed downstream.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import polars as pl

if TYPE_CHECKING:
    from ..binning.woe_binning import WoeBinner


def _table_to_markdown(df: pl.DataFrame, max_rows: int = 50) -> str:
    """Render a Polars DataFrame as a GitHub-flavored markdown table."""
    if df is None or df.is_empty():
        return "_(empty)_"

    view = df.head(max_rows)
    headers = view.columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in view.iter_rows():
        cells = []
        for v in row:
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    if df.height > max_rows:
        lines.append(f"\n_… {df.height - max_rows} more rows omitted_")
    return "\n".join(lines)


class _MarkdownReportMixin:
    """Shared markdown export for report objects."""

    _title: str = "Report"

    def _sections(self) -> List[tuple]:
        """Return (section_title, table_or_text) pairs. Overridden by subclasses."""
        raise NotImplementedError

    def to_markdown(self, max_rows: int = 50) -> str:
        parts = [f"# {self._title}", ""]
        meta = getattr(self, "metadata", None)
        if meta:
            parts.append("**Run metadata**")
            parts.append("")
            for k, v in meta.items():
                parts.append(f"- `{k}`: {v}")
            parts.append("")
        for title, content in self._sections():
            parts.append(f"## {title}")
            parts.append("")
            if isinstance(content, pl.DataFrame):
                parts.append(_table_to_markdown(content, max_rows=max_rows))
            else:
                parts.append(str(content))
            parts.append("")
        return "\n".join(parts)

    def save(self, path: Union[str, Path]) -> Path:
        """Write the markdown rendering of this report to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path


@dataclass
class DataProfileReport(_MarkdownReportMixin):
    """Data-quality snapshot produced by :func:`src.analysis.profile_data`.

    Attributes
    ----------
    overview_table : pl.DataFrame
        Dataset-level facts (rows, columns, dtype counts, duplicates).
    dq_table : pl.DataFrame
        Per-feature data quality: missing / special / unique / constant.
    stats_table : pl.DataFrame
        Per-feature descriptive statistics for numeric columns
        (special and missing values excluded).
    trend_tables : dict of str -> pl.DataFrame
        Present when ``group_col`` was given: ``row_count``,
        ``missing_rate`` expanded over groups.
    metadata : dict
        Run parameters and column roles.
    """

    overview_table: pl.DataFrame
    dq_table: pl.DataFrame
    stats_table: pl.DataFrame
    trend_tables: Dict[str, pl.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    _title = "Data Profile Report"

    def _sections(self) -> List[tuple]:
        sections = [
            ("Overview", self.overview_table),
            ("Data Quality", self.dq_table),
            ("Numeric Statistics", self.stats_table),
        ]
        for name, table in self.trend_tables.items():
            sections.append((f"Trend · {name}", table))
        return sections


@dataclass
class BinningReport(_MarkdownReportMixin):
    """Binning evaluation results (the ``report`` inside :class:`RiskProfile`).

    Attributes
    ----------
    summary_table : pl.DataFrame
        One row per feature: iv, ks, missing_rate, n_bins, monotonicity,
        psi_max (when groups are available), iv_strength label.
    detail_table : pl.DataFrame
        One row per (feature, bin): counts, bad_rate, woe, iv contribution.
    trend_tables : dict of str -> pl.DataFrame
        Present when ``group_col`` was given: ``psi`` / ``missing_rate`` /
        ``bad_rate`` expanded over groups.
    metadata : dict
        Binning parameters and run context.
    """

    summary_table: pl.DataFrame
    detail_table: pl.DataFrame
    trend_tables: Dict[str, pl.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    _title = "Binning & Risk Evaluation Report"

    def _sections(self) -> List[tuple]:
        sections = [
            ("Feature Summary", self.summary_table),
            ("Bin Detail", self.detail_table),
        ]
        for name, table in self.trend_tables.items():
            sections.append((f"Trend · {name}", table))
        return sections


@dataclass
class RiskProfile:
    """Unified result of :func:`src.analysis.profile_risk`.

    Holds the evaluation report together with the fitted binner so binning
    rules can be reused on new data (scoring, monitoring, benchmark periods).

    Attributes
    ----------
    report : BinningReport
        Structured evaluation tables.
    binner : WoeBinner
        Fitted binning transformer; reuse via ``binner.transform(new_df)``
        or pass to ``Monitor.monitor(..., binner=profile.binner)``.
    features : list of str
        Features that were actually binned.
    target : str
        Target column name used for evaluation.
    metadata : dict
        Run parameters and context.
    """

    report: BinningReport
    binner: "WoeBinner"
    features: List[str]
    target: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary_table(self) -> pl.DataFrame:
        """Shortcut for ``report.summary_table``."""
        return self.report.summary_table

    @property
    def detail_table(self) -> pl.DataFrame:
        """Shortcut for ``report.detail_table``."""
        return self.report.detail_table


@dataclass
class MonitoringReport(_MarkdownReportMixin):
    """Feature / model monitoring results from :class:`src.monitoring.Monitor`.

    Attributes
    ----------
    summary_table : pl.DataFrame
        One row per feature: psi_max, psi_latest, missing rates and deltas,
        and a ``status`` verdict (``正常`` / ``警告`` / ``严重``).
    detail_table : pl.DataFrame
        Bin-level distribution per (feature, bin, group).
    trend_tables : dict of str -> pl.DataFrame
        ``psi`` / ``missing_rate`` always; ``bad_rate`` when a target is
        given; ``score_mean`` when a score column is given.
    features : list of str
        Monitored feature names.
    target : str or None
        Target column, when performance monitoring was possible.
    binner : WoeBinner or None
        Binning rules used to bucket features (fitted on the benchmark).
    metadata : dict
        Benchmark definition, group column, row counts, parameters.
    """

    summary_table: pl.DataFrame
    detail_table: pl.DataFrame
    trend_tables: Dict[str, pl.DataFrame] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    target: Optional[str] = None
    binner: Optional["WoeBinner"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    _title = "Monitoring Report"

    def _sections(self) -> List[tuple]:
        sections = [("Feature Summary", self.summary_table)]
        for name, table in self.trend_tables.items():
            sections.append((f"Trend · {name}", table))
        sections.append(("Bin Detail", self.detail_table))
        return sections
