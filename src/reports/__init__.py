# -*- coding: utf-8 -*-
"""Structured report objects returned by high-level workflow entry points.

Following the "structured output first" convention: workflows return report
objects (summary_table / detail_table / trend_tables / metadata) instead of
writing files directly. Files are an export concern, handled by ``to_markdown``
/ ``save`` on each report object.
"""

from .objects import (
    BinningReport,
    DataProfileReport,
    MonitoringReport,
    RiskProfile,
)

__all__ = [
    "DataProfileReport",
    "BinningReport",
    "RiskProfile",
    "MonitoringReport",
]
