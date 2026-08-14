# -*- coding: utf-8 -*-
"""Excel audit report writer used by train and independent evaluation flows."""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import polars as pl

from ..core.logger import logger


def _rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pl.DataFrame):
        return value.to_dicts()
    if isinstance(value, dict):
        return [{"key": key, "value": value_item} for key, value_item in value.items()]
    if isinstance(value, list):
        return value if all(isinstance(item, dict) for item in value) else [
            {"value": item} for item in value
        ]
    return [{"value": value}]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_model_report(
    output_dir: Union[str, Path],
    metrics: Dict[str, Any],
    *,
    feature_importance: Optional[pl.DataFrame] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tables: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """Write a compact, deterministic audit workbook.

    The workbook is intentionally additive: existing JSON/CSV reports remain
    available, while this creates the first stable Excel contract.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        logger.warning("openpyxl is not installed; skipping Excel report")
        return None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if filename:
        report_path = output / filename
    else:
        index = 1
        while (output / f"Model_Report_{index}.xlsx").exists():
            index += 1
        report_path = output / f"Model_Report_{index}.xlsx"

    workbook = Workbook()
    overview = workbook.active
    overview.title = "Overview_Performance"
    overview.append(["Metric", "Value"])
    for cell in overview[1]:
        cell.font = Font(bold=True)
    for key, value in (metrics or {}).items():
        overview.append([key, _json_safe(value)])

    if feature_importance is not None:
        sheet = workbook.create_sheet("Feature_Importance")
        rows = feature_importance.to_dicts()
        if rows:
            headers = list(rows[0].keys())
            sheet.append(headers)
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            for row in rows:
                sheet.append([_json_safe(row.get(header)) for header in headers])

    meta_sheet = workbook.create_sheet("Artifact_Metadata")
    meta_sheet.append(["Key", "Value"])
    for cell in meta_sheet[1]:
        cell.font = Font(bold=True)
    for key, value in (metadata or {}).items():
        meta_sheet.append([key, _json_safe(value)])

    for sheet_name, table in (tables or {}).items():
        safe_name = str(sheet_name)[:31] or "Table"
        sheet = workbook.create_sheet(safe_name)
        rows = _rows(table)
        if rows:
            headers = list(rows[0].keys())
            sheet.append(headers)
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            for row in rows:
                sheet.append([_json_safe(row.get(header)) for header in headers])

    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        for column_cells in sheet.columns:
            width = min(
                40,
                max(len(str(cell.value or "")) for cell in column_cells) + 2,
            )
            sheet.column_dimensions[column_cells[0].column_letter].width = width
        if sheet.max_row > 1 and sheet.max_column > 1:
            sheet.auto_filter.ref = sheet.dimensions

    workbook.save(report_path)
    logger.info(f"Model report saved to {report_path}")
    return report_path
