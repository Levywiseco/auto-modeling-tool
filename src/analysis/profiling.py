# -*- coding: utf-8 -*-
"""Data profiling workflow: quality snapshot before any modeling decisions."""

from typing import Any, Dict, List, Optional

import polars as pl

from ..core.logger import logger
from ..reports import DataProfileReport
from ._frame import NUMERIC_DTYPES, ensure_polars, group_keys_sorted, resolve_features


def _special_mask(col: str, dtype: pl.DataType, special_values: List[Any]) -> Optional[pl.Expr]:
    """Expression matching special values, or None when not applicable."""
    if not special_values:
        return None
    if dtype in NUMERIC_DTYPES:
        vals = [v for v in special_values if isinstance(v, (int, float))]
    else:
        vals = [v for v in special_values if isinstance(v, str)]
    if not vals:
        return None
    return pl.col(col).is_in(vals)


def _missing_expr(col: str, dtype: pl.DataType) -> pl.Expr:
    expr = pl.col(col).is_null()
    if dtype in (pl.Float32, pl.Float64):
        expr = expr | pl.col(col).is_nan()
    return expr


def profile_data(
    df: Any,
    *,
    features: Optional[List[str]] = None,
    target: Optional[str] = None,
    group_col: Optional[str] = None,
    special_values: Optional[List[Any]] = None,
    high_missing_thr: float = 0.9,
) -> DataProfileReport:
    """Profile a dataset's quality and distributions before modeling.

    Accepts a Pandas or Polars DataFrame and returns a structured
    :class:`~src.reports.DataProfileReport` (no files are written).

    Parameters
    ----------
    df : DataFrame
        Full business table (Pandas or Polars).
    features : list of str, optional
        Columns to profile. Defaults to every column except role columns.
    target : str, optional
        Target column; profiled separately in the overview (bad rate).
    group_col : str, optional
        Existing grouping column (e.g. application month). When given,
        ``trend_tables`` expands row counts and missing rates over groups.
    special_values : list, optional
        Sentinel values (e.g. ``[-999, -1]``) counted separately from
        genuine missing values.
    high_missing_thr : float, default 0.9
        Threshold above which a feature is flagged ``high_missing``.

    Returns
    -------
    DataProfileReport
        With ``overview_table``, ``dq_table``, ``stats_table`` and, when
        ``group_col`` is given, ``trend_tables``.

    Example
    -------
    >>> report = profile_data(df, target="target", group_col="month",
    ...                       special_values=[-999])
    >>> report.dq_table
    >>> report.trend_tables["missing_rate"]
    """
    df = ensure_polars(df)
    special_values = special_values or []
    feats = resolve_features(df, features, exclude=[target, group_col])

    n_rows = df.height
    if n_rows == 0:
        raise ValueError("Cannot profile an empty dataframe")

    # ---- overview ------------------------------------------------------
    n_numeric = sum(1 for c in feats if df.schema[c] in NUMERIC_DTYPES)
    overview: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_features": len(feats),
        "n_numeric": n_numeric,
        "n_categorical": len(feats) - n_numeric,
        "n_duplicate_rows": n_rows - df.unique().height,
        "est_size_mb": round(df.estimated_size("mb"), 3),
    }
    if target and target in df.columns:
        overview["target"] = target
        overview["bad_rate"] = round(float(df[target].mean()), 6)
    if group_col and group_col in df.columns:
        overview["group_col"] = group_col
        overview["n_groups"] = df[group_col].n_unique()
    overview_table = pl.DataFrame(
        {"item": list(overview.keys()), "value": [str(v) for v in overview.values()]}
    )

    # ---- per-feature data quality (single pass) ------------------------
    dq_exprs: List[pl.Expr] = []
    for c in feats:
        dtype = df.schema[c]
        dq_exprs.append(_missing_expr(c, dtype).sum().alias(f"{c}__miss"))
        dq_exprs.append(pl.col(c).n_unique().alias(f"{c}__uniq"))
        sp = _special_mask(c, dtype, special_values)
        dq_exprs.append(
            (sp.sum() if sp is not None else pl.lit(0)).alias(f"{c}__spec")
        )
    dq_row = df.select(dq_exprs).row(0)

    dq_records = []
    for i, c in enumerate(feats):
        n_missing = int(dq_row[i * 3])
        n_unique = int(dq_row[i * 3 + 1])
        n_special = int(dq_row[i * 3 + 2])
        missing_rate = n_missing / n_rows
        flags = []
        if missing_rate >= high_missing_thr:
            flags.append("high_missing")
        if n_unique <= 1:
            flags.append("constant")
        dq_records.append(
            {
                "feature": c,
                "dtype": str(df.schema[c]),
                "n_missing": n_missing,
                "missing_rate": round(missing_rate, 6),
                "n_special": n_special,
                "special_rate": round(n_special / n_rows, 6),
                "n_unique": n_unique,
                "flags": ",".join(flags),
            }
        )
    dq_table = pl.DataFrame(dq_records).sort("missing_rate", descending=True)

    # ---- numeric statistics (special/missing excluded) -----------------
    stats_records = []
    for c in feats:
        dtype = df.schema[c]
        if dtype not in NUMERIC_DTYPES:
            continue
        valid = ~_missing_expr(c, dtype)
        sp = _special_mask(c, dtype, special_values)
        if sp is not None:
            valid = valid & ~sp
        clean = df.filter(valid)[c]
        if clean.len() == 0:
            continue
        stats_records.append(
            {
                "feature": c,
                "count": clean.len(),
                "mean": float(clean.mean()),
                "std": float(clean.std()) if clean.len() > 1 else 0.0,
                "min": float(clean.min()),
                "p25": float(clean.quantile(0.25)),
                "median": float(clean.median()),
                "p75": float(clean.quantile(0.75)),
                "max": float(clean.max()),
            }
        )
    stats_table = pl.DataFrame(stats_records) if stats_records else pl.DataFrame()

    # ---- trends over groups --------------------------------------------
    trend_tables: Dict[str, pl.DataFrame] = {}
    if group_col and group_col in df.columns:
        groups = group_keys_sorted(df, group_col)
        trend_tables["row_count"] = (
            df.group_by(group_col).len().sort(group_col)
            .rename({"len": "row_count"})
        )
        miss_rows = []
        for c in feats:
            agg = (
                df.group_by(group_col)
                .agg(_missing_expr(c, df.schema[c]).mean().alias("rate"))
                .sort(group_col)
            )
            rate_map = dict(zip(agg[group_col].to_list(), agg["rate"].to_list()))
            miss_rows.append(
                {"feature": c, **{str(g): round(float(rate_map.get(g, 0.0)), 6) for g in groups}}
            )
        trend_tables["missing_rate"] = pl.DataFrame(miss_rows)

    metadata = {
        "workflow": "profile_data",
        "n_rows": n_rows,
        "features": len(feats),
        "target": target,
        "group_col": group_col,
        "special_values": special_values,
        "high_missing_thr": high_missing_thr,
    }

    logger.info(
        f"📋 profile_data complete: {len(feats)} features, "
        f"{sum(1 for r in dq_records if r['flags'])} flagged"
    )
    return DataProfileReport(
        overview_table=overview_table,
        dq_table=dq_table,
        stats_table=stats_table,
        trend_tables=trend_tables,
        metadata=metadata,
    )
