# -*- coding: utf-8 -*-
"""Feature risk profiling workflow: binning + evaluation in one call."""

from typing import Any, Dict, List, Literal, Optional

import polars as pl

from ..binning.woe_binning import WoeBinner
from ..core.exceptions import ValidationError
from ..core.logger import logger
from ..evaluation.stability import bin_distribution, psi_from_distributions
from ..reports import BinningReport, RiskProfile
from ._frame import ensure_polars, group_keys_sorted, resolve_features


def _interpret_iv(iv: float) -> str:
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious"


def _ks_from_bin_stats(stats: pl.DataFrame) -> float:
    """Binned KS: max |cum bad share - cum good share| over ordered bins."""
    if stats.is_empty():
        return 0.0
    s = stats.sort("bin_idx")
    cum_bad = cum_good = 0.0
    ks = 0.0
    for db, dg in zip(s["dist_bad"].to_list(), s["dist_good"].to_list()):
        cum_bad += db or 0.0
        cum_good += dg or 0.0
        ks = max(ks, abs(cum_bad - cum_good))
    return ks


def _is_monotonic(woes: List[float]) -> bool:
    if len(woes) < 2:
        return True
    inc = all(b >= a for a, b in zip(woes, woes[1:]))
    dec = all(b <= a for a, b in zip(woes, woes[1:]))
    return inc or dec


def profile_risk(
    df: Any,
    *,
    target: str,
    features: Optional[List[str]] = None,
    group_col: Optional[str] = None,
    method: Literal["quantile", "uniform", "cart"] = "quantile",
    n_bins: int = 5,
    missing_values: Optional[List[Any]] = None,
    special_values: Optional[List[Any]] = None,
    psi_include_missing: bool = False,
    psi_include_special: bool = False,
    monotonic: bool = False,
    binner: Optional[WoeBinner] = None,
) -> RiskProfile:
    """Bin features, evaluate their risk-ranking power, and assemble a report.

    The one-call workflow entry point: accepts a Pandas or Polars business
    table, fits (or reuses) binning rules, and returns a
    :class:`~src.reports.RiskProfile` holding structured tables plus the
    fitted binner for reuse on new data.

    Parameters
    ----------
    df : DataFrame
        Full business table containing features, target, and role columns.
    target : str
        Binary target column (1 = bad).
    features : list of str, optional
        Features to evaluate. Defaults to all numeric non-role columns.
    group_col : str, optional
        Grouping column (e.g. application month). Enables PSI and trend
        tables expanded over groups; the first group is the PSI benchmark.
    method : {"quantile", "uniform", "cart"}, default "quantile"
        Binning strategy (ignored when ``binner`` is provided).
    n_bins : int, default 5
        Number of bins per feature (ignored when ``binner`` is provided).
    missing_values, special_values : list, optional
        Sentinels routed to dedicated bins (e.g. ``[-999]``).
    psi_include_missing, psi_include_special : bool, default False
        Whether missing / special bins participate in PSI.
    monotonic : bool, default False
        Enforce monotonic WOE for the CART method.
    binner : WoeBinner, optional
        A pre-fitted binner to reuse existing rules instead of refitting.

    Returns
    -------
    RiskProfile
        ``profile.report.summary_table`` — per-feature iv / ks / psi_max;
        ``profile.report.detail_table`` — per-bin stats;
        ``profile.report.trend_tables`` — psi / missing_rate / bad_rate over
        groups (when ``group_col`` given); ``profile.binner`` — fitted rules.

    Example
    -------
    >>> profile = profile_risk(df, target="target",
    ...                        features=["income", "utilization"],
    ...                        group_col="month", n_bins=4,
    ...                        missing_values=[-999], special_values=[-999])
    >>> profile.summary_table
    >>> profile.binner.transform(new_df)
    """
    df = ensure_polars(df)
    if target not in df.columns:
        raise ValidationError(f"Target column '{target}' not found in dataframe")

    feats = resolve_features(
        df, features, exclude=[target, group_col], numeric_only=True
    )
    if not feats:
        raise ValidationError("No numeric features available for risk profiling")

    y = df[target]
    X = df.select(feats)

    # ---- fit or reuse binning rules ------------------------------------
    if binner is None:
        binner = WoeBinner(
            n_bins=n_bins,
            method=method,
            features=feats,
            missing_values=missing_values,
            special_values=special_values,
            monotonic=monotonic,
        )
        binner.fit(X, y)
    elif not binner._is_fitted:
        binner.fit(X, y)

    fitted_feats = [c for c in feats if c in binner.bin_cuts_ and binner.bin_cuts_[c]]

    # ---- bin-level detail ----------------------------------------------
    detail = binner.compute_bin_stats(X, y)
    if not detail.is_empty():
        label_rows = [
            {"feature": c, "bin_idx": idx, "bin_label": lab}
            for c in fitted_feats
            for idx, lab in binner.bin_mappings_.get(c, {}).items()
        ]
        if label_rows:
            detail = detail.join(
                pl.DataFrame(label_rows).with_columns(pl.col("bin_idx").cast(pl.Int16)),
                on=["feature", "bin_idx"],
                how="left",
            )
        detail = detail.sort(["feature", "bin_idx"])

    # ---- per-group structures for PSI / trends -------------------------
    trend_tables: Dict[str, pl.DataFrame] = {}
    psi_max: Dict[str, float] = {}

    if group_col:
        if group_col not in df.columns:
            raise ValidationError(f"Group column '{group_col}' not found in dataframe")
        groups = group_keys_sorted(df, group_col)
        binned_all = binner.transform(X, return_type="index").with_columns(
            df[group_col].alias(group_col), y.alias(target)
        )

        psi_rows, miss_rows = [], []
        bench_group = groups[0]
        for c in fitted_feats:
            bin_col = f"{c}_bin"
            if bin_col not in binned_all.columns:
                continue
            bench_dist = bin_distribution(
                binned_all.filter(pl.col(group_col) == bench_group)[bin_col]
            )
            psi_row: Dict[str, Any] = {"feature": c}
            miss_row: Dict[str, Any] = {"feature": c}
            worst = 0.0
            for g in groups:
                part = binned_all.filter(pl.col(group_col) == g)
                dist = bin_distribution(part[bin_col])
                psi = psi_from_distributions(
                    bench_dist,
                    dist,
                    include_missing=psi_include_missing,
                    include_special=psi_include_special,
                )
                worst = max(worst, psi)
                psi_row[str(g)] = round(psi, 6)
                miss_row[str(g)] = round(dist.get(-1, 0.0), 6)
            psi_max[c] = worst
            psi_rows.append(psi_row)
            miss_rows.append(miss_row)

        trend_tables["psi"] = pl.DataFrame(psi_rows)
        trend_tables["missing_rate"] = pl.DataFrame(miss_rows)
        trend_tables["bad_rate"] = (
            df.group_by(group_col)
            .agg(pl.len().alias("count"), pl.col(target).mean().alias("bad_rate"))
            .sort(group_col)
        )

    # ---- feature-level summary -----------------------------------------
    n_rows = df.height
    summary_rows = []
    for c in fitted_feats:
        c_stats = (
            detail.filter(pl.col("feature") == c) if not detail.is_empty() else detail
        )
        normal_woes = [
            binner.bin_woes_[c][idx]
            for idx in sorted(k for k in binner.bin_woes_.get(c, {}) if k >= 0)
        ]
        missing_expr = pl.col(c).is_null()
        if X.schema[c] in (pl.Float32, pl.Float64):
            missing_expr = missing_expr | pl.col(c).is_nan()
        iv = binner.total_iv_.get(c, 0.0)
        row: Dict[str, Any] = {
            "feature": c,
            "iv": round(iv, 6),
            "iv_strength": _interpret_iv(iv),
            "ks": round(_ks_from_bin_stats(c_stats), 6),
            "missing_rate": round(
                X.select(missing_expr.mean()).item() or 0.0, 6
            ),
            "n_bins": len([k for k in binner.bin_mappings_.get(c, {}) if k >= 0]),
            "monotonic_woe": _is_monotonic(normal_woes),
        }
        if group_col:
            row["psi_max"] = round(psi_max.get(c, 0.0), 6)
        summary_rows.append(row)

    summary = pl.DataFrame(summary_rows).sort("iv", descending=True)

    metadata = {
        "workflow": "profile_risk",
        "n_rows": n_rows,
        "target": target,
        "bad_rate": round(float(y.mean()), 6),
        "features_requested": len(feats),
        "features_fitted": len(fitted_feats),
        "method": binner.method,
        "n_bins": binner.n_bins,
        "group_col": group_col,
        "psi_include_missing": psi_include_missing,
        "psi_include_special": psi_include_special,
    }

    logger.info(
        f"🎯 profile_risk complete: {len(fitted_feats)} features evaluated, "
        f"top IV = {summary['iv'][0] if summary.height else 0:.4f}"
    )

    report = BinningReport(
        summary_table=summary,
        detail_table=detail,
        trend_tables=trend_tables,
        metadata=metadata,
    )
    return RiskProfile(
        report=report,
        binner=binner,
        features=fitted_feats,
        target=target,
        metadata=metadata,
    )
