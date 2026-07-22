# -*- coding: utf-8 -*-
"""Monitoring orchestrator built on top of the binning pipeline."""

from typing import Any, Dict, List, Optional

import polars as pl

from ..analysis._frame import ensure_polars, group_keys_sorted, resolve_features
from ..binning.woe_binning import WoeBinner
from ..core.exceptions import ValidationError
from ..core.logger import logger
from ..evaluation.stability import bin_distribution, psi_from_distributions, psi_level
from ..reports import MonitoringReport


class Monitor:
    """Feature / model monitoring orchestrator.

    Stable strategy lives in the constructor; per-run data and column names
    are method parameters, so one Monitor instance can serve many datasets.

    Parameters
    ----------
    binner_params : dict, optional
        Parameters used to construct the default :class:`WoeBinner` when no
        explicit binner is passed to :meth:`monitor`
        (e.g. ``{"n_bins": 10, "missing_values": [-999]}``).
    psi_include_missing : bool, default False
        Include the missing bin in PSI calculations.
    psi_include_special : bool, default False
        Include special-value bins in PSI calculations.
    psi_warn, psi_critical : float
        Thresholds used for the ``status`` verdict in the summary table.

    Example
    -------
    >>> monitor = Monitor(binner_params={"n_bins": 10})
    >>> report = monitor.monitor(df, features=["income", "score"],
    ...                          target="target", group_col="month")
    >>> report.summary_table
    >>> report.trend_tables["psi"]
    """

    def __init__(
        self,
        *,
        binner_params: Optional[Dict[str, Any]] = None,
        psi_include_missing: bool = False,
        psi_include_special: bool = False,
        psi_warn: float = 0.1,
        psi_critical: float = 0.25,
    ) -> None:
        self.binner_params = binner_params or {}
        self.psi_include_missing = psi_include_missing
        self.psi_include_special = psi_include_special
        self.psi_warn = psi_warn
        self.psi_critical = psi_critical

    def monitor(
        self,
        df: Any,
        *,
        features: List[str],
        target: Optional[str] = None,
        score_col: Optional[str] = None,
        group_col: Optional[str] = None,
        binner: Optional[WoeBinner] = None,
        benchmark_df: Optional[Any] = None,
    ) -> MonitoringReport:
        """Run monitoring over one dataset and return a structured report.

        Parameters
        ----------
        df : DataFrame
            Observation table (Pandas or Polars) containing ``features`` and
            role columns.
        features : list of str
            Feature columns to monitor (may include the model score).
        target : str, optional
            Binary target column; enables bad-rate trend monitoring.
        score_col : str, optional
            Model score / probability column; enables score-mean trend.
        group_col : str, optional
            Period column (e.g. scoring month). Without it, ``df`` is
            compared against ``benchmark_df`` as a single period.
        binner : WoeBinner, optional
            Pre-fitted binning rules (e.g. from ``profile_risk(...).binner``)
            for consistent bucketing between development and monitoring.
        benchmark_df : DataFrame, optional
            Reference population (e.g. development sample). Defaults to the
            first group of ``group_col``.

        Returns
        -------
        MonitoringReport
        """
        df = ensure_polars(df)
        feats = resolve_features(df, features, exclude=[group_col])
        if not feats:
            raise ValidationError("No features to monitor")
        if group_col is None and benchmark_df is None:
            raise ValidationError(
                "Provide group_col (period column) or benchmark_df (reference data)"
            )

        # ---- benchmark slice -------------------------------------------
        groups: List[Any]
        if group_col is not None:
            if group_col not in df.columns:
                raise ValidationError(f"Group column '{group_col}' not found")
            groups = group_keys_sorted(df, group_col)
        else:
            groups = ["current"]

        if benchmark_df is not None:
            bench = ensure_polars(benchmark_df)
            benchmark_name = "benchmark_df"
        else:
            bench = df.filter(pl.col(group_col) == groups[0])
            benchmark_name = str(groups[0])

        # ---- binning rules ---------------------------------------------
        if binner is None:
            params = dict(self.binner_params)
            params["features"] = feats
            binner = WoeBinner(**params)
        if not binner._is_fitted:
            if target is not None and target in bench.columns:
                binner.fit(bench.select(feats), bench[target])
            else:
                # No target: fit unsupervised (quantile/uniform cuts only)
                pseudo_y = pl.Series("target", [0] * bench.height)
                binner.fit(bench.select(feats), pseudo_y)

        fitted = [c for c in feats if binner.bin_cuts_.get(c)]

        bench_binned = binner.transform(bench.select(fitted), return_type="index")
        bench_dists = {
            c: bin_distribution(bench_binned[f"{c}_bin"]) for c in fitted
        }
        bench_missing = {c: bench_dists[c].get(-1, 0.0) for c in fitted}

        # ---- per-period distributions ----------------------------------
        psi_rows: Dict[str, Dict[str, Any]] = {c: {"feature": c} for c in fitted}
        miss_rows: Dict[str, Dict[str, Any]] = {c: {"feature": c} for c in fitted}
        detail_records: List[Dict[str, Any]] = []
        psi_last: Dict[str, float] = {}
        psi_max: Dict[str, float] = {}
        miss_last: Dict[str, float] = {}

        for g in groups:
            part = df if group_col is None else df.filter(pl.col(group_col) == g)
            part_binned = binner.transform(part.select(fitted), return_type="index")
            for c in fitted:
                dist = bin_distribution(part_binned[f"{c}_bin"])
                psi = psi_from_distributions(
                    bench_dists[c],
                    dist,
                    include_missing=self.psi_include_missing,
                    include_special=self.psi_include_special,
                )
                psi_rows[c][str(g)] = round(psi, 6)
                miss_rows[c][str(g)] = round(dist.get(-1, 0.0), 6)
                psi_last[c] = psi
                psi_max[c] = max(psi_max.get(c, 0.0), psi)
                miss_last[c] = dist.get(-1, 0.0)
                labels = binner.bin_mappings_.get(c, {})
                for idx, share in sorted(dist.items()):
                    detail_records.append(
                        {
                            "feature": c,
                            "group": str(g),
                            "bin_idx": idx,
                            "bin_label": labels.get(idx, str(idx)),
                            "pct": round(share, 6),
                            "benchmark_pct": round(bench_dists[c].get(idx, 0.0), 6),
                        }
                    )

        trend_tables: Dict[str, pl.DataFrame] = {
            "psi": pl.DataFrame(list(psi_rows.values())),
            "missing_rate": pl.DataFrame(list(miss_rows.values())),
        }

        # ---- target / score trends -------------------------------------
        if target is not None and target in df.columns:
            key = group_col if group_col is not None else pl.lit("current").alias("group")
            trend_tables["bad_rate"] = (
                df.group_by(key)
                .agg(pl.len().alias("count"), pl.col(target).mean().alias("bad_rate"))
                .sort(group_col if group_col is not None else "group")
            )
        score_mean_delta: Optional[float] = None
        if score_col is not None and score_col in df.columns:
            key = group_col if group_col is not None else pl.lit("current").alias("group")
            score_trend = (
                df.group_by(key)
                .agg(
                    pl.col(score_col).mean().alias("score_mean"),
                    pl.col(score_col).std().alias("score_std"),
                )
                .sort(group_col if group_col is not None else "group")
            )
            trend_tables["score_mean"] = score_trend
            bench_score = bench[score_col].mean() if score_col in bench.columns else None
            if bench_score:
                latest_score = score_trend["score_mean"][-1]
                score_mean_delta = (latest_score - bench_score) / abs(bench_score)

        # ---- feature-level summary -------------------------------------
        summary_rows = []
        for c in fitted:
            row = {
                "feature": c,
                "psi_latest": round(psi_last.get(c, 0.0), 6),
                "psi_max": round(psi_max.get(c, 0.0), 6),
                "missing_rate_benchmark": round(bench_missing.get(c, 0.0), 6),
                "missing_rate_latest": round(miss_last.get(c, 0.0), 6),
                "missing_delta": round(
                    miss_last.get(c, 0.0) - bench_missing.get(c, 0.0), 6
                ),
                "status": psi_level(
                    psi_max.get(c, 0.0), self.psi_warn, self.psi_critical
                ),
            }
            summary_rows.append(row)
        summary = pl.DataFrame(summary_rows).sort("psi_max", descending=True)

        metadata = {
            "workflow": "monitor",
            "n_rows": df.height,
            "benchmark": benchmark_name,
            "group_col": group_col,
            "n_groups": len(groups),
            "target": target,
            "score_col": score_col,
            "score_mean_relative_delta": score_mean_delta,
            "psi_include_missing": self.psi_include_missing,
            "psi_include_special": self.psi_include_special,
        }

        n_alert = sum(1 for r in summary_rows if r["status"] != "正常")
        logger.info(
            f"📡 monitor complete: {len(fitted)} features over {len(groups)} periods, "
            f"{n_alert} flagged"
        )

        return MonitoringReport(
            summary_table=summary,
            detail_table=pl.DataFrame(detail_records),
            trend_tables=trend_tables,
            features=fitted,
            target=target,
            binner=binner,
            metadata=metadata,
        )
