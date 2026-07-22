# -*- coding: utf-8 -*-
"""Chinese-language alert text generation from monitoring reports."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..reports import MonitoringReport


@dataclass
class AlertConfig:
    """Alert thresholds for :func:`generate_monitoring_alert`.

    Severity levels: a metric ≥ its ``critical`` threshold is 🔴 严重,
    ≥ its ``warn`` threshold is 🟡 警告, otherwise silent.
    """

    psi_warn: float = 0.1
    psi_critical: float = 0.25
    missing_delta_warn: float = 0.05
    missing_delta_critical: float = 0.10
    score_mean_relative_delta_warn: float = 0.05
    score_mean_relative_delta_critical: float = 0.10
    max_items_per_priority: int = 10


def _classify(value: float, warn: float, critical: float) -> Optional[str]:
    if value >= critical:
        return "critical"
    if value >= warn:
        return "warn"
    return None


def generate_monitoring_alert(
    report: MonitoringReport,
    *,
    score_col: Optional[str] = None,
    model_features: Optional[List[str]] = None,
    config: Optional[AlertConfig] = None,
) -> str:
    """Render a monitoring report as a priority-sorted Chinese alert digest.

    Parameters
    ----------
    report : MonitoringReport
        Output of :meth:`Monitor.monitor`.
    score_col : str, optional
        Model score column name; score drift is reported as model-level risk.
    model_features : list of str, optional
        Features used by the deployed model; drift on these is prioritized
        over drift on merely-monitored features.
    config : AlertConfig, optional
        Custom thresholds; defaults to standard PSI conventions.

    Returns
    -------
    str
        Multi-line Chinese alert text, critical items first. Suitable for
        IM / email push.
    """
    cfg = config or AlertConfig()
    model_features = model_features or []
    critical: List[str] = []
    warn: List[str] = []

    def push(level: Optional[str], text: str, is_model: bool = False) -> None:
        if level == "critical":
            (critical.insert(0, text) if is_model else critical.append(text))
        elif level == "warn":
            (warn.insert(0, text) if is_model else warn.append(text))

    if not report.summary_table.is_empty():
        for row in report.summary_table.iter_rows(named=True):
            feat = row["feature"]
            is_model_feat = feat in model_features or feat == score_col
            prefix = "模型分" if feat == score_col else (
                "入模特征" if feat in model_features else "特征"
            )

            level = _classify(row.get("psi_max", 0.0), cfg.psi_warn, cfg.psi_critical)
            push(
                level,
                f"{prefix} `{feat}` PSI 达 {row['psi_max']:.4f}"
                f"（最新期 {row['psi_latest']:.4f}），分布相对基准发生"
                f"{'显著' if level == 'critical' else '一定'}偏移",
                is_model_feat,
            )

            miss_delta = abs(row.get("missing_delta", 0.0))
            level = _classify(
                miss_delta, cfg.missing_delta_warn, cfg.missing_delta_critical
            )
            direction = "上升" if row.get("missing_delta", 0.0) > 0 else "下降"
            push(
                level,
                f"{prefix} `{feat}` 缺失率较基准{direction} "
                f"{miss_delta:.2%}（{row['missing_rate_benchmark']:.2%} → "
                f"{row['missing_rate_latest']:.2%}），需排查上游取数链路",
                is_model_feat,
            )

    score_delta = report.metadata.get("score_mean_relative_delta")
    if score_delta is not None:
        level = _classify(
            abs(score_delta),
            cfg.score_mean_relative_delta_warn,
            cfg.score_mean_relative_delta_critical,
        )
        direction = "上移" if score_delta > 0 else "下移"
        push(
            level,
            f"模型分均值较基准{direction} {abs(score_delta):.2%}，"
            "客群或策略可能已变化，建议核对进件结构",
            is_model=True,
        )

    lines = ["📡 监控报警摘要", ""]
    meta = report.metadata
    lines.append(
        f"基准：{meta.get('benchmark', '-')} ｜ 周期列：{meta.get('group_col', '-')}"
        f" ｜ 期数：{meta.get('n_groups', '-')} ｜ 特征数：{len(report.features)}"
    )
    lines.append("")

    if critical:
        lines.append("🔴 严重（需立即处理）")
        for item in critical[: cfg.max_items_per_priority]:
            lines.append(f"  - {item}")
        if len(critical) > cfg.max_items_per_priority:
            lines.append(f"  - …另有 {len(critical) - cfg.max_items_per_priority} 条")
        lines.append("")
    if warn:
        lines.append("🟡 警告（建议关注）")
        for item in warn[: cfg.max_items_per_priority]:
            lines.append(f"  - {item}")
        if len(warn) > cfg.max_items_per_priority:
            lines.append(f"  - …另有 {len(warn) - cfg.max_items_per_priority} 条")
        lines.append("")
    if not critical and not warn:
        lines.append("✅ 各项指标均在阈值内，无需处理")

    return "\n".join(lines)
