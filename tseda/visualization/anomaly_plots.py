"""
Anomaly visualisation plots.

Functions
---------
plot_anomalies      — series with flagged anomaly markers
plot_anomaly_scores — anomaly score timeline with threshold line
plot_anomaly_heatmap— multi-method agreement heatmap  *(innovative)*
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from tseda.anomaly.detector import AnomalyReport
from tseda.core.timeseries import TimeSeries
from tseda.visualization.base import PALETTE, _make_fig_ax, _set_title

__all__ = ["plot_anomalies", "plot_anomaly_scores", "plot_anomaly_heatmap"]


def plot_anomalies(
    ts: TimeSeries,
    report: AnomalyReport,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Line plot of *ts* with anomaly positions highlighted as red markers.

    Parameters
    ----------
    ts : TimeSeries
    report : AnomalyReport
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_fig_ax(ax, figsize, (12, 4))
    ax.plot(ts.index, ts.values, color=PALETTE["accent"],
            linewidth=1.0, label=ts.name, zorder=2)
    if report.n_anomalies > 0:
        ax.scatter(
            report.timestamps, report.values,
            color=PALETTE["anomaly"], s=40, zorder=5,
            label=f"Anomaly ({report.method}, n={report.n_anomalies})",
        )
    ax.set_xlabel("Time")
    ax.set_ylabel(ts.unit or "Value")
    ax.legend(fontsize=9)
    _set_title(ax, title, f"Anomalies — {ts.name} [{report.method}]")
    fig.tight_layout()
    return fig


def plot_anomaly_scores(
    report: AnomalyReport,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Anomaly score timeline with a reference threshold line at 1.0.

    The ``scores`` field of :class:`~tseda.anomaly.detector.AnomalyReport` is
    normalised so that scores ≥ 1.0 correspond to detected anomalies.

    Parameters
    ----------
    report : AnomalyReport
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_fig_ax(ax, figsize, (12, 3))
    n = len(report.scores)
    ax.plot(range(n), report.scores, color=PALETTE["neutral"],
            linewidth=0.8, label="score")
    ax.axhline(1.0, color=PALETTE["anomaly"], linewidth=1.5,
               linestyle="--", alpha=0.9, label="threshold (1.0)")
    ax.fill_between(
        range(n), 0, report.scores,
        where=report.mask[:n], color=PALETTE["anomaly"], alpha=0.4,
        label="flagged",
    )
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Anomaly score")
    ax.legend(fontsize=9)
    _set_title(ax, title, f"Anomaly scores [{report.method}]")
    fig.tight_layout()
    return fig


def plot_anomaly_heatmap(
    ts: TimeSeries,
    reports: List[AnomalyReport],
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Multi-method anomaly agreement heatmap.

    Rows = detection methods; columns = time positions.
    Cells are 1 (anomaly detected) or 0 (normal).
    Persistent vertical columns indicate high-agreement anomalies.

    Parameters
    ----------
    ts : TimeSeries
    reports : list of AnomalyReport
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not reports:
        fig, ax = _make_fig_ax(ax, figsize, (10, 3))
        ax.text(0.5, 0.5, "No reports provided",
                transform=ax.transAxes, ha="center")
        _set_title(ax, title, "Anomaly heatmap")
        fig.tight_layout()
        return fig

    n = ts.n
    methods = [r.method for r in reports]
    grid = np.zeros((len(reports), n), dtype=float)
    for i, r in enumerate(reports):
        for idx in r.indices:
            if idx < n:
                grid[i, idx] = 1.0

    fig, ax = _make_fig_ax(ax, figsize, (12, max(2, len(reports) * 0.8 + 1)))
    im = ax.imshow(grid, aspect="auto", cmap="Reds",
                   vmin=0, vmax=1,
                   extent=[0, n, -0.5, len(reports) - 0.5])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Method")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Anomaly")
    _set_title(ax, title, f"Multi-method anomaly agreement — {ts.name}")
    fig.tight_layout()
    return fig