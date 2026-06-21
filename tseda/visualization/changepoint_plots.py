"""
Changepoint visualisation plots.

Functions
---------
plot_changepoints  — series with vertical break lines at detected changepoints
plot_cusum         — CUSUM score chart
plot_segment_means — series with per-segment mean overlaid  *(innovative)*
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from tseda.changepoint.detector import ChangepointReport
from tseda.core.timeseries import TimeSeries
from tseda.visualization.base import PALETTE, _make_fig_ax, _set_title

__all__ = ["plot_changepoints", "plot_cusum", "plot_segment_means"]


def plot_changepoints(
    ts: TimeSeries,
    report: ChangepointReport,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Line plot of *ts* with vertical dashed lines at each changepoint.

    Parameters
    ----------
    ts : TimeSeries
    report : ChangepointReport
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_fig_ax(ax, figsize, (12, 4))
    ax.plot(ts.index, ts.values, color=PALETTE["accent"],
            linewidth=1.0, label=ts.name, zorder=2)
    for i, cp in enumerate(report.changepoints):
        if cp < ts.n:
            label = f"Changepoint ({report.method})" if i == 0 else None
            ax.axvline(ts.index[cp], color=PALETTE["anomaly"],
                       linewidth=1.5, linestyle="--", alpha=0.85,
                       label=label, zorder=5)
    if report.n_changepoints > 0:
        ax.legend(fontsize=9)
    ax.set_xlabel("Time")
    ax.set_ylabel(ts.unit or "Value")
    _set_title(
        ax, title,
        f"Changepoints (n={report.n_changepoints}) — {ts.name} [{report.method}]",
    )
    fig.tight_layout()
    return fig


def plot_cusum(
    ts: TimeSeries,
    report: ChangepointReport,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """CUSUM score chart from a :class:`~tseda.changepoint.detector.ChangepointReport`.

    Plots the normalised CUSUM scores stored in *report.scores* and marks
    changepoint positions with vertical lines.

    Parameters
    ----------
    ts : TimeSeries
    report : ChangepointReport
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_fig_ax(ax, figsize, (12, 3))
    n = min(len(report.scores), ts.n)
    x_idx = range(n)
    ax.plot(x_idx, report.scores[:n], color=PALETTE["neutral"],
            linewidth=0.9, label="CUSUM score")
    for cp in report.changepoints:
        if cp < n:
            ax.axvline(cp, color=PALETTE["anomaly"], linewidth=1.5,
                       linestyle="--", alpha=0.85)
    ax.axhline(0, color=PALETTE["dark"], linewidth=0.6)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Normalised score")
    ax.legend(fontsize=9)
    _set_title(ax, title, f"CUSUM scores [{report.method}]")
    fig.tight_layout()
    return fig


def plot_segment_means(
    ts: TimeSeries,
    report: ChangepointReport,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Series with a step-function overlay of per-segment means.

    Each segment (between consecutive changepoints) is annotated with its
    sample mean drawn as a horizontal line.

    Parameters
    ----------
    ts : TimeSeries
    report : ChangepointReport
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_fig_ax(ax, figsize, (12, 4))
    ax.plot(ts.index, ts.values, color=PALETTE["accent"],
            linewidth=0.8, alpha=0.7, label=ts.name, zorder=2)

    breaks = sorted(set([0] + list(report.changepoints) + [ts.n]))
    cmap = plt.cm.tab10
    for i, (start, end) in enumerate(zip(breaks[:-1], breaks[1:])):
        seg = ts.values[start:end]
        seg_mean = float(np.nanmean(seg)) if len(seg) > 0 else 0.0
        seg_idx = ts.index[start:end]
        color = cmap(i % 10)
        ax.hlines(seg_mean, seg_idx[0], seg_idx[-1],
                  colors=color, linewidth=2.5, zorder=4)
        ax.axvline(seg_idx[0], color=PALETTE["anomaly"],
                   linewidth=1.0, linestyle="--", alpha=0.5)

    ax.set_xlabel("Time")
    ax.set_ylabel(ts.unit or "Value")
    ax.legend(fontsize=9)
    _set_title(
        ax, title,
        f"Segment means (n={report.n_changepoints} breaks) — {ts.name}",
    )
    fig.tight_layout()
    return fig