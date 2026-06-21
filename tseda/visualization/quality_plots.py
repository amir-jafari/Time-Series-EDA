"""
Data-quality visualisation plots.

Functions
---------
plot_missing_heatmap — NaN position heatmap  *(innovative)*
plot_outliers        — series with fence lines and flagged points
plot_outlier_score   — outlier score timeline
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from tseda.core.timeseries import TimeSeries
from tseda.quality.outliers import OutlierReport
from tseda.visualization.base import PALETTE, _make_fig_ax, _set_title

__all__ = ["plot_missing_heatmap", "plot_outliers", "plot_outlier_score"]


def plot_missing_heatmap(
    ts: TimeSeries,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """NaN positions displayed as a binary heatmap.

    For long series the observations are binned into columns to keep the
    plot readable.  Rows represent equal-sized chunks of the series; each
    cell is red if any NaN is present in that chunk, white otherwise.

    Parameters
    ----------
    ts : TimeSeries
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = ts.n
    n_cols = min(200, n)
    chunk = max(1, n // n_cols)
    n_chunks = (n + chunk - 1) // chunk

    grid = np.zeros((1, n_chunks), dtype=float)
    for c in range(n_chunks):
        seg = ts.values[c * chunk: (c + 1) * chunk]
        grid[0, c] = float(np.any(np.isnan(seg)))

    fig, ax = _make_fig_ax(ax, figsize, (12, 1.5))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1,
                   extent=[0, n, -0.5, 0.5])
    ax.set_yticks([])
    ax.set_xlabel("Observation index")
    pct = float(np.isnan(ts.values).mean() * 100)
    _set_title(ax, title, f"Missing value map — {ts.name} ({pct:.1f}% NaN)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Has NaN")
    fig.tight_layout()
    return fig


def plot_outliers(
    ts: TimeSeries,
    report: OutlierReport,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Series line plot with IQR fence lines and outlier markers.

    Parameters
    ----------
    ts : TimeSeries
    report : OutlierReport
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_fig_ax(ax, figsize, (12, 4))
    ax.plot(ts.index, ts.values, color=PALETTE["accent"],
            linewidth=1.0, label=ts.name, zorder=2)

    if report.lower_bound is not None:
        ax.axhline(report.lower_bound, color=PALETTE["warn"],
                   linewidth=1.2, linestyle="--", alpha=0.9,
                   label=f"Lower fence ({report.lower_bound:.2f})")
    if report.upper_bound is not None:
        ax.axhline(report.upper_bound, color=PALETTE["warn"],
                   linewidth=1.2, linestyle="--", alpha=0.9,
                   label=f"Upper fence ({report.upper_bound:.2f})")

    if report.n_outliers > 0:
        ax.scatter(
            report.timestamps, report.values,
            color=PALETTE["anomaly"], s=50, zorder=5,
            label=f"Outlier ({report.method}, n={report.n_outliers})",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(ts.unit or "Value")
    ax.legend(fontsize=9)
    _set_title(ax, title, f"Outliers — {ts.name} [{report.method}]")
    fig.tight_layout()
    return fig


def plot_outlier_score(
    ts: TimeSeries,
    report: OutlierReport,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Outlier score timeline with a threshold reference line at 1.0.

    Parameters
    ----------
    ts : TimeSeries
    report : OutlierReport
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_fig_ax(ax, figsize, (12, 3))
    n = min(len(report.indices), ts.n)

    # Build a score array aligned to ts positions
    scores = np.zeros(ts.n, dtype=float)
    for idx_pos in report.indices:
        if idx_pos < ts.n:
            scores[idx_pos] = 1.0

    ax.bar(range(ts.n), scores, color=PALETTE["anomaly"],
           alpha=0.7, width=1.0, label="outlier flag")
    ax.plot(ts.index, ts.values / max(abs(ts.values[~np.isnan(ts.values)]).max(), 1e-15),
            color=PALETTE["neutral"], linewidth=0.8, alpha=0.5, label="series (normalised)")
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Outlier flag")
    ax.legend(fontsize=9)
    _set_title(ax, title, f"Outlier positions [{report.method}] — {ts.name}")
    fig.tight_layout()
    return fig