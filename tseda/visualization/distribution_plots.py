"""
Distribution and rolling-statistics plots.

Functions
---------
plot_distribution — histogram + KDE + normal overlay
plot_qq           — quantile-quantile plot vs normal
plot_rolling_stats — rolling mean / std panel
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy import stats as sp_stats

from tseda.core.timeseries import TimeSeries
from tseda.visualization.base import PALETTE, _make_fig_ax, _set_title

__all__ = ["plot_distribution", "plot_qq", "plot_rolling_stats"]


def plot_distribution(
    ts: TimeSeries,
    *,
    bins: int = 30,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Histogram + KDE + fitted-normal overlay.

    Parameters
    ----------
    ts : TimeSeries
    bins : int
        Number of histogram bins.
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.stats import gaussian_kde

    fig, ax = _make_fig_ax(ax, figsize, (8, 5))
    x = ts.values[~np.isnan(ts.values)]
    if len(x) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        _set_title(ax, title, ts.name)
        return fig

    ax.hist(x, bins=bins, density=True, color=PALETTE["accent"],
            alpha=0.5, edgecolor="white", label="histogram")

    x_grid = np.linspace(x.min(), x.max(), 300)

    if len(x) >= 3:
        kde = gaussian_kde(x)
        ax.plot(x_grid, kde(x_grid), color=PALETTE["dark"],
                linewidth=2, label="KDE")

    mu, sigma = float(np.mean(x)), float(np.std(x))
    if sigma > 0:
        ax.plot(x_grid, sp_stats.norm.pdf(x_grid, mu, sigma),
                color=PALETTE["anomaly"], linewidth=1.5,
                linestyle="--", label=f"Normal(μ={mu:.2f}, σ={sigma:.2f})")

    ax.set_xlabel(ts.unit or "Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    _set_title(ax, title, f"Distribution — {ts.name}")
    fig.tight_layout()
    return fig


def plot_qq(
    ts: TimeSeries,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Quantile-quantile plot of *ts* values against a standard normal.

    Parameters
    ----------
    ts : TimeSeries
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_fig_ax(ax, figsize, (6, 6))
    x = ts.values[~np.isnan(ts.values)]

    if len(x) < 4:
        ax.text(0.5, 0.5, "Too few observations", transform=ax.transAxes,
                ha="center")
        _set_title(ax, title, ts.name)
        return fig

    (osm, osr), (slope, intercept, _) = sp_stats.probplot(x, dist="norm")
    ax.scatter(osm, osr, s=12, color=PALETTE["accent"], alpha=0.7, label="data")
    lo, hi = float(osm[0]), float(osm[-1])
    ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
            color=PALETTE["anomaly"], linewidth=1.5, label="Normal reference")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.legend(fontsize=9)
    _set_title(ax, title, f"Q-Q plot — {ts.name}")
    fig.tight_layout()
    return fig


def plot_rolling_stats(
    ts: TimeSeries,
    window: int,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Two-panel plot: rolling mean (top) and rolling std (bottom).

    Parameters
    ----------
    ts : TimeSeries
    window : int
        Rolling window width in observations.
    ax, title, figsize : optional
        *ax* is ignored for this multi-panel plot; a new figure is always
        created unless passed explicitly.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import pandas as pd
    window = max(2, int(window))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize or (12, 6), sharex=True)
    s = pd.Series(ts.values, index=ts.index)
    roll = s.rolling(window, center=True)

    ax1.plot(ts.index, ts.values, color=PALETTE["neutral"],
             linewidth=0.8, alpha=0.7, label="original")
    ax1.plot(ts.index, roll.mean().values, color=PALETTE["accent"],
             linewidth=1.5, label=f"Rolling mean ({window})")
    ax1.set_ylabel(ts.unit or "Value")
    ax1.legend(fontsize=9)

    ax2.plot(ts.index, roll.std().values, color=PALETTE["anomaly"],
             linewidth=1.5, label=f"Rolling std ({window})")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Std")
    ax2.legend(fontsize=9)

    fig.suptitle(title or f"Rolling statistics (window={window}) — {ts.name}")
    fig.tight_layout()
    return fig