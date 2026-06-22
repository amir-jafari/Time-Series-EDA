"""
Autocorrelation and partial-autocorrelation plots.

Functions
---------
plot_acf_pacf   — side-by-side stem plots with confidence bands
plot_acf_heatmap— rolling-window ACF heatmap (lag × time)  *(innovative)*
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from tseda.core.timeseries import TimeSeries
from tseda.statistics.autocorrelation import AutocorrelationResult
from tseda.visualization.base import PALETTE, _make_fig_ax, _set_title

__all__ = ["plot_acf_pacf", "plot_acf_heatmap"]


def plot_acf_pacf(
    result: AutocorrelationResult,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Figure:
    """Side-by-side ACF and PACF stem plots with ±95 % confidence bands.

    Parameters
    ----------
    result : AutocorrelationResult
        Output of :meth:`~tseda.statistics.autocorrelation.AutocorrelationAnalyzer.analyze`.
    figsize, title : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (14, 4))
    lags = result.lags
    ci = float(result.conf_upper[1])

    for ax, vals, label in [
        (ax1, result.acf, "ACF"),
        (ax2, result.pacf, "PACF"),
    ]:
        ax.axhline(0, color=PALETTE["neutral"], linewidth=0.8)
        ax.axhline(ci, color=PALETTE["accent"], linewidth=1.0,
                   linestyle="--", alpha=0.8, label="±95% CI")
        ax.axhline(-ci, color=PALETTE["accent"], linewidth=1.0,
                   linestyle="--", alpha=0.8)
        ax.fill_between(lags, -ci, ci, alpha=0.08, color=PALETTE["accent"])

        colors = [
            PALETTE["anomaly"] if abs(v) > ci else PALETTE["dark"]
            for v in vals
        ]
        for k, (v, c) in enumerate(zip(vals, colors)):
            ax.vlines(k, 0, v, colors=c, linewidth=1.2)
            ax.plot(k, v, "o", color=c, markersize=4)

        ax.set_xlabel("Lag")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xlim(-0.5, lags[-1] + 0.5)

    ax1.legend(fontsize=9)
    fig.suptitle(title or "ACF / PACF")
    fig.tight_layout()
    return fig


def plot_acf_heatmap(
    ts: TimeSeries,
    max_lag: int = 40,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Rolling-window ACF heatmap — lag on y-axis, time window on x-axis.

    For each overlapping window the ACF is computed and displayed as a
    colour column.  High ACF magnitude at a particular lag is visible as a persistent
    horizontal band.

    Parameters
    ----------
    ts : TimeSeries
    max_lag : int
        Maximum lag to show.
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = ts.values
    n = len(x)
    window_size = max(max_lag * 2 + 4, n // 5)
    window_size = min(window_size, n)

    if window_size < max_lag + 4 or n < window_size:
        fig, ax = _make_fig_ax(ax, figsize, (10, 5))
        ax.text(0.5, 0.5, "Series too short for rolling ACF",
                transform=ax.transAxes, ha="center")
        _set_title(ax, title, f"ACF heatmap — {ts.name}")
        fig.tight_layout()
        return fig

    step = max(1, (n - window_size) // 50)
    starts = list(range(0, n - window_size + 1, step))
    n_windows = len(starts)
    lag_cap = min(max_lag, window_size // 2 - 1)

    heatmap = np.full((lag_cap + 1, n_windows), np.nan)
    window_centers = []

    for j, s in enumerate(starts):
        w = x[s: s + window_size]
        valid = w[~np.isnan(w)]
        if len(valid) < lag_cap + 4:
            continue
        xc = valid - np.mean(valid)
        denom = float(np.dot(xc, xc))
        if denom == 0:
            continue
        for k in range(lag_cap + 1):
            heatmap[k, j] = float(np.dot(xc[k:], xc[:len(xc) - k])) / denom if k < len(xc) else np.nan
        center_idx = s + window_size // 2
        window_centers.append(ts.index[min(center_idx, n - 1)])

    fig, ax = _make_fig_ax(ax, figsize, (12, 5))
    im = ax.imshow(
        heatmap, aspect="auto", cmap="coolwarm",
        vmin=-1, vmax=1, origin="lower",
        extent=[0, n_windows, -0.5, lag_cap + 0.5],
    )
    ax.set_xlabel("Time window")
    ax.set_ylabel("Lag")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="ACF")
    _set_title(ax, title, f"Rolling ACF heatmap — {ts.name}")
    fig.tight_layout()
    return fig