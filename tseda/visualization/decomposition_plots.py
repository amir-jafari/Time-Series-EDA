"""
Decomposition visualisation plots.

Functions
---------
plot_decomposition        — 4-panel observed/trend/seasonal/residual
plot_strength_radar       — radar chart of decomposition quality metrics
plot_residual_diagnostics — residual distribution + ACF
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from tseda.decomposition.classical import DecompositionResult
from tseda.visualization.base import PALETTE, _make_fig_ax, _set_title

__all__ = ["plot_decomposition", "plot_strength_radar", "plot_residual_diagnostics"]


def plot_decomposition(
    result: DecompositionResult,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Figure:
    """Four-panel decomposition plot (observed / trend / seasonal / residual).

    Parameters
    ----------
    result : DecompositionResult
    figsize, title : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize or (12, 10), sharex=True)
    idx = result.original.index
    panels = [
        (result.original.values, "Observed", PALETTE["dark"]),
        (result.trend.values,    "Trend",    PALETTE["trend"]),
        (result.seasonal.values, "Seasonal", PALETTE["seasonal"]),
        (result.residual.values, "Residual", PALETTE["neutral"]),
    ]
    for ax, (vals, label, color) in zip(axes, panels):
        ax.plot(idx, vals, color=color, linewidth=1.0)
        ax.set_ylabel(label, fontsize=9)
        if label == "Residual":
            ax.axhline(0, color=PALETTE["anomaly"], linewidth=0.8,
                       linestyle="--", alpha=0.6)

    axes[-1].set_xlabel("Time")
    fig.suptitle(
        title or f"Decomposition ({result.method}, {result.model})",
    )
    fig.tight_layout()
    return fig


def plot_strength_radar(
    result: DecompositionResult,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Radar chart of four decomposition strength metrics.

    Metrics (all in [0, 1]):

    * Trend strength
    * Seasonal strength
    * Signal (1 − residual variance / total variance)
    * Smoothness (trend variance / original variance)

    Parameters
    ----------
    result : DecompositionResult
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    res_vals = result.residual.values
    orig_vals = result.original.values
    trend_vals = result.trend.values

    var_orig = float(np.nanvar(orig_vals))
    var_res = float(np.nanvar(res_vals))
    var_trend = float(np.nanvar(trend_vals))

    signal = float(np.clip(1.0 - var_res / max(var_orig, 1e-15), 0.0, 1.0))
    smoothness = float(np.clip(var_trend / max(var_orig, 1e-15), 0.0, 1.0))

    labels = ["Trend\nstrength", "Seasonal\nstrength", "Signal", "Smoothness"]
    values = [
        float(np.clip(result.strength_trend,    0.0, 1.0)),
        float(np.clip(result.strength_seasonal, 0.0, 1.0)),
        signal,
        smoothness,
    ]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    if ax is None:
        fig = plt.figure(figsize=figsize or (6, 6))
        ax = fig.add_subplot(111, polar=True)
    else:
        fig = ax.get_figure()

    ax.plot(angles_closed, values_closed, color=PALETTE["accent"],
            linewidth=2, marker="o", markersize=6)
    ax.fill(angles_closed, values_closed, color=PALETTE["accent"], alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.set_title(title or "Decomposition strength radar", pad=20)
    fig.tight_layout()
    return fig


def plot_residual_diagnostics(
    result: DecompositionResult,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Figure:
    """Two-panel residual diagnostics: histogram+KDE (left) and ACF (right).

    Parameters
    ----------
    result : DecompositionResult
    figsize, title : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.stats import gaussian_kde
    from tseda.core.timeseries import TimeSeries
    from tseda.statistics.autocorrelation import AutocorrelationAnalyzer

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (12, 4))

    res = result.residual.values[~np.isnan(result.residual.values)]
    n = len(res)

    # Panel 1: histogram + KDE
    if n > 0:
        ax1.hist(res, bins=min(30, n // 3 + 1), density=True,
                 color=PALETTE["neutral"], alpha=0.5, edgecolor="white")
        if n >= 3:
            x_grid = np.linspace(res.min(), res.max(), 200)
            kde = gaussian_kde(res)
            ax1.plot(x_grid, kde(x_grid), color=PALETTE["dark"], linewidth=2)
    ax1.axvline(0, color=PALETTE["anomaly"], linewidth=1.0, linestyle="--")
    ax1.set_xlabel("Residual value")
    ax1.set_ylabel("Density")
    ax1.set_title("Residual distribution")

    # Panel 2: ACF
    if n >= 10:
        ts_res = TimeSeries(
            result.residual.values,
            index=result.residual.index,
        )
        acf_r = AutocorrelationAnalyzer().analyze(ts_res, lags=min(40, n // 2 - 1))
        lags = acf_r.lags
        ci = float(acf_r.conf_upper[1])
        ax2.axhline(0, color=PALETTE["neutral"], linewidth=0.8)
        ax2.axhline(ci, color=PALETTE["accent"], linewidth=1.0,
                    linestyle="--", alpha=0.8)
        ax2.axhline(-ci, color=PALETTE["accent"], linewidth=1.0,
                    linestyle="--", alpha=0.8)
        colors = [
            PALETTE["anomaly"] if abs(v) > ci else PALETTE["dark"]
            for v in acf_r.acf
        ]
        for k, (v, c) in enumerate(zip(acf_r.acf, colors)):
            ax2.vlines(k, 0, v, colors=c, linewidth=1.2)
            ax2.plot(k, v, "o", color=c, markersize=3)
    else:
        ax2.text(0.5, 0.5, "Too few residuals for ACF",
                 transform=ax2.transAxes, ha="center")
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("ACF")
    ax2.set_title("Residual ACF")

    fig.suptitle(title or f"Residual diagnostics ({result.method})")
    fig.tight_layout()
    return fig