"""
Seasonality visualisation plots.

Functions
---------
plot_periodogram    — FFT power spectrum with marked dominant peaks
plot_polar_seasonal — values on a clock-face polar chart  *(innovative)*
plot_season_heatmap — period-phase × time heatmap  *(innovative)*
plot_monthly_boxplots — box per calendar month
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from tseda.core.timeseries import TimeSeries
from tseda.seasonality.detector import SeasonalityReport
from tseda.visualization.base import PALETTE, _make_fig_ax, _set_title

__all__ = [
    "plot_periodogram",
    "plot_polar_seasonal",
    "plot_season_heatmap",
    "plot_monthly_boxplots",
]


def plot_periodogram(
    report: SeasonalityReport,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """FFT power spectrum with dominant periods marked.

    Parameters
    ----------
    report : SeasonalityReport
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_fig_ax(ax, figsize, (12, 4))

    periods = np.array(report.periodogram_periods, dtype=float)
    strengths = np.array(report.strength_scores.get("periodogram", []), dtype=float)

    if len(periods) == 0 or len(strengths) == 0 or len(periods) != len(strengths):
        ax.text(0.5, 0.5, "No periodogram data available",
                transform=ax.transAxes, ha="center")
        _set_title(ax, title, "Periodogram")
        fig.tight_layout()
        return fig

    ax.plot(periods, strengths, color=PALETTE["accent"], linewidth=1.2)
    ax.fill_between(periods, strengths, alpha=0.2, color=PALETTE["accent"])

    top_n = min(3, len(report.candidate_periods))
    for p in report.candidate_periods[:top_n]:
        p_float = float(p)
        closest = int(np.argmin(np.abs(periods - p_float)))
        ax.axvline(p_float, color=PALETTE["anomaly"], linewidth=1.5,
                   linestyle="--", alpha=0.8)
        if closest < len(strengths):
            ax.annotate(
                f"T={p}",
                xy=(p_float, strengths[closest]),
                xytext=(p_float + 0.5, strengths[closest] * 1.1),
                fontsize=8,
                color=PALETTE["anomaly"],
            )

    ax.set_xlabel("Period (observations)")
    ax.set_ylabel("Power")
    _set_title(ax, title, "Periodogram")
    fig.tight_layout()
    return fig


def plot_polar_seasonal(
    ts: TimeSeries,
    period: int,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Values projected onto a clock-face polar chart.

    Each observation is plotted at angle = (position mod period) / period × 2π
    and radius proportional to the normalised value.

    Parameters
    ----------
    ts : TimeSeries
    period : int
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    period = max(2, int(period))
    x = ts.values[~np.isnan(ts.values)]
    n = len(x)

    if ax is None:
        fig = plt.figure(figsize=figsize or (7, 7))
        ax = fig.add_subplot(111, polar=True)
    else:
        fig = ax.get_figure()

    phases = (np.arange(n) % period) / period * 2 * np.pi
    # Normalise to [0.2, 1.0] for radius (avoid the origin)
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    if x_max > x_min:
        radii = 0.2 + 0.8 * (x - x_min) / (x_max - x_min)
    else:
        radii = np.full(n, 0.6)

    sc = ax.scatter(phases, radii, c=radii, cmap="viridis",
                    s=8, alpha=0.6, zorder=3)

    # Overlay per-phase mean
    phase_means = np.zeros(period)
    for s in range(period):
        idx_s = np.arange(s, n, period)
        vals_s = x[idx_s]
        if len(vals_s) > 0:
            phase_means[s] = np.nanmean(vals_s)
    mean_norm = 0.2 + 0.8 * (phase_means - x_min) / max(x_max - x_min, 1e-15)
    phase_angles = np.linspace(0, 2 * np.pi, period, endpoint=False)
    mean_angles = np.append(phase_angles, phase_angles[0])
    mean_radii = np.append(mean_norm, mean_norm[0])
    ax.plot(mean_angles, mean_radii, color=PALETTE["anomaly"],
            linewidth=2.0, linestyle="-", label="Phase mean")

    ax.set_xticks(np.linspace(0, 2 * np.pi, period, endpoint=False))
    ax.set_xticklabels([str(i + 1) for i in range(period)], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title or f"Polar seasonal (period={period}) — {ts.name}", pad=20)
    fig.tight_layout()
    return fig


def plot_season_heatmap(
    ts: TimeSeries,
    period: int,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Period-phase × cycle heatmap.

    Rows = phase within period (0 … period−1).
    Columns = cycle index.
    Cell colour = observed value.

    Parameters
    ----------
    ts : TimeSeries
    period : int
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import pandas as pd

    period = max(2, int(period))
    x = ts.values
    n = len(x)
    n_cycles = n // period
    if n_cycles < 1:
        fig, ax = _make_fig_ax(ax, figsize, (8, 4))
        ax.text(0.5, 0.5, "Too few cycles to build heatmap",
                transform=ax.transAxes, ha="center")
        _set_title(ax, title, f"Season heatmap (period={period})")
        fig.tight_layout()
        return fig

    grid = x[:n_cycles * period].reshape(n_cycles, period).T

    fig, ax = _make_fig_ax(ax, figsize, (max(6, n_cycles * 0.5 + 2), max(4, period * 0.4 + 2)))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanpercentile(grid, 5),
                   vmax=np.nanpercentile(grid, 95))
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Phase")
    ax.set_yticks(range(period))
    ax.set_yticklabels([str(i + 1) for i in range(period)], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=ts.unit or "Value")
    _set_title(ax, title, f"Season heatmap (period={period}) — {ts.name}")
    fig.tight_layout()
    return fig


def plot_monthly_boxplots(
    ts: TimeSeries,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Box-and-whisker plot per calendar month position.

    Parameters
    ----------
    ts : TimeSeries
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import pandas as pd

    fig, ax = _make_fig_ax(ax, figsize, (12, 4))
    s = pd.Series(ts.values, index=ts.index)
    months = s.index.month
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    groups = []
    labels = []
    for m in range(1, 13):
        g = s[months == m].dropna().values
        if len(g) > 0:
            groups.append(g)
            labels.append(month_names[m - 1])

    if not groups:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
    else:
        bp = ax.boxplot(groups, patch_artist=True,
                        medianprops={"color": PALETTE["anomaly"]})
        ax.set_xticklabels(labels)
        for patch in bp["boxes"]:
            patch.set_facecolor(PALETTE["seasonal"])
            patch.set_alpha(0.5)

    ax.set_xlabel("Month")
    ax.set_ylabel(ts.unit or "Value")
    _set_title(ax, title, f"Monthly boxplots — {ts.name}")
    fig.tight_layout()
    return fig