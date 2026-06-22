"""
Time-domain plots for :class:`~tseda.core.TimeSeries`.

Functions
---------
plot_series            — line plot with optional rolling mean overlay
plot_seasonal_subseries— one sub-panel per season position
plot_lag               — scatter plot matrix of ts vs lagged ts
plot_calendar_heatmap  — value by day-of-week × calendar week  *(innovative)*
plot_annual_boxplots   — distribution per calendar month
plot_density_ridge     — year-over-year KDE ridgeline  *(innovative)*
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from tseda.core.timeseries import TimeSeries
from tseda.visualization.base import PALETTE, _make_fig_ax, _set_title

__all__ = [
    "plot_series",
    "plot_seasonal_subseries",
    "plot_lag",
    "plot_calendar_heatmap",
    "plot_annual_boxplots",
    "plot_density_ridge",
]


def plot_series(
    ts: TimeSeries,
    *,
    rolling_window: Optional[int] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Line plot of *ts* with an optional rolling-mean overlay.

    Parameters
    ----------
    ts : TimeSeries
    rolling_window : int, optional
        If given, overlay a rolling mean of this width.
    ax, title, figsize : optional
        Standard plot arguments.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_fig_ax(ax, figsize, (12, 4))
    ax.plot(ts.index, ts.values, color=PALETTE["accent"], linewidth=1.0,
            label=ts.name)
    if rolling_window is not None and rolling_window > 1:
        rolled = (
            pd.Series(ts.values, index=ts.index)
            .rolling(rolling_window, center=True)
            .mean()
        )
        ax.plot(ts.index, rolled.values, color=PALETTE["anomaly"],
                linewidth=1.5, linestyle="--", label=f"MA({rolling_window})")
        ax.legend(fontsize=9)
    ax.set_xlabel("Time")
    ax.set_ylabel(ts.unit or "Value")
    _set_title(ax, title, ts.name)
    fig.tight_layout()
    return fig


def plot_seasonal_subseries(
    ts: TimeSeries,
    period: int,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """One sub-panel per season position, each showing all cycles.

    Parameters
    ----------
    ts : TimeSeries
    period : int
        Seasonal period (number of sub-panels).
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    period = max(2, int(period))
    ncols = min(period, 12)
    nrows = (period + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize or (max(12, 2 * period), 3 * nrows),
        sharey=True,
    )
    axes_flat = np.array(axes).flatten()
    x = ts.values
    n = len(x)

    for s in range(period):
        axi = axes_flat[s]
        positions = np.arange(s, n, period)
        vals = x[positions]
        axi.plot(range(len(vals)), vals, color=PALETTE["accent"],
                 marker="o", markersize=3, linewidth=0.8)
        mean_val = np.nanmean(vals)
        axi.axhline(mean_val, color=PALETTE["anomaly"], linewidth=1.0,
                    linestyle="--", alpha=0.7)
        axi.set_title(f"Season {s + 1}", fontsize=9)
        axi.tick_params(labelsize=8)

    for i in range(period, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(title or f"Seasonal subseries (period={period})")
    fig.tight_layout()
    return fig


def plot_lag(
    ts: TimeSeries,
    lags: Sequence[int] = (1, 2, 7),
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Figure:
    """Scatter-plot matrix of *ts* versus lagged copies.

    Parameters
    ----------
    ts : TimeSeries
    lags : sequence of int
        Lag values to plot.
    figsize, title : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    lags = [int(k) for k in lags if int(k) > 0]
    n_lags = len(lags)
    fig, axes = plt.subplots(
        1, n_lags, figsize=figsize or (4 * n_lags, 4), squeeze=False
    )
    x = ts.values
    n = len(x)

    for j, k in enumerate(lags):
        ax = axes[0, j]
        if k >= n:
            ax.set_visible(False)
            continue
        ax.scatter(x[:-k], x[k:], s=8, alpha=0.5, color=PALETTE["accent"])
        rng = np.nanmax(x) - np.nanmin(x)
        lo = np.nanmin(x) - 0.05 * rng
        hi = np.nanmax(x) + 0.05 * rng
        ax.plot([lo, hi], [lo, hi], color=PALETTE["neutral"],
                linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"t")
        ax.set_ylabel(f"t+{k}")
        ax.set_title(f"Lag {k}")

    fig.suptitle(title or f"Lag plots — {ts.name}")
    fig.tight_layout()
    return fig


def plot_calendar_heatmap(
    ts: TimeSeries,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Heatmap of values by day-of-week (columns) × calendar week (rows).

    Works best with daily or sub-daily data.  For non-daily series a
    weekly-aggregated heatmap (ISO week × year) is produced instead.

    Parameters
    ----------
    ts : TimeSeries
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.colors as mcolors

    idx = ts.index
    vals = ts.values

    # Determine if daily resolution
    freq = ts.freq
    is_daily = freq in ("D", "B") or (freq is None and len(idx) > 1 and
        np.median(np.diff(idx.astype(np.int64)) / 1e9) <= 86400)

    if is_daily:
        # Pivot: rows=weeks (ISO week), cols=day-of-week
        s = pd.Series(vals, index=idx)
        df = pd.DataFrame({
            "val": s.values,
            "week": idx.isocalendar().week.values,
            "year": idx.year,
            "dow":  idx.dayofweek,
        })
        df["week_key"] = df["year"] * 100 + df["week"]
        pivot = df.pivot_table(index="week_key", columns="dow", values="val", aggfunc="mean")
        xlabels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        xlabels = [xlabels[c] for c in pivot.columns]
        ylabel = "ISO week"
    else:
        # Monthly aggregation: rows=year, cols=month
        s = pd.Series(vals, index=idx)
        s = s.resample("MS").mean()
        df = pd.DataFrame({"val": s.values, "year": s.index.year, "month": s.index.month})
        pivot = df.pivot_table(index="year", columns="month", values="val", aggfunc="mean")
        xlabels = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
        xlabels = [xlabels[c - 1] for c in pivot.columns]
        ylabel = "Year"

    fig, ax = _make_fig_ax(ax, figsize, (12, max(4, len(pivot) * 0.25 + 2)))
    data_arr = pivot.values.astype(float)
    im = ax.imshow(data_arr, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanpercentile(data_arr, 5),
                   vmax=np.nanpercentile(data_arr, 95))
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel("Day of week" if is_daily else "Month")
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=ts.unit or "Value")
    _set_title(ax, title, f"Calendar heatmap — {ts.name}")
    fig.tight_layout()
    return fig


def plot_annual_boxplots(
    ts: TimeSeries,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Box-and-whisker plot of values grouped by calendar month.

    Parameters
    ----------
    ts : TimeSeries
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_fig_ax(ax, figsize, (12, 4))
    s = pd.Series(ts.values, index=ts.index)
    months = s.index.month
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    groups = [s[months == m].dropna().values for m in range(1, 13)]
    groups_nonempty = [(month_names[i], g) for i, g in enumerate(groups) if len(g) > 0]

    if not groups_nonempty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
    else:
        labels, data = zip(*groups_nonempty)
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops={"color": PALETTE["anomaly"]})
        ax.set_xticklabels(labels)
        for patch in bp["boxes"]:
            patch.set_facecolor(PALETTE["accent"])
            patch.set_alpha(0.6)

    ax.set_xlabel("Month")
    ax.set_ylabel(ts.unit or "Value")
    _set_title(ax, title, f"Monthly distribution — {ts.name}")
    fig.tight_layout()
    return fig


def plot_density_ridge(
    ts: TimeSeries,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Year-over-year density ridgeline (KDE per year).

    Each year is a separate KDE curve, offset vertically.

    Parameters
    ----------
    ts : TimeSeries
    ax, title, figsize : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.stats import gaussian_kde

    s = pd.Series(ts.values, index=ts.index)
    years = sorted(s.index.year.unique())

    if len(years) < 2:
        # Fallback: single KDE
        fig, ax = _make_fig_ax(ax, figsize, (10, 4))
        x_clean = s.dropna().values
        if len(x_clean) >= 2:
            x_grid = np.linspace(x_clean.min(), x_clean.max(), 200)
            kde = gaussian_kde(x_clean)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.5, color=PALETTE["accent"])
            ax.plot(x_grid, kde(x_grid), color=PALETTE["dark"], linewidth=1)
        _set_title(ax, title, f"Density — {ts.name}")
        fig.tight_layout()
        return fig

    fig, ax = _make_fig_ax(ax, figsize, (10, max(4, len(years) * 1.2)))
    cmap = plt.cm.viridis
    all_vals = s.dropna().values
    x_min, x_max = float(all_vals.min()), float(all_vals.max())
    x_grid = np.linspace(x_min, x_max, 300)
    scale = (x_max - x_min) * 0.15

    for i, yr in enumerate(years):
        yr_vals = s[s.index.year == yr].dropna().values
        if len(yr_vals) < 3:
            continue
        kde = gaussian_kde(yr_vals)
        density = kde(x_grid)
        density = density / density.max() * scale
        color = cmap(i / max(len(years) - 1, 1))
        ax.fill_between(x_grid, i, i + density, alpha=0.6, color=color)
        ax.plot(x_grid, i + density, color=color, linewidth=0.8)

    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], fontsize=9)
    ax.set_xlabel(ts.unit or "Value")
    ax.set_ylabel("Year")
    _set_title(ax, title, f"Density ridgeline — {ts.name}")
    fig.tight_layout()
    return fig