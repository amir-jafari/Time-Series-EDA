"""
Shared style, palette, and helpers for tseda visualizations.

Classes / constants
-------------------
PALETTE : dict
    Brand colour map.
set_style()
    Apply tseda rcParams globally.
_make_fig_ax(ax, figsize, default_figsize)
    Internal helper — returns (fig, ax), creating a new figure when ax is None.
"""
from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes

__all__ = ["PALETTE", "set_style"]

PALETTE: dict = {
    "dark":     "#2c3e50",
    "accent":   "#2980b9",
    "anomaly":  "#e74c3c",
    "seasonal": "#27ae60",
    "trend":    "#8e44ad",
    "neutral":  "#7f8c8d",
    "light":    "#ecf0f1",
    "warn":     "#e67e22",
}


def set_style() -> None:
    """Apply the tseda matplotlib style globally.

    Sets font size, grid style, and removes top/right spines.
    Safe to call multiple times.
    """
    mpl.rcParams.update({
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "figure.dpi":         100,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
    })


def _make_fig_ax(
    ax: Optional[Axes],
    figsize: Optional[Tuple[float, float]],
    default_figsize: Tuple[float, float],
) -> Tuple[Figure, Axes]:
    """Return (fig, ax).  Creates a new figure when *ax* is ``None``."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or default_figsize)
    else:
        fig = ax.get_figure()
        if figsize is not None and fig is not None:
            fig.set_size_inches(figsize)
    return fig, ax  # type: ignore[return-value]


def _set_title(ax: Axes, title: Optional[str], default: str) -> None:
    ax.set_title(title if title is not None else default)
