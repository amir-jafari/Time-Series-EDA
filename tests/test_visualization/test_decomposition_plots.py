"""Tests for tseda.visualization.decomposition_plots."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tseda import TimeSeries
from tseda.decomposition.classical import ClassicalDecomposer
from tseda.visualization.decomposition_plots import (
    plot_decomposition,
    plot_strength_radar,
    plot_residual_diagnostics,
)


def _seasonal_ts(n=120, period=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01", periods=n, freq="MS")
    t = np.arange(n)
    seas = np.tile(np.sin(2 * np.pi * np.arange(period) / period) * 5, n // period + 1)[:n]
    return TimeSeries(100 + 0.2 * t + seas + rng.standard_normal(n), index=idx)


def _decomp_result(n=120, period=12):
    ts = _seasonal_ts(n=n, period=period)
    return ClassicalDecomposer().decompose(ts, period=period)


class TestPlotDecomposition:
    def test_returns_figure(self):
        result = _decomp_result()
        fig = plot_decomposition(result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        result = _decomp_result()
        fig = plot_decomposition(result, figsize=(10, 8))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        result = _decomp_result()
        fig = plot_decomposition(result, title="Decomp check")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotStrengthRadar:
    def test_returns_figure(self):
        result = _decomp_result()
        fig = plot_strength_radar(result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_existing_polar_ax(self):
        result = _decomp_result()
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111, polar=True)
        fig = plot_strength_radar(result, ax=ax0)
        assert isinstance(fig, Figure)
        plt.close(fig0)

    def test_custom_title(self):
        result = _decomp_result()
        fig = plot_strength_radar(result, title="Radar")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotResidualDiagnostics:
    def test_returns_figure(self):
        result = _decomp_result()
        fig = plot_residual_diagnostics(result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_short_residuals(self):
        result = _decomp_result(n=30, period=6)
        fig = plot_residual_diagnostics(result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        result = _decomp_result()
        fig = plot_residual_diagnostics(result, figsize=(10, 3))
        assert isinstance(fig, Figure)
        plt.close(fig)