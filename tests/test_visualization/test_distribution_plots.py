"""Tests for tseda.visualization.distribution_plots."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tseda import TimeSeries
from tseda.visualization.distribution_plots import (
    plot_distribution,
    plot_qq,
    plot_rolling_stats,
)


def _ts(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return TimeSeries(rng.standard_normal(n), index=idx)


class TestPlotDistribution:
    def test_returns_figure(self):
        ts = _ts()
        fig = plot_distribution(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_bins(self):
        ts = _ts()
        fig = plot_distribution(ts, bins=15)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_all_nan(self):
        idx = pd.date_range("2020", periods=10, freq="D")
        ts = TimeSeries(np.full(10, np.nan), index=idx)
        fig = plot_distribution(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        ts = _ts()
        fig = plot_distribution(ts, figsize=(6, 4))
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotQQ:
    def test_returns_figure(self):
        ts = _ts()
        fig = plot_qq(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_too_few_obs(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        ts = TimeSeries([1.0, 2.0, 3.0], index=idx)
        fig = plot_qq(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        ts = _ts()
        fig = plot_qq(ts, title="QQ check")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotRollingStats:
    def test_returns_figure(self):
        ts = _ts()
        fig = plot_rolling_stats(ts, window=10)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_window_1_handled(self):
        ts = _ts()
        fig = plot_rolling_stats(ts, window=1)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        ts = _ts()
        fig = plot_rolling_stats(ts, window=7, figsize=(10, 5))
        assert isinstance(fig, Figure)
        plt.close(fig)