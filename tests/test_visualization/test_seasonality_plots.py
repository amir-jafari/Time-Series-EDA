"""Tests for tseda.visualization.seasonality_plots."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tseda import TimeSeries
from tseda.seasonality.detector import SeasonalityDetector
from tseda.visualization.seasonality_plots import (
    plot_periodogram,
    plot_polar_seasonal,
    plot_season_heatmap,
    plot_monthly_boxplots,
)


def _seasonal_ts(n=200, period=7, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n)
    seas = np.sin(2 * np.pi * t / period) * 4
    return TimeSeries(seas + rng.standard_normal(n), index=idx)


def _monthly_ts(n=48):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01", periods=n, freq="MS")
    seas = np.tile(np.sin(2 * np.pi * np.arange(12) / 12) * 5, n // 12 + 1)[:n]
    return TimeSeries(seas + rng.standard_normal(n), index=idx)


class TestPlotPeriodogram:
    def test_returns_figure(self):
        ts = _seasonal_ts()
        report = SeasonalityDetector().detect(ts)
        fig = plot_periodogram(report)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_empty_periodogram_data(self):
        ts = _seasonal_ts(n=30)
        report = SeasonalityDetector().detect(ts)
        fig = plot_periodogram(report)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        ts = _seasonal_ts()
        report = SeasonalityDetector().detect(ts)
        fig = plot_periodogram(report, figsize=(8, 3))
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotPolarSeasonal:
    def test_returns_figure(self):
        ts = _seasonal_ts()
        fig = plot_polar_seasonal(ts, period=7)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_period_12(self):
        ts = _monthly_ts()
        fig = plot_polar_seasonal(ts, period=12)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        ts = _seasonal_ts()
        fig = plot_polar_seasonal(ts, period=7, title="Polar")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotSeasonHeatmap:
    def test_returns_figure(self):
        ts = _seasonal_ts()
        fig = plot_season_heatmap(ts, period=7)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_too_few_cycles(self):
        idx = pd.date_range("2020", periods=10, freq="D")
        ts = TimeSeries(np.arange(10, dtype=float), index=idx)
        fig = plot_season_heatmap(ts, period=50)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_period_12(self):
        ts = _monthly_ts()
        fig = plot_season_heatmap(ts, period=12)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotMonthlyBoxplots:
    def test_daily_data(self):
        ts = _seasonal_ts()
        fig = plot_monthly_boxplots(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_monthly_data(self):
        ts = _monthly_ts()
        fig = plot_monthly_boxplots(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        ts = _monthly_ts()
        fig = plot_monthly_boxplots(ts, title="Boxplots")
        assert isinstance(fig, Figure)
        plt.close(fig)