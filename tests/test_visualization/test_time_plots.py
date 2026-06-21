"""Tests for tseda.visualization.time_plots."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tseda import TimeSeries
from tseda.visualization.time_plots import (
    plot_series,
    plot_seasonal_subseries,
    plot_lag,
    plot_calendar_heatmap,
    plot_annual_boxplots,
    plot_density_ridge,
)


def _daily(n=365, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return TimeSeries(rng.standard_normal(n), index=idx, name="test")


def _multi_year(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=730, freq="D")
    return TimeSeries(rng.standard_normal(730), index=idx, name="multi")


def _monthly(n=36, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01", periods=n, freq="MS")
    return TimeSeries(rng.standard_normal(n), index=idx, name="monthly")


class TestPlotSeries:
    def test_returns_figure(self):
        ts = _daily()
        fig = plot_series(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_rolling_window(self):
        ts = _daily()
        fig = plot_series(ts, rolling_window=7)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        ts = _daily()
        fig = plot_series(ts, title="My Title")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        ts = _daily()
        fig = plot_series(ts, figsize=(8, 3))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_existing_ax(self):
        ts = _daily()
        fig0, ax0 = plt.subplots()
        fig = plot_series(ts, ax=ax0)
        assert isinstance(fig, Figure)
        plt.close(fig0)

    def test_rolling_window_1_no_overlay(self):
        ts = _daily()
        fig = plot_series(ts, rolling_window=1)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotSeasonalSubseries:
    def test_returns_figure(self):
        ts = _daily()
        fig = plot_seasonal_subseries(ts, period=7)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_period_2(self):
        ts = _daily(n=50)
        fig = plot_seasonal_subseries(ts, period=2)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_large_period(self):
        ts = _daily()
        fig = plot_seasonal_subseries(ts, period=12)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotLag:
    def test_returns_figure(self):
        ts = _daily()
        fig = plot_lag(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_lags(self):
        ts = _daily()
        fig = plot_lag(ts, lags=(1, 3, 7, 14))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_lag_larger_than_n(self):
        ts = _daily(n=20)
        fig = plot_lag(ts, lags=(1, 50))
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotCalendarHeatmap:
    def test_daily_returns_figure(self):
        ts = _daily()
        fig = plot_calendar_heatmap(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_monthly_returns_figure(self):
        ts = _monthly()
        fig = plot_calendar_heatmap(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        ts = _daily(n=90)
        fig = plot_calendar_heatmap(ts, title="Heat")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotAnnualBoxplots:
    def test_returns_figure(self):
        ts = _daily()
        fig = plot_annual_boxplots(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_monthly_data(self):
        ts = _monthly()
        fig = plot_annual_boxplots(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_short_series(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        ts = TimeSeries(np.arange(10, dtype=float), index=idx)
        fig = plot_annual_boxplots(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotDensityRidge:
    def test_multi_year_returns_figure(self):
        ts = _multi_year()
        fig = plot_density_ridge(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_single_year_fallback(self):
        ts = _daily(n=100)
        fig = plot_density_ridge(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        ts = _multi_year()
        fig = plot_density_ridge(ts, figsize=(8, 5))
        assert isinstance(fig, Figure)
        plt.close(fig)