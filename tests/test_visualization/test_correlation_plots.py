"""Tests for tseda.visualization.correlation_plots."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tseda import TimeSeries
from tseda.statistics.autocorrelation import AutocorrelationAnalyzer
from tseda.visualization.correlation_plots import plot_acf_pacf, plot_acf_heatmap


def _ts(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x = np.zeros(n)
    eps = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + eps[i]
    return TimeSeries(x, index=idx)


class TestPlotAcfPacf:
    def test_returns_figure(self):
        ts = _ts()
        result = AutocorrelationAnalyzer().analyze(ts, lags=20)
        fig = plot_acf_pacf(result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        ts = _ts()
        result = AutocorrelationAnalyzer().analyze(ts, lags=20)
        fig = plot_acf_pacf(result, figsize=(10, 3))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        ts = _ts()
        result = AutocorrelationAnalyzer().analyze(ts, lags=10)
        fig = plot_acf_pacf(result, title="ACF check")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_few_lags(self):
        ts = _ts(n=30)
        result = AutocorrelationAnalyzer().analyze(ts, lags=5)
        fig = plot_acf_pacf(result)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotAcfHeatmap:
    def test_returns_figure(self):
        ts = _ts()
        fig = plot_acf_heatmap(ts, max_lag=20)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_short_series_fallback(self):
        idx = pd.date_range("2020", periods=15, freq="D")
        ts = TimeSeries(np.arange(15, dtype=float), index=idx)
        fig = plot_acf_heatmap(ts, max_lag=10)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        ts = _ts()
        fig = plot_acf_heatmap(ts, max_lag=15, figsize=(10, 4))
        assert isinstance(fig, Figure)
        plt.close(fig)