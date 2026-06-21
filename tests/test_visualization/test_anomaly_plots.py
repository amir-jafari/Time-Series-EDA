"""Tests for tseda.visualization.anomaly_plots."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tseda import TimeSeries
from tseda.anomaly.detector import AnomalyDetector
from tseda.visualization.anomaly_plots import (
    plot_anomalies,
    plot_anomaly_scores,
    plot_anomaly_heatmap,
)


def _ts_with_anomalies(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    vals = rng.standard_normal(n)
    vals[50] = 10.0
    vals[150] = -10.0
    return TimeSeries(vals, index=idx)


def _anomaly_report(ts, method="iqr"):
    det = AnomalyDetector()
    if method == "iqr":
        return det.rolling_iqr(ts)
    return det.rolling_z(ts)


class TestPlotAnomalies:
    def test_returns_figure(self):
        ts = _ts_with_anomalies()
        r = _anomaly_report(ts)
        fig = plot_anomalies(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_no_anomalies(self):
        idx = pd.date_range("2020", periods=100, freq="D")
        ts = TimeSeries(np.ones(100), index=idx)
        r = _anomaly_report(ts)
        fig = plot_anomalies(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        ts = _ts_with_anomalies()
        r = _anomaly_report(ts)
        fig = plot_anomalies(ts, r, title="Anomaly plot")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        ts = _ts_with_anomalies()
        r = _anomaly_report(ts)
        fig = plot_anomalies(ts, r, figsize=(8, 3))
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotAnomalyScores:
    def test_returns_figure(self):
        ts = _ts_with_anomalies()
        r = _anomaly_report(ts)
        fig = plot_anomaly_scores(r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        ts = _ts_with_anomalies()
        r = _anomaly_report(ts)
        fig = plot_anomaly_scores(r, figsize=(10, 2))
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotAnomalyHeatmap:
    def test_returns_figure(self):
        ts = _ts_with_anomalies()
        r1 = _anomaly_report(ts, "iqr")
        r2 = _anomaly_report(ts, "z")
        fig = plot_anomaly_heatmap(ts, [r1, r2])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_empty_reports(self):
        ts = _ts_with_anomalies()
        fig = plot_anomaly_heatmap(ts, [])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_single_report(self):
        ts = _ts_with_anomalies()
        r = _anomaly_report(ts)
        fig = plot_anomaly_heatmap(ts, [r])
        assert isinstance(fig, Figure)
        plt.close(fig)