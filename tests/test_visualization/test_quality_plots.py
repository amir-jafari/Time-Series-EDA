"""Tests for tseda.visualization.quality_plots."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tseda import TimeSeries
from tseda.quality.outliers import OutlierDetector
from tseda.visualization.quality_plots import (
    plot_missing_heatmap,
    plot_outliers,
    plot_outlier_score,
)


def _clean_ts(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return TimeSeries(rng.standard_normal(n), index=idx)


def _nan_ts(n=200, nan_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    vals = rng.standard_normal(n)
    nan_idx = rng.choice(n, size=int(n * nan_frac), replace=False)
    vals[nan_idx] = np.nan
    return TimeSeries(vals, index=idx)


def _outlier_ts(n=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    vals = rng.standard_normal(n)
    vals[30] = 15.0
    vals[70] = -15.0
    return TimeSeries(vals, index=idx)


class TestPlotMissingHeatmap:
    def test_no_nans(self):
        ts = _clean_ts()
        fig = plot_missing_heatmap(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_nans(self):
        ts = _nan_ts()
        fig = plot_missing_heatmap(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        ts = _nan_ts()
        fig = plot_missing_heatmap(ts, title="Missing map")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_short_series(self):
        idx = pd.date_range("2020", periods=5, freq="D")
        vals = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        ts = TimeSeries(vals, index=idx)
        fig = plot_missing_heatmap(ts)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        ts = _nan_ts()
        fig = plot_missing_heatmap(ts, figsize=(10, 2))
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotOutliers:
    def test_with_outliers(self):
        ts = _outlier_ts()
        r = OutlierDetector().iqr(ts)
        fig = plot_outliers(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_no_outliers(self):
        ts = _clean_ts()
        r = OutlierDetector().iqr(ts)
        fig = plot_outliers(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        ts = _outlier_ts()
        r = OutlierDetector().iqr(ts)
        fig = plot_outliers(ts, r, title="Outlier plot")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_zscore_method(self):
        ts = _outlier_ts()
        r = OutlierDetector().zscore(ts)
        fig = plot_outliers(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotOutlierScore:
    def test_with_outliers(self):
        ts = _outlier_ts()
        r = OutlierDetector().iqr(ts)
        fig = plot_outlier_score(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_no_outliers(self):
        ts = _clean_ts()
        r = OutlierDetector().iqr(ts)
        fig = plot_outlier_score(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        ts = _outlier_ts()
        r = OutlierDetector().iqr(ts)
        fig = plot_outlier_score(ts, r, figsize=(10, 2))
        assert isinstance(fig, Figure)
        plt.close(fig)