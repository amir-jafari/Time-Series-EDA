"""Tests for tseda.visualization.changepoint_plots."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tseda import TimeSeries
from tseda.changepoint.detector import ChangepointDetector
from tseda.visualization.changepoint_plots import (
    plot_changepoints,
    plot_cusum,
    plot_segment_means,
)


def _shift_ts(n=300, break_at=150, shift=5.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    vals = np.concatenate([
        rng.standard_normal(break_at),
        rng.standard_normal(n - break_at) + shift,
    ])
    return TimeSeries(vals, index=idx)


def _noshift_ts(n=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020", periods=n, freq="D")
    return TimeSeries(rng.standard_normal(n), index=idx)


class TestPlotChangepoints:
    def test_returns_figure_with_changepoints(self):
        ts = _shift_ts()
        r = ChangepointDetector().binary_segmentation(ts)
        fig = plot_changepoints(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_no_changepoints(self):
        ts = _noshift_ts()
        r = ChangepointDetector().binary_segmentation(ts)
        fig = plot_changepoints(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        ts = _shift_ts()
        r = ChangepointDetector().binary_segmentation(ts)
        fig = plot_changepoints(ts, r, title="CP plot")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotCusum:
    def test_returns_figure(self):
        ts = _shift_ts()
        r = ChangepointDetector().cusum(ts)
        fig = plot_cusum(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_no_changepoints(self):
        ts = _noshift_ts()
        r = ChangepointDetector().cusum(ts)
        fig = plot_cusum(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        ts = _shift_ts()
        r = ChangepointDetector().cusum(ts)
        fig = plot_cusum(ts, r, figsize=(8, 2))
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotSegmentMeans:
    def test_returns_figure_with_breaks(self):
        ts = _shift_ts()
        r = ChangepointDetector().binary_segmentation(ts)
        fig = plot_segment_means(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_no_breaks(self):
        ts = _noshift_ts()
        r = ChangepointDetector().binary_segmentation(ts)
        fig = plot_segment_means(ts, r)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        ts = _shift_ts()
        r = ChangepointDetector().binary_segmentation(ts)
        fig = plot_segment_means(ts, r, title="Segments")
        assert isinstance(fig, Figure)
        plt.close(fig)