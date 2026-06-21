"""
Tests for :mod:`tseda.quality.outliers`.

Coverage targets
----------------
* OutlierDetector.iqr()    — detection, fence values, edge cases.
* OutlierDetector.zscore() — detection, error on constant series.
* OutlierDetector.mad()    — detection, error on all-identical series.
* OutlierDetector.gesd()   — detection of planted spike.
* OutlierDetector.remove() — NaN replacement.
* OutlierDetector.clip()   — fence clamping; error when no bounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.quality.outliers import OutlierDetector, OutlierReport

det = OutlierDetector()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normal_ts(n: int = 100, seed: int = 0) -> TimeSeries:
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2020-01-01", periods=n, freq="D")
    vals = rng.standard_normal(n)
    return TimeSeries(vals, index=idx, name="normal")


def _ts_with_spike(pos: int = 10, spike: float = 50.0, n: int = 50, seed: int = 7) -> TimeSeries:
    """Normal(0,1) background with a single planted spike — MAD-safe."""
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2020-01-01", periods=n, freq="D")
    vals = rng.standard_normal(n)
    vals[pos] = spike
    return TimeSeries(vals, index=idx, name="spike")


# ===========================================================================
# IQR
# ===========================================================================


class TestIQR:
    def test_detects_spike(self):
        ts = _ts_with_spike(10, 100.0)
        r  = det.iqr(ts)
        assert r.n_outliers >= 1
        assert 10 in r.indices

    def test_no_outliers_on_normal(self):
        rng  = np.random.default_rng(42)
        idx  = pd.date_range("2020", periods=30, freq="D")
        # Tightly bounded data well within IQR fences
        vals = np.ones(30) * 5.0 + rng.uniform(-0.01, 0.01, 30)
        ts   = TimeSeries(vals, index=idx)
        r    = det.iqr(ts, k=1.5)
        assert r.n_outliers == 0

    def test_fence_values_set(self):
        ts = _normal_ts(100, seed=42)   # normally distributed — guarantees IQR > 0
        r  = det.iqr(ts)
        assert r.lower_bound is not None
        assert r.upper_bound is not None
        assert r.lower_bound < r.upper_bound

    def test_method_label(self):
        ts = _ts_with_spike()
        r  = det.iqr(ts, k=2.0)
        assert "IQR" in r.method
        assert "2.0" in r.method

    def test_returns_report(self):
        ts = _ts_with_spike()
        assert isinstance(det.iqr(ts), OutlierReport)

    def test_k_zero_raises(self):
        ts = _ts_with_spike()
        with pytest.raises(ValueError, match="positive"):
            det.iqr(ts, k=0.0)

    def test_too_few_values_raises(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        ts  = TimeSeries([1.0, 2.0, 3.0], index=idx)
        with pytest.raises(ValueError, match="4"):
            det.iqr(ts)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            det.iqr("not a series")  # type: ignore[arg-type]


# ===========================================================================
# Z-score
# ===========================================================================


class TestZScore:
    def test_detects_spike(self):
        ts = _ts_with_spike(5, 50.0)
        r  = det.zscore(ts)
        assert 5 in r.indices

    def test_threshold_respected(self):
        ts = _ts_with_spike(5, 5.0, n=100)  # moderate spike
        r_loose  = det.zscore(ts, threshold=1.0)
        r_strict = det.zscore(ts, threshold=10.0)
        assert r_loose.n_outliers >= r_strict.n_outliers

    def test_constant_series_raises(self):
        idx = pd.date_range("2020", periods=5, freq="D")
        ts  = TimeSeries([3.0, 3.0, 3.0, 3.0, 3.0], index=idx)
        with pytest.raises(ValueError, match="std"):
            det.zscore(ts)

    def test_negative_threshold_raises(self):
        ts = _ts_with_spike()
        with pytest.raises(ValueError):
            det.zscore(ts, threshold=-1.0)

    def test_method_label(self):
        ts = _ts_with_spike()
        r  = det.zscore(ts, threshold=2.5)
        assert "Z-score" in r.method


# ===========================================================================
# MAD
# ===========================================================================


class TestMAD:
    def test_detects_spike(self):
        ts = _ts_with_spike(7, 80.0)
        r  = det.mad(ts)
        assert 7 in r.indices

    def test_mad_zero_raises(self):
        # More than half the values are identical → MAD = 0
        idx = pd.date_range("2020", periods=8, freq="D")
        ts  = TimeSeries([5.0] * 5 + [6.0, 7.0, 8.0], index=idx)
        with pytest.raises(ValueError, match="MAD"):
            det.mad(ts)

    def test_negative_threshold_raises(self):
        ts = _ts_with_spike()
        with pytest.raises(ValueError):
            det.mad(ts, threshold=-1.0)

    def test_method_label(self):
        ts = _ts_with_spike()
        r  = det.mad(ts, threshold=4.0)
        assert "MAD" in r.method

    def test_bounds_set(self):
        ts = _ts_with_spike()
        r  = det.mad(ts)
        assert r.lower_bound is not None and r.upper_bound is not None


# ===========================================================================
# GESD
# ===========================================================================


class TestGESD:
    def test_detects_planted_spike(self):
        rng  = np.random.default_rng(1)
        idx  = pd.date_range("2020", periods=50, freq="D")
        vals = rng.standard_normal(50)
        vals[15] = 15.0
        ts   = TimeSeries(vals, index=idx)
        r    = det.gesd(ts)
        assert 15 in r.indices

    def test_no_bounds(self):
        ts = _normal_ts(50)
        r  = det.gesd(ts)
        assert r.lower_bound is None
        assert r.upper_bound is None

    def test_method_label(self):
        ts = _normal_ts(50)
        r  = det.gesd(ts)
        assert "GESD" in r.method

    def test_alpha_out_of_range_raises(self):
        ts = _normal_ts(50)
        with pytest.raises(ValueError, match="alpha"):
            det.gesd(ts, alpha=1.5)

    def test_max_outliers_too_large_raises(self):
        ts = _normal_ts(20)
        with pytest.raises(ValueError, match="max_outliers"):
            det.gesd(ts, max_outliers=15)


# ===========================================================================
# remove() and clip()
# ===========================================================================


class TestRemoveClip:
    def test_remove_replaces_with_nan(self):
        ts = _ts_with_spike(3, 100.0)
        r  = det.iqr(ts)
        cleaned = det.remove(ts, r)
        assert cleaned.has_nan
        assert np.isnan(cleaned.values[3])

    def test_remove_preserves_length(self):
        ts = _ts_with_spike()
        r  = det.iqr(ts)
        assert det.remove(ts, r).n == ts.n

    def test_clip_bounds_values(self):
        ts = _ts_with_spike(3, 100.0)
        r  = det.iqr(ts)
        clipped = det.clip(ts, r)
        assert clipped.values.max() <= r.upper_bound + 1e-9
        assert clipped.values.min() >= r.lower_bound - 1e-9

    def test_clip_no_nan_introduced(self):
        ts = _ts_with_spike()
        r  = det.iqr(ts)
        assert not det.clip(ts, r).has_nan

    def test_clip_gesd_no_bounds_raises(self):
        ts = _normal_ts(50)
        r  = det.gesd(ts)
        with pytest.raises(ValueError, match="bounds"):
            det.clip(ts, r)

    def test_remove_bad_report_raises(self):
        ts = _ts_with_spike()
        with pytest.raises(TypeError):
            det.remove(ts, "not a report")  # type: ignore[arg-type]