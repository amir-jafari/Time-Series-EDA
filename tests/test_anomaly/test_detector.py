"""
Tests for :mod:`tseda.anomaly.detector`.

Coverage targets
----------------
* AnomalyDetector.rolling_iqr() — spike detection, parameters, edges.
* AnomalyDetector.rolling_z()   — spike detection, parameters.
* AnomalyDetector.stl_residual()— three residual sub-methods, seasonal data.
* AnomalyDetector.gesd()        — known spike, clean data.
* AnomalyDetector.remove()      — NaN replacement.
* AnomalyDetector.label()       — 0/1 indicator series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.anomaly.detector import AnomalyDetector, AnomalyReport

det = AnomalyDetector()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_ts(n: int = 100, spike_pos: int = 50, spike_val: float = 20.0,
             seed: int = 0) -> TimeSeries:
    """Normal(0,1) background with one planted spike."""
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2020-01-01", periods=n, freq="D")
    vals = rng.standard_normal(n)
    vals[spike_pos] = spike_val
    return TimeSeries(vals, index=idx, name="test")


def _seasonal_ts(n: int = 60, period: int = 12,
                 spike_pos: int = 25, spike_val: float = 20.0,
                 seed: int = 0) -> TimeSeries:
    """Seasonal signal with one planted spike."""
    rng  = np.random.default_rng(seed)
    seas = np.tile(np.sin(2 * np.pi * np.arange(period) / period) * 5, n // period + 1)[:n]
    vals = seas + rng.standard_normal(n) * 0.3
    vals[spike_pos] = spike_val
    idx  = pd.date_range("2018-01", periods=n, freq="MS")
    return TimeSeries(vals, index=idx, name="seasonal")


# ===========================================================================
# rolling_iqr
# ===========================================================================

class TestRollingIQR:
    def test_detects_spike(self):
        ts = _flat_ts(spike_pos=50, spike_val=15.0)
        r  = det.rolling_iqr(ts, window=30)
        assert 50 in r.indices

    def test_no_detection_on_clean_data(self):
        rng  = np.random.default_rng(99)
        idx  = pd.date_range("2020", periods=100, freq="D")
        # Tight constant signal — should have very few anomalies
        vals = np.ones(100) * 5.0 + rng.uniform(-0.01, 0.01, 100)
        ts   = TimeSeries(vals, index=idx)
        r    = det.rolling_iqr(ts, window=30, k=3.0)
        assert r.n_anomalies == 0

    def test_returns_report(self):
        assert isinstance(det.rolling_iqr(_flat_ts()), AnomalyReport)

    def test_mask_length(self):
        ts = _flat_ts()
        r  = det.rolling_iqr(ts)
        assert len(r.mask) == ts.n

    def test_scores_in_range(self):
        ts = _flat_ts()
        r  = det.rolling_iqr(ts)
        assert np.all((r.scores >= 0) & (r.scores <= 1))

    def test_scores_nonzero_at_anomaly(self):
        ts = _flat_ts(spike_pos=50, spike_val=15.0)
        r  = det.rolling_iqr(ts, window=30)
        if 50 in r.indices:
            assert r.scores[50] > 0

    def test_method_label(self):
        r = det.rolling_iqr(_flat_ts(), window=20, k=2.0)
        assert "rolling_iqr" in r.method
        assert "20" in r.method

    def test_bad_k_raises(self):
        ts = _flat_ts()
        with pytest.raises(ValueError, match="positive"):
            det.rolling_iqr(ts, k=0.0)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            det.rolling_iqr("not a ts")  # type: ignore[arg-type]

    def test_too_few_obs_raises(self):
        idx = pd.date_range("2020", periods=5, freq="D")
        ts  = TimeSeries(np.ones(5), index=idx)
        with pytest.raises(ValueError, match="8"):
            det.rolling_iqr(ts)


# ===========================================================================
# rolling_z
# ===========================================================================

class TestRollingZ:
    def test_detects_spike(self):
        ts = _flat_ts(spike_pos=60, spike_val=10.0)
        r  = det.rolling_z(ts, window=30)
        assert 60 in r.indices

    def test_detects_negative_spike(self):
        rng  = np.random.default_rng(3)
        idx  = pd.date_range("2020", periods=100, freq="D")
        vals = rng.standard_normal(100)
        vals[40] = -10.0
        ts   = TimeSeries(vals, index=idx)
        r    = det.rolling_z(ts, window=30)
        assert 40 in r.indices

    def test_returns_report(self):
        assert isinstance(det.rolling_z(_flat_ts()), AnomalyReport)

    def test_method_label(self):
        r = det.rolling_z(_flat_ts(), window=25, threshold=2.5)
        assert "rolling_z" in r.method

    def test_scores_in_range(self):
        r = det.rolling_z(_flat_ts())
        assert np.all((r.scores >= 0) & (r.scores <= 1))

    def test_bad_threshold_raises(self):
        ts = _flat_ts()
        with pytest.raises(ValueError, match="positive"):
            det.rolling_z(ts, threshold=0.0)

    def test_preserves_length(self):
        ts = _flat_ts()
        r  = det.rolling_z(ts)
        assert len(r.mask) == ts.n

    def test_high_threshold_fewer_anomalies(self):
        ts = _flat_ts(spike_val=5.0)
        r_tight  = det.rolling_z(ts, threshold=2.0)
        r_loose  = det.rolling_z(ts, threshold=6.0)
        assert r_tight.n_anomalies >= r_loose.n_anomalies


# ===========================================================================
# stl_residual
# ===========================================================================

class TestSTLResidual:
    def test_detects_spike_iqr(self):
        ts = _seasonal_ts(spike_pos=25, spike_val=20.0)
        r  = det.stl_residual(ts, period=12, residual_method="iqr")
        assert 25 in r.indices

    def test_detects_spike_mad(self):
        ts = _seasonal_ts(spike_pos=25, spike_val=20.0)
        r  = det.stl_residual(ts, period=12, residual_method="mad")
        assert 25 in r.indices

    def test_detects_spike_z(self):
        ts = _seasonal_ts(spike_pos=25, spike_val=20.0)
        r  = det.stl_residual(ts, period=12, residual_method="z")
        assert 25 in r.indices

    def test_returns_report(self):
        ts = _seasonal_ts()
        assert isinstance(det.stl_residual(ts, period=12), AnomalyReport)

    def test_method_label_contains_period(self):
        ts = _seasonal_ts()
        r  = det.stl_residual(ts, period=12)
        assert "12" in r.method

    def test_bad_method_raises(self):
        ts = _seasonal_ts()
        with pytest.raises(ValueError, match="residual_method"):
            det.stl_residual(ts, period=12, residual_method="bad")

    def test_bad_k_raises(self):
        ts = _seasonal_ts()
        with pytest.raises(ValueError, match="k"):
            det.stl_residual(ts, period=12, k=0.0)

    def test_scores_in_range(self):
        ts = _seasonal_ts()
        r  = det.stl_residual(ts, period=12)
        assert np.all((r.scores >= 0) & (r.scores <= 1))

    def test_mask_length(self):
        ts = _seasonal_ts()
        r  = det.stl_residual(ts, period=12)
        assert len(r.mask) == ts.n


# ===========================================================================
# gesd
# ===========================================================================

class TestGESD:
    def test_detects_spike(self):
        ts = _flat_ts(spike_pos=10, spike_val=12.0)
        r  = det.gesd(ts)
        assert 10 in r.indices

    def test_no_anomaly_on_clean_normal(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2020", periods=100, freq="D")
        ts  = TimeSeries(rng.standard_normal(100), index=idx)
        r   = det.gesd(ts)
        # Very few or no anomalies expected for pure normal data
        assert r.n_anomalies <= 2

    def test_returns_report(self):
        assert isinstance(det.gesd(_flat_ts()), AnomalyReport)

    def test_scores_nonzero_at_anomaly(self):
        ts = _flat_ts(spike_val=12.0)
        r  = det.gesd(ts)
        if r.n_anomalies > 0:
            assert r.scores[r.indices[0]] > 0

    def test_method_label(self):
        r = det.gesd(_flat_ts())
        assert "gesd" in r.method

    def test_mask_and_indices_consistent(self):
        ts = _flat_ts()
        r  = det.gesd(ts)
        assert list(r.indices) == list(np.where(r.mask)[0])

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            det.gesd("not a ts")  # type: ignore[arg-type]


# ===========================================================================
# remove() and label()
# ===========================================================================

class TestRemoveLabel:
    def test_remove_inserts_nan(self):
        ts = _flat_ts(spike_pos=50, spike_val=15.0)
        r  = det.rolling_iqr(ts, window=30)
        cleaned = det.remove(ts, r)
        if 50 in r.indices:
            assert np.isnan(cleaned.values[50])

    def test_remove_preserves_length(self):
        ts = _flat_ts()
        r  = det.rolling_iqr(ts)
        assert det.remove(ts, r).n == ts.n

    def test_label_ones_at_anomaly(self):
        ts = _flat_ts(spike_pos=50, spike_val=15.0)
        r  = det.rolling_iqr(ts, window=30)
        lbl = det.label(ts, r)
        if 50 in r.indices:
            assert lbl.values[50] == 1.0

    def test_label_zeros_elsewhere(self):
        ts  = _flat_ts(spike_pos=50, spike_val=15.0)
        r   = det.rolling_iqr(ts, window=30)
        lbl = det.label(ts, r)
        non_anomaly_mask = ~r.mask
        assert np.all(lbl.values[non_anomaly_mask] == 0.0)

    def test_label_name(self):
        ts  = _flat_ts()
        r   = det.rolling_iqr(ts)
        lbl = det.label(ts, r)
        assert "anomaly_label" in lbl.name

    def test_label_length(self):
        ts  = _flat_ts()
        r   = det.rolling_iqr(ts)
        assert det.label(ts, r).n == ts.n

    def test_remove_bad_report_raises(self):
        ts = _flat_ts()
        with pytest.raises(TypeError):
            det.remove(ts, "not a report")  # type: ignore[arg-type]

    def test_label_bad_report_raises(self):
        ts = _flat_ts()
        with pytest.raises(TypeError):
            det.label(ts, "not a report")  # type: ignore[arg-type]