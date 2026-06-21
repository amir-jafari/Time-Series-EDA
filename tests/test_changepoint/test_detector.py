"""
Tests for :mod:`tseda.changepoint.detector`.

Coverage targets
----------------
* ChangepointDetector.cusum()              — shift detection, parameters.
* ChangepointDetector.binary_segmentation()— one / two shifts, no shift.
* ChangepointDetector.variance_ratio()     — variance shift detection.
* ChangepointDetector.segment()            — segment table structure.
* ChangepointReport.segment_labels()       — label array shape and values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.changepoint.detector import (
    ChangepointDetector,
    ChangepointReport,
    _binary_seg_recursive,
    _cusum_arrays,
)

det = ChangepointDetector()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _level_shift_ts(
    n: int = 300,
    break_at: int = 150,
    shift: float = 5.0,
    seed: int = 0,
) -> TimeSeries:
    """Normal(0,1) background with one planted mean shift at *break_at*."""
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2020-01-01", periods=n, freq="D")
    vals = np.concatenate([
        rng.standard_normal(break_at),
        rng.standard_normal(n - break_at) + shift,
    ])
    return TimeSeries(vals, index=idx, name="shift_ts")


def _two_shift_ts(n: int = 300, seed: int = 0) -> TimeSeries:
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2020-01-01", periods=n, freq="D")
    vals = np.concatenate([
        rng.standard_normal(100),
        rng.standard_normal(100) + 5.0,
        rng.standard_normal(100),
    ])
    return TimeSeries(vals, index=idx, name="two_shift")


def _variance_shift_ts(
    n: int = 200, break_at: int = 100, ratio: float = 5.0, seed: int = 0
) -> TimeSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020", periods=n, freq="D")
    vals = np.concatenate([
        rng.standard_normal(break_at) * 0.5,
        rng.standard_normal(n - break_at) * ratio * 0.5,
    ])
    return TimeSeries(vals, index=idx, name="var_shift")


def _stationary_ts(n: int = 200, seed: int = 0) -> TimeSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020", periods=n, freq="D")
    return TimeSeries(rng.standard_normal(n), index=idx, name="stationary")


# ===========================================================================
# Private helpers
# ===========================================================================

class TestCUSUMArrays:
    def test_flat_signal_zero_cusum(self):
        x = np.zeros(50)
        S_p, S_n = _cusum_arrays(x, target=0.0, sigma=1.0, k=0.5, h=5.0)
        assert np.all(S_p == 0.0)
        assert np.all(S_n == 0.0)

    def test_upward_shift_positive_cusum(self):
        x = np.ones(100) * 5.0   # strong upward deviation
        S_p, S_n = _cusum_arrays(x, target=0.0, sigma=1.0, k=0.5, h=5.0)
        assert S_p.max() > 0.0

    def test_lengths_match_input(self):
        x = np.arange(30, dtype=float)
        S_p, S_n = _cusum_arrays(x, target=15.0, sigma=1.0, k=0.5, h=5.0)
        assert len(S_p) == 30 and len(S_n) == 30


class TestBinarySegRecursive:
    def test_clear_shift_returns_one_cp(self):
        rng = np.random.default_rng(0)
        x   = np.concatenate([rng.standard_normal(50), rng.standard_normal(50) + 8.0])
        # penalty=50 → σ²×n ≈ 1×100 / 2 is typical; 50 avoids spurious noise splits
        cps = _binary_seg_recursive(x, offset=0, min_size=5, penalty=50.0)
        assert len(cps) == 1
        assert abs(cps[0] - 50) <= 5

    def test_no_shift_empty(self):
        rng = np.random.default_rng(1)
        x   = rng.standard_normal(50)
        # Very high penalty → no splits accepted
        cps = _binary_seg_recursive(x, offset=0, min_size=5, penalty=1e9)
        assert cps == []


# ===========================================================================
# CUSUM
# ===========================================================================

class TestCUSUM:
    def test_detects_mean_shift(self):
        ts = _level_shift_ts(shift=6.0)
        r  = det.cusum(ts, threshold=4.0, drift=0.5)
        assert r.n_changepoints >= 1

    def test_no_shift_in_stationary(self):
        ts = _stationary_ts()
        r  = det.cusum(ts, threshold=10.0)  # very high threshold
        assert r.n_changepoints == 0

    def test_returns_report(self):
        assert isinstance(det.cusum(_level_shift_ts()), ChangepointReport)

    def test_scores_in_range(self):
        r = det.cusum(_level_shift_ts())
        assert np.all((r.scores >= 0) & (r.scores <= 1))

    def test_method_label(self):
        r = det.cusum(_level_shift_ts(), threshold=5.0, drift=0.5)
        assert "cusum" in r.method

    def test_mask_length(self):
        ts = _level_shift_ts()
        r  = det.cusum(ts)
        assert len(r.scores) == ts.n

    def test_bad_threshold_raises(self):
        ts = _level_shift_ts()
        with pytest.raises(ValueError, match="positive"):
            det.cusum(ts, threshold=0.0)

    def test_bad_drift_raises(self):
        ts = _level_shift_ts()
        with pytest.raises(ValueError, match="positive"):
            det.cusum(ts, drift=-1.0)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            det.cusum("not a ts")  # type: ignore[arg-type]

    def test_too_few_obs_raises(self):
        idx = pd.date_range("2020", periods=5, freq="D")
        ts  = TimeSeries(np.ones(5), index=idx)
        with pytest.raises(ValueError, match="10"):
            det.cusum(ts)

    def test_timestamps_match_positions(self):
        ts = _level_shift_ts()
        r  = det.cusum(ts, threshold=4.0)
        for pos, ts_stamp in zip(r.changepoints, r.timestamps):
            assert ts.index[pos] == ts_stamp


# ===========================================================================
# Binary segmentation
# ===========================================================================

class TestBinarySegmentation:
    def test_detects_one_shift(self):
        ts = _level_shift_ts(shift=6.0)
        r  = det.binary_segmentation(ts)
        assert r.n_changepoints == 1

    def test_cp_near_true_break(self):
        ts = _level_shift_ts(n=300, break_at=150, shift=6.0)
        r  = det.binary_segmentation(ts)
        if r.n_changepoints >= 1:
            assert abs(r.changepoints[0] - 150) <= 15

    def test_detects_two_shifts(self):
        ts = _two_shift_ts()
        r  = det.binary_segmentation(ts)
        assert r.n_changepoints == 2

    def test_no_shift_returns_zero(self):
        ts = _stationary_ts()
        # Large penalty prevents spurious splits
        n  = ts.n
        x  = ts.values
        sigma2 = float(np.var(np.diff(x), ddof=1) / 2)
        r  = det.binary_segmentation(ts, penalty=sigma2 * n * 5)
        assert r.n_changepoints == 0

    def test_returns_report(self):
        assert isinstance(det.binary_segmentation(_level_shift_ts()), ChangepointReport)

    def test_scores_in_range(self):
        r = det.binary_segmentation(_level_shift_ts())
        assert np.all((r.scores >= 0) & (r.scores <= 1))

    def test_changepoints_sorted(self):
        ts  = _two_shift_ts()
        r   = det.binary_segmentation(ts)
        cps = r.changepoints
        assert cps == sorted(cps)

    def test_method_label(self):
        r = det.binary_segmentation(_level_shift_ts())
        assert "binary_segmentation" in r.method

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            det.binary_segmentation("not a ts")  # type: ignore[arg-type]

    def test_mask_length(self):
        ts = _level_shift_ts()
        r  = det.binary_segmentation(ts)
        assert len(r.scores) == ts.n


# ===========================================================================
# Variance ratio
# ===========================================================================

class TestVarianceRatio:
    def test_detects_variance_shift(self):
        ts = _variance_shift_ts(ratio=6.0)
        r  = det.variance_ratio(ts, window=20, alpha=0.01)
        assert r.n_changepoints >= 1

    def test_cp_near_true_break(self):
        ts = _variance_shift_ts(n=200, break_at=100, ratio=8.0)
        r  = det.variance_ratio(ts, window=20, alpha=0.01)
        if r.n_changepoints >= 1:
            assert abs(r.changepoints[0] - 100) <= 20

    def test_returns_report(self):
        assert isinstance(
            det.variance_ratio(_variance_shift_ts()), ChangepointReport
        )

    def test_scores_in_range(self):
        r = det.variance_ratio(_variance_shift_ts())
        assert np.all((r.scores >= 0) & (r.scores <= 1))

    def test_method_label(self):
        r = det.variance_ratio(_variance_shift_ts(), window=20, alpha=0.05)
        assert "variance_ratio" in r.method

    def test_bad_alpha_raises(self):
        ts = _variance_shift_ts()
        with pytest.raises(ValueError, match="alpha"):
            det.variance_ratio(ts, alpha=1.5)

    def test_window_too_small_raises(self):
        ts = _variance_shift_ts()
        with pytest.raises(ValueError, match="window"):
            det.variance_ratio(ts, window=2)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            det.variance_ratio("not a ts")  # type: ignore[arg-type]


# ===========================================================================
# segment()
# ===========================================================================

class TestSegment:
    def test_n_rows_equals_n_cps_plus_one(self):
        ts  = _level_shift_ts(shift=6.0)
        r   = det.binary_segmentation(ts)
        df  = det.segment(ts, r)
        assert len(df) == r.n_changepoints + 1

    def test_column_names(self):
        ts  = _level_shift_ts(shift=6.0)
        r   = det.binary_segmentation(ts)
        df  = det.segment(ts, r)
        assert set(df.columns) == {"segment", "start", "end", "n_obs",
                                    "mean", "std", "min", "max"}

    def test_n_obs_sum_equals_ts_n(self):
        ts  = _level_shift_ts(shift=6.0)
        r   = det.binary_segmentation(ts)
        df  = det.segment(ts, r)
        assert df["n_obs"].sum() == ts.n

    def test_segment_means_differ_after_level_shift(self):
        ts = _level_shift_ts(shift=8.0)
        r  = det.binary_segmentation(ts)
        if r.n_changepoints >= 1:
            df = det.segment(ts, r)
            means = df["mean"].values
            assert abs(means[-1] - means[0]) > 3.0

    def test_bad_report_type_raises(self):
        ts = _level_shift_ts()
        with pytest.raises(TypeError):
            det.segment(ts, "not a report")  # type: ignore[arg-type]

    def test_bad_ts_type_raises(self):
        ts = _level_shift_ts()
        r  = det.binary_segmentation(ts)
        with pytest.raises(TypeError):
            det.segment("not a ts", r)  # type: ignore[arg-type]


# ===========================================================================
# ChangepointReport.segment_labels()
# ===========================================================================

class TestSegmentLabels:
    def test_shape(self):
        r = det.binary_segmentation(_level_shift_ts(shift=6.0))
        lbl = r.segment_labels(300)
        assert len(lbl) == 300

    def test_labels_start_at_zero(self):
        r = det.binary_segmentation(_level_shift_ts(shift=6.0))
        lbl = r.segment_labels(300)
        assert lbl[0] == 0

    def test_label_count_equals_n_segments(self):
        ts  = _two_shift_ts()
        r   = det.binary_segmentation(ts)
        lbl = r.segment_labels(ts.n)
        assert len(set(lbl)) == r.n_changepoints + 1

    def test_known_positions(self):
        """segment_labels doctest example."""
        r = ChangepointReport(
            changepoints=[3, 7],
            timestamps=pd.DatetimeIndex([]),
            n_changepoints=2,
            scores=np.zeros(10),
            method="test",
        )
        assert r.segment_labels(10).tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]