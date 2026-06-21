"""
Tests for :mod:`tseda.quality.duplicates`.

Coverage targets
----------------
* DuplicateDetector.flatline()        — detection, min_run, atol, mask.
* DuplicateDetector.near_zero()       — detection of near-zero runs.
* DuplicateDetector.remove_flatlines() — keep_first flag, NaN count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.quality.duplicates import DuplicateDetector, FlatlineReport

det = DuplicateDetector()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(vals: list[float]) -> TimeSeries:
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="D")
    return TimeSeries(np.array(vals, dtype=float), index=idx, name="test")


# ===========================================================================
# flatline()
# ===========================================================================


class TestFlatline:
    def test_detects_single_run(self):
        ts = _ts([1.0, 3.0, 3.0, 3.0, 3.0, 2.0])
        r  = det.flatline(ts, min_run=3)
        assert r.n_flatline_runs == 1
        assert r.runs[0] == (1, 4, 3.0)

    def test_longest_run(self):
        ts = _ts([5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 2.0, 2.0])
        r  = det.flatline(ts, min_run=3)
        assert r.longest_run == 4

    def test_two_runs(self):
        ts = _ts([0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0])
        r  = det.flatline(ts, min_run=3)
        assert r.n_flatline_runs == 2

    def test_run_at_end(self):
        ts = _ts([1.0, 2.0, 7.0, 7.0, 7.0])
        r  = det.flatline(ts, min_run=3)
        assert r.n_flatline_runs == 1
        assert r.runs[0] == (2, 4, 7.0)

    def test_run_at_start(self):
        ts = _ts([4.0, 4.0, 4.0, 1.0, 2.0])
        r  = det.flatline(ts, min_run=3)
        assert r.n_flatline_runs == 1

    def test_no_run_below_min(self):
        ts = _ts([1.0, 1.0, 2.0, 3.0])
        r  = det.flatline(ts, min_run=3)
        assert r.n_flatline_runs == 0

    def test_mask_correct(self):
        ts = _ts([1.0, 5.0, 5.0, 5.0, 2.0])
        r  = det.flatline(ts, min_run=3)
        expected_mask = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(r.mask, expected_mask)

    def test_total_flatline_points(self):
        ts = _ts([1.0, 5.0, 5.0, 5.0, 5.0, 2.0])
        r  = det.flatline(ts, min_run=3)
        assert r.total_flatline_points == 4

    def test_atol_fuzzy_equality(self):
        ts = _ts([1.0, 2.000, 2.001, 2.002, 2.003, 1.0])
        r  = det.flatline(ts, min_run=3, atol=0.005)
        assert r.n_flatline_runs == 1

    def test_nan_breaks_run(self):
        ts = _ts([5.0, 5.0, float("nan"), 5.0, 5.0, 5.0])
        r  = det.flatline(ts, min_run=3)
        # NaN breaks the first pair; only the trailing run of 3 qualifies
        assert r.n_flatline_runs == 1

    def test_min_run_1_raises(self):
        ts = _ts([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="min_run"):
            det.flatline(ts, min_run=1)

    def test_negative_atol_raises(self):
        ts = _ts([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="atol"):
            det.flatline(ts, min_run=3, atol=-0.1)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            det.flatline("not a ts", min_run=3)  # type: ignore[arg-type]

    def test_returns_report_type(self, ts_daily):
        r = det.flatline(ts_daily)
        assert isinstance(r, FlatlineReport)

    def test_no_run_all_different(self):
        ts = _ts([1.0, 2.0, 3.0, 4.0, 5.0])
        r  = det.flatline(ts, min_run=2)
        assert r.n_flatline_runs == 0


# ===========================================================================
# near_zero()
# ===========================================================================


class TestNearZero:
    def test_detects_zero_run(self):
        ts = _ts([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        r  = det.near_zero(ts, min_run=3)
        assert r.n_flatline_runs == 1

    def test_detects_near_zero_run(self):
        ts = _ts([1.0, 1e-10, 1e-10, 1e-10, 1e-10, 1.0])
        r  = det.near_zero(ts, min_run=3, threshold=1e-8)
        assert r.n_flatline_runs == 1

    def test_no_detection_above_threshold(self):
        ts = _ts([1.0, 0.01, 0.01, 0.01, 1.0])
        r  = det.near_zero(ts, min_run=3, threshold=1e-8)
        assert r.n_flatline_runs == 0

    def test_negative_threshold_raises(self):
        ts = _ts([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="threshold"):
            det.near_zero(ts, threshold=-0.1)


# ===========================================================================
# remove_flatlines()
# ===========================================================================


class TestRemoveFlatlines:
    def test_keep_first_true(self):
        ts = _ts([1.0, 5.0, 5.0, 5.0, 2.0])
        r  = det.flatline(ts, min_run=3)
        cleaned = det.remove_flatlines(ts, r, keep_first=True)
        assert cleaned.values[1] == 5.0   # first of run kept
        assert np.isnan(cleaned.values[2])
        assert np.isnan(cleaned.values[3])
        assert cleaned.n_nan == 2

    def test_keep_first_false(self):
        ts = _ts([1.0, 5.0, 5.0, 5.0, 2.0])
        r  = det.flatline(ts, min_run=3)
        cleaned = det.remove_flatlines(ts, r, keep_first=False)
        assert np.isnan(cleaned.values[1])
        assert cleaned.n_nan == 3

    def test_no_flatlines_no_change(self, ts_daily):
        r = det.flatline(ts_daily, min_run=3)
        if r.n_flatline_runs == 0:
            cleaned = det.remove_flatlines(ts_daily, r)
            assert cleaned.n_nan == ts_daily.n_nan

    def test_bad_report_raises(self):
        ts = _ts([1.0, 2.0, 3.0])
        with pytest.raises(TypeError):
            det.remove_flatlines(ts, "not a report")  # type: ignore[arg-type]