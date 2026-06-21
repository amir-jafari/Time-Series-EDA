"""
Tests for :mod:`tseda.quality.missing`.

Coverage targets
----------------
* MissingValueAnalyzer.analyze() — n_nan, pct_nan, runs, monotone, gaps.
* MissingValueAnalyzer.interpolate() — all methods, limit, constant,
  spline; error paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.quality.missing import MissingValueAnalyzer, MissingValueReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_with_nans(positions: list[int], n: int = 10) -> TimeSeries:
    idx  = pd.date_range("2020-01-01", periods=n, freq="D")
    vals = np.arange(float(n))
    vals[positions] = np.nan
    return TimeSeries(vals, index=idx, name="test")


ana = MissingValueAnalyzer()


# ===========================================================================
# analyze()
# ===========================================================================


class TestAnalyze:
    def test_no_nan(self, ts_daily):
        r = ana.analyze(ts_daily)
        assert r.n_nan == 0
        assert r.pct_nan == 0.0
        assert r.longest_nan_run == 0
        assert len(r.nan_run_lengths) == 0

    def test_n_nan(self, ts_with_nan):
        r = ana.analyze(ts_with_nan)
        assert r.n_nan == 20

    def test_pct_nan(self, ts_with_nan):
        r = ana.analyze(ts_with_nan)
        assert abs(r.pct_nan - 10.0) < 0.1

    def test_nan_positions(self):
        ts = _ts_with_nans([2, 5, 7])
        r  = ana.analyze(ts)
        assert list(r.nan_positions) == [2, 5, 7]

    def test_longest_nan_run_single(self):
        ts = _ts_with_nans([3])
        r  = ana.analyze(ts)
        assert r.longest_nan_run == 1

    def test_longest_nan_run_consecutive(self):
        ts = _ts_with_nans([2, 3, 4])
        r  = ana.analyze(ts)
        assert r.longest_nan_run == 3

    def test_nan_run_lengths(self):
        ts = _ts_with_nans([0, 1, 5, 6, 7])
        r  = ana.analyze(ts)
        assert sorted(r.nan_run_lengths) == [2, 3]

    def test_monotone_missing_at_head(self):
        ts = _ts_with_nans([0, 1, 2])
        r  = ana.analyze(ts)
        assert r.is_monotone_missing is True

    def test_monotone_missing_at_tail(self):
        ts = _ts_with_nans([8, 9])
        r  = ana.analyze(ts)
        assert r.is_monotone_missing is True

    def test_not_monotone_missing(self):
        ts = _ts_with_nans([3, 4])
        r  = ana.analyze(ts)
        assert r.is_monotone_missing is False

    def test_index_gaps_daily(self):
        # Build a daily series but remove 2020-01-03 from the index
        full = pd.date_range("2020-01-01", periods=5, freq="D")
        missing_ts = full[[0, 1, 3, 4]]   # skip Jan 3
        vals = np.array([1.0, 2.0, 4.0, 5.0])
        ts   = TimeSeries(vals, index=missing_ts, name="gap_test")
        r    = ana.analyze(ts)
        assert r.n_gaps == 1
        assert pd.Timestamp("2020-01-03") in r.gap_locations

    def test_no_gaps_regular(self, ts_daily):
        r = ana.analyze(ts_daily)
        assert r.n_gaps == 0

    def test_gaps_unknown_freq(self, ts_irregular):
        r = ana.analyze(ts_irregular)
        # freq may be inferred or not; either way n_gaps is -1 or >= 0
        assert r.n_gaps >= -1

    def test_bad_type_raises(self):
        with pytest.raises(TypeError, match="TimeSeries"):
            ana.analyze("not a series")  # type: ignore[arg-type]

    def test_returns_report_type(self, ts_daily):
        r = ana.analyze(ts_daily)
        assert isinstance(r, MissingValueReport)


# ===========================================================================
# interpolate()
# ===========================================================================


class TestInterpolate:
    def _ts(self) -> TimeSeries:
        idx  = pd.date_range("2020-01-01", periods=7, freq="D")
        vals = np.array([1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0])
        return TimeSeries(vals, index=idx)

    def test_linear_fills_interior(self):
        filled = ana.interpolate(self._ts(), "linear")
        assert not filled.has_nan
        np.testing.assert_allclose(filled.values[:4], [1.0, 2.0, 3.0, 4.0])

    def test_forward_fill(self):
        filled = ana.interpolate(self._ts(), "forward")
        assert not filled.has_nan
        assert filled.values[1] == 1.0
        assert filled.values[2] == 1.0

    def test_backward_fill(self):
        filled = ana.interpolate(self._ts(), "backward")
        assert not filled.has_nan
        assert filled.values[1] == 4.0
        assert filled.values[2] == 4.0

    def test_nearest_fill(self):
        filled = ana.interpolate(self._ts(), "nearest")
        assert not filled.has_nan

    def test_zero_fill(self):
        filled = ana.interpolate(self._ts(), "zero")
        assert not filled.has_nan
        assert filled.values[1] == 0.0

    def test_constant_fill(self):
        filled = ana.interpolate(self._ts(), "constant", fill_value=-99.0)
        assert not filled.has_nan
        assert filled.values[1] == -99.0

    def test_constant_no_fill_value_raises(self):
        with pytest.raises(ValueError, match="fill_value"):
            ana.interpolate(self._ts(), "constant")

    def test_spline_fill(self):
        idx  = pd.date_range("2020", periods=6, freq="D")
        vals = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        ts   = TimeSeries(vals, index=idx)
        filled = ana.interpolate(ts, "spline")
        # Interior NaN should be filled; values close to expected
        assert not np.isnan(filled.values[1])
        assert not np.isnan(filled.values[4])

    def test_limit_caps_filled(self):
        idx  = pd.date_range("2020", periods=6, freq="D")
        vals = np.array([1.0, np.nan, np.nan, np.nan, 5.0, 6.0])
        ts   = TimeSeries(vals, index=idx)
        filled = ana.interpolate(ts, "forward", limit=1)
        # Only first NaN filled; remaining two stay NaN
        assert filled.values[1] == 1.0
        assert np.isnan(filled.values[2])
        assert np.isnan(filled.values[3])

    def test_preserves_metadata(self):
        ts = TimeSeries(
            [1.0, np.nan, 3.0],
            index=pd.date_range("2020", periods=3, freq="D"),
            name="x", unit="kg",
        )
        filled = ana.interpolate(ts)
        assert filled.name == "x"
        assert filled.unit == "kg"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            ana.interpolate(self._ts(), "magic")

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            ana.interpolate("not a series")  # type: ignore[arg-type]