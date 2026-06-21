"""
Tests for :class:`tseda.core.TimeSeries`.

Coverage targets
----------------
* Construction from all accepted input types.
* Validation errors on bad inputs.
* All read-only properties.
* All transform methods (diff, log, standardize, normalize, rolling, apply).
* Slicing (.slice, __getitem__).
* Resampling.
* Conversion helpers (to_series, to_frame, to_numpy, copy).
* Dunder methods (__len__, __contains__, __eq__, __repr__).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.core.types import AggMethod, DiffMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(n: int = 10, freq: str = "D") -> TimeSeries:
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    return TimeSeries(np.arange(float(n)), index=idx, name="test")


# ===========================================================================
# Construction
# ===========================================================================


class TestConstruction:
    def test_from_numpy_array(self):
        idx = pd.date_range("2020", periods=5, freq="D")
        ts = TimeSeries(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), index=idx)
        assert ts.n == 5

    def test_from_list(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        ts = TimeSeries([1, 2, 3], index=idx)
        assert ts.n == 3
        assert ts.values.dtype == float

    def test_from_tuple(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        ts = TimeSeries((1.0, 2.0, 3.0), index=idx)
        assert ts.n == 3

    def test_from_pandas_series_with_datetimeindex(self):
        idx = pd.date_range("2020", periods=4, freq="D")
        s = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx, name="px")
        ts = TimeSeries(s)
        assert ts.n == 4
        assert ts.name == "value"  # name kwarg defaults to "value"

    def test_from_series_explicit_index_overrides_series_index(self):
        idx_orig = pd.date_range("2020", periods=3, freq="D")
        idx_new  = pd.date_range("2021", periods=3, freq="D")
        s = pd.Series([1.0, 2.0, 3.0], index=idx_orig)
        ts = TimeSeries(s, index=idx_new)
        assert ts.start == pd.Timestamp("2021-01-01")

    def test_from_series_classmethod(self):
        idx = pd.date_range("2020", periods=5, freq="D")
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx, name="px")
        ts = TimeSeries.from_series(s)
        assert ts.name == "px"

    def test_from_series_classmethod_name_override(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        s = pd.Series([1.0, 2.0, 3.0], index=idx, name="old")
        ts = TimeSeries.from_series(s, name="new")
        assert ts.name == "new"

    def test_from_arrays_classmethod(self):
        vals = np.array([5.0, 10.0, 15.0])
        idx  = pd.date_range("2020", periods=3, freq="D")
        ts   = TimeSeries.from_arrays(vals, idx, name="arr")
        assert ts.n == 3
        assert ts.name == "arr"

    def test_from_dataframe_classmethod(self):
        idx = pd.date_range("2020", periods=4, freq="D")
        df  = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, index=idx)
        ts  = TimeSeries.from_dataframe(df, "a")
        assert ts.name == "a"
        assert ts.n == 4

    def test_string_datetime_index(self):
        idx = ["2020-01-01", "2020-01-02", "2020-01-03"]
        ts  = TimeSeries([1.0, 2.0, 3.0], index=idx)
        assert ts.n == 3

    def test_metadata_stored(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        ts  = TimeSeries([1, 2, 3], index=idx, name="x", unit="kg",
                         description="test series")
        assert ts.name == "x"
        assert ts.unit == "kg"
        assert ts.description == "test series"


# ===========================================================================
# Validation errors
# ===========================================================================


class TestValidationErrors:
    def test_missing_index_raises(self):
        with pytest.raises(ValueError, match="index"):
            TimeSeries([1.0, 2.0, 3.0])

    def test_length_mismatch_raises(self):
        idx = pd.date_range("2020", periods=5, freq="D")
        with pytest.raises(ValueError, match="same length"):
            TimeSeries([1.0, 2.0], index=idx)

    def test_non_monotonic_index_raises(self):
        idx = pd.DatetimeIndex(["2020-01-03", "2020-01-01", "2020-01-02"])
        with pytest.raises(ValueError, match="monotonically increasing"):
            TimeSeries([1.0, 2.0, 3.0], index=idx)

    def test_duplicate_timestamps_raise(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"])
        with pytest.raises(ValueError, match="duplicate"):
            TimeSeries([1.0, 2.0, 3.0], index=idx)

    def test_2d_array_raises(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        with pytest.raises(ValueError, match="1-D"):
            TimeSeries(np.ones((3, 2)), index=idx)

    def test_non_numeric_list_raises(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        with pytest.raises(ValueError):
            TimeSeries(["a", "b", "c"], index=idx)

    def test_empty_data_raises(self):
        idx = pd.DatetimeIndex([])
        with pytest.raises(ValueError):
            TimeSeries([], index=idx)

    def test_bad_type_raises(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        with pytest.raises(TypeError):
            TimeSeries({"a": 1}, index=idx)  # type: ignore[arg-type]

    def test_from_dataframe_missing_column_raises(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        df  = pd.DataFrame({"a": [1, 2, 3]}, index=idx)
        with pytest.raises(KeyError, match="b"):
            TimeSeries.from_dataframe(df, "b")

    def test_invalid_freq_string_raises(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        with pytest.raises(ValueError, match="offset alias"):
            TimeSeries([1, 2, 3], index=idx, freq="NOTAFREQ")


# ===========================================================================
# Properties
# ===========================================================================


class TestProperties:
    def test_n(self, ts_daily):
        assert ts_daily.n == 365

    def test_start_end(self, ts_daily):
        assert ts_daily.start == pd.Timestamp("2020-01-01")
        assert ts_daily.end   == pd.Timestamp("2020-12-30")

    def test_duration(self, ts_daily):
        assert ts_daily.duration == pd.Timedelta("364D")

    def test_freq_daily(self, ts_daily):
        assert ts_daily.freq == "D"

    def test_freq_hourly(self, ts_hourly):
        assert ts_hourly.freq == "h"

    def test_freq_monthly(self, ts_monthly):
        assert ts_monthly.freq == "MS"

    def test_freq_label_daily(self, ts_daily):
        assert ts_daily.freq_label == "Daily"

    def test_is_regular_true(self, ts_daily):
        assert ts_daily.is_regular is True

    def test_is_regular_false(self, ts_irregular):
        assert ts_irregular.is_regular is False

    def test_has_nan_false(self, ts_daily):
        assert ts_daily.has_nan is False

    def test_has_nan_true(self, ts_with_nan):
        assert ts_with_nan.has_nan is True

    def test_n_nan(self, ts_with_nan):
        assert ts_with_nan.n_nan == 20

    def test_values_returns_copy(self, ts_daily):
        v1 = ts_daily.values
        v2 = ts_daily.values
        v1[0] = 99_999.0
        assert ts_daily.values[0] != 99_999.0
        assert v2[0] != 99_999.0


# ===========================================================================
# Transforms
# ===========================================================================


class TestDiff:
    def test_simple_diff_length(self, ts_daily):
        d = ts_daily.diff()
        assert d.n == ts_daily.n - 1

    def test_simple_diff_periods(self, ts_daily):
        d = ts_daily.diff(7)
        assert d.n == ts_daily.n - 7

    def test_log_diff(self, ts_monthly):
        d = ts_monthly.diff(method=DiffMethod.LOG)
        assert d.n == ts_monthly.n - 1

    def test_percent_diff(self, ts_monthly):
        d = ts_monthly.diff(method=DiffMethod.PERCENT)
        assert d.n == ts_monthly.n - 1

    def test_log_diff_negative_raises(self):
        idx = pd.date_range("2020", periods=4, freq="D")
        ts  = TimeSeries([-1.0, 2.0, 3.0, 4.0], index=idx)
        with pytest.raises(ValueError, match="positive"):
            ts.diff(method="log")

    def test_diff_name_updated(self, ts_daily):
        d = ts_daily.diff()
        assert "_diff1" in d.name


class TestLog:
    def test_log_values(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        ts  = TimeSeries([1.0, np.e, np.e ** 2], index=idx)
        result = ts.log().values
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0])

    def test_log_non_positive_raises(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        ts  = TimeSeries([0.0, 1.0, 2.0], index=idx)
        with pytest.raises(ValueError, match="positive"):
            ts.log()

    def test_log_name(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        ts  = TimeSeries([1.0, 2.0, 3.0], index=idx, name="price")
        assert ts.log().name == "log(price)"


class TestStandardize:
    def test_mean_zero(self):
        ts = _make(100)
        z  = ts.standardize()
        assert abs(z.values.mean()) < 1e-10

    def test_std_one(self):
        ts = _make(100)
        z  = ts.standardize()
        assert abs(z.values.std(ddof=1) - 1.0) < 1e-10

    def test_constant_series_raises(self):
        idx = pd.date_range("2020", periods=4, freq="D")
        ts  = TimeSeries([5.0, 5.0, 5.0, 5.0], index=idx)
        with pytest.raises(ValueError, match="std"):
            ts.standardize()


class TestNormalize:
    def test_min_zero_max_one(self):
        ts  = _make(10)
        n   = ts.normalize()
        assert abs(n.values.min() - 0.0) < 1e-12
        assert abs(n.values.max() - 1.0) < 1e-12

    def test_custom_bounds(self):
        ts = _make(5)
        n  = ts.normalize(lower=-1.0, upper=1.0)
        assert abs(n.values.min() - (-1.0)) < 1e-12
        assert abs(n.values.max() - 1.0)   < 1e-12

    def test_invalid_bounds_raises(self):
        ts = _make(5)
        with pytest.raises(ValueError, match="lower"):
            ts.normalize(lower=1.0, upper=0.0)

    def test_constant_series_raises(self):
        idx = pd.date_range("2020", periods=4, freq="D")
        ts  = TimeSeries([3.0, 3.0, 3.0, 3.0], index=idx)
        with pytest.raises(ValueError, match="max"):
            ts.normalize()


class TestRolling:
    def test_rolling_mean_length(self):
        ts = _make(10)
        r  = ts.rolling(3)
        assert r.n == 8  # 10 - (3-1) NaN rows dropped

    def test_rolling_sum(self):
        idx = pd.date_range("2020", periods=5, freq="D")
        ts  = TimeSeries([1.0, 1.0, 1.0, 1.0, 1.0], index=idx)
        r   = ts.rolling(3, agg=AggMethod.SUM)
        np.testing.assert_array_equal(r.values, [3.0, 3.0, 3.0])

    def test_rolling_invalid_agg_raises(self):
        ts = _make(10)
        with pytest.raises(AttributeError):
            ts.rolling(3, agg="nonsense")  # type: ignore[arg-type]


class TestApply:
    def test_sqrt(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        ts  = TimeSeries([1.0, 4.0, 9.0], index=idx)
        r   = ts.apply(np.sqrt)
        np.testing.assert_allclose(r.values, [1.0, 2.0, 3.0])

    def test_wrong_length_raises(self):
        ts = _make(5)
        with pytest.raises(ValueError, match="same length"):
            ts.apply(lambda x: x[:3])

    def test_custom_name(self):
        ts = _make(5)
        r  = ts.apply(np.abs, name="abs_test")
        assert r.name == "abs_test"


# ===========================================================================
# Slicing
# ===========================================================================


class TestSlice:
    def test_slice_by_string(self, ts_daily):
        q1 = ts_daily.slice("2020-01-01", "2020-03-31")
        assert q1.n == 91

    def test_slice_open_start(self, ts_daily):
        s = ts_daily.slice(end="2020-01-31")
        assert s.n == 31

    def test_slice_open_end(self, ts_daily):
        s = ts_daily.slice(start="2020-12-01")
        assert s.n == 30

    def test_slice_empty_raises(self, ts_daily):
        with pytest.raises(ValueError, match="empty"):
            ts_daily.slice("2025-01-01", "2025-12-31")

    def test_getitem_integer(self, ts_short):
        assert ts_short[0] == 10.0
        assert ts_short[-1] == 50.0

    def test_getitem_slice(self, ts_short):
        sub = ts_short[1:3]
        assert isinstance(sub, TimeSeries)
        assert sub.n == 2

    def test_getitem_bad_type_raises(self, ts_short):
        with pytest.raises(TypeError):
            ts_short["2020-01-01"]  # type: ignore[index]


# ===========================================================================
# Resampling
# ===========================================================================


class TestResample:
    def test_daily_to_monthly(self, ts_daily):
        ts_m = ts_daily.resample("MS")
        assert ts_m.n == 12

    def test_resample_sum(self, ts_daily):
        ts_w = ts_daily.resample("W", agg=AggMethod.SUM)
        # 365 days / 7 ≈ 52 full weeks + partial
        assert 52 <= ts_w.n <= 53

    def test_invalid_agg_raises(self, ts_daily):
        with pytest.raises(AttributeError):
            ts_daily.resample("W", agg="bad_agg")  # type: ignore[arg-type]

    def test_invalid_freq_raises(self, ts_daily):
        with pytest.raises(ValueError):
            ts_daily.resample("NOTFREQ")


# ===========================================================================
# Conversions
# ===========================================================================


class TestConversions:
    def test_to_series_name(self, ts_daily):
        s = ts_daily.to_series()
        assert isinstance(s, pd.Series)
        assert s.name == ts_daily.name

    def test_to_frame(self, ts_daily):
        df = ts_daily.to_frame()
        assert isinstance(df, pd.DataFrame)
        assert ts_daily.name in df.columns

    def test_to_numpy(self, ts_daily):
        arr = ts_daily.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (ts_daily.n,)

    def test_copy_independent(self, ts_daily):
        ts2 = ts_daily.copy()
        assert ts2 == ts_daily
        # Modifying the copy's underlying data must not affect the original
        ts2_vals = ts2.values
        ts2_vals[0] = -9999.0
        assert ts_daily.values[0] != -9999.0


# ===========================================================================
# Dunder methods
# ===========================================================================


class TestDunders:
    def test_len(self, ts_daily):
        assert len(ts_daily) == 365

    def test_contains_true(self, ts_daily):
        assert pd.Timestamp("2020-06-15") in ts_daily

    def test_contains_false(self, ts_daily):
        assert pd.Timestamp("2025-01-01") not in ts_daily

    def test_eq_same(self, ts_daily):
        ts2 = ts_daily.copy()
        assert ts_daily == ts2

    def test_eq_different(self, ts_daily, ts_hourly):
        assert ts_daily != ts_hourly

    def test_repr_contains_key_info(self, ts_daily):
        r = repr(ts_daily)
        assert "daily" in r
        assert "365" in r
        assert "D" in r

    def test_repr_shows_unit(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        ts  = TimeSeries([1, 2, 3], index=idx, unit="kg")
        assert "kg" in repr(ts)