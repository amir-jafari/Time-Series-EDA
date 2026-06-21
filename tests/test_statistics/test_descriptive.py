"""Tests for :mod:`tseda.statistics.descriptive`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.statistics.descriptive import DescriptiveAnalyzer, DescriptiveStats

ana = DescriptiveAnalyzer()


def _ts(vals: list[float]) -> TimeSeries:
    idx = pd.date_range("2020", periods=len(vals), freq="D")
    return TimeSeries(np.array(vals, dtype=float), index=idx)


class TestDescriptiveAnalyze:
    def test_returns_report_type(self, ts_daily):
        r = ana.analyze(ts_daily)
        assert isinstance(r, DescriptiveStats)

    def test_n_total(self, ts_daily):
        r = ana.analyze(ts_daily)
        assert r.n_total == 365

    def test_n_valid_no_nan(self, ts_daily):
        r = ana.analyze(ts_daily)
        assert r.n_valid == 365
        assert r.n_nan == 0
        assert r.pct_nan == 0.0

    def test_n_nan_counted(self, ts_with_nan):
        r = ana.analyze(ts_with_nan)
        assert r.n_nan == 20
        assert abs(r.pct_nan - 10.0) < 0.1

    def test_mean_known(self):
        r = ana.analyze(_ts([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert abs(r.mean - 3.0) < 1e-12

    def test_median_known(self):
        r = ana.analyze(_ts([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert abs(r.median - 3.0) < 1e-12

    def test_std_known(self):
        r = ana.analyze(_ts([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]))
        assert abs(r.std - 2.138089935299395) < 1e-8

    def test_min_max(self):
        r = ana.analyze(_ts([3.0, 1.0, 4.0, 1.0, 5.0]))
        assert r.min == 1.0
        assert r.max == 5.0
        assert r.range == 4.0

    def test_first_last(self):
        r = ana.analyze(_ts([10.0, 20.0, 30.0]))
        assert r.first == 10.0
        assert r.last == 30.0

    def test_first_last_skips_nan(self):
        idx  = pd.date_range("2020", periods=5, freq="D")
        vals = np.array([np.nan, 2.0, 3.0, 4.0, 5.0])
        ts   = TimeSeries(vals, index=idx)
        r    = ana.analyze(ts)
        assert r.first == 2.0
        assert r.last  == 5.0

    def test_cv_nonzero_mean(self):
        r = ana.analyze(_ts([10.0, 12.0, 8.0, 11.0, 9.0]))
        assert r.cv > 0

    def test_cv_zero_mean(self):
        r = ana.analyze(_ts([-1.0, 0.0, 1.0]))
        assert np.isnan(r.cv)

    def test_skewness_symmetric(self):
        r = ana.analyze(_ts([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert abs(r.skewness) < 0.05   # symmetric → near 0

    def test_kurtosis_normal_approx(self):
        rng  = np.random.default_rng(0)
        idx  = pd.date_range("2020", periods=1000, freq="D")
        ts   = TimeSeries(rng.standard_normal(1000), index=idx)
        r    = ana.analyze(ts)
        assert abs(r.kurtosis) < 0.5    # normal → excess kurtosis ≈ 0

    def test_quantiles_present(self, ts_daily):
        r = ana.analyze(ts_daily)
        assert 0.25 in r.quantiles
        assert 0.50 in r.quantiles
        assert 0.75 in r.quantiles
        assert r.quantiles[0.25] < r.quantiles[0.75]

    def test_n_zeros(self):
        r = ana.analyze(_ts([0.0, 1.0, 0.0, 2.0, 0.0]))
        assert r.n_zeros == 3

    def test_n_positive_negative(self):
        r = ana.analyze(_ts([-1.0, 0.0, 1.0, 2.0, -3.0]))
        assert r.n_positive == 2
        assert r.n_negative == 2
        assert r.n_zeros == 1

    def test_mad_constant(self):
        r = ana.analyze(_ts([5.0, 5.0, 5.0, 5.0]))
        assert r.mad == 0.0

    def test_trimmed_mean_close_to_mean_for_normal(self):
        rng  = np.random.default_rng(1)
        idx  = pd.date_range("2020", periods=500, freq="D")
        ts   = TimeSeries(rng.standard_normal(500), index=idx)
        r    = ana.analyze(ts)
        assert abs(r.trimmed_mean - r.mean) < 0.15

    def test_all_nan_raises(self):
        idx  = pd.date_range("2020", periods=3, freq="D")
        ts   = TimeSeries([np.nan, np.nan, np.nan], index=idx)
        with pytest.raises(ValueError, match="non-NaN"):
            ana.analyze(ts)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            ana.analyze("not a ts")  # type: ignore[arg-type]