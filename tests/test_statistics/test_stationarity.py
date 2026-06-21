"""Tests for :mod:`tseda.statistics.stationarity`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.statistics.stationarity import StationarityResult, StationarityTester

tester = StationarityTester()


def _white_noise(n: int = 200, seed: int = 0) -> TimeSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020", periods=n, freq="D")
    return TimeSeries(rng.standard_normal(n), index=idx)


def _random_walk(n: int = 200, seed: int = 0) -> TimeSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020", periods=n, freq="D")
    return TimeSeries(np.cumsum(rng.standard_normal(n)), index=idx)


class TestADF:
    def test_returns_result(self):
        r = tester.adf(_white_noise())
        assert isinstance(r, StationarityResult)

    def test_white_noise_is_stationary(self):
        r = tester.adf(_white_noise(300, seed=42))
        assert r.is_stationary is True

    def test_random_walk_not_stationary(self):
        r = tester.adf(_random_walk(300, seed=0))
        assert r.is_stationary is False

    def test_test_name(self):
        assert tester.adf(_white_noise()).test_name == "ADF"

    def test_critical_values_present(self):
        r = tester.adf(_white_noise())
        assert "5%" in r.critical_values

    def test_p_value_range(self):
        r = tester.adf(_white_noise())
        assert 0.0 <= r.p_value <= 1.0

    def test_regression_ct(self):
        r = tester.adf(_white_noise(), regression="ct")
        assert r.regression == "ct"

    def test_invalid_regression_raises(self):
        with pytest.raises(ValueError, match="regression"):
            tester.adf(_white_noise(), regression="bad")

    def test_too_few_obs_raises(self):
        idx = pd.date_range("2020", periods=5, freq="D")
        ts  = TimeSeries(np.random.randn(5), index=idx)
        with pytest.raises(ValueError, match="10"):
            tester.adf(ts)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            tester.adf("not a ts")  # type: ignore[arg-type]

    def test_n_lags_not_negative(self):
        r = tester.adf(_white_noise())
        assert r.n_lags is None or r.n_lags >= 0


class TestKPSS:
    def test_returns_result(self):
        r = tester.kpss(_white_noise())
        assert isinstance(r, StationarityResult)

    def test_white_noise_is_stationary(self):
        r = tester.kpss(_white_noise(300, seed=10))
        assert r.is_stationary is True

    def test_random_walk_not_stationary(self):
        r = tester.kpss(_random_walk(300, seed=5))
        assert r.is_stationary is False

    def test_test_name(self):
        assert tester.kpss(_white_noise()).test_name == "KPSS"

    def test_regression_ct(self):
        r = tester.kpss(_white_noise(), regression="ct")
        assert r.regression == "ct"

    def test_invalid_regression_raises(self):
        with pytest.raises(ValueError, match="regression"):
            tester.kpss(_white_noise(), regression="nc")

    def test_p_value_range(self):
        r = tester.kpss(_white_noise())
        assert 0.0 <= r.p_value <= 1.0

    def test_statistic_positive(self):
        r = tester.kpss(_white_noise())
        assert r.statistic >= 0.0


class TestSummary:
    def test_returns_string(self):
        s = tester.summary(_white_noise(200))
        assert isinstance(s, str)

    def test_contains_adf_and_kpss(self):
        s = tester.summary(_white_noise(200))
        assert "ADF" in s
        assert "KPSS" in s

    def test_stationary_verdict_for_white_noise(self):
        s = tester.summary(_white_noise(300, seed=42))
        assert "STATIONARY" in s

    def test_non_stationary_for_random_walk(self):
        s = tester.summary(_random_walk(300, seed=0))
        assert "NON-STATIONARY" in s or "DIFFERENCE" in s or "TREND" in s