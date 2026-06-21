"""Tests for :mod:`tseda.statistics.autocorrelation`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.statistics.autocorrelation import AutocorrelationAnalyzer, AutocorrelationResult

ana = AutocorrelationAnalyzer()


def _white_noise(n: int = 200, seed: int = 0) -> TimeSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020", periods=n, freq="D")
    return TimeSeries(rng.standard_normal(n), index=idx)


def _ar1(phi: float = 0.7, n: int = 300, seed: int = 7) -> TimeSeries:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x   = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    idx = pd.date_range("2020", periods=n, freq="D")
    return TimeSeries(x, index=idx)


class TestAnalyze:
    def test_returns_result(self):
        r = ana.analyze(_white_noise())
        assert isinstance(r, AutocorrelationResult)

    def test_acf_lag0_is_one(self):
        r = ana.analyze(_white_noise())
        assert r.acf[0] == 1.0

    def test_pacf_lag0_is_one(self):
        r = ana.analyze(_white_noise())
        assert r.pacf[0] == 1.0

    def test_acf_length(self):
        r = ana.analyze(_white_noise(), lags=20)
        assert len(r.acf) == 21   # lags 0..20

    def test_pacf_length(self):
        r = ana.analyze(_white_noise(), lags=15)
        assert len(r.pacf) == 16

    def test_lags_array(self):
        r = ana.analyze(_white_noise(), lags=10)
        np.testing.assert_array_equal(r.lags, np.arange(11))

    def test_conf_bounds_symmetric(self):
        r = ana.analyze(_white_noise())
        np.testing.assert_allclose(r.conf_lower[1:], -r.conf_upper[1:])

    def test_lb_length(self):
        r = ana.analyze(_white_noise(), lags=10)
        assert len(r.lb_statistic) == 10
        assert len(r.lb_pvalue)    == 10

    def test_white_noise_is_wn(self):
        # seed=0 gives LB-20 p≈0.62 — reliably above any common α
        r = ana.analyze(_white_noise(500, seed=0), lags=20)
        assert r.is_white_noise is True

    def test_ar1_not_white_noise(self):
        r = ana.analyze(_ar1(), lags=20)
        assert r.is_white_noise is False

    def test_ar1_strong_lag1(self):
        r = ana.analyze(_ar1(phi=0.8), lags=10)
        assert r.acf[1] > 0.5

    def test_lb_pvalue_range(self):
        r = ana.analyze(_white_noise())
        assert np.all((r.lb_pvalue >= 0.0) & (r.lb_pvalue <= 1.0))

    def test_too_few_obs_raises(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        ts  = TimeSeries([1.0, 2.0, 3.0], index=idx)
        with pytest.raises(ValueError, match="4"):
            ana.analyze(ts)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            ana.analyze("not a ts")  # type: ignore[arg-type]

    def test_nan_dropped(self, ts_with_nan):
        r = ana.analyze(ts_with_nan, lags=10)
        assert r.n_obs == ts_with_nan.n - ts_with_nan.n_nan

    def test_lags_capped_at_half_n(self):
        idx = pd.date_range("2020", periods=20, freq="D")
        ts  = TimeSeries(np.arange(20.0), index=idx)
        with pytest.raises(ValueError, match="n // 2"):
            ana.analyze(ts, lags=15)


class TestSignificantLags:
    def test_returns_array(self):
        r    = ana.analyze(_white_noise(), lags=20)
        lags = ana.significant_lags(r)
        assert isinstance(lags, np.ndarray)

    def test_ar1_has_significant_acf_lags(self):
        r    = ana.analyze(_ar1(), lags=20)
        lags = ana.significant_lags(r, which="acf")
        assert len(lags) > 0

    def test_white_noise_few_significant(self):
        r    = ana.analyze(_white_noise(400, seed=3), lags=20)
        lags = ana.significant_lags(r, which="acf")
        # By chance ≤ 5 % of lags (≤1 out of 20) may be significant
        assert len(lags) <= 3

    def test_invalid_which_raises(self):
        r = ana.analyze(_white_noise(), lags=10)
        with pytest.raises(ValueError, match="which"):
            ana.significant_lags(r, which="bad")

    def test_bad_result_type_raises(self):
        with pytest.raises(TypeError):
            ana.significant_lags("not a result")  # type: ignore[arg-type]

    def test_pacf_ar1_lag1_significant(self):
        r    = ana.analyze(_ar1(phi=0.8, n=300), lags=15)
        lags = ana.significant_lags(r, which="pacf")
        assert 1 in lags
