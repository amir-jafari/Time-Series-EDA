"""
Tests for :mod:`tseda.forecastability.scorer`.

Coverage targets
----------------
* ForecastabilityScorer.score()          — return type, score bounds, sub-scores
* Sub-score correctness                  — data_quality, stationarity, etc.
* recommended_model logic                — all five model labels
* recommended_diff                       — 0 vs 1
* recommended_period                     — auto-detect vs explicit
* Edge cases                             — minimal series, all-NaN, period arg
* Input validation                       — wrong type, period < 2, too few obs
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.forecastability.scorer import (
    ForecastabilityReport,
    ForecastabilityScorer,
    _pct_outlier_iqr,
    _recommend_model,
    _has_large_gaps,
)

scorer = ForecastabilityScorer()


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_ts(
    n: int = 200,
    *,
    freq: str = "D",
    seed: int = 0,
    noise: float = 1.0,
    trend: float = 0.0,
    seasonal_period: int = 0,
    seasonal_amp: float = 0.0,
    nan_frac: float = 0.0,
) -> TimeSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    t = np.arange(n, dtype=float)
    vals = rng.standard_normal(n) * noise + trend * t
    if seasonal_period > 0:
        vals += np.sin(2 * np.pi * t / seasonal_period) * seasonal_amp
    if nan_frac > 0:
        nan_idx = rng.choice(n, size=int(n * nan_frac), replace=False)
        vals[nan_idx] = np.nan
    return TimeSeries(vals, index=idx)


def _ar1_ts(n: int = 300, phi: float = 0.8, seed: int = 0) -> TimeSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020", periods=n, freq="D")
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return TimeSeries(x, index=idx)


# ===========================================================================
# Return type & structure
# ===========================================================================


class TestReturnType:
    def test_returns_report(self):
        ts = _make_ts()
        r = scorer.score(ts)
        assert isinstance(r, ForecastabilityReport)

    def test_score_in_range(self):
        ts = _make_ts()
        r = scorer.score(ts)
        assert 0.0 <= r.score <= 100.0

    def test_sub_scores_keys(self):
        ts = _make_ts()
        r = scorer.score(ts)
        expected = {
            "data_quality", "stationarity", "signal_to_noise",
            "autocorrelation", "sample_size", "regularity",
        }
        assert set(r.sub_scores.keys()) == expected

    def test_sub_scores_in_range(self):
        ts = _make_ts()
        r = scorer.score(ts)
        for k, v in r.sub_scores.items():
            assert 0.0 <= v <= 100.0, f"sub-score {k}={v} out of [0, 100]"

    def test_n_obs(self):
        ts = _make_ts(n=150)
        r = scorer.score(ts)
        assert r.n_obs == 150

    def test_recommended_model_valid(self):
        ts = _make_ts()
        r = scorer.score(ts)
        assert r.recommended_model in ("ARIMA", "SARIMA", "ETS", "Prophet", "ML")

    def test_recommended_diff_binary(self):
        ts = _make_ts()
        r = scorer.score(ts)
        assert r.recommended_diff in (0, 1)


# ===========================================================================
# Sub-score: data_quality
# ===========================================================================


class TestDataQuality:
    def test_clean_series_high_score(self):
        ts = _make_ts(nan_frac=0.0)
        r = scorer.score(ts)
        assert r.sub_scores["data_quality"] >= 80.0

    def test_many_nans_lower_score(self):
        ts_clean = _make_ts(nan_frac=0.0)
        ts_nan = _make_ts(nan_frac=0.30)
        r_clean = scorer.score(ts_clean)
        r_nan = scorer.score(ts_nan)
        assert r_clean.sub_scores["data_quality"] > r_nan.sub_scores["data_quality"]

    def test_pct_missing_stored(self):
        ts = _make_ts(n=100, nan_frac=0.10)
        r = scorer.score(ts)
        assert 8.0 <= r.pct_missing <= 15.0


class TestPctOutlierHelper:
    def test_no_outliers(self):
        x = np.arange(100, dtype=float)
        assert _pct_outlier_iqr(x) == pytest.approx(0.0)

    def test_one_outlier(self):
        x = np.concatenate([np.arange(99, dtype=float), [1000.0]])
        pct = _pct_outlier_iqr(x)
        assert pct > 0.0

    def test_empty_array(self):
        assert _pct_outlier_iqr(np.array([])) == 0.0

    def test_constant_array(self):
        assert _pct_outlier_iqr(np.ones(50)) == 0.0


# ===========================================================================
# Sub-score: stationarity
# ===========================================================================


class TestStationarity:
    def test_white_noise_stationary(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2020", periods=300, freq="D")
        ts = TimeSeries(rng.standard_normal(300), index=idx)
        r = scorer.score(ts)
        assert r.is_stationary is True
        assert r.recommended_diff == 0

    def test_random_walk_nonstationary(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2020", periods=300, freq="D")
        ts = TimeSeries(np.cumsum(rng.standard_normal(300)), index=idx)
        r = scorer.score(ts)
        assert r.recommended_diff == 1

    def test_stationary_score_higher_for_stationary(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2020", periods=300, freq="D")
        ts_stat = TimeSeries(rng.standard_normal(300), index=idx)
        ts_rw = TimeSeries(np.cumsum(rng.standard_normal(300)), index=idx)
        r_stat = scorer.score(ts_stat)
        r_rw = scorer.score(ts_rw)
        assert r_stat.sub_scores["stationarity"] >= r_rw.sub_scores["stationarity"]


# ===========================================================================
# Sub-score: signal_to_noise
# ===========================================================================


class TestSignalToNoise:
    def test_seasonal_signal_higher(self):
        n = 200
        idx = pd.date_range("2020", periods=n, freq="D")
        t = np.arange(n)
        rng = np.random.default_rng(0)
        seas = np.sin(2 * np.pi * t / 7) * 5
        ts_seasonal = TimeSeries(seas + rng.standard_normal(n) * 0.3, index=idx)
        ts_noise = TimeSeries(rng.standard_normal(n), index=idx)
        r_seas = scorer.score(ts_seasonal, period=7)
        r_noise = scorer.score(ts_noise)
        assert r_seas.sub_scores["signal_to_noise"] > r_noise.sub_scores["signal_to_noise"]


# ===========================================================================
# Sub-score: autocorrelation
# ===========================================================================


class TestAutocorrelation:
    def test_ar1_has_nonzero_ac_score(self):
        ts = _ar1_ts(phi=0.8)
        r = scorer.score(ts)
        assert r.sub_scores["autocorrelation"] > 0.0

    def test_white_noise_low_ac_score(self):
        rng = np.random.default_rng(10)
        idx = pd.date_range("2020", periods=200, freq="D")
        ts = TimeSeries(rng.standard_normal(200), index=idx)
        r = scorer.score(ts)
        assert r.sub_scores["autocorrelation"] < 50.0


# ===========================================================================
# Sub-score: sample_size
# ===========================================================================


class TestSampleSize:
    def test_longer_series_higher_score(self):
        ts_short = _make_ts(n=50)
        ts_long = _make_ts(n=500)
        r_short = scorer.score(ts_short)
        r_long = scorer.score(ts_long)
        assert r_long.sub_scores["sample_size"] >= r_short.sub_scores["sample_size"]

    def test_explicit_period_used_in_score(self):
        ts = _make_ts(n=200)
        r_short_period = scorer.score(ts, period=2)
        r_long_period = scorer.score(ts, period=50)
        assert r_short_period.sub_scores["sample_size"] > r_long_period.sub_scores["sample_size"]


# ===========================================================================
# Sub-score: regularity
# ===========================================================================


class TestRegularity:
    def test_regular_ts_full_regularity_score(self):
        ts = _make_ts(n=100)
        assert ts.is_regular
        r = scorer.score(ts)
        assert r.sub_scores["regularity"] >= 50.0

    def test_irregular_ts_lower_score(self):
        rng = np.random.default_rng(0)
        base = pd.Timestamp("2020-01-01")
        offsets = np.cumsum(rng.integers(1, 48, size=100))
        idx = pd.DatetimeIndex([base + pd.Timedelta(hours=int(h)) for h in offsets])
        ts = TimeSeries(rng.standard_normal(100), index=idx)
        r = scorer.score(ts)
        assert r.sub_scores["regularity"] <= 100.0


class TestHasLargeGapsHelper:
    def test_regular_no_gaps(self):
        idx = pd.date_range("2020", periods=50, freq="D")
        ts = TimeSeries(np.ones(50), index=idx)
        assert _has_large_gaps(ts) is False

    def test_large_gap_detected(self):
        idx1 = pd.date_range("2020-01-01", periods=25, freq="D")
        idx2 = pd.date_range("2020-06-01", periods=25, freq="D")
        idx = idx1.append(idx2)
        ts = TimeSeries(np.ones(50), index=idx)
        assert _has_large_gaps(ts) is True

    def test_short_series_no_gap(self):
        idx = pd.date_range("2020", periods=2, freq="D")
        ts = TimeSeries(np.array([1.0, 2.0]), index=idx)
        assert _has_large_gaps(ts) is False


# ===========================================================================
# Recommended model logic
# ===========================================================================


class TestRecommendModel:
    def test_sarima_when_seasonal_stationary(self):
        assert _recommend_model(
            is_seasonal=True, is_stationary=True, n_obs=200, overall_score=70.0
        ) == "SARIMA"

    def test_ets_when_seasonal_nonstationary(self):
        assert _recommend_model(
            is_seasonal=True, is_stationary=False, n_obs=200, overall_score=60.0
        ) == "ETS"

    def test_arima_when_no_seasonal_stationary(self):
        assert _recommend_model(
            is_seasonal=False, is_stationary=True, n_obs=100, overall_score=55.0
        ) == "ARIMA"

    def test_prophet_when_long_non_seasonal_nonstationary(self):
        assert _recommend_model(
            is_seasonal=False, is_stationary=False, n_obs=300, overall_score=40.0
        ) == "Prophet"

    def test_ml_when_very_low_score(self):
        assert _recommend_model(
            is_seasonal=False, is_stationary=False, n_obs=100, overall_score=15.0
        ) == "ML"


# ===========================================================================
# Period parameter
# ===========================================================================


class TestPeriodParameter:
    def test_explicit_period_stored(self):
        ts = _make_ts(n=300, seasonal_period=7, seasonal_amp=3.0)
        r = scorer.score(ts, period=7)
        assert r.recommended_period == 7
        assert r.dominant_period == 7

    def test_no_period_returns_none_or_int(self):
        ts = _make_ts(n=200)
        r = scorer.score(ts)
        assert r.recommended_period is None or isinstance(r.recommended_period, int)


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="TimeSeries"):
            scorer.score([1, 2, 3])

    def test_too_few_obs_raises(self):
        idx = pd.date_range("2020", periods=3, freq="D")
        ts = TimeSeries([1.0, 2.0, 3.0], index=idx)
        with pytest.raises(ValueError, match="4 observations"):
            scorer.score(ts)

    def test_period_less_than_2_raises(self):
        ts = _make_ts()
        with pytest.raises(ValueError, match="period"):
            scorer.score(ts, period=1)

    def test_period_zero_raises(self):
        ts = _make_ts()
        with pytest.raises(ValueError):
            scorer.score(ts, period=0)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_minimal_series(self):
        idx = pd.date_range("2020", periods=20, freq="D")
        ts = TimeSeries(np.arange(20, dtype=float), index=idx)
        r = scorer.score(ts)
        assert 0.0 <= r.score <= 100.0

    def test_series_with_nans_does_not_crash(self):
        ts = _make_ts(n=100, nan_frac=0.15)
        r = scorer.score(ts)
        assert isinstance(r, ForecastabilityReport)

    def test_constant_series(self):
        idx = pd.date_range("2020", periods=50, freq="D")
        ts = TimeSeries(np.ones(50), index=idx)
        r = scorer.score(ts)
        assert isinstance(r, ForecastabilityReport)

    def test_monthly_series(self):
        idx = pd.date_range("2018-01", periods=60, freq="MS")
        rng = np.random.default_rng(0)
        seas = np.tile(np.sin(2 * np.pi * np.arange(12) / 12) * 5, 5)
        ts = TimeSeries(seas + rng.standard_normal(60), index=idx)
        r = scorer.score(ts, period=12)
        assert r.recommended_period == 12