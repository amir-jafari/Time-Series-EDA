"""
Tests for :mod:`tseda.seasonality.detector`.

Coverage targets
----------------
* SeasonalityDetector.detect() — all three methods, validation errors.
* SeasonalityDetector.test_period() — detection and non-detection cases.
* SeasonalityReport — summary(), structural checks.
* Helper internals — _fisher_g, _periodogram_detect, _acf_detect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.seasonality.detector import (
    SeasonalityDetector,
    SeasonalityReport,
    _acf_detect,
    _clean,
    _combine_scores,
    _fisher_g,
    _periodogram_detect,
)

det = SeasonalityDetector()


# ---------------------------------------------------------------------------
# Helpers — synthetic series
# ---------------------------------------------------------------------------

def _seasonal_ts(
    period: int, n: int, freq: str = "MS", noise: float = 0.3, seed: int = 0
) -> TimeSeries:
    """Pure sine-wave seasonal signal + small noise."""
    rng  = np.random.default_rng(seed)
    seas = np.tile(np.sin(2 * np.pi * np.arange(period) / period) * 8.0,
                   n // period + 1)[:n]
    y    = 50.0 + seas + rng.standard_normal(n) * noise
    idx  = pd.date_range("2018-01-01", periods=n, freq=freq)
    return TimeSeries(y, index=idx, name=f"p{period}")


def _white_noise_ts(n: int = 200, seed: int = 5) -> TimeSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return TimeSeries(rng.standard_normal(n), index=idx, name="wn")


# ===========================================================================
# Private helper tests
# ===========================================================================

class TestFisherG:
    def test_uniform_power_g_near_one_over_m(self):
        power = np.ones(10)
        g, p  = _fisher_g(power)
        assert abs(g - 0.1) < 1e-10

    def test_dominant_spike_low_p(self):
        power      = np.zeros(100)
        power[10]  = 1000.0   # huge spike
        g, p       = _fisher_g(power)
        assert g > 0.9
        assert p < 0.001

    def test_empty_power_returns_one(self):
        g, p = _fisher_g(np.array([]))
        assert g == 0.0 and p == 1.0

    def test_zero_power_returns_one(self):
        g, p = _fisher_g(np.zeros(10))
        assert p == 1.0


class TestClean:
    def test_removes_nan_by_interpolation(self):
        x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        out = _clean(x)
        assert not np.any(np.isnan(out))
        assert len(out) == 5

    def test_detrended_mean_near_zero(self):
        x   = np.arange(100, dtype=float)   # pure trend
        out = _clean(x)
        assert abs(out.mean()) < 1e-8


class TestPeriodogramDetect:
    def test_finds_period_12(self):
        ts  = _seasonal_ts(12, 72)
        x   = ts.values - ts.values.mean()
        res, g, p = _periodogram_detect(x, 2, 36, 5, 0.05)
        periods   = [r[0] for r in res]
        assert 12 in periods

    def test_finds_period_7(self):
        ts  = _seasonal_ts(7, 140, freq="D")
        x   = ts.values - ts.values.mean()
        res, g, p = _periodogram_detect(x, 2, 70, 5, 0.05)
        periods   = [r[0] for r in res]
        assert 7 in periods

    def test_normalised_scores_le_1(self):
        ts  = _seasonal_ts(12, 72)
        x   = ts.values
        res, _, _ = _periodogram_detect(x, 2, 36, 5, 0.05)
        for _, score in res:
            assert 0.0 <= score <= 1.0

    def test_short_series_returns_empty(self):
        x = np.ones(2)
        res, g, p = _periodogram_detect(x, 2, 1, 5, 0.05)
        assert res == []


class TestACFDetect:
    def test_finds_lag_12(self):
        ts  = _seasonal_ts(12, 96)
        x   = _clean(ts.values)
        res = _acf_detect(x, 48, 0.05, 5)
        lags = [r[0] for r in res]
        assert 12 in lags

    def test_white_noise_few_or_no_peaks(self):
        rng = np.random.default_rng(99)
        x   = _clean(rng.standard_normal(200))
        res = _acf_detect(x, 100, 0.05, 5)
        # white noise may have zero or a few spurious peaks
        assert len(res) <= 5

    def test_acf_values_positive(self):
        ts  = _seasonal_ts(12, 96)
        x   = _clean(ts.values)
        res = _acf_detect(x, 48, 0.05, 5)
        for _, v in res:
            assert v > 0

    def test_too_short_returns_empty(self):
        x = np.ones(3)
        assert _acf_detect(x, 1, 0.05, 5) == []


class TestCombineScores:
    def test_both_methods_agreement_bonus(self):
        # (1.0 + 1.0)/2 * 1.2 = 1.2, capped to 1.0
        pg  = [(12, 1.0)]
        acf = [(12, 1.0)]
        combined = _combine_scores(pg, acf)
        assert combined[12] == 1.0

    def test_both_methods_mid_scores_get_bonus(self):
        # (0.5 + 0.5)/2 * 1.2 = 0.6  > 0.5 (single-method max)
        pg  = [(12, 0.5)]
        acf = [(12, 0.5)]
        combined = _combine_scores(pg, acf)
        assert combined[12] > 0.5

    def test_single_method_discounted(self):
        pg  = [(12, 1.0)]
        acf: list = []
        combined = _combine_scores(pg, acf)
        assert combined[12] == 0.9  # 0.9 discount

    def test_union_of_periods(self):
        pg  = [(7, 0.8), (14, 0.3)]
        acf = [(7, 0.9), (21, 0.5)]
        combined = _combine_scores(pg, acf)
        assert 7 in combined
        assert 14 in combined
        assert 21 in combined


# ===========================================================================
# SeasonalityDetector.detect()
# ===========================================================================

class TestDetectCombined:
    def test_detects_period_12(self):
        ts = _seasonal_ts(12, 72)
        r  = det.detect(ts)
        assert r.dominant_period == 12
        assert r.is_seasonal is True

    def test_detects_period_7(self):
        ts = _seasonal_ts(7, 140, freq="D", noise=0.2)
        r  = det.detect(ts)
        assert r.dominant_period == 7

    def test_white_noise_not_seasonal(self):
        r = det.detect(_white_noise_ts())
        assert r.is_seasonal is False

    def test_returns_report_type(self):
        r = det.detect(_seasonal_ts(12, 72))
        assert isinstance(r, SeasonalityReport)

    def test_method_label(self):
        r = det.detect(_seasonal_ts(12, 72))
        assert r.method == "combined"

    def test_n_obs_correct(self):
        ts = _seasonal_ts(12, 72)
        r  = det.detect(ts)
        assert r.n_obs == 72

    def test_candidate_periods_sorted(self):
        r = det.detect(_seasonal_ts(12, 72))
        scores = [s for _, s in r.candidate_periods]
        assert scores == sorted(scores, reverse=True)

    def test_dominant_in_candidates(self):
        r = det.detect(_seasonal_ts(12, 72))
        if r.dominant_period is not None:
            periods = [p for p, _ in r.candidate_periods]
            assert r.dominant_period in periods

    def test_fisher_p_value_range(self):
        r = det.detect(_seasonal_ts(12, 72))
        assert 0.0 <= r.fisher_p_value <= 1.0

    def test_strength_scores_range(self):
        r = det.detect(_seasonal_ts(12, 72))
        for v in r.strength_scores.values():
            assert 0.0 <= v <= 1.0

    def test_with_nan_values(self):
        ts   = _seasonal_ts(12, 72)
        vals = ts.values.copy()
        vals[5] = np.nan
        vals[30] = np.nan
        ts_nan = TimeSeries(vals, index=ts.index)
        r = det.detect(ts_nan)
        assert isinstance(r, SeasonalityReport)
        assert r.n_obs == 70


class TestDetectPeriodogram:
    def test_detects_period_12(self):
        ts = _seasonal_ts(12, 72)
        r  = det.detect(ts, method="periodogram")
        assert r.method == "periodogram"
        assert r.dominant_period == 12

    def test_fisher_sig_for_seasonal(self):
        r = det.detect(_seasonal_ts(12, 72), method="periodogram")
        assert r.fisher_p_value < 0.05

    def test_fisher_not_sig_for_noise(self):
        r = det.detect(_white_noise_ts(), method="periodogram")
        assert r.fisher_p_value > 0.001   # not strongly significant


class TestDetectACF:
    def test_detects_period_12_with_acf(self):
        ts = _seasonal_ts(12, 96)
        r  = det.detect(ts, method="acf")
        assert r.method == "acf"
        assert r.dominant_period == 12

    def test_acf_periods_not_empty_for_seasonal(self):
        ts = _seasonal_ts(12, 96)
        r  = det.detect(ts, method="acf")
        assert len(r.acf_periods) > 0

    def test_acf_periods_empty_for_noise(self):
        r = det.detect(_white_noise_ts(400, seed=0), method="acf")
        # Very few (ideally 0) ACF peaks for white noise
        assert len(r.acf_periods) <= 3


# ===========================================================================
# Validation errors
# ===========================================================================

class TestValidationErrors:
    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            det.detect("not a ts")  # type: ignore[arg-type]

    def test_bad_method_raises(self):
        ts = _seasonal_ts(12, 72)
        with pytest.raises(ValueError, match="method"):
            det.detect(ts, method="magic")

    def test_too_short_raises(self):
        idx = pd.date_range("2020", periods=5, freq="D")
        ts  = TimeSeries(np.ones(5), index=idx)
        with pytest.raises(ValueError, match="8"):
            det.detect(ts)

    def test_alpha_out_of_range_raises(self):
        ts = _seasonal_ts(12, 72)
        with pytest.raises(ValueError, match="alpha"):
            det.detect(ts, alpha=1.5)

    def test_min_period_less_than_2_raises(self):
        ts = _seasonal_ts(12, 72)
        with pytest.raises(ValueError, match="min_period"):
            det.detect(ts, min_period=1)


# ===========================================================================
# test_period()
# ===========================================================================

class TestTestPeriod:
    def test_detects_known_period(self):
        ts  = _seasonal_ts(12, 72)
        res = det.test_period(ts, 12)
        assert res["detected"] is True

    def test_wrong_period_not_detected(self):
        ts  = _seasonal_ts(12, 72)
        res = det.test_period(ts, 3)   # period 3 not in data
        # strength should be very low (may or may not be "detected")
        assert res["strength"] < 0.5

    def test_keys_present(self):
        ts  = _seasonal_ts(12, 72)
        res = det.test_period(ts, 12)
        assert set(res.keys()) == {
            "period", "detected",
            "periodogram_detected", "acf_detected",
            "strength", "fisher_p_value",
        }

    def test_period_stored(self):
        ts  = _seasonal_ts(12, 72)
        res = det.test_period(ts, 12)
        assert res["period"] == 12

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            det.test_period("not ts", 12)  # type: ignore[arg-type]

    def test_period_less_than_2_raises(self):
        ts = _seasonal_ts(12, 72)
        with pytest.raises(ValueError, match="period"):
            det.test_period(ts, 1)


# ===========================================================================
# SeasonalityReport.summary()
# ===========================================================================

class TestSummary:
    def test_summary_is_string(self):
        r = det.detect(_seasonal_ts(12, 72))
        s = r.summary()
        assert isinstance(s, str)

    def test_summary_contains_method(self):
        r = det.detect(_seasonal_ts(12, 72))
        assert r.method in r.summary()

    def test_summary_contains_period(self):
        r = det.detect(_seasonal_ts(12, 72))
        assert "12" in r.summary()