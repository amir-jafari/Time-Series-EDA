"""Tests for :mod:`tseda.decomposition.classical`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.decomposition.classical import (
    ClassicalDecomposer,
    DecompositionResult,
    _centered_trend,
    _strength,
)

dec = ClassicalDecomposer()


# ---------------------------------------------------------------------------
# Test series builders
# ---------------------------------------------------------------------------

def _additive_ts(n: int = 60, period: int = 12, seed: int = 0) -> TimeSeries:
    """Clean synthetic additive series: trend + seasonal + small noise."""
    rng  = np.random.default_rng(seed)
    t    = np.arange(n, dtype=float)
    seas = np.tile(np.sin(2 * np.pi * np.arange(period) / period) * 5.0, n // period + 1)[:n]
    y    = 100.0 + 0.4 * t + seas + rng.standard_normal(n) * 0.3
    idx  = pd.date_range("2020-01", periods=n, freq="MS")
    return TimeSeries(y, index=idx, name="sales")


def _multiplicative_ts(n: int = 48, period: int = 12, seed: int = 1) -> TimeSeries:
    """Positive synthetic multiplicative series."""
    rng  = np.random.default_rng(seed)
    t    = np.arange(n, dtype=float)
    seas = np.tile(np.exp(np.sin(2 * np.pi * np.arange(period) / period) * 0.3), n // period + 1)[:n]
    y    = (50.0 + 0.5 * t) * seas * (1 + rng.standard_normal(n) * 0.02)
    idx  = pd.date_range("2020-01", periods=n, freq="MS")
    return TimeSeries(y, index=idx, name="revenue")


# ===========================================================================
# DecompositionResult
# ===========================================================================

class TestDecompositionResult:
    def test_to_dataframe_columns(self):
        ts = _additive_ts()
        r  = dec.decompose(ts, period=12)
        df = r.to_dataframe()
        assert list(df.columns) == ["observed", "trend", "seasonal", "residual"]

    def test_to_dataframe_length(self):
        ts = _additive_ts()
        r  = dec.decompose(ts, period=12)
        assert len(r.to_dataframe()) == ts.n

    def test_summary_is_string(self):
        r = dec.decompose(_additive_ts(), period=12)
        s = r.summary()
        assert isinstance(s, str)
        assert "classical" in s

    def test_method_and_model(self):
        r = dec.decompose(_additive_ts(), period=12)
        assert r.method == "classical"
        assert r.model  == "additive"

    def test_period_stored(self):
        r = dec.decompose(_additive_ts(), period=12)
        assert r.period == 12


# ===========================================================================
# Classical — additive
# ===========================================================================

class TestClassicalAdditive:
    def test_returns_result_type(self):
        r = dec.decompose(_additive_ts(), period=12)
        assert isinstance(r, DecompositionResult)

    def test_component_lengths(self):
        ts = _additive_ts()
        r  = dec.decompose(ts, period=12)
        assert r.trend.n    == ts.n
        assert r.seasonal.n == ts.n
        assert r.residual.n == ts.n

    def test_reconstruction_identity(self):
        """T + S + R == y everywhere trend is not NaN."""
        ts = _additive_ts()
        r  = dec.decompose(ts, period=12)
        T  = r.trend.values
        S  = r.seasonal.values
        R  = r.residual.values
        y  = ts.values
        valid = ~np.isnan(T)
        np.testing.assert_allclose(T[valid] + S[valid] + R[valid], y[valid], rtol=1e-8)

    def test_trend_nan_at_edges(self):
        """Odd period: NaN in first and last period//2 positions."""
        ts = _additive_ts(period=7, n=56)
        r  = dec.decompose(ts, period=7)
        assert np.isnan(r.trend.values[0])
        assert np.isnan(r.trend.values[-1])

    def test_even_period_trend_nan_at_edges(self):
        ts = _additive_ts()
        r  = dec.decompose(ts, period=12)
        # Even period 12 → NaN in first and last 6 positions
        assert np.isnan(r.trend.values[:6]).all()
        assert np.isnan(r.trend.values[-6:]).all()

    def test_seasonal_no_nan(self):
        r = dec.decompose(_additive_ts(), period=12)
        assert not r.seasonal.has_nan

    def test_seasonal_period_pattern(self):
        """Values 12 positions apart should be the same."""
        r = dec.decompose(_additive_ts(), period=12)
        S = r.seasonal.values
        np.testing.assert_allclose(S[0], S[12], atol=1e-10)
        np.testing.assert_allclose(S[6], S[18], atol=1e-10)

    def test_seasonal_sums_to_zero(self):
        r = dec.decompose(_additive_ts(), period=12)
        np.testing.assert_allclose(np.sum(r.seasonal.values[:12]), 0.0, atol=1e-10)

    def test_strength_seasonal_high_for_seasonal_data(self):
        r = dec.decompose(_additive_ts(), period=12)
        assert r.strength_seasonal > 0.5

    def test_strength_trend_high_for_trending_data(self):
        r = dec.decompose(_additive_ts(), period=12)
        assert r.strength_trend > 0.5

    def test_strength_in_range(self):
        r = dec.decompose(_additive_ts(), period=12)
        assert 0.0 <= r.strength_trend    <= 1.0
        assert 0.0 <= r.strength_seasonal <= 1.0

    def test_freq_inferred_monthly(self):
        ts = _additive_ts()   # MS freq → period 12 inferred
        r  = dec.decompose(ts)
        assert r.period == 12

    def test_n_obs_used(self):
        r = dec.decompose(_additive_ts(), period=12)
        assert r.n_obs_used > 0
        assert r.n_obs_used <= r.original.n

    def test_original_unchanged(self):
        ts = _additive_ts()
        r  = dec.decompose(ts, period=12)
        assert r.original is ts


# ===========================================================================
# Classical — multiplicative
# ===========================================================================

class TestClassicalMultiplicative:
    def test_reconstruction_multiplicative(self):
        """T × S × R == y everywhere trend is not NaN."""
        ts = _multiplicative_ts()
        r  = dec.decompose(ts, period=12, model="multiplicative")
        T  = r.trend.values
        S  = r.seasonal.values
        R  = r.residual.values
        y  = ts.values
        valid = ~np.isnan(T)
        np.testing.assert_allclose(T[valid] * S[valid] * R[valid], y[valid], rtol=1e-8)

    def test_model_stored(self):
        r = dec.decompose(_multiplicative_ts(), period=12, model="multiplicative")
        assert r.model == "multiplicative"

    def test_seasonal_mean_one(self):
        r = dec.decompose(_multiplicative_ts(), period=12, model="multiplicative")
        S = r.seasonal.values
        np.testing.assert_allclose(np.mean(S[:12]), 1.0, atol=1e-10)


# ===========================================================================
# Validation errors
# ===========================================================================

class TestValidationErrors:
    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            dec.decompose("not a ts")  # type: ignore[arg-type]

    def test_bad_model_raises(self):
        ts = _additive_ts()
        with pytest.raises(ValueError, match="model"):
            dec.decompose(ts, period=12, model="magic")

    def test_series_too_short_raises(self):
        idx = pd.date_range("2020", periods=10, freq="MS")
        ts  = TimeSeries(np.ones(10), index=idx)
        with pytest.raises(ValueError, match="2 × period"):
            dec.decompose(ts, period=12)

    def test_period_infer_fails_without_freq(self, ts_irregular):
        with pytest.raises(ValueError, match="period"):
            dec.decompose(ts_irregular)

    def test_period_one_raises(self):
        ts = _additive_ts()
        with pytest.raises(ValueError, match="period"):
            dec.decompose(ts, period=1)

    def test_multiplicative_negative_trend_raises(self):
        idx = pd.date_range("2020", periods=48, freq="MS")
        # Strongly negative mean → trend will be negative
        y   = np.tile(np.arange(12, dtype=float) - 10, 4)
        ts  = TimeSeries(y, index=idx)
        with pytest.raises(ValueError, match="positive"):
            dec.decompose(ts, period=12, model="multiplicative")


# ===========================================================================
# Private helper — _centered_trend
# ===========================================================================

class TestCenteredTrend:
    def test_odd_period_symmetric_nan(self):
        s   = pd.Series(np.ones(20))
        out = _centered_trend(s, period=7)
        assert np.isnan(out.iloc[0])
        assert np.isnan(out.iloc[-1])
        assert not np.isnan(out.iloc[3])

    def test_even_period_constant_series(self):
        s   = pd.Series(np.ones(30) * 5.0)
        out = _centered_trend(s, period=12)
        valid = out.dropna()
        np.testing.assert_allclose(valid.values, 5.0, atol=1e-10)

    def test_odd_period_constant_series(self):
        s   = pd.Series(np.ones(30) * 3.0)
        out = _centered_trend(s, period=7)
        valid = out.dropna()
        np.testing.assert_allclose(valid.values, 3.0, atol=1e-10)


# ===========================================================================
# _strength helper
# ===========================================================================

class TestStrength:
    def test_perfect_trend_strength_one(self):
        # R = 0 everywhere → strength = 1
        R = np.zeros(100)
        T = np.linspace(0, 1, 100)
        assert _strength(R, T + R) == 1.0

    def test_no_trend_strength_zero(self):
        # T = 0, R = noise → Var(T+R) = Var(R) → strength = 0
        R = np.random.randn(100)
        T = np.zeros(100)
        assert _strength(R, T + R) == 0.0

    def test_strength_in_range(self):
        R = np.random.randn(100)
        T = np.random.randn(100) * 3
        s = _strength(R, T + R)
        assert 0.0 <= s <= 1.0