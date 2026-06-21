"""Tests for :mod:`tseda.decomposition.stl`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.decomposition.classical import DecompositionResult
from tseda.decomposition.stl import STLDecomposer

dec = STLDecomposer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seasonal_ts(n: int = 60, period: int = 12, seed: int = 0) -> TimeSeries:
    rng  = np.random.default_rng(seed)
    seas = np.tile(np.sin(2 * np.pi * np.arange(period) / period) * 8.0, n // period + 1)[:n]
    y    = 50.0 + np.linspace(0, 5, n) + seas + rng.standard_normal(n) * 0.3
    idx  = pd.date_range("2020-01", periods=n, freq="MS")
    return TimeSeries(y, index=idx, name="temp", unit="°C")


# ===========================================================================
# Core functionality
# ===========================================================================

class TestSTLDecompose:
    def test_returns_result_type(self):
        r = dec.decompose(_seasonal_ts(), period=12)
        assert isinstance(r, DecompositionResult)

    def test_method_label(self):
        r = dec.decompose(_seasonal_ts(), period=12)
        assert r.method in ("stl", "stl-fallback")

    def test_model_always_additive(self):
        r = dec.decompose(_seasonal_ts(), period=12)
        assert r.model == "additive"

    def test_component_lengths_equal_original(self):
        ts = _seasonal_ts()
        r  = dec.decompose(ts, period=12)
        assert r.trend.n    == ts.n
        assert r.seasonal.n == ts.n
        assert r.residual.n == ts.n

    def test_no_nan_in_trend(self):
        """STL fills edges — trend should not have NaN (for complete series)."""
        r = dec.decompose(_seasonal_ts(), period=12)
        assert not r.trend.has_nan

    def test_reconstruction_identity(self):
        """T + S + R == y (at all non-NaN positions)."""
        ts = _seasonal_ts()
        r  = dec.decompose(ts, period=12)
        T  = r.trend.values
        S  = r.seasonal.values
        R  = r.residual.values
        y  = ts.values
        valid = ~(np.isnan(T) | np.isnan(S) | np.isnan(R))
        np.testing.assert_allclose(
            T[valid] + S[valid] + R[valid], y[valid], rtol=1e-6
        )

    def test_strength_seasonal_high(self):
        r = dec.decompose(_seasonal_ts(), period=12)
        assert r.strength_seasonal > 0.6

    def test_strength_in_range(self):
        r = dec.decompose(_seasonal_ts(), period=12)
        assert 0.0 <= r.strength_trend    <= 1.0
        assert 0.0 <= r.strength_seasonal <= 1.0

    def test_period_stored(self):
        r = dec.decompose(_seasonal_ts(), period=12)
        assert r.period == 12

    def test_period_inferred_from_freq(self):
        ts = _seasonal_ts()   # MS → period 12
        r  = dec.decompose(ts)
        assert r.period == 12

    def test_robust_true_vs_false(self):
        ts  = _seasonal_ts()
        r1  = dec.decompose(ts, period=12, robust=True)
        r2  = dec.decompose(ts, period=12, robust=False)
        # Both should reconstruct; robust may differ slightly
        assert r1.method == r2.method
        assert r1.strength_seasonal > 0.5

    def test_original_stored(self):
        ts = _seasonal_ts()
        r  = dec.decompose(ts, period=12)
        assert r.original is ts

    def test_to_dataframe(self):
        r  = dec.decompose(_seasonal_ts(), period=12)
        df = r.to_dataframe()
        assert set(df.columns) == {"observed", "trend", "seasonal", "residual"}

    def test_n_obs_used_positive(self):
        r = dec.decompose(_seasonal_ts(), period=12)
        assert r.n_obs_used > 0

    def test_handles_series_with_nan(self):
        """NaN positions in original should produce NaN in components."""
        ts_clean = _seasonal_ts()
        vals     = ts_clean.values.copy()
        vals[5]  = np.nan
        ts_nan   = TimeSeries(vals, index=ts_clean.index, name="with_nan")
        r = dec.decompose(ts_nan, period=12)
        assert np.isnan(r.residual.values[5])


# ===========================================================================
# Validation errors
# ===========================================================================

class TestSTLValidationErrors:
    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            dec.decompose("not a ts")  # type: ignore[arg-type]

    def test_series_too_short_raises(self):
        idx = pd.date_range("2020", periods=10, freq="MS")
        ts  = TimeSeries(np.ones(10), index=idx)
        with pytest.raises(ValueError, match="2 × period"):
            dec.decompose(ts, period=12)

    def test_period_infer_fails_without_freq(self, ts_irregular):
        with pytest.raises(ValueError, match="period"):
            dec.decompose(ts_irregular)

    def test_period_one_raises(self):
        ts = _seasonal_ts()
        with pytest.raises(ValueError, match="period"):
            dec.decompose(ts, period=1)