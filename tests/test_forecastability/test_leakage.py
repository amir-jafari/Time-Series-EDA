"""
Tests for :mod:`tseda.forecastability.leakage`.

Coverage targets
----------------
* LeakageDetector.check() — return type, report fields, warnings
* Target leakage           — exact copy, linear transform, near-copy
* Temporal leakage         — future-shifted features
* No leakage               — lagged features, random features
* None features_df         — warning, empty report
* Input validation         — type errors, wrong horizon, bad threshold
* Edge cases               — very short series, single column, all-NaN column
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.forecastability.leakage import (
    LeakageDetector,
    LeakageReport,
    _cross_corr_at_lag,
)

det = LeakageDetector()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts(n: int = 100, seed: int = 0) -> TimeSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return TimeSeries(rng.standard_normal(n), index=idx)


# ===========================================================================
# Return type & structure
# ===========================================================================


class TestReturnType:
    def test_returns_report(self):
        ts = _make_ts()
        r = det.check(ts, horizon=5)
        assert isinstance(r, LeakageReport)

    def test_no_features_returns_empty_report(self):
        ts = _make_ts()
        r = det.check(ts, horizon=5)
        assert r.n_features == 0
        assert r.has_temporal_leakage is False
        assert r.has_target_leakage is False
        assert len(r.warnings) > 0

    def test_report_stores_horizon(self):
        ts = _make_ts()
        r = det.check(ts, horizon=10)
        assert r.horizon == 10

    def test_report_stores_n_obs(self):
        ts = _make_ts(n=80)
        r = det.check(ts, horizon=3)
        assert r.n_obs == 80

    def test_report_stores_n_features(self):
        ts = _make_ts(n=80)
        idx = ts.index
        y = ts.values
        df = pd.DataFrame({"f1": np.roll(y, 1), "f2": np.roll(y, 2)}, index=idx)
        r = det.check(ts, horizon=3, features_df=df)
        assert r.n_features == 2


# ===========================================================================
# Target leakage
# ===========================================================================


class TestTargetLeakage:
    def test_exact_copy_detected(self):
        ts = _make_ts(n=100)
        y = ts.values
        df = pd.DataFrame({"target_copy": y}, index=ts.index)
        r = det.check(ts, horizon=5, features_df=df)
        assert r.has_target_leakage is True
        assert "target_copy" in r.target_leakage_columns

    def test_linear_transform_detected(self):
        ts = _make_ts(n=100)
        y = ts.values
        df = pd.DataFrame({"scaled": y * 2.0 + 10.0}, index=ts.index)
        r = det.check(ts, horizon=5, features_df=df)
        assert r.has_target_leakage is True

    def test_lagged_feature_not_target_leakage(self):
        ts = _make_ts(n=100)
        y = ts.values
        lagged = np.roll(y, 5).astype(float)
        lagged[:5] = np.nan
        df = pd.DataFrame({"lag5": lagged}, index=ts.index)
        r = det.check(ts, horizon=5, features_df=df)
        assert r.has_target_leakage is False

    def test_random_feature_not_target_leakage(self):
        ts = _make_ts(n=100, seed=0)
        rng = np.random.default_rng(99)
        df = pd.DataFrame({"random": rng.standard_normal(100)}, index=ts.index)
        r = det.check(ts, horizon=5, features_df=df)
        assert r.has_target_leakage is False

    def test_correlation_stored(self):
        ts = _make_ts(n=100)
        y = ts.values
        df = pd.DataFrame({"target_copy": y}, index=ts.index)
        r = det.check(ts, horizon=5, features_df=df)
        assert "target_copy" in r.target_leakage_correlations
        assert abs(r.target_leakage_correlations["target_copy"]) > 0.99

    def test_custom_threshold_lower(self):
        ts = _make_ts(n=100, seed=0)
        y = ts.values
        rng = np.random.default_rng(1)
        # y*0.5 + noise*0.5 → corr ≈ 0.7, below strict threshold (0.95)
        slightly_corr = y * 0.5 + rng.standard_normal(100) * 0.5
        df = pd.DataFrame({"slightly_corr": slightly_corr}, index=ts.index)
        r_strict = det.check(ts, horizon=5, features_df=df, target_corr_threshold=0.95)
        r_loose = det.check(ts, horizon=5, features_df=df, target_corr_threshold=0.50)
        assert r_loose.has_target_leakage is True
        assert not r_strict.has_target_leakage


# ===========================================================================
# Temporal leakage
# ===========================================================================


class TestTemporalLeakage:
    def test_future_shifted_feature_flagged(self):
        rng = np.random.default_rng(42)
        n = 200
        idx = pd.date_range("2020", periods=n, freq="D")
        y = rng.standard_normal(n)
        ts = TimeSeries(y, index=idx)
        future_feat = np.roll(y, -5).astype(float)
        future_feat[-5:] = np.nan
        df = pd.DataFrame({"future_5": future_feat}, index=idx)
        r = det.check(ts, horizon=10, features_df=df)
        assert r.has_temporal_leakage is True
        assert "future_5" in r.temporal_leakage_columns

    def test_past_shifted_feature_not_flagged(self):
        rng = np.random.default_rng(42)
        n = 200
        idx = pd.date_range("2020", periods=n, freq="D")
        y = rng.standard_normal(n)
        ts = TimeSeries(y, index=idx)
        past_feat = np.roll(y, 5).astype(float)
        past_feat[:5] = np.nan
        df = pd.DataFrame({"lag5": past_feat}, index=idx)
        r = det.check(ts, horizon=10, features_df=df)
        assert r.has_temporal_leakage is False

    def test_peak_lag_stored(self):
        rng = np.random.default_rng(0)
        n = 150
        idx = pd.date_range("2020", periods=n, freq="D")
        y = rng.standard_normal(n)
        ts = TimeSeries(y, index=idx)
        df = pd.DataFrame({"lag3": np.roll(y, 3)}, index=idx)
        r = det.check(ts, horizon=10, features_df=df)
        assert "lag3" in r.temporal_peak_lags


# ===========================================================================
# No leakage cases
# ===========================================================================


class TestNoLeakage:
    def test_pure_noise_features(self):
        ts = _make_ts(n=100, seed=0)
        rng = np.random.default_rng(77)
        df = pd.DataFrame(
            {f"noise_{i}": rng.standard_normal(100) for i in range(5)},
            index=ts.index,
        )
        r = det.check(ts, horizon=5, features_df=df)
        assert r.has_target_leakage is False

    def test_lagged_features_no_target_leakage(self):
        ts = _make_ts(n=100, seed=0)
        y = ts.values
        lags = {f"lag{k}": np.roll(y, k) for k in range(1, 6)}
        for k in range(1, 6):
            lags[f"lag{k}"][:k] = np.nan
        df = pd.DataFrame(lags, index=ts.index)
        r = det.check(ts, horizon=5, features_df=df)
        assert r.has_target_leakage is False


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    def test_wrong_ts_type_raises(self):
        with pytest.raises(TypeError, match="TimeSeries"):
            det.check([1, 2, 3], horizon=5)

    def test_horizon_zero_raises(self):
        ts = _make_ts()
        with pytest.raises(ValueError, match="horizon"):
            det.check(ts, horizon=0)

    def test_horizon_negative_raises(self):
        ts = _make_ts()
        with pytest.raises(ValueError, match="horizon"):
            det.check(ts, horizon=-1)

    def test_bad_threshold_zero_raises(self):
        ts = _make_ts()
        with pytest.raises(ValueError, match="target_corr_threshold"):
            det.check(ts, horizon=5, target_corr_threshold=0.0)

    def test_bad_threshold_negative_raises(self):
        ts = _make_ts()
        with pytest.raises(ValueError, match="target_corr_threshold"):
            det.check(ts, horizon=5, target_corr_threshold=-0.1)

    def test_features_wrong_type_raises(self):
        ts = _make_ts()
        with pytest.raises(TypeError, match="DataFrame"):
            det.check(ts, horizon=5, features_df=[[1, 2], [3, 4]])

    def test_features_wrong_length_raises(self):
        ts = _make_ts(n=100)
        df = pd.DataFrame({"x": np.ones(50)})
        with pytest.raises(ValueError, match="rows"):
            det.check(ts, horizon=5, features_df=df)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_column_dataframe(self):
        ts = _make_ts(n=80)
        y = ts.values
        df = pd.DataFrame({"f": y * 0.3}, index=ts.index)
        r = det.check(ts, horizon=3, features_df=df)
        assert isinstance(r, LeakageReport)

    def test_empty_dataframe_columns(self):
        ts = _make_ts(n=80)
        df = pd.DataFrame(index=ts.index)
        r = det.check(ts, horizon=3, features_df=df)
        assert r.n_features == 0
        assert r.has_temporal_leakage is False
        assert r.has_target_leakage is False

    def test_all_nan_column_no_crash(self):
        ts = _make_ts(n=80)
        df = pd.DataFrame({"all_nan": np.full(80, np.nan)}, index=ts.index)
        r = det.check(ts, horizon=5, features_df=df)
        assert isinstance(r, LeakageReport)

    def test_horizon_1(self):
        ts = _make_ts(n=80)
        y = ts.values
        df = pd.DataFrame({"f": y * 0.5}, index=ts.index)
        r = det.check(ts, horizon=1, features_df=df)
        assert isinstance(r, LeakageReport)

    def test_multiple_columns(self):
        ts = _make_ts(n=100, seed=0)
        rng = np.random.default_rng(5)
        y = ts.values
        df = pd.DataFrame(
            {
                "leak": y,
                "safe": rng.standard_normal(100),
                "lag2": np.roll(y, 2),
            },
            index=ts.index,
        )
        r = det.check(ts, horizon=5, features_df=df)
        assert "leak" in r.target_leakage_columns
        assert "safe" not in r.target_leakage_columns


# ===========================================================================
# Private helper: _cross_corr_at_lag
# ===========================================================================


class TestCrossCorr:
    def test_lag0_perfect_correlation(self):
        x = np.arange(50, dtype=float)
        assert _cross_corr_at_lag(x, x, lag=0) == pytest.approx(1.0)

    def test_lag0_anti_correlation(self):
        x = np.arange(50, dtype=float)
        y = -x
        assert _cross_corr_at_lag(x, y, lag=0) == pytest.approx(-1.0)

    def test_positive_lag(self):
        x = np.arange(50, dtype=float)
        corr = _cross_corr_at_lag(x, x, lag=3)
        assert corr == pytest.approx(1.0)

    def test_negative_lag(self):
        x = np.arange(50, dtype=float)
        corr = _cross_corr_at_lag(x, x, lag=-3)
        assert corr == pytest.approx(1.0)

    def test_too_short_returns_zero(self):
        x = np.array([1.0, 2.0])
        assert _cross_corr_at_lag(x, x, lag=5) == 0.0

    def test_constant_returns_zero(self):
        x = np.ones(50)
        assert _cross_corr_at_lag(x, x, lag=0) == 0.0