"""Tests for :mod:`tseda.features.statistical`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.features.statistical import (
    StatisticalFeatureExtractor,
    _approx_entropy,
    _linear_trend,
    _sample_entropy,
)

ext = StatisticalFeatureExtractor()


def _ts(vals, freq: str = "D") -> TimeSeries:
    idx = pd.date_range("2020-01-01", periods=len(vals), freq=freq)
    return TimeSeries(np.array(vals, dtype=float), index=idx)


class TestStatisticalExtract:
    def test_returns_single_row_dataframe(self):
        ts = _ts(range(20))
        df = ext.extract(ts, entropy=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_mean_known(self):
        ts  = _ts([1.0, 2.0, 3.0, 4.0, 5.0])
        df  = ext.extract(ts, entropy=False)
        assert abs(float(df["mean"].iloc[0]) - 3.0) < 1e-10

    def test_std_known(self):
        ts  = _ts([0.0, 1.0, 2.0, 3.0, 4.0])
        df  = ext.extract(ts, entropy=False)
        expected = float(np.std([0, 1, 2, 3, 4], ddof=1))
        assert abs(float(df["std"].iloc[0]) - expected) < 1e-8

    def test_min_max_range(self):
        ts = _ts([3.0, 1.0, 4.0, 1.0, 5.0])
        df = ext.extract(ts, entropy=False)
        assert float(df["min"].iloc[0]) == 1.0
        assert float(df["max"].iloc[0]) == 5.0
        assert float(df["range"].iloc[0]) == 4.0

    def test_linear_series_slope_one(self):
        ts = _ts(range(100))
        df = ext.extract(ts, entropy=False)
        assert abs(float(df["linear_slope"].iloc[0]) - 1.0) < 1e-8

    def test_linear_r2_one_for_perfect_trend(self):
        ts = _ts(range(100))
        df = ext.extract(ts, entropy=False)
        assert abs(float(df["linear_r2"].iloc[0]) - 1.0) < 1e-8

    def test_constant_series_slope_zero(self):
        ts = _ts([5.0] * 20)
        df = ext.extract(ts, entropy=False)
        assert abs(float(df["linear_slope"].iloc[0])) < 1e-10

    def test_skewness_symmetric_near_zero(self):
        ts  = _ts([1.0, 2.0, 3.0, 4.0, 5.0])
        df  = ext.extract(ts, entropy=False)
        assert abs(float(df["skewness"].iloc[0])) < 0.1

    def test_iqr_known(self):
        ts = _ts([1.0, 2.0, 3.0, 4.0, 5.0])
        df = ext.extract(ts, entropy=False)
        assert abs(float(df["iqr"].iloc[0]) - 2.0) < 1e-8

    def test_lag1_acf_positive_for_trending(self):
        ts = _ts(range(50))
        df = ext.extract(ts, entropy=False)
        assert float(df["lag1_acf"].iloc[0]) > 0.8

    def test_turning_points_oscillating(self):
        ts = _ts([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        df = ext.extract(ts, entropy=False)
        # Oscillating → high turning points ratio
        assert float(df["turning_points_ratio"].iloc[0]) > 0.5

    def test_mean_crossing_rate_oscillating(self):
        vals = np.tile([1.0, -1.0], 25)
        ts   = _ts(vals)
        df   = ext.extract(ts, entropy=False)
        assert float(df["mean_crossing_rate"].iloc[0]) > 0.5

    def test_flatness_ratio_constant(self):
        ts = _ts([3.0] * 10)
        df = ext.extract(ts, entropy=False)
        assert float(df["flatness_ratio"].iloc[0]) == 1.0

    def test_n_peaks_counted(self):
        ts = _ts([0, 1, 0, 2, 0, 3, 0])
        df = ext.extract(ts, entropy=False)
        assert int(df["n_peaks"].iloc[0]) >= 2

    def test_approx_entropy_regular_low(self):
        ts = _ts([0.0, 1.0] * 50)
        df = ext.extract(ts, entropy=True)
        assert float(df["approx_entropy"].iloc[0]) < 0.5  # very regular

    def test_required_columns_present(self):
        ts  = _ts(range(20))
        df  = ext.extract(ts, entropy=False)
        required = {"mean", "std", "skewness", "kurtosis", "min", "max",
                    "lag1_acf", "linear_slope", "linear_r2", "n_peaks"}
        assert required.issubset(set(df.columns))

    def test_too_few_obs_raises(self):
        ts = _ts([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="4"):
            ext.extract(ts)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            ext.extract("not a ts")  # type: ignore[arg-type]


class TestHelpers:
    def test_linear_trend_perfect(self):
        x = np.arange(100, dtype=float)
        slope, r2 = _linear_trend(x)
        assert abs(slope - 1.0) < 1e-8
        assert abs(r2 - 1.0) < 1e-8

    def test_linear_trend_constant(self):
        x = np.ones(50) * 7.0
        slope, r2 = _linear_trend(x)
        assert abs(slope) < 1e-10

    def test_approx_entropy_constant_zero(self):
        x = np.ones(50)
        assert _approx_entropy(x) == 0.0

    def test_sample_entropy_constant_zero(self):
        x = np.ones(50)
        assert _sample_entropy(x) == 0.0