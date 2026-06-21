"""Tests for :mod:`tseda.features.spectral`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.features.spectral import SpectralFeatureExtractor

ext = SpectralFeatureExtractor()


def _sine_ts(period: int = 7, n: int = 256, freq: str = "D") -> TimeSeries:
    """Pure sine wave with given period."""
    t   = np.arange(n, dtype=float)
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    return TimeSeries(np.sin(2 * np.pi * t / period), index=idx)


def _noise_ts(n: int = 128, seed: int = 0) -> TimeSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020", periods=n, freq="D")
    return TimeSeries(rng.standard_normal(n), index=idx)


class TestSpectralExtract:
    def test_returns_single_row_dataframe(self):
        df = ext.extract(_sine_ts())
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_dominant_period_sine_7(self):
        df  = ext.extract(_sine_ts(period=7, n=252))
        dom = int(round(float(df["dominant_period"].iloc[0])))
        assert dom == 7

    def test_dominant_period_sine_12(self):
        df  = ext.extract(_sine_ts(period=12, n=240))
        dom = int(round(float(df["dominant_period"].iloc[0])))
        assert dom == 12

    def test_total_power_positive(self):
        df = ext.extract(_sine_ts())
        assert float(df["total_power"].iloc[0]) > 0

    def test_spectral_entropy_range(self):
        df = ext.extract(_noise_ts())
        se = float(df["spectral_entropy"].iloc[0])
        assert 0.0 <= se <= 1.0

    def test_spectral_flatness_range(self):
        df = ext.extract(_noise_ts())
        sf = float(df["spectral_flatness"].iloc[0])
        assert 0.0 <= sf <= 1.0

    def test_spectral_flatness_sine_low(self):
        """Pure sine has most energy at one frequency → flatness close to 0."""
        df = ext.extract(_sine_ts(period=7, n=252))
        assert float(df["spectral_flatness"].iloc[0]) < 0.1

    def test_spectral_centroid_present(self):
        df = ext.extract(_sine_ts())
        assert "spectral_centroid" in df.columns
        assert float(df["spectral_centroid"].iloc[0]) > 0

    def test_band_power_columns(self):
        df = ext.extract(_sine_ts(), n_bands=3)
        assert "band_power_0" in df.columns
        assert "band_power_1" in df.columns
        assert "band_power_2" in df.columns

    def test_n_bands_respected(self):
        for b in [1, 2, 5]:
            df = ext.extract(_sine_ts(), n_bands=b)
            assert f"band_power_{b-1}" in df.columns

    def test_n_spectral_peaks_positive_for_sine(self):
        df = ext.extract(_sine_ts(period=7, n=252))
        assert int(df["n_spectral_peaks"].iloc[0]) >= 1

    def test_handles_nan(self):
        ts   = _sine_ts()
        vals = ts.values.copy()
        vals[5:10] = np.nan
        ts_nan = TimeSeries(vals, index=ts.index)
        df = ext.extract(ts_nan)
        assert not df.isnull().any().any()

    def test_rolloff_order(self):
        df = ext.extract(_sine_ts())
        r50 = float(df["spectral_rolloff_0.5"].iloc[0])
        r85 = float(df["spectral_rolloff_0.85"].iloc[0])
        assert r50 <= r85

    def test_too_few_obs_raises(self):
        idx = pd.date_range("2020", periods=5, freq="D")
        ts  = TimeSeries(np.ones(5), index=idx)
        with pytest.raises(ValueError, match="8"):
            ext.extract(ts)

    def test_bad_n_bands_raises(self):
        ts = _sine_ts()
        with pytest.raises(ValueError, match="n_bands"):
            ext.extract(ts, n_bands=0)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            ext.extract("not a ts")  # type: ignore[arg-type]