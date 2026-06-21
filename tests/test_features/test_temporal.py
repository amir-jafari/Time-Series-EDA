"""Tests for :mod:`tseda.features.temporal`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.features.temporal import TemporalFeatureExtractor

ext = TemporalFeatureExtractor()


def _ts(n: int = 10, freq: str = "D") -> TimeSeries:
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    return TimeSeries(np.arange(float(n)), index=idx)


class TestTemporalExtract:
    def test_returns_dataframe(self):
        df = ext.extract(_ts())
        assert isinstance(df, pd.DataFrame)

    def test_index_matches_ts(self):
        ts = _ts(20)
        df = ext.extract(ts)
        assert (df.index == ts.index).all()

    def test_row_count_equals_n(self):
        ts = _ts(50)
        df = ext.extract(ts)
        assert len(df) == ts.n

    def test_year_column(self):
        ts = _ts(3)
        df = ext.extract(ts)
        assert int(df["year"].iloc[0]) == 2020

    def test_month_column(self):
        ts = _ts(3)
        df = ext.extract(ts)
        assert int(df["month"].iloc[0]) == 1

    def test_dayofweek_range(self):
        ts = _ts(30)
        df = ext.extract(ts)
        assert df["dayofweek"].between(0, 6).all()

    def test_is_weekend_flag(self):
        # 2020-01-01 = Wednesday, 2020-01-04 = Saturday, 2020-01-05 = Sunday
        ts = _ts(7)
        df = ext.extract(ts)
        assert df["is_weekend"].iloc[0] == 0.0   # Wed
        assert df["is_weekend"].iloc[3] == 1.0   # Sat (2020-01-04)
        assert df["is_weekend"].iloc[4] == 1.0   # Sun (2020-01-05)

    def test_cyclic_columns_present(self):
        df = ext.extract(_ts(), cyclic=True)
        assert "month_sin" in df.columns
        assert "month_cos" in df.columns
        assert "dow_sin"   in df.columns
        assert "dow_cos"   in df.columns

    def test_cyclic_false_no_sin_cos(self):
        df = ext.extract(_ts(), cyclic=False)
        assert "month_sin" not in df.columns
        assert "dow_sin"   not in df.columns

    def test_time_index_columns(self):
        df = ext.extract(_ts(5), time_index=True)
        assert "days_since_start" in df.columns
        assert "time_norm"        in df.columns

    def test_time_norm_starts_zero_ends_one(self):
        df = ext.extract(_ts(10), time_index=True)
        assert float(df["time_norm"].iloc[0]) == 0.0
        assert abs(float(df["time_norm"].iloc[-1]) - 1.0) < 1e-6

    def test_time_index_false(self):
        df = ext.extract(_ts(), time_index=False)
        assert "days_since_start" not in df.columns
        assert "time_norm"        not in df.columns

    def test_hour_column_for_hourly(self):
        ts = _ts(24, freq="h")
        df = ext.extract(ts)
        assert int(df["hour"].iloc[0]) == 0
        assert int(df["hour"].iloc[1]) == 1

    def test_month_sin_cos_identity(self):
        """sin²+cos² must equal 1 for every row."""
        df = ext.extract(_ts(12), cyclic=True)
        identity = df["month_sin"] ** 2 + df["month_cos"] ** 2
        np.testing.assert_allclose(identity.values, 1.0, atol=1e-10)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            ext.extract("not a ts")  # type: ignore[arg-type]

    def test_quarter_range(self):
        ts = _ts(366)
        df = ext.extract(ts)
        assert df["quarter"].between(1, 4).all()

    def test_weekofyear_range(self):
        ts = _ts(366)
        df = ext.extract(ts)
        assert df["weekofyear"].between(1, 53).all()