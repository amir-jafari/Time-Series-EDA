"""
Shared pytest fixtures for the tseda test suite.

All fixtures produce :class:`~tseda.core.TimeSeries` objects that cover the
common edge cases exercised across multiple test modules:

* ``ts_daily``       — 365 observations, daily, no NaN, regular.
* ``ts_hourly``      — 720 observations, hourly, no NaN, regular.
* ``ts_monthly``     — 36 observations, monthly (MS), no NaN, regular.
* ``ts_with_nan``    — daily, contains 10 % NaN values.
* ``ts_short``       — 5 observations (boundary cases).
* ``ts_irregular``   — randomly-spaced timestamps, irregular.
"""

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def ts_daily(rng: np.random.Generator) -> TimeSeries:
    """365-point daily time series (2020-01-01 → 2020-12-30)."""
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    vals = np.cumsum(rng.standard_normal(365))
    return TimeSeries(vals, index=idx, name="daily", unit="units")


@pytest.fixture(scope="session")
def ts_hourly(rng: np.random.Generator) -> TimeSeries:
    """720-point hourly time series (30 days)."""
    idx = pd.date_range("2020-01-01", periods=720, freq="h")
    vals = np.cumsum(rng.standard_normal(720))
    return TimeSeries(vals, index=idx, name="hourly", unit="°C")


@pytest.fixture(scope="session")
def ts_monthly(rng: np.random.Generator) -> TimeSeries:
    """36-point monthly (start-of-month) time series (3 years)."""
    idx = pd.date_range("2018-01-01", periods=36, freq="MS")
    vals = rng.standard_normal(36) * 10 + 100
    return TimeSeries(vals, index=idx, name="monthly", unit="USD")


@pytest.fixture(scope="session")
def ts_with_nan(rng: np.random.Generator) -> TimeSeries:
    """Daily series with ~10 % NaN values."""
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    vals = rng.standard_normal(200).astype(float)
    nan_idx = rng.choice(200, size=20, replace=False)
    vals[nan_idx] = np.nan
    return TimeSeries(vals, index=idx, name="with_nan")


@pytest.fixture(scope="session")
def ts_short() -> TimeSeries:
    """Minimal 5-observation series for boundary-condition tests."""
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    return TimeSeries([10.0, 20.0, 30.0, 40.0, 50.0], index=idx, name="short")


@pytest.fixture(scope="session")
def ts_irregular(rng: np.random.Generator) -> TimeSeries:
    """50-point series with irregular (non-uniform) timestamps."""
    base = pd.Timestamp("2020-01-01")
    offsets = np.cumsum(rng.integers(1, 10, size=50))  # random gaps in hours
    idx = pd.DatetimeIndex([base + pd.Timedelta(hours=int(h)) for h in offsets])
    vals = rng.standard_normal(50)
    return TimeSeries(vals, index=idx, name="irregular")