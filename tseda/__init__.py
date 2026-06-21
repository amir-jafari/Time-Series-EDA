"""
tseda — Time Series Exploratory Data Analysis
=============================================

A comprehensive, dependency-light Python toolkit for understanding time series
data before forecasting it.

Quick Start
-----------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries

>>> idx = pd.date_range("2020-01-01", periods=365, freq="D")
>>> ts = TimeSeries(np.cumsum(np.random.randn(365)), index=idx,
...                 name="stock_price", unit="USD")
>>> print(ts)  # doctest: +SKIP

Modules
-------
core
    :class:`~tseda.core.TimeSeries` data structure and validators.
quality
    Missing-value analysis, outlier detection, and duplicate checks.
statistics
    Descriptive statistics, stationarity tests, and autocorrelation.
decomposition
    Classical (additive/multiplicative) and STL decomposition.
seasonality
    Period detection via periodogram and autocorrelation.
anomaly
    Point and contextual anomaly detection.
changepoint
    Structural break detection.
features
    Temporal, statistical, and spectral feature extraction.
forecastability
    Forecast-readiness scoring and data-leakage detection.
visualization
    Matplotlib-based plot suite.
report
    HTML and console report generation.
"""

from importlib.metadata import PackageNotFoundError, version

from tseda.core import (
    AggMethod,
    ArrayLike,
    DatetimeLike,
    DiffMethod,
    Frequency,
    TimeSeries,
)

try:
    __version__: str = version("tseda")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__author__  = "Amirhossein Jafari"
__email__   = "ajafari@gwu.edu"
__license__ = "MIT"

__all__ = [
    "__version__",
    "TimeSeries",
    "Frequency",
    "AggMethod",
    "DiffMethod",
    "ArrayLike",
    "DatetimeLike",
]