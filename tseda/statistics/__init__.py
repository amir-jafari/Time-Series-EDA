"""
tseda.statistics
================

Statistical analysis for time series.

Public API
----------
DescriptiveStats
    Frozen result of :class:`DescriptiveAnalyzer`.
DescriptiveAnalyzer
    Comprehensive descriptive statistics (mean, std, MAD, skew, kurtosis, quantiles, …).
StationarityResult
    Frozen result of :class:`StationarityTester`.
StationarityTester
    ADF, KPSS, and Phillips-Perron stationarity tests with combined summary.
AutocorrelationResult
    Frozen result of :class:`AutocorrelationAnalyzer`.
AutocorrelationAnalyzer
    ACF, PACF, Ljung-Box test, and significant-lag detection.
"""
from tseda.statistics.autocorrelation import (
    AutocorrelationAnalyzer,
    AutocorrelationResult,
)
from tseda.statistics.descriptive import DescriptiveAnalyzer, DescriptiveStats
from tseda.statistics.stationarity import StationarityResult, StationarityTester

__all__ = [
    "DescriptiveStats",
    "DescriptiveAnalyzer",
    "StationarityResult",
    "StationarityTester",
    "AutocorrelationResult",
    "AutocorrelationAnalyzer",
]