"""
tseda.seasonality
=================

Seasonal period detection for time series.

Public API
----------
SeasonalityReport
    Frozen dataclass with dominant period, candidate list, Fisher G-test,
    and ``is_seasonal`` flag.
SeasonalityDetector
    Stateless detector supporting periodogram (FFT), ACF-peak, and
    combined strategies.
"""
from tseda.seasonality.detector import SeasonalityDetector, SeasonalityReport

__all__ = ["SeasonalityReport", "SeasonalityDetector"]
