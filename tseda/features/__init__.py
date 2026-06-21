"""
tseda.features
==============

Feature extraction for time series.

Public API
----------
TemporalFeatureExtractor
    Calendar and cyclic time-index features → :class:`pandas.DataFrame`.
StatisticalFeatureExtractor
    Distribution, complexity, and linear-structure features → single-row DataFrame.
SpectralFeatureExtractor
    Frequency-domain (FFT) features → single-row DataFrame.
"""
from tseda.features.spectral import SpectralFeatureExtractor
from tseda.features.statistical import StatisticalFeatureExtractor
from tseda.features.temporal import TemporalFeatureExtractor

__all__ = [
    "TemporalFeatureExtractor",
    "StatisticalFeatureExtractor",
    "SpectralFeatureExtractor",
]