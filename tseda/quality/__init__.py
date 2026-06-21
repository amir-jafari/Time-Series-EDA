"""
tseda.quality
=============

Data quality diagnostics for time series.

Public API
----------
MissingValueReport
    Immutable result of :class:`MissingValueAnalyzer`.
MissingValueAnalyzer
    Detect NaN values, index gaps, and interpolate.
OutlierReport
    Immutable result of :class:`OutlierDetector`.
OutlierDetector
    IQR, Z-score, MAD, and GESD outlier detection.
FlatlineReport
    Immutable result of :class:`DuplicateDetector`.
DuplicateDetector
    Consecutive flat-line and near-zero segment detection.
"""
from tseda.quality.duplicates import DuplicateDetector, FlatlineReport
from tseda.quality.missing import MissingValueAnalyzer, MissingValueReport
from tseda.quality.outliers import OutlierDetector, OutlierReport

__all__ = [
    "MissingValueReport",
    "MissingValueAnalyzer",
    "OutlierReport",
    "OutlierDetector",
    "FlatlineReport",
    "DuplicateDetector",
]