"""
tseda.changepoint
=================

Structural break (changepoint) detection for time series.

Public API
----------
ChangepointReport
    Frozen dataclass with changepoint positions, timestamps, scores,
    and a segment-label helper.
ChangepointDetector
    Stateless detector: CUSUM, binary segmentation, variance ratio.
"""
from tseda.changepoint.detector import ChangepointDetector, ChangepointReport

__all__ = ["ChangepointReport", "ChangepointDetector"]