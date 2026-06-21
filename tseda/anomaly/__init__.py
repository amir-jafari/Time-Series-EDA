"""
tseda.anomaly
=============

Point and contextual anomaly detection for time series.

Public API
----------
AnomalyReport
    Frozen dataclass with mask, indices, timestamps, values, scores,
    method, and n_anomalies.
AnomalyDetector
    Stateless detector: rolling IQR, rolling Z-score, STL-residual,
    and GESD methods.
"""
from tseda.anomaly.detector import AnomalyDetector, AnomalyReport

__all__ = ["AnomalyReport", "AnomalyDetector"]
