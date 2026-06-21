"""
tseda.forecastability — Forecast readiness scoring and leakage detection.

Classes
-------
ForecastabilityReport
    Immutable result of :meth:`ForecastabilityScorer.score`.
ForecastabilityScorer
    Composite 0–100 forecastability scorer.
LeakageReport
    Immutable result of :meth:`LeakageDetector.check`.
LeakageDetector
    Temporal and target leakage detector for feature sets.
"""
from tseda.forecastability.leakage import LeakageDetector, LeakageReport
from tseda.forecastability.scorer import ForecastabilityReport, ForecastabilityScorer

__all__ = [
    "ForecastabilityReport",
    "ForecastabilityScorer",
    "LeakageReport",
    "LeakageDetector",
]