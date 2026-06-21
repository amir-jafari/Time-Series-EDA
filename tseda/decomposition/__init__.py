"""
tseda.decomposition
===================

Time series decomposition into trend, seasonal, and residual components.

Public API
----------
DecompositionResult
    Frozen dataclass holding all four components and quality metrics.
ClassicalDecomposer
    Centered moving-average decomposition (additive or multiplicative).
    Pure numpy — no extra dependencies.
STLDecomposer
    LOESS-based STL decomposition (statsmodels primary, scipy fallback).
"""
from tseda.decomposition.classical import ClassicalDecomposer, DecompositionResult
from tseda.decomposition.stl import STLDecomposer

__all__ = [
    "DecompositionResult",
    "ClassicalDecomposer",
    "STLDecomposer",
]