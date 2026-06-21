"""
Classical time series decomposition.

Decomposes a :class:`~tseda.core.TimeSeries` into **trend**, **seasonal**,
and **residual** components using the centered moving-average method
(classical / X-11 style).

Two models are supported:

* **Additive** — ``y = T + S + R``  (default)
* **Multiplicative** — ``y = T × S × R``  (use when variance scales with level)

The trend is estimated by a centered moving average whose window width equals
the seasonal *period*. For even periods (e.g., 12) a 2×MA is applied to
obtain a truly centered estimate.

All arithmetic is pure numpy / pandas — no extra dependencies.

Classes
-------
DecompositionResult
    Frozen dataclass holding all four components and quality metrics.
    *Shared by* :mod:`tseda.decomposition.stl`.
ClassicalDecomposer
    Stateless decomposer.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.decomposition.classical import ClassicalDecomposer

Monthly sales with trend + seasonality:

>>> rng  = np.random.default_rng(0)
>>> n    = 60
>>> t    = np.arange(n, dtype=float)
>>> seas = np.tile([-2, -1, 0, 1, 2, 3, 3, 2, 1, 0, -1, -2], 5)
>>> y    = 100 + 0.5 * t + seas + rng.standard_normal(n) * 0.5
>>> idx  = pd.date_range("2020-01", periods=n, freq="MS")
>>> ts   = TimeSeries(y, index=idx, name="sales", unit="units")
>>> dec  = ClassicalDecomposer().decompose(ts, period=12)
>>> dec.method
'classical'
>>> dec.model
'additive'
>>> round(dec.strength_seasonal, 2) > 0.5
True
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from tseda.core.timeseries import TimeSeries
from tseda.core.validator import validate_positive_int

__all__ = ["DecompositionResult", "ClassicalDecomposer"]

# ---------------------------------------------------------------------------
# Period lookup: default seasonal period for common frequencies
# ---------------------------------------------------------------------------

_FREQ_DEFAULT_PERIOD: dict[str, int] = {
    "s":   60,   # secondly → minute cycle
    "min": 60,   # minutely → hour cycle
    "T":   60,
    "h":   24,   # hourly → daily cycle
    "H":   24,
    "D":   7,    # daily → weekly cycle
    "B":   5,    # business-daily → weekly cycle
    "W":   52,   # weekly → annual cycle
    "MS":  12,   # monthly → annual cycle
    "M":   12,
    "ME":  12,
    "QS":  4,    # quarterly → annual cycle
    "Q":   4,
    "QE":  4,
}


def _default_period(freq: Optional[str]) -> Optional[int]:
    """Return the default seasonal period for *freq*, or ``None``."""
    if freq is None:
        return None
    key = freq.lstrip("0123456789")   # strip leading multiplier
    return _FREQ_DEFAULT_PERIOD.get(key)


# ---------------------------------------------------------------------------
# Shared result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecompositionResult:
    """Immutable time series decomposition result.

    Attributes
    ----------
    original : TimeSeries
        The input series.
    trend : TimeSeries
        Smooth trend component.  May contain NaN at the edges (classical
        decomposition only; STL fills the edges with LOESS extrapolation).
    seasonal : TimeSeries
        Periodic seasonal component with the same length as *original*.
    residual : TimeSeries
        Remainder after removing trend and seasonal.  NaN wherever *trend*
        is NaN.
    period : int
        Number of observations per seasonal cycle (e.g., 12 for monthly
        data with an annual pattern).
    model : str
        ``"additive"`` or ``"multiplicative"``.
    method : str
        ``"classical"`` or ``"stl"``.
    strength_trend : float
        Wang et al. (2006) trend strength:
        ``max(0, 1 − Var(R) / Var(T + R))``.  In [0, 1].
    strength_seasonal : float
        Wang et al. (2006) seasonality strength:
        ``max(0, 1 − Var(R) / Var(S + R))``.  In [0, 1].
    n_obs_used : int
        Number of non-NaN residual observations used for strength metrics.
    """

    original: TimeSeries
    trend: TimeSeries
    seasonal: TimeSeries
    residual: TimeSeries
    period: int
    model: str
    method: str
    strength_trend: float
    strength_seasonal: float
    n_obs_used: int

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Return all four components as a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            Columns: ``observed``, ``trend``, ``seasonal``, ``residual``.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.decomposition.classical import ClassicalDecomposer
        >>> idx = pd.date_range("2020", periods=36, freq="MS")
        >>> ts  = TimeSeries(np.ones(36) + np.tile(np.arange(12), 3), index=idx)
        >>> df  = ClassicalDecomposer().decompose(ts, period=12).to_dataframe()
        >>> list(df.columns)
        ['observed', 'trend', 'seasonal', 'residual']
        """
        return pd.DataFrame(
            {
                "observed": self.original.to_series().rename("observed"),
                "trend":    self.trend.to_series().rename("trend"),
                "seasonal": self.seasonal.to_series().rename("seasonal"),
                "residual": self.residual.to_series().rename("residual"),
            }
        )

    def summary(self) -> str:
        """Return a plain-text summary of the decomposition.

        Returns
        -------
        str
        """
        return (
            f"DecompositionResult\n"
            f"{'─' * 40}\n"
            f"  method           : {self.method}\n"
            f"  model            : {self.model}\n"
            f"  period           : {self.period}\n"
            f"  n_obs_used       : {self.n_obs_used}\n"
            f"  strength_trend   : {self.strength_trend:.4f}\n"
            f"  strength_seasonal: {self.strength_seasonal:.4f}\n"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _centered_trend(s: pd.Series, period: int) -> pd.Series:
    """Compute the centered moving-average trend.

    For odd *period*: single MA of width *period*.
    For even *period*: 2×MA (MA of *period* followed by MA of 2, then
    shifted to centre) — the standard X-11 / Census Method I approach.
    """
    if period % 2 == 1:
        return s.rolling(window=period, center=True, min_periods=period).mean()

    # Even period: trailing MA of period → trailing MA of 2 → shift to centre
    ma1 = s.rolling(window=period, min_periods=period).mean()
    ma2 = ma1.rolling(window=2, min_periods=2).mean()
    # ma2[i] is centred at position i − period/2; shift left by period//2 − 1
    # so that trend[i] is centred at position i.
    # Derivation:
    #   ma1[i] centres at i − (period−1)/2  (trailing window)
    #   ma2[i] centres at i − period/2
    #   We want trend[i] centred at i → need ma2[i + period/2]
    #   → trend = ma2.shift(−period//2)
    return ma2.shift(-(period // 2))


def _seasonal_and_residual_additive(
    y: np.ndarray, trend: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (seasonal, residual) for the additive model."""
    n = len(y)
    detrended = y - trend  # NaN where trend is NaN — that's fine

    # Average at each phase (position within period), ignoring NaN
    phase_avg = np.array(
        [np.nanmean(detrended[phase::period]) for phase in range(period)]
    )
    # Normalise so that the seasonal factors sum to zero over one period
    phase_avg -= np.nanmean(phase_avg)

    # Tile to full length
    seasonal = np.array([phase_avg[i % period] for i in range(n)])
    residual  = y - trend - seasonal
    return seasonal, residual


def _seasonal_and_residual_multiplicative(
    y: np.ndarray, trend: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (seasonal, residual) for the multiplicative model."""
    n = len(y)
    if np.any(trend[~np.isnan(trend)] <= 0):
        raise ValueError(
            "Multiplicative decomposition requires a positive trend; "
            "the estimated trend contains non-positive values.  "
            "Use model='additive' or apply a log-transform first."
        )

    ratio = y / trend  # NaN where trend is NaN

    phase_avg = np.array(
        [np.nanmean(ratio[phase::period]) for phase in range(period)]
    )
    # Normalise so the average seasonal factor = 1
    avg_factor = np.nanmean(phase_avg)
    if avg_factor == 0:
        raise ValueError("All seasonal factors averaged to zero — cannot normalise.")
    phase_avg /= avg_factor

    seasonal = np.array([phase_avg[i % period] for i in range(n)])
    residual  = y / (trend * seasonal)
    return seasonal, residual


def _strength(residual: np.ndarray, combined: np.ndarray) -> float:
    """Wang et al. (2006) component strength: max(0, 1 − Var(R)/Var(C+R))."""
    mask = ~(np.isnan(residual) | np.isnan(combined))
    r = residual[mask]
    c = combined[mask]
    if len(r) < 2 or np.var(c) == 0:
        return 0.0
    return float(max(0.0, 1.0 - np.var(r) / np.var(c)))


def _wrap(
    arr: np.ndarray,
    orig: TimeSeries,
    name_suffix: str,
) -> TimeSeries:
    """Wrap a numpy array into a TimeSeries, inheriting metadata from *orig*."""
    return TimeSeries(
        arr,
        index=orig.index,
        name=f"{orig.name}_{name_suffix}",
        freq=orig.freq,
        unit=orig.unit if name_suffix == "trend" else None,
    )


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------


class ClassicalDecomposer:
    """Decompose a :class:`~tseda.core.TimeSeries` using the centered
    moving-average (classical) method.

    The decomposer is **stateless** — one instance may be reused.

    Methods
    -------
    decompose(ts, period, model)
        Return a :class:`DecompositionResult`.

    Notes
    -----
    The classical method has well-known limitations:

    * The trend component has NaN at both ends (half the period width).
    * It assumes a fixed seasonal pattern throughout the series.
    * It can be sensitive to outliers.

    For more robust results, use :class:`~tseda.decomposition.stl.STLDecomposer`.

    Examples
    --------
    Additive decomposition:

    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.decomposition.classical import ClassicalDecomposer

    >>> rng  = np.random.default_rng(1)
    >>> n    = 48
    >>> seas = np.tile(np.sin(2 * np.pi * np.arange(12) / 12) * 5, 4)
    >>> y    = np.arange(n, dtype=float) * 0.3 + seas + rng.standard_normal(n)
    >>> idx  = pd.date_range("2020-01", periods=n, freq="MS")
    >>> ts   = TimeSeries(y, index=idx)
    >>> r    = ClassicalDecomposer().decompose(ts, period=12)
    >>> r.model
    'additive'
    >>> r.strength_seasonal > 0.5
    True
    """

    def decompose(
        self,
        ts: TimeSeries,
        period: Optional[int] = None,
        *,
        model: str = "additive",
    ) -> DecompositionResult:
        """Decompose *ts* into trend, seasonal, and residual components.

        Parameters
        ----------
        ts : TimeSeries
            Input series.  Should be regularly spaced and have length
            ``>= 2 × period``.
        period : int, optional
            Seasonal period (number of observations per cycle).  When
            omitted the period is inferred from ``ts.freq``:
            daily → 7, monthly → 12, quarterly → 4, etc.
        model : str, optional
            ``"additive"`` (default) or ``"multiplicative"``.

        Returns
        -------
        DecompositionResult

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *period* cannot be inferred, is < 2, or the series is too
            short; also if *model* is not recognised.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.decomposition.classical import ClassicalDecomposer

        >>> idx = pd.date_range("2020", periods=36, freq="MS")
        >>> y   = np.tile(np.arange(12, dtype=float), 3) + np.linspace(0, 6, 36)
        >>> ts  = TimeSeries(y, index=idx)
        >>> r   = ClassicalDecomposer().decompose(ts, period=12)
        >>> r.seasonal.n
        36
        >>> r.trend.n
        36
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )
        if model not in ("additive", "multiplicative"):
            raise ValueError(
                f"'model' must be 'additive' or 'multiplicative'; got {model!r}."
            )

        # --- resolve period ---
        if period is None:
            period = _default_period(ts.freq)
            if period is None:
                raise ValueError(
                    "'period' could not be inferred from the series frequency "
                    f"({ts.freq!r}).  Pass 'period' explicitly."
                )

        period = validate_positive_int(period, name="period")
        if period < 2:
            raise ValueError(f"'period' must be >= 2, got {period}.")
        if ts.n < 2 * period:
            raise ValueError(
                f"Series length ({ts.n}) must be at least 2 × period "
                f"({2 * period}) for classical decomposition."
            )

        y  = ts.values                 # float64 ndarray
        s  = ts.to_series()

        # --- 1. Trend: centered MA ---
        trend_series = _centered_trend(s, period)
        T = trend_series.values

        # --- 2. Seasonal + Residual ---
        if model == "additive":
            S, R = _seasonal_and_residual_additive(y, T, period)
        else:
            S, R = _seasonal_and_residual_multiplicative(y, T, period)

        # --- 3. Strength metrics ---
        if model == "additive":
            st = _strength(R, T + R)
            ss = _strength(R, S + R)
        else:
            # Work in log space for multiplicative model
            with np.errstate(divide="ignore", invalid="ignore"):
                log_T = np.log(np.where(T > 0, T, np.nan))
                log_S = np.log(np.where(S > 0, S, np.nan))
                log_R = np.log(np.where(R > 0, R, np.nan))
            st = _strength(log_R, log_T + log_R)
            ss = _strength(log_R, log_S + log_R)

        valid_mask = ~np.isnan(R)
        n_used = int(valid_mask.sum())

        return DecompositionResult(
            original=ts,
            trend=_wrap(T, ts, "trend"),
            seasonal=_wrap(S, ts, "seasonal"),
            residual=_wrap(R, ts, "residual"),
            period=period,
            model=model,
            method="classical",
            strength_trend=round(st, 6),
            strength_seasonal=round(ss, 6),
            n_obs_used=n_used,
        )