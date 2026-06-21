"""
STL (Seasonal-Trend decomposition using LOESS) for time series.

STL (Cleveland et al., 1990) is preferred over the classical moving-average
approach because it:

* Fills the trend at the edges (no NaN border effects).
* Allows the seasonal component to evolve slowly over time.
* Provides a robust fitting option that down-weights outliers.
* Handles any period length, including non-integer values.

**Primary path** — delegates to :class:`statsmodels.tsa.seasonal.STL` when
``statsmodels`` is installed (``pip install tseda[stats]``).

**Fallback path** — when ``statsmodels`` is absent, a simplified iterative
decomposition is used:

1. Trend via Savitzky-Golay smoothing (:func:`scipy.signal.savgol_filter`).
2. Seasonal component via per-phase averaging (same as classical).
3. Residual = ``y − trend − seasonal``.

The fallback is clearly labelled ``"stl-fallback"`` in
:attr:`~tseda.decomposition.classical.DecompositionResult.method`.

Classes
-------
STLDecomposer
    Stateless decomposer.

Notes
-----
STL always produces an **additive** decomposition.  For multiplicative
behaviour apply :meth:`~tseda.core.TimeSeries.log` before decomposing and
exponentiate the components afterwards.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.decomposition.stl import STLDecomposer

Monthly temperature with annual seasonality:

>>> rng  = np.random.default_rng(0)
>>> n    = 60
>>> seas = np.tile(np.array([3, 5, 8, 12, 16, 19, 21, 20, 16, 11, 6, 3], dtype=float), 5)
>>> y    = seas + np.linspace(0, 3, n) + rng.standard_normal(n) * 0.3
>>> idx  = pd.date_range("2020-01", periods=n, freq="MS")
>>> ts   = TimeSeries(y, index=idx, name="temp", unit="°C")
>>> r    = STLDecomposer().decompose(ts, period=12)
>>> r.method in ("stl", "stl-fallback")
True
>>> r.trend.n == ts.n
True
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tseda.core.timeseries import TimeSeries
from tseda.core.validator import validate_positive_int
from tseda.decomposition.classical import (
    DecompositionResult,
    _default_period,
    _seasonal_and_residual_additive,
    _strength,
    _wrap,
)

__all__ = ["STLDecomposer"]


# ---------------------------------------------------------------------------
# Fallback implementation (scipy only)
# ---------------------------------------------------------------------------


def _savgol_trend(y: np.ndarray, period: int) -> np.ndarray:
    """Estimate trend using a Savitzky-Golay smoother.

    The window length is chosen as the largest odd number ≤ ``period * 1.5``
    that is at most ``n``.  Polynomial order is 2.
    """
    from scipy.signal import savgol_filter

    n = len(y)
    window = int(period * 1.5)
    if window % 2 == 0:
        window += 1                          # must be odd
    window = min(window, n if n % 2 == 1 else n - 1)
    window = max(window, 3)                  # minimum valid window
    poly   = min(2, window - 1)

    # savgol_filter handles NaN poorly; replace with linear interpolation first
    s = np.copy(y).astype(float)
    nan_mask = np.isnan(s)
    if nan_mask.any():
        idx_arr = np.arange(n)
        s[nan_mask] = np.interp(idx_arr[nan_mask], idx_arr[~nan_mask], s[~nan_mask])

    return savgol_filter(s, window_length=window, polyorder=poly)


def _stl_fallback(
    ts: TimeSeries, period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simplified iterative decomposition used when statsmodels is absent.

    Returns (trend, seasonal, residual) arrays.
    """
    y = ts.values.copy()

    # Iteration 1 — coarse trend via Savitzky-Golay
    T = _savgol_trend(y, period)
    S, R = _seasonal_and_residual_additive(y, T, period)

    # Iteration 2 — refine trend on de-seasonalised series
    T = _savgol_trend(y - S, period)
    S, R = _seasonal_and_residual_additive(y, T, period)

    return T, S, R


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------


class STLDecomposer:
    """Decompose a :class:`~tseda.core.TimeSeries` using the STL algorithm.

    The decomposer is **stateless** — one instance may be reused.

    Methods
    -------
    decompose(ts, period, robust, seasonal_deg, trend_deg)
        Return a :class:`~tseda.decomposition.classical.DecompositionResult`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.decomposition.stl import STLDecomposer

    >>> rng  = np.random.default_rng(2)
    >>> n    = 48
    >>> seas = np.tile(np.sin(2 * np.pi * np.arange(12) / 12) * 8, 4)
    >>> y    = np.linspace(0, 10, n) + seas + rng.standard_normal(n) * 0.3
    >>> idx  = pd.date_range("2020-01", periods=n, freq="MS")
    >>> ts   = TimeSeries(y, index=idx)
    >>> r    = STLDecomposer().decompose(ts, period=12)
    >>> r.strength_seasonal > 0.7
    True
    """

    def decompose(
        self,
        ts: TimeSeries,
        period: Optional[int] = None,
        *,
        robust: bool = True,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
    ) -> DecompositionResult:
        """Decompose *ts* using STL.

        Parameters
        ----------
        ts : TimeSeries
            Input series.  Must be regularly spaced and length
            ``>= 2 × period``.
        period : int, optional
            Seasonal period.  Inferred from ``ts.freq`` when omitted.
        robust : bool, optional
            When ``True`` (default), use robust LOESS fitting to
            down-weight outliers.  Has no effect on the fallback path.
        seasonal_deg : int, optional
            Polynomial degree for seasonal LOESS smoother (0 or 1).
            Passed to :class:`statsmodels.tsa.seasonal.STL`.  Default 1.
        trend_deg : int, optional
            Polynomial degree for trend LOESS smoother (0 or 1).
            Default 1.

        Returns
        -------
        DecompositionResult
            :attr:`~DecompositionResult.method` is ``"stl"`` when
            statsmodels is used, or ``"stl-fallback"`` otherwise.

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *period* cannot be inferred, is < 2, or the series is too
            short.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.decomposition.stl import STLDecomposer
        >>> rng = np.random.default_rng(3)
        >>> idx = pd.date_range("2020-01", periods=36, freq="MS")
        >>> seas = np.tile(np.arange(12, dtype=float), 3)
        >>> ts  = TimeSeries(seas + rng.standard_normal(36) * 0.2, index=idx)
        >>> r   = STLDecomposer().decompose(ts, period=12)
        >>> r.residual.n
        36
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
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
                f"({2 * period}) for STL decomposition."
            )

        y = ts.values

        # --- primary path: statsmodels STL ---
        method = "stl"
        try:
            from statsmodels.tsa.seasonal import STL as _STL

            # statsmodels STL does not accept NaN; fill forward then back
            import pandas as pd

            s_filled = (
                ts.to_series()
                .ffill()
                .bfill()
            )
            stl_res = _STL(
                s_filled,
                period=period,
                robust=robust,
                seasonal_deg=seasonal_deg,
                trend_deg=trend_deg,
            ).fit()

            T = stl_res.trend
            S = stl_res.seasonal
            R = stl_res.resid

            # Restore NaN at positions where original was NaN
            nan_positions = np.isnan(y)
            T[nan_positions] = np.nan
            S[nan_positions] = np.nan
            R[nan_positions] = np.nan

        except ImportError:
            method = "stl-fallback"
            T, S, R = _stl_fallback(ts, period)

        # --- strength metrics ---
        st = _strength(R, T + R)
        ss = _strength(R, S + R)

        valid_mask = ~np.isnan(R)
        n_used = int(valid_mask.sum())

        return DecompositionResult(
            original=ts,
            trend=_wrap(T, ts, "trend"),
            seasonal=_wrap(S, ts, "seasonal"),
            residual=_wrap(R, ts, "residual"),
            period=period,
            model="additive",
            method=method,
            strength_trend=round(st, 6),
            strength_seasonal=round(ss, 6),
            n_obs_used=n_used,
        )