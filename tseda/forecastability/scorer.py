"""
Forecastability scoring for time series.

Computes a composite 0–100 readiness score from six diagnostic sub-scores
and recommends a modelling strategy.

Sub-scores and weights
----------------------

+-------------------+--------+---------------------------------------------+
| Sub-score         | Weight | Measures                                    |
+===================+========+=============================================+
| data_quality      |  20 %  | Inverse of missing % + IQR-outlier %        |
+-------------------+--------+---------------------------------------------+
| stationarity      |  15 %  | ADF + KPSS combined verdict                 |
+-------------------+--------+---------------------------------------------+
| signal_to_noise   |  20 %  | STL/classical strength_trend + strength_sea |
+-------------------+--------+---------------------------------------------+
| autocorrelation   |  15 %  | Max |ACF| at significant lags               |
+-------------------+--------+---------------------------------------------+
| sample_size       |  15 %  | n / (2 × period) — enough seasonal cycles   |
+-------------------+--------+---------------------------------------------+
| regularity        |  15 %  | is_regular flag + absence of large gaps      |
+-------------------+--------+---------------------------------------------+

Classes
-------
ForecastabilityReport
    Frozen dataclass returned by :meth:`ForecastabilityScorer.score`.
ForecastabilityScorer
    Stateless scorer.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.forecastability.scorer import ForecastabilityScorer

Simple AR(1) process — moderate forecastability:

>>> rng = np.random.default_rng(0)
>>> n   = 300
>>> idx = pd.date_range("2020-01-01", periods=n, freq="D")
>>> eps = rng.standard_normal(n)
>>> x   = np.zeros(n)
>>> for i in range(1, n): x[i] = 0.7 * x[i-1] + eps[i]
>>> ts  = TimeSeries(x, index=idx)
>>> r   = ForecastabilityScorer().score(ts)
>>> 0 <= r.score <= 100
True
>>> r.recommended_model in ("ARIMA", "SARIMA", "ETS", "Prophet", "ML")
True
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tseda.core.timeseries import TimeSeries
from tseda.statistics.autocorrelation import AutocorrelationAnalyzer
from tseda.statistics.stationarity import StationarityTester

__all__ = ["ForecastabilityReport", "ForecastabilityScorer"]

_WEIGHTS: Dict[str, float] = {
    "data_quality":    0.20,
    "stationarity":    0.15,
    "signal_to_noise": 0.20,
    "autocorrelation": 0.15,
    "sample_size":     0.15,
    "regularity":      0.15,
}

assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastabilityReport:
    """Immutable forecastability assessment.

    Attributes
    ----------
    score : float
        Overall forecastability score in [0, 100].  Higher is better.
    sub_scores : dict of str → float
        Individual sub-scores (0–100 each) keyed by sub-score name.
    recommended_model : str
        Suggested modelling approach: ``"ARIMA"``, ``"SARIMA"``, ``"ETS"``,
        ``"Prophet"``, or ``"ML"``.
    recommended_diff : int
        Recommended differencing order: 0 (already stationary) or 1.
    recommended_period : int or None
        Dominant seasonal period detected, or ``None`` if no seasonality found.
    n_obs : int
        Number of observations in the series.
    pct_missing : float
        Percentage of NaN values.
    pct_outlier : float
        Percentage of IQR-flagged outliers.
    is_stationary : bool
        ``True`` when the ADF test rejects the unit-root null.
    dominant_period : int or None
        Same as :attr:`recommended_period`.
    """

    score: float
    sub_scores: Dict[str, float]
    recommended_model: str
    recommended_diff: int
    recommended_period: Optional[int]
    n_obs: int
    pct_missing: float
    pct_outlier: float
    is_stationary: bool
    dominant_period: Optional[int]

    def __repr__(self) -> str:  # pragma: no cover
        lines = [
            "ForecastabilityReport(",
            f"  score            : {self.score:.1f}/100",
            "  sub_scores       :",
        ]
        for k, v in self.sub_scores.items():
            lines.append(f"    {k:<20}: {v:.1f}")
        lines += [
            f"  recommended_model: {self.recommended_model}",
            f"  recommended_diff : {self.recommended_diff}",
            f"  recommended_period: {self.recommended_period}",
            f"  n_obs            : {self.n_obs}",
            f"  pct_missing      : {self.pct_missing:.2f} %",
            f"  pct_outlier      : {self.pct_outlier:.2f} %",
            f"  is_stationary    : {self.is_stationary}",
            ")",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _pct_outlier_iqr(x: np.ndarray) -> float:
    """Return percentage of values outside 1.5×IQR fences (no NaN)."""
    if len(x) == 0:
        return 0.0
    q1, q3 = float(np.percentile(x, 25)), float(np.percentile(x, 75))
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    fence_lo = q1 - 1.5 * iqr
    fence_hi = q3 + 1.5 * iqr
    n_out = int(np.sum((x < fence_lo) | (x > fence_hi)))
    return n_out / len(x) * 100.0


def _stl_strengths(
    x: np.ndarray, period: int
) -> tuple[float, float]:
    """Return (strength_trend, strength_seasonal) via STL or fallback.

    Both values are in [0, 1].  Uses statsmodels STL when available;
    falls back to a Savitzky-Golay trend + per-phase seasonal average.
    """
    n = len(x)
    if n < 2 * period:
        return 0.0, 0.0

    try:
        from statsmodels.tsa.seasonal import STL

        result = STL(x, period=period, robust=True).fit()
        trend_comp = result.trend
        seasonal_comp = result.seasonal
        resid = result.resid
    except ImportError:
        from scipy.signal import savgol_filter

        wl = min(period * 2 + 1, n if n % 2 == 1 else n - 1)
        if wl < 3:
            wl = 3
        wl = wl if wl % 2 == 1 else wl + 1
        trend_comp = savgol_filter(x, window_length=min(wl, n if n % 2 == 1 else n - 1), polyorder=2)
        detrended = x - trend_comp
        seasonal_comp = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            seasonal_comp[indices] = np.mean(detrended[indices])
        resid = x - trend_comp - seasonal_comp

    var_resid = float(np.var(resid))
    var_x = float(np.var(x))

    strength_trend = max(0.0, 1.0 - var_resid / max(np.var(x - seasonal_comp), 1e-15))
    strength_seasonal = max(0.0, 1.0 - var_resid / max(np.var(x - trend_comp), 1e-15))

    return float(np.clip(strength_trend, 0.0, 1.0)), float(np.clip(strength_seasonal, 0.0, 1.0))


def _detect_period(ts: TimeSeries) -> Optional[int]:
    """Return the dominant seasonal period via FFT periodogram, or None."""
    try:
        from tseda.seasonality.detector import SeasonalityDetector

        report = SeasonalityDetector().detect(ts)
        if report.is_seasonal and report.dominant_period is not None:
            return int(report.dominant_period)
    except Exception:
        pass
    return None


def _has_large_gaps(ts: TimeSeries) -> bool:
    """Return True if any time gap exceeds 3× the median gap."""
    if ts.n < 3:
        return False
    gaps = np.diff(ts.index.astype(np.int64))
    median_gap = float(np.median(gaps))
    if median_gap <= 0:
        return False
    return bool(np.any(gaps > 3 * median_gap))


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class ForecastabilityScorer:
    """Assess how forecastable a :class:`~tseda.core.TimeSeries` is.

    The scorer is **stateless** — calling :meth:`score` multiple times is safe.

    Methods
    -------
    score(ts, period)
        Return a :class:`ForecastabilityReport` with an overall 0–100 score.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.forecastability.scorer import ForecastabilityScorer

    >>> rng = np.random.default_rng(1)
    >>> idx = pd.date_range("2020", periods=200, freq="D")
    >>> ts  = TimeSeries(rng.standard_normal(200), index=idx)
    >>> r   = ForecastabilityScorer().score(ts)
    >>> isinstance(r.score, float)
    True
    """

    def score(
        self,
        ts: TimeSeries,
        *,
        period: Optional[int] = None,
        alpha: float = 0.05,
    ) -> ForecastabilityReport:
        """Compute the forecastability score for *ts*.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        period : int, optional
            Seasonal period.  When ``None`` the period is detected
            automatically via the FFT periodogram.
        alpha : float, optional
            Significance level used for stationarity and ACF tests.
            Default ``0.05``.

        Returns
        -------
        ForecastabilityReport

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *period* is given and is < 2, or *ts* has fewer than 4 obs.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.forecastability.scorer import ForecastabilityScorer
        >>> rng = np.random.default_rng(2)
        >>> idx = pd.date_range("2020", periods=365, freq="D")
        >>> n   = 365
        >>> seas = np.sin(2 * np.pi * np.arange(n) / 7) * 3
        >>> ts  = TimeSeries(seas + rng.standard_normal(n) * 0.5, index=idx)
        >>> r   = ForecastabilityScorer().score(ts, period=7)
        >>> r.recommended_period
        7
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )
        x_full = ts.values
        n_full = len(x_full)
        if n_full < 4:
            raise ValueError(
                "ForecastabilityScorer requires at least 4 observations."
            )
        if period is not None and period < 2:
            raise ValueError(
                f"'period' must be >= 2; got {period}."
            )

        x_clean = x_full[~np.isnan(x_full)]
        n_clean = len(x_clean)

        # ── 1. Data quality ────────────────────────────────────────────
        pct_nan = (n_full - n_clean) / n_full * 100.0
        pct_out = _pct_outlier_iqr(x_clean)
        dq_score = float(np.clip(100.0 - pct_nan - pct_out, 0.0, 100.0))

        # ── 2. Stationarity ────────────────────────────────────────────
        tester = StationarityTester()
        try:
            adf_result = tester.adf(ts, alpha=alpha)
            adf_stat = adf_result.is_stationary
        except Exception:
            adf_stat = True  # constant / zero-variance series treated as stationary
        try:
            kpss_result = tester.kpss(ts, alpha=alpha)
            kpss_stat = kpss_result.is_stationary
        except Exception:
            kpss_stat = True

        if adf_stat and kpss_stat:
            stat_score = 100.0
        elif adf_stat and not kpss_stat:
            stat_score = 75.0
        elif not adf_stat and kpss_stat:
            stat_score = 50.0
        else:
            stat_score = 0.0

        recommended_diff = 0 if adf_stat else 1

        # ── 3. Signal-to-noise ─────────────────────────────────────────
        detected_period = period if period is not None else _detect_period(ts)
        effective_period = detected_period if detected_period is not None else 12

        if n_clean >= 2 * effective_period:
            str_trend, str_seasonal = _stl_strengths(x_clean, effective_period)
        else:
            str_trend = float(
                np.clip(1.0 - np.var(np.diff(x_clean)) / max(np.var(x_clean), 1e-15), 0.0, 1.0)
            )
            str_seasonal = 0.0

        sn_score = float(np.clip((str_trend + str_seasonal) / 2.0 * 100.0, 0.0, 100.0))

        # ── 4. Autocorrelation ────────────────────────────────────────
        max_lag = min(40, n_clean // 2 - 1)
        if max_lag >= 1 and n_clean >= 4:
            acf_result = AutocorrelationAnalyzer().analyze(ts, lags=max_lag, alpha=alpha)
            acf_vals = acf_result.acf[1:]
            ci = float(acf_result.conf_upper[1])
            sig_mask = np.abs(acf_vals) > ci
            if sig_mask.any():
                max_acf = float(np.max(np.abs(acf_vals[sig_mask])))
            else:
                max_acf = 0.0
        else:
            max_acf = 0.0
        ac_score = float(np.clip(max_acf * 100.0, 0.0, 100.0))

        # ── 5. Sample size ────────────────────────────────────────────
        ratio = n_clean / (2.0 * effective_period)
        ss_score = float(np.clip(ratio / 5.0 * 100.0, 0.0, 100.0))

        # ── 6. Regularity ─────────────────────────────────────────────
        is_reg = ts.is_regular
        has_gaps = _has_large_gaps(ts)
        reg_score = 50.0 * float(is_reg) + 50.0 * float(not has_gaps)

        # ── Weighted overall score ────────────────────────────────────
        sub_scores: Dict[str, float] = {
            "data_quality":    round(dq_score, 2),
            "stationarity":    round(stat_score, 2),
            "signal_to_noise": round(sn_score, 2),
            "autocorrelation": round(ac_score, 2),
            "sample_size":     round(ss_score, 2),
            "regularity":      round(reg_score, 2),
        }
        overall = sum(_WEIGHTS[k] * v for k, v in sub_scores.items())
        overall = float(np.clip(overall, 0.0, 100.0))

        # ── Recommended model ─────────────────────────────────────────
        is_seasonal = detected_period is not None
        recommended_model = _recommend_model(
            is_seasonal=is_seasonal,
            is_stationary=adf_stat,
            n_obs=n_clean,
            overall_score=overall,
        )

        return ForecastabilityReport(
            score=round(overall, 2),
            sub_scores=sub_scores,
            recommended_model=recommended_model,
            recommended_diff=recommended_diff,
            recommended_period=detected_period,
            n_obs=n_full,
            pct_missing=round(pct_nan, 4),
            pct_outlier=round(pct_out, 4),
            is_stationary=adf_stat,
            dominant_period=detected_period,
        )


def _recommend_model(
    *,
    is_seasonal: bool,
    is_stationary: bool,
    n_obs: int,
    overall_score: float,
) -> str:
    """Select a recommended modelling approach from a fixed set.

    Decision rules (evaluated in order):

    1. Very low score (< 25) → ``"ML"`` (non-linear, complex patterns)
    2. Seasonal + stationary → ``"SARIMA"``
    3. Seasonal + non-stationary → ``"ETS"`` (handles trend/level updates)
    4. Non-seasonal + stationary → ``"ARIMA"``
    5. Long non-seasonal series with trend → ``"Prophet"``
    6. Fallback → ``"ARIMA"``
    """
    if overall_score < 25.0:
        return "ML"
    if is_seasonal and is_stationary:
        return "SARIMA"
    if is_seasonal and not is_stationary:
        return "ETS"
    if not is_seasonal and is_stationary:
        return "ARIMA"
    if not is_seasonal and not is_stationary and n_obs >= 100:
        return "Prophet"
    return "ARIMA"