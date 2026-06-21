"""
Seasonality detection for time series.

Three detection strategies are available, all implemented in pure
numpy / scipy:

+---------------+--------------------------------------+-------------------------------+
| Method        | Mechanism                            | Best for                      |
+===============+======================================+===============================+
| periodogram   | FFT power spectrum peaks             | Any length; fast              |
+---------------+--------------------------------------+-------------------------------+
| acf           | Autocorrelation peaks above 95 % CI  | Short to medium series        |
+---------------+--------------------------------------+-------------------------------+
| combined      | Both methods with agreement bonus    | General use (default)         |
+---------------+--------------------------------------+-------------------------------+

A Fisher G-test is always computed on the periodogram and contributes to the
:attr:`~SeasonalityReport.is_seasonal` verdict.

Classes
-------
SeasonalityReport
    Frozen dataclass returned by :meth:`SeasonalityDetector.detect`.
SeasonalityDetector
    Stateless detector.

Examples
--------
Monthly data with a clear 12-month season:

>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.seasonality.detector import SeasonalityDetector

>>> rng  = np.random.default_rng(0)
>>> n    = 72
>>> seas = np.tile(np.sin(2 * np.pi * np.arange(12) / 12) * 6, 6)
>>> y    = 50 + np.linspace(0, 4, n) + seas + rng.standard_normal(n) * 0.3
>>> idx  = pd.date_range("2018-01", periods=n, freq="MS")
>>> ts   = TimeSeries(y, index=idx, name="sales")
>>> r    = SeasonalityDetector().detect(ts)
>>> r.dominant_period
12
>>> r.is_seasonal
True
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats

from tseda.core.timeseries import TimeSeries
from tseda.core.validator import validate_positive_int

__all__ = ["SeasonalityReport", "SeasonalityDetector"]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonalityReport:
    """Immutable seasonality detection result.

    Attributes
    ----------
    dominant_period : int or None
        The single most likely seasonal period (in observations), or
        ``None`` if no seasonality was detected.
    candidate_periods : list of (int, float)
        All detected candidate periods as ``(period, score)`` pairs
        sorted by score descending.  Score is normalised to [0, 1].
    is_seasonal : bool
        ``True`` when the evidence for a dominant seasonal period is
        statistically significant at :attr:`alpha`.
    method : str
        Detection method used: ``"periodogram"``, ``"acf"``, or
        ``"combined"``.
    n_obs : int
        Number of non-NaN observations used.
    alpha : float
        Significance level.
    periodogram_periods : list of (int, float)
        Top-k candidate periods from the FFT periodogram,
        as ``(period, normalised_power)`` pairs.
    acf_periods : list of (int, float)
        Candidate periods from significant ACF peaks,
        as ``(period, acf_value)`` pairs.
    fisher_g_stat : float
        Fisher's G test statistic for the dominant periodogram peak.
    fisher_p_value : float
        P-value of the Fisher G test.  Small values (< alpha) indicate
        that at least one spectral peak is too strong to be explained by
        white noise.
    strength_scores : dict of int → float
        Normalised combined score for every candidate period.
    """

    dominant_period: Optional[int]
    candidate_periods: List[Tuple[int, float]]
    is_seasonal: bool
    method: str
    n_obs: int
    alpha: float
    periodogram_periods: List[Tuple[int, float]]
    acf_periods: List[Tuple[int, float]]
    fisher_g_stat: float
    fisher_p_value: float
    strength_scores: Dict[int, float]

    def summary(self) -> str:
        """Return a plain-text summary.

        Returns
        -------
        str
        """
        top = "\n".join(
            f"    period={p:>4d}  score={s:.4f}"
            for p, s in self.candidate_periods[:5]
        )
        return (
            f"SeasonalityReport\n"
            f"{'─' * 42}\n"
            f"  method          : {self.method}\n"
            f"  n_obs           : {self.n_obs}\n"
            f"  is_seasonal     : {self.is_seasonal}  (α={self.alpha})\n"
            f"  dominant_period : {self.dominant_period}\n"
            f"  Fisher G        : stat={self.fisher_g_stat:.4f}  "
            f"p={self.fisher_p_value:.4f}\n"
            f"  top candidates  :\n{top}\n"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clean(x: np.ndarray) -> np.ndarray:
    """Replace NaN by linear interpolation; detrend and demean."""
    x = x.astype(float).copy()
    n = len(x)
    nan_mask = np.isnan(x)
    if nan_mask.any():
        idx = np.arange(n)
        x[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x[~nan_mask])
    x = sp_signal.detrend(x, type="linear")
    return x


def _fisher_g(power: np.ndarray) -> Tuple[float, float]:
    """Fisher's G test: ``G = max(I) / sum(I)``.

    P-value approximation (Percival & Walden 1993):
    ``p ≈ 1 − (1 − exp(−m·G))^m``

    Parameters
    ----------
    power : numpy.ndarray
        Periodogram ordinates (excluding the DC component at index 0).

    Returns
    -------
    (g_stat, p_value)
    """
    m = len(power)
    if m == 0 or power.sum() == 0:
        return 0.0, 1.0
    g = float(power.max() / power.sum())
    p = float(1.0 - (1.0 - np.exp(-g * m)) ** m)
    return g, min(p, 1.0)


def _periodogram_detect(
    x: np.ndarray,
    min_period: int,
    max_period: int,
    top_k: int,
    alpha: float,
) -> Tuple[List[Tuple[int, float]], float, float]:
    """FFT periodogram peak detection.

    Returns
    -------
    (candidates, fisher_g, fisher_p)
        *candidates* is a list of ``(period, normalised_power)`` sorted by
        power descending.
    """
    n = len(x)
    if n < 4:
        return [], 0.0, 1.0

    # Hann window to reduce spectral leakage
    window = np.hanning(n)
    x_win  = x * window

    # FFT power spectrum (one-sided)
    fft_vals = np.fft.rfft(x_win)
    power    = np.abs(fft_vals) ** 2

    # DC (index 0) excluded from peak search and G-test
    power_no_dc = power[1:]

    g_stat, g_pval = _fisher_g(power_no_dc)

    # Convert frequency indices → periods
    # freq[k] = k / n  →  period = n / k
    n_freqs = len(power)
    freq_idx = np.arange(1, n_freqs)  # skip DC
    with np.errstate(divide="ignore"):
        raw_periods = n / freq_idx.astype(float)

    # Find local maxima in the power spectrum (excluding DC)
    peaks, _ = sp_signal.find_peaks(power_no_dc, height=0)

    if len(peaks) == 0:
        return [], g_stat, g_pval

    # Map each peak to the nearest integer period, filter by [min_period, max_period]
    candidates_raw: Dict[int, float] = {}
    for pk in peaks:
        raw_p = raw_periods[pk]
        for p_int in [int(np.floor(raw_p)), int(np.ceil(raw_p))]:
            if min_period <= p_int <= max_period:
                pw = float(power_no_dc[pk])
                if p_int not in candidates_raw or pw > candidates_raw[p_int]:
                    candidates_raw[p_int] = pw

    if not candidates_raw:
        return [], g_stat, g_pval

    # Normalise power to [0, 1]
    max_pw = max(candidates_raw.values())
    norm   = {p: pw / max_pw for p, pw in candidates_raw.items()}

    # Sort by power descending, return top-k
    sorted_cands = sorted(norm.items(), key=lambda t: -t[1])[:top_k]
    return [(p, s) for p, s in sorted_cands], g_stat, g_pval


def _acf_detect(
    x: np.ndarray,
    max_lag: int,
    alpha: float,
    top_k: int,
) -> List[Tuple[int, float]]:
    """ACF-peak-based seasonality detection.

    Returns
    -------
    list of (lag, acf_value)
        Significant positive ACF peaks, sorted by ACF value descending.
    """
    n = len(x)
    if max_lag < 2 or n < 4:
        return []

    # Compute biased ACF at lags 1 … max_lag
    xc    = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom == 0:
        return []

    acf = np.array([float(np.dot(xc[k:], xc[:-k])) / denom for k in range(1, max_lag + 1)])

    # Bartlett 95 % CI for white noise
    z_crit = float(sp_stats.norm.ppf(1 - alpha / 2))
    ci     = z_crit / np.sqrt(n)

    # Find positive peaks above the CI with minimum spacing of 2 lags
    peaks, _ = sp_signal.find_peaks(acf, height=ci, distance=2)

    if len(peaks) == 0:
        return []

    # Lag number is peak index + 1 (lags start at 1)
    result = [(int(pk + 1), float(acf[pk])) for pk in peaks]
    result.sort(key=lambda t: -t[1])
    return result[:top_k]


def _combine_scores(
    periodogram: List[Tuple[int, float]],
    acf: List[Tuple[int, float]],
) -> Dict[int, float]:
    """Merge periodogram and ACF scores with an agreement bonus.

    When both methods agree on the same period the combined score gets a
    20 % boost (capped at 1.0).
    """
    pg_dict  = dict(periodogram)
    acf_dict = dict(acf)
    all_periods = set(pg_dict) | set(acf_dict)

    combined: Dict[int, float] = {}
    for p in all_periods:
        ps = pg_dict.get(p, 0.0)
        as_ = acf_dict.get(p, 0.0)
        if ps > 0 and as_ > 0:
            # Both methods detected this period → agreement bonus
            combined[p] = min(1.0, (ps + as_) / 2.0 * 1.2)
        else:
            # Single-method detection → slight discount
            combined[p] = max(ps, as_) * 0.9

    return combined


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class SeasonalityDetector:
    """Detect seasonal periods in a :class:`~tseda.core.TimeSeries`.

    The detector is **stateless** — one instance, many series.

    Methods
    -------
    detect(ts, method, top_k, alpha, min_period, max_period)
        Return a :class:`SeasonalityReport`.
    test_period(ts, period, alpha)
        Test a specific period for significance.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.seasonality.detector import SeasonalityDetector

    Weekly pattern in daily data:

    >>> rng  = np.random.default_rng(1)
    >>> n    = 140
    >>> seas = np.tile(np.array([0, 1, 2, 2, 1, -1, -2], dtype=float), 20)
    >>> y    = 10 + seas + rng.standard_normal(n) * 0.2
    >>> idx  = pd.date_range("2020-01-06", periods=n, freq="D")
    >>> ts   = TimeSeries(y, index=idx)
    >>> r    = SeasonalityDetector().detect(ts)
    >>> r.dominant_period
    7
    """

    def detect(
        self,
        ts: TimeSeries,
        method: str = "combined",
        *,
        top_k: int = 5,
        alpha: float = 0.05,
        min_period: int = 2,
        max_period: Optional[int] = None,
    ) -> SeasonalityReport:
        """Detect seasonal periods in *ts*.

        Parameters
        ----------
        ts : TimeSeries
            Input series.  NaN values are filled by linear interpolation
            before spectral analysis.
        method : str, optional
            Detection strategy:

            * ``"periodogram"`` — FFT power spectrum only.
            * ``"acf"``         — ACF peaks only.
            * ``"combined"``    — both methods with agreement bonus (default).

        top_k : int, optional
            Maximum number of candidate periods to return per method.
            Default 5.
        alpha : float, optional
            Significance level for Fisher G-test and ACF confidence
            interval.  Default 0.05.
        min_period : int, optional
            Minimum period to search for.  Default 2.
        max_period : int, optional
            Maximum period to search for.  Defaults to ``n // 2``.

        Returns
        -------
        SeasonalityReport

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *method* is not recognised, *top_k* < 1, *alpha* outside
            (0, 1), or the series is too short.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.seasonality.detector import SeasonalityDetector

        Monthly series — detect 12-month period:

        >>> rng  = np.random.default_rng(0)
        >>> idx  = pd.date_range("2018-01", periods=60, freq="MS")
        >>> seas = np.tile(np.sin(2 * np.pi * np.arange(12) / 12) * 8, 5)
        >>> ts   = TimeSeries(seas + rng.standard_normal(60) * 0.5, index=idx)
        >>> r    = SeasonalityDetector().detect(ts)
        >>> r.dominant_period
        12
        """
        # ── validate inputs ────────────────────────────────────────────
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )
        if method not in ("periodogram", "acf", "combined"):
            raise ValueError(
                f"'method' must be 'periodogram', 'acf', or 'combined'; "
                f"got {method!r}."
            )
        top_k = validate_positive_int(top_k, name="top_k")
        if not (0 < alpha < 1):
            raise ValueError(f"'alpha' must be in (0, 1); got {alpha}.")
        min_period = validate_positive_int(min_period, name="min_period")
        if min_period < 2:
            raise ValueError(f"'min_period' must be >= 2; got {min_period}.")

        # ── clean series ───────────────────────────────────────────────
        x = ts.values[~np.isnan(ts.values)]
        n = len(x)
        if n < 8:
            raise ValueError(
                f"Seasonality detection requires at least 8 non-NaN "
                f"observations; got {n}."
            )

        max_p  = max_period if max_period is not None else n // 2
        max_p  = max(min_period, min(max_p, n // 2))

        x_clean = _clean(x)

        # ── run selected method(s) ─────────────────────────────────────
        pg_periods: List[Tuple[int, float]] = []
        acf_periods: List[Tuple[int, float]] = []
        g_stat = 0.0
        g_pval = 1.0

        if method in ("periodogram", "combined"):
            pg_periods, g_stat, g_pval = _periodogram_detect(
                x_clean, min_period, max_p, top_k, alpha
            )

        if method in ("acf", "combined"):
            acf_periods = _acf_detect(x_clean, max_p, alpha, top_k)

        # ── build combined score dict ──────────────────────────────────
        if method == "periodogram":
            score_dict = dict(pg_periods)
        elif method == "acf":
            # Normalise ACF values to [0, 1]
            if acf_periods:
                max_acf = max(v for _, v in acf_periods)
                score_dict = {
                    p: v / max_acf for p, v in acf_periods
                } if max_acf > 0 else {}
            else:
                score_dict = {}
        else:  # combined
            # Normalise ACF values to [0, 1] for combining
            if acf_periods:
                max_acf = max(v for _, v in acf_periods)
                acf_norm = [(p, v / max_acf) for p, v in acf_periods] if max_acf > 0 else []
            else:
                acf_norm = []
            score_dict = _combine_scores(pg_periods, acf_norm)

        # ── dominant period ────────────────────────────────────────────
        if score_dict:
            dominant_period: Optional[int] = max(score_dict, key=lambda p: score_dict[p])
            dom_score = score_dict[dominant_period]
        else:
            dominant_period = None
            dom_score = 0.0

        # ── is_seasonal decision ───────────────────────────────────────
        # Significant if:
        #   - periodogram: Fisher G p-value < alpha
        #   - acf: at least one peak above CI
        #   - combined: either of the above
        fisher_sig = g_pval < alpha
        acf_sig    = len(acf_periods) > 0
        dom_strong = dom_score > 0.05   # guard against near-zero scores

        if method == "periodogram":
            is_seasonal = fisher_sig and dom_strong
        elif method == "acf":
            is_seasonal = acf_sig and dom_strong
        else:
            is_seasonal = (fisher_sig or acf_sig) and dom_strong

        # ── build sorted candidate list ────────────────────────────────
        candidates = sorted(score_dict.items(), key=lambda t: -t[1])

        return SeasonalityReport(
            dominant_period=dominant_period if is_seasonal else None,
            candidate_periods=candidates,
            is_seasonal=is_seasonal,
            method=method,
            n_obs=n,
            alpha=alpha,
            periodogram_periods=pg_periods,
            acf_periods=acf_periods,
            fisher_g_stat=g_stat,
            fisher_p_value=g_pval,
            strength_scores=score_dict,
        )

    def test_period(
        self,
        ts: TimeSeries,
        period: int,
        *,
        alpha: float = 0.05,
    ) -> dict:
        """Test whether a specific *period* is present in *ts*.

        Runs both the periodogram and ACF detectors and checks whether the
        requested period appears among their significant candidates.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        period : int
            The period to test (must be >= 2).
        alpha : float, optional
            Significance level.  Default 0.05.

        Returns
        -------
        dict
            Keys:

            * ``"period"`` — the period tested.
            * ``"detected"`` — ``True`` if the period was found by at least
              one method.
            * ``"periodogram_detected"`` — ``True`` if it appears in the FFT
              peaks.
            * ``"acf_detected"`` — ``True`` if the ACF at this lag is a
              significant positive peak.
            * ``"strength"`` — combined strength score in [0, 1].
            * ``"fisher_p_value"`` — p-value of the overall Fisher G-test.

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *period* < 2.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.seasonality.detector import SeasonalityDetector

        >>> rng  = np.random.default_rng(0)
        >>> idx  = pd.date_range("2018-01", periods=60, freq="MS")
        >>> seas = np.tile(np.sin(2 * np.pi * np.arange(12) / 12) * 8, 5)
        >>> ts   = TimeSeries(seas + rng.standard_normal(60) * 0.5, index=idx)
        >>> SeasonalityDetector().test_period(ts, 12)["detected"]
        True
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )
        period = validate_positive_int(period, name="period")
        if period < 2:
            raise ValueError(f"'period' must be >= 2; got {period}.")

        r = self.detect(ts, method="combined", top_k=20, alpha=alpha)

        pg_detected  = any(p == period for p, _ in r.periodogram_periods)
        acf_detected = any(p == period for p, _ in r.acf_periods)
        strength     = r.strength_scores.get(period, 0.0)

        return {
            "period":                period,
            "detected":              pg_detected or acf_detected,
            "periodogram_detected":  pg_detected,
            "acf_detected":          acf_detected,
            "strength":              strength,
            "fisher_p_value":        r.fisher_p_value,
        }