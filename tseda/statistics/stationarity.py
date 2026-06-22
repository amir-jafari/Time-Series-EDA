"""
Stationarity testing for time series.

Three widely-used tests are implemented with a dual-path strategy:

1. **Primary path** — pure numpy / scipy implementation so the package works
   without statsmodels.
2. **Fast path** — if ``statsmodels`` is installed the well-tested
   :mod:`statsmodels.tsa.stattools` implementations are used instead,
   which have more reliable critical-value tables.

+---------+------------------+-----------------------------------+
| Test    | H₀               | Detects                           |
+=========+==================+===================================+
| ADF     | Unit root exists | Evidence *against* unit root      |
+---------+------------------+-----------------------------------+
| KPSS    | Series is level  | Evidence *of* non-stationarity    |
|         | (or trend)       |                                   |
|         | stationary       |                                   |
+---------+------------------+-----------------------------------+
| PP      | Unit root exists | Robust to serial correlation      |
|         |                  | without requiring lag selection   |
+---------+------------------+-----------------------------------+

The combined :meth:`StationarityTester.summary` method reconciles all three
tests and returns a human-readable verdict with recommended action.

Classes
-------
StationarityResult
    Frozen dataclass for a single test's output.
StationarityTester
    Stateless tester.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.statistics.stationarity import StationarityTester

Stationary white noise:

>>> rng = np.random.default_rng(42)
>>> idx = pd.date_range("2020", periods=300, freq="D")
>>> ts  = TimeSeries(rng.standard_normal(300), index=idx)
>>> r   = StationarityTester().adf(ts)
>>> r.is_stationary   # p < 0.05
True

Random walk (non-stationary):

>>> rw  = TimeSeries(np.cumsum(rng.standard_normal(300)), index=idx)
>>> r2  = StationarityTester().adf(rw)
>>> r2.is_stationary
False
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from tseda.core.timeseries import TimeSeries
from tseda.core.validator import validate_positive_int

__all__ = ["StationarityResult", "StationarityTester"]

# ---------------------------------------------------------------------------
# Critical-value lookup tables (5 % significance level)
# ---------------------------------------------------------------------------

# ADF critical values (MacKinnon 1994, n → ∞ asymptotic).
# Keys: regression type ('nc', 'c', 'ct'); values: (1%, 5%, 10%)
_ADF_CV: dict[str, tuple[float, float, float]] = {
    "nc": (-2.5658, -1.9393, -1.6156),
    "c":  (-3.4336, -2.8621, -2.5671),
    "ct": (-3.9638, -3.4126, -3.1279),
}

# KPSS critical values (Kwiatkowski et al. 1992, Table 1).
# Keys: regression type ('c', 'ct'); values: (10%, 5%, 2.5%, 1%)
_KPSS_CV: dict[str, tuple[float, float, float, float]] = {
    "c":  (0.347, 0.463, 0.574, 0.739),
    "ct": (0.119, 0.146, 0.176, 0.216),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationarityResult:
    """Immutable result of a stationarity test.

    Attributes
    ----------
    test_name : str
        Name of the test (e.g., ``"ADF"``).
    statistic : float
        Test statistic value.
    p_value : float
        Approximate p-value.
    critical_values : dict of str → float
        Critical values at standard significance levels (``"1%"``, ``"5%"``,
        ``"10%"``).
    n_lags : int or None
        Number of lags used (``None`` for tests that do not select lags).
    regression : str
        Regression type used (``"nc"``, ``"c"``, or ``"ct"``).
    is_stationary : bool
        Convenience flag.  For ADF / PP: ``p_value < alpha`` (reject unit
        root → evidence of stationarity).  For KPSS: ``p_value > alpha``
        (fail to reject stationarity null).
    alpha : float
        Significance level used to set :attr:`is_stationary`.
    interpretation : str
        One-sentence plain-English summary of the result.
    """

    test_name: str
    statistic: float
    p_value: float
    critical_values: dict
    n_lags: Optional[int]
    regression: str
    is_stationary: bool
    alpha: float
    interpretation: str

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"StationarityResult(\n"
            f"  test           : {self.test_name}\n"
            f"  statistic      : {self.statistic:.4f}\n"
            f"  p_value        : {self.p_value:.4f}\n"
            f"  is_stationary  : {self.is_stationary}  (α={self.alpha})\n"
            f"  {self.interpretation}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Private helpers — pure numpy/scipy ADF
# ---------------------------------------------------------------------------


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS: return (beta, residuals, sigma^2)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta     = XtX_inv @ X.T @ y
    resid    = y - X @ beta
    n, k     = X.shape
    s2       = float(np.dot(resid, resid) / max(n - k, 1))
    return beta, resid, s2


def _adf_native(
    x: np.ndarray,
    maxlag: int,
    regression: str,
) -> tuple[float, float, int]:
    """Pure-numpy Augmented Dickey-Fuller test.

    Returns
    -------
    (test_stat, p_value, n_lags_used)
    """
    # First-difference
    dx = np.diff(x)
    n  = len(dx)

    # Lag selection via AIC (Akaike Information Criterion)
    best_aic = np.inf
    best_lag = 0
    for lag in range(0, maxlag + 1):
        start  = lag + 1
        y      = dx[start:]
        regs   = [x[start - 1 : n - (lag if lag else 0)]]  # lagged level
        if regression in ("c", "ct"):
            regs.append(np.ones(len(y)))
        if regression == "ct":
            regs.append(np.arange(len(y), dtype=float))
        for k in range(1, lag + 1):
            regs.append(dx[start - k : n - (lag - k) if lag - k else None])
        X   = np.column_stack(regs) if len(regs) > 1 else regs[0].reshape(-1, 1)
        _, resid, s2 = _ols(y, X)
        k_params = X.shape[1]
        aic = len(y) * np.log(s2 + 1e-15) + 2 * k_params
        if aic < best_aic:
            best_aic = aic
            best_lag = lag

    # Final regression with best lag
    lag    = best_lag
    start  = lag + 1
    y      = dx[start:]
    regs   = [x[start - 1 : n - (lag if lag else 0)]]
    if regression in ("c", "ct"):
        regs.append(np.ones(len(y)))
    if regression == "ct":
        regs.append(np.arange(len(y), dtype=float))
    for k in range(1, lag + 1):
        regs.append(dx[start - k : n - (lag - k) if lag - k else None])
    X = np.column_stack(regs) if len(regs) > 1 else regs[0].reshape(-1, 1)

    beta, resid, s2 = _ols(y, X)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se      = np.sqrt(np.diag(XtX_inv) * s2)
    t_stat  = float(beta[0] / se[0]) if se[0] > 0 else float("nan")

    # Approximate p-value via MacKinnon (1994) response surface
    cv      = _ADF_CV[regression]
    # Use a simple linear interpolation over the asymptotic critical values
    # to approximate p_value (very rough but avoids dependency on tables)
    t_vals  = np.array([cv[0], cv[1], cv[2]])
    p_vals  = np.array([0.01,  0.05,  0.10])
    if t_stat <= t_vals[0]:
        p_val = 0.005
    elif t_stat >= t_vals[2]:
        p_val = 0.15
    else:
        p_val = float(np.interp(t_stat, t_vals, p_vals))

    return t_stat, p_val, lag


def _kpss_native(x: np.ndarray, regression: str) -> tuple[float, float]:
    """Pure-numpy KPSS test.

    Returns
    -------
    (test_stat, p_value)
    """
    n = len(x)
    t = np.arange(1, n + 1, dtype=float)

    if regression == "c":
        resid = x - np.mean(x)
    else:  # ct
        X     = np.column_stack([np.ones(n), t])
        beta, resid, _ = _ols(x, X)

    # Partial sums
    S    = np.cumsum(resid)
    # Long-run variance via Bartlett kernel (bandwidth = int(4*(n/100)^0.25))
    bw   = max(1, int(4 * (n / 100) ** 0.25))
    lrv  = float(np.dot(resid, resid) / n)
    for j in range(1, bw + 1):
        w    = 1 - j / (bw + 1)
        auto = float(np.dot(resid[j:], resid[:-j]) / n)
        lrv += 2 * w * auto

    stat = float(np.sum(S ** 2) / (n ** 2 * lrv)) if lrv > 0 else float("nan")

    # p-value approximation via critical-value interpolation
    cv_table = _KPSS_CV[regression]   # (10%, 5%, 2.5%, 1%)
    cv_stats = np.array(cv_table)
    cv_ps    = np.array([0.10, 0.05, 0.025, 0.01])

    if stat < cv_stats[0]:
        p_val = 0.15
    elif stat > cv_stats[-1]:
        p_val = 0.005
    else:
        p_val = float(np.interp(stat, cv_stats, cv_ps))

    return stat, p_val


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------


class StationarityTester:
    """Test a :class:`~tseda.core.TimeSeries` for stationarity.

    All methods return a :class:`StationarityResult` and are stateless.

    Methods
    -------
    adf(ts, maxlag, regression, alpha)
        Augmented Dickey-Fuller test.
    kpss(ts, regression, alpha)
        KPSS test.
    pp(ts, regression, alpha)
        Phillips-Perron test (delegates to statsmodels if available).
    summary(ts, alpha)
        Run ADF + KPSS and return a combined verdict string.

    Notes
    -----
    When ``statsmodels`` is installed the ``adf`` and ``kpss`` methods
    automatically use its implementations, which have more accurate
    critical-value tables.  Install with ``pip install statsmodels``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.statistics.stationarity import StationarityTester

    >>> rng = np.random.default_rng(0)
    >>> idx = pd.date_range("2020", periods=200, freq="D")
    >>> ts  = TimeSeries(rng.standard_normal(200), index=idx)
    >>> r   = StationarityTester().adf(ts)
    >>> r.is_stationary
    True
    """

    @staticmethod
    def _validate(ts: object) -> TimeSeries:
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )
        ts = ts  # type: ignore[assignment]
        assert isinstance(ts, TimeSeries)
        x = ts.values[~np.isnan(ts.values)]
        if len(x) < 10:
            raise ValueError(
                "Stationarity tests require at least 10 non-NaN observations."
            )
        return ts

    # ------------------------------------------------------------------
    # ADF
    # ------------------------------------------------------------------

    def adf(
        self,
        ts: TimeSeries,
        *,
        maxlag: Optional[int] = None,
        regression: str = "c",
        alpha: float = 0.05,
    ) -> StationarityResult:
        """Augmented Dickey-Fuller unit-root test.

        **H₀**: The series has a unit root (is non-stationary).
        **H₁**: The series is stationary.

        Reject H₀ (small p-value) → evidence of stationarity.

        Parameters
        ----------
        ts : TimeSeries
            Input series.  NaN values are dropped before testing.
        maxlag : int, optional
            Maximum lag to consider for AIC-based lag selection.
            Defaults to ``int(12 * (n / 100) ** 0.25)`` (Schwert 1989).
        regression : str, optional
            Deterministic terms to include in the test equation.

            * ``"nc"`` — no constant, no trend.
            * ``"c"``  — constant only (default).
            * ``"ct"`` — constant + linear trend.

        alpha : float, optional
            Significance level for :attr:`~StationarityResult.is_stationary`.
            Default ``0.05``.

        Returns
        -------
        StationarityResult

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If fewer than 10 non-NaN observations or *regression* is invalid.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.statistics.stationarity import StationarityTester
        >>> rng = np.random.default_rng(1)
        >>> idx = pd.date_range("2020", periods=150, freq="D")
        >>> ts  = TimeSeries(rng.standard_normal(150), index=idx)
        >>> StationarityTester().adf(ts).is_stationary
        True
        """
        if regression not in ("nc", "c", "ct"):
            raise ValueError(
                f"'regression' must be one of 'nc', 'c', 'ct'; got {regression!r}."
            )
        ts  = self._validate(ts)
        x   = ts.values[~np.isnan(ts.values)]
        n   = len(x)
        ml  = maxlag if maxlag is not None else int(12 * (n / 100) ** 0.25)

        # --- try statsmodels fast path ---
        try:
            import warnings as _w
            from statsmodels.tsa.stattools import adfuller as sm_adf

            with _w.catch_warnings():
                _w.simplefilter("ignore")
                res = sm_adf(x, maxlag=ml, regression=regression, autolag="AIC")
            stat, p_val, n_lags = float(res[0]), float(res[1]), int(res[2])
            cv = {k: float(v) for k, v in res[4].items()}
        except ImportError:
            stat, p_val, n_lags = _adf_native(x, ml, regression)
            _cv = _ADF_CV[regression]
            cv  = {"1%": _cv[0], "5%": _cv[1], "10%": _cv[2]}

        is_stat = p_val < alpha
        if is_stat:
            interp = (
                f"p={p_val:.4f} < α={alpha}: reject H₀ — evidence of stationarity."
            )
        else:
            interp = (
                f"p={p_val:.4f} ≥ α={alpha}: fail to reject H₀ — "
                "series may have a unit root (non-stationary)."
            )

        return StationarityResult(
            test_name="ADF",
            statistic=stat,
            p_value=p_val,
            critical_values=cv,
            n_lags=n_lags,
            regression=regression,
            is_stationary=is_stat,
            alpha=alpha,
            interpretation=interp,
        )

    # ------------------------------------------------------------------
    # KPSS
    # ------------------------------------------------------------------

    def kpss(
        self,
        ts: TimeSeries,
        *,
        regression: str = "c",
        alpha: float = 0.05,
    ) -> StationarityResult:
        """Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity test.

        **H₀**: The series is level (or trend) stationary.
        **H₁**: The series has a unit root.

        Fail to reject H₀ (large p-value) → evidence of stationarity.
        This is the *opposite* null from ADF.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        regression : str, optional
            ``"c"`` — test for level stationarity (default).
            ``"ct"`` — test for trend stationarity.
        alpha : float, optional
            Significance level.  Default ``0.05``.

        Returns
        -------
        StationarityResult

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.statistics.stationarity import StationarityTester
        >>> rng = np.random.default_rng(2)
        >>> idx = pd.date_range("2020", periods=150, freq="D")
        >>> ts  = TimeSeries(rng.standard_normal(150), index=idx)
        >>> StationarityTester().kpss(ts).is_stationary
        True
        """
        if regression not in ("c", "ct"):
            raise ValueError(
                f"'regression' must be 'c' or 'ct' for KPSS; got {regression!r}."
            )
        ts = self._validate(ts)
        x  = ts.values[~np.isnan(ts.values)]

        # --- try statsmodels fast path ---
        try:
            import warnings as _w
            from statsmodels.tsa.stattools import kpss as sm_kpss

            with _w.catch_warnings():
                _w.simplefilter("ignore")
                stat, p_val, n_lags, cv_dict = sm_kpss(x, regression=regression)
            stat    = float(stat)
            p_val   = float(p_val)
            n_lags  = int(n_lags)
            cv      = {k: float(v) for k, v in cv_dict.items()}
        except ImportError:
            stat, p_val = _kpss_native(x, regression)
            _cv  = _KPSS_CV[regression]
            cv   = {"10%": _cv[0], "5%": _cv[1], "2.5%": _cv[2], "1%": _cv[3]}
            n_lags = None  # type: ignore[assignment]

        # KPSS: is_stationary = fail to reject H₀ = p > alpha
        is_stat = p_val > alpha
        if is_stat:
            interp = (
                f"p={p_val:.4f} > α={alpha}: fail to reject H₀ — "
                "evidence of stationarity."
            )
        else:
            interp = (
                f"p={p_val:.4f} ≤ α={alpha}: reject H₀ — "
                "evidence of non-stationarity (unit root)."
            )

        return StationarityResult(
            test_name="KPSS",
            statistic=stat,
            p_value=p_val,
            critical_values=cv,
            n_lags=n_lags,
            regression=regression,
            is_stationary=is_stat,
            alpha=alpha,
            interpretation=interp,
        )

    # ------------------------------------------------------------------
    # Phillips-Perron
    # ------------------------------------------------------------------

    def pp(
        self,
        ts: TimeSeries,
        *,
        regression: str = "c",
        alpha: float = 0.05,
    ) -> StationarityResult:
        """Phillips-Perron unit-root test.

        Like ADF but uses a non-parametric correction for serial correlation
        (no lag selection required).  Requires ``statsmodels``.

        **H₀**: The series has a unit root.
        **H₁**: The series is stationary.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        regression : str, optional
            ``"c"`` (default) or ``"ct"``.
        alpha : float, optional
            Significance level.  Default ``0.05``.

        Returns
        -------
        StationarityResult

        Raises
        ------
        ImportError
            If ``statsmodels`` is not installed.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.statistics.stationarity import StationarityTester
        >>> rng = np.random.default_rng(3)
        >>> idx = pd.date_range("2020", periods=200, freq="D")
        >>> ts  = TimeSeries(rng.standard_normal(200), index=idx)
        >>> StationarityTester().pp(ts).is_stationary
        True
        """
        ts = self._validate(ts)
        x  = ts.values[~np.isnan(ts.values)]

        try:
            from statsmodels.tsa.stattools import PhillipsPerron as sm_pp

            res    = sm_pp(x, trend=regression)
            stat   = float(res.stat)
            p_val  = float(res.pvalue)
            cv     = {k: float(v) for k, v in res.critical_values.items()}
            n_lags = None
        except ImportError as exc:
            raise ImportError(
                "Phillips-Perron test requires statsmodels. "
                "Install with: pip install statsmodels"
            ) from exc

        is_stat = p_val < alpha
        interp  = (
            f"p={p_val:.4f} {'<' if is_stat else '≥'} α={alpha}: "
            f"{'reject H₀ — evidence of stationarity.' if is_stat else 'fail to reject H₀ — possible unit root.'}"
        )

        return StationarityResult(
            test_name="PP",
            statistic=stat,
            p_value=p_val,
            critical_values=cv,
            n_lags=n_lags,
            regression=regression,
            is_stationary=is_stat,
            alpha=alpha,
            interpretation=interp,
        )

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------

    def summary(
        self,
        ts: TimeSeries,
        *,
        regression: str = "c",
        alpha: float = 0.05,
    ) -> str:
        """Run ADF + KPSS and return a human-readable combined verdict.

        The two tests have opposite nulls, so their results can be
        reconciled:

        +-------+--------+---------------------------------------------+
        | ADF   | KPSS   | Verdict                                     |
        +=======+========+=============================================+
        | stat. | stat.  | Strong evidence of stationarity             |
        +-------+--------+---------------------------------------------+
        | stat. | non-s. | Trend stationary — consider detrending      |
        +-------+--------+---------------------------------------------+
        | non-s.| stat.  | Difference stationary — try differencing    |
        +-------+--------+---------------------------------------------+
        | non-s.| non-s. | Strong evidence of non-stationarity         |
        +-------+--------+---------------------------------------------+

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        regression : str, optional
            Passed to both ADF and KPSS.  Default ``"c"``.
        alpha : float, optional
            Significance level.  Default ``0.05``.

        Returns
        -------
        str
            Multi-line plain-English summary.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.statistics.stationarity import StationarityTester
        >>> rng = np.random.default_rng(0)
        >>> idx = pd.date_range("2020", periods=200, freq="D")
        >>> ts  = TimeSeries(rng.standard_normal(200), index=idx)
        >>> print(StationarityTester().summary(ts))  # doctest: +SKIP
        """
        adf_r  = self.adf(ts,  regression=regression, alpha=alpha)
        kpss_r = self.kpss(ts, regression=regression if regression != "nc" else "c",
                            alpha=alpha)

        adf_s  = adf_r.is_stationary
        kpss_s = kpss_r.is_stationary

        if adf_s and kpss_s:
            verdict = "STATIONARY — both ADF and KPSS agree."
            action  = "No differencing or detrending required."
        elif adf_s and not kpss_s:
            verdict = "TREND STATIONARY — ADF rejects unit root, KPSS rejects level stationarity."
            action  = "Consider detrending (remove deterministic trend)."
        elif not adf_s and kpss_s:
            verdict = "DIFFERENCE STATIONARY — ADF cannot reject unit root; KPSS passes."
            action  = "Consider first-differencing (d=1)."
        else:
            verdict = "NON-STATIONARY — both ADF and KPSS indicate non-stationarity."
            action  = "Consider differencing and/or detrending."

        return (
            f"Stationarity Summary (α={alpha})\n"
            f"{'─' * 45}\n"
            f"ADF  : p={adf_r.p_value:.4f}  → "
            f"{'stationary' if adf_s else 'non-stationary'}\n"
            f"KPSS : p={kpss_r.p_value:.4f}  → "
            f"{'stationary' if kpss_s else 'non-stationary'}\n"
            f"{'─' * 45}\n"
            f"Verdict : {verdict}\n"
            f"Action  : {action}\n"
        )
