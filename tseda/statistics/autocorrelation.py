"""
Autocorrelation analysis for time series.

Implements ACF, PACF, and the Ljung-Box portmanteau test entirely in
numpy / scipy — no statsmodels dependency.

Classes
-------
AutocorrelationResult
    Frozen dataclass containing ACF, PACF, confidence bounds, and
    Ljung-Box statistics.
AutocorrelationAnalyzer
    Stateless analyzer.

Theory
------
**ACF** at lag *k*:

.. math::

    \\hat{\\rho}(k) = \\frac{\\sum_{t=k+1}^{n}(x_t - \\bar{x})(x_{t-k} - \\bar{x})}
                           {\\sum_{t=1}^{n}(x_t - \\bar{x})^2}

**PACF** via Durbin-Levinson recursion on the ACF values:

.. math::

    \\phi_{k,k} = \\frac{\\hat{\\rho}(k) - \\sum_{j=1}^{k-1} \\phi_{k-1,j} \\hat{\\rho}(k-j)}
                        {1 - \\sum_{j=1}^{k-1} \\phi_{k-1,j} \\hat{\\rho}(j)}

**95 % confidence interval** for ACF (Bartlett's formula assuming white noise):
``±1.96 / √n``.

**Ljung-Box** test statistic:

.. math::

    Q = n(n+2) \\sum_{k=1}^{m} \\frac{\\hat{\\rho}(k)^2}{n - k}

    Q \\sim \\chi^2(m) \\text{ under H₀ (white noise)}

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.statistics.autocorrelation import AutocorrelationAnalyzer

>>> rng = np.random.default_rng(0)
>>> idx = pd.date_range("2020", periods=200, freq="D")
>>> ts  = TimeSeries(rng.standard_normal(200), index=idx)
>>> r   = AutocorrelationAnalyzer().analyze(ts, lags=20)
>>> len(r.acf)     # lag 0 … 20
21
>>> r.is_white_noise   # white noise → True
True
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from tseda.core.timeseries import TimeSeries
from tseda.core.validator import validate_lags

__all__ = ["AutocorrelationResult", "AutocorrelationAnalyzer"]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutocorrelationResult:
    """Immutable autocorrelation analysis result.

    Attributes
    ----------
    acf : numpy.ndarray
        Autocorrelation function values at lags 0, 1, …, *n_lags*.
        ``acf[0]`` is always 1.0 (lag-0 autocorrelation).
    pacf : numpy.ndarray
        Partial autocorrelation function values at lags 0, 1, …, *n_lags*.
        ``pacf[0]`` is always 1.0 by convention.
    lags : numpy.ndarray
        Integer array ``[0, 1, …, n_lags]``.
    conf_lower : numpy.ndarray
        Lower 95 % confidence bound at each lag (Bartlett's approximation).
    conf_upper : numpy.ndarray
        Upper 95 % confidence bound at each lag.
    lb_statistic : numpy.ndarray
        Ljung-Box Q-statistic at each lag from 1 to *n_lags*.
    lb_pvalue : numpy.ndarray
        P-value of the Ljung-Box test at each lag.
    n_lags : int
        Number of lags requested (excluding lag 0).
    n_obs : int
        Number of non-NaN observations used.
    is_white_noise : bool
        ``True`` when the Ljung-Box p-value at lag ``min(n_lags, 20)``
        exceeds ``alpha``.
    alpha : float
        Significance level used for ``is_white_noise`` and confidence bounds.
    """

    acf: np.ndarray
    pacf: np.ndarray
    lags: np.ndarray
    conf_lower: np.ndarray
    conf_upper: np.ndarray
    lb_statistic: np.ndarray
    lb_pvalue: np.ndarray
    n_lags: int
    n_obs: int
    is_white_noise: bool
    alpha: float

    def __repr__(self) -> str:  # pragma: no cover
        sig_acf  = int(np.sum(np.abs(self.acf[1:]) > np.abs(self.conf_upper[1:])))
        sig_pacf = int(np.sum(np.abs(self.pacf[1:]) > np.abs(self.conf_upper[1:])))
        return (
            f"AutocorrelationResult(\n"
            f"  n_obs           : {self.n_obs}\n"
            f"  n_lags          : {self.n_lags}\n"
            f"  significant ACF : {sig_acf} lag(s) outside 95% CI\n"
            f"  significant PACF: {sig_pacf} lag(s) outside 95% CI\n"
            f"  is_white_noise  : {self.is_white_noise}  (α={self.alpha})\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _acf(x: np.ndarray, n_lags: int) -> np.ndarray:
    """Compute biased sample ACF at lags 0 … n_lags."""
    n    = len(x)
    xc   = x - np.mean(x)
    denom = np.dot(xc, xc)
    if denom == 0:
        return np.zeros(n_lags + 1)
    result = np.empty(n_lags + 1)
    result[0] = 1.0
    for k in range(1, n_lags + 1):
        result[k] = float(np.dot(xc[k:], xc[:-k])) / denom
    return result


def _pacf_durbin_levinson(acf_vals: np.ndarray) -> np.ndarray:
    """Compute PACF via Durbin-Levinson recursion from ACF values.

    Parameters
    ----------
    acf_vals : array of shape (n_lags + 1,)
        ACF values including lag 0.

    Returns
    -------
    numpy.ndarray of shape (n_lags + 1,)
        PACF values (lag 0 = 1.0 by convention).
    """
    n_lags  = len(acf_vals) - 1
    pacf    = np.zeros(n_lags + 1)
    pacf[0] = 1.0

    if n_lags == 0:
        return pacf

    # phi[k, j] : phi at order k, coefficient j (1-indexed)
    phi = np.zeros((n_lags + 1, n_lags + 1))

    # Order 1
    phi[1, 1] = acf_vals[1]
    pacf[1]   = acf_vals[1]

    for k in range(2, n_lags + 1):
        # Numerator
        num = acf_vals[k] - sum(phi[k - 1, j] * acf_vals[k - j] for j in range(1, k))
        # Denominator
        den = 1.0 - sum(phi[k - 1, j] * acf_vals[j] for j in range(1, k))
        phi[k, k] = num / den if den != 0 else 0.0
        pacf[k]   = phi[k, k]
        # Update lower-order coefficients
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

    return pacf


def _ljung_box(x: np.ndarray, acf_vals: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute Ljung-Box Q-statistic for lags 1 … n_lags.

    Returns
    -------
    (lb_stat, lb_pval) each of shape (n_lags,)
    """
    n       = len(x)
    rho     = acf_vals[1:]          # shape (n_lags,)
    lb_stat = np.empty(n_lags)
    lb_pval = np.empty(n_lags)

    for m in range(1, n_lags + 1):
        q_m = float(
            n * (n + 2) * np.sum(rho[:m] ** 2 / (n - np.arange(1, m + 1)))
        )
        p_m = float(sp_stats.chi2.sf(q_m, df=m))
        lb_stat[m - 1] = q_m
        lb_pval[m - 1] = p_m

    return lb_stat, lb_pval


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class AutocorrelationAnalyzer:
    """Compute ACF, PACF, and Ljung-Box statistics for a
    :class:`~tseda.core.TimeSeries`.

    This class is **stateless**.

    Methods
    -------
    analyze(ts, lags, alpha)
        Return an :class:`AutocorrelationResult`.
    significant_lags(result)
        Return the lag numbers where ACF or PACF exceeds the CI.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.statistics.autocorrelation import AutocorrelationAnalyzer

    AR(1) process:

    >>> rng = np.random.default_rng(7)
    >>> n   = 300
    >>> idx = pd.date_range("2020", periods=n, freq="D")
    >>> eps = rng.standard_normal(n)
    >>> x   = np.zeros(n)
    >>> for i in range(1, n): x[i] = 0.7 * x[i-1] + eps[i]
    >>> ts  = TimeSeries(x, index=idx)
    >>> r   = AutocorrelationAnalyzer().analyze(ts, lags=10)
    >>> r.acf[1] > 0.5          # strong lag-1 autocorrelation
    True
    >>> r.is_white_noise         # definitely not white noise
    False
    """

    def analyze(
        self,
        ts: TimeSeries,
        lags: int = 40,
        *,
        alpha: float = 0.05,
    ) -> AutocorrelationResult:
        """Compute ACF, PACF, and Ljung-Box statistics.

        Parameters
        ----------
        ts : TimeSeries
            Input series.  NaN values are dropped before analysis.
        lags : int, optional
            Number of lags to compute (lag 0 is always included).
            Capped at ``n // 2``.  Default 40.
        alpha : float, optional
            Significance level for confidence bounds and
            :attr:`~AutocorrelationResult.is_white_noise`.  Default 0.05.

        Returns
        -------
        AutocorrelationResult

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *lags* is out of range or fewer than 4 non-NaN observations.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.statistics.autocorrelation import AutocorrelationAnalyzer
        >>> idx = pd.date_range("2020", periods=50, freq="D")
        >>> ts  = TimeSeries(np.ones(50), index=idx)
        >>> r   = AutocorrelationAnalyzer().analyze(ts, lags=5)
        >>> r.acf[0]
        1.0
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )
        x = ts.values[~np.isnan(ts.values)]
        if len(x) < 4:
            raise ValueError(
                "AutocorrelationAnalyzer requires at least 4 non-NaN observations."
            )

        n_lags = validate_lags(lags, len(x), name="lags")
        n      = len(x)

        acf_vals  = _acf(x, n_lags)
        pacf_vals = _pacf_durbin_levinson(acf_vals)
        lb_stat, lb_pval = _ljung_box(x, acf_vals, n_lags)

        # Bartlett 95% CI  ±z * se, se ≈ 1/√n for white noise
        z_crit     = float(sp_stats.norm.ppf(1 - alpha / 2))
        ci_half    = z_crit / np.sqrt(n)
        conf_lower = np.full(n_lags + 1, -ci_half)
        conf_upper = np.full(n_lags + 1,  ci_half)
        conf_lower[0] = conf_upper[0] = 1.0   # lag 0 is exactly 1

        # White noise check at lag min(n_lags, 20)
        wn_lag     = min(n_lags, 20) - 1      # 0-indexed
        is_wn      = bool(lb_pval[wn_lag] > alpha)

        return AutocorrelationResult(
            acf=acf_vals,
            pacf=pacf_vals,
            lags=np.arange(n_lags + 1),
            conf_lower=conf_lower,
            conf_upper=conf_upper,
            lb_statistic=lb_stat,
            lb_pvalue=lb_pval,
            n_lags=n_lags,
            n_obs=n,
            is_white_noise=is_wn,
            alpha=alpha,
        )

    def significant_lags(
        self,
        result: AutocorrelationResult,
        *,
        which: str = "acf",
    ) -> np.ndarray:
        """Return lag numbers (> 0) where the function exceeds the CI.

        Parameters
        ----------
        result : AutocorrelationResult
            Output of :meth:`analyze`.
        which : str, optional
            ``"acf"`` (default) or ``"pacf"``.

        Returns
        -------
        numpy.ndarray
            Integer array of significant lag numbers.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.statistics.autocorrelation import AutocorrelationAnalyzer

        >>> rng = np.random.default_rng(7)
        >>> n   = 300
        >>> idx = pd.date_range("2020", periods=n, freq="D")
        >>> eps = rng.standard_normal(n)
        >>> x   = np.zeros(n)
        >>> for i in range(1, n): x[i] = 0.7 * x[i-1] + eps[i]
        >>> ts  = TimeSeries(x, index=idx)
        >>> r   = AutocorrelationAnalyzer().analyze(ts, lags=10)
        >>> len(AutocorrelationAnalyzer().significant_lags(r)) > 0
        True
        """
        if not isinstance(result, AutocorrelationResult):
            raise TypeError(
                f"'result' must be an AutocorrelationResult, "
                f"got {type(result).__name__!r}."
            )
        if which not in ("acf", "pacf"):
            raise ValueError(f"'which' must be 'acf' or 'pacf'; got {which!r}.")

        vals = result.acf if which == "acf" else result.pacf
        ub   = result.conf_upper

        # Exclude lag 0 (always 1.0)
        mask = np.abs(vals[1:]) > np.abs(ub[1:])
        return result.lags[1:][mask]