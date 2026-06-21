"""
Statistical feature extraction for time series.

Extracts a rich set of statistical descriptors that characterise the
distribution, complexity, and structure of a :class:`~tseda.core.TimeSeries`.
All features are computed in pure numpy.

Feature groups
--------------
* **Distribution** — mean, std, skewness, kurtosis, quantiles, range, CV.
* **Spread / Robust** — MAD, trimmed mean, IQR.
* **Complexity** — approximate entropy, sample entropy, turning points ratio,
  mean-crossing rate.
* **Linear structure** — lag-1 autocorrelation, linear-trend slope and R².
* **Nonlinearity** — number of peaks and troughs, flatness ratio.

Classes
-------
StatisticalFeatureExtractor
    Stateless extractor.

Examples
--------
>>> import pandas as pd, numpy as np
>>> from tseda import TimeSeries
>>> from tseda.features.statistical import StatisticalFeatureExtractor

>>> rng = np.random.default_rng(0)
>>> idx = pd.date_range("2020", periods=200, freq="D")
>>> ts  = TimeSeries(rng.standard_normal(200), index=idx)
>>> df  = StatisticalFeatureExtractor().extract(ts)
>>> "mean" in df.columns and "std" in df.columns
True
>>> df.shape[0]
1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tseda.core.timeseries import TimeSeries

__all__ = ["StatisticalFeatureExtractor"]


# ---------------------------------------------------------------------------
# Private feature computers
# ---------------------------------------------------------------------------


def _approx_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Approximate entropy (ApEn) — measures regularity / predictability.

    Parameters
    ----------
    x : numpy.ndarray  (1-D, no NaN)
    m : int            Template length (default 2).
    r : float          Tolerance as fraction of std (default 0.2).

    Returns
    -------
    float
        ApEn value.  Lower = more regular.
    """
    n   = len(x)
    tol = r * float(np.std(x, ddof=1))
    if tol == 0:
        return 0.0

    def _phi(m_: int) -> float:
        templates = np.array([x[i : i + m_] for i in range(n - m_ + 1)])
        count = np.array([
            np.sum(np.max(np.abs(templates - templates[i]), axis=1) <= tol)
            for i in range(len(templates))
        ])
        return float(np.mean(np.log(count / (n - m_ + 1))))

    return abs(_phi(m) - _phi(m + 1))


def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Sample entropy (SampEn) — more robust ApEn variant.

    Lower = more regular; ``0`` for constant series.
    """
    n   = len(x)
    tol = r * float(np.std(x, ddof=1))
    if tol == 0:
        return 0.0

    def _count(m_: int) -> int:
        total = 0
        for i in range(n - m_):
            template = x[i : i + m_]
            for j in range(i + 1, n - m_):
                if np.max(np.abs(x[j : j + m_] - template)) < tol:
                    total += 1
        return total

    A = _count(m + 1)
    B = _count(m)
    if B == 0:
        return 0.0
    return float(-np.log(A / B)) if A > 0 else float("nan")


def _linear_trend(x: np.ndarray) -> tuple[float, float]:
    """Return (slope, r_squared) of a least-squares linear fit."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    t    = np.arange(n, dtype=float)
    t_c  = t - t.mean()
    x_c  = x - x.mean()
    denom = float(np.dot(t_c, t_c))
    if denom == 0:
        return 0.0, 0.0
    slope  = float(np.dot(t_c, x_c) / denom)
    y_hat  = slope * t_c + x.mean()
    ss_res = float(np.sum((x - y_hat) ** 2))
    ss_tot = float(np.sum(x_c ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, max(0.0, r2)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class StatisticalFeatureExtractor:
    """Extract statistical features from a :class:`~tseda.core.TimeSeries`.

    The extractor is **stateless**.  It operates on the non-NaN values of
    the series.

    Methods
    -------
    extract(ts, entropy)
        Return a single-row :class:`pandas.DataFrame` of features.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from tseda import TimeSeries
    >>> from tseda.features.statistical import StatisticalFeatureExtractor

    >>> idx = pd.date_range("2020", periods=100, freq="D")
    >>> ts  = TimeSeries(np.arange(100.0), index=idx)
    >>> df  = StatisticalFeatureExtractor().extract(ts)
    >>> round(float(df["linear_slope"].iloc[0]), 1)
    1.0
    >>> round(float(df["linear_r2"].iloc[0]), 2)
    1.0
    """

    def extract(
        self,
        ts: TimeSeries,
        *,
        entropy: bool = True,
    ) -> pd.DataFrame:
        """Compute statistical features for *ts*.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        entropy : bool, optional
            When ``True`` (default), compute approximate entropy and sample
            entropy.  These are O(n²) — set ``False`` for large series
            (n > 2000) to save time.

        Returns
        -------
        pandas.DataFrame
            One row, columns:

            Distribution:
              ``mean``, ``std``, ``var``, ``skewness``, ``kurtosis``,
              ``min``, ``max``, ``range``, ``median``, ``iqr``,
              ``mad``, ``cv``, ``trimmed_mean``,
              ``q25``, ``q75``, ``q05``, ``q95``.

            Complexity:
              ``turning_points_ratio``, ``mean_crossing_rate``,
              ``flatness_ratio``.
              If ``entropy=True``:  ``approx_entropy``, ``sample_entropy``.

            Linear structure:
              ``lag1_acf``, ``linear_slope``, ``linear_r2``.

            Nonlinearity:
              ``n_peaks``, ``n_troughs``.

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *ts* has fewer than 4 non-NaN observations.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> from tseda import TimeSeries
        >>> from tseda.features.statistical import StatisticalFeatureExtractor
        >>> idx = pd.date_range("2020", periods=5, freq="D")
        >>> ts  = TimeSeries([1.0, 2.0, 1.5, 2.5, 2.0], index=idx)
        >>> df  = StatisticalFeatureExtractor().extract(ts, entropy=False)
        >>> float(df["mean"].iloc[0])
        1.8
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )

        vals   = ts.values
        x      = vals[~np.isnan(vals)]
        n      = len(x)

        if n < 4:
            raise ValueError(
                "StatisticalFeatureExtractor requires at least 4 non-NaN "
                f"observations; got {n}."
            )

        # ── Distribution ──────────────────────────────────────────────
        mean_v  = float(np.mean(x))
        std_v   = float(np.std(x, ddof=1))
        var_v   = std_v ** 2
        med_v   = float(np.median(x))
        mn, mx  = float(np.min(x)), float(np.max(x))
        rng_v   = mx - mn
        q25, q75 = float(np.percentile(x, 25)), float(np.percentile(x, 75))
        q05, q95 = float(np.percentile(x, 5)),  float(np.percentile(x, 95))
        iqr_v   = q75 - q25
        mad_v   = float(np.median(np.abs(x - med_v)))
        cv_v    = std_v / abs(mean_v) if mean_v != 0 else float("nan")

        # Trimmed mean (5 % each tail)
        k       = max(1, int(np.floor(n * 0.05)))
        xs      = np.sort(x)
        tr_mean = float(np.mean(xs[k : n - k]))

        # Skewness and excess kurtosis (bias-corrected)
        m2 = float(np.mean((x - mean_v) ** 2))
        m3 = float(np.mean((x - mean_v) ** 3))
        m4 = float(np.mean((x - mean_v) ** 4))

        if m2 > 0:
            g1 = m3 / m2 ** 1.5
            skew = float(g1 * np.sqrt(n * (n - 1)) / (n - 2)) if n >= 3 else 0.0
            g2   = m4 / m2 ** 2 - 3.0
            kurt = float((n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)) if n >= 4 else 0.0
        else:
            skew = 0.0
            kurt = 0.0

        # ── Complexity ────────────────────────────────────────────────
        # Turning points (local max or min)
        diff1 = np.sign(np.diff(x))
        sign_changes = np.where(diff1[:-1] != diff1[1:])[0]
        tp_ratio = float(len(sign_changes)) / max(n - 2, 1)

        # Mean-crossing rate
        xc = x - mean_v
        crossings = np.sum((xc[:-1] * xc[1:]) < 0)
        mc_rate = float(crossings) / max(n - 1, 1)

        # Flatness: fraction of consecutive pairs with zero difference
        flat = np.sum(np.diff(x) == 0)
        flat_ratio = float(flat) / max(n - 1, 1)

        # Entropy (optional, O(n²))
        ap_ent = _approx_entropy(x) if entropy and n <= 2000 else float("nan")
        sa_ent = _sample_entropy(x)  if entropy and n <= 500  else float("nan")

        # ── Linear structure ──────────────────────────────────────────
        lag1_acf = 0.0
        if n >= 2:
            xc_arr = x - mean_v
            denom  = float(np.dot(xc_arr, xc_arr))
            if denom > 0:
                lag1_acf = float(np.dot(xc_arr[1:], xc_arr[:-1]) / denom)

        slope, r2 = _linear_trend(x)

        # ── Nonlinearity ──────────────────────────────────────────────
        d       = np.diff(x)
        n_peaks  = int(np.sum((d[:-1] > 0) & (d[1:] < 0)))
        n_trough = int(np.sum((d[:-1] < 0) & (d[1:] > 0)))

        row = {
            # Distribution
            "mean":            mean_v,
            "std":             std_v,
            "var":             var_v,
            "skewness":        skew,
            "kurtosis":        kurt,
            "min":             mn,
            "max":             mx,
            "range":           rng_v,
            "median":          med_v,
            "iqr":             iqr_v,
            "mad":             mad_v,
            "cv":              cv_v,
            "trimmed_mean":    tr_mean,
            "q05":             q05,
            "q25":             q25,
            "q75":             q75,
            "q95":             q95,
            # Complexity
            "turning_points_ratio": tp_ratio,
            "mean_crossing_rate":   mc_rate,
            "flatness_ratio":       flat_ratio,
            "approx_entropy":       ap_ent,
            "sample_entropy":       sa_ent,
            # Linear structure
            "lag1_acf":        lag1_acf,
            "linear_slope":    slope,
            "linear_r2":       r2,
            # Nonlinearity
            "n_peaks":         float(n_peaks),
            "n_troughs":       float(n_trough),
        }

        return pd.DataFrame([row], index=[ts.index[0]])
