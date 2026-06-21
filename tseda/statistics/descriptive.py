"""
Descriptive statistics for time series.

Provides a single :class:`DescriptiveStats` result object and a stateless
:class:`DescriptiveAnalyzer` that computes it.  All arithmetic uses numpy
so there are no extra dependencies beyond the core stack.

The statistics reported go beyond what :func:`pandas.Series.describe` offers:

* Robust location / spread (median, MAD, trimmed mean).
* Shape (skewness, excess kurtosis).
* Quantiles at multiple probability levels.
* First/last value, range, coefficient of variation.
* Count of zeros and near-zero values.

Classes
-------
DescriptiveStats
    Frozen dataclass containing every computed statistic.
DescriptiveAnalyzer
    Stateless analyzer that produces :class:`DescriptiveStats`.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.statistics.descriptive import DescriptiveAnalyzer

>>> rng = np.random.default_rng(0)
>>> idx = pd.date_range("2020-01-01", periods=200, freq="D")
>>> ts  = TimeSeries(rng.standard_normal(200), index=idx, name="returns")
>>> r   = DescriptiveAnalyzer().analyze(ts)
>>> round(r.mean, 3)
0.024
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from tseda.core.timeseries import TimeSeries

__all__ = ["DescriptiveStats", "DescriptiveAnalyzer"]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DescriptiveStats:
    """Comprehensive descriptive statistics for a :class:`~tseda.core.TimeSeries`.

    All statistics are computed on the **non-NaN** subset unless otherwise
    noted.

    Attributes
    ----------
    n_total : int
        Total number of observations (including NaN).
    n_valid : int
        Number of non-NaN observations.
    n_nan : int
        Number of NaN observations.
    pct_nan : float
        Percentage of NaN observations (0–100).
    mean : float
        Arithmetic mean.
    median : float
        50th percentile.
    std : float
        Sample standard deviation (ddof=1).
    var : float
        Sample variance (ddof=1).
    mad : float
        Median absolute deviation: ``median(|x - median(x)|)``.
    trimmed_mean : float
        Mean with the top and bottom 5 % of values removed.
    min : float
        Minimum value.
    max : float
        Maximum value.
    range : float
        ``max - min``.
    first : float
        First (earliest) non-NaN value.
    last : float
        Last (most recent) non-NaN value.
    cv : float
        Coefficient of variation: ``std / |mean|``.  ``nan`` when mean == 0.
    skewness : float
        Fisher's moment coefficient of skewness (bias-corrected).
    kurtosis : float
        Excess kurtosis (Fisher definition, bias-corrected).  0 for a
        normal distribution.
    quantiles : dict of float → float
        Mapping from probability level to quantile value.
        Keys: ``[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]``.
    n_zeros : int
        Number of exact zeros.
    n_positive : int
        Number of strictly positive values.
    n_negative : int
        Number of strictly negative values.
    """

    # Sample size
    n_total: int
    n_valid: int
    n_nan: int
    pct_nan: float

    # Central tendency
    mean: float
    median: float
    trimmed_mean: float

    # Spread
    std: float
    var: float
    mad: float
    cv: float

    # Range
    min: float
    max: float
    range: float
    first: float
    last: float

    # Shape
    skewness: float
    kurtosis: float

    # Quantiles
    quantiles: Dict[float, float]

    # Value-type counts
    n_zeros: int
    n_positive: int
    n_negative: int

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DescriptiveStats(\n"
            f"  n_valid    : {self.n_valid:,} / {self.n_total:,}  "
            f"({self.pct_nan:.1f}% NaN)\n"
            f"  mean       : {self.mean:.6g}\n"
            f"  median     : {self.median:.6g}\n"
            f"  std        : {self.std:.6g}\n"
            f"  [min, max] : [{self.min:.6g}, {self.max:.6g}]\n"
            f"  skewness   : {self.skewness:.4f}\n"
            f"  kurtosis   : {self.kurtosis:.4f}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_QUANTILE_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _trimmed_mean(x: np.ndarray, trim: float = 0.05) -> float:
    """Return mean of *x* after removing the *trim* fraction from each tail."""
    n = len(x)
    k = int(np.floor(n * trim))
    if k == 0:
        return float(np.mean(x))
    xs = np.sort(x)
    return float(np.mean(xs[k : n - k]))


def _skewness(x: np.ndarray) -> float:
    """Bias-corrected sample skewness (Fisher's g1)."""
    n = len(x)
    if n < 3:
        return float("nan")
    m = x - x.mean()
    m2 = float(np.mean(m ** 2))
    m3 = float(np.mean(m ** 3))
    if m2 == 0:
        return float("nan")
    g1 = m3 / m2 ** 1.5
    # bias correction
    return float(g1 * np.sqrt(n * (n - 1)) / (n - 2))


def _kurtosis(x: np.ndarray) -> float:
    """Bias-corrected excess kurtosis (Fisher's g2)."""
    n = len(x)
    if n < 4:
        return float("nan")
    m = x - x.mean()
    m2 = float(np.mean(m ** 2))
    m4 = float(np.mean(m ** 4))
    if m2 == 0:
        return float("nan")
    # Fisher's excess kurtosis (normal = 0)
    g2 = m4 / m2 ** 2 - 3.0
    # Bias correction (excess kurtosis)
    correction = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)
    return float(correction)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class DescriptiveAnalyzer:
    """Compute comprehensive descriptive statistics for a
    :class:`~tseda.core.TimeSeries`.

    This class is **stateless** — one instance, many series.

    Methods
    -------
    analyze(ts)
        Return a :class:`DescriptiveStats` for *ts*.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.statistics.descriptive import DescriptiveAnalyzer

    >>> idx = pd.date_range("2020", periods=5, freq="D")
    >>> ts  = TimeSeries([2.0, 4.0, 4.0, 4.0, 5.0], index=idx)
    >>> r   = DescriptiveAnalyzer().analyze(ts)
    >>> r.mean
    3.8
    >>> r.std  # doctest: +ELLIPSIS
    1.09...
    """

    def analyze(self, ts: TimeSeries) -> DescriptiveStats:
        """Compute descriptive statistics for *ts*.

        Parameters
        ----------
        ts : TimeSeries
            Input series.

        Returns
        -------
        DescriptiveStats

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *ts* has no non-NaN values.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.statistics.descriptive import DescriptiveAnalyzer

        >>> idx = pd.date_range("2020", periods=4, freq="D")
        >>> ts  = TimeSeries([1.0, 2.0, 3.0, 4.0], index=idx)
        >>> r   = DescriptiveAnalyzer().analyze(ts)
        >>> r.median
        2.5
        >>> r.n_positive
        4
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )

        vals     = ts.values
        not_nan  = ~np.isnan(vals)
        x        = vals[not_nan]
        n_valid  = int(x.size)

        if n_valid == 0:
            raise ValueError("'ts' has no non-NaN values; cannot compute statistics.")

        n_total = ts.n
        n_nan   = n_total - n_valid
        pct_nan = 100.0 * n_nan / max(n_total, 1)

        # Central tendency
        mean     = float(np.mean(x))
        median   = float(np.median(x))
        tr_mean  = _trimmed_mean(x, trim=0.05)

        # Spread
        std = float(np.std(x, ddof=1)) if n_valid > 1 else float("nan")
        var = float(np.var(x, ddof=1)) if n_valid > 1 else float("nan")
        mad = float(np.median(np.abs(x - median)))
        cv  = (std / abs(mean)) if (mean != 0 and not np.isnan(std)) else float("nan")

        # Range
        mn    = float(np.min(x))
        mx    = float(np.max(x))
        rng   = mx - mn

        # First / last non-NaN values (positional)
        not_nan_idx = np.where(not_nan)[0]
        first = float(vals[not_nan_idx[0]])
        last  = float(vals[not_nan_idx[-1]])

        # Shape
        skew = _skewness(x)
        kurt = _kurtosis(x)

        # Quantiles
        quantiles = {
            q: float(np.quantile(x, q)) for q in _QUANTILE_LEVELS
        }

        # Value-type counts
        n_zeros    = int(np.sum(x == 0.0))
        n_positive = int(np.sum(x > 0.0))
        n_negative = int(np.sum(x < 0.0))

        return DescriptiveStats(
            n_total=n_total,
            n_valid=n_valid,
            n_nan=n_nan,
            pct_nan=round(pct_nan, 4),
            mean=mean,
            median=median,
            trimmed_mean=tr_mean,
            std=std,
            var=var,
            mad=mad,
            cv=cv,
            min=mn,
            max=mx,
            range=rng,
            first=first,
            last=last,
            skewness=skew,
            kurtosis=kurt,
            quantiles=quantiles,
            n_zeros=n_zeros,
            n_positive=n_positive,
            n_negative=n_negative,
        )