"""
Outlier detection for time series.

Four detection methods are provided, all implemented with pure numpy / scipy —
no machine-learning dependencies:

+----------+----------------------------+-------------------------------+
| Method   | Statistic                  | Best for                      |
+==========+============================+===============================+
| IQR      | Tukey fences               | Symmetric or skewed data      |
+----------+----------------------------+-------------------------------+
| Z-score  | Mean / std deviation       | Approximately normal data     |
+----------+----------------------------+-------------------------------+
| MAD      | Median absolute deviation  | Skewed data, heavy tails      |
+----------+----------------------------+-------------------------------+
| GESD     | Generalized ESD test       | Known-normal; # unknown       |
+----------+----------------------------+-------------------------------+

All detectors return an :class:`OutlierReport` and expose ``.remove()`` /
``.clip()`` helpers that return cleaned :class:`~tseda.core.TimeSeries`
objects.

Classes
-------
OutlierReport
    Immutable result dataclass.
OutlierDetector
    Stateless detector.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.quality.outliers import OutlierDetector

>>> idx = pd.date_range("2020-01-01", periods=10, freq="D")
>>> vals = np.array([1, 2, 2, 3, 100, 2, 3, 2, 1, 2], dtype=float)
>>> ts  = TimeSeries(vals, index=idx)
>>> det = OutlierDetector()
>>> r   = det.iqr(ts)
>>> r.n_outliers
1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from tseda.core.timeseries import TimeSeries
from tseda.core.validator import validate_positive_int

__all__ = ["OutlierReport", "OutlierDetector"]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutlierReport:
    """Immutable outlier detection result.

    Attributes
    ----------
    mask : numpy.ndarray
        Boolean array of shape ``(n,)``; ``True`` where an outlier was found.
    indices : numpy.ndarray
        Integer positions (0-based) of detected outliers.
    timestamps : pandas.DatetimeIndex
        Timestamps of detected outliers.
    values : numpy.ndarray
        Observed values at outlier positions.
    method : str
        Name of the detection method used.
    n_outliers : int
        Number of outliers detected.
    lower_bound : float or None
        Lower fence / threshold (when applicable).
    upper_bound : float or None
        Upper fence / threshold (when applicable).
    """

    mask: np.ndarray
    indices: np.ndarray
    timestamps: pd.DatetimeIndex
    values: np.ndarray
    method: str
    n_outliers: int
    lower_bound: Optional[float]
    upper_bound: Optional[float]

    def __repr__(self) -> str:  # pragma: no cover
        lb = f"{self.lower_bound:.4f}" if self.lower_bound is not None else "—"
        ub = f"{self.upper_bound:.4f}" if self.upper_bound is not None else "—"
        return (
            f"OutlierReport(\n"
            f"  method      : {self.method}\n"
            f"  n_outliers  : {self.n_outliers}\n"
            f"  lower_bound : {lb}\n"
            f"  upper_bound : {ub}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _build_report(
    ts: TimeSeries,
    mask: np.ndarray,
    method: str,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> OutlierReport:
    """Construct an :class:`OutlierReport` from a boolean mask."""
    idx = np.where(mask)[0]
    return OutlierReport(
        mask=mask,
        indices=idx,
        timestamps=ts.index[idx],
        values=ts.values[idx],
        method=method,
        n_outliers=int(mask.sum()),
        lower_bound=lower,
        upper_bound=upper,
    )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class OutlierDetector:
    """Detect, remove, or clip outliers in a :class:`~tseda.core.TimeSeries`.

    This class is **stateless** — create one instance and reuse across many
    series.

    Methods
    -------
    iqr(ts, k=1.5)
        Tukey IQR fence method.
    zscore(ts, threshold=3.0)
        Standard Z-score method.
    mad(ts, threshold=3.5)
        Median Absolute Deviation method.
    gesd(ts, alpha=0.05, max_outliers=10)
        Generalized Extreme Studentized Deviate test.
    remove(ts, report)
        Replace outlier values with NaN.
    clip(ts, report)
        Clip outlier values to the fence bounds.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.quality.outliers import OutlierDetector

    >>> idx = pd.date_range("2020", periods=6, freq="D")
    >>> ts  = TimeSeries([2.0, 2.1, 1.9, 2.0, 50.0, 2.0], index=idx)
    >>> det = OutlierDetector()

    IQR detection:

    >>> r = det.iqr(ts)
    >>> r.n_outliers
    1
    >>> int(r.indices[0])
    4

    Remove the outlier (replace with NaN):

    >>> cleaned = det.remove(ts, r)
    >>> cleaned.has_nan
    True
    """

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(ts: object) -> TimeSeries:
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )
        ts = ts  # type: ignore[assignment]
        assert isinstance(ts, TimeSeries)
        vals = ts.values
        finite = vals[~np.isnan(vals)]
        if finite.size < 4:
            raise ValueError(
                "Outlier detection requires at least 4 non-NaN observations."
            )
        return ts

    # ------------------------------------------------------------------
    # IQR method
    # ------------------------------------------------------------------

    def iqr(self, ts: TimeSeries, k: float = 1.5) -> OutlierReport:
        """Detect outliers using Tukey's IQR fences.

        Points below ``Q1 - k * IQR`` or above ``Q3 + k * IQR`` are
        flagged.  NaN values are excluded from the quartile computation
        but *not* flagged as outliers.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        k : float, optional
            Fence multiplier.  Common choices:

            * ``1.5`` — standard outlier fence (default).
            * ``3.0`` — extreme outlier fence.

        Returns
        -------
        OutlierReport

        Raises
        ------
        ValueError
            If *k* is not positive, or if fewer than 4 non-NaN values exist.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.quality.outliers import OutlierDetector
        >>> idx = pd.date_range("2020", periods=5, freq="D")
        >>> ts  = TimeSeries([1.0, 2.0, 2.0, 2.0, 100.0], index=idx)
        >>> OutlierDetector().iqr(ts).n_outliers
        1
        """
        if k <= 0:
            raise ValueError(f"'k' must be positive, got {k}.")
        ts = self._validate(ts)
        vals = ts.values
        finite = vals[~np.isnan(vals)]
        q1, q3 = float(np.percentile(finite, 25)), float(np.percentile(finite, 75))
        iqr_val = q3 - q1
        lower = q1 - k * iqr_val
        upper = q3 + k * iqr_val
        mask = ~np.isnan(vals) & ((vals < lower) | (vals > upper))
        return _build_report(ts, mask, f"IQR(k={k})", lower, upper)

    # ------------------------------------------------------------------
    # Z-score method
    # ------------------------------------------------------------------

    def zscore(
        self, ts: TimeSeries, threshold: float = 3.0
    ) -> OutlierReport:
        """Detect outliers using the standard Z-score.

        A value is flagged when ``|z| > threshold``, where
        ``z = (x - mean) / std``.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        threshold : float, optional
            Z-score cut-off.  Default ``3.0`` (≈ 0.3 % false-positive rate
            under normality).

        Returns
        -------
        OutlierReport

        Raises
        ------
        ValueError
            If *threshold* is not positive or if std == 0.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.quality.outliers import OutlierDetector
        >>> idx = pd.date_range("2020", periods=5, freq="D")
        >>> ts  = TimeSeries([0.0, 0.1, 0.0, -0.1, 10.0], index=idx)
        >>> OutlierDetector().zscore(ts).n_outliers
        1
        """
        if threshold <= 0:
            raise ValueError(f"'threshold' must be positive, got {threshold}.")
        ts = self._validate(ts)
        vals = ts.values
        finite = vals[~np.isnan(vals)]
        mean = float(np.mean(finite))
        std  = float(np.std(finite, ddof=1))
        if std == 0.0:
            raise ValueError(
                "Z-score requires non-constant data (std == 0)."
            )
        z = (vals - mean) / std
        mask = ~np.isnan(vals) & (np.abs(z) > threshold)
        lower = mean - threshold * std
        upper = mean + threshold * std
        return _build_report(ts, mask, f"Z-score(t={threshold})", lower, upper)

    # ------------------------------------------------------------------
    # MAD method
    # ------------------------------------------------------------------

    def mad(
        self, ts: TimeSeries, threshold: float = 3.5
    ) -> OutlierReport:
        """Detect outliers using the Median Absolute Deviation (MAD).

        A value is flagged when the modified Z-score exceeds *threshold*:

        .. math::

            M_i = \\frac{0.6745 \\,(x_i - \\tilde{x})}{\\text{MAD}}

        where :math:`\\tilde{x}` is the median and
        :math:`\\text{MAD} = \\text{median}(|x_i - \\tilde{x}|)`.

        This is more robust than the Z-score for skewed distributions or
        heavy-tailed noise.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        threshold : float, optional
            Modified Z-score cut-off.  Iglewicz & Hoaglin (1993) recommend
            ``3.5`` (default).

        Returns
        -------
        OutlierReport

        Raises
        ------
        ValueError
            If *threshold* is not positive or MAD == 0.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.quality.outliers import OutlierDetector
        >>> idx = pd.date_range("2020", periods=5, freq="D")
        >>> ts  = TimeSeries([2.0, 2.1, 1.9, 2.0, 50.0], index=idx)
        >>> OutlierDetector().mad(ts).n_outliers
        1
        """
        if threshold <= 0:
            raise ValueError(f"'threshold' must be positive, got {threshold}.")
        ts = self._validate(ts)
        vals = ts.values
        finite = vals[~np.isnan(vals)]
        med = float(np.median(finite))
        mad_val = float(np.median(np.abs(finite - med)))
        if mad_val == 0.0:
            raise ValueError(
                "MAD == 0: more than half the non-NaN values are identical. "
                "MAD-based detection cannot distinguish outliers."
            )
        modified_z = 0.6745 * (vals - med) / mad_val
        mask = ~np.isnan(vals) & (np.abs(modified_z) > threshold)
        lower = med - (threshold / 0.6745) * mad_val
        upper = med + (threshold / 0.6745) * mad_val
        return _build_report(ts, mask, f"MAD(t={threshold})", lower, upper)

    # ------------------------------------------------------------------
    # GESD method
    # ------------------------------------------------------------------

    def gesd(
        self,
        ts: TimeSeries,
        *,
        alpha: float = 0.05,
        max_outliers: int = 10,
    ) -> OutlierReport:
        """Detect outliers using the Generalized ESD (Extreme Studentized Deviate) test.

        The GESD test (Rosner, 1983) sequentially removes the most extreme
        value and tests whether it is a statistical outlier, up to
        *max_outliers* iterations.  It assumes the underlying data are
        approximately normal *after* removing the outliers.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        alpha : float, optional
            Significance level.  Default ``0.05``.
        max_outliers : int, optional
            Upper bound on the number of outliers to test for.  Default 10.
            Must be less than ``n // 2``.

        Returns
        -------
        OutlierReport
            ``lower_bound`` and ``upper_bound`` are ``None`` (GESD uses
            per-iteration critical values, not fixed fences).

        Raises
        ------
        ValueError
            If *alpha* is not in ``(0, 1)`` or *max_outliers* is invalid.
        ImportError
            If scipy is not installed (needed for the t-distribution CDF).

        Notes
        -----
        The critical value at each step *i* (1-indexed) is:

        .. math::

            \\lambda_i = \\frac{(n - i) \\, t_{p, n-i-1}}
                               {\\sqrt{(n-i-1+t_{p,n-i-1}^2)(n-i+1)}}

        where :math:`p = 1 - \\alpha / (2(n - i + 1))` and
        :math:`t_{p, \\nu}` is the :math:`p`-quantile of the
        t-distribution with :math:`\\nu` degrees of freedom.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.quality.outliers import OutlierDetector
        >>> rng = np.random.default_rng(0)
        >>> idx = pd.date_range("2020", periods=50, freq="D")
        >>> vals = rng.standard_normal(50)
        >>> vals[10] = 15.0   # plant a spike
        >>> ts  = TimeSeries(vals, index=idx)
        >>> r   = OutlierDetector().gesd(ts)
        >>> 10 in r.indices
        True
        """
        try:
            from scipy import stats as sp_stats
        except ImportError as exc:
            raise ImportError(
                "gesd() requires scipy. Install it with: pip install scipy"
            ) from exc

        if not (0 < alpha < 1):
            raise ValueError(f"'alpha' must be in (0, 1), got {alpha}.")

        ts = self._validate(ts)
        vals = ts.values.copy()
        n = ts.n

        max_outliers = validate_positive_int(max_outliers, name="max_outliers")
        if max_outliers >= n // 2:
            raise ValueError(
                f"'max_outliers' ({max_outliers}) must be less than n // 2 = {n // 2}."
            )

        # Work on the subset without NaN
        finite_mask = ~np.isnan(vals)
        finite_vals = vals[finite_mask]
        n_finite = len(finite_vals)

        # Store (test_stat, critical_val, candidate_index_in_finite) per step
        removed_positions: list[int] = []  # positions in finite_vals
        test_stats: list[float] = []
        crit_vals: list[float] = []

        working = finite_vals.copy()
        working_positions = list(range(n_finite))

        for i in range(1, max_outliers + 1):
            n_w = len(working)
            if n_w < 3:
                break
            mean_w = float(np.mean(working))
            std_w  = float(np.std(working, ddof=1))
            if std_w == 0:
                break
            abs_dev = np.abs(working - mean_w)
            local_idx = int(np.argmax(abs_dev))
            R_i = float(abs_dev[local_idx] / std_w)
            test_stats.append(R_i)

            # Critical value
            p = 1.0 - alpha / (2.0 * (n_finite - i + 1))
            t_crit = float(sp_stats.t.ppf(p, df=n_finite - i - 1))
            lam = (
                (n_finite - i) * t_crit
                / np.sqrt(
                    (n_finite - i - 1 + t_crit ** 2) * (n_finite - i + 1)
                )
            )
            crit_vals.append(lam)

            removed_positions.append(working_positions[local_idx])
            working = np.delete(working, local_idx)
            working_positions.pop(local_idx)

        # Find the largest h such that R_h > lambda_h
        n_detected = 0
        for h in range(len(test_stats) - 1, -1, -1):
            if test_stats[h] > crit_vals[h]:
                n_detected = h + 1
                break

        # Map back to original (full) positions
        finite_indices = np.where(finite_mask)[0]
        detected_finite = set(removed_positions[:n_detected])
        outlier_orig = {
            int(finite_indices[p]) for p in detected_finite
        }

        mask = np.zeros(ts.n, dtype=bool)
        for pos in outlier_orig:
            mask[pos] = True

        return _build_report(ts, mask, f"GESD(alpha={alpha})", None, None)

    # ------------------------------------------------------------------
    # Repair methods
    # ------------------------------------------------------------------

    def remove(self, ts: TimeSeries, report: OutlierReport) -> TimeSeries:
        """Replace detected outlier values with NaN.

        Parameters
        ----------
        ts : TimeSeries
            The original series.
        report : OutlierReport
            Result from one of the detection methods.

        Returns
        -------
        TimeSeries
            A new series with outliers replaced by NaN.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.quality.outliers import OutlierDetector
        >>> idx = pd.date_range("2020", periods=5, freq="D")
        >>> ts  = TimeSeries([1.0, 2.0, 100.0, 2.0, 1.0], index=idx)
        >>> det = OutlierDetector()
        >>> cleaned = det.remove(ts, det.iqr(ts))
        >>> cleaned.has_nan
        True
        """
        if not isinstance(report, OutlierReport):
            raise TypeError(
                f"'report' must be an OutlierReport, got {type(report).__name__!r}."
            )
        vals = ts.values
        vals[report.mask] = np.nan
        return TimeSeries(
            vals, index=ts.index, name=ts.name,
            freq=ts.freq, unit=ts.unit, description=ts.description,
        )

    def clip(self, ts: TimeSeries, report: OutlierReport) -> TimeSeries:
        """Clip outlier values to the fence bounds of *report*.

        Parameters
        ----------
        ts : TimeSeries
            The original series.
        report : OutlierReport
            Must have non-``None`` ``lower_bound`` and ``upper_bound``
            (i.e., from ``iqr``, ``zscore``, or ``mad``).

        Returns
        -------
        TimeSeries
            A new series with values clamped to ``[lower_bound, upper_bound]``.

        Raises
        ------
        ValueError
            If *report* has no bounds (e.g., from ``gesd``).

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.quality.outliers import OutlierDetector
        >>> idx = pd.date_range("2020", periods=5, freq="D")
        >>> ts  = TimeSeries([1.0, 2.0, 100.0, 2.0, 1.0], index=idx)
        >>> det = OutlierDetector()
        >>> r   = det.iqr(ts)
        >>> clipped = det.clip(ts, r)
        >>> float(clipped.values.max()) < 100.0
        True
        """
        if not isinstance(report, OutlierReport):
            raise TypeError(
                f"'report' must be an OutlierReport, got {type(report).__name__!r}."
            )
        if report.lower_bound is None or report.upper_bound is None:
            raise ValueError(
                "clip() requires an OutlierReport with numeric bounds. "
                f"'{report.method}' does not produce bounds. "
                "Use iqr(), zscore(), or mad() instead."
            )
        vals = np.clip(ts.values, report.lower_bound, report.upper_bound)
        return TimeSeries(
            vals, index=ts.index, name=ts.name,
            freq=ts.freq, unit=ts.unit, description=ts.description,
        )