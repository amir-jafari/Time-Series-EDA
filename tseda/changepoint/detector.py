"""
Changepoint (structural break) detection for time series.

A changepoint is a position in the series where the underlying data-generating
process shifts — typically a sudden change in mean level, variance, or trend
slope.

Three complementary methods are provided, all in pure numpy / scipy:

+-----------------------+----------------------------+-----------------------------+
| Method                | Detects                    | Best for                    |
+=======================+============================+=============================+
| ``cusum``             | Mean shift                 | Online-style detection;     |
|                       |                            | sensitive to gradual drift  |
+-----------------------+----------------------------+-----------------------------+
| ``binary_segmentation``| Mean shift (multiple)     | Batch; finds exact number  |
|                       |                            | of breaks automatically     |
+-----------------------+----------------------------+-----------------------------+
| ``variance_ratio``    | Variance shift             | Detecting regime changes in |
|                       |                            | volatility                  |
+-----------------------+----------------------------+-----------------------------+

All methods return a :class:`ChangepointReport` and share a repair helper
:meth:`ChangepointDetector.segment` that returns labelled segment indices.

Classes
-------
ChangepointReport
    Frozen dataclass returned by every detection method.
ChangepointDetector
    Stateless detector.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.changepoint.detector import ChangepointDetector

Series with a single level shift at position 200:

>>> rng   = np.random.default_rng(0)
>>> n     = 400
>>> idx   = pd.date_range("2020-01-01", periods=n, freq="D")
>>> vals  = np.concatenate([rng.standard_normal(200),
...                         rng.standard_normal(200) + 5.0])
>>> ts    = TimeSeries(vals, index=idx)
>>> det   = ChangepointDetector()
>>> r     = det.binary_segmentation(ts)
>>> len(r.changepoints) >= 1
True
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from tseda.core.timeseries import TimeSeries
from tseda.core.validator import validate_positive_int

__all__ = ["ChangepointReport", "ChangepointDetector"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangepointReport:
    """Immutable changepoint detection result.

    Attributes
    ----------
    changepoints : list of int
        0-based integer positions of detected changepoints, sorted ascending.
        A changepoint at position ``k`` means the break occurs *between*
        observations ``k-1`` and ``k``.
    timestamps : pandas.DatetimeIndex
        Timestamps corresponding to each changepoint position.
    n_changepoints : int
        Number of detected changepoints.
    scores : numpy.ndarray
        Continuous changepoint score in [0, 1] for each observation.
        Higher values indicate stronger evidence of a structural break at
        or near that position.
    method : str
        Name of the detection method.
    """

    changepoints: List[int]
    timestamps: pd.DatetimeIndex
    n_changepoints: int
    scores: np.ndarray
    method: str

    def segment_labels(self, n: int) -> np.ndarray:
        """Return a 0-indexed integer segment label for each of *n* observations.

        Segment 0 spans ``[0, changepoints[0])``, segment 1 spans
        ``[changepoints[0], changepoints[1])``, and so on.

        Parameters
        ----------
        n : int
            Total number of observations.

        Returns
        -------
        numpy.ndarray of int
            Shape ``(n,)``.

        Examples
        --------
        >>> from tseda.changepoint.detector import ChangepointReport
        >>> import numpy as np, pandas as pd
        >>> r = ChangepointReport(
        ...     changepoints=[3, 7],
        ...     timestamps=pd.DatetimeIndex([]),
        ...     n_changepoints=2,
        ...     scores=np.zeros(10),
        ...     method="test",
        ... )
        >>> r.segment_labels(10).tolist()
        [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        """
        labels = np.zeros(n, dtype=int)
        for seg_idx, cp in enumerate(sorted(self.changepoints), start=1):
            labels[cp:] = seg_idx
        return labels

    def __repr__(self) -> str:  # pragma: no cover
        ts_str = (
            [str(t.date()) for t in self.timestamps[:5]]
            if len(self.timestamps) > 0 else []
        )
        return (
            f"ChangepointReport(\n"
            f"  method          : {self.method}\n"
            f"  n_changepoints  : {self.n_changepoints}\n"
            f"  changepoints    : {self.changepoints[:5]}{'...' if self.n_changepoints > 5 else ''}\n"
            f"  timestamps      : {ts_str}{'...' if self.n_changepoints > 5 else ''}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_report(
    ts: TimeSeries,
    changepoints: List[int],
    scores: np.ndarray,
    method: str,
) -> ChangepointReport:
    cps = sorted(changepoints)
    idx = ts.index[cps] if cps else pd.DatetimeIndex([])
    return ChangepointReport(
        changepoints=cps,
        timestamps=idx,
        n_changepoints=len(cps),
        scores=scores,
        method=method,
    )


def _clean_array(ts: TimeSeries) -> np.ndarray:
    """Return non-NaN values as a numpy array (linear interp for NaN)."""
    vals = ts.values.astype(float)
    nan_mask = np.isnan(vals)
    if nan_mask.any():
        idx_arr = np.arange(len(vals))
        vals[nan_mask] = np.interp(
            idx_arr[nan_mask], idx_arr[~nan_mask], vals[~nan_mask]
        )
    return vals


def _cusum_arrays(
    x: np.ndarray, target: float, sigma: float, k: float, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute two-sided CUSUM arrays S+ and S−."""
    n = len(x)
    S_pos = np.zeros(n)
    S_neg = np.zeros(n)
    for i in range(1, n):
        S_pos[i] = max(0.0, S_pos[i - 1] + (x[i] - target) - k * sigma)
        S_neg[i] = max(0.0, S_neg[i - 1] - (x[i] - target) - k * sigma)
    return S_pos, S_neg


def _binary_seg_recursive(
    x: np.ndarray, offset: int, min_size: int, penalty: float
) -> List[int]:
    """Recursive binary segmentation — returns changepoints as global indices."""
    n = len(x)
    if n < 2 * min_size:
        return []

    # Sum of squared errors for the full segment
    mu_all = np.nanmean(x)
    sse_all = float(np.nansum((x - mu_all) ** 2))

    best_gain  = -np.inf
    best_split = -1

    for t in range(min_size, n - min_size + 1):
        left, right = x[:t], x[t:]
        sse_l = float(np.nansum((left  - np.nanmean(left))  ** 2))
        sse_r = float(np.nansum((right - np.nanmean(right)) ** 2))
        gain  = sse_all - sse_l - sse_r
        if gain > best_gain:
            best_gain  = gain
            best_split = t

    if best_gain <= penalty or best_split < 0:
        return []

    # Found a split — recurse on both halves
    cps  = [offset + best_split]
    cps += _binary_seg_recursive(x[:best_split], offset, min_size, penalty)
    cps += _binary_seg_recursive(x[best_split:], offset + best_split, min_size, penalty)
    return sorted(cps)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class ChangepointDetector:
    """Detect structural breaks in a :class:`~tseda.core.TimeSeries`.

    This class is **stateless** — one instance, many series.

    Methods
    -------
    cusum(ts, threshold, drift, target)
        Two-sided CUSUM control chart for mean shift.
    binary_segmentation(ts, min_size, penalty)
        Recursive mean-shift changepoint detection.
    variance_ratio(ts, window, alpha)
        Sliding F-test for variance shifts.
    segment(ts, report)
        Return segment labels and per-segment statistics.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.changepoint.detector import ChangepointDetector

    Single level shift:

    >>> rng  = np.random.default_rng(0)
    >>> idx  = pd.date_range("2020", periods=200, freq="D")
    >>> vals = np.concatenate([rng.standard_normal(100),
    ...                        rng.standard_normal(100) + 4.0])
    >>> ts   = TimeSeries(vals, index=idx)
    >>> det  = ChangepointDetector()
    >>> r    = det.binary_segmentation(ts)
    >>> abs(r.changepoints[0] - 100) <= 5   # within 5 obs of true break
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
                "Changepoint detection requires at least 10 non-NaN observations."
            )
        return ts

    # ------------------------------------------------------------------
    # CUSUM
    # ------------------------------------------------------------------

    def cusum(
        self,
        ts: TimeSeries,
        *,
        threshold: float = 5.0,
        drift: float = 0.5,
        target: Optional[float] = None,
    ) -> ChangepointReport:
        """Two-sided CUSUM (Cumulative Sum) control chart for mean shift.

        CUSUM accumulates deviations from a *target* mean.  When the
        cumulative sum exceeds a *threshold* (expressed in units of σ),
        a changepoint is signalled.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        threshold : float, optional
            Decision interval in multiples of σ (default ``5.0``).
            Higher values = less sensitive / fewer false alarms.
        drift : float, optional
            Allowance parameter *k* (default ``0.5``).  Typically set to
            half the magnitude of the smallest shift to detect, in units
            of σ.
        target : float, optional
            Reference (in-control) mean.  Defaults to the series mean.

        Returns
        -------
        ChangepointReport

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If fewer than 10 non-NaN observations or *threshold* / *drift* ≤ 0.

        Notes
        -----
        The CUSUM chart for detecting upward shifts:

        .. math::

            S_t^+ = \\max\\bigl(0,\\; S_{t-1}^+ + (x_t - \\mu_0) - k\\sigma\\bigr)

        A changepoint is signalled when :math:`S_t^+ > h\\sigma`
        (or similarly for :math:`S_t^-`).  After each signal the
        accumulator is reset to zero.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.changepoint.detector import ChangepointDetector
        >>> rng  = np.random.default_rng(1)
        >>> idx  = pd.date_range("2020", periods=300, freq="D")
        >>> vals = np.concatenate([rng.standard_normal(150),
        ...                        rng.standard_normal(150) + 3.0])
        >>> ts   = TimeSeries(vals, index=idx)
        >>> r    = ChangepointDetector().cusum(ts, threshold=5.0, drift=0.5)
        >>> r.n_changepoints >= 1
        True
        """
        if threshold <= 0:
            raise ValueError(f"'threshold' must be positive, got {threshold}.")
        if drift <= 0:
            raise ValueError(f"'drift' must be positive, got {drift}.")

        ts = self._validate(ts)
        x  = _clean_array(ts)
        n  = len(x)

        # Estimate in-control mean and sigma
        mu     = float(target) if target is not None else float(np.mean(x))
        diffs  = np.diff(x)
        sigma  = float(np.std(diffs, ddof=1) / np.sqrt(2)) if len(diffs) > 1 else 1.0
        if sigma == 0:
            sigma = 1.0

        h = threshold * sigma
        k = drift    # allowance (no sigma scaling — drift is already a fraction)

        S_pos, S_neg = _cusum_arrays(x, mu, sigma, k, h)

        # Detect signals and reset
        changepoints: List[int] = []
        S_p, S_n = S_pos.copy(), S_neg.copy()
        i = 1
        while i < n:
            if S_p[i] > h or S_n[i] > h:
                changepoints.append(i)
                # Reset from next observation
                if i + 1 < n:
                    S_p[i + 1] = 0.0
                    S_n[i + 1] = 0.0
                    S_p, S_n = _cusum_arrays(x[i + 1:], mu, sigma, k, h)
                    S_p = np.concatenate([np.zeros(i + 1), S_p])
                    S_n = np.concatenate([np.zeros(i + 1), S_n])
            i += 1

        # Scores: normalised max(S+, S−)
        max_s  = max(float(S_pos.max()), float(S_neg.max()), 1e-10)
        scores = np.maximum(S_pos, S_neg) / max_s

        return _build_report(ts, changepoints, scores,
                             f"cusum(t={threshold},d={drift})")

    # ------------------------------------------------------------------
    # Binary segmentation
    # ------------------------------------------------------------------

    def binary_segmentation(
        self,
        ts: TimeSeries,
        *,
        min_size: int = 10,
        penalty: Optional[float] = None,
    ) -> ChangepointReport:
        """Recursive binary segmentation for mean-shift changepoints.

        Iteratively finds the position that maximises the reduction in
        within-segment sum-of-squares error.  A split is accepted when the
        gain exceeds *penalty*; recursion continues on each sub-segment.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        min_size : int, optional
            Minimum number of observations per segment (default ``10``).
            Prevents detecting breaks on very small sub-sequences.
        penalty : float, optional
            Minimum SSE gain required to accept a split.  Defaults to
            ``n × σ²`` where σ is estimated from first-differences.  A
            higher penalty → fewer changepoints.

        Returns
        -------
        ChangepointReport

        Notes
        -----
        The algorithm has O(n²) time complexity per level of recursion.
        For very long series (n > 5000) consider using a larger *min_size*
        or restricting the search.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.changepoint.detector import ChangepointDetector
        >>> rng  = np.random.default_rng(0)
        >>> idx  = pd.date_range("2020", periods=300, freq="D")
        >>> vals = np.concatenate([rng.standard_normal(100),
        ...                        rng.standard_normal(100) + 5.0,
        ...                        rng.standard_normal(100)])
        >>> ts   = TimeSeries(vals, index=idx)
        >>> r    = ChangepointDetector().binary_segmentation(ts)
        >>> r.n_changepoints
        2
        """
        ts = self._validate(ts)
        min_size = validate_positive_int(min_size, name="min_size")
        x  = _clean_array(ts)
        n  = len(x)

        # Default penalty: σ² × n (BIC-like)
        if penalty is None:
            diffs  = np.diff(x)
            sigma2 = float(np.var(diffs, ddof=1) / 2) if len(diffs) > 1 else 1.0
            penalty = sigma2 * n

        cps = _binary_seg_recursive(x, 0, min_size, float(penalty))

        # Build per-position scores: proportion of total SSE gain at nearest cp
        scores = np.zeros(n)
        if cps:
            total_sse = float(np.sum((x - np.mean(x)) ** 2))
            for cp in cps:
                left, right = x[:cp], x[cp:]
                gain = total_sse - (
                    float(np.sum((left - np.mean(left)) ** 2)) +
                    float(np.sum((right - np.mean(right)) ** 2))
                )
                if total_sse > 0:
                    scores[cp] = min(1.0, gain / total_sse)

        return _build_report(ts, cps, scores, f"binary_segmentation(p={penalty:.1f})")

    # ------------------------------------------------------------------
    # Variance ratio
    # ------------------------------------------------------------------

    def variance_ratio(
        self,
        ts: TimeSeries,
        *,
        window: int = 30,
        alpha: float = 0.05,
    ) -> ChangepointReport:
        """Detect variance shifts via a sliding two-sample F-test.

        Two adjacent windows of width *window* are compared at each position.
        A significant difference in variance (p < alpha) signals a
        variance-change changepoint.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        window : int, optional
            Half-window width for each sample.  Default 30.
        alpha : float, optional
            Significance level for the F-test.  Default 0.05.

        Returns
        -------
        ChangepointReport
            ``changepoints`` are positions with a significant variance shift,
            with consecutive positives merged into the maximum-score position.

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *window* < 3 or *alpha* outside (0, 1).

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.changepoint.detector import ChangepointDetector
        >>> rng  = np.random.default_rng(2)
        >>> idx  = pd.date_range("2020", periods=200, freq="D")
        >>> vals = np.concatenate([rng.standard_normal(100) * 0.5,
        ...                        rng.standard_normal(100) * 3.0])
        >>> ts   = TimeSeries(vals, index=idx)
        >>> r    = ChangepointDetector().variance_ratio(ts, window=20)
        >>> r.n_changepoints >= 1
        True
        """
        if not (0 < alpha < 1):
            raise ValueError(f"'alpha' must be in (0, 1), got {alpha}.")
        ts     = self._validate(ts)
        window = validate_positive_int(window, name="window")
        if window < 3:
            raise ValueError(f"'window' must be >= 3, got {window}.")

        x      = _clean_array(ts)
        n      = len(x)
        scores = np.zeros(n)
        sig    = np.zeros(n, dtype=bool)

        for t in range(window, n - window):
            left  = x[t - window : t]
            right = x[t : t + window]

            nl, nr  = len(left), len(right)
            var_l   = float(np.var(left, ddof=1))
            var_r   = float(np.var(right, ddof=1))

            if var_l <= 0 and var_r <= 0:
                continue
            if var_l <= 0:
                var_l = 1e-12
            if var_r <= 0:
                var_r = 1e-12

            # Two-sided F-test: always put larger variance in numerator
            F    = max(var_l, var_r) / min(var_l, var_r)
            df1  = nl - 1
            df2  = nr - 1
            pval = float(2 * sp_stats.f.sf(F, df1, df2))

            scores[t] = 1.0 - pval
            sig[t]    = pval < alpha

        # Merge consecutive significant positions — keep local maximum
        cps: List[int] = []
        in_run = False
        run_start = 0
        run_best  = 0
        run_score = -1.0

        for i in range(n):
            if sig[i]:
                if not in_run:
                    in_run, run_start = True, i
                    run_best, run_score = i, scores[i]
                else:
                    if scores[i] > run_score:
                        run_best, run_score = i, scores[i]
            else:
                if in_run:
                    cps.append(run_best)
                    in_run = False

        if in_run:
            cps.append(run_best)

        return _build_report(ts, cps, scores,
                             f"variance_ratio(w={window},α={alpha})")

    # ------------------------------------------------------------------
    # Segment analysis
    # ------------------------------------------------------------------

    def segment(
        self,
        ts: TimeSeries,
        report: ChangepointReport,
    ) -> pd.DataFrame:
        """Return per-segment statistics for a change-point report.

        Parameters
        ----------
        ts : TimeSeries
            The original series.
        report : ChangepointReport
            Output of any detection method.

        Returns
        -------
        pandas.DataFrame
            One row per segment with columns:
            ``segment``, ``start``, ``end``, ``n_obs``,
            ``mean``, ``std``, ``min``, ``max``.

        Raises
        ------
        TypeError
            If either argument has the wrong type.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.changepoint.detector import ChangepointDetector
        >>> rng  = np.random.default_rng(0)
        >>> idx  = pd.date_range("2020", periods=200, freq="D")
        >>> vals = np.concatenate([rng.standard_normal(100),
        ...                        rng.standard_normal(100) + 4.0])
        >>> ts   = TimeSeries(vals, index=idx)
        >>> det  = ChangepointDetector()
        >>> r    = det.binary_segmentation(ts)
        >>> df   = det.segment(ts, r)
        >>> len(df) == r.n_changepoints + 1
        True
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )
        if not isinstance(report, ChangepointReport):
            raise TypeError(
                f"'report' must be a ChangepointReport, "
                f"got {type(report).__name__!r}."
            )

        breaks = sorted(report.changepoints)
        bounds = [0] + breaks + [ts.n]
        rows   = []

        for seg_idx, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            seg_vals = ts.values[lo:hi]
            finite   = seg_vals[~np.isnan(seg_vals)]
            rows.append({
                "segment": seg_idx,
                "start":   ts.index[lo],
                "end":     ts.index[hi - 1],
                "n_obs":   hi - lo,
                "mean":    float(np.mean(finite)) if len(finite) else np.nan,
                "std":     float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                "min":     float(np.min(finite)) if len(finite) else np.nan,
                "max":     float(np.max(finite)) if len(finite) else np.nan,
            })

        return pd.DataFrame(rows)