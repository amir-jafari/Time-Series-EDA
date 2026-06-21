"""
Flat-line and near-constant segment detection for time series.

*Timestamp* duplicates are rejected at construction time by
:func:`~tseda.core.validator.validate_datetime_index`.  This module
addresses the complementary problem: consecutive identical or near-zero
*values*, which typically signal:

* A stuck sensor / ADC saturation.
* A data-pipeline bug that forward-filled data without marking it.
* A genuine flat segment that may confuse differencing-based methods.

Classes
-------
FlatlineReport
    Immutable result dataclass returned by :meth:`DuplicateDetector.flatline`.
DuplicateDetector
    Stateless detector.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.quality.duplicates import DuplicateDetector

>>> idx  = pd.date_range("2020-01-01", periods=10, freq="D")
>>> vals = np.array([1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0])
>>> ts   = TimeSeries(vals, index=idx)
>>> det  = DuplicateDetector()
>>> r    = det.flatline(ts, min_run=3)
>>> r.n_flatline_runs
1
>>> r.longest_run
4
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

from tseda.core.timeseries import TimeSeries
from tseda.core.validator import validate_positive_int

__all__ = ["FlatlineReport", "DuplicateDetector"]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlatlineReport:
    """Immutable summary of flat-line segments in a :class:`~tseda.core.TimeSeries`.

    Attributes
    ----------
    n_flatline_runs : int
        Number of runs that meet or exceed *min_run* in length.
    longest_run : int
        Length of the single longest flat-line run.
    total_flatline_points : int
        Total number of observations that belong to a qualifying flat-line
        run (includes the first observation of each run).
    runs : list of (start_pos, end_pos, value)
        Each element is a tuple ``(start_pos, end_pos, value)`` where
        *start_pos* and *end_pos* are 0-based integer positions and
        *value* is the repeated value.  Only runs of length >= *min_run*
        are included.
    mask : numpy.ndarray
        Boolean array; ``True`` at every position that is part of a
        qualifying flat-line run.
    min_run : int
        The minimum run length used for this report.
    """

    n_flatline_runs: int
    longest_run: int
    total_flatline_points: int
    runs: List[Tuple[int, int, float]]
    mask: np.ndarray
    min_run: int

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FlatlineReport(\n"
            f"  n_flatline_runs       : {self.n_flatline_runs}\n"
            f"  longest_run           : {self.longest_run}\n"
            f"  total_flatline_points : {self.total_flatline_points}\n"
            f"  min_run               : {self.min_run}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_runs(
    values: np.ndarray, *, min_run: int, atol: float
) -> List[Tuple[int, int, float]]:
    """Return all consecutive runs of near-equal values.

    Parameters
    ----------
    values:
        1-D float array (NaN values break any run they belong to).
    min_run:
        Minimum run length to report.
    atol:
        Absolute tolerance for equality.

    Returns
    -------
    list of (start, end, value)
        *end* is inclusive.
    """
    n = len(values)
    if n == 0:
        return []

    runs: List[Tuple[int, int, float]] = []
    start = 0
    ref   = values[0]

    for i in range(1, n):
        v = values[i]
        same = (not np.isnan(ref)) and (not np.isnan(v)) and abs(v - ref) <= atol
        if same:
            continue
        run_len = i - start
        if run_len >= min_run and not np.isnan(ref):
            runs.append((start, i - 1, float(ref)))
        start = i
        ref   = v

    # Handle last run
    run_len = n - start
    if run_len >= min_run and not np.isnan(ref):
        runs.append((start, n - 1, float(ref)))

    return runs


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class DuplicateDetector:
    """Detect consecutive duplicate (flat-line) value runs.

    Methods
    -------
    flatline(ts, min_run=3, atol=0.0)
        Detect flat-line segments of repeated values.
    near_zero(ts, min_run=3, threshold=1e-8)
        Detect segments where the series is stuck near zero.
    remove_flatlines(ts, report)
        Replace flat-line positions with NaN (keeping the first value).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.quality.duplicates import DuplicateDetector

    >>> idx  = pd.date_range("2020", periods=8, freq="D")
    >>> vals = np.array([1.0, 5.0, 5.0, 5.0, 5.0, 2.0, 3.0, 4.0])
    >>> ts   = TimeSeries(vals, index=idx)
    >>> det  = DuplicateDetector()
    >>> r    = det.flatline(ts, min_run=3)
    >>> r.n_flatline_runs
    1
    >>> r.longest_run
    4
    """

    @staticmethod
    def _validate(ts: object) -> TimeSeries:
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )
        return ts  # type: ignore[return-value]

    def flatline(
        self,
        ts: TimeSeries,
        min_run: int = 3,
        *,
        atol: float = 0.0,
    ) -> FlatlineReport:
        """Detect consecutive runs of identical (or near-identical) values.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        min_run : int, optional
            Minimum number of consecutive identical observations to constitute
            a "flat line".  Default ``3``.
        atol : float, optional
            Absolute tolerance for equality.  Two values ``a`` and ``b`` are
            considered equal when ``|a - b| <= atol``.  Default ``0.0``
            (exact equality).

        Returns
        -------
        FlatlineReport

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *min_run* < 2.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.quality.duplicates import DuplicateDetector

        Exact flat line of length 4:

        >>> idx  = pd.date_range("2020", periods=7, freq="D")
        >>> vals = np.array([1.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0])
        >>> ts   = TimeSeries(vals, index=idx)
        >>> r    = DuplicateDetector().flatline(ts, min_run=3)
        >>> r.n_flatline_runs
        1
        >>> r.runs[0]
        (1, 4, 3.0)

        No flat line (min_run too high):

        >>> r2 = DuplicateDetector().flatline(ts, min_run=5)
        >>> r2.n_flatline_runs
        0
        """
        ts = self._validate(ts)
        min_run = validate_positive_int(min_run, name="min_run")
        if min_run < 2:
            raise ValueError(
                f"'min_run' must be >= 2 to detect a repeated run, got {min_run}."
            )
        if atol < 0:
            raise ValueError(f"'atol' must be >= 0, got {atol}.")

        vals = ts.values
        runs = _find_runs(vals, min_run=min_run, atol=atol)

        mask = np.zeros(ts.n, dtype=bool)
        for start, end, _ in runs:
            mask[start : end + 1] = True

        longest = max((e - s + 1 for s, e, _ in runs), default=0)
        total   = int(mask.sum())

        return FlatlineReport(
            n_flatline_runs=len(runs),
            longest_run=longest,
            total_flatline_points=total,
            runs=runs,
            mask=mask,
            min_run=min_run,
        )

    def near_zero(
        self,
        ts: TimeSeries,
        min_run: int = 3,
        *,
        threshold: float = 1e-8,
    ) -> FlatlineReport:
        """Detect segments where the series is stuck near zero.

        Only consecutive runs where **every** value satisfies
        ``|x| <= threshold`` are reported.  This differs from
        :meth:`flatline`, which detects any repeated value regardless of
        magnitude.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        min_run : int, optional
            Minimum run length.  Default ``3``.
        threshold : float, optional
            Maximum absolute value to count as "near zero".
            Default ``1e-8``.

        Returns
        -------
        FlatlineReport
            Runs where every value satisfies ``|x| <= threshold``.

        Raises
        ------
        ValueError
            If *threshold* < 0.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.quality.duplicates import DuplicateDetector

        >>> idx  = pd.date_range("2020", periods=8, freq="D")
        >>> vals = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        >>> ts   = TimeSeries(vals, index=idx)
        >>> r    = DuplicateDetector().near_zero(ts, min_run=3)
        >>> r.n_flatline_runs
        1
        """
        ts = self._validate(ts)
        min_run = validate_positive_int(min_run, name="min_run")
        if threshold < 0:
            raise ValueError(f"'threshold' must be >= 0, got {threshold}.")

        vals = ts.values
        nz_mask = np.abs(vals) <= threshold  # True where near-zero

        # Find runs of consecutive True values in nz_mask
        n = len(vals)
        runs: List[Tuple[int, int, float]] = []
        in_run = False
        run_start = 0

        for i in range(n):
            if nz_mask[i]:
                if not in_run:
                    run_start = i
                    in_run = True
            else:
                if in_run:
                    run_len = i - run_start
                    if run_len >= min_run:
                        runs.append((run_start, i - 1, float(vals[run_start])))
                    in_run = False

        if in_run:
            run_len = n - run_start
            if run_len >= min_run:
                runs.append((run_start, n - 1, float(vals[run_start])))

        mask = np.zeros(n, dtype=bool)
        for s, e, _ in runs:
            mask[s : e + 1] = True

        longest = max((e - s + 1 for s, e, _ in runs), default=0)

        return FlatlineReport(
            n_flatline_runs=len(runs),
            longest_run=longest,
            total_flatline_points=int(mask.sum()),
            runs=runs,
            mask=mask,
            min_run=min_run,
        )

    def remove_flatlines(
        self,
        ts: TimeSeries,
        report: FlatlineReport,
        *,
        keep_first: bool = True,
    ) -> TimeSeries:
        """Replace flat-line positions with NaN.

        Parameters
        ----------
        ts : TimeSeries
            The original series.
        report : FlatlineReport
            Result from :meth:`flatline` or :meth:`near_zero`.
        keep_first : bool, optional
            When ``True`` (default), the *first* observation of each
            flat-line run is preserved; only the *repeated* copies are
            set to NaN.  When ``False``, the entire run including the
            first observation is set to NaN.

        Returns
        -------
        TimeSeries
            A new series with flat-line values replaced by NaN.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.quality.duplicates import DuplicateDetector

        >>> idx  = pd.date_range("2020", periods=6, freq="D")
        >>> vals = np.array([1.0, 5.0, 5.0, 5.0, 2.0, 3.0])
        >>> ts   = TimeSeries(vals, index=idx)
        >>> det  = DuplicateDetector()
        >>> r    = det.flatline(ts, min_run=3)
        >>> cleaned = det.remove_flatlines(ts, r, keep_first=True)
        >>> cleaned.n_nan
        2
        """
        if not isinstance(report, FlatlineReport):
            raise TypeError(
                f"'report' must be a FlatlineReport, got {type(report).__name__!r}."
            )
        ts = self._validate(ts)
        vals = ts.values.copy()

        for start, end, _ in report.runs:
            replace_from = start + 1 if keep_first else start
            vals[replace_from : end + 1] = np.nan

        return TimeSeries(
            vals, index=ts.index, name=ts.name,
            freq=ts.freq, unit=ts.unit, description=ts.description,
        )