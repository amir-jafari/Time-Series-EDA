"""
Missing-value analysis for time series.

Two distinct concepts are handled here:

* **Value NaN** — a timestamp is present in the index but its observed value
  is :data:`numpy.nan`.
* **Index gap** — a timestamp that *should* exist (given the series frequency)
  is absent from the index entirely.

Both are reported by :class:`MissingValueAnalyzer`.  Interpolation of NaN
values is also provided via :meth:`MissingValueAnalyzer.interpolate`.

Classes
-------
MissingValueReport
    Immutable result dataclass returned by :meth:`MissingValueAnalyzer.analyze`.
MissingValueAnalyzer
    Stateless analyzer; all methods accept a :class:`~tseda.core.TimeSeries`
    and return plain Python / numpy objects or a new :class:`~tseda.core.TimeSeries`.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.quality.missing import MissingValueAnalyzer

>>> idx = pd.date_range("2020-01-01", periods=10, freq="D")
>>> vals = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0, 7.0, 8.0, np.nan, 10.0])
>>> ts  = TimeSeries(vals, index=idx)
>>> ana = MissingValueAnalyzer()
>>> report = ana.analyze(ts)
>>> report.n_nan
3
>>> report.pct_nan
30.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from tseda.core.timeseries import TimeSeries
from tseda.core.validator import validate_freq_string

__all__ = ["MissingValueReport", "MissingValueAnalyzer"]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissingValueReport:
    """Immutable summary of missing values in a :class:`~tseda.core.TimeSeries`.

    Attributes
    ----------
    n_nan : int
        Number of NaN values in the observed array.
    pct_nan : float
        Percentage of NaN observations (0–100).
    n_gaps : int
        Number of missing *timestamps* (index gaps) when the series
        frequency is known.  ``-1`` when frequency is unknown.
    gap_locations : list of pandas.Timestamp
        Start timestamp of each index gap.  Empty when ``n_gaps <= 0``.
    longest_nan_run : int
        Length of the longest consecutive run of NaN values.
    nan_run_lengths : list of int
        Lengths of every consecutive NaN run (ascending order).
    nan_positions : numpy.ndarray
        Integer positions (0-based) of all NaN values.
    is_monotone_missing : bool
        ``True`` when all NaN values cluster at the start or end of the
        series (monotone missing pattern — easier to handle).
    """

    n_nan: int
    pct_nan: float
    n_gaps: int
    gap_locations: List[pd.Timestamp]
    longest_nan_run: int
    nan_run_lengths: List[int]
    nan_positions: np.ndarray
    is_monotone_missing: bool

    def __repr__(self) -> str:  # pragma: no cover
        gap_str = (
            f"{self.n_gaps} gap(s)" if self.n_gaps >= 0 else "unknown (no freq)"
        )
        return (
            f"MissingValueReport(\n"
            f"  n_nan              : {self.n_nan} ({self.pct_nan:.1f}%)\n"
            f"  index gaps         : {gap_str}\n"
            f"  longest NaN run    : {self.longest_nan_run}\n"
            f"  is_monotone        : {self.is_monotone_missing}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nan_runs(mask: np.ndarray) -> List[int]:
    """Return a sorted list of consecutive-NaN run lengths."""
    runs: List[int] = []
    run = 0
    for v in mask:
        if v:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return sorted(runs)


def _index_gaps(
    index: pd.DatetimeIndex, freq: str
) -> Tuple[int, List[pd.Timestamp]]:
    """Count and locate timestamps missing from *index* given *freq*.

    Parameters
    ----------
    index:
        The actual datetime index of the series.
    freq:
        Pandas offset alias (e.g., ``"D"``).

    Returns
    -------
    n_gaps : int
    gap_locations : list of pd.Timestamp
        The first missing timestamp for each gap.
    """
    expected = pd.date_range(start=index[0], end=index[-1], freq=freq)
    actual_set = set(index)
    missing = [ts for ts in expected if ts not in actual_set]
    return len(missing), missing


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class MissingValueAnalyzer:
    """Analyze and repair missing values in a :class:`~tseda.core.TimeSeries`.

    This class is **stateless** — instantiate once and call its methods on
    different series objects.

    Methods
    -------
    analyze(ts)
        Return a :class:`MissingValueReport` for *ts*.
    interpolate(ts, method)
        Fill NaN values and return a new :class:`~tseda.core.TimeSeries`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.quality.missing import MissingValueAnalyzer

    >>> idx  = pd.date_range("2020-01-01", periods=5, freq="D")
    >>> vals = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    >>> ts   = TimeSeries(vals, index=idx)
    >>> ana  = MissingValueAnalyzer()
    >>> r = ana.analyze(ts)
    >>> r.n_nan
    2
    >>> filled = ana.interpolate(ts)
    >>> filled.has_nan
    False
    """

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def analyze(self, ts: TimeSeries) -> MissingValueReport:
        """Compute a complete missing-value summary for *ts*.

        Parameters
        ----------
        ts : TimeSeries
            The series to analyze.

        Returns
        -------
        MissingValueReport

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.quality.missing import MissingValueAnalyzer
        >>> idx  = pd.date_range("2020", periods=4, freq="D")
        >>> vals = np.array([1.0, np.nan, np.nan, 4.0])
        >>> report = MissingValueAnalyzer().analyze(TimeSeries(vals, index=idx))
        >>> report.n_nan
        2
        >>> report.longest_nan_run
        2
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )

        values = ts.values
        nan_mask = np.isnan(values)
        n_nan = int(nan_mask.sum())
        pct_nan = 100.0 * n_nan / max(ts.n, 1)
        nan_positions = np.where(nan_mask)[0]

        # Consecutive NaN runs
        runs = _nan_runs(nan_mask)
        longest = runs[-1] if runs else 0

        # Index gaps (only when freq is known)
        if ts.freq is not None:
            try:
                n_gaps, gap_locs = _index_gaps(ts.index, ts.freq)
            except Exception:
                n_gaps, gap_locs = -1, []
        else:
            n_gaps, gap_locs = -1, []

        # Monotone missing: all NaN are at the head or tail
        is_monotone = False
        if n_nan > 0:
            first_nan = int(nan_positions[0])
            last_nan  = int(nan_positions[-1])
            is_monotone = (first_nan == 0) or (last_nan == ts.n - 1)

        return MissingValueReport(
            n_nan=n_nan,
            pct_nan=round(pct_nan, 4),
            n_gaps=n_gaps,
            gap_locations=gap_locs,
            longest_nan_run=longest,
            nan_run_lengths=runs,
            nan_positions=nan_positions,
            is_monotone_missing=is_monotone,
        )

    def interpolate(
        self,
        ts: TimeSeries,
        method: str = "linear",
        *,
        limit: Optional[int] = None,
        fill_value: Optional[float] = None,
    ) -> TimeSeries:
        """Fill NaN values and return a new :class:`~tseda.core.TimeSeries`.

        Parameters
        ----------
        ts : TimeSeries
            Series to fill.
        method : str, optional
            Interpolation strategy.  One of:

            * ``"linear"``   — linear interpolation between neighbours
              (default).  Leading and trailing NaN are filled with the
              nearest observed boundary value when *limit* is ``None``.
            * ``"forward"``  — forward-fill (carry last observed value).
            * ``"backward"`` — backward-fill (carry next observed value).
            * ``"nearest"``  — fill with the nearest non-NaN value.
            * ``"zero"``     — fill with 0.0.
            * ``"constant"`` — fill with *fill_value* (must be provided).
            * ``"spline"``   — cubic spline (requires scipy).

        limit : int, optional
            Maximum number of consecutive NaN values to fill.  ``None``
            fills all gaps.
        fill_value : float, optional
            Used only with ``method="constant"``.

        Returns
        -------
        TimeSeries
            A new series with NaN values replaced.  Metadata (name, unit,
            freq, description) is preserved.

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *method* is not recognised, or if ``"constant"`` is chosen
            without supplying *fill_value*.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.quality.missing import MissingValueAnalyzer
        >>> idx  = pd.date_range("2020", periods=5, freq="D")
        >>> vals = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        >>> ts   = TimeSeries(vals, index=idx)
        >>> ana  = MissingValueAnalyzer()

        Linear interpolation:

        >>> filled = ana.interpolate(ts, "linear")
        >>> filled.values.tolist()
        [1.0, 2.0, 3.0, 4.0, 5.0]

        Forward fill:

        >>> fwd = ana.interpolate(ts, "forward")
        >>> fwd.values.tolist()
        [1.0, 1.0, 1.0, 4.0, 5.0]
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )

        _VALID = {"linear", "forward", "backward", "nearest", "zero",
                  "constant", "spline"}
        if method not in _VALID:
            raise ValueError(
                f"Unknown interpolation method {method!r}. "
                f"Valid options: {sorted(_VALID)}."
            )

        if method == "constant":
            if fill_value is None:
                raise ValueError(
                    "method='constant' requires a numeric 'fill_value'."
                )

        series = ts.to_series().copy()

        if method == "linear":
            filled = series.interpolate(method="index", limit=limit)
            # pandas interpolate(method='index') leaves leading/trailing NaN;
            # fill them with nearest boundary value when no limit is imposed.
            if limit is None:
                filled = filled.ffill().bfill()
        elif method == "forward":
            filled = series.ffill(limit=limit)
        elif method == "backward":
            filled = series.bfill(limit=limit)
        elif method == "nearest":
            filled = series.interpolate(method="nearest", limit=limit)
        elif method == "zero":
            filled = series.fillna(0.0)
        elif method == "constant":
            filled = series.fillna(float(fill_value))  # type: ignore[arg-type]
        else:  # spline
            try:
                from scipy.interpolate import CubicSpline
            except ImportError as exc:
                raise ImportError(
                    "method='spline' requires scipy. "
                    "Install it with: pip install scipy"
                ) from exc
            not_nan = ~series.isna()
            if not_nan.sum() < 2:
                raise ValueError(
                    "method='spline' requires at least 2 non-NaN observations."
                )
            x_all  = np.arange(len(series), dtype=float)
            x_obs  = x_all[not_nan.values]
            y_obs  = series.values[not_nan.values]
            cs     = CubicSpline(x_obs, y_obs, extrapolate=False)
            filled_vals = series.values.copy()
            nan_idx = np.where(series.isna().values)[0]
            # Only fill within the observed range
            in_range = (nan_idx >= x_obs[0]) & (nan_idx <= x_obs[-1])
            filled_vals[nan_idx[in_range]] = cs(x_all[nan_idx[in_range]])
            filled = pd.Series(filled_vals, index=series.index)

        # Preserve leading/trailing NaN if limit was applied
        return TimeSeries(
            filled.values,
            index=filled.index,
            name=ts.name,
            freq=ts.freq,
            unit=ts.unit,
            description=ts.description,
        )