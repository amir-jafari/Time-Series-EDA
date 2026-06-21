"""
Input validation utilities for tseda.

Every public function in this module raises a descriptive :class:`TypeError`
or :class:`ValueError` on bad input and returns the canonicalised value on
success.  All heavy lifting of data coercion lives here so that
:class:`~tseda.core.TimeSeries` and analysis modules stay clean.

Functions
---------
validate_data_array
    Coerce arbitrary numeric input to a 1-D ``float64`` :class:`numpy.ndarray`.
validate_datetime_index
    Coerce arbitrary input to a sorted, duplicate-free
    :class:`pandas.DatetimeIndex`.
validate_positive_int
    Assert that a value is a positive integer.
validate_lags
    Assert that the requested lag count is sensible relative to series length.
validate_freq_string
    Assert that a string is a recognised pandas offset alias.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

__all__ = [
    "validate_data_array",
    "validate_datetime_index",
    "validate_positive_int",
    "validate_lags",
    "validate_freq_string",
]

# ---------------------------------------------------------------------------
# Public validators
# ---------------------------------------------------------------------------


def validate_data_array(data: Any, *, name: str = "data") -> np.ndarray:
    """Coerce *data* to a 1-D ``float64`` :class:`numpy.ndarray`.

    Parameters
    ----------
    data:
        Numeric input.  Accepted types:

        * :class:`numpy.ndarray` — must be 1-D.
        * :class:`pandas.Series` — values extracted; index ignored.
        * :class:`list` or :class:`tuple` — must be flat and numeric.

    name:
        Variable name used in error messages (default ``"data"``).

    Returns
    -------
    numpy.ndarray
        1-D array of dtype ``float64``.  NaN values are preserved.

    Raises
    ------
    TypeError
        If *data* is not a recognised type.
    ValueError
        If *data* is not 1-D or contains non-numeric elements.

    Examples
    --------
    >>> validate_data_array([1.0, 2.0, 3.0])
    array([1., 2., 3.])
    >>> validate_data_array(pd.Series([1, 2, 3]))
    array([1., 2., 3.])
    """
    if isinstance(data, pd.Series):
        arr = data.to_numpy(dtype=float, na_value=np.nan)
    elif isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError(
                f"'{name}' must be 1-D, got shape {data.shape}. "
                "Use MultiTimeSeries for multivariate data."
            )
        arr = data.astype(float, copy=True)
    elif isinstance(data, (list, tuple)):
        try:
            arr = np.asarray(data, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'{name}' could not be converted to a numeric array: {exc}"
            ) from exc
        if arr.ndim != 1:
            raise ValueError(
                f"'{name}' must be a flat 1-D sequence, got shape {arr.shape}."
            )
    else:
        raise TypeError(
            f"'{name}' must be array-like (ndarray, Series, list, or tuple), "
            f"got {type(data).__name__!r}."
        )

    if arr.size == 0:
        raise ValueError(f"'{name}' must contain at least one element.")

    return arr


def validate_datetime_index(index: Any, *, name: str = "index") -> pd.DatetimeIndex:
    """Coerce *index* to a sorted, duplicate-free :class:`pandas.DatetimeIndex`.

    Parameters
    ----------
    index:
        Datetime-like input.  Accepted types:

        * :class:`pandas.DatetimeIndex`
        * :class:`pandas.Series` with datetime dtype
        * :class:`list` or :class:`numpy.ndarray` of datetime-like strings
          or :class:`numpy.datetime64` values

    name:
        Variable name used in error messages (default ``"index"``).

    Returns
    -------
    pandas.DatetimeIndex
        Validated, monotonically increasing, duplicate-free index.

    Raises
    ------
    TypeError
        If *index* is not a recognised type.
    ValueError
        If *index* is not monotonically increasing or contains duplicates.

    Examples
    --------
    >>> idx = pd.date_range("2020-01-01", periods=5, freq="D")
    >>> validate_datetime_index(idx)  # doctest: +ELLIPSIS
    DatetimeIndex(['2020-01-01', ..., '2020-01-05'], dtype='datetime64[ns]', freq='D')
    """
    if isinstance(index, pd.DatetimeIndex):
        dti = index
    elif isinstance(index, pd.Series):
        try:
            dti = pd.DatetimeIndex(index)
        except Exception as exc:
            raise TypeError(
                f"'{name}' Series could not be converted to DatetimeIndex: {exc}"
            ) from exc
    elif isinstance(index, (list, np.ndarray)):
        try:
            dti = pd.DatetimeIndex(index)
        except Exception as exc:
            raise TypeError(
                f"'{name}' sequence could not be parsed as datetimes: {exc}"
            ) from exc
    else:
        raise TypeError(
            f"'{name}' must be a DatetimeIndex or datetime-like sequence, "
            f"got {type(index).__name__!r}."
        )

    if len(dti) == 0:
        raise ValueError(f"'{name}' must contain at least one timestamp.")

    if not dti.is_monotonic_increasing:
        raise ValueError(
            f"'{name}' must be monotonically increasing (time-sorted). "
            "Sort your data before constructing a TimeSeries."
        )

    if dti.has_duplicates:
        n_dupes = int(dti.duplicated().sum())
        raise ValueError(
            f"'{name}' contains {n_dupes} duplicate timestamp(s). "
            "Aggregate or drop duplicates before constructing a TimeSeries."
        )

    return dti


def validate_positive_int(value: Any, *, name: str = "value") -> int:
    """Assert that *value* is a positive integer.

    Parameters
    ----------
    value:
        The candidate value.
    name:
        Variable name used in error messages.

    Returns
    -------
    int
        The validated integer.

    Raises
    ------
    TypeError
        If *value* is not an integer type.
    ValueError
        If *value* is less than 1.

    Examples
    --------
    >>> validate_positive_int(5)
    5
    """
    if not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"'{name}' must be an integer, got {type(value).__name__!r}."
        )
    v = int(value)
    if v < 1:
        raise ValueError(f"'{name}' must be >= 1, got {v}.")
    return v


def validate_lags(lags: int, n: int, *, name: str = "lags") -> int:
    """Assert that *lags* is a sensible lag count for a series of length *n*.

    The upper bound is ``n // 2`` because computing autocorrelations at lags
    approaching *n* produces unreliable estimates.

    Parameters
    ----------
    lags:
        Requested number of lags.
    n:
        Length of the time series.
    name:
        Variable name used in error messages.

    Returns
    -------
    int
        The validated lag count.

    Raises
    ------
    ValueError
        If *lags* is not in ``[1, n // 2]``.

    Examples
    --------
    >>> validate_lags(40, 100)
    40
    """
    lags = validate_positive_int(lags, name=name)
    max_lags = n // 2
    if lags > max_lags:
        raise ValueError(
            f"'{name}' ({lags}) exceeds the maximum allowed value of n // 2 = {max_lags} "
            f"for a series of length {n}."
        )
    return lags


def validate_freq_string(freq: Any, *, name: str = "freq") -> str:
    """Assert that *freq* is a non-empty string accepted by :func:`pandas.tseries.frequencies.to_offset`.

    Parameters
    ----------
    freq:
        Candidate frequency string (e.g., ``"D"``, ``"h"``, ``"MS"``).
    name:
        Variable name used in error messages.

    Returns
    -------
    str
        The validated frequency string.

    Raises
    ------
    TypeError
        If *freq* is not a string.
    ValueError
        If *freq* is not recognised by pandas.

    Examples
    --------
    >>> validate_freq_string("D")
    'D'
    >>> validate_freq_string("15min")
    '15min'
    """
    if not isinstance(freq, str):
        raise TypeError(
            f"'{name}' must be a string (e.g., 'D', 'h', 'MS'), "
            f"got {type(freq).__name__!r}."
        )
    freq = freq.strip()
    if not freq:
        raise ValueError(f"'{name}' must not be empty.")

    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"'{name}' = {freq!r} is not a recognised pandas offset alias. "
            "See https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases"
        ) from exc
    if offset is None:
        raise ValueError(
            f"'{name}' = {freq!r} is not a recognised pandas offset alias. "
            "See https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases"
        )
    return freq