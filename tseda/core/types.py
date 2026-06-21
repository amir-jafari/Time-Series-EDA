"""
Type aliases and enumerations used throughout tseda.

All public symbols here are re-exported from :mod:`tseda.core` so downstream
modules import from one place::

    from tseda.core import ArrayLike, Frequency, AggMethod
"""
from __future__ import annotations

from enum import Enum
from typing import Union

import numpy as np
import pandas as pd

__all__ = [
    "ArrayLike",
    "DatetimeLike",
    "Frequency",
    "AggMethod",
    "DiffMethod",
]

# ---------------------------------------------------------------------------
# Scalar / array type aliases
# ---------------------------------------------------------------------------

#: Any 1-D numeric input that can be coerced to a numpy array.
ArrayLike = Union[np.ndarray, list, tuple, pd.Series]

#: Any input that can be coerced to a :class:`pandas.DatetimeIndex`.
DatetimeLike = Union[pd.DatetimeIndex, pd.Series, list, np.ndarray]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Frequency(str, Enum):
    """Canonical pandas offset aliases recognised by tseda.

    The string value of each member is a valid ``freq`` argument to
    :func:`pandas.date_range` and :meth:`pandas.Series.resample`.

    Examples
    --------
    >>> Frequency.DAILY.value
    'D'
    >>> Frequency.DAILY == "D"
    True
    """

    SECONDLY = "S"
    MINUTELY = "min"
    HOURLY = "h"
    DAILY = "D"
    BUSINESS_DAILY = "B"
    WEEKLY = "W"
    MONTHLY_START = "MS"
    MONTHLY_END = "ME"
    QUARTERLY_START = "QS"
    QUARTERLY_END = "QE"
    ANNUAL_START = "YS"
    ANNUAL_END = "YE"


class AggMethod(str, Enum):
    """Aggregation functions available when resampling a :class:`~tseda.core.TimeSeries`.

    The string value matches the :class:`pandas.core.resample.Resampler`
    method name.

    Examples
    --------
    >>> AggMethod.MEAN.value
    'mean'
    """

    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    FIRST = "first"
    LAST = "last"
    STD = "std"
    VAR = "var"
    COUNT = "count"


class DiffMethod(str, Enum):
    """Differencing mode for :meth:`~tseda.core.TimeSeries.diff`.

    Attributes
    ----------
    SIMPLE :
        ``y[t] - y[t-k]``  (standard first/kth difference).
    LOG :
        ``log(y[t]) - log(y[t-k])``  (log return / percent change in log scale).
    PERCENT :
        ``(y[t] - y[t-k]) / y[t-k]``  (relative change).
    """

    SIMPLE = "simple"
    LOG = "log"
    PERCENT = "percent"