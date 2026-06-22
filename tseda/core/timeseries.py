"""
Core :class:`TimeSeries` data structure.

This module defines the single object that every tseda analysis operates on.
It wraps a :class:`pandas.Series` with a :class:`pandas.DatetimeIndex`,
validates inputs, infers frequency, and exposes a clean, chainable API for
basic transforms.

Design Principles
-----------------
* **Effectively immutable** — transform methods return new :class:`TimeSeries`
  objects; the original is never modified.
* **Single source of truth** — all data lives in one private ``pd.Series``;
  ``values`` and ``index`` are read-only views.
* **Explicit over implicit** — every parameter has a clear type and a
  descriptive error message when violated.
* **Minimal dependencies** — only numpy, pandas, and scipy at runtime.

Examples
--------
Build from a numpy array and a date range:

>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> idx = pd.date_range("2020-01-01", periods=365, freq="D")
>>> ts = TimeSeries(np.random.randn(365), index=idx, name="returns", unit="USD")
>>> ts.n
365
>>> ts.freq
'D'

Build from an existing :class:`pandas.Series`:

>>> s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020", periods=3, freq="D"))
>>> ts2 = TimeSeries.from_series(s, name="price")
>>> ts2.start
Timestamp('2020-01-01 00:00:00')
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from tseda.core.types import AggMethod, ArrayLike, DatetimeLike, DiffMethod, Frequency
from tseda.core.validator import (
    validate_data_array,
    validate_datetime_index,
    validate_freq_string,
    validate_positive_int,
)

__all__ = ["TimeSeries"]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Map median gap (seconds) → pandas freq alias.
# Ordered from smallest to largest; each entry is (seconds, alias).
_GAP_TO_ALIAS: list[tuple[float, str]] = [
    (1.0,           "s"),
    (60.0,          "min"),
    (3_600.0,       "h"),
    (86_400.0,      "D"),
    (7 * 86_400.0,  "W"),
    (30 * 86_400.0, "MS"),
    (91 * 86_400.0, "QS"),
    (365 * 86_400.0, "YS"),
]

# Mapping from freq alias prefix → human-readable label.
_ALIAS_TO_LABEL: dict[str, str] = {
    "s":   "Secondly",
    "T":   "Minutely",
    "min": "Minutely",
    "H":   "Hourly",
    "h":   "Hourly",
    "D":   "Daily",
    "B":   "Business daily",
    "W":   "Weekly",
    "M":   "Monthly (end)",
    "MS":  "Monthly (start)",
    "ME":  "Monthly (end)",
    "Q":   "Quarterly (end)",
    "QS":  "Quarterly (start)",
    "QE":  "Quarterly (end)",
    "A":   "Annual (end)",
    "AS":  "Annual (start)",
    "Y":   "Annual (end)",
    "YS":  "Annual (start)",
    "YE":  "Annual (end)",
}


def _infer_freq(index: pd.DatetimeIndex) -> Optional[str]:
    """Return a pandas offset alias for *index*, or ``None`` if indeterminate.

    Tries :func:`pandas.infer_freq` first; falls back to a median-gap
    heuristic (within 10 % of a known period).
    """
    if len(index) < 3:
        return None

    freq = pd.infer_freq(index)
    if freq is not None:
        return freq

    # Median gap in seconds — compute via timedelta so result is unit-independent
    # (pandas 2.2+ uses second-resolution DatetimeIndex for day-level freq, so
    # asi8 / astype(int64) is no longer guaranteed to be nanoseconds).
    try:
        gaps_td = np.diff(index.to_numpy())            # timedelta64[*]
        gaps_s  = gaps_td / np.timedelta64(1, "s")    # → float seconds
        median_s = float(np.median(gaps_s))
    except Exception:
        return None

    for threshold_s, alias in _GAP_TO_ALIAS:
        if abs(median_s - threshold_s) / threshold_s < 0.10:
            return alias

    return None


def _freq_label(freq: Optional[str]) -> str:
    """Return a human-readable label for a pandas freq alias."""
    if freq is None:
        return "Irregular / unknown"
    # Strip leading multiplier digits (e.g. "15min" → "min")
    key = freq.lstrip("0123456789")
    return _ALIAS_TO_LABEL.get(key, freq)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TimeSeries:
    """Univariate time series with a :class:`pandas.DatetimeIndex`.

    Parameters
    ----------
    data:
        Numeric values.  Accepted types:

        * 1-D :class:`numpy.ndarray`
        * :class:`pandas.Series` — values are extracted; the Series index
          is used unless *index* is also provided.
        * :class:`list` or :class:`tuple` of numbers

    index:
        Datetime timestamps aligned with *data*.  When *data* is a
        :class:`pandas.Series` with a :class:`pandas.DatetimeIndex` this
        argument may be omitted.  Accepted types:

        * :class:`pandas.DatetimeIndex`
        * :class:`list` / :class:`numpy.ndarray` of datetime-like strings
          or :class:`numpy.datetime64` objects

    name:
        Short identifier for the series (used in plots and reports).
        Default ``"value"``.
    freq:
        Pandas offset alias (e.g., ``"D"``, ``"h"``, ``"MS"``).
        When ``None`` (default) the frequency is inferred automatically.
    unit:
        Physical unit of the values (e.g., ``"USD"``, ``"°C"``).
        Purely informational — used in axis labels.
    description:
        Free-text description stored in :attr:`metadata`.

    Raises
    ------
    TypeError
        If *data* or *index* have an unsupported type.
    ValueError
        If *data* and *index* have different lengths, if *index* is not
        monotonically increasing, or if *index* contains duplicates.

    Examples
    --------
    From a numpy array:

    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> idx = pd.date_range("2020-01-01", periods=5, freq="D")
    >>> ts = TimeSeries([10.0, 11.5, 9.8, 12.0, 11.0], index=idx)
    >>> ts.n
    5

    From a pandas Series:

    >>> s = pd.Series([1, 2, 3], index=pd.date_range("2020", periods=3, freq="D"))
    >>> ts = TimeSeries.from_series(s)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        data: Union[ArrayLike, pd.Series],
        *,
        index: Optional[DatetimeLike] = None,
        name: str = "value",
        freq: Optional[str] = None,
        unit: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        # ── resolve index ──────────────────────────────────────────────
        if isinstance(data, pd.Series) and index is None:
            # Use the Series' own index if it is already a DatetimeIndex.
            if isinstance(data.index, pd.DatetimeIndex):
                index = data.index
            else:
                raise ValueError(
                    "When constructing from a pandas Series without an explicit "
                    "'index', the Series must already have a DatetimeIndex.  "
                    "Pass 'index=' or use TimeSeries.from_series()."
                )
        elif index is None:
            raise ValueError(
                "'index' is required when 'data' is not a pandas Series "
                "with a DatetimeIndex."
            )

        # ── validate data ──────────────────────────────────────────────
        values: np.ndarray = validate_data_array(data, name="data")

        # ── validate index ─────────────────────────────────────────────
        dti: pd.DatetimeIndex = validate_datetime_index(index, name="index")

        # ── length check ───────────────────────────────────────────────
        if len(values) != len(dti):
            raise ValueError(
                f"'data' and 'index' must have the same length; "
                f"got {len(values)} values and {len(dti)} timestamps."
            )

        # ── store immutable internal Series ───────────────────────────
        self._data: pd.Series = pd.Series(values, index=dti, dtype=float)

        # ── metadata ──────────────────────────────────────────────────
        self._name: str = str(name)
        self._unit: Optional[str] = str(unit) if unit is not None else None
        self._description: Optional[str] = (
            str(description) if description is not None else None
        )

        # ── frequency ─────────────────────────────────────────────────
        if freq is not None:
            self._freq: Optional[str] = validate_freq_string(freq, name="freq")
        else:
            self._freq = _infer_freq(dti)

        # ── cached derived attributes ──────────────────────────────────
        self._is_regular: bool = self._compute_is_regular()

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_series(
        cls,
        series: pd.Series,
        *,
        name: Optional[str] = None,
        freq: Optional[str] = None,
        unit: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "TimeSeries":
        """Construct a :class:`TimeSeries` from a :class:`pandas.Series`.

        Parameters
        ----------
        series:
            Must have a :class:`pandas.DatetimeIndex`.
        name:
            Override the Series' ``.name`` attribute.  When ``None`` the
            Series name (if any) is used, falling back to ``"value"``.
        freq, unit, description:
            Forwarded to :class:`TimeSeries.__init__`.

        Returns
        -------
        TimeSeries

        Examples
        --------
        >>> s = pd.Series([1.0, 2.0], index=pd.date_range("2020", periods=2, freq="D"))
        >>> TimeSeries.from_series(s, name="x").name
        'x'
        """
        if not isinstance(series, pd.Series):
            raise TypeError(
                f"'series' must be a pandas.Series, got {type(series).__name__!r}."
            )
        _name = name if name is not None else (str(series.name) if series.name is not None else "value")
        return cls(
            series,
            name=_name,
            freq=freq,
            unit=unit,
            description=description,
        )

    @classmethod
    def from_arrays(
        cls,
        values: ArrayLike,
        index: DatetimeLike,
        *,
        name: str = "value",
        freq: Optional[str] = None,
        unit: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "TimeSeries":
        """Construct a :class:`TimeSeries` from parallel arrays.

        Parameters
        ----------
        values:
            1-D numeric array.
        index:
            Datetime-like array of the same length.
        name, freq, unit, description:
            Forwarded to :class:`TimeSeries.__init__`.

        Returns
        -------
        TimeSeries

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> vals = np.array([1.0, 2.0, 3.0])
        >>> idx  = pd.date_range("2021-01-01", periods=3, freq="D")
        >>> TimeSeries.from_arrays(vals, idx).n
        3
        """
        return cls(values, index=index, name=name, freq=freq, unit=unit,
                   description=description)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        column: str,
        *,
        name: Optional[str] = None,
        freq: Optional[str] = None,
        unit: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "TimeSeries":
        """Extract one column from a :class:`pandas.DataFrame`.

        Parameters
        ----------
        df:
            Source DataFrame.  Must have a :class:`pandas.DatetimeIndex`.
        column:
            Column name to extract.
        name:
            Override the column name as the series name.
        freq, unit, description:
            Forwarded to :class:`TimeSeries.__init__`.

        Returns
        -------
        TimeSeries

        Raises
        ------
        KeyError
            If *column* is not in *df*.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"temp": [20.0, 21.0, 19.5]},
        ...                    index=pd.date_range("2020", periods=3, freq="D"))
        >>> TimeSeries.from_dataframe(df, "temp").name
        'temp'
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"'df' must be a pandas.DataFrame, got {type(df).__name__!r}."
            )
        if column not in df.columns:
            raise KeyError(
                f"Column {column!r} not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        series = df[column]
        _name = name if name is not None else column
        return cls.from_series(series, name=_name, freq=freq, unit=unit,
                               description=description)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_is_regular(self) -> bool:
        """Return ``True`` when all consecutive time gaps are identical."""
        if len(self._data) < 2:
            return True
        gaps_td = np.diff(self._data.index.to_numpy())   # timedelta64[*] — unit-agnostic
        return bool(np.all(gaps_td == gaps_td[0]))

    def _copy_with(self, new_data: pd.Series, **meta_overrides) -> "TimeSeries":
        """Return a new :class:`TimeSeries` sharing metadata with *self*."""
        return TimeSeries(
            new_data.values,
            index=new_data.index,
            name=meta_overrides.get("name", self._name),
            freq=meta_overrides.get("freq", None),   # re-infer from new index
            unit=meta_overrides.get("unit", self._unit),
            description=meta_overrides.get("description", self._description),
        )

    # ------------------------------------------------------------------
    # Core properties — data access
    # ------------------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """1-D ``float64`` array of observed values.

        Returns
        -------
        numpy.ndarray
            A *copy* to protect the internal state.
        """
        return self._data.values.copy()

    @property
    def index(self) -> pd.DatetimeIndex:
        """Datetime index of the series.

        Returns
        -------
        pandas.DatetimeIndex
        """
        return self._data.index

    @property
    def n(self) -> int:
        """Number of observations.

        Returns
        -------
        int
        """
        return len(self._data)

    @property
    def start(self) -> pd.Timestamp:
        """Timestamp of the first observation.

        Returns
        -------
        pandas.Timestamp
        """
        return self._data.index[0]

    @property
    def end(self) -> pd.Timestamp:
        """Timestamp of the last observation.

        Returns
        -------
        pandas.Timestamp
        """
        return self._data.index[-1]

    @property
    def duration(self) -> pd.Timedelta:
        """Wall-clock span from the first to the last observation.

        Returns
        -------
        pandas.Timedelta
        """
        return self.end - self.start

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier for the series.

        Returns
        -------
        str
        """
        return self._name

    @property
    def unit(self) -> Optional[str]:
        """Physical unit of the values, or ``None`` if unspecified.

        Returns
        -------
        str or None
        """
        return self._unit

    @property
    def description(self) -> Optional[str]:
        """Free-text description, or ``None`` if unspecified.

        Returns
        -------
        str or None
        """
        return self._description

    # ------------------------------------------------------------------
    # Frequency properties
    # ------------------------------------------------------------------

    @property
    def freq(self) -> Optional[str]:
        """Pandas offset alias (e.g., ``"D"``), or ``None`` for irregular data.

        Returns
        -------
        str or None
        """
        return self._freq

    @property
    def freq_label(self) -> str:
        """Human-readable frequency label (e.g., ``"Daily"``).

        Returns
        -------
        str
        """
        return _freq_label(self._freq)

    # ------------------------------------------------------------------
    # Quality properties
    # ------------------------------------------------------------------

    @property
    def has_nan(self) -> bool:
        """``True`` when at least one value is NaN.

        Returns
        -------
        bool
        """
        return bool(self._data.isna().any())

    @property
    def n_nan(self) -> int:
        """Number of NaN values.

        Returns
        -------
        int
        """
        return int(self._data.isna().sum())

    @property
    def is_regular(self) -> bool:
        """``True`` when all consecutive time gaps are identical.

        A regular series has no missing timestamps (assuming a fixed
        sampling interval).  An irregular series may be the result of
        market holidays, sensor outages, or event-driven sampling.

        Returns
        -------
        bool
        """
        return self._is_regular

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_series(self) -> pd.Series:
        """Return the data as a :class:`pandas.Series`.

        The returned Series uses the same DatetimeIndex and the
        :attr:`name` attribute as its Series name.

        Returns
        -------
        pandas.Series
        """
        s = self._data.copy()
        s.name = self._name
        return s

    def to_frame(self) -> pd.DataFrame:
        """Return the data as a single-column :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            Column name equals :attr:`name`.
        """
        return self._data.rename(self._name).to_frame()

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the raw values as a 1-D numpy array.

        Returns
        -------
        numpy.ndarray
        """
        return self.values

    def copy(self) -> "TimeSeries":
        """Return a deep copy of this :class:`TimeSeries`.

        Returns
        -------
        TimeSeries
        """
        return TimeSeries(
            self._data.values.copy(),
            index=self._data.index.copy(),
            name=self._name,
            freq=self._freq,
            unit=self._unit,
            description=self._description,
        )

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------

    def slice(
        self,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
    ) -> "TimeSeries":
        """Return a time-bounded subset of the series.

        Both *start* and *end* are **inclusive**.  Either may be ``None``
        to leave that boundary open.

        Parameters
        ----------
        start:
            Start timestamp (inclusive).  Accepts any value parseable by
            :func:`pandas.Timestamp` (e.g., ``"2020-01-01"``).
        end:
            End timestamp (inclusive).

        Returns
        -------
        TimeSeries

        Raises
        ------
        ValueError
            If the resulting slice is empty.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020-01-01", periods=365, freq="D")
        >>> ts = TimeSeries(np.arange(365.0), index=idx)
        >>> q1 = ts.slice("2020-01-01", "2020-03-31")
        >>> q1.n
        91
        """
        sliced = self._data.loc[start:end]
        if sliced.empty:
            raise ValueError(
                f"Slice [{start!r} : {end!r}] produced an empty series. "
                "Check that the bounds fall within the series range "
                f"[{self.start} : {self.end}]."
            )
        return self._copy_with(sliced)

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample(
        self,
        freq: str,
        *,
        agg: Union[str, AggMethod] = AggMethod.MEAN,
    ) -> "TimeSeries":
        """Resample the series to a new frequency.

        Parameters
        ----------
        freq:
            Target pandas offset alias (e.g., ``"W"``, ``"MS"``).
        agg:
            Aggregation method.  Either an :class:`~tseda.core.AggMethod`
            member or its string value.  Default ``"mean"``.

        Returns
        -------
        TimeSeries

        Raises
        ------
        ValueError
            If *freq* is not recognised by pandas.
        AttributeError
            If *agg* is not a valid resampler method.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020-01-01", periods=365, freq="D")
        >>> ts = TimeSeries(np.ones(365), index=idx)
        >>> ts.resample("MS").n    # 12 monthly values
        12
        """
        validate_freq_string(freq, name="freq")
        agg_str = agg.value if isinstance(agg, AggMethod) else str(agg)

        resampler = self._data.resample(freq)
        try:
            resampled: pd.Series = getattr(resampler, agg_str)()
        except AttributeError:
            raise AttributeError(
                f"'{agg_str}' is not a valid resampling aggregation. "
                f"Valid options: {[m.value for m in AggMethod]}."
            )
        resampled = resampled.dropna()
        return self._copy_with(resampled, freq=freq)

    # ------------------------------------------------------------------
    # Transforms — all return new TimeSeries objects
    # ------------------------------------------------------------------

    def diff(
        self,
        periods: int = 1,
        *,
        method: Union[str, DiffMethod] = DiffMethod.SIMPLE,
    ) -> "TimeSeries":
        """Difference the series.

        Parameters
        ----------
        periods:
            Number of periods to lag.  Default 1 (first difference).
        method:
            One of:

            * ``"simple"`` — ``y[t] - y[t-k]``
            * ``"log"``    — ``log(y[t]) - log(y[t-k])``
            * ``"percent"``— ``(y[t] - y[t-k]) / y[t-k]``

        Returns
        -------
        TimeSeries
            The leading NaN rows introduced by differencing are dropped.

        Raises
        ------
        ValueError
            If *method* is ``"log"`` or ``"percent"`` and the series
            contains non-positive values.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020", periods=5, freq="D")
        >>> ts = TimeSeries([10.0, 11.0, 12.0, 11.0, 13.0], index=idx)
        >>> ts.diff().values
        array([1., 1., -1., 2.])
        """
        periods = validate_positive_int(periods, name="periods")
        m = DiffMethod(method) if isinstance(method, str) else method

        vals = self._data.values.copy()

        if m in (DiffMethod.LOG, DiffMethod.PERCENT):
            if np.any(vals <= 0):
                raise ValueError(
                    f"DiffMethod.{m.name} requires strictly positive values; "
                    "the series contains zero or negative observations."
                )

        if m == DiffMethod.SIMPLE:
            result = self._data.diff(periods)
        elif m == DiffMethod.LOG:
            log_series = np.log(self._data)
            result = log_series.diff(periods)
        else:  # PERCENT
            result = self._data.pct_change(periods)

        result = result.dropna()
        suffix = f"_diff{periods}" if m == DiffMethod.SIMPLE else f"_{m.value}{periods}"
        return self._copy_with(result, name=self._name + suffix)

    def log(self) -> "TimeSeries":
        """Apply the natural logarithm element-wise.

        Returns
        -------
        TimeSeries

        Raises
        ------
        ValueError
            If the series contains non-positive values.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020", periods=3, freq="D")
        >>> TimeSeries([1.0, np.e, np.e**2], index=idx).log().values
        array([0., 1., 2.])
        """
        if np.any(self._data.values <= 0):
            raise ValueError(
                "log() requires strictly positive values; "
                "the series contains zero or negative observations."
            )
        result = np.log(self._data)
        return self._copy_with(result, name=f"log({self._name})")

    def standardize(self) -> "TimeSeries":
        """Standardise to zero mean and unit variance (z-score).

        The transform is ``(x - mean) / std``.  NaN values are ignored
        when computing statistics but preserved in position.

        Returns
        -------
        TimeSeries

        Raises
        ------
        ValueError
            If the standard deviation is zero (constant series).

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020", periods=4, freq="D")
        >>> ts = TimeSeries([2.0, 4.0, 6.0, 8.0], index=idx)
        >>> z = ts.standardize()
        >>> round(float(z.values.mean()), 10)
        0.0
        """
        mean = float(np.nanmean(self._data.values))
        std  = float(np.nanstd(self._data.values, ddof=1))
        if std == 0.0:
            raise ValueError(
                "standardize() requires a non-constant series (std == 0)."
            )
        result = (self._data - mean) / std
        return self._copy_with(result, name=f"z({self._name})")

    def normalize(
        self,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> "TimeSeries":
        """Min-max normalise the series to [*lower*, *upper*].

        Parameters
        ----------
        lower:
            Target minimum value.  Default ``0.0``.
        upper:
            Target maximum value.  Default ``1.0``.

        Returns
        -------
        TimeSeries

        Raises
        ------
        ValueError
            If the series has zero range (max == min) or *lower* >= *upper*.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020", periods=3, freq="D")
        >>> ts = TimeSeries([0.0, 5.0, 10.0], index=idx)
        >>> ts.normalize().values
        array([0. , 0.5, 1. ])
        """
        if lower >= upper:
            raise ValueError(
                f"'lower' ({lower}) must be less than 'upper' ({upper})."
            )
        mn = float(np.nanmin(self._data.values))
        mx = float(np.nanmax(self._data.values))
        if mx == mn:
            raise ValueError(
                "normalize() requires a non-constant series (max == min)."
            )
        result = lower + (self._data - mn) / (mx - mn) * (upper - lower)
        return self._copy_with(result, name=f"norm({self._name})")

    def rolling(
        self,
        window: int,
        *,
        agg: Union[str, AggMethod] = AggMethod.MEAN,
        center: bool = False,
        min_periods: Optional[int] = None,
    ) -> "TimeSeries":
        """Apply a rolling-window aggregation.

        Parameters
        ----------
        window:
            Size of the rolling window in number of observations.
        agg:
            Aggregation method (default ``"mean"``).
        center:
            Whether to set the window labels as the centre of the window
            (default ``False`` — trailing window).
        min_periods:
            Minimum number of non-NaN observations required to produce a
            value.  Defaults to *window*.

        Returns
        -------
        TimeSeries
            Leading/trailing NaNs introduced by the window are dropped.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020", periods=6, freq="D")
        >>> ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        >>> ts.rolling(3).values
        array([2., 3., 4., 5.])
        """
        validate_positive_int(window, name="window")
        agg_str = agg.value if isinstance(agg, AggMethod) else str(agg)

        roller = self._data.rolling(
            window=window, center=center, min_periods=min_periods
        )
        try:
            result: pd.Series = getattr(roller, agg_str)()
        except AttributeError:
            raise AttributeError(
                f"'{agg_str}' is not a valid rolling aggregation. "
                f"Valid options: {[m.value for m in AggMethod]}."
            )
        result = result.dropna()
        return self._copy_with(
            result,
            name=f"rolling_{window}_{agg_str}({self._name})",
        )

    def apply(self, func: Callable[[np.ndarray], np.ndarray], *, name: Optional[str] = None) -> "TimeSeries":
        """Apply an arbitrary element-wise function to the values.

        Parameters
        ----------
        func:
            Callable that takes a 1-D ``numpy.ndarray`` and returns a 1-D
            array of the same length.
        name:
            Name for the resulting series.  Defaults to
            ``"f({self.name})"``.

        Returns
        -------
        TimeSeries

        Raises
        ------
        ValueError
            If *func* changes the length of the array.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020", periods=3, freq="D")
        >>> ts = TimeSeries([1.0, 4.0, 9.0], index=idx)
        >>> ts.apply(np.sqrt).values
        array([1., 2., 3.])
        """
        result_vals = func(self._data.values.copy())
        if len(result_vals) != len(self._data):
            raise ValueError(
                f"'func' must return an array of the same length as the input "
                f"({len(self._data)}); got {len(result_vals)}."
            )
        result = pd.Series(result_vals, index=self._data.index, dtype=float)
        _name = name if name is not None else f"f({self._name})"
        return self._copy_with(result, name=_name)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, timestamp: object) -> bool:
        """Check whether a timestamp exists in the index.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020", periods=3, freq="D")
        >>> ts = TimeSeries([1.0, 2.0, 3.0], index=idx)
        >>> pd.Timestamp("2020-01-02") in ts
        True
        """
        try:
            ts = pd.Timestamp(timestamp)  # type: ignore[arg-type]
        except Exception:
            return False
        return ts in self._data.index

    def __getitem__(self, key: Union[int, slice]) -> Union[float, "TimeSeries"]:
        """Positional indexing by integer or slice.

        Parameters
        ----------
        key:
            * ``int`` — return the scalar value at that position.
            * ``slice`` — return a new :class:`TimeSeries` for that range.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020", periods=5, freq="D")
        >>> ts = TimeSeries([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)
        >>> ts[0]
        10.0
        >>> ts[-1]
        50.0
        >>> ts[1:3].values
        array([20., 30.])
        """
        if isinstance(key, int):
            return float(self._data.iloc[key])
        if isinstance(key, slice):
            sliced = self._data.iloc[key]
            return self._copy_with(sliced)
        raise TypeError(
            f"Indices must be integers or slices, not {type(key).__name__!r}. "
            "For datetime-based slicing use ts.slice()."
        )

    def __eq__(self, other: object) -> bool:
        """Two :class:`TimeSeries` objects are equal when values and index match."""
        if not isinstance(other, TimeSeries):
            return NotImplemented
        return (
            self._name == other._name
            and self._data.index.equals(other._data.index)
            and np.array_equal(self._data.values, other._data.values, equal_nan=True)
        )

    def __repr__(self) -> str:
        unit_line = f"\n  unit        : {self._unit}" if self._unit else ""
        desc_line = (
            f"\n  description : {self._description[:60]}{'...' if len(self._description) > 60 else ''}"
            if self._description
            else ""
        )
        nan_pct = 100.0 * self.n_nan / max(self.n, 1)
        return (
            f"TimeSeries(\n"
            f"  name        : {self._name}\n"
            f"  n           : {self.n:,}\n"
            f"  start       : {self.start}\n"
            f"  end         : {self.end}\n"
            f"  duration    : {self.duration}\n"
            f"  freq        : {self._freq or 'unknown'} ({self.freq_label})\n"
            f"  is_regular  : {self.is_regular}\n"
            f"  has_nan     : {self.has_nan} ({self.n_nan} / {nan_pct:.1f}%)"
            f"{unit_line}{desc_line}\n"
            f")"
        )