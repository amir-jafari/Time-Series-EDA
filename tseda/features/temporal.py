"""
Temporal feature extraction for time series.

Extracts calendar-based and time-index features from a
:class:`~tseda.core.TimeSeries`.  All features are deterministic functions
of the datetime index — no statistical estimation required.

Two categories are produced:

* **Calendar features** — year, month, day, hour, day-of-week, quarter,
  and boolean flags (is_weekend, is_month_start, is_month_end).
* **Cyclic encodings** — sine/cosine projections of periodic calendar
  fields (month, day-of-week, hour) so that ``month 12`` and ``month 1``
  are close in feature space.

Classes
-------
TemporalFeatureExtractor
    Stateless extractor returning a :class:`pandas.DataFrame`.

Examples
--------
>>> import pandas as pd, numpy as np
>>> from tseda import TimeSeries
>>> from tseda.features.temporal import TemporalFeatureExtractor

>>> idx = pd.date_range("2020-01-01", periods=10, freq="D")
>>> ts  = TimeSeries(np.arange(10.0), index=idx)
>>> df  = TemporalFeatureExtractor().extract(ts)
>>> list(df.columns[:4])
['year', 'month', 'day', 'dayofweek']
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tseda.core.timeseries import TimeSeries

__all__ = ["TemporalFeatureExtractor"]


class TemporalFeatureExtractor:
    """Extract calendar and cyclic time features from a
    :class:`~tseda.core.TimeSeries`.

    Methods
    -------
    extract(ts, cyclic, time_index)
        Return a :class:`pandas.DataFrame` with one feature column per row
        aligned to ``ts.index``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from tseda import TimeSeries
    >>> from tseda.features.temporal import TemporalFeatureExtractor

    >>> idx = pd.date_range("2020-01-01", periods=5, freq="D")
    >>> ts  = TimeSeries([10.0, 11.0, 12.0, 11.5, 10.5], index=idx)
    >>> df  = TemporalFeatureExtractor().extract(ts)
    >>> int(df["year"].iloc[0])
    2020
    >>> int(df["month"].iloc[0])
    1
    """

    def extract(
        self,
        ts: TimeSeries,
        *,
        cyclic: bool = True,
        time_index: bool = True,
    ) -> pd.DataFrame:
        """Extract temporal features aligned to ``ts.index``.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        cyclic : bool, optional
            When ``True`` (default), add sine/cosine encodings for
            ``month``, ``dayofweek``, and ``hour``.
        time_index : bool, optional
            When ``True`` (default), add ``days_since_start`` and
            ``time_norm`` (0 → 1 over the series span).

        Returns
        -------
        pandas.DataFrame
            Index matches ``ts.index``.  Columns:

            Always present:
              ``year``, ``month``, ``day``, ``dayofweek``, ``hour``,
              ``quarter``, ``weekofyear``, ``is_weekend``,
              ``is_month_start``, ``is_month_end``.

            When ``cyclic=True``:
              ``month_sin``, ``month_cos``,
              ``dow_sin``, ``dow_cos``,
              ``hour_sin``, ``hour_cos``.

            When ``time_index=True``:
              ``days_since_start``, ``time_norm``.

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> from tseda import TimeSeries
        >>> from tseda.features.temporal import TemporalFeatureExtractor
        >>> idx = pd.date_range("2020-01-01", periods=7, freq="D")
        >>> ts  = TimeSeries(np.ones(7), index=idx)
        >>> df  = TemporalFeatureExtractor().extract(ts, cyclic=False, time_index=False)
        >>> set(df.columns) >= {"year", "month", "day", "dayofweek", "is_weekend"}
        True
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )

        idx = ts.index

        features: dict[str, np.ndarray] = {}

        # ── Calendar ──────────────────────────────────────────────────
        features["year"]          = idx.year.to_numpy()
        features["month"]         = idx.month.to_numpy()
        features["day"]           = idx.day.to_numpy()
        features["dayofweek"]     = idx.dayofweek.to_numpy()       # 0=Mon, 6=Sun
        features["hour"]          = idx.hour.to_numpy()
        features["quarter"]       = idx.quarter.to_numpy()
        features["weekofyear"]    = idx.isocalendar().week.to_numpy(dtype=int)
        features["is_weekend"]    = (idx.dayofweek >= 5).astype(float)
        features["is_month_start"] = idx.is_month_start.astype(float)
        features["is_month_end"]   = idx.is_month_end.astype(float)

        # ── Cyclic encodings ──────────────────────────────────────────
        if cyclic:
            # Month (1–12)
            m = features["month"]
            features["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
            features["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)
            # Day-of-week (0–6)
            d = features["dayofweek"]
            features["dow_sin"] = np.sin(2 * np.pi * d / 7)
            features["dow_cos"] = np.cos(2 * np.pi * d / 7)
            # Hour (0–23)
            h = features["hour"]
            features["hour_sin"] = np.sin(2 * np.pi * h / 24)
            features["hour_cos"] = np.cos(2 * np.pi * h / 24)

        # ── Time index ────────────────────────────────────────────────
        if time_index:
            days = (idx - idx[0]).total_seconds() / 86_400.0
            features["days_since_start"] = days
            span = float(days[-1]) if len(days) > 1 else 1.0
            features["time_norm"] = days / span

        return pd.DataFrame(features, index=idx)