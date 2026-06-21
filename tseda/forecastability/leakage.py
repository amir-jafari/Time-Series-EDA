"""
Leakage detection for time series feature sets.

Two classes of leakage are detected:

+--------------------+-------------------------------------------------------+
| Type               | Definition                                            |
+====================+=======================================================+
| Temporal leakage   | A feature at time *t* correlates more strongly with  |
|                    | future target values (t+1 … t+horizon) than with     |
|                    | past or present target values.                       |
+--------------------+-------------------------------------------------------+
| Target leakage     | A feature is so highly correlated with the target at |
|                    | lag 0 that it almost certainly encodes the target    |
|                    | itself (e.g., a lagged copy or a near-identity       |
|                    | transform).                                          |
+--------------------+-------------------------------------------------------+

When *features_df* is ``None`` the report is returned with empty leakage sets
and a warning that no features were provided.

Classes
-------
LeakageReport
    Frozen dataclass returned by :meth:`LeakageDetector.check`.
LeakageDetector
    Stateless detector.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.forecastability.leakage import LeakageDetector

No leakage — lagged features only:

>>> rng  = np.random.default_rng(0)
>>> n    = 100
>>> idx  = pd.date_range("2020", periods=n, freq="D")
>>> y    = rng.standard_normal(n)
>>> ts   = TimeSeries(y, index=idx)
>>> feat = pd.DataFrame({"lag1": np.roll(y, 1), "lag2": np.roll(y, 2)}, index=idx)
>>> feat.iloc[:2] = np.nan
>>> r    = LeakageDetector().check(ts, horizon=5, features_df=feat)
>>> r.has_target_leakage
False
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tseda.core.timeseries import TimeSeries

__all__ = ["LeakageReport", "LeakageDetector"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakageReport:
    """Immutable leakage detection result.

    Attributes
    ----------
    has_temporal_leakage : bool
        ``True`` if any feature shows stronger correlation with future target
        than with current / past target.
    has_target_leakage : bool
        ``True`` if any feature is correlated with the target at lag 0 above
        *target_corr_threshold*.
    temporal_leakage_columns : list of str
        Names of feature columns flagged for temporal leakage.
    target_leakage_columns : list of str
        Names of feature columns flagged for target leakage.
    target_leakage_correlations : dict of str → float
        Lag-0 Pearson correlation for each column in
        :attr:`target_leakage_columns`.
    temporal_peak_lags : dict of str → int
        For each feature column, the lag at which the cross-correlation with
        the target is maximised.  Positive lag means feature correlates with
        *future* target.
    horizon : int
        Forecast horizon passed to :meth:`~LeakageDetector.check`.
    n_features : int
        Number of feature columns examined.
    n_obs : int
        Number of observations in the target series.
    warnings : list of str
        Human-readable diagnostic messages.
    """

    has_temporal_leakage: bool
    has_target_leakage: bool
    temporal_leakage_columns: List[str]
    target_leakage_columns: List[str]
    target_leakage_correlations: Dict[str, float]
    temporal_peak_lags: Dict[str, int]
    horizon: int
    n_features: int
    n_obs: int
    warnings: List[str]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LeakageReport(\n"
            f"  has_temporal_leakage : {self.has_temporal_leakage}\n"
            f"  has_target_leakage   : {self.has_target_leakage}\n"
            f"  temporal_columns     : {self.temporal_leakage_columns}\n"
            f"  target_columns       : {self.target_leakage_columns}\n"
            f"  n_features           : {self.n_features}\n"
            f"  horizon              : {self.horizon}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _cross_corr_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """Pearson correlation of x[:-lag] vs y[lag:] (positive lag = y is ahead).

    Returns 0.0 if either slice has zero variance or is too short.
    """
    if lag == 0:
        a, b = x, y
    elif lag > 0:
        a, b = x[:-lag], y[lag:]
    else:
        k = -lag
        a, b = x[k:], y[:-k]

    if len(a) < 3:
        return 0.0
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 3:
        return 0.0
    std_a, std_b = float(np.std(a)), float(np.std(b))
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class LeakageDetector:
    """Detect temporal and target leakage in a feature set.

    The detector is **stateless**.

    Methods
    -------
    check(ts, horizon, features_df, target_corr_threshold)
        Return a :class:`LeakageReport`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.forecastability.leakage import LeakageDetector

    Target leakage — a feature that *is* the target:

    >>> rng = np.random.default_rng(0)
    >>> n   = 80
    >>> idx = pd.date_range("2020", periods=n, freq="D")
    >>> y   = rng.standard_normal(n)
    >>> ts  = TimeSeries(y, index=idx)
    >>> feat = pd.DataFrame({"target_copy": y}, index=idx)
    >>> r = LeakageDetector().check(ts, horizon=1, features_df=feat)
    >>> r.has_target_leakage
    True
    >>> "target_copy" in r.target_leakage_columns
    True
    """

    def check(
        self,
        ts: TimeSeries,
        horizon: int,
        *,
        features_df: Optional[pd.DataFrame] = None,
        target_corr_threshold: float = 0.95,
    ) -> LeakageReport:
        """Check *features_df* for leakage against target *ts*.

        Parameters
        ----------
        ts : TimeSeries
            Target time series.
        horizon : int
            Forecast horizon in time steps.  Must be >= 1.
        features_df : pandas.DataFrame, optional
            Feature matrix with the same :class:`~pandas.DatetimeIndex` as
            *ts*, one column per feature.  When ``None`` the report is empty
            with a warning.
        target_corr_threshold : float, optional
            |Pearson r| threshold above which a feature is flagged as
            target-leaking.  Default ``0.95``.

        Returns
        -------
        LeakageReport

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *horizon* < 1, *target_corr_threshold* ∉ (0, 1], or
            *features_df* has a different number of rows from *ts*.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.forecastability.leakage import LeakageDetector

        >>> rng = np.random.default_rng(1)
        >>> n   = 60
        >>> idx = pd.date_range("2020", periods=n, freq="D")
        >>> ts  = TimeSeries(rng.standard_normal(n), index=idx)
        >>> r   = LeakageDetector().check(ts, horizon=3)
        >>> r.n_features
        0
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError(
                f"'horizon' must be a positive integer; got {horizon!r}."
            )
        if not (0 < target_corr_threshold <= 1.0):
            raise ValueError(
                f"'target_corr_threshold' must be in (0, 1]; "
                f"got {target_corr_threshold!r}."
            )

        warn_msgs: List[str] = []

        if features_df is None:
            warn_msgs.append(
                "No features_df provided — leakage check skipped. "
                "Pass a DataFrame of feature columns to enable full analysis."
            )
            return LeakageReport(
                has_temporal_leakage=False,
                has_target_leakage=False,
                temporal_leakage_columns=[],
                target_leakage_columns=[],
                target_leakage_correlations={},
                temporal_peak_lags={},
                horizon=horizon,
                n_features=0,
                n_obs=ts.n,
                warnings=warn_msgs,
            )

        if not isinstance(features_df, pd.DataFrame):
            raise TypeError(
                f"'features_df' must be a pandas.DataFrame, "
                f"got {type(features_df).__name__!r}."
            )
        if len(features_df) != ts.n:
            raise ValueError(
                f"'features_df' must have the same number of rows as 'ts' "
                f"({ts.n}); got {len(features_df)} rows."
            )

        y = ts.values.copy()
        n_features = len(features_df.columns)

        temporal_leakage_cols: List[str] = []
        target_leakage_cols: List[str] = []
        target_leakage_corrs: Dict[str, float] = {}
        temporal_peak_lags: Dict[str, int] = {}

        for col in features_df.columns:
            f = features_df[col].to_numpy(dtype=float, na_value=np.nan)

            # ── Target leakage: |corr at lag 0| > threshold ──────────
            r0 = _cross_corr_at_lag(f, y, lag=0)
            if abs(r0) >= target_corr_threshold:
                target_leakage_cols.append(str(col))
                target_leakage_corrs[str(col)] = round(r0, 6)

            # ── Temporal leakage: peak cross-correlation at positive lag ──
            max_shift = min(horizon, ts.n // 4)
            if max_shift < 1:
                temporal_peak_lags[str(col)] = 0
                continue

            lags = range(-max_shift, max_shift + 1)
            corr_by_lag = {k: _cross_corr_at_lag(f, y, lag=k) for k in lags}

            peak_lag = max(corr_by_lag, key=lambda k: abs(corr_by_lag[k]))
            temporal_peak_lags[str(col)] = int(peak_lag)

            if peak_lag > 0:
                corr_future = abs(corr_by_lag[peak_lag])
                corr_present = abs(corr_by_lag[0])
                if corr_future > corr_present + 0.05:
                    temporal_leakage_cols.append(str(col))

        return LeakageReport(
            has_temporal_leakage=len(temporal_leakage_cols) > 0,
            has_target_leakage=len(target_leakage_cols) > 0,
            temporal_leakage_columns=temporal_leakage_cols,
            target_leakage_columns=target_leakage_cols,
            target_leakage_correlations=target_leakage_corrs,
            temporal_peak_lags=temporal_peak_lags,
            horizon=horizon,
            n_features=n_features,
            n_obs=ts.n,
            warnings=warn_msgs,
        )