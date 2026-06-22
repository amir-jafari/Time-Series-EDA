"""
Anomaly detection for time series.

Detects two classes of anomaly:

* **Point anomalies** — isolated observations that deviate sharply from
  the surrounding distribution (spikes, dips).
* **Contextual anomalies** — observations that are surprising given their
  local neighbourhood (e.g., a value that is normal globally but anomalous
  within a specific season).

Four detection methods are provided, all in pure numpy / scipy:

+------------------+------------------------------+---------------------------------+
| Method           | Mechanism                    | Best for                        |
+==================+==============================+=================================+
| ``rolling_iqr``  | Rolling IQR fence            | Non-stationary level data       |
+------------------+------------------------------+---------------------------------+
| ``rolling_z``    | Rolling mean ± k × std       | Approximately normal windows    |
+------------------+------------------------------+---------------------------------+
| ``stl_residual`` | STL decompose → flag residual| Seasonal + trend data           |
+------------------+------------------------------+---------------------------------+
| ``gesd``         | Global GESD test             | Stationary data, known # spikes |
+------------------+------------------------------+---------------------------------+

All methods return an :class:`AnomalyReport` and share the repair helpers
:meth:`AnomalyDetector.remove` (replace with NaN) and
:meth:`AnomalyDetector.label` (0/1 indicator series).

Classes
-------
AnomalyReport
    Frozen dataclass returned by every detection method.
AnomalyDetector
    Stateless detector.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.anomaly.detector import AnomalyDetector

Plant a spike and recover it:

>>> rng  = np.random.default_rng(0)
>>> idx  = pd.date_range("2020-01-01", periods=200, freq="D")
>>> vals = rng.standard_normal(200)
>>> vals[50] = 15.0   # spike
>>> ts   = TimeSeries(vals, index=idx)
>>> det  = AnomalyDetector()
>>> r    = det.rolling_iqr(ts)
>>> 50 in r.indices
True
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from tseda.core.timeseries import TimeSeries
from tseda.core.validator import validate_positive_int

__all__ = ["AnomalyReport", "AnomalyDetector"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalyReport:
    """Immutable anomaly detection result.

    Attributes
    ----------
    mask : numpy.ndarray
        Boolean array of shape ``(n,)``; ``True`` where an anomaly was detected.
    indices : numpy.ndarray
        Integer positions (0-based) of detected anomalies.
    timestamps : pandas.DatetimeIndex
        Timestamps of detected anomalies.
    values : numpy.ndarray
        Observed values at anomaly positions.
    scores : numpy.ndarray
        Continuous anomaly score in [0, 1] for each observation.
        Higher values indicate stronger evidence of anomaly.
        ``0`` for normal observations.
    method : str
        Name of the detection method.
    n_anomalies : int
        Number of anomalies detected.
    """

    mask: np.ndarray
    indices: np.ndarray
    timestamps: pd.DatetimeIndex
    values: np.ndarray
    scores: np.ndarray
    method: str
    n_anomalies: int

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AnomalyReport(\n"
            f"  method      : {self.method}\n"
            f"  n_anomalies : {self.n_anomalies}\n"
            f"  top scores  : {np.sort(self.scores)[::-1][:5].round(4).tolist()}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_report(
    ts: TimeSeries, mask: np.ndarray, scores: np.ndarray, method: str
) -> AnomalyReport:
    idx = np.where(mask)[0]
    return AnomalyReport(
        mask=mask,
        indices=idx,
        timestamps=ts.index[idx],
        values=ts.values[idx],
        scores=scores,
        method=method,
        n_anomalies=int(mask.sum()),
    )


def _safe_iqr(arr: np.ndarray) -> tuple[float, float, float, float]:
    """Return (q1, q3, iqr, median) ignoring NaN."""
    finite = arr[~np.isnan(arr)]
    if len(finite) < 4:
        return np.nan, np.nan, np.nan, np.nan
    q1  = float(np.percentile(finite, 25))
    q3  = float(np.percentile(finite, 75))
    iqr = q3 - q1
    med = float(np.median(finite))
    return q1, q3, iqr, med


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """Detect point and contextual anomalies in a
    :class:`~tseda.core.TimeSeries`.

    This class is **stateless** — one instance, many series.

    Methods
    -------
    rolling_iqr(ts, window, k)
        Rolling Tukey IQR fence.
    rolling_z(ts, window, threshold)
        Rolling mean ± k × std.
    stl_residual(ts, period, method, threshold)
        STL decompose then flag large residuals.
    gesd(ts, alpha, max_outliers)
        Global Generalized ESD test (re-uses quality module).
    remove(ts, report)
        Replace anomaly positions with NaN.
    label(ts, report)
        Return a 0/1 :class:`~tseda.core.TimeSeries` of anomaly labels.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.anomaly.detector import AnomalyDetector

    >>> rng  = np.random.default_rng(1)
    >>> idx  = pd.date_range("2020", periods=100, freq="D")
    >>> vals = rng.standard_normal(100)
    >>> vals[20] = 12.0
    >>> ts   = TimeSeries(vals, index=idx)
    >>> det  = AnomalyDetector()
    >>> r    = det.rolling_z(ts, window=30)
    >>> 20 in r.indices
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
        if len(x) < 8:
            raise ValueError(
                "Anomaly detection requires at least 8 non-NaN observations."
            )
        return ts

    # ------------------------------------------------------------------
    # Rolling IQR
    # ------------------------------------------------------------------

    def rolling_iqr(
        self,
        ts: TimeSeries,
        window: int = 30,
        *,
        k: float = 2.5,
        center: bool = True,
        min_periods: Optional[int] = None,
    ) -> AnomalyReport:
        """Detect anomalies using a rolling IQR fence.

        An observation ``y[t]`` is flagged when it falls outside
        ``[Q1(t) − k×IQR(t),  Q3(t) + k×IQR(t)]`` where the quartiles
        are computed over a rolling window centred at ``t``.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        window : int, optional
            Rolling window width in observations.  Default 30.
        k : float, optional
            Fence multiplier.  Default 2.5 (tighter than the global 1.5
            because the window is local and thus more sensitive).
        center : bool, optional
            Centre the window on each observation.  Default ``True``.
        min_periods : int, optional
            Minimum non-NaN observations required in each window.
            Defaults to ``window // 2``.

        Returns
        -------
        AnomalyReport

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If fewer than 8 non-NaN observations or *k* ≤ 0.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.anomaly.detector import AnomalyDetector
        >>> rng  = np.random.default_rng(0)
        >>> idx  = pd.date_range("2020", periods=100, freq="D")
        >>> vals = rng.standard_normal(100)
        >>> vals[40] = 10.0
        >>> ts   = TimeSeries(vals, index=idx)
        >>> r    = AnomalyDetector().rolling_iqr(ts)
        >>> 40 in r.indices
        True
        """
        if k <= 0:
            raise ValueError(f"'k' must be positive, got {k}.")
        ts = self._validate(ts)
        validate_positive_int(window, name="window")
        mp = min_periods if min_periods is not None else window // 2

        s = ts.to_series()
        q1 = s.rolling(window, center=center, min_periods=mp).quantile(0.25)
        q3 = s.rolling(window, center=center, min_periods=mp).quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr

        vals   = ts.values
        lo_arr = lower.to_numpy()
        hi_arr = upper.to_numpy()
        iqr_arr = hi_arr - lo_arr

        valid  = ~(np.isnan(vals) | np.isnan(lo_arr) | np.isnan(hi_arr))
        below  = valid & (vals < lo_arr)
        above  = valid & (vals > hi_arr)
        mask   = below | above

        scores = np.zeros(ts.n, dtype=float)
        scores[below] = np.minimum(1.0, (lo_arr[below] - vals[below]) / (iqr_arr[below] + 1e-10))
        scores[above] = np.minimum(1.0, (vals[above]   - hi_arr[above]) / (iqr_arr[above] + 1e-10))

        return _build_report(ts, mask, scores, f"rolling_iqr(w={window},k={k})")

    # ------------------------------------------------------------------
    # Rolling Z-score
    # ------------------------------------------------------------------

    def rolling_z(
        self,
        ts: TimeSeries,
        window: int = 30,
        *,
        threshold: float = 3.0,
        center: bool = True,
        min_periods: Optional[int] = None,
    ) -> AnomalyReport:
        """Detect anomalies using a rolling Z-score.

        An observation is flagged when
        ``|y[t] − μ(t)| / σ(t) > threshold``, where ``μ`` and ``σ`` are
        the rolling mean and standard deviation computed over a window.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        window : int, optional
            Rolling window width.  Default 30.
        threshold : float, optional
            Z-score cut-off.  Default 3.0.
        center : bool, optional
            Centre the window.  Default ``True``.
        min_periods : int, optional
            Minimum non-NaN observations per window.  Defaults to
            ``window // 2``.

        Returns
        -------
        AnomalyReport

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.anomaly.detector import AnomalyDetector
        >>> rng  = np.random.default_rng(2)
        >>> idx  = pd.date_range("2020", periods=100, freq="D")
        >>> vals = rng.standard_normal(100)
        >>> vals[60] = -9.0
        >>> ts   = TimeSeries(vals, index=idx)
        >>> r    = AnomalyDetector().rolling_z(ts)
        >>> 60 in r.indices
        True
        """
        if threshold <= 0:
            raise ValueError(f"'threshold' must be positive, got {threshold}.")
        ts = self._validate(ts)
        validate_positive_int(window, name="window")
        mp = min_periods if min_periods is not None else window // 2

        s    = ts.to_series()
        mu   = s.rolling(window, center=center, min_periods=mp).mean()
        sigma = s.rolling(window, center=center, min_periods=mp).std(ddof=1)

        vals   = ts.values
        mu_arr = mu.to_numpy()
        sg_arr = sigma.to_numpy()

        valid  = ~(np.isnan(vals) | np.isnan(mu_arr) | np.isnan(sg_arr) | (sg_arr == 0))
        z      = np.where(valid, np.abs(vals - mu_arr) / (sg_arr + 1e-300), 0.0)
        mask   = valid & (z > threshold)
        scores = np.where(mask, np.minimum(1.0, (z - threshold) / (threshold + 1e-10)), 0.0)

        return _build_report(ts, mask, scores, f"rolling_z(w={window},t={threshold})")

    # ------------------------------------------------------------------
    # STL residual
    # ------------------------------------------------------------------

    def stl_residual(
        self,
        ts: TimeSeries,
        period: Optional[int] = None,
        *,
        residual_method: str = "iqr",
        k: float = 3.0,
        robust: bool = True,
    ) -> AnomalyReport:
        """Detect anomalies in the STL residual component.

        Decomposes *ts* into trend + seasonal + residual using STL, then
        flags residual values that are extreme under the chosen criterion.

        Parameters
        ----------
        ts : TimeSeries
            Input series.  Must be long enough for STL decomposition
            (at least 2 × *period* observations).
        period : int, optional
            Seasonal period.  Inferred from ``ts.freq`` when omitted.
        residual_method : str, optional
            Criterion to flag residuals:

            * ``"iqr"`` — Tukey IQR fence with multiplier *k* (default).
            * ``"mad"`` — Median Absolute Deviation threshold *k*.
            * ``"z"``   — Z-score threshold *k*.

        k : float, optional
            Threshold multiplier / fence factor.  Default 3.0.
        robust : bool, optional
            Use robust LOESS in STL.  Default ``True``.

        Returns
        -------
        AnomalyReport

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If *residual_method* is not recognised.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.anomaly.detector import AnomalyDetector
        >>> rng  = np.random.default_rng(3)
        >>> n    = 60
        >>> seas = np.tile(np.sin(2*np.pi*np.arange(12)/12)*5, 5)
        >>> vals = seas + rng.standard_normal(n) * 0.3
        >>> vals[25] = 20.0
        >>> idx  = pd.date_range("2018-01", periods=n, freq="MS")
        >>> ts   = TimeSeries(vals, index=idx)
        >>> r    = AnomalyDetector().stl_residual(ts, period=12)
        >>> 25 in r.indices
        True
        """
        if residual_method not in ("iqr", "mad", "z"):
            raise ValueError(
                f"'residual_method' must be 'iqr', 'mad', or 'z'; "
                f"got {residual_method!r}."
            )
        if k <= 0:
            raise ValueError(f"'k' must be positive, got {k}.")
        ts = self._validate(ts)

        # Decompose
        from tseda.decomposition.stl import STLDecomposer
        dec = STLDecomposer().decompose(ts, period=period, robust=robust)
        R   = dec.residual.values

        # Flag residuals
        finite = R[~np.isnan(R)]
        mask   = np.zeros(ts.n, dtype=bool)
        scores = np.zeros(ts.n, dtype=float)

        if residual_method == "iqr":
            q1, q3 = float(np.percentile(finite, 25)), float(np.percentile(finite, 75))
            iqr_val = q3 - q1
            lo, hi  = q1 - k * iqr_val, q3 + k * iqr_val
            for i, r in enumerate(R):
                if np.isnan(r):
                    continue
                if r < lo:
                    mask[i]   = True
                    scores[i] = min(1.0, (lo - r) / (iqr_val + 1e-10))
                elif r > hi:
                    mask[i]   = True
                    scores[i] = min(1.0, (r - hi) / (iqr_val + 1e-10))

        elif residual_method == "mad":
            med     = float(np.median(finite))
            mad_val = float(np.median(np.abs(finite - med)))
            if mad_val == 0:
                mad_val = 1e-10
            for i, r in enumerate(R):
                if np.isnan(r):
                    continue
                mod_z = 0.6745 * abs(r - med) / mad_val
                if mod_z > k:
                    mask[i]   = True
                    scores[i] = min(1.0, (mod_z - k) / (k + 1e-10))

        else:  # z
            mean = float(np.mean(finite))
            std  = float(np.std(finite, ddof=1))
            if std == 0:
                std = 1e-10
            for i, r in enumerate(R):
                if np.isnan(r):
                    continue
                z = abs(r - mean) / std
                if z > k:
                    mask[i]   = True
                    scores[i] = min(1.0, (z - k) / (k + 1e-10))

        return _build_report(
            ts, mask, scores,
            f"stl_residual({residual_method},k={k},p={dec.period})",
        )

    # ------------------------------------------------------------------
    # GESD (re-uses quality module)
    # ------------------------------------------------------------------

    def gesd(
        self,
        ts: TimeSeries,
        *,
        alpha: float = 0.05,
        max_outliers: int = 10,
    ) -> AnomalyReport:
        """Global GESD anomaly detection.

        Delegates to :meth:`tseda.quality.OutlierDetector.gesd` and wraps
        the result as an :class:`AnomalyReport`.  Best suited for stationary
        series where the number of anomalies is small relative to *n*.

        Parameters
        ----------
        ts : TimeSeries
            Input series.
        alpha : float, optional
            Significance level.  Default 0.05.
        max_outliers : int, optional
            Maximum number of anomalies to test for.  Default 10.

        Returns
        -------
        AnomalyReport

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.anomaly.detector import AnomalyDetector
        >>> rng  = np.random.default_rng(0)
        >>> idx  = pd.date_range("2020", periods=50, freq="D")
        >>> vals = rng.standard_normal(50)
        >>> vals[10] = 12.0
        >>> ts   = TimeSeries(vals, index=idx)
        >>> r    = AnomalyDetector().gesd(ts)
        >>> 10 in r.indices
        True
        """
        ts = self._validate(ts)
        from tseda.quality.outliers import OutlierDetector
        n_finite = int(np.sum(~np.isnan(ts.values)))
        safe_max = min(max_outliers, n_finite // 2 - 1)
        if safe_max < 1:
            return _build_report(ts, np.zeros(ts.n, dtype=bool),
                                 np.zeros(ts.n), f"gesd(alpha={alpha})")
        od_report = OutlierDetector().gesd(ts, alpha=alpha, max_outliers=safe_max)

        # Build continuous scores: |z-score| normalised to [0,1]
        vals   = ts.values.copy()
        finite = vals[~np.isnan(vals)]
        mean_v = float(np.mean(finite))
        std_v  = float(np.std(finite, ddof=1)) if len(finite) > 1 else 1.0
        z_abs  = np.where(
            ~np.isnan(vals),
            np.abs((vals - mean_v) / (std_v + 1e-10)),
            0.0,
        )
        # Normalise scores to [0, 1] relative to max z
        max_z  = z_abs.max()
        scores = (z_abs / max_z) if max_z > 0 else z_abs
        # Zero out non-anomaly scores
        scores[~od_report.mask] = 0.0

        return _build_report(
            ts, od_report.mask, scores,
            f"gesd(alpha={alpha})",
        )

    # ------------------------------------------------------------------
    # Repair helpers
    # ------------------------------------------------------------------

    def remove(self, ts: TimeSeries, report: AnomalyReport) -> TimeSeries:
        """Replace detected anomaly values with NaN.

        Parameters
        ----------
        ts : TimeSeries
            The original series.
        report : AnomalyReport
            Output of any detection method.

        Returns
        -------
        TimeSeries
            A new series with anomaly positions set to NaN.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.anomaly.detector import AnomalyDetector
        >>> idx  = pd.date_range("2020", periods=50, freq="D")
        >>> vals = np.zeros(50)
        >>> vals[5] = 100.0
        >>> ts   = TimeSeries(vals, index=idx)
        >>> det  = AnomalyDetector()
        >>> cleaned = det.remove(ts, det.rolling_iqr(ts))
        >>> cleaned.has_nan
        True
        """
        if not isinstance(report, AnomalyReport):
            raise TypeError(
                f"'report' must be an AnomalyReport, got {type(report).__name__!r}."
            )
        vals = ts.values
        vals[report.mask] = np.nan
        return TimeSeries(
            vals, index=ts.index, name=ts.name,
            freq=ts.freq, unit=ts.unit, description=ts.description,
        )

    def label(self, ts: TimeSeries, report: AnomalyReport) -> TimeSeries:
        """Return a 0/1 :class:`~tseda.core.TimeSeries` of anomaly labels.

        Parameters
        ----------
        ts : TimeSeries
            The original series (used for index and metadata).
        report : AnomalyReport
            Output of any detection method.

        Returns
        -------
        TimeSeries
            Values are ``1`` at anomaly positions, ``0`` elsewhere.
            Name is ``"{ts.name}_anomaly_label"``.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.anomaly.detector import AnomalyDetector
        >>> idx  = pd.date_range("2020", periods=50, freq="D")
        >>> vals = np.zeros(50); vals[5] = 100.0
        >>> ts   = TimeSeries(vals, index=idx)
        >>> det  = AnomalyDetector()
        >>> lbl  = det.label(ts, det.rolling_iqr(ts))
        >>> lbl.values[5]
        1.0
        >>> lbl.values[0]
        0.0
        """
        if not isinstance(report, AnomalyReport):
            raise TypeError(
                f"'report' must be an AnomalyReport, got {type(report).__name__!r}."
            )
        labels = report.mask.astype(float)
        return TimeSeries(
            labels, index=ts.index,
            name=f"{ts.name}_anomaly_label",
            freq=ts.freq,
        )