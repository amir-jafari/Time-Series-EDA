"""
Spectral feature extraction for time series.

Computes frequency-domain descriptors of a :class:`~tseda.core.TimeSeries`
using the FFT power spectrum.  All computations use pure numpy / scipy.

Feature groups
--------------
* **Energy / Power** — total spectral power, power in low / mid / high bands.
* **Shape** — spectral centroid, bandwidth, rolloff frequency, spectral entropy.
* **Peak** — dominant frequency, dominant period, number of spectral peaks.
* **Temporal** — spectral flatness (ratio of geometric to arithmetic mean power).

Classes
-------
SpectralFeatureExtractor
    Stateless extractor.

Examples
--------
>>> import pandas as pd, numpy as np
>>> from tseda import TimeSeries
>>> from tseda.features.spectral import SpectralFeatureExtractor

>>> rng = np.random.default_rng(0)
>>> idx = pd.date_range("2020", periods=256, freq="D")
>>> ts  = TimeSeries(np.sin(2 * np.pi * np.arange(256) / 7), index=idx)
>>> df  = SpectralFeatureExtractor().extract(ts)
>>> int(round(float(df["dominant_period"].iloc[0])))
7
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from tseda.core.timeseries import TimeSeries

__all__ = ["SpectralFeatureExtractor"]


class SpectralFeatureExtractor:
    """Extract frequency-domain features from a :class:`~tseda.core.TimeSeries`.

    The extractor is **stateless**.  NaN values in the series are replaced
    by linear interpolation before FFT analysis.

    Methods
    -------
    extract(ts, n_bands)
        Return a single-row :class:`pandas.DataFrame` of spectral features.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from tseda import TimeSeries
    >>> from tseda.features.spectral import SpectralFeatureExtractor

    >>> idx = pd.date_range("2020", periods=128, freq="h")
    >>> ts  = TimeSeries(np.sin(2 * np.pi * np.arange(128) / 24), index=idx)
    >>> df  = SpectralFeatureExtractor().extract(ts)
    >>> "spectral_centroid" in df.columns
    True
    """

    def extract(
        self,
        ts: TimeSeries,
        *,
        n_bands: int = 3,
    ) -> pd.DataFrame:
        """Compute spectral features for *ts*.

        Parameters
        ----------
        ts : TimeSeries
            Input series.  NaN values are linearly interpolated before
            the FFT.
        n_bands : int, optional
            Number of equal-width frequency bands for band power features.
            Default 3 (low / mid / high).  Must be >= 1.

        Returns
        -------
        pandas.DataFrame
            One row, columns:

            Energy:
              ``total_power``,
              ``band_power_0`` … ``band_power_{n_bands-1}``.

            Shape:
              ``spectral_centroid``, ``spectral_bandwidth``,
              ``spectral_rolloff_0.5``, ``spectral_rolloff_0.85``,
              ``spectral_entropy``, ``spectral_flatness``.

            Peak:
              ``dominant_freq``, ``dominant_period``, ``n_spectral_peaks``.

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        ValueError
            If the series has fewer than 8 non-NaN observations or
            *n_bands* < 1.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> from tseda import TimeSeries
        >>> from tseda.features.spectral import SpectralFeatureExtractor
        >>> idx = pd.date_range("2020", periods=256, freq="D")
        >>> ts  = TimeSeries(np.cos(2*np.pi*np.arange(256)/7), index=idx)
        >>> df  = SpectralFeatureExtractor().extract(ts)
        >>> int(round(float(df["dominant_period"].iloc[0])))
        7
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )
        if n_bands < 1:
            raise ValueError(f"'n_bands' must be >= 1, got {n_bands}.")

        vals = ts.values.astype(float)
        n    = len(vals)

        if np.sum(~np.isnan(vals)) < 8:
            raise ValueError(
                "SpectralFeatureExtractor requires at least 8 non-NaN "
                f"observations; got {int(np.sum(~np.isnan(vals)))}."
            )

        # Fill NaN by linear interpolation
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            idx_arr    = np.arange(n)
            vals[nan_mask] = np.interp(
                idx_arr[nan_mask], idx_arr[~nan_mask], vals[~nan_mask]
            )

        # Detrend + Hann window
        x = sp_signal.detrend(vals, type="linear")
        x = x * np.hanning(n)

        # FFT power spectrum (one-sided, excluding DC)
        fft_vals = np.fft.rfft(x)
        power    = np.abs(fft_vals) ** 2

        # Frequency axis (normalised: 0 → 0.5 cycles/sample)
        freqs    = np.fft.rfftfreq(n)        # shape (n//2 + 1,)
        # Exclude DC (index 0)
        power_ac = power[1:]
        freqs_ac = freqs[1:]
        n_ac     = len(power_ac)

        total_power = float(power_ac.sum()) if power_ac.sum() > 0 else 1e-12

        # ── Band power ────────────────────────────────────────────────
        band_edges = np.linspace(0, n_ac, n_bands + 1, dtype=int)
        band_power = {
            f"band_power_{i}": float(power_ac[band_edges[i]:band_edges[i + 1]].sum())
            for i in range(n_bands)
        }

        # ── Shape ─────────────────────────────────────────────────────
        # Spectral centroid (weighted mean frequency)
        centroid = float(np.sum(freqs_ac * power_ac) / total_power)

        # Spectral bandwidth (weighted std of frequency)
        bw = float(
            np.sqrt(np.sum(((freqs_ac - centroid) ** 2) * power_ac) / total_power)
        )

        # Spectral rolloff (frequency below which X% of power resides)
        cumpower  = np.cumsum(power_ac)
        rolloff_50  = float(freqs_ac[np.searchsorted(cumpower, 0.50 * total_power)])
        rolloff_85  = float(freqs_ac[np.searchsorted(cumpower, 0.85 * total_power)])

        # Spectral entropy (normalised power distribution)
        p_norm   = power_ac / total_power
        # Avoid log(0)
        p_norm   = np.where(p_norm > 0, p_norm, 1e-12)
        sp_ent   = float(-np.sum(p_norm * np.log(p_norm)) / np.log(n_ac))

        # Spectral flatness (geometric / arithmetic mean ratio)
        log_mean  = float(np.exp(np.mean(np.log(power_ac + 1e-12))))
        arith_mean = float(np.mean(power_ac))
        sp_flat   = log_mean / arith_mean if arith_mean > 0 else 0.0

        # ── Peak ──────────────────────────────────────────────────────
        dom_idx  = int(np.argmax(power_ac))
        dom_freq = float(freqs_ac[dom_idx])
        dom_per  = float(1.0 / dom_freq) if dom_freq > 0 else float("nan")

        peaks, _ = sp_signal.find_peaks(power_ac, height=0)
        n_peaks  = int(len(peaks))

        row: dict = {
            "total_power":          total_power,
            **band_power,
            "spectral_centroid":    centroid,
            "spectral_bandwidth":   bw,
            "spectral_rolloff_0.5": rolloff_50,
            "spectral_rolloff_0.85":rolloff_85,
            "spectral_entropy":     sp_ent,
            "spectral_flatness":    sp_flat,
            "dominant_freq":        dom_freq,
            "dominant_period":      dom_per,
            "n_spectral_peaks":     float(n_peaks),
        }

        return pd.DataFrame([row], index=[ts.index[0]])
