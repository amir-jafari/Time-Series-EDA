.. _user_guide_seasonality:

Seasonality Detection
=====================

The ``tseda.seasonality`` module detects periodic patterns in a time series
using two complementary methods: an FFT-based periodogram and the ACF.

Basic usage
-----------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tseda import TimeSeries
   from tseda.seasonality import SeasonalityDetector

   # Construct a series with a clear weekly (period=7) cycle
   idx  = pd.date_range("2021-01-01", periods=365, freq="D")
   t    = np.arange(365)
   vals = 10 * np.sin(2 * np.pi * t / 7) + np.random.randn(365)
   ts   = TimeSeries(vals, index=idx, name="energy", unit="kWh")

   season = SeasonalityDetector().detect(ts)

   print(season.dominant_period)     # most prominent period (e.g. 7)
   print(season.all_periods)         # all detected periods, ranked by strength
   print(season.seasonal_strength)   # 0–1 strength of the dominant period
   print(season.is_seasonal)         # True / False

Result attributes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``dominant_period``
     - Integer period (in samples) with the highest spectral power.
   * - ``all_periods``
     - List of detected periods sorted by strength (descending).
   * - ``period_strengths``
     - Dict mapping period → spectral strength score.
   * - ``seasonal_strength``
     - 0–1 strength estimate of the dominant period.
   * - ``is_seasonal``
     - ``True`` when at least one period is detected above the threshold.

Detection methods
-----------------

The detector combines two signals:

1. **Periodogram** — FFT power spectrum.  Peaks identify candidate
   periods.  Harmonics are automatically suppressed so that the
   fundamental period is returned rather than its multiples.

2. **ACF** — Autocorrelation function.  Confirms candidate periods
   by checking that the ACF has a local maximum at the expected lag.

Using the detected period for decomposition
-------------------------------------------

.. code-block:: python

   from tseda.decomposition import ClassicalDecomposer

   period = season.dominant_period
   result = ClassicalDecomposer().decompose(ts, period=period)
   print(result.summary())