.. _user_guide_forecastability:

Forecastability
===============

The ``tseda.forecastability`` module answers two questions before modelling:

1. **Is this series forecastable?**  — :class:`~tseda.forecastability.ForecastabilityScorer`
2. **Is there data leakage?**  — :class:`~tseda.forecastability.LeakageDetector`

Forecastability scoring
-----------------------

:class:`~tseda.forecastability.ForecastabilityScorer` computes an
``overall_score`` in [0, 1] from several sub-scores and returns a plain-
language recommendation.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tseda import TimeSeries
   from tseda.forecastability import ForecastabilityScorer

   idx  = pd.date_range("2020-01-01", periods=500, freq="D")
   vals = np.cumsum(np.random.randn(500))
   ts   = TimeSeries(vals, index=idx, name="price", unit="USD")

   report = ForecastabilityScorer().score(ts)
   print(report.overall_score)      # 0–1 (higher = more forecastable)
   print(report.recommendation)     # plain-language summary

Sub-scores available on the report object:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - What it measures
   * - ``stationarity_score``
     - Penalises strong unit roots.
   * - ``autocorrelation_score``
     - Rewards significant autocorrelation (signal to exploit).
   * - ``noise_score``
     - Penalises high-entropy / near-white-noise series.
   * - ``sample_size_score``
     - Rewards longer series (more data → better forecasts).
   * - ``missing_score``
     - Penalises high rates of NaN or index gaps.

Leakage detection
-----------------

:class:`~tseda.forecastability.LeakageDetector` identifies high correlation
between the target series and lagged versions of itself or of a covariate,
which can indicate that the feature would not be available at prediction time.

.. code-block:: python

   from tseda.forecastability import LeakageDetector

   det    = LeakageDetector()
   report = det.detect(ts, max_lag=30)

   print(report.leaky_lags)          # list of lag indices with high correlation
   print(report.max_pearson_r)       # maximum Pearson r across all lags
   print(report.is_leaky)            # True if any lag exceeds the threshold

Interpreting the overall score
-------------------------------

+------------------+------------------------------------------------------------+
| Score range      | Interpretation                                             |
+==================+============================================================+
| 0.75 – 1.00      | Highly forecastable.  Standard models should work well.    |
+------------------+------------------------------------------------------------+
| 0.50 – 0.74      | Moderately forecastable.  Feature engineering may help.    |
+------------------+------------------------------------------------------------+
| 0.25 – 0.49      | Difficult.  Consider longer history or external covariates.|
+------------------+------------------------------------------------------------+
| 0.00 – 0.24      | Near-random.  Forecast uncertainty will be very high.      |
+------------------+------------------------------------------------------------+