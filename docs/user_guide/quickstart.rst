.. _user_guide_quickstart:

Quickstart
==========

This page walks through a complete tseda workflow in under two minutes.

Build a TimeSeries
------------------

Every tseda analyser takes a :class:`~tseda.core.TimeSeries` object as input.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tseda import TimeSeries

   np.random.seed(42)
   idx = pd.date_range("2020-01-01", periods=365, freq="D")
   vals = np.cumsum(np.random.randn(365)) + 50   # random walk around 50

   ts = TimeSeries(vals, index=idx, name="price", unit="USD")
   print(ts)

Data quality check
------------------

.. code-block:: python

   from tseda.quality import MissingValueAnalyzer, OutlierDetector

   missing = MissingValueAnalyzer().analyze(ts)
   print(f"NaN count : {missing.n_nan}")
   print(f"Index gaps: {missing.n_gaps}")

   outliers = OutlierDetector().mad(ts)
   print(f"Outliers  : {outliers.n_outliers}")

Descriptive statistics and stationarity
----------------------------------------

.. code-block:: python

   from tseda.statistics import DescriptiveAnalyzer, StationarityTester

   desc = DescriptiveAnalyzer().analyze(ts)
   print(desc.summary())

   # ADF test (requires pip install timeseries-eda[stats])
   adf = StationarityTester().adf(ts)
   print(adf.summary())

Decomposition
-------------

.. code-block:: python

   from tseda.decomposition import ClassicalDecomposer

   result = ClassicalDecomposer().decompose(ts, period=7, model="additive")
   print(result.summary())

Seasonality detection
---------------------

.. code-block:: python

   from tseda.seasonality import SeasonalityDetector

   season = SeasonalityDetector().detect(ts)
   print(f"Dominant period: {season.dominant_period}")
   print(f"Seasonal strength: {season.seasonal_strength:.3f}")

Anomaly detection
-----------------

.. code-block:: python

   from tseda.anomaly import AnomalyDetector

   report = AnomalyDetector().rolling_iqr(ts)
   print(f"Anomalies found: {report.n_anomalies}")

Forecastability score
---------------------

.. code-block:: python

   from tseda.forecastability import ForecastabilityScorer

   score = ForecastabilityScorer().score(ts)
   print(f"Forecastability: {score.overall_score:.2f} / 1.00")
   print(score.recommendation)

Visualise
---------

.. code-block:: python

   import matplotlib
   matplotlib.use("Agg")   # or "TkAgg" / "QtAgg" for interactive windows

   from tseda.visualization import plot_series, plot_acf_pacf

   fig = plot_series(ts)
   fig.savefig("price.png")

   fig2 = plot_acf_pacf(ts, lags=40)
   fig2.savefig("acf_pacf.png")

Full HTML report
----------------

.. code-block:: python

   from tseda.report import HTMLReport

   HTMLReport(ts).build("report.html")
   # Open report.html in your browser