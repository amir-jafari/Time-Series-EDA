.. _user_guide_visualization:

Visualization
=============

The ``tseda.visualization`` module provides a full suite of
Matplotlib-based plots.  Every function accepts an optional ``ax`` argument
to embed the plot in an existing figure, and returns the ``Figure`` object.

Setup
-----

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib
   matplotlib.use("Agg")   # use "TkAgg" or "QtAgg" for interactive windows

   from tseda import TimeSeries

   idx  = pd.date_range("2021-01-01", periods=365, freq="D")
   t    = np.arange(365)
   vals = 0.05 * t + 5 * np.sin(2 * np.pi * t / 7) + np.random.randn(365)
   ts   = TimeSeries(vals, index=idx, name="sales", unit="units")

Time plots
----------

.. code-block:: python

   from tseda.visualization import (
       plot_series,
       plot_annual_boxplots,
       plot_seasonal_subseries,
       plot_lag,
       plot_calendar_heatmap,
       plot_density_ridge,
   )

   plot_series(ts).savefig("series.png")
   plot_annual_boxplots(ts).savefig("annual_boxplots.png")
   plot_seasonal_subseries(ts, period=7).savefig("subseries.png")
   plot_lag(ts, lags=[1, 7, 14]).savefig("lag.png")
   plot_calendar_heatmap(ts).savefig("heatmap.png")
   plot_density_ridge(ts).savefig("ridge.png")

Seasonality plots
-----------------

.. code-block:: python

   from tseda.visualization import (
       plot_periodogram,
       plot_polar_seasonal,
       plot_season_heatmap,
       plot_monthly_boxplots,
   )

   plot_periodogram(ts).savefig("periodogram.png")
   plot_polar_seasonal(ts, period=7).savefig("polar.png")
   plot_season_heatmap(ts, period=7).savefig("season_heatmap.png")
   plot_monthly_boxplots(ts).savefig("monthly_boxplots.png")

Correlation plots
-----------------

.. code-block:: python

   from tseda.visualization import plot_acf_pacf, plot_acf_heatmap

   plot_acf_pacf(ts, lags=40).savefig("acf_pacf.png")
   plot_acf_heatmap(ts, max_lag=40).savefig("acf_heatmap.png")

Distribution plots
------------------

.. code-block:: python

   from tseda.visualization import plot_distribution, plot_qq, plot_rolling_stats

   plot_distribution(ts).savefig("distribution.png")
   plot_qq(ts).savefig("qq.png")
   plot_rolling_stats(ts, window=30).savefig("rolling_stats.png")

Quality plots
-------------

.. code-block:: python

   from tseda.quality import MissingValueAnalyzer, OutlierDetector
   from tseda.visualization import plot_missing_heatmap, plot_outliers, plot_outlier_score

   mv_report  = MissingValueAnalyzer().analyze(ts)
   out_report = OutlierDetector().mad(ts)

   plot_missing_heatmap(ts, mv_report).savefig("missing.png")
   plot_outliers(ts, out_report).savefig("outliers.png")
   plot_outlier_score(ts, out_report).savefig("outlier_scores.png")

Decomposition plots
-------------------

.. code-block:: python

   from tseda.decomposition import ClassicalDecomposer
   from tseda.visualization import (
       plot_decomposition,
       plot_strength_radar,
       plot_residual_diagnostics,
   )

   dec = ClassicalDecomposer().decompose(ts, period=7)

   plot_decomposition(dec).savefig("decomposition.png")
   plot_strength_radar(dec).savefig("radar.png")
   plot_residual_diagnostics(dec).savefig("residuals.png")

Anomaly and changepoint plots
------------------------------

.. code-block:: python

   from tseda.anomaly import AnomalyDetector
   from tseda.visualization import plot_anomalies, plot_anomaly_scores

   ar = AnomalyDetector().rolling_iqr(ts)
   plot_anomalies(ts, ar).savefig("anomalies.png")
   plot_anomaly_scores(ts, ar).savefig("anomaly_scores.png")

Embedding in an existing figure
--------------------------------

All plot functions accept an ``ax`` keyword to embed the chart:

.. code-block:: python

   import matplotlib.pyplot as plt
   from tseda.visualization import plot_series, plot_acf_pacf

   fig, axes = plt.subplots(1, 2, figsize=(14, 4))
   plot_series(ts, ax=axes[0])
   plot_acf_pacf(ts, lags=20, ax=axes[1])
   fig.tight_layout()
   fig.savefig("combined.png")