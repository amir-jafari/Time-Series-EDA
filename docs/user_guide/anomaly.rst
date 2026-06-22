.. _user_guide_anomaly:

Anomaly Detection
=================

The ``tseda.anomaly`` module finds point anomalies — individual observations
that deviate unusually from the surrounding data.  Four detection strategies
are available, each returning an :class:`~tseda.anomaly.AnomalyReport`.

Detection methods
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - When to use
   * - ``rolling_iqr``
     - General-purpose; robust to skew and heavy tails.
   * - ``rolling_z``
     - When the data is approximately Gaussian.
   * - ``stl_residual``
     - When trend and seasonality are present — anomalies in the residual.
   * - ``gesd``
     - GESD test; best for series with at most a few outliers.

Basic usage
-----------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tseda import TimeSeries
   from tseda.anomaly import AnomalyDetector

   np.random.seed(0)
   idx  = pd.date_range("2022-01-01", periods=200, freq="D")
   vals = np.random.randn(200)
   vals[[30, 80, 150]] = [8.0, -7.5, 9.0]   # inject spikes

   ts  = TimeSeries(vals, index=idx, name="sensor")
   det = AnomalyDetector()

   report = det.rolling_iqr(ts, window=21, threshold=3.0)
   print(report.n_anomalies)
   print(report.anomaly_indices)
   print(report.anomaly_timestamps)

STL-residual detection (handles seasonality)
--------------------------------------------

.. code-block:: python

   t    = np.arange(365)
   idx2 = pd.date_range("2021-01-01", periods=365, freq="D")
   vals2 = 5 * np.sin(2 * np.pi * t / 7) + np.random.randn(365)
   vals2[100] = 30.0           # large spike
   ts2 = TimeSeries(vals2, index=idx2, name="energy", unit="kWh")

   # stl_residual requires statsmodels: pip install timeseries-eda[stats]
   report2 = det.stl_residual(ts2, period=7)
   print(report2.n_anomalies)

Removing or labelling anomalies
--------------------------------

.. code-block:: python

   # Replace anomalies with NaN
   ts_clean = det.remove(ts, report)

   # Add a boolean label column (returns TimeSeries with values 0/1)
   ts_labeled = det.label(ts, report)