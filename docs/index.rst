.. tseda documentation master file

tseda — Time Series EDA
========================

**Version:** |release| · **License:** MIT

*"Understand your time series before you forecast it."*

tseda is a comprehensive, dependency-light Python toolkit that provides the
time-series equivalent of `YData-Profiling
<https://docs.profiling.ydata.ai/>`_ for tabular data.  A single call
produces a complete picture of your data: quality diagnostics,
statistical properties, seasonality, anomalies, structural breaks,
forecastability scores, and a model-recommendation report.

.. code-block:: python

   import numpy as np, pandas as pd
   from tseda import TimeSeries

   idx = pd.date_range("2020-01-01", periods=365, freq="D")
   ts  = TimeSeries(np.cumsum(np.random.randn(365)), index=idx,
                    name="returns", unit="USD")
   print(ts)

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/timeseries_object
   user_guide/quality
   user_guide/statistics
   user_guide/decomposition
   user_guide/seasonality
   user_guide/anomaly
   user_guide/features
   user_guide/forecastability
   user_guide/visualization
   user_guide/reports

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/core
   api/quality
   api/statistics
   api/decomposition
   api/seasonality
   api/anomaly
   api/changepoint
   api/features
   api/forecastability
   api/visualization
   api/report

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`