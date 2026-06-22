tseda — Time Series EDA
=======================

.. image:: https://img.shields.io/pypi/v/tseda.svg
   :target: https://pypi.org/project/tseda/
   :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/tseda.svg
   :target: https://pypi.org/project/tseda/
   :alt: Python Versions

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/amir-jafari/Time-Series-EDA/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/github/actions/workflow/status/amir-jafari/Time-Series-EDA/tests.yml?branch=main
   :target: https://github.com/amir-jafari/Time-Series-EDA/actions
   :alt: Tests

**"Understand your time series before you forecast it."**

``tseda`` is a comprehensive, dependency-light Python toolkit for time series
Exploratory Data Analysis (EDA).  It is to time series what
`YData-Profiling <https://docs.profiling.ydata.ai/>`_ is to tabular data:
a single command that produces a complete understanding of any time series
dataset before you start modelling.

----

Why tseda?
----------

Existing libraries solve individual problems.  No single package provides all of:

* Comprehensive EDA & data auditing
* Forecastability assessment
* Automated diagnostics (stationarity, seasonality, anomalies)
* Structural break / changepoint detection
* Feature engineering
* Model recommendations
* Interactive reports

``tseda`` fills that gap, using only **numpy**, **pandas**, **scipy**, and
**matplotlib** as core dependencies.

----

Installation
------------

.. code-block:: bash

   pip install tseda

For stationarity tests that use statsmodels (ADF, KPSS, Phillips-Perron):

.. code-block:: bash

   pip install tseda[stats]

For building the documentation:

.. code-block:: bash

   pip install tseda[docs]

----

Quick Start
-----------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tseda import TimeSeries

   # Build a TimeSeries object
   idx = pd.date_range("2020-01-01", periods=365, freq="D")
   ts  = TimeSeries(
       np.cumsum(np.random.randn(365)),
       index=idx,
       name="stock_price",
       unit="USD",
   )
   print(ts)

   # Data quality
   from tseda.quality import MissingValueAnalyzer, OutlierDetector
   missing = MissingValueAnalyzer().analyze(ts)
   outliers = OutlierDetector().mad(ts)

   # Statistics
   from tseda.statistics import DescriptiveAnalyzer, StationarityTester
   stats = DescriptiveAnalyzer().analyze(ts)
   adf   = StationarityTester().adf(ts)
   print(adf.summary() if hasattr(adf, "summary") else adf)

   # Decomposition
   from tseda.decomposition import STLDecomposer
   dec = STLDecomposer().decompose(ts, period=7)
   print(dec.summary())

   # Seasonality
   from tseda.seasonality import SeasonalityDetector
   season = SeasonalityDetector().detect(ts)
   print(f"Dominant period: {season.dominant_period}")

----

Modules
-------

+---------------------+------------------------------------------------------+
| Module              | Capability                                           |
+=====================+======================================================+
| ``core``            | ``TimeSeries`` data structure & validators           |
+---------------------+------------------------------------------------------+
| ``quality``         | Missing values, outlier detection, flat-line checks  |
+---------------------+------------------------------------------------------+
| ``statistics``      | Descriptive stats, stationarity, ACF/PACF            |
+---------------------+------------------------------------------------------+
| ``decomposition``   | Classical & STL decomposition                        |
+---------------------+------------------------------------------------------+
| ``seasonality``     | FFT periodogram + ACF-based period detection         |
+---------------------+------------------------------------------------------+
| ``anomaly``         | Rolling IQR/Z-score, STL-residual anomaly detection  |
+---------------------+------------------------------------------------------+
| ``changepoint``     | Structural break / CUSUM detection                   |
+---------------------+------------------------------------------------------+
| ``features``        | Temporal, statistical, spectral feature extraction   |
+---------------------+------------------------------------------------------+
| ``forecastability`` | Forecast-readiness scoring & leakage detection       |
+---------------------+------------------------------------------------------+
| ``visualization``   | Matplotlib plot suite                                |
+---------------------+------------------------------------------------------+
| ``report``          | HTML & console report generation                     |
+---------------------+------------------------------------------------------+

----

Dependencies
------------

**Core** (always installed):

* ``numpy >= 1.23``
* ``pandas >= 1.5``
* ``scipy >= 1.9``
* ``matplotlib >= 3.6``

**Optional**:

* ``statsmodels >= 0.14`` — ADF, KPSS, Phillips-Perron, STL (``pip install tseda[stats]``)

----

Documentation
-------------

`https://amir-jafari.github.io/Time-Series-EDA <https://amir-jafari.github.io/Time-Series-EDA>`_

----

Contributing
------------

Contributions are welcome!  Please open an issue or pull request at
`https://github.com/amir-jafari/Time-Series-EDA <https://github.com/amir-jafari/Time-Series-EDA>`_.

----

License
-------

MIT © 2026 Amirhossein Jafari
