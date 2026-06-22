.. _user_guide_statistics:

Statistics
==========

The ``tseda.statistics`` module provides three groups of analyses:
descriptive statistics, stationarity tests, and autocorrelation.

Descriptive statistics
----------------------

:class:`~tseda.statistics.DescriptiveAnalyzer` computes a comprehensive
summary of a series in one call.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tseda import TimeSeries
   from tseda.statistics import DescriptiveAnalyzer

   idx  = pd.date_range("2020-01-01", periods=500, freq="D")
   vals = np.cumsum(np.random.randn(500))
   ts   = TimeSeries(vals, index=idx, name="price", unit="USD")

   desc = DescriptiveAnalyzer().analyze(ts)
   print(desc.summary())

The result object exposes attributes such as ``mean``, ``std``, ``skewness``,
``kurtosis``, ``min``, ``max``, ``median``, ``q25``, ``q75``.

Stationarity tests
------------------

:class:`~tseda.statistics.StationarityTester` wraps the three standard
unit-root and stationarity tests from statsmodels.

.. note::

   These methods require ``statsmodels``.  Install with
   ``pip install "timeseries-eda[stats]"``.

.. code-block:: python

   from tseda.statistics import StationarityTester

   tester = StationarityTester()

   adf  = tester.adf(ts)   # Augmented Dickey-Fuller (H0: unit root)
   kpss = tester.kpss(ts)  # KPSS (H0: stationary)
   pp   = tester.pp(ts)    # Phillips-Perron (H0: unit root)

   print(adf.summary())

Each result object provides:

* ``stat`` — the test statistic.
* ``pvalue`` — p-value.
* ``is_stationary`` — ``True`` when the test rejects the null at 5 %.
* ``critical_values`` — dictionary of critical values at 1 %, 5 %, 10 %.
* ``summary()`` — formatted table combining all of the above.

Interpreting ADF vs KPSS
~~~~~~~~~~~~~~~~~~~~~~~~~

ADF and KPSS have **opposite** null hypotheses:

+------+-----+------------------------------------------------------------------+
| ADF  |KPSS | Interpretation                                                   |
+======+=====+==================================================================+
| fail |fail | Evidence of non-stationarity is unclear; inspect visually.       |
+------+-----+------------------------------------------------------------------+
| fail |pass | Likely non-stationary (unit root present).                       |
+------+-----+------------------------------------------------------------------+
| pass |fail | Trend-stationary (consider detrending).                          |
+------+-----+------------------------------------------------------------------+
| pass |pass | Likely stationary.                                               |
+------+-----+------------------------------------------------------------------+

Autocorrelation
---------------

:class:`~tseda.statistics.AutocorrelationAnalyzer` computes ACF and PACF
coefficients and returns them in a result object.

.. code-block:: python

   from tseda.statistics import AutocorrelationAnalyzer

   r = AutocorrelationAnalyzer().analyze(ts, lags=40)

   print(r.acf[:5])    # first 5 ACF coefficients
   print(r.pacf[:5])   # first 5 PACF coefficients

   # Lag at which ACF first crosses the significance bound
   print(r.first_insignificant_lag)