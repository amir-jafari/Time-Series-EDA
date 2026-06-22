.. _user_guide_decomposition:

Decomposition
=============

The ``tseda.decomposition`` module splits a time series into its structural
components: **trend**, **seasonal**, and **residual**.  Two decomposers are
provided: classical (moving-average) and STL (Seasonal and Trend decomposition
using Loess).

Classical decomposition
-----------------------

:class:`~tseda.decomposition.ClassicalDecomposer` implements the standard
moving-average approach.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tseda import TimeSeries
   from tseda.decomposition import ClassicalDecomposer

   # Build a series with a weekly cycle
   idx  = pd.date_range("2021-01-01", periods=365, freq="D")
   t    = np.arange(365)
   vals = 0.05 * t + 5 * np.sin(2 * np.pi * t / 7) + np.random.randn(365)
   ts   = TimeSeries(vals, index=idx, name="sales", unit="units")

   result = ClassicalDecomposer().decompose(ts, period=7, model="additive")
   print(result.summary())

The result exposes four ``TimeSeries`` objects:

* ``result.trend``
* ``result.seasonal``
* ``result.residual``
* ``result.observed``

And strength metrics:

* ``result.seasonal_strength`` — proportion of variance explained by the seasonal component.
* ``result.trend_strength`` — proportion of variance explained by the trend.

Additive vs multiplicative
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Additive: observed = trend + seasonal + residual  (default)
   r_add  = ClassicalDecomposer().decompose(ts, period=7, model="additive")

   # Multiplicative: observed = trend × seasonal × residual
   r_mult = ClassicalDecomposer().decompose(ts, period=7, model="multiplicative")

STL decomposition
-----------------

:class:`~tseda.decomposition.STLDecomposer` uses the Loess-based STL algorithm
which is more robust to outliers and supports non-integer seasonality.

.. note::

   STL requires ``statsmodels``.  Install with
   ``pip install "timeseries-eda[stats]"``.

.. code-block:: python

   from tseda.decomposition import STLDecomposer

   result = STLDecomposer().decompose(ts, period=7)
   print(result.seasonal_strength)

The result object has the same interface as ``ClassicalDecomposer`` output:
``trend``, ``seasonal``, ``residual``, ``observed``, ``seasonal_strength``,
and ``trend_strength``.

Choosing a period
-----------------

If you are unsure of the period, run
:class:`~tseda.seasonality.SeasonalityDetector` first:

.. code-block:: python

   from tseda.seasonality import SeasonalityDetector

   season = SeasonalityDetector().detect(ts)
   period = season.dominant_period

   result = ClassicalDecomposer().decompose(ts, period=period)