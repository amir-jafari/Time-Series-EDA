.. _user_guide_timeseries_object:

The TimeSeries Object
=====================

:class:`~tseda.core.TimeSeries` is the central data structure in tseda.
Every analyser, plotter, and report builder accepts one as input.

Construction
------------

From a numpy array and an index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tseda import TimeSeries

   idx  = pd.date_range("2021-01-01", periods=100, freq="D")
   vals = np.random.randn(100)

   ts = TimeSeries(vals, index=idx, name="returns", unit="%")

From an existing pandas Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   s  = pd.Series(np.random.randn(50),
                  index=pd.date_range("2022-06-01", periods=50, freq="h"),
                  name="temp")
   ts = TimeSeries.from_series(s, unit="°C")

From a DataFrame column
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   df = pd.DataFrame(
       {"close": np.random.randn(200) + 100},
       index=pd.date_range("2020-01-01", periods=200, freq="B"),
   )
   ts = TimeSeries.from_dataframe(df, column="close", name="AAPL", unit="USD")

Key properties
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Description
   * - ``ts.n``
     - Number of observations.
   * - ``ts.values``
     - 1-D ``float64`` numpy array (copy).
   * - ``ts.index``
     - ``pandas.DatetimeIndex``.
   * - ``ts.start`` / ``ts.end``
     - First and last timestamps.
   * - ``ts.duration``
     - ``pandas.Timedelta`` from start to end.
   * - ``ts.freq``
     - Inferred pandas offset alias (``"D"``, ``"h"``, …) or ``None``.
   * - ``ts.name``
     - Short identifier string.
   * - ``ts.unit``
     - Physical unit label used in plot axes.
   * - ``ts.has_nan``
     - ``True`` when any value is NaN.
   * - ``ts.is_regular``
     - ``True`` when all time steps are equal.

Transforms
----------

All transform methods return a **new** ``TimeSeries``; the original is
never modified.

.. code-block:: python

   # Slice by date
   ts_2021 = ts.slice("2021-01-01", "2021-12-31")

   # Resample to monthly mean
   ts_monthly = ts.resample("MS")

   # First-order differencing
   ts_diff = ts.diff(order=1)

   # Log transform
   ts_log = ts.log()

   # Normalise to zero mean, unit variance
   ts_norm = ts.normalize()

   # Rolling mean and standard deviation
   ts_roll = ts.rolling_mean(window=7)
   ts_std  = ts.rolling_std(window=7)

Converting back to pandas
--------------------------

.. code-block:: python

   s  = ts.to_series()     # pandas.Series
   df = ts.to_dataframe()  # pandas.DataFrame (one column)

Frequency inference
-------------------

When ``freq`` is omitted, tseda infers it from the index.  It first
tries ``pandas.infer_freq``; if that fails (e.g. because the index has
gaps) it uses a median time-step heuristic.  The result is stored in
``ts.freq`` and is required by gap-detection, decomposition, and other
analyses.

Pass ``freq`` explicitly to override:

.. code-block:: python

   ts = TimeSeries(vals, index=idx, freq="D")