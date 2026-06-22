.. _user_guide_quality:

Data Quality
============

The ``tseda.quality`` module covers three common data-quality issues:
missing values (NaN and index gaps), outliers, and flat-line / duplicate
segments.

Missing values
--------------

:class:`~tseda.quality.MissingValueAnalyzer` distinguishes two concepts:

* **Value NaN** — a timestamp is present but the observation is ``numpy.nan``.
* **Index gap** — a timestamp that should exist (given the series frequency)
  is absent from the index entirely.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tseda import TimeSeries
   from tseda.quality import MissingValueAnalyzer

   idx  = pd.date_range("2022-01-01", periods=90, freq="D")
   vals = np.random.randn(90)
   vals[[10, 11, 50]] = np.nan           # inject NaN values

   ts = TimeSeries(vals, index=idx, name="sensor")
   r  = MissingValueAnalyzer().analyze(ts)

   print(r.n_nan)            # 3
   print(r.pct_nan)          # 3.33
   print(r.longest_nan_run)  # 2  (positions 10–11)
   print(r.n_gaps)           # 0  (index is complete)

Interpolation
~~~~~~~~~~~~~

.. code-block:: python

   filled = MissingValueAnalyzer().interpolate(ts, method="linear")
   # Other methods: "forward", "backward", "nearest", "zero",
   #                "constant" (needs fill_value=), "spline"

Outlier detection
-----------------

:class:`~tseda.quality.OutlierDetector` provides four detection strategies:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Description
   * - ``mad(ts)``
     - Median Absolute Deviation — robust to skew.
   * - ``iqr(ts)``
     - Inter-Quartile Range fence (1.5 × IQR by default).
   * - ``zscore(ts)``
     - Standard Z-score threshold.
   * - ``rolling_zscore(ts)``
     - Z-score computed in a rolling window — catches local outliers.

.. code-block:: python

   from tseda.quality import OutlierDetector

   det = OutlierDetector()

   r_mad = det.mad(ts, threshold=3.5)
   print(r_mad.n_outliers)
   print(r_mad.outlier_indices)

   # Clean the series (replace outliers with NaN)
   ts_clean = det.remove(ts, r_mad)

   # Or label with a boolean mask
   ts_labeled = det.label(ts, r_mad)

Flat-line and duplicate detection
----------------------------------

:class:`~tseda.quality.DuplicateDetector` finds segments where the value
does not change (flat-lines) and timestamps that appear more than once.

.. code-block:: python

   from tseda.quality import DuplicateDetector

   r = DuplicateDetector().analyze(ts)
   print(r.n_flatline_segments)
   print(r.n_duplicate_timestamps)