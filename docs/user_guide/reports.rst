.. _user_guide_reports:

Reports
=======

The ``tseda.report`` module assembles the results of every analysis into a
single deliverable — either an interactive HTML file or a plain-text console
summary.

HTML report
-----------

:class:`~tseda.report.HTMLReport` runs all analysers automatically and writes
a self-contained HTML file.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tseda import TimeSeries
   from tseda.report import HTMLReport

   idx  = pd.date_range("2020-01-01", periods=500, freq="D")
   vals = np.cumsum(np.random.randn(500)) + 100
   ts   = TimeSeries(vals, index=idx, name="close", unit="USD")

   HTMLReport(ts).build("eda_report.html")
   # Open eda_report.html in any browser

The HTML report includes:

* Series overview (n, start, end, freq, NaN count)
* Data quality section (missing values, outliers, flat-lines)
* Descriptive statistics table
* Stationarity test results (requires ``[stats]`` extra)
* Decomposition chart (trend, seasonal, residual)
* Seasonality summary (dominant period, strength)
* Anomaly summary
* Forecastability score and recommendation
* Embedded Matplotlib figures

Console report
--------------

:class:`~tseda.report.ConsoleReport` prints a structured text summary to
stdout, suitable for notebooks and scripts.

.. code-block:: python

   from tseda.report import ConsoleReport

   ConsoleReport(ts).print()

Example output::

   ╔══════════════════════════════════════════╗
   ║   tseda — Time Series EDA Report        ║
   ╠══════════════════════════════════════════╣
   ║  Series  : close                        ║
   ║  Length  : 500  (2020-01-01 → 2021-05-15)
   ║  Freq    : D                            ║
   ║  NaN     : 0 (0.0%)                     ║
   ║  Gaps    : 0                            ║
   ╠══════════════════════════════════════════╣
   ║  Mean    :  ...                         ║
   ║  Std     :  ...                         ║
   ╚══════════════════════════════════════════╝