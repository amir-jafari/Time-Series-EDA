.. _examples:

Examples
========

.. toctree::
   :maxdepth: 1

   Global_Air_Pollution_EDA

Global Air Pollution EDA
------------------------

A complete end-to-end tseda workflow applied to the
`Global Air Pollution Dataset <https://www.kaggle.com/datasets/hasibalmurad/global-air-pollution-dataset>`_
(23 463 cities × 5 pollutants).

**What the notebook covers:**

* Loading and aggregating the dataset to country-level mean values
* Constructing :class:`~tseda.core.TimeSeries` objects for each pollutant
* Data quality analysis (missing values, outliers)
* Descriptive statistics and stationarity testing
* Decomposition, seasonality detection, anomaly detection
* Forecastability scoring
* Generating HTML EDA reports for each pollutant

The generated HTML reports are available in the :doc:`../reports/index` section.