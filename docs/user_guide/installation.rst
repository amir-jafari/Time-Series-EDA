.. _user_guide_installation:

Installation
============

Requirements
------------

tseda requires Python 3.9 or later and the following core libraries:

* ``numpy >= 1.23``
* ``pandas >= 1.5``
* ``scipy >= 1.9``
* ``matplotlib >= 3.6``

Basic install
-------------

.. code-block:: bash

   pip install timeseries-eda

This installs everything needed for all modules **except** the
statsmodels-backed stationarity tests.

With stationarity tests (ADF, KPSS, Phillips-Perron)
-----------------------------------------------------

.. code-block:: bash

   pip install "timeseries-eda[stats]"

This adds ``statsmodels >= 0.14``, which is required by
:meth:`~tseda.statistics.StationarityTester.adf`,
:meth:`~tseda.statistics.StationarityTester.kpss`, and
:meth:`~tseda.statistics.StationarityTester.pp`.

Development install (from source)
----------------------------------

.. code-block:: bash

   git clone https://github.com/amir-jafari/Time-Series-EDA.git
   cd Time-Series-EDA
   pip install -e ".[stats,dev]"

Verifying the install
---------------------

.. code-block:: python

   import tseda
   print(tseda.__version__)