.. _user_guide_installation:

Installation
============

Basic install
-------------

.. code-block:: bash

   pip install timeseries-eda

With stationarity tests (ADF, KPSS, Phillips-Perron)
-----------------------------------------------------

.. code-block:: bash

   pip install "timeseries-eda[stats]"

Development install (from source)
----------------------------------

.. code-block:: bash

   git clone https://github.com/amir-jafari/Time-Series-EDA.git
   cd Time-Series-EDA
   pip install -e ".[stats,dev]"

Python and dependency requirements are listed in :doc:`../changelog`.