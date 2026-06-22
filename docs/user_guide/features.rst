.. _user_guide_features:

Feature Extraction
==================

The ``tseda.features`` module extracts numerical features from a
:class:`~tseda.core.TimeSeries` that can be used as model inputs, for
clustering, or for comparing multiple series.  Three extractors are provided.

Temporal features
-----------------

:class:`~tseda.features.TemporalFeatureExtractor` derives calendar-based
features from the datetime index.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tseda import TimeSeries
   from tseda.features import TemporalFeatureExtractor

   idx  = pd.date_range("2022-01-01", periods=365, freq="D")
   vals = np.random.randn(365)
   ts   = TimeSeries(vals, index=idx, name="demand")

   df = TemporalFeatureExtractor().extract(ts)
   print(df.columns.tolist())
   # e.g. ['hour', 'day_of_week', 'day_of_month', 'week_of_year',
   #        'month', 'quarter', 'year', 'is_weekend', ...]

Statistical features
--------------------

:class:`~tseda.features.StatisticalFeatureExtractor` returns a flat
dictionary of summary statistics useful as ML features.

.. code-block:: python

   from tseda.features import StatisticalFeatureExtractor

   feats = StatisticalFeatureExtractor().extract(ts)
   print(feats)
   # e.g. {'mean': ..., 'std': ..., 'skewness': ..., 'kurtosis': ...,
   #        'autocorr_lag1': ..., 'entropy': ..., ...}

Spectral features
-----------------

:class:`~tseda.features.SpectralFeatureExtractor` extracts frequency-domain
features via FFT.

.. code-block:: python

   from tseda.features import SpectralFeatureExtractor

   feats = SpectralFeatureExtractor().extract(ts)
   print(feats)
   # e.g. {'dominant_frequency': ..., 'spectral_entropy': ...,
   #        'spectral_centroid': ..., 'spectral_rolloff': ..., ...}

Combining all features
----------------------

.. code-block:: python

   from tseda.features import (
       TemporalFeatureExtractor,
       StatisticalFeatureExtractor,
       SpectralFeatureExtractor,
   )
   import pandas as pd

   temporal = TemporalFeatureExtractor().extract(ts)
   stats    = StatisticalFeatureExtractor().extract(ts)
   spectral = SpectralFeatureExtractor().extract(ts)

   # Scalar features → single-row DataFrame for ML pipelines
   scalar_df = pd.DataFrame({**stats, **spectral}, index=[ts.name])
   print(scalar_df)