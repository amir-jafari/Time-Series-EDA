Changelog
=========

All notable changes to ``tseda`` are documented here.
This project follows `Semantic Versioning <https://semver.org/>`_.

----

0.1.3 (2026-06-22)
-------------------

Fix incorrect PyPI install command in README and STL docstring
(``pip install tseda`` → ``pip install timeseries-eda``).

----

0.1.2 (2026-06-22)
-------------------

Bug fixes.

**Correctness**

* ``core.TimeSeries._compute_is_regular`` — replaced
  ``DatetimeIndex.astype(np.int64)`` with unit-agnostic ``to_numpy()``
  timedelta comparison; fixes ``TypeError`` on pandas 2.x with
  non-nanosecond resolution indexes.
* ``forecastability.scorer._has_large_gaps`` — same timedelta fix.
* ``statistics.StationarityTester`` (native ADF fallback) — rewrote
  regressor construction in ``_adf_native`` to use a fixed sample size
  (``maxlag+1`` start) for all lag models, making AIC comparisons valid
  and eliminating ``ValueError`` crashes for lag ≥ 1.
* ``anomaly.AnomalyDetector.gesd`` — clamps ``max_outliers`` to
  ``n_finite // 2 − 1`` before delegating to ``OutlierDetector``;
  prevents crash on series with fewer than 22 observations.
* ``changepoint.ChangepointDetector.cusum`` — replaced post-hoc reset
  logic with a single incremental accumulator so consecutive changepoints
  are detected correctly; scores now reflect the reset-adjusted CUSUM
  values rather than the pre-reset arrays.
* ``changepoint.ChangepointDetector.variance_ratio`` — F-test degrees of
  freedom now assigned to the window with the larger variance (numerator),
  fixing an incorrect two-sided p-value.
* ``quality.MissingValueAnalyzer.interpolate`` (``method="linear"``) —
  leading and trailing NaN are now filled with the nearest boundary value
  when no ``limit`` is set, matching the documented "fill NaN values"
  contract.

**Performance**

* ``anomaly.AnomalyDetector.rolling_iqr`` and ``rolling_z`` — replaced
  O(n) Python loops (``series.iloc[i]`` per observation) with vectorized
  NumPy array operations.

----

0.1.1 (2026-06-21)
-------------------

Documentation and CI improvements (no API changes).

----

0.1.0 (2026-06-21)
-------------------

Initial release.

**Modules**

* ``tseda.core`` — :class:`~tseda.core.TimeSeries` data structure,
  type aliases (:class:`~tseda.core.Frequency`, :class:`~tseda.core.AggMethod`,
  :class:`~tseda.core.DiffMethod`), and validators.

* ``tseda.quality`` — Missing-value analysis
  (:class:`~tseda.quality.MissingValueAnalyzer`), outlier detection with
  IQR / Z-score / MAD / GESD (:class:`~tseda.quality.OutlierDetector`),
  and flat-line / near-zero detection
  (:class:`~tseda.quality.DuplicateDetector`).

* ``tseda.statistics`` — Comprehensive descriptive statistics
  (:class:`~tseda.statistics.DescriptiveAnalyzer`), stationarity tests
  ADF / KPSS / Phillips-Perron (:class:`~tseda.statistics.StationarityTester`),
  and ACF / PACF / Ljung-Box autocorrelation analysis
  (:class:`~tseda.statistics.AutocorrelationAnalyzer`).

* ``tseda.decomposition`` — Classical additive / multiplicative decomposition
  (:class:`~tseda.decomposition.ClassicalDecomposer`) and STL decomposition
  (:class:`~tseda.decomposition.STLDecomposer`).

* ``tseda.seasonality`` — Seasonal period detection via FFT periodogram,
  ACF peaks, and combined scoring with Fisher G-test
  (:class:`~tseda.seasonality.SeasonalityDetector`).
