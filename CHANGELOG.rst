Changelog
=========

All notable changes to ``tseda`` are documented here.
This project follows `Semantic Versioning <https://semver.org/>`_.

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
