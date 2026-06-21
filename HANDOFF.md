# tseda — Session Handoff

> **Update this file every time a phase is completed or a file is changed.**
> A new Claude Code session must read this file first before touching any code.

---

## Project Identity

| Field | Value |
|-------|-------|
| Package name (import) | `tseda` |
| PyPI name | `tseda` |
| Root directory | `/Users/ajafari/NAS/Deep Learning Book/All_repo_source_file/Packages/TimeSeriesEDA/` |
| Python | 3.12.7 (requires >=3.9) |
| Installed | `pip install -e ".[dev]"` — already done |

---

## Vision (non-negotiable)

Build the **YData-Profiling of time series**: one import, one object, complete understanding of any time series dataset before forecasting.  
Mission: *"Understand your time series before you forecast it."*

Core dependencies only: **numpy, pandas, scipy, matplotlib**.  
Optional: statsmodels (for ADF/KPSS/etc) — import-guarded, never a hard requirement.

---

## Coding Rules (enforce in every session)

1. **One module per session** — confirm with user before starting the next.
2. **NumPy-style docstrings** everywhere (Parameters / Returns / Raises / Examples sections).
3. **No comments** unless the WHY is non-obvious.
4. **All transforms return new `TimeSeries` objects** — `TimeSeries` is effectively immutable.
5. **No ML deps** (no scikit-learn, torch, tensorflow).
6. **Validators live in `tseda/core/validator.py`** — import from there, never duplicate logic.
7. **Tests first** — every new module gets a matching `tests/test_<module>/test_<file>.py`.
8. **Run `pytest` before declaring a phase done** — all tests must pass.
9. **Update this file** (`HANDOFF.md`) when a phase completes or a file changes.

---

## Architecture Map

```
tseda/
├── core/               ✅ DONE — Phase 1
│   ├── types.py        ArrayLike, DatetimeLike, Frequency, AggMethod, DiffMethod
│   ├── validator.py    validate_data_array, validate_datetime_index,
│   │                   validate_positive_int, validate_lags, validate_freq_string
│   └── timeseries.py   TimeSeries class (see API below)
│
├── quality/            🔄 IN PROGRESS — Phase 2
│   ├── missing.py      MissingValueAnalyzer
│   ├── outliers.py     OutlierDetector
│   └── duplicates.py   DuplicateDetector
│
├── statistics/         ⬜ Phase 3
│   ├── descriptive.py  DescriptiveStats
│   ├── stationarity.py StationarityTester
│   └── autocorrelation.py AutocorrelationAnalyzer
│
├── decomposition/      ⬜ Phase 4
│   ├── classical.py    ClassicalDecomposer
│   └── stl.py          STLDecomposer
│
├── seasonality/        ⬜ Phase 5
│   └── detector.py     SeasonalityDetector
│
├── anomaly/            ⬜ Phase 6
│   └── detector.py     AnomalyDetector
│
├── changepoint/        ⬜ Phase 7
│   └── detector.py     ChangepointDetector
│
├── features/           ⬜ Phase 8
│   ├── temporal.py     TemporalFeatures
│   ├── statistical.py  StatisticalFeatures
│   └── spectral.py     SpectralFeatures
│
├── forecastability/    ⬜ Phase 9
│   ├── scorer.py       ForecastabilityScorer
│   └── leakage.py      LeakageDetector
│
├── visualization/      ⬜ Phase 10
│   ├── base.py         plotting utilities / style
│   ├── time_plots.py   plot_series, plot_decomposition, ...
│   ├── distribution_plots.py
│   ├── correlation_plots.py
│   └── diagnostic_plots.py
│
└── report/             ⬜ Phase 11
    ├── html_report.py  HTMLReport
    └── console_report.py ConsoleReport
```

---

## Phase Status

| Phase | Module | Status | Tests | Coverage |
|-------|--------|--------|-------|----------|
| 1 | `core` | ✅ Complete | 78/78 pass | 91 % |
| 2 | `quality` | ✅ Complete | 157/157 pass (all) | 94 % overall |
| 3 | `statistics` | ✅ Complete | 223/223 pass (all) | 86 % overall |
| 4 | `decomposition` | ✅ Complete | 276/276 pass (all) | 86 % overall |
| 5 | `seasonality` | ✅ Complete | 325/325 pass (all) | 87 % overall |
| 6 | `anomaly` | ✅ Complete | 367/367 pass (all) | 88 % overall |
| 7 | `changepoint` | ⬜ Todo | — | — |
| 8 | `features` | ⬜ Todo | — | — |
| 9 | `forecastability` | ⬜ Todo | — | — |
| 10 | `visualization` | ⬜ Todo | — | — |
| 11 | `report` | ⬜ Todo | — | — |

---

## Phase 1 — `tseda/core/` (Complete)

### TimeSeries class API

**Constructor:**
```python
TimeSeries(data, *, index=None, name="value", freq=None, unit=None, description=None)
```
Accepted `data`: `np.ndarray` (1-D), `list`, `tuple`, `pd.Series`.  
`index` required unless `data` is a `pd.Series` with a `DatetimeIndex`.

**Class methods:**
```python
TimeSeries.from_series(series, *, name, freq, unit, description)
TimeSeries.from_arrays(values, index, *, name, freq, unit, description)
TimeSeries.from_dataframe(df, column, *, name, freq, unit, description)
```

**Properties (read-only):**
```
.values       → np.ndarray (copy)   .index       → pd.DatetimeIndex
.n            → int                 .start       → pd.Timestamp
.end          → pd.Timestamp        .duration    → pd.Timedelta
.freq         → str | None          .freq_label  → str
.name         → str                 .unit        → str | None
.description  → str | None          .has_nan     → bool
.n_nan        → int                 .is_regular  → bool
```

**Transforms (each returns a new `TimeSeries`):**
```python
.diff(periods=1, *, method="simple"|"log"|"percent")
.log()
.standardize()
.normalize(*, lower=0.0, upper=1.0)
.rolling(window, *, agg="mean", center=False, min_periods=None)
.apply(func, *, name=None)
.resample(freq, *, agg="mean")
.slice(start=None, end=None)
.copy()
```

**Conversions:** `.to_series()`, `.to_frame()`, `.to_numpy()`

**Dunders:** `__len__`, `__contains__`, `__getitem__` (int or slice), `__eq__`, `__repr__`

### Enums (tseda/core/types.py)
- `Frequency` — pandas freq alias strings (DAILY="D", HOURLY="h", etc.)
- `AggMethod` — MEAN, SUM, MIN, MAX, MEDIAN, FIRST, LAST, STD, VAR, COUNT
- `DiffMethod` — SIMPLE, LOG, PERCENT

### Validators (tseda/core/validator.py)
All raise `TypeError` or `ValueError` with clear messages:
- `validate_data_array(data, *, name="data") → np.ndarray`
- `validate_datetime_index(index, *, name="index") → pd.DatetimeIndex`
- `validate_positive_int(value, *, name="value") → int`
- `validate_lags(lags, n, *, name="lags") → int`
- `validate_freq_string(freq, *, name="freq") → str`

---

## Phase 2 — `tseda/quality/` (Complete)

### Design decisions agreed upon:

**`MissingValueAnalyzer`** (`missing.py`):
- Counts NaN values; computes % missing.
- Detects **timestamp gaps** when `freq` is known (missing rows in the index).
- Returns a `MissingValueReport` dataclass with fields: `n_nan`, `pct_nan`, `n_gaps`, `gap_locations`, `longest_gap`, `missing_pattern` ("MCAR" hint via Little's-style run test).
- Method `interpolate(method="linear"|"forward"|"backward"|"nearest") → TimeSeries` — uses only numpy/pandas, no scipy for basic methods.

**`OutlierDetector`** (`outliers.py`):
- Four methods, all pure numpy/scipy, no ML:
  - `iqr(k=1.5)` — classic Tukey fence.
  - `zscore(threshold=3.0)` — standard Z-score.
  - `mad(threshold=3.5)` — Median Absolute Deviation (robust).
  - `gesd(alpha=0.05, max_outliers=10)` — Generalized ESD test.
- Returns `OutlierReport` dataclass: `mask` (bool array), `indices`, `values`, `method`, `n_outliers`.
- Method `remove() → TimeSeries` replaces outliers with NaN.
- Method `clip() → TimeSeries` clips to fence bounds.

**`DuplicateDetector`** (`duplicates.py`):
- Detects **value** duplicates (consecutive equal values = flat lines), not timestamp duplicates (those are caught by the validator at construction time).
- `flatline(min_run=3) → FlatlineReport`: finds runs of identical consecutive values.
- `near_zero(threshold=1e-8)` variant for floating-point "stuck" sensors.

### Files created
```
tseda/quality/__init__.py          — re-exports all 6 public symbols
tseda/quality/missing.py           — MissingValueReport, MissingValueAnalyzer
tseda/quality/outliers.py          — OutlierReport, OutlierDetector
tseda/quality/duplicates.py        — FlatlineReport, DuplicateDetector
tests/test_quality/__init__.py
tests/test_quality/test_missing.py  — 23 tests
tests/test_quality/test_outliers.py — 23 tests
tests/test_quality/test_duplicates.py — 19 tests
docs/api/quality.rst
```

### Key implementation notes
- `MissingValueAnalyzer.interpolate(method="spline")` requires scipy; import-guarded.
- `OutlierDetector.gesd()` requires scipy for t-distribution CDF; import-guarded.
- `OutlierDetector.clip()` raises `ValueError` for GESD results (no bounds).
- `DuplicateDetector.near_zero()` finds runs where `|x| <= threshold` directly
  (does NOT reuse `flatline` internally — different semantics).
- All `_ts_with_spike()` test fixtures use Normal(0,1) background so MAD ≠ 0.

---

## Phase 3 — `tseda/statistics/` (Next)

### Planned files
```
tseda/statistics/__init__.py
tseda/statistics/descriptive.py    — DescriptiveStats dataclass + analyzer
tseda/statistics/stationarity.py   — StationarityTester (ADF, KPSS, PP — numpy/scipy impl)
tseda/statistics/autocorrelation.py— AutocorrelationAnalyzer (ACF, PACF, Ljung-Box)
tests/test_statistics/test_descriptive.py
tests/test_statistics/test_stationarity.py
tests/test_statistics/test_autocorrelation.py
docs/api/statistics.rst
```

### Design decisions for Phase 3
- ADF test: pure numpy implementation (OLS regression + critical values lookup table).
  Optional statsmodels path if installed: `from statsmodels.tsa.stattools import adfuller`.
- KPSS test: same pattern — native numpy fallback + optional statsmodels.
- ACF / PACF: pure numpy (Durbin-Levinson / Yule-Walker equations).
- Ljung-Box: pure numpy + scipy.stats.chi2.
- All result classes are frozen dataclasses.

---

## Test fixtures (tests/conftest.py)

```python
ts_daily    # 365 pts, daily, no NaN, regular
ts_hourly   # 720 pts, hourly, no NaN, regular
ts_monthly  # 36 pts, monthly MS, no NaN, regular
ts_with_nan # 200 pts, daily, 20 NaN (10 %)
ts_short    # 5 pts, daily [10,20,30,40,50]
ts_irregular# 50 pts, random hourly gaps
```
All use `rng = np.random.default_rng(42)` (session-scoped) for reproducibility.

---

## Key files to read at session start

Before writing any code in a new session, read these files in order:

1. `HANDOFF.md` ← this file
2. `tseda/core/timeseries.py` — understand the `TimeSeries` API
3. `tseda/core/validator.py` — understand what validators exist
4. `tests/conftest.py` — understand available fixtures
5. The target module's `__init__.py` (currently empty stubs)

---

## How to run tests

```bash
cd "/Users/ajafari/NAS/Deep Learning Book/All_repo_source_file/Packages/TimeSeriesEDA"
python -m pytest tests/ -v --tb=short
```

To run only one module's tests:
```bash
python -m pytest tests/test_quality/ -v --tb=short
```

---

## Sphinx docs structure

`docs/conf.py` is configured with:
- Extensions: `numpydoc`, `sphinx.ext.napoleon`, `sphinx.ext.autodoc`,
  `sphinx_autodoc_typehints`, `sphinx_copybutton`, `sphinx.ext.mathjax`,
  `sphinx.ext.githubpages`
- **Theme: `sphinx_rtd_theme` v3.1** (Read the Docs) — matches reference site
  https://amir-jafari.github.io/TimeSeries/
- Custom CSS: `docs/_static/css/custom.css` — One Dark code blocks, brand colours,
  polished table/admonition styles, responsive layout
- Sidebar: dark blue-grey (`#2c3e50`), 4-level deep navigation, "Edit on GitHub" links
- Intersphinx: numpy, pandas, scipy, matplotlib, python

**Do NOT switch back to `pydata_sphinx_theme`.** User explicitly chose RTD theme.

Each new module needs a `docs/api/<module>.rst` file added to the toctree.

---

## Phase 3 — `tseda/statistics/` (Complete)

### Files created
```
tseda/statistics/__init__.py           — re-exports all 6 public symbols
tseda/statistics/descriptive.py        — DescriptiveStats, DescriptiveAnalyzer
tseda/statistics/stationarity.py       — StationarityResult, StationarityTester
tseda/statistics/autocorrelation.py    — AutocorrelationResult, AutocorrelationAnalyzer
tests/test_statistics/test_descriptive.py   — 20 tests
tests/test_statistics/test_stationarity.py  — 18 tests
tests/test_statistics/test_autocorrelation.py — 18 tests
docs/api/statistics.rst
```

### Key implementation notes
- **Dual-path strategy**: statsmodels fast path (ADF, KPSS via `adfuller`, `kpss`);
  pure numpy/scipy fallback when statsmodels absent. Both produce `StationarityResult`.
- **PP test** (`StationarityTester.pp`) requires statsmodels — raises `ImportError` if absent.
- **ACF/PACF**: pure numpy (Durbin-Levinson recursion). No statsmodels dependency.
- **Ljung-Box**: pure numpy + `scipy.stats.chi2.sf`.
- `stationarity.py` has 45 % coverage because the native fallback paths are only
  exercised without statsmodels — this is expected and intentional.
- White-noise Ljung-Box test uses `seed=0, n=500` to guarantee p≈0.62 reliably.

---

## Phase 4 — `tseda/decomposition/` (Complete)

### Files created
```
tseda/decomposition/__init__.py        — re-exports 3 public symbols
tseda/decomposition/classical.py       — DecompositionResult, ClassicalDecomposer
tseda/decomposition/stl.py             — STLDecomposer
tests/test_decomposition/__init__.py
tests/test_decomposition/test_classical.py — 34 tests
tests/test_decomposition/test_stl.py       — 19 tests
docs/api/decomposition.rst
```

### Key implementation notes
- **`DecompositionResult`** lives in `classical.py` and is imported by `stl.py`.
  Contains: `original`, `trend`, `seasonal`, `residual` (all TimeSeries),
  plus `period`, `model`, `method`, `strength_trend`, `strength_seasonal`, `n_obs_used`.
  Methods: `to_dataframe()`, `summary()`.
- **`ClassicalDecomposer`**: pure numpy/pandas.
  - Even-period centered MA: trailing `MA(p)` → trailing `MA(2)` → shift `-(period//2)`.
  - Additive: seasonal normalised to sum=0 over one period.
  - Multiplicative: seasonal normalised to mean=1; raises `ValueError` for negative trends.
  - Trend has NaN at edges; seasonal covers full length; residual NaN where trend is NaN.
- **Reconstruction identity** verified in tests: `T+S+R == y` (additive), `T×S×R == y` (mult).
- **Strength metrics**: Wang et al. (2006) — `max(0, 1 - Var(R)/Var(C+R))`.
- **`STLDecomposer`**: statsmodels `STL` primary path; `stl-fallback` (2-pass Savitzky-Golay)
  when statsmodels absent. STL always produces additive decomposition.
  STL trend covers full length (no NaN edges).
- **NaN handling in STL**: original NaN positions are re-applied to components
  after statsmodels fitting (which requires NaN-free input filled via `ffill+bfill`).
- `stl.py` coverage is 64 % — the fallback path (savgol) is only hit without statsmodels.

---

## Phase 5 — `tseda/seasonality/` (Complete)

### Files created
```
tseda/seasonality/__init__.py          — re-exports 2 public symbols
tseda/seasonality/detector.py          — SeasonalityReport, SeasonalityDetector
tests/test_seasonality/__init__.py
tests/test_seasonality/test_detector.py — 49 tests
docs/api/seasonality.rst
```

### Key implementation notes
- **Three detection methods**: `"periodogram"`, `"acf"`, `"combined"` (default).
- **Periodogram path**: Hann window + `scipy.fft` + `scipy.signal.find_peaks`;
  peaks rounded to nearest integer period; normalised to [0,1] by max power.
- **ACF path**: biased ACF in numpy + `find_peaks` above Bartlett 95% CI
  (`±1.96/√n`); raw ACF values normalised to [0,1].
- **Combined scoring**: agreement bonus — when both methods detect the same period,
  score = min(1.0, mean × 1.2); single-method score gets 0.9 discount.
- **Fisher G-test** always computed on periodogram; contributes to `is_seasonal`.
- `is_seasonal` = Fisher G significant OR ≥1 ACF peak significant, AND dom_score > 0.05.
- **NaN handling**: filled by linear interpolation before FFT/ACF; original NaN
  count subtracted from `n_obs`.
- Pre-processing: `scipy.signal.detrend(type="linear")` before FFT to suppress
  spectral leakage from trend and mean.
- `SeasonalityReport.dominant_period` is `None` when `is_seasonal=False`
  (safe for downstream use with `DecompositionResult.period`).
- `test_period(ts, period)` is a convenience wrapper that runs combined detection
  with `top_k=20` and looks up a specific period in the results.

---

## Phase 6 — `tseda/anomaly/` (Complete)

### Files created
```
tseda/anomaly/__init__.py          — re-exports 2 public symbols
tseda/anomaly/detector.py          — AnomalyReport, AnomalyDetector
tests/test_anomaly/__init__.py
tests/test_anomaly/test_detector.py — 42 tests
docs/api/anomaly.rst
```

### Key implementation notes
- `AnomalyReport`: `mask`, `indices`, `timestamps`, `values`, `scores` (0→1),
  `method`, `n_anomalies`.
- `rolling_iqr(window, k)`: pandas rolling Q1/Q3 + IQR fence; score = excess/IQR.
- `rolling_z(window, threshold)`: pandas rolling mean/std; score = (|z|−threshold)/threshold.
- `stl_residual(period, residual_method, k)`: delegates to `STLDecomposer`; then
  flags residuals via IQR/MAD/Z. Period auto-inferred from `ts.freq`.
- `gesd(alpha, max_outliers)`: re-uses `quality.OutlierDetector.gesd`; scores = |z|/max_z.
- `remove(ts, report)` → NaN at anomaly positions.
- `label(ts, report)` → 0/1 TimeSeries named `"{name}_anomaly_label"`.
- All rolling methods use `center=True` by default (look at surrounding context).

## Git / GitHub

- Remote: `git@github.com:amir-jafari/Time-Series-EDA.git`
  (HTTPS: `https://github.com/amir-jafari/Time-Series-EDA.git`)
- Branch: `main`
- Push command (from repo root): `git push origin main`
- After every phase, stage all new/changed files and push.
- `.gitignore` tracks `docs/api/*.rst` (manually written, not auto-generated).

---

## Phase 7 — `tseda/changepoint/` (Next)

### Planned files
```
tseda/changepoint/__init__.py
tseda/changepoint/detector.py   — ChangepointDetector
tests/test_changepoint/__init__.py
tests/test_changepoint/test_detector.py
docs/api/changepoint.rst
```

### Design decisions for Phase 7
- Detect structural breaks (mean shift, variance shift, trend change).
- Methods (pure numpy/scipy):
  1. **CUSUM** — Cumulative sum control chart for mean shift.
  2. **BOCPD-lite** — Bayesian online changepoint detection (simplified).
  3. **Variance ratio** — sliding-window F-test for variance shifts.
- `ChangepointReport`: `changepoints` (list of timestamps), `n_changepoints`,
  `scores` (continuous score per observation), `method`.
- Pure numpy fallback for all methods.

---

*Last updated: Phase 6 complete + pushed to GitHub — 2026-06-21*