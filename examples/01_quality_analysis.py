"""
01_quality_analysis.py — Data Quality Assessment
=================================================

Demonstrates:
  * Missing value analysis and interpolation
  * Outlier detection (IQR, Z-score, MAD, GESD)
  * Flat-line / duplicate value detection

Usage::

    python examples/01_quality_analysis.py
"""
import numpy as np
import pandas as pd

from tseda import TimeSeries
from tseda.quality import (
    DuplicateDetector,
    MissingValueAnalyzer,
    OutlierDetector,
)

# Load daily sample
df = pd.read_csv("data/sample/daily_sample.csv", parse_dates=["date"])
ts = TimeSeries.from_dataframe(
    df.set_index("date"), column="value", name="daily_price", unit="USD"
)
print(f"Loaded: {ts}\n")

# ---------------------------------------------------------------------------
# 1. Missing value analysis
# ---------------------------------------------------------------------------
print("=" * 55)
print("1. MISSING VALUE ANALYSIS")
print("=" * 55)
missing_ana = MissingValueAnalyzer()
missing_rep = missing_ana.analyze(ts)
print(missing_rep)

print("\n  Filling NaN with linear interpolation...")
ts_filled = missing_ana.interpolate(ts, method="linear")
print(f"  NaN after fill: {ts_filled.n_nan}")

# ---------------------------------------------------------------------------
# 2. Outlier detection
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("2. OUTLIER DETECTION")
print("=" * 55)
det = OutlierDetector()

# IQR
r_iqr = det.iqr(ts, k=3.0)
print(f"\n  IQR (k=3.0)  → {r_iqr.n_outliers} outliers  "
      f"fences: [{r_iqr.lower_bound:.2f}, {r_iqr.upper_bound:.2f}]")

# MAD
r_mad = det.mad(ts, threshold=5.0)
print(f"  MAD (t=5.0)  → {r_mad.n_outliers} outliers")

# GESD
r_gesd = det.gesd(ts, alpha=0.01, max_outliers=15)
print(f"  GESD (α=.01) → {r_gesd.n_outliers} outliers")
if r_gesd.n_outliers > 0:
    print(f"  Detected at  : {list(r_gesd.timestamps[:5].strftime('%Y-%m-%d'))} ...")

# Clean the series
ts_clean = det.remove(ts, r_iqr)
print(f"\n  After IQR remove: {ts_clean.n_nan} NaN (was {ts.n_nan})")

# ---------------------------------------------------------------------------
# 3. Flat-line detection
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("3. FLAT-LINE / DUPLICATE DETECTION")
print("=" * 55)
dup_det = DuplicateDetector()

# Inject a flat-line segment for demonstration
vals_fl = ts.values.copy()
vals_fl[200:205] = 120.0   # 5 identical values
ts_fl = TimeSeries(vals_fl, index=ts.index, name="with_flatline")
r_fl = dup_det.flatline(ts_fl, min_run=3)
print(f"\n  Flat-line runs (min_run=3): {r_fl.n_flatline_runs}")
if r_fl.runs:
    s, e, v = r_fl.runs[0]
    print(f"  First run: positions [{s}:{e}]  value={v:.2f}  length={e-s+1}")

# Near-zero check
r_nz = dup_det.near_zero(ts, min_run=3, threshold=0.5)
print(f"  Near-zero runs (|x|<0.5): {r_nz.n_flatline_runs}")

print("\nQuality analysis complete.")