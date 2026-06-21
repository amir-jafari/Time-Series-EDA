"""
05_anomaly_detection.py — Anomaly Detection
============================================

Demonstrates:
  * Rolling IQR and rolling Z-score detection
  * STL-residual anomaly detection
  * GESD global test
  * Removing and labelling anomalies

Usage::

    python examples/05_anomaly_detection.py
"""
import pandas as pd

from tseda import TimeSeries
from tseda.anomaly import AnomalyDetector

det = AnomalyDetector()

df = pd.read_csv("data/sample/daily_sample.csv", parse_dates=["date"])
ts = TimeSeries.from_dataframe(df.set_index("date"), column="value",
                               name="daily_price", unit="USD")
print(f"Loaded: {ts.n} obs,  {ts.n_nan} NaN\n")

# ---------------------------------------------------------------------------
# 1. Rolling IQR
# ---------------------------------------------------------------------------
print("=" * 55)
print("1. ROLLING IQR  (window=30, k=2.5)")
print("=" * 55)
r_iqr = det.rolling_iqr(ts, window=30, k=2.5)
print(f"  Anomalies detected : {r_iqr.n_anomalies}")
if r_iqr.n_anomalies > 0:
    top3 = r_iqr.timestamps[r_iqr.scores.argsort()[::-1]][:3]
    print(f"  Top-3 timestamps   : {list(top3.strftime('%Y-%m-%d'))}")

# ---------------------------------------------------------------------------
# 2. Rolling Z-score
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("2. ROLLING Z-SCORE  (window=30, threshold=3.0)")
print("=" * 55)
r_z = det.rolling_z(ts, window=30, threshold=3.0)
print(f"  Anomalies detected : {r_z.n_anomalies}")

# ---------------------------------------------------------------------------
# 3. STL residual
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("3. STL-RESIDUAL  (period=7, method=iqr, k=3.0)")
print("=" * 55)
r_stl = det.stl_residual(ts, period=7, residual_method="iqr", k=3.0)
print(f"  Anomalies detected : {r_stl.n_anomalies}")
print(f"  Method label       : {r_stl.method}")

# ---------------------------------------------------------------------------
# 4. GESD
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("4. GESD  (alpha=0.01, max_outliers=15)")
print("=" * 55)
r_gesd = det.gesd(ts, alpha=0.01, max_outliers=15)
print(f"  Anomalies detected : {r_gesd.n_anomalies}")

# ---------------------------------------------------------------------------
# 5. Repair
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("5. REPAIR")
print("=" * 55)
ts_clean  = det.remove(ts, r_iqr)
ts_labels = det.label(ts, r_iqr)
print(f"  After remove : {ts_clean.n_nan} NaN (was {ts.n_nan})")
print(f"  Label sum    : {int(ts_labels.values.sum())} marked anomalies")
print(f"  Label name   : {ts_labels.name}")

# ---------------------------------------------------------------------------
# 6. Summary comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("6. METHOD COMPARISON")
print("=" * 55)
print(f"  {'Method':<20} {'Anomalies':>10}")
for name, r in [("rolling_iqr", r_iqr), ("rolling_z", r_z),
                ("stl_residual", r_stl), ("gesd", r_gesd)]:
    print(f"  {name:<20} {r.n_anomalies:>10}")

print("\nAnomaly detection complete.")