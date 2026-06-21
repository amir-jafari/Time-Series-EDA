"""
06_changepoint_detection.py — Structural Break Detection
=========================================================

Demonstrates:
  * CUSUM for mean-shift detection
  * Binary segmentation (multiple changepoints)
  * Variance ratio test for volatility shifts
  * Segment analysis table

Usage::

    python examples/06_changepoint_detection.py
"""
import pandas as pd

from tseda import TimeSeries
from tseda.changepoint import ChangepointDetector

det = ChangepointDetector()

# The daily sample has a known level shift at ~day 600
df = pd.read_csv("data/sample/daily_sample.csv", parse_dates=["date"])
ts = TimeSeries.from_dataframe(df.set_index("date"), column="value",
                               name="daily_price", unit="USD")
print(f"Loaded: {ts.n} obs  [{ts.start.date()} → {ts.end.date()}]\n")

# ---------------------------------------------------------------------------
# 1. CUSUM
# ---------------------------------------------------------------------------
print("=" * 55)
print("1. CUSUM  (threshold=5.0, drift=0.5)")
print("=" * 55)
r_cusum = det.cusum(ts, threshold=5.0, drift=0.5)
print(f"  Changepoints detected : {r_cusum.n_changepoints}")
if r_cusum.n_changepoints > 0:
    for pos, ts_stamp in zip(r_cusum.changepoints[:5], r_cusum.timestamps[:5]):
        print(f"    pos={pos:>4d}  date={ts_stamp.date()}  score={r_cusum.scores[pos]:.4f}")

# ---------------------------------------------------------------------------
# 2. Binary segmentation
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("2. BINARY SEGMENTATION  (auto penalty)")
print("=" * 55)
r_bs = det.binary_segmentation(ts)
print(f"  Changepoints detected : {r_bs.n_changepoints}")
if r_bs.n_changepoints > 0:
    for pos, ts_stamp in zip(r_bs.changepoints, r_bs.timestamps):
        print(f"    pos={pos:>4d}  date={ts_stamp.date()}  score={r_bs.scores[pos]:.4f}")

# Segment statistics
print("\n  Segment statistics:")
seg_df = det.segment(ts, r_bs)
print(seg_df[["segment", "start", "end", "n_obs", "mean", "std"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Variance ratio
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("3. VARIANCE RATIO  (window=60, alpha=0.01)")
print("=" * 55)
r_vr = det.variance_ratio(ts, window=60, alpha=0.01)
print(f"  Changepoints detected : {r_vr.n_changepoints}")
if r_vr.n_changepoints > 0:
    for pos, ts_stamp in zip(r_vr.changepoints[:5], r_vr.timestamps[:5]):
        print(f"    pos={pos:>4d}  date={ts_stamp.date()}  score={r_vr.scores[pos]:.4f}")

# ---------------------------------------------------------------------------
# 4. Segment labels
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("4. SEGMENT LABELS")
print("=" * 55)
labels = r_bs.segment_labels(ts.n)
import numpy as np
unique_segs = set(labels)
print(f"  Segments : {len(unique_segs)}")
for seg in sorted(unique_segs):
    mask = labels == seg
    vals = ts.values[mask]
    finite = vals[~np.isnan(vals)]
    print(f"    Segment {seg}: n={mask.sum():<4d}  mean={finite.mean():.3f}")

print("\nChangepoint detection complete.")