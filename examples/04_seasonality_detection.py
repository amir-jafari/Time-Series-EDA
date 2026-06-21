"""
04_seasonality_detection.py — Seasonality Detection
=====================================================

Demonstrates:
  * Period detection via periodogram, ACF, and combined methods
  * Fisher G-test for spectral significance
  * Testing a specific period

Usage::

    python examples/04_seasonality_detection.py
"""
import pandas as pd

from tseda import TimeSeries
from tseda.seasonality import SeasonalityDetector

det = SeasonalityDetector()

# ----- Daily data (should find weekly period=7) -----
print("=" * 55)
print("DAILY DATA — expecting weekly period = 7")
print("=" * 55)
df_d = pd.read_csv("data/sample/daily_sample.csv", parse_dates=["date"])
ts_d = TimeSeries.from_dataframe(df_d.set_index("date"), column="value",
                                 name="daily", unit="USD")

r_d = det.detect(ts_d, method="combined", top_k=5)
print(r_d.summary())
print(f"  Fisher G stat  : {r_d.fisher_g_stat:.4f}   p = {r_d.fisher_p_value:.4f}")
print(f"  Top 5 periods  : {[(p, round(s,3)) for p,s in r_d.candidate_periods[:5]]}")

# Explicitly test period=7
t7 = det.test_period(ts_d, period=7)
print(f"\n  test_period(7) → detected={t7['detected']}  "
      f"strength={t7['strength']:.4f}  "
      f"periodogram={t7['periodogram_detected']}  "
      f"acf={t7['acf_detected']}")

# ----- Monthly data (should find annual period=12) -----
print("\n" + "=" * 55)
print("MONTHLY DATA — expecting annual period = 12")
print("=" * 55)
df_m = pd.read_csv("data/sample/monthly_sample.csv", parse_dates=["date"])
ts_m = TimeSeries.from_dataframe(df_m.set_index("date"), column="value",
                                 name="monthly", unit="k$")

r_m = det.detect(ts_m, method="combined")
print(r_m.summary())

# Compare methods
print("\n  Method comparison:")
for method in ("periodogram", "acf", "combined"):
    r = det.detect(ts_m, method=method)
    print(f"    {method:<12} dominant={r.dominant_period}  "
          f"is_seasonal={r.is_seasonal}")

print("\nSeasonality detection complete.")
