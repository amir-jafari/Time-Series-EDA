"""
02_statistics_analysis.py — Statistical Analysis
=================================================

Demonstrates:
  * Descriptive statistics
  * Stationarity tests (ADF, KPSS, summary)
  * ACF / PACF / Ljung-Box autocorrelation

Usage::

    python examples/02_statistics_analysis.py
"""
import numpy as np
import pandas as pd

from tseda import TimeSeries
from tseda.statistics import (
    AutocorrelationAnalyzer,
    DescriptiveAnalyzer,
    StationarityTester,
)

# Load
df = pd.read_csv("data/sample/daily_sample.csv", parse_dates=["date"])
ts = TimeSeries.from_dataframe(
    df.set_index("date"), column="value", name="daily_price", unit="USD"
)

# ---------------------------------------------------------------------------
# 1. Descriptive statistics
# ---------------------------------------------------------------------------
print("=" * 55)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 55)
desc = DescriptiveAnalyzer().analyze(ts)
print(desc)
print(f"\n  Quantiles:")
for q, v in desc.quantiles.items():
    print(f"    P{int(q*100):>3d} = {v:>10.4f}")

# ---------------------------------------------------------------------------
# 2. Stationarity
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("2. STATIONARITY TESTS")
print("=" * 55)
tester = StationarityTester()

adf = tester.adf(ts)
print(f"\n  ADF  stat={adf.statistic:.4f}  p={adf.p_value:.4f}  "
      f"→ {'STATIONARY' if adf.is_stationary else 'NON-STATIONARY'}")

kpss = tester.kpss(ts)
print(f"  KPSS stat={kpss.statistic:.4f}  p={kpss.p_value:.4f}  "
      f"→ {'STATIONARY' if kpss.is_stationary else 'NON-STATIONARY'}")

print("\n" + tester.summary(ts))

# First-differenced series
ts_diff = ts.diff()
adf_d = tester.adf(ts_diff)
print(f"  ADF on diff(1): p={adf_d.p_value:.4f}  "
      f"→ {'STATIONARY' if adf_d.is_stationary else 'NON-STATIONARY'}")

# ---------------------------------------------------------------------------
# 3. Autocorrelation
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("3. AUTOCORRELATION (ACF / PACF / Ljung-Box)")
print("=" * 55)
acf_ana = AutocorrelationAnalyzer()
acf_res = acf_ana.analyze(ts, lags=30)

print(f"\n  n_obs         : {acf_res.n_obs}")
print(f"  is_white_noise: {acf_res.is_white_noise}")
print(f"  ACF[1:5]      : {acf_res.acf[1:6].round(4).tolist()}")
print(f"  PACF[1:5]     : {acf_res.pacf[1:6].round(4).tolist()}")

sig_lags = acf_ana.significant_lags(acf_res, which="acf")
print(f"  Significant ACF lags: {sig_lags.tolist()}")

lb_lag7 = acf_res.lb_pvalue[6]  # Ljung-Box at lag 7
print(f"  Ljung-Box p (lag=7) : {lb_lag7:.4f} "
      f"→ {'white noise' if lb_lag7 > 0.05 else 'autocorrelated'}")

print("\nStatistics analysis complete.")