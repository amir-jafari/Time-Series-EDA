"""
00_load_data.py — Loading data into tseda.TimeSeries
=====================================================

Demonstrates how to construct a TimeSeries from:
  * A CSV file (the most common real-world case)
  * A pandas Series
  * A numpy array + date range
  * A pandas DataFrame column

Usage::

    python examples/00_load_data.py
"""
import numpy as np
import pandas as pd

from tseda import TimeSeries

# ---------------------------------------------------------------------------
# 1. From CSV
# ---------------------------------------------------------------------------
df = pd.read_csv("data/sample/daily_sample.csv", parse_dates=["date"])
ts_csv = TimeSeries.from_dataframe(
    df.set_index("date"), column="value",
    name="stock_price", unit="USD",
    description="3-year synthetic daily series with trend, weekly seasonality, "
                "anomalies, and a structural break at day 600",
)
print("=== From CSV ===")
print(ts_csv)

# ---------------------------------------------------------------------------
# 2. From pandas Series
# ---------------------------------------------------------------------------
s = pd.read_csv(
    "data/sample/monthly_sample.csv", parse_dates=["date"], index_col="date"
)["value"]
ts_series = TimeSeries.from_series(s, name="monthly_revenue", unit="k$")
print("\n=== From pandas Series ===")
print(ts_series)

# ---------------------------------------------------------------------------
# 3. From numpy array + date range
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
idx = pd.date_range("2021-01-01", periods=52, freq="W")
ts_np = TimeSeries(
    rng.standard_normal(52).cumsum() + 50,
    index=idx,
    name="weekly_kpi",
    unit="units",
)
print("\n=== From numpy array ===")
print(ts_np)

# ---------------------------------------------------------------------------
# 4. Basic transforms — all return new TimeSeries
# ---------------------------------------------------------------------------
print("\n=== Transforms ===")
print(f"  diff(1) length  : {ts_csv.diff().n}")
print(f"  resample W n    : {ts_csv.resample('W').n}")
print(f"  slice Q1-2020   : {ts_csv.slice('2020-01-01', '2020-03-31').n} obs")
print(f"  standardize mean: {ts_csv.standardize().values.mean():.6f}")
print(f"  normalize range : [{ts_csv.normalize().values.min():.4f}, "
      f"{ts_csv.normalize().values.max():.4f}]")
print(f"  rolling(30) n   : {ts_csv.rolling(30).n}")