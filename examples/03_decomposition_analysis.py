"""
03_decomposition_analysis.py — Time Series Decomposition
=========================================================

Demonstrates:
  * Classical decomposition (additive and multiplicative)
  * STL decomposition
  * Inspecting trend / seasonal / residual components

Usage::

    python examples/03_decomposition_analysis.py
"""
import pandas as pd

from tseda import TimeSeries
from tseda.decomposition import ClassicalDecomposer, STLDecomposer

# Load monthly (clear annual seasonality, good for decomposition)
df = pd.read_csv("data/sample/monthly_sample.csv", parse_dates=["date"])
ts = TimeSeries.from_dataframe(
    df.set_index("date"), column="value", name="monthly_revenue", unit="k$"
)
print(f"Loaded: {ts}\n")

# ---------------------------------------------------------------------------
# 1. Classical — additive
# ---------------------------------------------------------------------------
print("=" * 55)
print("1. CLASSICAL DECOMPOSITION (Additive, period=12)")
print("=" * 55)
classical = ClassicalDecomposer()
r_add = classical.decompose(ts, period=12, model="additive")
print(r_add.summary())
print(f"  Trend  range : [{r_add.trend.values[~__import__('numpy').isnan(r_add.trend.values)].min():.2f}, "
      f"{r_add.trend.values[~__import__('numpy').isnan(r_add.trend.values)].max():.2f}]")
print(f"  Seasonal NaN : {r_add.seasonal.has_nan}")
print(f"  Residual NaN : {r_add.residual.n_nan} positions")
df_add = r_add.to_dataframe()
print(f"\n  Component DataFrame shape: {df_add.shape}")
print(df_add.head(3).round(3).to_string())

# ---------------------------------------------------------------------------
# 2. Classical — multiplicative
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("2. CLASSICAL DECOMPOSITION (Multiplicative, period=12)")
print("=" * 55)
r_mul = classical.decompose(ts, period=12, model="multiplicative")
print(r_mul.summary())

# ---------------------------------------------------------------------------
# 3. STL
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("3. STL DECOMPOSITION (period=12, robust=True)")
print("=" * 55)
stl = STLDecomposer()
r_stl = stl.decompose(ts, period=12, robust=True)
print(r_stl.summary())
print(f"  Method        : {r_stl.method}")
print(f"  Trend has NaN : {r_stl.trend.has_nan}  (STL fills edges)")
print(f"  Strength trend    : {r_stl.strength_trend:.4f}")
print(f"  Strength seasonal : {r_stl.strength_seasonal:.4f}")

# Compare strengths
print("\n  Additive vs STL comparison:")
print(f"  {'Method':<12} {'Trend':>10} {'Seasonal':>10}")
print(f"  {'classical':<12} {r_add.strength_trend:>10.4f} {r_add.strength_seasonal:>10.4f}")
print(f"  {'STL':<12} {r_stl.strength_trend:>10.4f} {r_stl.strength_seasonal:>10.4f}")

print("\nDecomposition analysis complete.")