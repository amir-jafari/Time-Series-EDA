"""
Example 08 — Forecastability scoring and leakage detection.

Demonstrates:
* ForecastabilityScorer.score()  — overall + sub-scores, model recommendation
* LeakageDetector.check()        — target leakage, temporal leakage, no leakage

Run:
    python examples/08_forecastability.py
    python examples/08_forecastability.py data/sample/Series_G_Airline_Passengers.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tseda import TimeSeries
from tseda.forecastability import ForecastabilityScorer, LeakageDetector


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Load series
# ---------------------------------------------------------------------------

def load_series(path: str | None) -> TimeSeries:
    if path is None:
        rng = np.random.default_rng(0)
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        t = np.arange(n)
        seas = np.sin(2 * np.pi * t / 7) * 4
        vals = 100 + 0.05 * t + seas + rng.standard_normal(n) * 0.8
        return TimeSeries(vals, index=idx, name="synthetic_daily")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    col = df.columns[0]
    return TimeSeries(df[col].values, index=df.index, name=col)


# ---------------------------------------------------------------------------
# 2. Forecastability scoring
# ---------------------------------------------------------------------------

def run_forecastability(ts: TimeSeries) -> None:
    _section("Forecastability Scoring")
    scorer = ForecastabilityScorer()
    report = scorer.score(ts)

    print(f"\nSeries      : {ts.name}  (n={ts.n}, freq={ts.freq})")
    print(f"Overall score: {report.score:.1f} / 100")
    print()
    print(f"{'Sub-score':<22} {'Score':>7}   Weight")
    weights = {"data_quality": 20, "stationarity": 15, "signal_to_noise": 20,
               "autocorrelation": 15, "sample_size": 15, "regularity": 15}
    for k, v in report.sub_scores.items():
        bar = "█" * int(v / 5)
        print(f"  {k:<20} {v:>6.1f}   ({weights[k]}%)  {bar}")

    print()
    print(f"Recommended model  : {report.recommended_model}")
    print(f"Recommended diff   : d={report.recommended_diff}")
    print(f"Recommended period : {report.recommended_period}")
    print(f"Missing values     : {report.pct_missing:.2f} %")
    print(f"IQR outliers       : {report.pct_outlier:.2f} %")
    print(f"Is stationary      : {report.is_stationary}")


# ---------------------------------------------------------------------------
# 3. Leakage detection
# ---------------------------------------------------------------------------

def run_leakage(ts: TimeSeries) -> None:
    _section("Leakage Detection")
    det = LeakageDetector()
    y = ts.values
    n = ts.n
    idx = ts.index

    # --- Build feature matrix ---
    rng = np.random.default_rng(42)

    lag1 = np.roll(y, 1).astype(float)
    lag1[0] = np.nan

    lag7 = np.roll(y, 7).astype(float)
    lag7[:7] = np.nan

    # Simulated "future leak": rolling mean that inadvertently includes future
    future_mean = np.roll(y, -3).astype(float)
    future_mean[-3:] = np.nan

    target_copy = y.copy()

    random_feat = rng.standard_normal(n)

    features = pd.DataFrame(
        {
            "lag_1":       lag1,
            "lag_7":       lag7,
            "future_mean": future_mean,
            "target_copy": target_copy,
            "random":      random_feat,
        },
        index=idx,
    )

    report = det.check(ts, horizon=5, features_df=features)

    print(f"\nFeatures examined : {report.n_features}")
    print(f"Horizon           : {report.horizon} steps")
    print()

    if report.has_target_leakage:
        print(f"[!] TARGET LEAKAGE detected in: {report.target_leakage_columns}")
        for col, corr in report.target_leakage_correlations.items():
            print(f"    {col}: r={corr:.4f} at lag 0")
    else:
        print("[ok] No target leakage detected.")

    if report.has_temporal_leakage:
        print(f"[!] TEMPORAL LEAKAGE detected in: {report.temporal_leakage_columns}")
    else:
        print("[ok] No temporal leakage detected.")

    print()
    print("Peak-correlation lags per feature (positive = future):")
    for col, lag in report.temporal_peak_lags.items():
        direction = "future ⚠" if lag > 0 else ("present" if lag == 0 else "past")
        print(f"  {col:<18} peak lag = {lag:+d}  ({direction})")


# ---------------------------------------------------------------------------
# 4. No-features check
# ---------------------------------------------------------------------------

def run_no_features(ts: TimeSeries) -> None:
    _section("No-Features Baseline Check")
    det = LeakageDetector()
    report = det.check(ts, horizon=5)
    print(f"\nWarning: {report.warnings[0]}")
    print(f"n_features = {report.n_features}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    ts = load_series(csv_path)

    run_forecastability(ts)
    run_leakage(ts)
    run_no_features(ts)

    print("\nDone.")