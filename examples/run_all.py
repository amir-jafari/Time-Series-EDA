"""
run_all.py — Run the complete tseda analysis pipeline on any CSV file
======================================================================

Usage::

    python examples/run_all.py <csv_file> [options]

Arguments
---------
csv_file
    Path to a CSV with at least a date column and a value column.

Options
-------
--date-col   Name of the date column (default: "date")
--value-col  Name of the value column (default: "value")
--name       Series name (default: filename stem)
--unit       Unit label, e.g. "USD" (default: "")
--freq       Pandas offset alias if auto-inference fails, e.g. "D", "MS"
--period     Seasonal period to use (default: auto-detect)
--out        Directory for output summary (default: stdout only)

Example::

    python examples/run_all.py data/sample/daily_sample.csv \\
        --date-col date --value-col value --period 7

"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure tseda is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tseda import TimeSeries
from tseda.anomaly import AnomalyDetector
from tseda.decomposition import STLDecomposer
from tseda.quality import MissingValueAnalyzer, OutlierDetector
from tseda.seasonality import SeasonalityDetector
from tseda.statistics import (
    AutocorrelationAnalyzer,
    DescriptiveAnalyzer,
    StationarityTester,
)


def _hr(title: str, width: int = 60) -> str:
    return f"\n{'='*width}\n  {title}\n{'='*width}"


def run_pipeline(
    csv_path: str,
    *,
    date_col: str = "date",
    value_col: str = "value",
    name: str | None = None,
    unit: str = "",
    freq: str | None = None,
    period: int | None = None,
) -> dict:
    """Run the full EDA pipeline and return a results dict."""
    path  = Path(csv_path)
    _name = name or path.stem

    # ── Load ──────────────────────────────────────────────────────────
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    ts = TimeSeries.from_dataframe(df, column=value_col, name=_name,
                                   unit=unit, freq=freq)
    print(_hr(f"tseda EDA Pipeline  →  {_name}"))
    print(ts)

    results: dict = {"ts": ts}

    # ── Quality ───────────────────────────────────────────────────────
    print(_hr("QUALITY"))
    mva = MissingValueAnalyzer()
    missing = mva.analyze(ts)
    print(missing)
    results["missing"] = missing

    od  = OutlierDetector()
    out_iqr  = od.iqr(ts, k=3.0)
    out_mad  = od.mad(ts, threshold=4.5)
    print(f"  IQR (k=3.0)  outliers : {out_iqr.n_outliers}")
    print(f"  MAD (t=4.5)  outliers : {out_mad.n_outliers}")
    results["outliers_iqr"] = out_iqr

    # ── Statistics ────────────────────────────────────────────────────
    print(_hr("STATISTICS — DESCRIPTIVE"))
    desc = DescriptiveAnalyzer().analyze(ts)
    print(desc)
    results["descriptive"] = desc

    print(_hr("STATISTICS — STATIONARITY"))
    tst = StationarityTester()
    print(tst.summary(ts))
    results["stationarity_adf"] = tst.adf(ts)

    print(_hr("STATISTICS — AUTOCORRELATION"))
    lags = min(40, ts.n // 3)
    acf  = AutocorrelationAnalyzer().analyze(ts, lags=lags)
    print(f"  is_white_noise : {acf.is_white_noise}")
    print(f"  Significant ACF lags : {AutocorrelationAnalyzer().significant_lags(acf).tolist()[:10]}")
    results["acf"] = acf

    # ── Seasonality ───────────────────────────────────────────────────
    print(_hr("SEASONALITY DETECTION"))
    sd   = SeasonalityDetector()
    seas = sd.detect(ts, method="combined", top_k=5)
    print(seas.summary())
    results["seasonality"] = seas

    # Infer period for decomposition
    _period = period or seas.dominant_period
    if _period is None:
        from tseda.decomposition.classical import _default_period
        _period = _default_period(ts.freq)

    # ── Decomposition ─────────────────────────────────────────────────
    if _period and ts.n >= 2 * _period:
        print(_hr(f"DECOMPOSITION  (period={_period})"))
        dec = STLDecomposer().decompose(ts, period=_period, robust=True)
        print(dec.summary())
        results["decomposition"] = dec
    else:
        print(f"\n  [Skipping decomposition — period={_period}, n={ts.n}]")

    # ── Anomaly ───────────────────────────────────────────────────────
    print(_hr("ANOMALY DETECTION"))
    det  = AnomalyDetector()
    w    = min(30, ts.n // 5)
    a_iqr = det.rolling_iqr(ts, window=w, k=2.5)
    a_stl = det.stl_residual(ts, period=_period) if _period else None
    print(f"  Rolling IQR (w={w})  : {a_iqr.n_anomalies} anomalies")
    if a_stl:
        print(f"  STL residual IQR     : {a_stl.n_anomalies} anomalies")
    results["anomalies"] = a_iqr

    # ── Summary ───────────────────────────────────────────────────────
    print(_hr("SUMMARY"))
    dom_p  = seas.dominant_period if seas.is_seasonal else "none detected"
    adf_r  = results["stationarity_adf"]
    print(textwrap.dedent(f"""
      Series        : {_name}
      Observations  : {ts.n:,}
      Period        : {ts.start.date()} → {ts.end.date()}
      Frequency     : {ts.freq or 'irregular'}
      ─────────────────────────────────
      Missing (NaN) : {missing.n_nan} ({missing.pct_nan:.1f}%)
      Index gaps    : {missing.n_gaps if missing.n_gaps >= 0 else 'n/a'}
      Outliers (IQR): {out_iqr.n_outliers}
      ─────────────────────────────────
      Mean / Std    : {desc.mean:.4g} / {desc.std:.4g}
      Skewness      : {desc.skewness:.4f}
      Excess kurt.  : {desc.kurtosis:.4f}
      ─────────────────────────────────
      Stationary?   : {'Yes' if adf_r.is_stationary else 'No'} (ADF p={adf_r.p_value:.4f})
      White noise?  : {acf.is_white_noise}
      ─────────────────────────────────
      Seasonal?     : {seas.is_seasonal}  dominant period = {dom_p}
      Anomalies     : {a_iqr.n_anomalies} (rolling IQR w={w})
    """).strip())

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the tseda EDA pipeline on a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv_file", help="Path to input CSV")
    parser.add_argument("--date-col",  default="date",  help="Date column name")
    parser.add_argument("--value-col", default="value", help="Value column name")
    parser.add_argument("--name",      default=None,    help="Series name")
    parser.add_argument("--unit",      default="",      help="Unit label")
    parser.add_argument("--freq",      default=None,    help="Pandas freq alias")
    parser.add_argument("--period",    default=None, type=int, help="Seasonal period")
    args = parser.parse_args()

    run_pipeline(
        args.csv_file,
        date_col=args.date_col,
        value_col=args.value_col,
        name=args.name,
        unit=args.unit,
        freq=args.freq,
        period=args.period,
    )


if __name__ == "__main__":
    main()