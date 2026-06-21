"""
Console EDA report for :class:`~tseda.core.TimeSeries`.

Runs all tseda analysis modules and prints a structured plain-text
summary to stdout (or returns it as a string).

Classes
-------
ConsoleReport
    Stateless report generator.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from tseda import TimeSeries
>>> from tseda.report.console_report import ConsoleReport

>>> rng = np.random.default_rng(0)
>>> idx = pd.date_range("2020-01-01", periods=200, freq="D")
>>> ts  = TimeSeries(rng.standard_normal(200), index=idx, name="demo")
>>> report_str = ConsoleReport().to_string(ts)
>>> "demo" in report_str
True
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tseda.core.timeseries import TimeSeries

__all__ = ["ConsoleReport"]

_W = 68  # column width


def _hr(char: str = "─") -> str:
    return char * _W


def _header(text: str, char: str = "═") -> str:
    return f"\n{char * _W}\n  {text}\n{char * _W}"


def _section(title: str) -> str:
    return f"\n{_hr('─')}\n  {title}\n{_hr('─')}"


def _kv(key: str, value: str, width: int = 26) -> str:
    return f"  {key:<{width}}: {value}"


class ConsoleReport:
    """Generate a plain-text EDA report for a :class:`~tseda.core.TimeSeries`.

    Methods
    -------
    to_string(ts, period, alpha)
        Build the report and return it as a string.
    generate(ts, period, alpha)
        Print the report to stdout.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from tseda import TimeSeries
    >>> from tseda.report.console_report import ConsoleReport
    >>> rng = np.random.default_rng(0)
    >>> idx = pd.date_range("2020", periods=100, freq="D")
    >>> ts  = TimeSeries(rng.standard_normal(100), index=idx)
    >>> s = ConsoleReport().to_string(ts)
    >>> isinstance(s, str) and len(s) > 0
    True
    """

    def to_string(
        self,
        ts: TimeSeries,
        *,
        period: Optional[int] = None,
        alpha: float = 0.05,
    ) -> str:
        """Return the full EDA report as a string.

        Parameters
        ----------
        ts : TimeSeries
        period : int, optional
            Seasonal period.  Auto-detected when ``None``.
        alpha : float, optional
            Significance level for statistical tests.  Default ``0.05``.

        Returns
        -------
        str
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )
        lines: list[str] = []

        # ── Title ──────────────────────────────────────────────────
        lines.append(_header(f"tseda EDA Report — {ts.name}"))
        lines.append(_kv("Series name", ts.name))
        lines.append(_kv("Observations", str(ts.n)))
        lines.append(_kv("Frequency", ts.freq or "irregular"))
        lines.append(_kv("Start", str(ts.start)[:19]))
        lines.append(_kv("End",   str(ts.end)[:19]))
        lines.append(_kv("Is regular", str(ts.is_regular)))
        lines.append(_kv("Unit", ts.unit or "—"))

        # ── 1. Data Quality ────────────────────────────────────────
        lines.append(_section("1. DATA QUALITY"))
        try:
            from tseda.quality.missing import MissingValueAnalyzer
            from tseda.quality.outliers import OutlierDetector

            mq = MissingValueAnalyzer().analyze(ts)
            lines.append(_kv("Missing values", f"{mq.n_nan}  ({mq.pct_nan:.2f} %)"))
            lines.append(_kv("Longest NaN run", str(mq.longest_nan_run)))
            lines.append(_kv("Index gaps", str(mq.n_gaps)))

            oq = OutlierDetector().iqr(ts)
            lines.append(_kv("IQR outliers", f"{oq.n_outliers}  ({oq.n_outliers / max(ts.n, 1) * 100:.2f} %)"))
        except Exception as exc:
            lines.append(f"  [warning] Quality analysis failed: {exc}")

        # ── 2. Descriptive Statistics ──────────────────────────────
        lines.append(_section("2. DESCRIPTIVE STATISTICS"))
        try:
            from tseda.statistics.descriptive import DescriptiveAnalyzer

            ds = DescriptiveAnalyzer().analyze(ts)
            lines.append(_kv("Mean",     f"{ds.mean:.6g}"))
            lines.append(_kv("Std",      f"{ds.std:.6g}"))
            lines.append(_kv("Min",      f"{ds.min:.6g}"))
            lines.append(_kv("Max",      f"{ds.max:.6g}"))
            lines.append(_kv("Median",   f"{ds.median:.6g}"))
            lines.append(_kv("Skewness", f"{ds.skewness:.4f}"))
            lines.append(_kv("Kurtosis", f"{ds.kurtosis:.4f}"))
            lines.append(_kv("CV",       f"{ds.cv:.4f}"))
        except Exception as exc:
            lines.append(f"  [warning] Descriptive stats failed: {exc}")

        # ── 3. Stationarity ────────────────────────────────────────
        lines.append(_section("3. STATIONARITY"))
        try:
            from tseda.statistics.stationarity import StationarityTester

            tester = StationarityTester()
            adf = tester.adf(ts, alpha=alpha)
            kpss = tester.kpss(ts, alpha=alpha)
            lines.append(_kv("ADF p-value", f"{adf.p_value:.4f}  → {'stationary' if adf.is_stationary else 'non-stationary'}"))
            lines.append(_kv("KPSS p-value", f"{kpss.p_value:.4f}  → {'stationary' if kpss.is_stationary else 'non-stationary'}"))
            # Combined verdict
            if adf.is_stationary and kpss.is_stationary:
                verdict = "STATIONARY (both tests agree)"
            elif adf.is_stationary:
                verdict = "TREND STATIONARY (consider detrending)"
            elif kpss.is_stationary:
                verdict = "DIFFERENCE STATIONARY (consider d=1)"
            else:
                verdict = "NON-STATIONARY (strong evidence)"
            lines.append(_kv("Verdict", verdict))
        except Exception as exc:
            lines.append(f"  [warning] Stationarity tests failed: {exc}")

        # ── 4. Autocorrelation ─────────────────────────────────────
        lines.append(_section("4. AUTOCORRELATION"))
        try:
            from tseda.statistics.autocorrelation import AutocorrelationAnalyzer

            ac = AutocorrelationAnalyzer().analyze(ts, lags=min(40, ts.n // 2 - 1))
            ci = float(ac.conf_upper[1])
            n_sig_acf  = int(np.sum(np.abs(ac.acf[1:]) > ci))
            n_sig_pacf = int(np.sum(np.abs(ac.pacf[1:]) > ci))
            lines.append(_kv("Significant ACF lags",  f"{n_sig_acf}"))
            lines.append(_kv("Significant PACF lags", f"{n_sig_pacf}"))
            lines.append(_kv("Is white noise",        str(ac.is_white_noise)))
            lines.append(_kv("Ljung-Box p (lag 10)",
                f"{ac.lb_pvalue[min(9, len(ac.lb_pvalue)-1)]:.4f}"))
        except Exception as exc:
            lines.append(f"  [warning] Autocorrelation failed: {exc}")

        # ── 5. Seasonality ─────────────────────────────────────────
        lines.append(_section("5. SEASONALITY"))
        detected_period = period
        try:
            from tseda.seasonality.detector import SeasonalityDetector

            sr = SeasonalityDetector().detect(ts)
            lines.append(_kv("Is seasonal",      str(sr.is_seasonal)))
            lines.append(_kv("Dominant period",  str(sr.dominant_period)))
            lines.append(_kv("Candidate periods", str(sr.candidate_periods[:5])))
            if period is None and sr.is_seasonal:
                detected_period = sr.dominant_period
        except Exception as exc:
            lines.append(f"  [warning] Seasonality detection failed: {exc}")

        # ── 6. Forecastability ─────────────────────────────────────
        lines.append(_section("6. FORECASTABILITY"))
        try:
            from tseda.forecastability.scorer import ForecastabilityScorer

            fr = ForecastabilityScorer().score(ts, period=detected_period, alpha=alpha)
            lines.append(_kv("Overall score", f"{fr.score:.1f} / 100"))
            lines.append("")
            sub_w = {"data_quality": 20, "stationarity": 15, "signal_to_noise": 20,
                     "autocorrelation": 15, "sample_size": 15, "regularity": 15}
            lines.append(f"  {'Sub-score':<22} {'Score':>7}   {'Weight':>6}")
            lines.append(f"  {'─'*22}  {'─'*7}   {'─'*6}")
            for k, v in fr.sub_scores.items():
                bar = "█" * int(v / 10)
                lines.append(f"  {k:<22} {v:>7.1f}   ({sub_w[k]:>2}%)  {bar}")
            lines.append("")
            lines.append(_kv("Recommended model",  fr.recommended_model))
            lines.append(_kv("Recommended diff",   f"d={fr.recommended_diff}"))
            lines.append(_kv("Recommended period", str(fr.recommended_period)))
        except Exception as exc:
            lines.append(f"  [warning] Forecastability scoring failed: {exc}")

        lines.append(f"\n{_hr('═')}\n")
        return "\n".join(lines)

    def generate(
        self,
        ts: TimeSeries,
        *,
        period: Optional[int] = None,
        alpha: float = 0.05,
    ) -> None:
        """Print the EDA report to stdout.

        Parameters
        ----------
        ts : TimeSeries
        period : int, optional
        alpha : float, optional

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from tseda import TimeSeries
        >>> from tseda.report.console_report import ConsoleReport
        >>> rng = np.random.default_rng(0)
        >>> idx = pd.date_range("2020", periods=100, freq="D")
        >>> ts  = TimeSeries(rng.standard_normal(100), index=idx)
        >>> ConsoleReport().generate(ts)  # doctest: +SKIP
        """
        print(self.to_string(ts, period=period, alpha=alpha))