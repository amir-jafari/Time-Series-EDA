"""
HTML EDA report for :class:`~tseda.core.TimeSeries`.

Generates a self-contained HTML file with:

* Summary scorecard table at the top
* Collapsible sections for each analysis module
* Matplotlib figures embedded as base64 PNG — no external assets

Classes
-------
HTMLReport
    Stateless HTML report generator.

Examples
--------
>>> import numpy as np, pandas as pd, tempfile, os
>>> from tseda import TimeSeries
>>> from tseda.report.html_report import HTMLReport

>>> rng = np.random.default_rng(0)
>>> idx = pd.date_range("2020-01-01", periods=100, freq="D")
>>> ts  = TimeSeries(rng.standard_normal(100), index=idx, name="demo")
>>> with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
...     path = HTMLReport().generate(ts, f.name)
>>> os.path.exists(path)
True
>>> os.unlink(path)
"""
from __future__ import annotations

import base64
import io
import os
from typing import Optional

import numpy as np

from tseda.core.timeseries import TimeSeries

__all__ = ["HTMLReport"]

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_CSS = """
<style>
  * { box-sizing: border-box; }
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px 30px;
         background: #f4f6f8; color: #2c3e50; font-size: 14px; }
  h1   { color: #2c3e50; border-bottom: 3px solid #2980b9; padding-bottom: 8px; margin-bottom: 4px; }
  h2   { color: #2980b9; margin: 6px 0 4px 0; font-size: 1.05em; }
  .meta { color: #7f8c8d; font-size: 0.88em; margin-bottom: 16px; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0; background: white;
          border-radius: 4px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  th   { background: #2c3e50; color: white; padding: 8px 14px; text-align: left; font-weight: 600; }
  td   { padding: 6px 14px; border-bottom: 1px solid #e8ecef; }
  tr:last-child td { border-bottom: none; }
  tr:nth-child(even) td { background: #f8fafc; }
  .score-card { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }
  .card { background: white; border-radius: 6px; padding: 14px 18px; min-width: 160px;
          box-shadow: 0 1px 4px rgba(0,0,0,.1); text-align: center; }
  .card .label { font-size: 0.82em; color: #7f8c8d; text-transform: uppercase; letter-spacing: .04em; }
  .card .value { font-size: 1.7em; font-weight: 700; margin-top: 4px; }
  .good  { color: #27ae60; }
  .mid   { color: #e67e22; }
  .bad   { color: #e74c3c; }
  .neutral { color: #2980b9; }
  details { background: white; border: 1px solid #dce1e7; border-radius: 6px;
            margin: 10px 0; padding: 0; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
  summary { font-weight: 600; color: #2c3e50; cursor: pointer; padding: 12px 16px;
            font-size: 1.02em; user-select: none; }
  summary:hover { background: #f0f4f8; }
  .section-body { padding: 12px 16px 16px; border-top: 1px solid #dce1e7; }
  img { max-width: 100%; height: auto; display: block; margin: 10px auto;
        border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  .bar-bg { background: #eaecef; border-radius: 3px; height: 8px; overflow: hidden; margin-top: 4px; }
  .bar-fill { height: 100%; border-radius: 3px; }
  .warn-box { background: #fff8e1; border-left: 4px solid #e67e22; padding: 8px 12px;
              margin: 6px 0; border-radius: 0 4px 4px 0; font-size: 0.9em; color: #7f5200; }
  .kv-row  { display: flex; gap: 8px; margin: 3px 0; }
  .kv-key  { color: #7f8c8d; min-width: 200px; }
  .kv-val  { font-weight: 500; }
</style>
"""

_SCORE_WEIGHTS = {
    "data_quality": 20, "stationarity": 15, "signal_to_noise": 20,
    "autocorrelation": 15, "sample_size": 15, "regularity": 15,
}


def _score_class(score: float) -> str:
    if score >= 70:
        return "good"
    if score >= 40:
        return "mid"
    return "bad"


def _fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _img(b64: str, width: str = "100%") -> str:
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:{width};" />'
    )


def _section(title: str, body: str, open_: bool = False) -> str:
    open_attr = " open" if open_ else ""
    return (
        f"<details{open_attr}>"
        f"<summary>{title}</summary>"
        f'<div class="section-body">{body}</div>'
        f"</details>"
    )


def _kv(key: str, value: str) -> str:
    return (
        f'<div class="kv-row">'
        f'<span class="kv-key">{key}</span>'
        f'<span class="kv-val">{value}</span>'
        f"</div>"
    )


def _warn(msg: str) -> str:
    return f'<div class="warn-box">⚠ {msg}</div>'


def _table(headers: list, rows: list) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _bar(score: float) -> str:
    color = {"good": "#27ae60", "mid": "#e67e22", "bad": "#e74c3c"}[_score_class(score)]
    return (
        f'<div class="bar-bg">'
        f'<div class="bar-fill" style="width:{score:.0f}%;background:{color};"></div>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_overview(ts: TimeSeries) -> str:
    lines = [
        _kv("Series name", ts.name),
        _kv("Observations", str(ts.n)),
        _kv("Frequency", ts.freq or "irregular"),
        _kv("Start", str(ts.start)[:19]),
        _kv("End",   str(ts.end)[:19]),
        _kv("Is regular", str(ts.is_regular)),
        _kv("Unit", ts.unit or "—"),
    ]
    return "\n".join(lines)


def _build_quality(ts: TimeSeries) -> tuple[str, dict]:
    meta: dict = {}
    try:
        from tseda.quality.missing import MissingValueAnalyzer
        from tseda.quality.outliers import OutlierDetector
        from tseda.visualization.quality_plots import (
            plot_missing_heatmap, plot_outliers,
        )

        mq = MissingValueAnalyzer().analyze(ts)
        oq = OutlierDetector().iqr(ts)
        meta["pct_missing"] = mq.pct_nan
        meta["n_outliers"] = oq.n_outliers

        html = _table(
            ["Metric", "Value"],
            [
                ["Missing values", f"{mq.n_nan} ({mq.pct_nan:.2f} %)"],
                ["Longest NaN run", str(mq.longest_nan_run)],
                ["Index gaps", str(mq.n_gaps)],
                ["IQR outliers", f"{oq.n_outliers} ({oq.n_outliers / max(ts.n, 1) * 100:.2f} %)"],
            ],
        )
        html += _img(_fig_to_b64(plot_missing_heatmap(ts, figsize=(10, 1.5))))
        html += _img(_fig_to_b64(plot_outliers(ts, oq, figsize=(10, 3))))
        return html, meta
    except Exception as exc:
        return _warn(f"Quality section unavailable: {exc}"), meta


def _build_statistics(ts: TimeSeries, alpha: float) -> str:
    try:
        from tseda.statistics.descriptive import DescriptiveAnalyzer
        from tseda.statistics.autocorrelation import AutocorrelationAnalyzer
        from tseda.visualization.distribution_plots import (
            plot_distribution, plot_qq, plot_rolling_stats,
        )
        from tseda.visualization.correlation_plots import plot_acf_pacf

        ds = DescriptiveAnalyzer().analyze(ts)
        html = _table(
            ["Statistic", "Value"],
            [
                ["Mean",     f"{ds.mean:.6g}"],
                ["Std",      f"{ds.std:.6g}"],
                ["Min",      f"{ds.min:.6g}"],
                ["Max",      f"{ds.max:.6g}"],
                ["Median",   f"{ds.median:.6g}"],
                ["Skewness", f"{ds.skewness:.4f}"],
                ["Kurtosis", f"{ds.kurtosis:.4f}"],
                ["CV",       f"{ds.cv:.4f}"],
            ],
        )
        html += _img(_fig_to_b64(plot_distribution(ts, figsize=(10, 4))))
        html += _img(_fig_to_b64(plot_qq(ts, figsize=(5, 5))), width="50%")

        window = max(5, ts.n // 10)
        html += _img(_fig_to_b64(plot_rolling_stats(ts, window=window, figsize=(10, 5))))

        ac_lags = min(40, ts.n // 2 - 1)
        if ac_lags >= 1:
            ac_result = AutocorrelationAnalyzer().analyze(ts, lags=ac_lags, alpha=alpha)
            html += _img(_fig_to_b64(plot_acf_pacf(ac_result, figsize=(12, 4))))
        return html
    except Exception as exc:
        return _warn(f"Statistics section unavailable: {exc}")


def _build_stationarity(ts: TimeSeries, alpha: float) -> str:
    try:
        from tseda.statistics.stationarity import StationarityTester

        t = StationarityTester()
        adf  = t.adf(ts, alpha=alpha)
        kpss = t.kpss(ts, alpha=alpha)

        if adf.is_stationary and kpss.is_stationary:
            verdict = "STATIONARY — both tests agree."
        elif adf.is_stationary:
            verdict = "TREND STATIONARY — consider detrending."
        elif kpss.is_stationary:
            verdict = "DIFFERENCE STATIONARY — consider d=1."
        else:
            verdict = "NON-STATIONARY — strong evidence."

        return _table(
            ["Test", "Statistic", "p-value", "Result"],
            [
                ["ADF",  f"{adf.statistic:.4f}",  f"{adf.p_value:.4f}",
                 "stationary" if adf.is_stationary else "non-stationary"],
                ["KPSS", f"{kpss.statistic:.4f}", f"{kpss.p_value:.4f}",
                 "stationary" if kpss.is_stationary else "non-stationary"],
            ],
        ) + f"<p><strong>Verdict:</strong> {verdict}</p>"
    except Exception as exc:
        return _warn(f"Stationarity section unavailable: {exc}")


def _build_decomposition(ts: TimeSeries, period: int) -> str:
    try:
        from tseda.decomposition.stl import STLDecomposer
        from tseda.visualization.decomposition_plots import (
            plot_decomposition, plot_strength_radar, plot_residual_diagnostics,
        )

        result = STLDecomposer().decompose(ts, period=period)
        html = _table(
            ["Metric", "Value"],
            [
                ["Method", result.method],
                ["Period", str(result.period)],
                ["Trend strength",    f"{result.strength_trend:.4f}"],
                ["Seasonal strength", f"{result.strength_seasonal:.4f}"],
            ],
        )
        html += _img(_fig_to_b64(plot_decomposition(result, figsize=(10, 9))))
        html += _img(_fig_to_b64(plot_strength_radar(result, figsize=(5, 5))), width="45%")
        html += _img(_fig_to_b64(plot_residual_diagnostics(result, figsize=(10, 4))))
        return html
    except Exception as exc:
        return _warn(f"Decomposition unavailable: {exc}")


def _build_seasonality(ts: TimeSeries, period: Optional[int]) -> tuple[str, Optional[int]]:
    detected: Optional[int] = period
    try:
        from tseda.seasonality.detector import SeasonalityDetector
        from tseda.visualization.seasonality_plots import (
            plot_periodogram, plot_season_heatmap, plot_monthly_boxplots,
        )

        sr = SeasonalityDetector().detect(ts)
        if period is None and sr.is_seasonal:
            detected = sr.dominant_period

        html = _table(
            ["Metric", "Value"],
            [
                ["Is seasonal",      str(sr.is_seasonal)],
                ["Dominant period",  str(sr.dominant_period)],
                ["Candidate periods", str(sr.candidate_periods[:5])],
                ["Fisher G p-value", f"{sr.fisher_p_value:.4f}" if sr.fisher_p_value is not None else "—"],
            ],
        )
        html += _img(_fig_to_b64(plot_periodogram(sr, figsize=(10, 3))))
        eff_period = detected or 12
        if ts.n >= 2 * eff_period:
            html += _img(_fig_to_b64(plot_season_heatmap(ts, period=eff_period, figsize=(10, 4))))
        html += _img(_fig_to_b64(plot_monthly_boxplots(ts, figsize=(10, 3))))
        return html, detected
    except Exception as exc:
        return _warn(f"Seasonality section unavailable: {exc}"), detected


def _build_anomaly(ts: TimeSeries) -> str:
    try:
        from tseda.anomaly.detector import AnomalyDetector
        from tseda.visualization.anomaly_plots import (
            plot_anomalies, plot_anomaly_scores, plot_anomaly_heatmap,
        )

        det = AnomalyDetector()
        r_iqr = det.rolling_iqr(ts)
        r_z   = det.rolling_z(ts)

        html = _table(
            ["Method", "Anomalies", "Rate"],
            [
                ["Rolling IQR", str(r_iqr.n_anomalies), f"{r_iqr.n_anomalies / ts.n * 100:.2f} %"],
                ["Rolling Z",   str(r_z.n_anomalies),   f"{r_z.n_anomalies   / ts.n * 100:.2f} %"],
            ],
        )
        html += _img(_fig_to_b64(plot_anomalies(ts, r_iqr, figsize=(10, 4))))
        html += _img(_fig_to_b64(plot_anomaly_scores(r_iqr, figsize=(10, 3))))
        html += _img(_fig_to_b64(plot_anomaly_heatmap(ts, [r_iqr, r_z], figsize=(10, 2))))
        return html
    except Exception as exc:
        return _warn(f"Anomaly section unavailable: {exc}")


def _build_changepoint(ts: TimeSeries) -> str:
    try:
        from tseda.changepoint.detector import ChangepointDetector
        from tseda.visualization.changepoint_plots import (
            plot_changepoints, plot_segment_means, plot_cusum,
        )

        det = ChangepointDetector()
        r_bs   = det.binary_segmentation(ts)
        r_cusum = det.cusum(ts)

        html = _table(
            ["Method", "Changepoints"],
            [
                ["Binary segmentation", str(r_bs.n_changepoints)],
                ["CUSUM",               str(r_cusum.n_changepoints)],
            ],
        )
        html += _img(_fig_to_b64(plot_changepoints(ts, r_bs, figsize=(10, 4))))
        html += _img(_fig_to_b64(plot_segment_means(ts, r_bs, figsize=(10, 4))))
        html += _img(_fig_to_b64(plot_cusum(ts, r_cusum, figsize=(10, 3))))
        return html
    except Exception as exc:
        return _warn(f"Changepoint section unavailable: {exc}")


def _build_forecastability(ts: TimeSeries, period: Optional[int], alpha: float) -> tuple[str, dict]:
    meta: dict = {}
    try:
        from tseda.forecastability.scorer import ForecastabilityScorer

        fr = ForecastabilityScorer().score(ts, period=period, alpha=alpha)
        meta["score"] = fr.score
        meta["recommended_model"] = fr.recommended_model
        meta["recommended_diff"]  = fr.recommended_diff
        meta["recommended_period"] = fr.recommended_period

        sub_rows = [
            [k, f"{v:.1f}", f"{_SCORE_WEIGHTS[k]} %", _bar(v)]
            for k, v in fr.sub_scores.items()
        ]
        html = _table(
            ["Sub-score", "Score", "Weight", "Bar"],
            sub_rows,
        )
        html += _table(
            ["Recommendation", "Value"],
            [
                ["Recommended model",  fr.recommended_model],
                ["Recommended diff",   f"d = {fr.recommended_diff}"],
                ["Recommended period", str(fr.recommended_period) if fr.recommended_period else "—"],
                ["Missing %",          f"{fr.pct_missing:.2f} %"],
                ["Outlier %",          f"{fr.pct_outlier:.2f} %"],
            ],
        )
        return html, meta
    except Exception as exc:
        return _warn(f"Forecastability section unavailable: {exc}"), meta


def _build_scorecard(ts: TimeSeries, quality_meta: dict, cast_meta: dict) -> str:
    score = cast_meta.get("score")
    pct_miss = quality_meta.get("pct_missing", 0.0)
    n_out    = quality_meta.get("n_outliers", 0)
    model    = cast_meta.get("recommended_model", "—")

    def _card(label: str, value: str, cls: str = "neutral") -> str:
        return (
            f'<div class="card">'
            f'<div class="label">{label}</div>'
            f'<div class="value {cls}">{value}</div>'
            f"</div>"
        )

    score_cls = _score_class(score) if score is not None else "neutral"
    score_str = f"{score:.0f}" if score is not None else "—"

    return (
        f'<div class="score-card">'
        + _card("Forecast score", score_str, score_cls)
        + _card("Observations", str(ts.n), "neutral")
        + _card("Missing %", f"{pct_miss:.1f}%",
                "good" if pct_miss < 5 else ("mid" if pct_miss < 20 else "bad"))
        + _card("Outliers", str(n_out),
                "good" if n_out == 0 else "mid")
        + _card("Best model", model, "neutral")
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class HTMLReport:
    """Generate a self-contained HTML EDA report.

    Methods
    -------
    generate(ts, output_path, period, alpha)
        Write the HTML report and return the output path.

    Examples
    --------
    >>> import numpy as np, pandas as pd, tempfile, os
    >>> from tseda import TimeSeries
    >>> from tseda.report.html_report import HTMLReport
    >>> rng = np.random.default_rng(0)
    >>> idx = pd.date_range("2020", periods=100, freq="D")
    >>> ts  = TimeSeries(rng.standard_normal(100), index=idx)
    >>> with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
    ...     path = HTMLReport().generate(ts, f.name)
    >>> os.path.exists(path)
    True
    >>> os.unlink(path)
    """

    def generate(
        self,
        ts: TimeSeries,
        output_path: str,
        *,
        period: Optional[int] = None,
        alpha: float = 0.05,
    ) -> str:
        """Generate the HTML EDA report.

        Parameters
        ----------
        ts : TimeSeries
            Series to analyse.
        output_path : str or path-like
            Destination file.  Created / overwritten.
        period : int, optional
            Seasonal period.  Auto-detected when ``None``.
        alpha : float, optional
            Significance level.  Default ``0.05``.

        Returns
        -------
        str
            Absolute path to the written HTML file.

        Raises
        ------
        TypeError
            If *ts* is not a :class:`~tseda.core.TimeSeries`.
        """
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"'ts' must be a TimeSeries, got {type(ts).__name__!r}."
            )

        # ── Run sections ───────────────────────────────────────────
        overview_html = _build_overview(ts)
        quality_html,    quality_meta   = _build_quality(ts)
        statistics_html                 = _build_statistics(ts, alpha)
        stationarity_html               = _build_stationarity(ts, alpha)
        seasonality_html, eff_period    = _build_seasonality(ts, period)
        forecast_html,   forecast_meta  = _build_forecastability(ts, eff_period, alpha)
        anomaly_html                    = _build_anomaly(ts)
        changepoint_html                = _build_changepoint(ts)

        decomp_html = ""
        if eff_period is not None and ts.n >= 2 * eff_period:
            decomp_html = _build_decomposition(ts, eff_period)

        scorecard_html = _build_scorecard(ts, quality_meta, forecast_meta)

        # ── Assemble HTML ──────────────────────────────────────────
        body_parts = [
            f"<h1>tseda EDA Report</h1>",
            f'<p class="meta">{ts.name} &nbsp;|&nbsp; {ts.n} observations '
            f"&nbsp;|&nbsp; {ts.freq or 'irregular'} "
            f"&nbsp;|&nbsp; {str(ts.start)[:10]} → {str(ts.end)[:10]}</p>",
            scorecard_html,
            _section("Series Overview",       overview_html,      open_=True),
            _section("Data Quality",           quality_html,       open_=True),
            _section("Descriptive Statistics", statistics_html),
            _section("Stationarity",           stationarity_html),
            _section("Seasonality",            seasonality_html),
        ]
        if decomp_html:
            body_parts.append(_section("Decomposition", decomp_html))
        body_parts += [
            _section("Anomaly Detection", anomaly_html),
            _section("Changepoint Detection", changepoint_html),
            _section("Forecastability",       forecast_html),
        ]

        html = (
            "<!DOCTYPE html>"
            "<html lang='en'>"
            f"<head><meta charset='utf-8'>"
            f"<title>tseda EDA — {ts.name}</title>"
            f"{_CSS}</head>"
            "<body>"
            + "\n".join(body_parts)
            + "<p style='color:#aaa;font-size:.8em;margin-top:30px;'>"
            "Generated by <strong>tseda</strong></p>"
            "</body></html>"
        )

        output_path = os.path.abspath(str(output_path))
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        return output_path