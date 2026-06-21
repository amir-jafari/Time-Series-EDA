"""
Tests for :mod:`tseda.report.console_report`.

Coverage targets
----------------
* ConsoleReport.to_string()  — returns string, contains expected content
* ConsoleReport.generate()   — prints without error (captured by capsys)
* Section presence            — each section name appears in output
* Input validation            — wrong type raises TypeError
* Edge cases                  — short series, series with NaN, custom period
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.report.console_report import ConsoleReport

rep = ConsoleReport()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(n: int = 100, seed: int = 0, nan_frac: float = 0.0) -> TimeSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    vals = rng.standard_normal(n)
    if nan_frac > 0:
        nan_idx = rng.choice(n, size=int(n * nan_frac), replace=False)
        vals[nan_idx] = np.nan
    return TimeSeries(vals, index=idx, name="test_series")


def _seasonal_ts(n: int = 200, period: int = 7) -> TimeSeries:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n)
    vals = np.sin(2 * np.pi * t / period) * 3 + rng.standard_normal(n) * 0.5
    return TimeSeries(vals, index=idx, name="seasonal")


# ===========================================================================
# Return type and content
# ===========================================================================


class TestToString:
    def test_returns_string(self):
        ts = _ts()
        s = rep.to_string(ts)
        assert isinstance(s, str)

    def test_non_empty(self):
        ts = _ts()
        s = rep.to_string(ts)
        assert len(s) > 100

    def test_contains_series_name(self):
        ts = _ts()
        s = rep.to_string(ts)
        assert "test_series" in s

    def test_contains_all_section_headers(self):
        ts = _ts()
        s = rep.to_string(ts)
        for section in [
            "DATA QUALITY",
            "DESCRIPTIVE STATISTICS",
            "STATIONARITY",
            "AUTOCORRELATION",
            "SEASONALITY",
            "FORECASTABILITY",
        ]:
            assert section in s, f"Section '{section}' missing from report"

    def test_contains_n_obs(self):
        ts = _ts(n=80)
        s = rep.to_string(ts)
        assert "80" in s

    def test_contains_frequency(self):
        ts = _ts()
        s = rep.to_string(ts)
        assert "D" in s or "daily" in s.lower() or "D" in s


class TestGenerate:
    def test_prints_to_stdout(self, capsys):
        ts = _ts()
        rep.generate(ts)
        captured = capsys.readouterr()
        assert len(captured.out) > 50

    def test_no_stderr(self, capsys):
        ts = _ts()
        rep.generate(ts)
        captured = capsys.readouterr()
        assert captured.err == ""


# ===========================================================================
# Section content
# ===========================================================================


class TestSectionContent:
    def test_mean_in_output(self):
        ts = _ts()
        s = rep.to_string(ts)
        assert "Mean" in s

    def test_stationarity_has_adf(self):
        ts = _ts()
        s = rep.to_string(ts)
        assert "ADF" in s

    def test_forecastability_has_score(self):
        ts = _ts()
        s = rep.to_string(ts)
        assert "score" in s.lower() or "Score" in s

    def test_recommended_model_present(self):
        ts = _ts()
        s = rep.to_string(ts)
        models = ("ARIMA", "SARIMA", "ETS", "Prophet", "ML")
        assert any(m in s for m in models)

    def test_missing_count_present(self):
        ts = _ts(nan_frac=0.1)
        s = rep.to_string(ts)
        assert "Missing" in s


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="TimeSeries"):
            rep.to_string([1, 2, 3])

    def test_generate_wrong_type_raises(self):
        with pytest.raises(TypeError):
            rep.generate(42)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_short_series(self):
        idx = pd.date_range("2020", periods=20, freq="D")
        ts = TimeSeries(np.arange(20, dtype=float), index=idx)
        s = rep.to_string(ts)
        assert isinstance(s, str)

    def test_series_with_nans(self):
        ts = _ts(nan_frac=0.15)
        s = rep.to_string(ts)
        assert isinstance(s, str)

    def test_explicit_period(self):
        ts = _seasonal_ts()
        s = rep.to_string(ts, period=7)
        assert isinstance(s, str)

    def test_monthly_series(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2018-01", periods=36, freq="MS")
        ts = TimeSeries(rng.standard_normal(36) * 10 + 100, index=idx, name="monthly")
        s = rep.to_string(ts)
        assert isinstance(s, str)

    def test_constant_series_no_crash(self):
        idx = pd.date_range("2020", periods=30, freq="D")
        ts = TimeSeries(np.ones(30), index=idx)
        s = rep.to_string(ts)
        assert isinstance(s, str)

    def test_custom_alpha(self):
        ts = _ts()
        s = rep.to_string(ts, alpha=0.01)
        assert isinstance(s, str)