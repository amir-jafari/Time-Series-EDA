"""
Tests for :mod:`tseda.report.html_report`.

Coverage targets
----------------
* HTMLReport.generate()   — creates file, returns path, valid HTML
* File content            — contains expected sections, base64 images
* Input validation        — wrong ts type
* Edge cases              — short series, NaN series, explicit period,
                            custom output directory
"""
from __future__ import annotations

import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from tseda import TimeSeries
from tseda.report.html_report import HTMLReport

rep = HTMLReport()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(n: int = 80, seed: int = 0, nan_frac: float = 0.0) -> TimeSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    vals = rng.standard_normal(n)
    if nan_frac > 0:
        nan_idx = rng.choice(n, size=int(n * nan_frac), replace=False)
        vals[nan_idx] = np.nan
    return TimeSeries(vals, index=idx, name="html_test")


def _generate_to_tmp(ts: TimeSeries, **kwargs) -> tuple[str, str]:
    """Return (path, html_content) after generating to a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name
    try:
        returned = rep.generate(ts, path, **kwargs)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return returned, content
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ===========================================================================
# Return value and file creation
# ===========================================================================


class TestFileCreation:
    def test_file_exists(self):
        ts = _ts()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            rep.generate(ts, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_returns_absolute_path(self):
        ts = _ts()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            returned = rep.generate(ts, path)
            assert os.path.isabs(returned)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_returns_same_path(self):
        ts = _ts()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            returned = rep.generate(ts, path)
            assert os.path.abspath(path) == returned
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_file_is_non_empty(self):
        ts = _ts()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            rep.generate(ts, path)
            assert os.path.getsize(path) > 500
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ===========================================================================
# HTML structure
# ===========================================================================


class TestHtmlStructure:
    def test_valid_html_doctype(self):
        ts = _ts()
        _, html = _generate_to_tmp(ts)
        assert "<!DOCTYPE html>" in html

    def test_has_html_tag(self):
        ts = _ts()
        _, html = _generate_to_tmp(ts)
        assert "<html" in html and "</html>" in html

    def test_has_head_and_body(self):
        ts = _ts()
        _, html = _generate_to_tmp(ts)
        assert "<head>" in html and "<body>" in html

    def test_has_style(self):
        ts = _ts()
        _, html = _generate_to_tmp(ts)
        assert "<style>" in html

    def test_series_name_in_html(self):
        ts = _ts()
        _, html = _generate_to_tmp(ts)
        assert "html_test" in html

    def test_has_details_sections(self):
        ts = _ts()
        _, html = _generate_to_tmp(ts)
        assert "<details" in html and "<summary" in html

    def test_has_base64_image(self):
        ts = _ts()
        _, html = _generate_to_tmp(ts)
        assert "data:image/png;base64," in html

    def test_has_scorecard(self):
        ts = _ts()
        _, html = _generate_to_tmp(ts)
        assert "score-card" in html or "Forecast score" in html

    def test_section_names_present(self):
        ts = _ts()
        _, html = _generate_to_tmp(ts)
        for section in [
            "Data Quality",
            "Descriptive Statistics",
            "Stationarity",
            "Seasonality",
            "Anomaly Detection",
            "Changepoint Detection",
            "Forecastability",
        ]:
            assert section in html, f"Section '{section}' not found in HTML"


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    def test_wrong_ts_type_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".html") as f:
            with pytest.raises(TypeError, match="TimeSeries"):
                rep.generate([1, 2, 3], f.name)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_series_with_nans(self):
        ts = _ts(nan_frac=0.15)
        _, html = _generate_to_tmp(ts)
        assert "<!DOCTYPE html>" in html

    def test_explicit_period(self):
        ts = _ts(n=100)
        _, html = _generate_to_tmp(ts, period=7)
        assert "<!DOCTYPE html>" in html

    def test_short_series(self):
        idx = pd.date_range("2020", periods=25, freq="D")
        ts = TimeSeries(np.arange(25, dtype=float), index=idx)
        _, html = _generate_to_tmp(ts)
        assert "<!DOCTYPE html>" in html

    def test_seasonal_series(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2018-01", periods=60, freq="MS")
        seas = np.tile(np.sin(2 * np.pi * np.arange(12) / 12) * 5, 5)
        ts = TimeSeries(seas + rng.standard_normal(60), index=idx, name="monthly_seasonal")
        _, html = _generate_to_tmp(ts, period=12)
        assert "Decomposition" in html

    def test_custom_alpha(self):
        ts = _ts()
        _, html = _generate_to_tmp(ts, alpha=0.01)
        assert "<!DOCTYPE html>" in html

    def test_utf8_encoding(self):
        ts = _ts()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            rep.generate(ts, path)
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            assert len(content) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)