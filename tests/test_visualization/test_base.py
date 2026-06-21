"""Tests for tseda.visualization.base."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from tseda.visualization.base import PALETTE, set_style, _make_fig_ax, _set_title


class TestPalette:
    def test_required_keys(self):
        for key in ("dark", "accent", "anomaly", "seasonal", "trend", "neutral"):
            assert key in PALETTE

    def test_hex_values(self):
        for v in PALETTE.values():
            assert v.startswith("#")
            assert len(v) == 7


class TestSetStyle:
    def test_runs_without_error(self):
        set_style()

    def test_idempotent(self):
        set_style()
        set_style()


class TestMakeFigAx:
    def test_creates_new_fig_when_ax_none(self):
        fig, ax = _make_fig_ax(None, None, (6, 4))
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_respects_figsize(self):
        fig, ax = _make_fig_ax(None, (8, 3), (6, 4))
        w, h = fig.get_size_inches()
        assert abs(w - 8) < 0.1
        assert abs(h - 3) < 0.1
        plt.close(fig)

    def test_uses_existing_ax(self):
        fig0, ax0 = plt.subplots()
        fig, ax = _make_fig_ax(ax0, None, (6, 4))
        assert ax is ax0
        plt.close(fig0)