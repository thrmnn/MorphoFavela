"""Unit test for the round-3 §0 text-overflow gate (fig_style.check_text_overflow)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fig_style", ROOT / "outputs" / "paper_figures" / "fig_style.py"
)
fig_style = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(ROOT / "outputs" / "paper_figures"))
_spec.loader.exec_module(fig_style)


def test_clean_figure_passes():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_title("a clean title")
    ax.text(0.5, 0.5, "centered", transform=ax.transAxes, ha="center")
    assert fig_style.check_text_overflow(fig) == []
    plt.close(fig)


def test_offcanvas_text_flagged():
    fig, ax = plt.subplots(figsize=(4, 3))
    # figure-fraction y > 1 places the label above the canvas top → must be caught
    fig.text(0.5, 1.15, "this label sits off the top of the canvas")
    bad = fig_style.check_text_overflow(fig)
    assert bad, "off-canvas text should be flagged"
    assert bad[0][1] > 1.0  # overflow in pixels
    plt.close(fig)


def test_gate_raises_on_overflow():
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.text(-0.2, 0.5, "off the left edge")
    try:
        fig_style.save_fig(fig, "_gate_test_should_not_write", gate=True)
        raised = False
    except ValueError:
        raised = True
    finally:
        plt.close("all")
    assert raised
