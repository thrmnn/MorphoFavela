"""Unit test for the neighbourhood-λf windowing primitive.

``neighbourhood_mean`` must average a value field over all points within the
radius of each evaluated point, preserving the open-area (zero) denominator —
the property that makes the window mean a proper Grimmond–Oke λf.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "run_lambda_f_neighbourhood", ROOT / "scripts" / "run_lambda_f_neighbourhood.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_window_mean_includes_zero_cells():
    # 1×3 row of cells at x = 0, 10, 20; one built (value 3.0) flanked by two
    # unbuilt (value 0). A 15 m window at the centre averages all three → 1.0,
    # NOT 3.0 — the open-area denominator is preserved.
    cx = np.array([0.0, 10.0, 20.0])
    cy = np.array([0.0, 0.0, 0.0])
    values = np.array([0.0, 3.0, 0.0])
    eval_mask = np.array([False, True, False])
    out = mod.neighbourhood_mean(cx, cy, values, radius=15.0, eval_mask=eval_mask)
    assert out.shape == (1,)
    assert abs(out[0] - 1.0) < 1e-9


def test_tight_radius_recovers_self():
    # A radius smaller than the spacing sees only the point itself.
    cx = np.array([0.0, 10.0, 20.0])
    cy = np.zeros(3)
    values = np.array([0.0, 3.0, 0.0])
    out = mod.neighbourhood_mean(cx, cy, values, radius=5.0, eval_mask=np.array([False, True, False]))
    assert abs(out[0] - 3.0) < 1e-9


def test_summary_regime_fractions_sum_to_one():
    vals = np.array([0.05, 0.20, 0.40, 0.80, np.nan])
    s = mod.summarise(vals)
    total = (
        s["frac_isolated_lt_0_15"]
        + s["frac_wake_0_15_to_0_35"]
        + s["frac_skimming_ge_0_35"]
    )
    assert abs(total - 1.0) < 1e-9
    assert s["n"] == 4
