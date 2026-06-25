"""Unit test for the λf flow-regime classification."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "run_lambda_f_regime", ROOT / "scripts" / "run_lambda_f_regime.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_regime_bins_and_shares_sum_to_one():
    # one cell per regime: < 0.15 isolated, [0.15,0.65) wake, ≥ 0.65 skimming
    c = mod.classify_regime(np.array([0.10, 0.30, 1.20, np.nan]))
    assert c["n"] == 3
    assert abs(c["isolated"] - 1 / 3) < 1e-9
    assert abs(c["wake"] - 1 / 3) < 1e-9
    assert abs(c["skimming"] - 1 / 3) < 1e-9
    assert abs(c["isolated"] + c["wake"] + c["skimming"] - 1.0) < 1e-9


def test_uniformly_skimming():
    c = mod.classify_regime(np.array([0.8, 1.1, 2.0, 3.5]))
    assert c["skimming"] == 1.0
    assert c["wake"] == 0.0
    assert c["isolated"] == 0.0


def test_boundary_values():
    # exactly at thresholds: 0.15 → wake (isolated is strict <), 0.65 → skimming
    c = mod.classify_regime(np.array([0.15, 0.65]))
    assert c["isolated"] == 0.0
    assert abs(c["wake"] - 0.5) < 1e-9
    assert abs(c["skimming"] - 0.5) < 1e-9
