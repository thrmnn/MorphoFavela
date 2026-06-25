"""Regime-stratified taxonomy (script 09) — axis logic + canonical denominator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "regime_tax", ROOT / "scripts" / "brisa_ventilation" / "09_regime_taxonomy.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _df(sun, lf, hw):
    n = len(sun)
    return pd.DataFrame({
        "built": np.ones(n, bool),
        "sun_known": np.isfinite(sun),
        "lf_known": np.isfinite(lf),
        "hw_known": np.isfinite(hw),
        "sun": np.asarray(sun, float),
        "lf": np.asarray(lf, float),
        "hw": np.asarray(hw, float),
    })


def test_skimming_boundary_is_065_not_275():
    # λf = 1.0 sits in skimming (>0.65) → ventilation-constrained;
    # under the deprecated 2.75 pre-screen it would have been adequate.
    df = _df(sun=[5.0], lf=[1.0], hw=[np.nan])
    r = mod.shares_lf(df)
    assert r["counts"]["ventilation_constraint"] == 1
    assert r["counts"]["adequate"] == 0


def test_four_states_partition_the_denominator():
    df = _df(sun=[5, 5, 1, 1], lf=[0.2, 1.0, 0.2, 1.0], hw=[np.nan] * 4)
    r = mod.shares_lf(df)
    assert r["n_classified"] == 4
    assert sum(r["counts"].values()) == 4
    assert r["counts"] == {"adequate": 1, "sunlight_constraint": 1,
                           "ventilation_constraint": 1, "compound_constraint": 1}


def test_hw_axis_uses_own_denominator():
    # one cell has no H/W sample → excluded from the H/W denominator only.
    df = _df(sun=[1, 1, 1], lf=[1.0, 1.0, 1.0], hw=[1.0, 1.0, np.nan])
    assert mod.shares_lf(df)["n_classified"] == 3
    assert mod.shares_hw(df)["n_classified_hw"] == 2


def test_canonical_denominator_is_56k_not_64k():
    """Live regime JSON must pool to the street-solar denominator (~56.6k),
    not the 64,389 raw built mask."""
    p = ROOT / "outputs" / "brisa_ventilation_fix" / "taxonomy_regime.json"
    if not p.exists():
        return  # output not generated in this checkout
    j = json.loads(p.read_text())
    n = j["aggregates_lambda_f"]["all"]["n_classified"]
    assert 55_000 < n < 58_000, n
    assert n < 64_389
