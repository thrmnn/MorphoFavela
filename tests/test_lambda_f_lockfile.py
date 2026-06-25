"""λf canonical lock file (script 10) — pins the invariants brisaverse integrates."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "outputs" / "brisa_ventilation_fix" / "lambda_f_canonical.json"


def _lock():
    if not LOCK.exists():
        return None
    return json.loads(LOCK.read_text())


def test_canonical_denominator_is_full_built_mask():
    j = _lock()
    if j is None:
        return  # output not generated in this checkout
    assert j["groupings"]["pooled"]["n"] == 64389  # full built mask, not 56,631


def test_signature_family_reproduces_tr_table():
    """TR §5.2 λf row 1.04 / 0.68 / 0.87 must reproduce exactly from the lock file."""
    j = _lock()
    if j is None:
        return
    fam = j["groupings"]["signature_family"]
    assert round(fam["Hillside"]["median"], 2) == 1.04
    assert round(fam["Mixed"]["median"], 2) == 0.68
    assert round(fam["Flatland"]["median"], 2) == 0.87


def test_dissolved_below_summed_everywhere():
    j = _lock()
    if j is None:
        return
    for s, p in j["per_site"].items():
        assert p["over_count_factor_median"] > 1.0, s
        assert p["lambda_f_dissolved"]["median"] < p["lambda_f_summed"]["median"], s


def test_pooled_regime_is_predominantly_skimming():
    j = _lock()
    if j is None:
        return
    assert round(j["flow_regime_pooled"]["skimming"], 2) == 0.65


def test_two_groupings_are_distinct():
    """terrain_binary folds CDA into hillside; signature_family keeps it Mixed."""
    j = _lock()
    if j is None:
        return
    assert set(j["groupings"]["terrain_binary"]) == {"hillside", "flatland"}
    assert "Mixed" in j["groupings"]["signature_family"]
    # hillside (V+R+A) has more cells than signature Hillside (V+R)
    assert (j["groupings"]["terrain_binary"]["hillside"]["n"]
            > j["groupings"]["signature_family"]["Hillside"]["n"])
