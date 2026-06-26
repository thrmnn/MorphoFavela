"""Unit tests for the MAUP grid-resolution sensitivity A/B (S2).

Pure-function checks on the regime-share + median/IQR computation using a
small synthetic grid (no disk I/O), plus a guard that the script's flow-regime
threshold constants stay pinned to the canonical 0.15 / 0.65.
"""

import numpy as np

from scripts.run_maup_sensitivity import (
    ISOLATED_MAX,
    SKIMMING_MIN,
    median_delta,
    median_iqr,
    regime_shares,
    share_delta_pp,
)


def test_thresholds_match_canonical():
    # Oke (1988) / Grimmond & Oke (1999), as locked in lambda_f_canonical.json.
    assert ISOLATED_MAX == 0.15
    assert SKIMMING_MIN == 0.65


def test_regime_shares_boundaries():
    # 0.15 is wake (>= isolated_max, < skimming_min); 0.65 is skimming (>= min).
    lf = np.array([0.10, 0.149, 0.15, 0.40, 0.649, 0.65, 0.90])
    s = regime_shares(lf)
    assert s["n"] == 7
    assert s["isolated"] == 2 / 7  # 0.10, 0.149
    assert s["skimming"] == 2 / 7  # 0.65, 0.90
    assert s["wake"] == 3 / 7  # 0.15, 0.40, 0.649
    assert abs(s["isolated"] + s["wake"] + s["skimming"] - 1.0) < 1e-12


def test_regime_shares_drops_nan():
    lf = np.array([0.05, np.nan, 0.70, np.nan])
    s = regime_shares(lf)
    assert s["n"] == 2
    assert s["isolated"] == 0.5
    assert s["skimming"] == 0.5
    assert s["wake"] == 0.0


def test_regime_shares_empty():
    s = regime_shares(np.array([]))
    assert s["n"] == 0
    assert s["isolated"] == s["wake"] == s["skimming"] == 0.0


def test_median_iqr_known_values():
    vals = np.arange(1, 101, dtype=float)  # 1..100
    m = median_iqr(vals)
    assert m["n"] == 100
    assert abs(m["median"] - 50.5) < 1e-9
    assert abs(m["q1"] - 25.75) < 1e-9
    assert abs(m["q3"] - 75.25) < 1e-9
    assert abs(m["iqr"] - 49.5) < 1e-9


def test_median_iqr_drops_nan_and_empty():
    m = median_iqr(np.array([2.0, np.nan, 4.0]))
    assert m["n"] == 2
    assert abs(m["median"] - 3.0) < 1e-9
    empty = median_iqr(np.array([np.nan, np.nan]))
    assert empty["n"] == 0
    assert np.isnan(empty["median"])


def test_share_delta_pp():
    a = regime_shares(np.array([0.10, 0.40, 0.70, 0.80]))  # 25/25/50
    b = regime_shares(np.array([0.10, 0.70, 0.80, 0.90]))  # 25/0/75
    d = share_delta_pp(a, b)
    assert abs(d["isolated"] - 0.0) < 1e-9
    assert abs(d["wake"] - (-25.0)) < 1e-9
    assert abs(d["skimming"] - 25.0) < 1e-9


def test_median_delta_abs_and_pct():
    a = median_iqr(np.array([10.0, 10.0]))
    b = median_iqr(np.array([15.0, 15.0]))
    d = median_delta(a, b)
    assert abs(d["abs"] - 5.0) < 1e-9
    assert abs(d["pct"] - 50.0) < 1e-9
