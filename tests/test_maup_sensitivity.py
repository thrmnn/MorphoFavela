"""Unit tests for the MAUP grid-resolution sensitivity A/B (S2).

Pure-function checks on the regime-share + median/IQR computation using a
small synthetic grid (no disk I/O), plus a guard that the script's flow-regime
threshold constants stay pinned to the canonical 0.15 / 0.65.
"""

import numpy as np

from scripts.run_maup_sensitivity import (
    ISOLATED_MAX,
    SITES,
    SKIMMING_MIN,
    build_curve,
    median_delta,
    median_iqr,
    regime_shares,
    share_delta_pp,
    spearman,
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


# ── resolution-curve helpers ────────────────────────────────────────────


def test_spearman_monotone():
    assert abs(spearman(np.array([1.0, 2, 3, 4]), np.array([2.0, 4, 6, 8])) - 1.0) < 1e-9
    assert abs(spearman(np.array([1.0, 2, 3, 4]), np.array([8.0, 6, 4, 2])) + 1.0) < 1e-9


def test_spearman_ties_and_degenerate():
    # a constant vector has no rank variance → nan, not a crash
    assert np.isnan(spearman(np.array([1.0, 1, 1]), np.array([3.0, 2, 1])))
    # ties handled via averaged ranks
    rho = spearman(np.array([1.0, 1, 2, 3]), np.array([1.0, 2, 2, 4]))
    assert 0.8 <= rho <= 1.0


def _summary(skimming, lf_med, sh_med, per_site):
    """Minimal resolution_summary-shaped dict for build_curve tests."""
    wake = 1.0 - skimming
    return {
        "pooled": {
            "regime_shares": {"isolated": 0.0, "wake": wake, "skimming": skimming, "n": 100},
            "lambda_f_mean": {"median": lf_med},
            "sigma_h": {"median": sh_med},
        },
        "per_site": per_site,
    }


def _site_block(label, skimming):
    return {
        "label": label,
        "regime_shares": {"isolated": 0.0, "wake": 1.0 - skimming, "skimming": skimming, "n": 50},
        "lambda_f_mean": {"median": skimming},
        "sigma_h": {"median": 5.0},
    }


def test_build_curve_trajectory_and_ordering():
    s0, s1 = SITES[0], SITES[1]
    # skimming falls as cells coarsen; site s0 always > s1 (ordering preserved)
    summaries = {
        5: _summary(0.60, 0.80, 4.0, {s0: _site_block("A", 0.70), s1: _site_block("B", 0.50)}),
        10: _summary(0.45, 0.60, 4.5, {s0: _site_block("A", 0.55), s1: _site_block("B", 0.35)}),
        30: _summary(0.30, 0.40, 5.0, {s0: _site_block("A", 0.40), s1: _site_block("B", 0.20)}),
    }
    curve = build_curve(summaries)
    assert curve["cells_m"] == [5, 10, 30]
    # pooled skimming monotone decreasing
    assert curve["pooled"]["skimming"] == [0.60, 0.45, 0.30]
    assert curve["pooled"]["lambda_f_median"] == [0.80, 0.60, 0.40]
    # both sites tracked
    assert set(curve["per_site"]) == {s0, s1}
    assert curve["per_site"][s0]["skimming"] == [0.70, 0.55, 0.40]
    # ordering preserved 5↔30 m → perfect Spearman
    os_ = curve["ordering_stability"]
    assert os_["fine_m"] == 5 and os_["coarse_m"] == 30
    assert abs(os_["rho"] - 1.0) < 1e-9


def test_build_curve_drops_site_missing_at_a_resolution():
    s0, s1 = SITES[0], SITES[1]
    summaries = {
        5: _summary(0.6, 0.8, 4.0, {s0: _site_block("A", 0.7), s1: _site_block("B", 0.5)}),
        10: _summary(0.4, 0.6, 4.5, {s0: _site_block("A", 0.5)}),  # s1 absent here
    }
    # only 2 resolutions but that's fine for build_curve; s1 must be dropped
    curve = build_curve(summaries)
    assert set(curve["per_site"]) == {s0}
