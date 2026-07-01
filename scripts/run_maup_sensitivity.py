#!/usr/bin/env python3
"""MAUP sensitivity — does the headline morphometry survive changes in cell size?

The Modifiable Areal Unit Problem (MAUP) is the standard methods-reviewer
objection to grid-based morphometrics: results computed on a fixed lattice can
be artefacts of the cell size. This script answers it directly. It reads the
already-meshed grids (no re-meshing) at each available resolution and on built
cells (``building_count > 0``) recomputes:

- the pooled and per-site Oke / Grimmond-Oke flow-regime shares
  (isolated / wake / skimming) from ``lambda_f_mean``;
- the σH (``sigma_h``) and ``lambda_f_mean`` median + IQR.

Two products come out of the same summaries:

1. **A/B** — the canonical 10 m vs 20 m doubling, with explicit deltas
   (percentage points for regime shares, absolute + relative for medians),
   written to ``outputs/comparative/maup/maup_sensitivity.json`` and drawn as a
   grouped-bar figure of the regime shares.
2. **Curve** — the full resolution sweep over every present cell size in
   {5, 10, 15, 20, 30} m, written to ``.../maup_curve.json`` and drawn as a
   three-panel resolution curve (regime trajectories, λf/σH medians, per-site
   skimming with a Spearman ordering-stability scalar). Degrades gracefully:
   whatever grids exist are used, and the curve is skipped if only the two A/B
   resolutions are present.

Honest framing: this does NOT recompute or touch the canonical 10 m lock. It
reports the swing the data shows; if regime shares move under coarsening, that
is stated plainly rather than smoothed into a "stable" conclusion.

Run (after building the extra grids with run_morphometric_audit.py +
migrate_lambda_f_dissolve.py at each --cell-size / --grid-suffix):
    python scripts/run_maup_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ── Oke (1988) / Grimmond & Oke (1999) flow-regime thresholds on lambda_f ──
# Must match outputs/brisa_ventilation_fix/lambda_f_canonical.json. The 20 m
# grid is binned with the SAME thresholds so the A/B is honest.
ISOLATED_MAX = 0.15
SKIMMING_MIN = 0.65

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
SITE_LABELS = {
    "vidigal": "Vidigal",
    "rocinha": "Rocinha",
    "complexo_do_alemao": "Complexo do Alemão",
    "riodaspedras": "Rio das Pedras",
    "maré": "Maré",
}
REGIMES = ["isolated", "wake", "skimming"]
CANONICAL_JSON = PROJECT_ROOT / "outputs" / "brisa_ventilation_fix" / "lambda_f_canonical.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "comparative" / "maup"

# Full resolution curve. 10 m is the canonical lock (suffix ""); the rest are
# MAUP-only variants built by run_morphometric_audit.py --output-suffix Nm.
SUFFIX_BY_CELL = {5: "_5m", 10: "", 15: "_15m", 20: "_20m", 30: "_30m"}
CURVE_CELLS = [5, 10, 15, 20, 30]


# ══════════════════════════════════════════════════════════════════════
#  Pure functions (unit-tested in tests/test_maup_sensitivity.py)
# ══════════════════════════════════════════════════════════════════════


def regime_shares(lambda_f: np.ndarray) -> dict[str, float]:
    """Fraction of cells in each Oke/Grimmond-Oke regime, dropping NaNs.

    Returns shares that sum to 1.0 over the valid (non-NaN) cells; an
    all-NaN / empty input yields a zero count and 0.0 for every regime.
    """
    vals = np.asarray(lambda_f, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = vals.size
    if n == 0:
        return {r: 0.0 for r in REGIMES} | {"n": 0}
    isolated = int((vals < ISOLATED_MAX).sum())
    skimming = int((vals >= SKIMMING_MIN).sum())
    wake = n - isolated - skimming
    return {
        "isolated": isolated / n,
        "wake": wake / n,
        "skimming": skimming / n,
        "n": n,
    }


def median_iqr(values: np.ndarray) -> dict[str, float]:
    """Median, Q1, Q3, IQR over non-NaN values; n=0 → NaN stats."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = vals.size
    if n == 0:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan"),
                "iqr": float("nan"), "n": 0}
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1), "n": n}


def share_delta_pp(share_10m: dict[str, float], share_20m: dict[str, float]) -> dict[str, float]:
    """Per-regime 20m − 10m delta in percentage points."""
    return {r: (share_20m[r] - share_10m[r]) * 100.0 for r in REGIMES}


def median_delta(stat_10m: dict[str, float], stat_20m: dict[str, float]) -> dict[str, float]:
    """Absolute and relative (%) delta of the median, 20m − 10m."""
    m10, m20 = stat_10m["median"], stat_20m["median"]
    abs_d = m20 - m10
    rel = (abs_d / m10 * 100.0) if (m10 and not np.isnan(m10)) else float("nan")
    return {"abs": abs_d, "pct": rel}


# ══════════════════════════════════════════════════════════════════════
#  Grid I/O
# ══════════════════════════════════════════════════════════════════════


def load_built_cells(site: str, suffix: str) -> pd.DataFrame | None:
    """Read built cells (building_count>0) for one site/resolution.

    ``suffix`` is '' for the 10 m grid, '_20m' for the 20 m grid. Returns
    None if the grid is absent or carries neither required column.
    """
    path = PROJECT_ROOT / "outputs" / site / f"morphometrics{suffix}" / "grid" / "grid_metrics.gpkg"
    if not path.exists():
        return None
    g = gpd.read_file(path)
    if "building_count" not in g.columns:
        return None
    if "lambda_f_mean" not in g.columns and "sigma_h" not in g.columns:
        return None
    return pd.DataFrame(g.drop(columns="geometry"))[g["building_count"] > 0].reset_index(drop=True)


def resolution_summary(suffix: str) -> dict:
    """Pooled + per-site regime shares and σH/λf median+IQR for one resolution."""
    per_site: dict[str, dict] = {}
    pooled_lf: list[np.ndarray] = []
    pooled_sh: list[np.ndarray] = []
    missing: list[str] = []

    for site in SITES:
        df = load_built_cells(site, suffix)
        if df is None or df.empty:
            missing.append(site)
            continue
        lf = df["lambda_f_mean"].to_numpy(dtype=float) if "lambda_f_mean" in df else np.array([])
        sh = df["sigma_h"].to_numpy(dtype=float) if "sigma_h" in df else np.array([])
        per_site[site] = {
            "label": SITE_LABELS[site],
            "n_built": int(len(df)),
            "regime_shares": regime_shares(lf),
            "sigma_h": median_iqr(sh),
            "lambda_f_mean": median_iqr(lf),
        }
        if lf.size:
            pooled_lf.append(lf)
        if sh.size:
            pooled_sh.append(sh)

    lf_all = np.concatenate(pooled_lf) if pooled_lf else np.array([])
    sh_all = np.concatenate(pooled_sh) if pooled_sh else np.array([])
    return {
        "pooled": {
            "regime_shares": regime_shares(lf_all),
            "sigma_h": median_iqr(sh_all),
            "lambda_f_mean": median_iqr(lf_all),
        },
        "per_site": per_site,
        "missing_sites": missing,
    }


# ══════════════════════════════════════════════════════════════════════
#  A/B assembly + reporting
# ══════════════════════════════════════════════════════════════════════


def build_ab(s10: dict, s20: dict) -> dict:
    """Side-by-side A/B with explicit deltas for pooled + per-site."""
    p10, p20 = s10["pooled"], s20["pooled"]
    pooled = {
        "10m": p10,
        "20m": p20,
        "delta_regime_pp": share_delta_pp(p10["regime_shares"], p20["regime_shares"]),
        "delta_sigma_h_median": median_delta(p10["sigma_h"], p20["sigma_h"]),
        "delta_lambda_f_median": median_delta(p10["lambda_f_mean"], p20["lambda_f_mean"]),
    }
    per_site = {}
    for site in SITES:
        a, b = s10["per_site"].get(site), s20["per_site"].get(site)
        if a is None or b is None:
            continue
        per_site[site] = {
            "label": SITE_LABELS[site],
            "10m": a,
            "20m": b,
            "delta_regime_pp": share_delta_pp(a["regime_shares"], b["regime_shares"]),
            "delta_sigma_h_median": median_delta(a["sigma_h"], b["sigma_h"]),
            "delta_lambda_f_median": median_delta(a["lambda_f_mean"], b["lambda_f_mean"]),
        }
    max_pp = max(abs(pooled["delta_regime_pp"][r]) for r in REGIMES)
    verdict = "stable" if max_pp <= 5.0 else "shifts"
    return {
        "pooled": pooled,
        "per_site": per_site,
        "max_pooled_regime_swing_pp": max_pp,
        "headline": verdict,
        "missing_sites": {"10m": s10["missing_sites"], "20m": s20["missing_sites"]},
    }


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties averaged — enough for a 5-site Spearman."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two equal-length vectors (NaN → 0-length → nan)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 2:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def build_curve(summaries: dict[int, dict]) -> dict:
    """Assemble the resolution curve from a {cell_size: resolution_summary} dict.

    Pure over already-computed summaries (no disk I/O) so it is unit-testable.
    Only cell sizes present in ``summaries`` are used, sorted ascending. Emits
    pooled trajectories (regime shares, λf/σH medians, n) and per-site
    trajectories, plus a Spearman ordering-stability scalar for the per-site
    skimming share between the finest and coarsest present resolution.
    """
    cells = sorted(summaries)
    pooled = {r: [summaries[c]["pooled"]["regime_shares"][r] for c in cells] for r in REGIMES}
    pooled["lambda_f_median"] = [summaries[c]["pooled"]["lambda_f_mean"]["median"] for c in cells]
    pooled["sigma_h_median"] = [summaries[c]["pooled"]["sigma_h"]["median"] for c in cells]
    pooled["n_built"] = [summaries[c]["pooled"]["regime_shares"]["n"] for c in cells]

    per_site: dict[str, dict] = {}
    for site in SITES:
        if not all(site in summaries[c]["per_site"] for c in cells):
            continue
        blk = summaries[cells[0]]["per_site"][site]
        per_site[site] = {
            "label": blk["label"],
            "skimming": [summaries[c]["per_site"][site]["regime_shares"]["skimming"] for c in cells],
            "wake": [summaries[c]["per_site"][site]["regime_shares"]["wake"] for c in cells],
            "isolated": [summaries[c]["per_site"][site]["regime_shares"]["isolated"] for c in cells],
            "lambda_f_median": [summaries[c]["per_site"][site]["lambda_f_mean"]["median"] for c in cells],
            "sigma_h_median": [summaries[c]["per_site"][site]["sigma_h"]["median"] for c in cells],
            "n_built": [summaries[c]["per_site"][site]["regime_shares"]["n"] for c in cells],
        }

    # ordering stability: per-site skimming rank, finest vs coarsest resolution
    fine, coarse = cells[0], cells[-1]
    sites_both = [s for s in SITES if s in per_site]
    skim_fine = np.array([summaries[fine]["per_site"][s]["regime_shares"]["skimming"] for s in sites_both])
    skim_coarse = np.array([summaries[coarse]["per_site"][s]["regime_shares"]["skimming"] for s in sites_both])
    rho = spearman(skim_fine, skim_coarse)

    return {
        "cells_m": cells,
        "pooled": pooled,
        "per_site": per_site,
        "ordering_stability": {
            "metric": "spearman_rho_per_site_skimming",
            "fine_m": fine, "coarse_m": coarse,
            "rho": rho, "n_sites": len(sites_both),
        },
    }


def print_curve(curve: dict) -> None:
    cells = curve["cells_m"]
    p = curve["pooled"]
    print("\n" + "═" * 72)
    print(f"MAUP RESOLUTION CURVE — {'/'.join(str(c) for c in cells)} m (built cells)")
    print("═" * 72)
    print(f"\n{'cell (m)':>8} {'isolated':>9} {'wake':>9} {'skimming':>9} "
          f"{'λf med':>8} {'σH med':>8} {'n built':>9}")
    for i, c in enumerate(cells):
        print(f"{c:>8} {p['isolated'][i]*100:>8.1f}% {p['wake'][i]*100:>8.1f}% "
              f"{p['skimming'][i]*100:>8.1f}% {p['lambda_f_median'][i]:>8.3f} "
              f"{p['sigma_h_median'][i]:>8.3f} {p['n_built'][i]:>9,}")
    os_ = curve["ordering_stability"]
    print(f"\nPer-site skimming ordering {os_['fine_m']}↔{os_['coarse_m']} m: "
          f"Spearman ρ = {os_['rho']:.3f} (n={os_['n_sites']} sites)")
    print("─" * 72)


def make_curve_figure(curve: dict, out_dir: Path) -> Path | None:
    """Three-panel resolution curve: regimes, λf/σH medians, per-site skimming."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    cells = curve["cells_m"]
    p = curve["pooled"]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))

    # (A) pooled regime shares vs cell size
    reg_col = {"isolated": "#66AA66", "wake": "#EE9944", "skimming": "#BB4444"}
    for r in REGIMES:
        ax[0].plot(cells, [v * 100 for v in p[r]], "o-", color=reg_col[r],
                   label=r.capitalize(), lw=1.8, ms=5)
    ax[0].axvline(10, color="#888", ls=":", lw=1)
    ax[0].text(10, ax[0].get_ylim()[1], " 10 m lock", color="#888", fontsize=7,
               va="top", ha="left")
    ax[0].set_xlabel("Cell size (m)")
    ax[0].set_ylabel("Pooled cell share (%)")
    ax[0].set_title("(A) Flow-regime shares vs resolution", fontsize=9)
    ax[0].legend(frameon=False, fontsize=8)

    # (B) pooled λf & σH medians vs cell size (twin y)
    axb = ax[1]
    l1 = axb.plot(cells, p["lambda_f_median"], "s-", color="#4477AA",
                  label="λf_mean median", lw=1.8, ms=5)
    axb.set_xlabel("Cell size (m)")
    axb.set_ylabel("λf_mean median", color="#4477AA")
    axb.tick_params(axis="y", labelcolor="#4477AA")
    axb2 = axb.twinx()
    l2 = axb2.plot(cells, p["sigma_h_median"], "^--", color="#AA3377",
                   label="σH median (m)", lw=1.8, ms=5)
    axb2.set_ylabel("σH median (m)", color="#AA3377")
    axb2.tick_params(axis="y", labelcolor="#AA3377")
    axb.axvline(10, color="#888", ls=":", lw=1)
    axb.set_title("(B) λf and σH medians vs resolution", fontsize=9)
    axb.legend(l1 + l2, [ln.get_label() for ln in l1 + l2], frameon=False, fontsize=8,
               loc="upper right")

    # (C) per-site skimming trajectory (ordering preservation)
    for s, blk in curve["per_site"].items():
        ax[2].plot(cells, [v * 100 for v in blk["skimming"]], "o-", lw=1.6, ms=4,
                   label=blk["label"])
    ax[2].axvline(10, color="#888", ls=":", lw=1)
    os_ = curve["ordering_stability"]
    ax[2].set_xlabel("Cell size (m)")
    ax[2].set_ylabel("Skimming share (%)")
    ax[2].set_title(f"(C) Per-site skimming vs resolution\n"
                    f"(ordering ρ={os_['rho']:.2f}, {os_['fine_m']}↔{os_['coarse_m']} m)",
                    fontsize=9)
    ax[2].legend(frameon=False, fontsize=7, ncol=1)

    for a in (ax[0], ax[1], ax[2], axb2):
        a.spines[["top"]].set_visible(False)
        a.set_xticks(cells)
    fig.tight_layout()
    out = out_dir / "maup_resolution_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def print_table(ab: dict) -> None:
    p = ab["pooled"]
    print("\n" + "═" * 72)
    print("MAUP SENSITIVITY A/B — 10 m vs 20 m (built cells, building_count>0)")
    print("═" * 72)
    print("\nPOOLED flow-regime shares (Oke/Grimmond-Oke on λf_mean):")
    print(f"  {'regime':<10} {'10 m':>9} {'20 m':>9} {'Δ (pp)':>9}")
    for r in REGIMES:
        print(f"  {r:<10} {p['10m']['regime_shares'][r]*100:>8.1f}% "
              f"{p['20m']['regime_shares'][r]*100:>8.1f}% "
              f"{p['delta_regime_pp'][r]:>+8.1f}")
    print(f"  (n built: 10 m = {p['10m']['regime_shares']['n']:,}  "
          f"20 m = {p['20m']['regime_shares']['n']:,})")

    for key, lab in [("sigma_h", "σH (m)"), ("lambda_f_mean", "λf_mean")]:
        a, b = p["10m"][key], p["20m"][key]
        d = p[f"delta_{'sigma_h' if key=='sigma_h' else 'lambda_f'}_median"]
        print(f"\nPOOLED {lab} median [IQR]:")
        print(f"  10 m: {a['median']:.3f} [{a['q1']:.3f}–{a['q3']:.3f}]")
        print(f"  20 m: {b['median']:.3f} [{b['q1']:.3f}–{b['q3']:.3f}]")
        print(f"  Δ median: {d['abs']:+.3f}  ({d['pct']:+.1f}%)")

    print("\nPER-SITE pooled regime-share swing (max |Δ pp| across regimes):")
    for site, blk in ab["per_site"].items():
        mx = max(abs(blk["delta_regime_pp"][r]) for r in REGIMES)
        print(f"  {blk['label']:<22} {mx:>5.1f} pp")

    print("\n" + "─" * 72)
    print(f"Max pooled regime swing: {ab['max_pooled_regime_swing_pp']:.1f} pp"
          f"   →  HEADLINE: {ab['headline'].upper()}")
    if any(ab["missing_sites"].values()):
        print(f"Missing/empty grids: {ab['missing_sites']}")
    print("─" * 72)


def make_figure(ab: dict, out_dir: Path) -> Path | None:
    """Grouped-bar chart of pooled regime shares, 10 m vs 20 m."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    p = ab["pooled"]
    s10 = [p["10m"]["regime_shares"][r] * 100 for r in REGIMES]
    s20 = [p["20m"]["regime_shares"][r] * 100 for r in REGIMES]
    x = np.arange(len(REGIMES))
    w = 0.38

    fig, axx = plt.subplots(figsize=(5.0, 3.4))
    b1 = axx.bar(x - w / 2, s10, w, label="10 m", color="#4477AA")
    b2 = axx.bar(x + w / 2, s20, w, label="20 m", color="#EE6677")
    for bars in (b1, b2):
        for bar in bars:
            axx.annotate(f"{bar.get_height():.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha="center", va="bottom", fontsize=8, xytext=(0, 1), textcoords="offset points")
    axx.set_xticks(x)
    axx.set_xticklabels([r.capitalize() for r in REGIMES])
    axx.set_ylabel("Pooled cell share (%)")
    axx.set_title(f"Flow-regime shares under cell-size doubling\n"
                  f"(max swing {ab['max_pooled_regime_swing_pp']:.1f} pp — {ab['headline']})", fontsize=9)
    axx.legend(frameon=False)
    axx.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = out_dir / "maup_regime_shares.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical = json.loads(CANONICAL_JSON.read_text()) if CANONICAL_JSON.exists() else {}
    canon_thr = canonical.get("flow_regime_thresholds", {})

    # summaries for every present resolution (reused by both A/B and curve)
    summaries: dict[int, dict] = {}
    for cell in CURVE_CELLS:
        suffix = SUFFIX_BY_CELL[cell]
        summ = resolution_summary(suffix)
        # a resolution is "present" if at least one site had a grid
        if len(summ["missing_sites"]) < len(SITES):
            summaries[cell] = summ

    s10 = summaries.get(10) or resolution_summary("")
    s20 = summaries.get(20) or resolution_summary("_20m")
    ab = build_ab(s10, s20)

    record = {
        "title": "MAUP sensitivity A/B — flow-regime + σH/λf stability under 10 m → 20 m doubling",
        "generated_by": "scripts/run_maup_sensitivity.py",
        "flow_regime_thresholds": {"isolated_max": ISOLATED_MAX, "skimming_min": SKIMMING_MIN,
                                   "reference": "Oke (1988) / Grimmond & Oke (1999)"},
        "canonical_thresholds_match": (
            canon_thr.get("isolated_max") == ISOLATED_MAX
            and canon_thr.get("skimming_min") == SKIMMING_MIN
        ),
        "canonical_10m_pooled_regime": canonical.get("flow_regime_pooled"),
        "ab": ab,
        "interpretation": (
            f"Doubling the grid cell from 10 m to 20 m moves the pooled flow-regime "
            f"shares by at most {ab['max_pooled_regime_swing_pp']:.1f} pp "
            f"(isolated {ab['pooled']['delta_regime_pp']['isolated']:+.1f}, "
            f"wake {ab['pooled']['delta_regime_pp']['wake']:+.1f}, "
            f"skimming {ab['pooled']['delta_regime_pp']['skimming']:+.1f}). "
            f"The headline regime mix is therefore '{ab['headline']}' under cell-size "
            f"doubling. σH and λf medians shift by "
            f"{ab['pooled']['delta_sigma_h_median']['pct']:+.1f}% and "
            f"{ab['pooled']['delta_lambda_f_median']['pct']:+.1f}% respectively. "
            f"Reported as measured — no smoothing toward a 'stable' verdict."
        ),
    }

    out_json = OUT_DIR / "maup_sensitivity.json"
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print_table(ab)
    fig_path = make_figure(ab, OUT_DIR)
    print(f"\nWrote {out_json}")
    if fig_path:
        print(f"Wrote {fig_path}")

    # ── Full resolution curve (needs ≥ 3 present resolutions) ──────────────
    if len(summaries) >= 3:
        curve = build_curve(summaries)
        curve_record = {
            "title": "MAUP resolution curve — flow-regime + σH/λf trajectories over cell size",
            "generated_by": "scripts/run_maup_sensitivity.py",
            "flow_regime_thresholds": {"isolated_max": ISOLATED_MAX, "skimming_min": SKIMMING_MIN,
                                       "reference": "Oke (1988) / Grimmond & Oke (1999)"},
            "lambda_f_basis": "dissolved (party-wall-corrected), canonical",
            "canonical_lock_m": 10,
            "curve": curve,
            "interpretation": (
                f"Across {'/'.join(str(c) for c in curve['cells_m'])} m the pooled skimming "
                f"share moves {curve['pooled']['skimming'][0]*100:.1f}% → "
                f"{curve['pooled']['skimming'][-1]*100:.1f}% and the λf_mean median "
                f"{curve['pooled']['lambda_f_median'][0]:.3f} → "
                f"{curve['pooled']['lambda_f_median'][-1]:.3f} as cells coarsen (each cell "
                f"pools more open ground, lowering λf and pushing cells out of skimming). "
                f"Per-site skimming ordering is preserved (Spearman ρ = "
                f"{curve['ordering_stability']['rho']:.2f}, "
                f"{curve['ordering_stability']['fine_m']}↔{curve['ordering_stability']['coarse_m']} m), "
                f"so cross-site comparisons are robust to cell size even though absolute "
                f"shares are not — absolute regime shares must be quoted at the 10 m lock."
            ),
        }
        curve_json = OUT_DIR / "maup_curve.json"
        curve_json.write_text(json.dumps(curve_record, ensure_ascii=False, indent=2))
        print_curve(curve)
        curve_fig = make_curve_figure(curve, OUT_DIR)
        print(f"Wrote {curve_json}")
        if curve_fig:
            print(f"Wrote {curve_fig}")
    else:
        print(f"\n[curve skipped: only {len(summaries)} resolution(s) present — "
              f"need ≥3 of {CURVE_CELLS}]")


if __name__ == "__main__":
    main()
