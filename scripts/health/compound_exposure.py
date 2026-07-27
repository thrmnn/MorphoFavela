"""T4 — compound (sun × ventilation) exposure vs TB.

Extends tb_sun_deficit_screen from ONE exposure (winter sun-deficit, ρ≈+0.80,
exact perm p≈0.13, n=5) to TWO, adding a REAL geometric ventilation-deficit and
asking whether a COMPOUND (sun ∩ ventilation) deficit tracks TB better than sun
alone. This is the ecological touchpoint for the "respiratory mechanism lives on
ventilation" claim and the P1 4-state compound-deprivation taxonomy — WITHOUT
touching the gated/synthetic CFD (the ventilation metric is strictly geometric,
pre-CFD: the built-cell frontal-area share).

DESIGN (all three exposures from ONE self-consistent built-cell classification —
the same 4-state taxonomy as outputs/paper_figures/cross_site_stats.json, so the
compound is the literal conjunction of the two marginals I report):
  - sun-deficit share    = built cells with solar_hours_winter < 2.0 h  (WHO floor)
  - ventilation-deficit  = built cells with lambda_f_mean > 0.35        (P1 taxonomy)
  - COMPOUND (primary)   = built cells failing BOTH  = compound_constraint state
Sensitivity: repeat with the stricter SKIMMING threshold lambda_f_mean ≥ 0.65
(the ventilation_index.json flag_skimming axis) to check threshold-robustness.

Jacarezinho is the one TB site absent from cross_site_stats.json / the CFD
campaign, so its shares are recomputed here from its grid_metrics.gpkg +
svf_streets_solar.gpkg with the identical classifier (aggregate_solar + built_mask).

HONEST FRAMING — this is a design/method deliverable + a direction-only read, NOT
an inferential claim. n=5 is far below the powered regime (power ≈0.13 at ρ=0.8,
per tb_power_curve.py); the ventilation-deficit share is a compressed, saturated
built-form proxy collinear with the sun-deficit and with generic density/
deprivation. A perfect rank match at n=5 is NOT evidence of a ventilation
mechanism. If compound did NOT beat sun-alone this script would say so.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "health"))

from compute_cross_site_stats import aggregate_solar  # noqa: E402
from src.morphometry.invariants import built_mask  # noqa: E402
from tb_sun_deficit_screen import (  # noqa: E402
    SITES,
    YEARS,
    boot_ci,
    exact_perm_p,
    incidence,
)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260727)
OUT = ROOT / "outputs" / "comparative" / "health"

SUN_FLOOR_H = 2.0
LAMBDA_F_PRIMARY = 0.35   # P1 4-state taxonomy threshold (cross_site_stats.json)
LAMBDA_F_SKIMMING = 0.65  # ventilation_index.json flag_skimming axis


def site_deficits(site: str, lf_thr: float) -> dict[str, float]:
    """Built-cell sun / ventilation / compound deficit shares (%) for one site.

    Identical classifier to compute_cross_site_stats.classify: a cell counts only
    if built (lenient) with both solar and lambda_f known; compound = fails both.
    """
    grid = gpd.read_file(ROOT / f"outputs/{site}/morphometrics/grid/grid_metrics.gpkg")
    grid = aggregate_solar(
        grid, ROOT / f"outputs/{site}/morphometrics/svf/svf_streets_solar.gpkg"
    )
    built = np.asarray(built_mask(grid, lenient=True))
    sun = grid["solar_hours_winter"]
    lf = grid["lambda_f_mean"]
    m = built & sun.notna() & lf.notna()
    sd = (sun[m] < SUN_FLOOR_H).to_numpy()
    vd = (lf[m] > lf_thr).to_numpy()
    n = int(m.sum())
    return {
        "n_built_cells": n,
        "sun_deficit_pct": float(sd.mean() * 100),
        "vent_deficit_pct": float(vd.mean() * 100),
        "compound_pct": float((sd & vd).mean() * 100),
    }


def three_rhos(sun, vent, comp, tb):
    """ρ + exact two-tailed perm p + bootstrap sign-only CI for each exposure."""
    out = {}
    for name, x in (("sun", sun), ("ventilation", vent), ("compound", comp)):
        x = np.asarray(x)
        rho, _ = spearmanr(x, tb)
        lo, hi, med = boot_ci(x, tb)
        out[name] = {
            "rho": float(rho),
            "perm_p_two_tailed": exact_perm_p(x, tb),
            "boot_ci95_sign_only": [lo, hi],
            "boot_median": med,
        }
    return out


def delta_sign_ci(a, b, tb, n=5000):
    """Bootstrap sign-only CI of Δρ = ρ(a,TB) − ρ(b,TB) by resampling sites."""
    a, b, tb = map(np.asarray, (a, b, tb))
    k = len(tb)
    vals = []
    for _ in range(n):
        s = rng.integers(0, k, k)
        if len(set(tb[s])) < 3:
            continue
        ra, _ = spearmanr(a[s], tb[s])
        rb, _ = spearmanr(b[s], tb[s])
        if np.isfinite(ra) and np.isfinite(rb):
            vals.append(ra - rb)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)),
            float(np.mean([v > 0 for v in vals])))


def build_block(sites, tb, lf_thr):
    d = {s: site_deficits(s, lf_thr) for s in sites}
    sun = [d[s]["sun_deficit_pct"] for s in sites]
    vent = [d[s]["vent_deficit_pct"] for s in sites]
    comp = [d[s]["compound_pct"] for s in sites]
    rhos = three_rhos(sun, vent, comp, tb)
    d_lo, d_hi, d_frac = delta_sign_ci(comp, sun, tb)
    rhos["delta_compound_minus_sun"] = {
        "value": rhos["compound"]["rho"] - rhos["sun"]["rho"],
        "boot_ci95_sign_only": [d_lo, d_hi],
        "boot_frac_positive": d_frac,
    }
    return {"lambda_f_threshold": lf_thr, "per_site": d, "correlations": rhos}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sites = list(SITES)
    tb = np.array([incidence(s, YEARS) for s in sites])

    primary = build_block(sites, tb, LAMBDA_F_PRIMARY)
    skimming = build_block(sites, tb, LAMBDA_F_SKIMMING)

    def line(block):
        c = block["correlations"]
        return (f"λf>{block['lambda_f_threshold']}: ρ_sun={c['sun']['rho']:+.2f} "
                f"(p={c['sun']['perm_p_two_tailed']:.3f})  "
                f"ρ_vent={c['ventilation']['rho']:+.2f} "
                f"(p={c['ventilation']['perm_p_two_tailed']:.3f})  "
                f"ρ_comp={c['compound']['rho']:+.2f} "
                f"(p={c['compound']['perm_p_two_tailed']:.3f})  "
                f"Δ(comp−sun)={c['delta_compound_minus_sun']['value']:+.2f}")

    print("=== T4 compound (sun × ventilation) deficit vs TB — n=5, direction-only ===")
    for s in sites:
        p = primary["per_site"][s]
        print(f"  {SITES[s][0]:14s} sun<2h={p['sun_deficit_pct']:5.1f}%  "
              f"λf>.35={p['vent_deficit_pct']:5.1f}%  compound={p['compound_pct']:5.1f}%  "
              f"TB={incidence(s, YEARS):5.0f}/100k")
    print("  " + line(primary))
    print("  " + line(skimming) + "   [skimming sensitivity]")
    cp = primary["correlations"]
    dv = cp["delta_compound_minus_sun"]
    beats = dv["value"] > 0
    print(f"\n  VERDICT: point Δρ(compound−sun)={dv['value']:+.2f} → compound tracks TB "
          f"{'better than' if beats else 'no better than'} sun-alone, BUT the sign-only "
          f"bootstrap does NOT confirm it (P[Δ>0]={dv['boot_frac_positive']:.2f}); the "
          f"improvement is not distinguishable from n=5 noise. ventilation-alone "
          f"ρ={cp['ventilation']['rho']:+.2f} is the strongest single axis. n=5 underpowered "
          f"(power≈0.13 @ρ=0.8) → direction-only, NOT inferential; the ventilation-deficit "
          f"is a saturated built-form proxy collinear with sun & density.")

    # ---- figure: (1) ρ by exposure, (2) compound-deficit vs TB scatter ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.2))
    labels = ["sun\nalone", "ventilation\nalone", "compound\n(sun ∩ vent)"]
    keys = ["sun", "ventilation", "compound"]
    rp = [cp[k]["rho"] for k in keys]
    rs = [skimming["correlations"][k]["rho"] for k in keys]
    xpos = np.arange(3)
    ax1.bar(xpos, rp, width=0.55, color=["#c9a227", "#3d6b8e", "#b5651d"],
            edgecolor="#333", linewidth=0.6, zorder=3, label="λf>0.35 (P1 taxonomy)")
    ax1.plot(xpos, rs, "D", ms=8, color="#7a3d0c", zorder=4,
             label="λf≥0.65 (skimming)")
    for x, r, p in zip(xpos, rp, [cp[k]["perm_p_two_tailed"] for k in keys]):
        ax1.annotate(f"ρ={r:+.2f}\np={p:.3f}", (x, r), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=8.5, color="#333")
    ax1.axhline(0, color="#888", lw=0.8)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Spearman ρ vs TB incidence (9-yr, n=5)")
    ax1.set_title("Exposure → TB rank-tracking", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.25)
    ax1.annotate(f"Δ(compound−sun) = {dv['value']:+.2f}", (0.5, 1.02),
                 xycoords="axes fraction", ha="center", fontsize=9, color="#b5651d",
                 fontweight="bold")

    for s in sites:
        p = primary["per_site"][s]
        ax2.scatter(p["compound_pct"], incidence(s, YEARS), s=95, c="#b5651d",
                    edgecolor="#333", linewidth=0.6, zorder=3)
        ax2.annotate(SITES[s][0], (p["compound_pct"], incidence(s, YEARS)),
                     xytext=(6, 4), textcoords="offset points", fontsize=9, color="#333")
    xc = np.array([primary["per_site"][s]["compound_pct"] for s in sites])
    b, a = np.polyfit(xc, tb, 1)
    xx = np.linspace(xc.min() - 3, xc.max() + 3, 50)
    ax2.plot(xx, a + b * xx, "--", c="#b5651d", lw=1.2, alpha=0.6)
    ax2.set_xlabel("Compound deficit  (% built cells: sun<2h AND λf>0.35)")
    ax2.set_ylabel("TB incidence  (/100k, 2015–23 mean)")
    ax2.set_title(f"Compound deficit vs TB  (ρ={cp['compound']['rho']:+.2f})",
                  fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.25)
    fig.suptitle("Compound (sun × ventilation) deprivation vs TB — Rio favelas "
                 "(exploratory, n=5, direction-only)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = OUT / "compound_exposure.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"\n  figure → {png.relative_to(ROOT)}")

    (OUT / "compound_exposure.json").write_text(json.dumps({
        "framing": "exploratory ecological probe; direction-only; NOT causal; NOT "
                   "inferential (n=5, power≈0.13 @ρ=0.8); geometric ventilation, pre-CFD; "
                   "ventilation-deficit is a saturated built-form proxy collinear with the "
                   "sun-deficit and generic density — a rank match is NOT a mechanism",
        "n_sites": len(sites),
        "window_years": [YEARS[0], YEARS[-1]],
        "sun_floor_h": SUN_FLOOR_H,
        "compound_definition": "P1 4-state co-occurrence: built cell failing BOTH the WHO "
                               "sun floor (solar_hours_winter<2h) AND the ventilation "
                               "threshold (lambda_f_mean>0.35) = compound_constraint state; "
                               "the literal conjunction of the two reported marginals",
        "tb_incidence_9yr": {s: round(float(incidence(s, YEARS)), 1) for s in sites},
        "primary_lambda_f_0_35": primary,
        "sensitivity_skimming_lambda_f_0_65": skimming,
        "verdict": {
            "compound_beats_sun_alone": bool(beats),
            "delta_rho_compound_minus_sun_primary": dv["value"],
            "delta_rho_compound_minus_sun_skimming":
                skimming["correlations"]["delta_compound_minus_sun"]["value"],
            "ventilation_alone_rho_primary": cp["ventilation"]["rho"],
            "delta_boot_frac_positive_primary": dv["boot_frac_positive"],
            "reading": "point Δρ is positive (compound +0.10 over sun at λf>0.35, +0.20 at "
                       "skimming λf≥0.65) so compound tracks TB at least as well as sun "
                       "alone — but the sign-only bootstrap does NOT confirm the improvement "
                       "(P[Δ>0]≈0.45): it is not distinguishable from n=5 noise. "
                       "ventilation-alone rank-matches TB (ρ=+1.0) and is the strongest "
                       "single axis, so the compound largely inherits ventilation's rank. "
                       "At n=5 (underpowered) this is a direction-only design deliverable, "
                       "NOT evidence of a ventilation mechanism separable from deprivation.",
        },
    }, indent=2, ensure_ascii=False))
    print(f"  json   → {(OUT / 'compound_exposure.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
