#!/usr/bin/env python3
"""λf flow-regime classification — the "uniformly skimming" saturation finding.

Per the round-3 urban-physics council: geometry classifies the FLOW REGIME (it
cannot grade per-cell ventilation adequacy — that is CFD-τ, Tier 2). Each built
cell is binned by its (now canonical, dissolved) frontal-area density λf against
the Oke / Grimmond–Oke flow-regime thresholds:

    isolated roughness   λf < 0.15
    wake interference    0.15 ≤ λf < 0.65
    skimming flow        λf ≥ 0.65   (the most ventilation-suppressed canopy)

The result is the at-scale ventilation statement: the favela fabric sits almost
entirely in skimming flow, so frontal-area density classifies *which regime* a
cell is in but not *how ventilated* it is. No fitting, no per-cell adequacy claim.

Outputs:
  outputs/paper_figures/lambda_f_regime.json
  outputs/paper_figures/exports/lambda_f_regime.png (+ .svg)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import SITE_COLORS, SITE_LABELS, apply_style, save_fig  # noqa: E402

# Hillside → flatland order (round-2 cross-figure consistency).
SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
ISOLATED_MAX = 0.15
SKIMMING_MIN = 0.65
REGIMES = ("isolated", "wake", "skimming")


def classify_regime(lf: np.ndarray) -> dict:
    """Shares of cells in each Oke/GO flow regime (NaNs dropped)."""
    v = np.asarray(lf, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return {"n": 0, **{r: float("nan") for r in REGIMES}}
    iso = float((v < ISOLATED_MAX).mean())
    sky = float((v >= SKIMMING_MIN).mean())
    wake = 1.0 - iso - sky
    return {"n": int(n), "isolated": iso, "wake": wake, "skimming": sky,
            "median": float(np.median(v))}


def site_lambda_f(site: str) -> np.ndarray:
    grid = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    )
    built = grid[(grid["building_count"] > 0)]
    return built["lambda_f_mean"].to_numpy()


def make_figure(per_site: dict[str, np.ndarray], pooled_skim: float) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xmax = 4.0
    # Regime bands behind everything.
    ax.axvspan(0, ISOLATED_MAX, color="#F2F2F2", zorder=0)
    ax.axvspan(ISOLATED_MAX, SKIMMING_MIN, color="#E4ECEF", zorder=0)
    ax.axvspan(SKIMMING_MIN, xmax, color="#CBDCE3", zorder=0)
    ax.axvline(ISOLATED_MAX, color="#999999", lw=0.6, ls=":")
    ax.axvline(SKIMMING_MIN, color="#2A6F8E", lw=1.0, ls="--")
    label_y = len(SITES) - 0.25
    for x, lab in [(0.075, "isolated"), (0.40, "wake"), (xmax * 0.7, "skimming flow")]:
        ax.text(x, label_y, lab, ha="center", va="bottom", fontsize=6.5,
                color="#2A6F8E" if lab.startswith("skim") else "#888888")

    positions = np.arange(len(SITES))[::-1]
    data = [np.clip(per_site[s][np.isfinite(per_site[s])], 0, xmax) for s in SITES]
    bp = ax.boxplot(data, positions=positions, vert=False, widths=0.55,
                    patch_artist=True, showfliers=False, whis=(5, 95))
    for patch, s in zip(bp["boxes"], SITES):
        patch.set_facecolor(SITE_COLORS[s])
        patch.set_alpha(0.85)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.6)
    for med in bp["medians"]:
        med.set_color("#111111")
        med.set_linewidth(1.1)
    for whisk in bp["whiskers"]:
        whisk.set_color("#555555")
        whisk.set_linewidth(0.6)
    for cap in bp["caps"]:
        cap.set_color("#555555")

    ax.set_yticks(positions)
    ax.set_yticklabels([SITE_LABELS[s] for s in SITES], fontsize=8)
    ax.set_ylim(-0.7, len(SITES) - 0.1)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(r"dissolved frontal-area density $\lambda_f$ (cell, 5–95% box)", fontsize=8)
    ax.set_title(
        f"Flow-regime classification — favela fabric sits in skimming flow "
        f"({pooled_skim * 100:.0f}% of cells ≥ 0.65)",
        loc="left", fontsize=8.5, pad=4,
    )
    ax.text(0.0, -0.22,
            "Geometry classifies the regime, not per-cell ventilation adequacy "
            "(that is CFD age-of-air τ, Tier 2). Oke/Grimmond–Oke thresholds.",
            transform=ax.transAxes, fontsize=6.0, color="#666666", va="top")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save_fig(fig, "lambda_f_regime")


def main() -> None:
    per_site_vals = {s: site_lambda_f(s) for s in SITES}
    per_site = {s: classify_regime(v) for s, v in per_site_vals.items()}
    pooled = classify_regime(np.concatenate([per_site_vals[s] for s in SITES]))
    for s in SITES:
        c = per_site[s]
        print(f"  {s:20s} median {c['median']:.2f}  "
              f"skimming {c['skimming'] * 100:.0f}%  wake {c['wake'] * 100:.0f}%  "
              f"isolated {c['isolated'] * 100:.0f}%")
    print(f"  POOLED median {pooled['median']:.2f}  skimming {pooled['skimming'] * 100:.0f}%")

    out = {
        "lambda_f": "dissolved (canonical, party-wall corrected)",
        "thresholds": {"isolated_max": ISOLATED_MAX, "skimming_min": SKIMMING_MIN},
        "reference": "Oke (1988) / Grimmond & Oke (1999) flow-regime boundaries",
        "per_site": per_site,
        "pooled": pooled,
    }
    out_json = PROJECT_ROOT / "outputs" / "paper_figures" / "lambda_f_regime.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  wrote {out_json.relative_to(PROJECT_ROOT)}")
    make_figure(per_site_vals, pooled["skimming"])


if __name__ == "__main__":
    main()
