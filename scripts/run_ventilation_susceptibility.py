"""2-D geometric ventilation-susceptibility map — vertical regime × lateral depth.

Roadmap #6. Two independent geometry-only signals, kept as SEPARATE axes (never
summed into a single index — the council was explicit):

  VERTICAL   — the Oke/Grimmond-&-Oke λf flow regime: isolated (<0.15) /
               wake (0.15–0.65) / skimming (≥0.65).
  HORIZONTAL — depth into contiguous fabric (distance to the nearest open edge,
               run_lateral_connectivity): shallow vs deep, split at the pooled median.

Their per-cell positive coupling (ρ ≈ +0.49) is the "doubly constrained" finding;
this promotes it to a 3×2 bivariate classification + map. The worst geometric
susceptibility is skimming × deep (vertically suppressed AND laterally buried), the
best is isolated × shallow.

STRICTLY GEOMETRIC SUSCEPTIBILITY — not air exchange, not an adequacy. Age-of-air τ
is CFD-gated and superseding; this is a pre-CFD prioritisation surface.

Outputs:
  outputs/paper_figures/ventilation_susceptibility.json
  outputs/paper_figures/exports/ventilation_susceptibility.png (+ .svg)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import SITE_LABELS, add_provenance, apply_style, save_fig  # noqa: E402

from scripts.run_lateral_connectivity import open_edge_distance  # noqa: E402

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
ISO_MAX, SKIM_MIN = 0.15, 0.65
REGIMES = ["isolated", "wake", "skimming"]
REGIME_FILL = {"isolated": "#FDD9A8", "wake": "#F08A3C", "skimming": "#B2182B"}


def _darken(hex_color: str, f: float = 0.5):
    return tuple(c * f for c in to_rgb(hex_color))


# 3×2 bivariate palette: regime hue × depth lightness (shallow=base, deep=darkened).
CLASS_COLOR = {}
for _r in REGIMES:
    CLASS_COLOR[(_r, "shallow")] = to_rgb(REGIME_FILL[_r])
    CLASS_COLOR[(_r, "deep")] = _darken(REGIME_FILL[_r], 0.55)


def built(site: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg")
    g = g[g["building_count"] > 0].copy()
    built_mask = (g["building_count"] > 0).to_numpy()
    g["open_edge_dist_m"] = open_edge_distance(
        g["centroid_x"].to_numpy(), g["centroid_y"].to_numpy(), built_mask)
    lf = g["lambda_f_mean"].to_numpy()
    reg = np.full(len(g), 1)  # wake
    reg[lf < ISO_MAX] = 0
    reg[lf >= SKIM_MIN] = 2
    g["regime_int"] = reg
    return g


def main() -> None:
    grids = {s: built(s) for s in SITES}
    depth_median = float(np.median(np.concatenate(
        [grids[s]["open_edge_dist_m"].to_numpy() for s in SITES])))
    for s in SITES:
        g = grids[s]
        g["depth_class"] = np.where(g["open_edge_dist_m"] >= depth_median, "deep", "shallow")
        g["regime"] = [REGIMES[i] for i in g["regime_int"]]
        g["rgb"] = list(zip(g["regime"], g["depth_class"]))

    classes = [(r, d) for r in REGIMES for d in ("shallow", "deep")]
    per_site, pooled_counts = {}, {c: 0 for c in classes}
    for s in SITES:
        g = grids[s]
        vc = g["rgb"].value_counts()
        n = len(g)
        shares = {f"{r}|{d}": float(vc.get((r, d), 0)) / n for r, d in classes}
        for c in classes:
            pooled_counts[c] += int(vc.get(c, 0))
        per_site[s] = {"n": n, "shares": shares,
                       "skimming_deep_frac": shares["skimming|deep"]}
    n_pool = sum(pooled_counts.values())
    pooled = {f"{r}|{d}": pooled_counts[(r, d)] / n_pool for r, d in classes}

    payload = {
        "title": "2-D geometric ventilation susceptibility — regime × depth (2026-06-25)",
        "axes": {"vertical": "λf flow regime (isolated/wake/skimming)",
                 "horizontal": f"open-edge depth, split at pooled median {depth_median:.0f} m"},
        "status": "STRICTLY GEOMETRIC susceptibility, pre-CFD; NOT air exchange / adequacy (τ CFD-gated)",
        "depth_median_m": depth_median,
        "worst_class": "skimming|deep (vertically suppressed AND laterally buried)",
        "per_site": per_site,
        "pooled_shares": pooled,
    }
    OUT = PROJECT_ROOT / "outputs" / "paper_figures" / "ventilation_susceptibility.json"
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT.relative_to(PROJECT_ROOT)}  (depth median {depth_median:.0f} m)\n")
    print(f"{'site':<22s} {'skim∩deep':>10s}")
    for s in SITES:
        print(f"{SITE_LABELS[s]:<22s} {per_site[s]['skimming_deep_frac']*100:>9.1f}%")
    print(f"{'POOLED':<22s} {pooled['skimming|deep']*100:>9.1f}%")

    apply_style()
    fig = plt.figure(figsize=(7.09, 3.5))
    gs = fig.add_gridspec(1, len(SITES) + 1, width_ratios=[1] * len(SITES) + [0.5],
                          wspace=0.06, top=0.78, bottom=0.04, left=0.01, right=0.99)
    for i, s in enumerate(SITES):
        ax = fig.add_subplot(gs[0, i])
        g = grids[s]
        g.plot(ax=ax, color=[CLASS_COLOR[c] for c in g["rgb"]], edgecolor="none", linewidth=0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{'(A) ' if i == 0 else ''}{SITE_LABELS[s]}\n"
                     f"skim∩deep {per_site[s]['skimming_deep_frac']*100:.0f}%", fontsize=6.2, pad=2)
    # 3×2 legend matrix
    lax = fig.add_subplot(gs[0, -1])
    lax.set_axis_off()
    handles = [Patch(facecolor=CLASS_COLOR[(r, d)], edgecolor="#444444", linewidth=0.4,
                     label=f"{r} · {d}") for r in REGIMES for d in ("shallow", "deep")]
    lax.legend(handles=handles, loc="center", fontsize=5.4, frameon=False,
               title="regime × depth", title_fontsize=6, labelspacing=0.5,
               handlelength=1.1, handleheight=1.1)
    fig.text(0.5, 0.95,
             "2-D geometric ventilation susceptibility — vertical λ$_f$ regime × lateral "
             "depth-to-open-edge\n(two separate geometry-only axes; worst = skimming × deep; "
             "pre-CFD, not air exchange)", ha="center", va="top", fontsize=7.0)
    add_provenance(fig)
    save_fig(fig, "ventilation_susceptibility", gate=True)


if __name__ == "__main__":
    main()
