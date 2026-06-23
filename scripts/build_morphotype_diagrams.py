"""Explainer visuals for the morphotype signature (track/viz):

1. morphotype_schematics.png — an idealized street section per type (the
   "hypothetical scenario": density, height spread, slope, canyon depth at a glance).
2. composition_by_site.png — % of each morphotype per favela (stacked bars).
3. map_<site>.png — a per-favela full morphotype map (campaign sites).

Driven by the canonical fabric centroids + the written-back morphotype_smooth
labels. Decisions: docs/morpho_signature_decisions.md.

    python scripts/build_morphotype_diagrams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from src.morphometry.signature import CAMPAIGN_SITES  # noqa: E402
from src.viz.signature_style import NULL_COLOR, TYPE_COLORS, TYPE_LABEL  # noqa: E402

FIGS = ROOT / "outputs" / "cross_site" / "signature" / "figures_v2"
SITE_NAMES = {"vidigal": "Vidigal", "rocinha": "Rocinha", "riodaspedras": "Rio das Pedras",
              "complexo_do_alemao": "Complexo do Alemão", "maré": "Maré"}

# centroid drivers (median, original units): λp, H_mean, σH, slope°, H/W
CENTROIDS = {
    0: dict(lp=0.22, H=4.35, sH=0.0, slope=3.0, hw=0.35),
    1: dict(lp=0.51, H=4.57, sH=1.30, slope=18.6, hw=0.86),
    2: dict(lp=0.68, H=6.83, sH=1.34, slope=1.4, hw=0.79),
    3: dict(lp=0.69, H=7.89, sH=2.06, slope=1.4, hw=1.49),
    4: dict(lp=0.78, H=7.76, sH=2.00, slope=19.8, hw=2.58),
    5: dict(lp=1.0, H=7.46, sH=2.09, slope=2.6, hw=3.47),
}
# deterministic height multipliers (mean 1.0) so σH shows without RNG
_HSEQ = np.array([0.78, 1.18, 0.88, 1.22, 0.82, 1.12, 0.95, 1.05])


def schematic():
    """Idealized 12 m-deep street section per type — density, σH, slope, canyon."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    W = 40.0  # section width (m)
    for c, ax in zip(range(6), axes.flat):
        p = CENTROIDS[c]
        n = max(2, int(round(p["lp"] * 8)))      # building count from coverage
        gap = W / n
        bw = gap * p["lp"]                         # footprint width from λp
        tan = np.tan(np.deg2rad(p["slope"]))
        for i in range(n):
            x = i * gap + (gap - bw) / 2
            g0 = x * tan                           # ground rises with slope
            h = p["H"] * (1 + (_HSEQ[i % 8] - 1) * (p["sH"] / max(p["H"], 1e-6)) * 3)
            h = max(h, 1.5)
            ax.add_patch(Rectangle((x, g0), bw, h, facecolor=TYPE_COLORS[c],
                                   edgecolor="0.25", lw=0.6))
        # ground line
        ax.add_patch(Polygon([(0, 0), (W, W * tan), (W, -3), (0, -3)],
                             facecolor="#efe7da", edgecolor="none", zorder=0))
        ax.plot([0, W], [0, W * tan], color="0.4", lw=1)
        ax.set_xlim(0, W)
        ax.set_ylim(-3, 26)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"T{c} · {TYPE_LABEL[c]}", fontsize=10, color=TYPE_COLORS[c],
                     fontweight="bold")
        ax.text(0.5, 0.93, f"λp {p['lp']:.2f} · H {p['H']:.0f} m · σH {p['sH']:.1f} · "
                f"slope {p['slope']:.0f}° · H/W {p['hw']:.1f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7, color="0.3")
    fig.suptitle("Idealized morphotype sections — a 40 m street slice of each type "
                 "(schematic, driven by the cluster centroids)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "morphotype_schematics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def composition(per_site):
    """Stacked % of each morphotype per favela."""
    sites = [s for s in CAMPAIGN_SITES if s in per_site]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bottom = np.zeros(len(sites))
    for c in range(6):
        vals = np.array([per_site[s].get(c, 0.0) for s in sites]) * 100
        ax.barh(range(len(sites)), vals, left=bottom, color=TYPE_COLORS[c],
                label=f"T{c} {TYPE_LABEL[c]}")
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 6:
                ax.text(b + v / 2, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += vals
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels([SITE_NAMES[s] for s in sites])
    ax.set_xlabel("% of built cells")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_title("Morphotype composition per favela (%)", fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(FIGS / "composition_by_site.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_favela_maps():
    cmap = ListedColormap([TYPE_COLORS[c] for c in range(6)])
    per_site = {}
    for s in CAMPAIGN_SITES:
        p = ROOT / "outputs" / s / "features" / "features_grid.parquet"
        if not p.exists():
            continue
        g = gpd.read_parquet(p)
        col = "morphotype_smooth" if "morphotype_smooth" in g else "morphotype"
        gg = g.dropna(subset=[col])
        per_site[s] = (gg[col].value_counts(normalize=True)).to_dict()
        fig, ax = plt.subplots(figsize=(6, 6))
        g.plot(ax=ax, color=NULL_COLOR, linewidth=0)
        diss = gg.dissolve(by=col).reset_index().explode(index_parts=False)
        diss = diss[diss.geometry.area >= 10 * float(g.geometry.area.median())]
        diss.plot(ax=ax, column=col, categorical=True, cmap=cmap, vmin=0, vmax=5,
                  edgecolor="white", linewidth=0.25)
        gpd.GeoSeries([g.geometry.union_all()], crs=g.crs).boundary.plot(
            ax=ax, color="0.25", lw=0.6)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{SITE_NAMES[s]} — morphotypes", fontsize=12)
        handles = [Rectangle((0, 0), 1, 1, color=TYPE_COLORS[c]) for c in range(6)]
        ax.legend(handles, [f"T{c} {TYPE_LABEL[c]}" for c in range(6)],
                  fontsize=7, loc="lower right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(FIGS / f"map_{s}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    return per_site


def main():
    schematic()
    per_site = per_favela_maps()
    composition(per_site)
    pd.DataFrame(per_site).T.reindex(columns=range(6)).to_csv(
        ROOT / "outputs" / "cross_site" / "signature" / "composition_by_site.csv")
    print("schematics + composition + per-favela maps written")
    for s, comp in per_site.items():
        print(f"{s:20s} " + " ".join(f"T{c}:{comp.get(c, 0)*100:4.0f}%" for c in range(6)))


if __name__ == "__main__":
    main()
