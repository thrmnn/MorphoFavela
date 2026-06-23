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


def _shade(hex_color, factor):
    """Darken (factor<1) or lighten (factor>1) a hex colour for roof/shadow."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    if factor < 1:
        return (r * factor, g * factor, b * factor)
    return tuple(min(1, x + (1 - x) * (factor - 1)) for x in (r, g, b))


def _person(ax, x, y):
    """A 1.7 m human silhouette for scale."""
    ax.add_patch(plt.Circle((x, y + 1.45), 0.32, color="#33373c", zorder=6))
    ax.add_patch(Rectangle((x - 0.3, y), 0.6, 1.2, color="#33373c", zorder=6))


def schematic():
    """Idealized street section per type — same scale across panels so heights,
    density, slope and canyon depth are directly comparable. Designer treatment:
    sky, earth, roof shading, a person for scale, and the H/W canyon dimension."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    W, YMAX = 42.0, 30.0
    for c, ax in zip(range(6), axes.flat):
        p = CENTROIDS[c]
        col = TYPE_COLORS[c]
        tan = np.tan(np.deg2rad(p["slope"]))
        ax.add_patch(Rectangle((0, -4), W, YMAX + 4, facecolor="#eaf2f7",  # sky
                               edgecolor="none", zorder=0))
        ax.add_patch(Polygon([(0, 0), (W, W * tan), (W, -4), (0, -4)],     # earth
                             facecolor="#d8cbb6", edgecolor="none", zorder=1))
        ax.plot([0, W], [0, W * tan], color="#9b8a6b", lw=1.4, zorder=2)
        ax.plot([2, 4.4], [YMAX - 2.5, YMAX - 2.5], color="#f3b340", lw=2.4)  # sun ray
        ax.add_patch(plt.Circle((1.2, YMAX - 2.5), 1.0, color="#f6c659", zorder=2))
        n = max(2, int(round(p["lp"] * 8)))
        gap = W / n
        bw = gap * p["lp"]
        tops = []
        for i in range(n):
            x = i * gap + (gap - bw) / 2
            g0 = x * tan
            h = max(p["H"] * (1 + (_HSEQ[i % 8] - 1) * (p["sH"] / max(p["H"], 1e-6)) * 3), 2)
            ax.add_patch(Rectangle((x, g0), bw, h, facecolor=col,
                                   edgecolor=_shade(col, 0.5), lw=0.8, zorder=4))
            ax.add_patch(Rectangle((x, g0 + h - 0.7), bw, 0.7,        # roof band
                                   facecolor=_shade(col, 0.7), edgecolor="none", zorder=5))
            tops.append((x, x + bw, g0 + h))
        # widest gap → a clean canyon dimension arrow + a person standing in it
        gaps = [(tops[i][0] - tops[i - 1][1], tops[i - 1][1], tops[i][0],
                 min(tops[i - 1][2], tops[i][2])) for i in range(1, len(tops))]
        if gaps:
            wid, gx0, gx1, gtop = max(gaps, key=lambda t: t[0])
            ymid = (gx0 + gx1) / 2
            _person(ax, ymid, ymid * tan)
            if wid > 2.0:                                  # only annotate a readable gap
                yb = gtop + 1.2
                ax.annotate("", (gx0, yb), (gx1, yb),
                            arrowprops=dict(arrowstyle="<->", color="#b3261e", lw=1.1))
        # 10 m scale tick (left, always clear)
        ax.plot([1.0, 1.0], [12, 22], color="0.35", lw=1.4)
        ax.text(1.5, 17, "10 m", rotation=90, va="center", ha="left", fontsize=6.5,
                color="0.35")
        ax.set_xlim(0, W)
        ax.set_ylim(-4, YMAX)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("0.8")
        ax.set_title(f"T{c} · {TYPE_LABEL[c]}", fontsize=11, color=_shade(col, 0.75),
                     fontweight="bold", pad=2)
        ax.text(0.5, 0.965, f"λp {p['lp']:.2f}  ·  H {p['H']:.0f} m  ·  σH {p['sH']:.1f}"
                f"  ·  slope {p['slope']:.0f}°  ·  H/W {p['hw']:.1f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7, color="0.35")
    fig.suptitle("The six morphotypes as idealized street sections — same scale, "
                 "so density · height spread · slope · canyon depth compare directly",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "morphotype_schematics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _text_on(hex_color):
    """Black or white label, whichever has contrast on the segment colour."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.6 else "#1d1d1f"


# topography-ordered (hillside group, then flat group) for a readable comparison
_COMP_ORDER = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
_HILLSIDE = {"vidigal", "rocinha", "complexo_do_alemao"}


def composition(per_site):
    """100%-stacked composition per favela — white separators, direct labels,
    grouped hillside-then-flat, named legend strip."""
    sites = [s for s in _COMP_ORDER if s in per_site]
    ypos = []
    y = 0
    for s in sites:                       # gap between the hillside and flat groups
        if s == "riodaspedras":
            y += 0.6
        ypos.append(y)
        y += 1
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, s in enumerate(sites):
        left = 0.0
        for c in range(6):
            v = per_site[s].get(c, 0.0) * 100
            if v <= 0:
                continue
            ax.barh(ypos[i], v, left=left, color=TYPE_COLORS[c], height=0.78,
                    edgecolor="white", linewidth=1.6, zorder=3)
            if v >= 4:
                ax.text(left + v / 2, ypos[i], f"{v:.0f}", ha="center", va="center",
                        fontsize=7.5, color=_text_on(TYPE_COLORS[c]), fontweight="bold")
            left += v
    ax.set_yticks(ypos)
    ax.set_yticklabels([SITE_NAMES[s] for s in sites], fontsize=10)
    for tick, s in zip(ax.get_yticklabels(), sites):
        tick.set_color("#8a5a00" if s in _HILLSIDE else "#1f5fa8")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of built cells")
    ax.invert_yaxis()
    handles = [Rectangle((0, 0), 1, 1, color=TYPE_COLORS[c]) for c in range(6)]
    ax.legend(handles, [f"T{c} {TYPE_LABEL[c]}" for c in range(6)], fontsize=8,
              ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              handlelength=1.1, columnspacing=1.0, frameon=False)
    ax.set_title("Morphotype composition per favela — hillside (amber labels) is "
                 "T1/T4-heavy, flat (blue) is T3/T5", fontsize=9.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
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
