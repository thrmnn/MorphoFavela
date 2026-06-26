#!/usr/bin/env python3
"""BRISA slide deck — fig_morpho_boxplots.png (slide 9, replaces 3-row ridges).

Same three metrics as fig_morpho_distributions_3row.py
(λf · SVF · slope) but rendered as side-by-side box plots — easier to
read at a glance than ridges.

Layout:
  * 3 panels left-to-right (λf · SVF · slope)
  * Each panel: 5 boxes, hillside-first ordering
    (Vidigal, Rocinha, Alemão, Rio das Pedras, Maré)
  * Hillside boxes: black outline + light grey fill
  * Flatland boxes: black outline + white fill + hatch
  * Median + IQR + whiskers; no outlier flier dots
  * Typology legend top-right of figure

Aspect ≈ 1.6. 200 DPI.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_morpho_boxplots.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
sys.path.insert(0, str(PAPER_FIG_DIR))
from _assets import BRISA_ASSETS_DIR
from fig_style import SITE_LABELS, apply_style

OUT_DIR = BRISA_ASSETS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
HILLSIDE = {"vidigal", "rocinha", "complexo_do_alemao"}

STYLE = {
    "hillside": dict(facecolor="#cfcfcf", edgecolor="#111111", hatch=None),
    "flatland": dict(facecolor="#ffffff", edgecolor="#111111", hatch="///"),
}


def load_grid(site: str) -> gpd.GeoDataFrame:
    p = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    grid = gpd.read_file(p)
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    return grid[built].copy()


def boxplot_panel(ax, data: dict, *, ylim, ylabel, fmt="{:.2f}"):
    positions = np.arange(len(SITES))
    arrays = []
    for s in SITES:
        v = data[s]
        v = v[(v >= ylim[0]) & (v <= ylim[1])]
        arrays.append(v)

    bp = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops=dict(color="#000000", linewidth=1.8),
        whiskerprops=dict(color="#111111", linewidth=1.0),
        capprops=dict(color="#111111", linewidth=1.0),
    )

    for i, (s, box) in enumerate(zip(SITES, bp["boxes"])):
        typology = "hillside" if s in HILLSIDE else "flatland"
        st = STYLE[typology]
        box.set(facecolor=st["facecolor"], edgecolor=st["edgecolor"],
                linewidth=1.2, hatch=st["hatch"])

    # Median labels above each box.
    for i, vals in enumerate(arrays):
        if len(vals) == 0:
            continue
        med = float(np.median(vals))
        ax.text(
            positions[i], med, fmt.format(med),
            ha="center", va="bottom",
            fontsize=9.5, color="#111111",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85),
            zorder=5,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [SITE_LABELS[s] for s in SITES],
        rotation=25, ha="right", fontsize=10.5,
    )
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=11.5)
    ax.tick_params(axis="y", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)
    ax.yaxis.grid(True, linestyle=":", color="#bbbbbb", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def main() -> None:
    apply_style()
    grids = {s: load_grid(s) for s in SITES}

    lambdaf_data = {s: grids[s]["lambda_f_mean"].dropna().values for s in SITES}
    svf_data = {s: grids[s]["svf"].dropna().values for s in SITES}
    slope_data = {s: grids[s]["slope_deg"].dropna().values for s in SITES}

    fig_w = 14.0
    fig_h = fig_w / 1.6  # ~8.75 in
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    boxplot_panel(
        axes[0], lambdaf_data,
        ylim=(0.0, 8.0),
        ylabel=r"$\lambda_f$ — frontal-area density",
        fmt="{:.2f}",
    )
    boxplot_panel(
        axes[1], svf_data,
        ylim=(0.0, 1.0),
        ylabel="Sky View Factor",
        fmt="{:.2f}",
    )
    boxplot_panel(
        axes[2], slope_data,
        ylim=(0.0, 45.0),
        ylabel="Slope (degrees)",
        fmt="{:.1f}°",
    )

    handles = [
        Patch(facecolor=STYLE["hillside"]["facecolor"],
              edgecolor=STYLE["hillside"]["edgecolor"],
              label="hillside (Vidigal, Rocinha, Alemão)"),
        Patch(facecolor=STYLE["flatland"]["facecolor"],
              edgecolor=STYLE["flatland"]["edgecolor"],
              hatch=STYLE["flatland"]["hatch"],
              label="flatland (Rio das Pedras, Maré)"),
    ]
    leg = fig.legend(
        handles=handles, loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=True, fontsize=10, title="Typology", title_fontsize=10.5,
        handlelength=2.2,
    )
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("#888888")
    leg.get_frame().set_facecolor("white")

    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.16, wspace=0.28)

    out = OUT_DIR / "fig_morpho_boxplots.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches=None)
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
