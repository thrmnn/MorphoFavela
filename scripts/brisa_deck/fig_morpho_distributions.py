#!/usr/bin/env python3
"""BRISA slide deck — fig_morpho_distributions.png (slide 8, REPLACES scatter).

Per-favela distribution comparison: SVF on the top row, slope on the bottom
row. Five ridge/KDE distributions per row (Vidigal, Rocinha, Alemão, Rio das
Pedras, Maré), hillside-first, with typology style-coded:
  * Hillside (Vidigal, Rocinha, Alemão): solid black-filled KDEs.
  * Flatland (Rio das Pedras, Maré): grey + hatched fill.

Each ridge has its median marked + numerically labelled. Shared x-axis per row.
A short typology call-out is annotated under each row.

Aspect ≈ 1.6 (wider than tall). The side-by-side ridge view reads instantly,
which the SVF×slope scatter did not.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_morpho_distributions.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

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
    "hillside": dict(facecolor="#222222", alpha=0.78, hatch=None, edgecolor="#111111"),
    "flatland": dict(facecolor="#cfcfcf", alpha=0.85, hatch="///", edgecolor="#444444"),
}


def load_grid(site: str) -> gpd.GeoDataFrame:
    p = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    grid = gpd.read_file(p)
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    return grid[built].copy()


def ridge_row(ax, data: dict, *, xlim, xlabel, bw=0.22, fmt="{:.2f}", height=0.85):
    n = len(SITES)
    offsets = np.arange(n)
    xs = np.linspace(xlim[0], xlim[1], 500)
    for i, site in enumerate(SITES):
        vals = data[site]
        vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
        if len(vals) < 30:
            continue
        kde = gaussian_kde(vals, bw_method=bw)
        density = kde(xs)
        if density.max() == 0:
            continue
        density = density / density.max() * height
        typology = "hillside" if site in HILLSIDE else "flatland"
        st = STYLE[typology]
        ax.fill_between(
            xs, offsets[i], offsets[i] + density,
            facecolor=st["facecolor"],
            alpha=st["alpha"],
            hatch=st["hatch"],
            edgecolor=st["edgecolor"],
            linewidth=0.6,
        )
        ax.plot(xs, offsets[i] + density, color="#111111", lw=0.7)
        median = float(np.median(vals))
        ax.plot([median, median], [offsets[i], offsets[i] + height * 0.7],
                color="white" if typology == "hillside" else "black",
                lw=1.6, solid_capstyle="butt", zorder=5)
        ax.text(median, offsets[i] + height * 0.78, fmt.format(median),
                ha="center", va="bottom", fontsize=9.5, color="#111111",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))

    ax.set_yticks(offsets)
    ax.set_yticklabels([SITE_LABELS[s] for s in SITES], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)


def main() -> None:
    apply_style()
    grids = {s: load_grid(s) for s in SITES}

    svf_data = {s: grids[s]["svf"].dropna().values for s in SITES}
    slope_data = {s: grids[s]["slope_deg"].dropna().values for s in SITES}

    fig_w = 13.0
    fig_h = fig_w / 1.6  # ~8.1 in
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Two stacked rows.
    left, right = 0.10, 0.97
    row_w = right - left
    top_y = 0.55
    bot_y = 0.08
    row_h = 0.36

    ax_top = fig.add_axes([left, top_y, row_w, row_h])
    ax_bot = fig.add_axes([left, bot_y, row_w, row_h])

    ridge_row(ax_top, svf_data, xlim=(0.0, 1.0), xlabel="Sky View Factor", bw=0.22, fmt="{:.2f}")
    ridge_row(ax_bot, slope_data, xlim=(0.0, 45.0), xlabel="Slope (degrees)", bw=0.30, fmt="{:.1f}°")

    ax_top.set_title("Sky View Factor — per favela", loc="left", fontsize=12.5, pad=6, color="#111111")
    ax_bot.set_title("Slope — per favela", loc="left", fontsize=12.5, pad=6, color="#111111")

    # Typology annotation under each row.
    ax_top.text(
        0.99, -0.18,
        "Hillside cores (Vidigal, Rocinha, Alemão) extend to SVF < 0.2; flatland fabrics (Rio das Pedras, Maré) sit above 0.4.",
        transform=ax_top.transAxes, ha="right", va="top",
        fontsize=10.0, color="#333333", style="italic",
    )
    ax_bot.text(
        0.99, -0.18,
        "Vidigal / Rocinha / Alemão extend past 30°; Rio das Pedras / Maré cluster around 0°.",
        transform=ax_bot.transAxes, ha="right", va="top",
        fontsize=10.0, color="#333333", style="italic",
    )

    # Typology legend (top right of figure).
    handles = [
        Patch(facecolor=STYLE["hillside"]["facecolor"], edgecolor=STYLE["hillside"]["edgecolor"],
              alpha=STYLE["hillside"]["alpha"], label="hillside (Vidigal, Rocinha, Alemão)"),
        Patch(facecolor=STYLE["flatland"]["facecolor"], edgecolor=STYLE["flatland"]["edgecolor"],
              alpha=STYLE["flatland"]["alpha"], hatch=STYLE["flatland"]["hatch"],
              label="flatland (Rio das Pedras, Maré)"),
    ]
    leg = fig.legend(handles=handles, loc="upper right",
                     bbox_to_anchor=(0.98, 0.99),
                     frameon=True, fontsize=10, title="Typology", title_fontsize=10.5,
                     handlelength=2.0)
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("#888888")
    leg.get_frame().set_facecolor("white")

    out = OUT_DIR / "fig_morpho_distributions.png"
    fig.savefig(out, dpi=220, facecolor="white", bbox_inches=None)
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
