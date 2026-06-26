#!/usr/bin/env python3
"""BRISA slide deck — fig_lambdaf_cross_site_v2_2row.png (λf sibling of SVF v2).

Same 2-row layout as fig_svf_cross_site_v2_2row.py, but for λf (frontal-area
density, 8-direction mean) instead of SVF.

  * Top row: Vidigal · Rocinha · Complexo do Alemão (3 hillside)
  * Bottom row: Rio das Pedras · Maré (2 flatland), centred
  * Shared λf colorbar on the right, spanning both rows
  * 200 m scale bar on each row's leftmost panel
  * Per-favela ridge plot below the maps with vertical dashed line at
    λf = 0.35 (skimming-flow regime threshold, Grimmond & Oke 1999).

Greyscale colormap (darker = higher λf = denser canopy = poorer ventilation),
inverse semantics to the SVF figure (darker = lower SVF = enclosed canyon).

Aspect ≈ 1.4 (slightly wider than tall). 200 DPI.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_lambdaf_cross_site_v2_2row.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
sys.path.insert(0, str(PAPER_FIG_DIR))
from _assets import BRISA_ASSETS_DIR
from fig_style import SITE_LABELS, apply_style

OUT_DIR = BRISA_ASSETS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
ROW_TOP = ["vidigal", "rocinha", "complexo_do_alemao"]
ROW_BOT = ["riodaspedras", "maré"]

LAMBDA_COL = "lambda_f_mean"
LAMBDA_CMAP = "Greys"  # darker = higher λf
# Data range: medians 2.4–5.6, p90 ~5–9 across sites. vmax=8 lets the
# choropleths show variation; the 0.35 threshold sits in the low tail.
LAMBDA_VMIN, LAMBDA_VMAX = 0.0, 8.0
THRESHOLD_LAMBDA_F = 0.35
RIDGE_XMAX = 10.0


def load_grid(site: str) -> gpd.GeoDataFrame:
    p = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    grid = gpd.read_file(p)
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    return grid[built].copy()


def choropleth(ax, gdf: gpd.GeoDataFrame, title: str, *, with_scalebar: bool = False) -> None:
    valid = gdf[gdf[LAMBDA_COL].notna()].copy()
    valid[LAMBDA_COL] = valid[LAMBDA_COL].clip(LAMBDA_VMIN, LAMBDA_VMAX)
    valid.plot(
        ax=ax, column=LAMBDA_COL, cmap=LAMBDA_CMAP,
        vmin=LAMBDA_VMIN, vmax=LAMBDA_VMAX,
        edgecolor="none", linewidth=0.0,
    )
    bbox = valid.total_bounds
    pad = (bbox[2] - bbox[0]) * 0.03
    ax.set_xlim(bbox[0] - pad, bbox[2] + pad)
    ax.set_ylim(bbox[1] - pad, bbox[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=13, color="#111111", pad=4)

    if with_scalebar:
        x0 = bbox[0] + pad
        y0 = bbox[1] + pad
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), 200.0, (bbox[3] - bbox[1]) * 0.010,
                facecolor="black", edgecolor="black", linewidth=0.0,
                zorder=20, clip_on=False,
            )
        )
        ax.text(
            x0 + 100, y0 + (bbox[3] - bbox[1]) * 0.030,
            "200 m", ha="center", va="bottom",
            fontsize=9, color="#111111",
        )


def add_colorbar(fig, top_anchor_ax, bot_anchor_ax) -> None:
    top_pos = top_anchor_ax.get_position()
    bot_pos = bot_anchor_ax.get_position()
    cb_x = top_pos.x1 + 0.012
    cb_w = 0.014
    cb_y = bot_pos.y0
    cb_h = top_pos.y1 - bot_pos.y0
    cax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
    sm = ScalarMappable(norm=Normalize(vmin=LAMBDA_VMIN, vmax=LAMBDA_VMAX), cmap=LAMBDA_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.5)
    cb.set_label(r"$\lambda_f$ (8-direction mean, 10 m grid)",
                 fontsize=11, color="#111111", labelpad=6)
    cb.ax.tick_params(labelsize=10, color="#333333", length=2.5)


def ridge(ax, data: dict) -> None:
    n = len(SITES)
    offsets = np.arange(n)
    xlim = (0.0, RIDGE_XMAX)
    height = 0.9
    for i, site in enumerate(SITES):
        vals = data[site]
        vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
        if len(vals) < 30:
            continue
        kde = gaussian_kde(vals, bw_method=0.22)
        xs = np.linspace(xlim[0], xlim[1], 400)
        density = kde(xs)
        density = density / density.max() * height
        ax.fill_between(xs, offsets[i], offsets[i] + density,
                        color="#bbbbbb", alpha=0.85, linewidth=0)
        ax.plot(xs, offsets[i] + density, color="#222222", lw=0.7)
        median = float(np.median(vals))
        ax.plot([median, median], [offsets[i], offsets[i] + height * 0.65],
                color="#000000", lw=1.3, solid_capstyle="butt")
        ax.text(median, offsets[i] + height * 0.72, f"{median:.2f}",
                ha="center", va="bottom", fontsize=9.5, color="#111111")

    # Skimming-flow regime threshold (Grimmond & Oke 1999).
    # All five favelas sit deep into skimming-flow territory — the threshold
    # is plotted to anchor the finding, not to differentiate sites.
    ax.axvline(THRESHOLD_LAMBDA_F, color="#b00000", lw=1.2, ls="--",
               alpha=0.95, zorder=5)
    # Anchor the 0.35 threshold call-out near the bottom of the ridge stack
    # so it does not collide with the per-site distributions or the title.
    ax.text(THRESHOLD_LAMBDA_F + 0.15, n - 0.35,
            r"$\lambda_f = 0.35$" "\nskimming-flow"
            "\nthreshold",
            fontsize=9.0, ha="left", va="top", color="#b00000",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#b00000", linewidth=0.6))

    ax.set_yticks(offsets)
    ax.set_yticklabels([SITE_LABELS[s] for s in SITES], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_ylim(n - 0.15, -0.85)  # leave headroom for the threshold label
    ax.set_xlabel(r"$\lambda_f$ (frontal-area density, 8-direction mean)",
                  fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_title(r"Per-favela $\lambda_f$ distribution (median marked)",
                 loc="left", fontsize=12, pad=4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)


def main() -> None:
    apply_style()
    grids = {s: load_grid(s) for s in SITES}

    fig_w = 14.0
    fig_h = fig_w / 1.4  # ~10 in tall
    fig = plt.figure(figsize=(fig_w, fig_h))

    bboxes = {}
    for s in SITES:
        valid = grids[s][grids[s][LAMBDA_COL].notna()]
        bboxes[s] = valid.total_bounds
    aspects = {s: (bboxes[s][2] - bboxes[s][0]) / (bboxes[s][3] - bboxes[s][1]) for s in SITES}

    left, right = 0.03, 0.88
    cb_pad = 0.005
    band_w = right - left
    map_top = 0.97
    map_bottom = 0.40
    row_gap = 0.025
    row_h = (map_top - map_bottom - row_gap) / 2.0
    top_row_y = map_bottom + row_h + row_gap
    bot_row_y = map_bottom

    inter_pad = 0.015

    def allocate_widths(sites, band_width):
        raw = np.array([np.sqrt(aspects[s]) for s in sites])
        total_gap = inter_pad * (len(sites) - 1)
        raw_n = raw / raw.sum()
        return raw_n * (band_width - total_gap)

    top_widths = allocate_widths(ROW_TOP, band_w)
    bot_band_w = band_w * 0.72
    bot_widths = allocate_widths(ROW_BOT, bot_band_w)
    bot_left = left + (band_w - bot_band_w) / 2.0 + cb_pad / 2

    top_axes = []
    x = left
    for i, s in enumerate(ROW_TOP):
        ax = fig.add_axes([x, top_row_y, top_widths[i], row_h])
        choropleth(ax, grids[s], SITE_LABELS[s], with_scalebar=(i == 0))
        top_axes.append(ax)
        x += top_widths[i] + inter_pad

    bot_axes = []
    x = bot_left
    for i, s in enumerate(ROW_BOT):
        ax = fig.add_axes([x, bot_row_y, bot_widths[i], row_h])
        choropleth(ax, grids[s], SITE_LABELS[s], with_scalebar=(i == 0))
        bot_axes.append(ax)
        x += bot_widths[i] + inter_pad

    fig.canvas.draw()
    add_colorbar(fig, top_axes[-1], bot_axes[-1])

    ridge_h = 0.32
    ridge_bot = 0.04
    ridge_top = ridge_bot + ridge_h
    if ridge_top > map_bottom - 0.005:
        ridge_h = (map_bottom - 0.005) - ridge_bot
    ridge_left = 0.085
    ridge_right = right
    ax_r = fig.add_axes([ridge_left, ridge_bot, ridge_right - ridge_left, ridge_h])
    data = {s: grids[s][LAMBDA_COL].dropna().values for s in SITES}
    ridge(ax_r, data)

    out = OUT_DIR / "fig_lambdaf_cross_site_v2_2row.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches=None)
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
