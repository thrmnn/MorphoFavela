#!/usr/bin/env python3
"""BRISA slide deck — fig_svf_streets_cross_site.png (slide 8).

Cross-site street-level SVF: 5 favelas in a unified 2-row layout, each
rendered from svf_streets_segments.gpkg with a shared colorbar and no
in-panel title clutter. Mirrors the layout logic of
fig_svf_cross_site_v2_2row.py but uses street segments (not grid cells)
and removes the ridge plot — the slide is dominated by the maps.

Layout:
  * Top row: Vidigal · Rocinha · Complexo do Alemão
  * Bottom row: Rio das Pedras · Maré (centred)
  * Shared vertical SVF colorbar on the right (Greys, vmin=0 vmax=1)
  * 200 m scalebar on the leftmost panel of each row
  * No in-figure title

Aspect ≈ 1.4. 200 DPI.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_svf_streets_cross_site.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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

SVF_CMAP = "Greys_r"  # lighter = higher SVF
SVF_VMIN, SVF_VMAX = 0.0, 1.0
SVF_COL = "svf_mean"


def load_segments(site: str) -> gpd.GeoDataFrame:
    p = PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_segments.gpkg"
    g = gpd.read_file(p)
    return g[g[SVF_COL].notna()].copy()


def streets_panel(ax, gdf: gpd.GeoDataFrame, title: str, *, with_scalebar: bool = False) -> None:
    valid = gdf.copy()
    valid[SVF_COL] = valid[SVF_COL].clip(SVF_VMIN, SVF_VMAX)
    # Dark backdrop helps streets read against light=high-SVF.
    bbox = valid.total_bounds
    pad = (bbox[2] - bbox[0]) * 0.03

    # Soft grey context rectangle to make light streets visible.
    ax.add_patch(
        plt.Rectangle(
            (bbox[0] - pad, bbox[1] - pad),
            (bbox[2] - bbox[0]) + 2 * pad,
            (bbox[3] - bbox[1]) + 2 * pad,
            facecolor="#2a2a2a", edgecolor="none", zorder=0,
        )
    )

    valid.plot(
        ax=ax,
        column=SVF_COL,
        cmap=SVF_CMAP,
        vmin=SVF_VMIN, vmax=SVF_VMAX,
        linewidth=1.6,
    )

    ax.set_xlim(bbox[0] - pad, bbox[2] + pad)
    ax.set_ylim(bbox[1] - pad, bbox[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Site label small, top-left, on dark backdrop.
    ax.text(
        0.02, 0.97, title,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13, color="white",
        bbox=dict(facecolor="#111111", edgecolor="none", pad=2.5, alpha=0.7),
    )

    if with_scalebar:
        x0 = bbox[0] + pad
        y0 = bbox[1] + pad
        bar_h = (bbox[3] - bbox[1]) * 0.012
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), 200.0, bar_h,
                facecolor="white", edgecolor="white", linewidth=0.0,
                zorder=20, clip_on=False,
            )
        )
        ax.text(
            x0 + 100, y0 + (bbox[3] - bbox[1]) * 0.035,
            "200 m", ha="center", va="bottom",
            fontsize=9.5, color="white",
        )


def add_colorbar(fig, top_anchor_ax, bot_anchor_ax) -> None:
    top_pos = top_anchor_ax.get_position()
    bot_pos = bot_anchor_ax.get_position()
    cb_x = top_pos.x1 + 0.014
    cb_w = 0.016
    cb_y = bot_pos.y0
    cb_h = top_pos.y1 - bot_pos.y0
    cax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
    sm = ScalarMappable(norm=Normalize(vmin=SVF_VMIN, vmax=SVF_VMAX), cmap=SVF_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.5)
    cb.set_label("Street-level Sky View Factor", fontsize=12, color="#111111", labelpad=8)
    cb.ax.tick_params(labelsize=10, color="#333333", length=2.5)


def main() -> None:
    apply_style()
    segs = {s: load_segments(s) for s in SITES}

    fig_w = 16.0
    fig_h = fig_w / 1.4  # ~11.4 in tall
    fig = plt.figure(figsize=(fig_w, fig_h))

    bboxes = {s: segs[s].total_bounds for s in SITES}
    aspects = {
        s: (bboxes[s][2] - bboxes[s][0]) / (bboxes[s][3] - bboxes[s][1])
        for s in SITES
    }

    left, right = 0.025, 0.89
    band_w = right - left

    # Two map rows occupy nearly the full figure height.
    map_top = 0.985
    map_bottom = 0.03
    row_gap = 0.025
    row_h = (map_top - map_bottom - row_gap) / 2.0
    top_row_y = map_bottom + row_h + row_gap
    bot_row_y = map_bottom

    inter_pad = 0.012

    def allocate_widths(sites, band_width):
        raw = np.array([np.sqrt(aspects[s]) for s in sites])
        total_gap = inter_pad * (len(sites) - 1)
        raw_n = raw / raw.sum()
        return raw_n * (band_width - total_gap)

    top_widths = allocate_widths(ROW_TOP, band_w)
    bot_band_w = band_w * 0.72
    bot_widths = allocate_widths(ROW_BOT, bot_band_w)
    bot_left = left + (band_w - bot_band_w) / 2.0

    top_axes = []
    x = left
    for i, s in enumerate(ROW_TOP):
        ax = fig.add_axes([x, top_row_y, top_widths[i], row_h])
        streets_panel(ax, segs[s], SITE_LABELS[s], with_scalebar=(i == 0))
        top_axes.append(ax)
        x += top_widths[i] + inter_pad

    bot_axes = []
    x = bot_left
    for i, s in enumerate(ROW_BOT):
        ax = fig.add_axes([x, bot_row_y, bot_widths[i], row_h])
        streets_panel(ax, segs[s], SITE_LABELS[s], with_scalebar=(i == 0))
        bot_axes.append(ax)
        x += bot_widths[i] + inter_pad

    fig.canvas.draw()
    add_colorbar(fig, top_axes[-1], bot_axes[-1])

    out = OUT_DIR / "fig_svf_streets_cross_site.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches=None)
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
