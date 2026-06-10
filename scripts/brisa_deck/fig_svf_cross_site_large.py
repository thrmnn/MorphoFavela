#!/usr/bin/env python3
"""BRISA slide deck — fig_svf_cross_site_large.png (slide 7).

ONE METRIC (SVF), FIVE LARGE PANELS, shared grayscale colorbar, with a single
ridge plot underneath summarising per-favela SVF distributions.

Each per-favela panel is rendered ~2x larger than fig03_ventilation_solar.png so
the urban fabric reads clearly at full slide width. 200 m scale bar on the first
panel for spatial anchoring.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_svf_cross_site_large.png
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
from fig_style import SITE_LABELS, apply_style  # noqa: E402

OUT_DIR = Path("/home/theo/brisa_paper/artifacts/slides/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]

SVF_CMAP = "Greys_r"  # lighter = higher SVF (more open)
SVF_VMIN, SVF_VMAX = 0.0, 1.0


def load_grid(site: str) -> gpd.GeoDataFrame:
    p = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    grid = gpd.read_file(p)
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    return grid[built].copy()


def choropleth(ax, gdf: gpd.GeoDataFrame, title: str, *, with_scalebar: bool = False) -> None:
    valid = gdf[gdf["svf"].notna()].copy()
    valid["svf"] = valid["svf"].clip(SVF_VMIN, SVF_VMAX)
    valid.plot(
        ax=ax,
        column="svf",
        cmap=SVF_CMAP,
        vmin=SVF_VMIN,
        vmax=SVF_VMAX,
        edgecolor="none",
        linewidth=0.0,
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
    ax.set_title(title, fontsize=11, color="#111111", pad=4)

    if with_scalebar:
        x0 = bbox[0] + pad
        y0 = bbox[1] + pad
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                200.0,
                (bbox[3] - bbox[1]) * 0.008,
                facecolor="black",
                edgecolor="black",
                linewidth=0.0,
                zorder=20,
                clip_on=False,
            )
        )
        ax.text(
            x0 + 100,
            y0 + (bbox[3] - bbox[1]) * 0.025,
            "200 m",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111111",
        )


def add_colorbar(fig, panel_axes) -> None:
    # Place a single tall colorbar to the right of the rightmost panel.
    right_pos = panel_axes[-1].get_position()
    cb_x = right_pos.x1 + 0.012
    cb_w = 0.014
    cb_y = right_pos.y0
    cb_h = right_pos.height
    cax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
    sm = ScalarMappable(norm=Normalize(vmin=SVF_VMIN, vmax=SVF_VMAX), cmap=SVF_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.5)
    cb.set_label("Sky View Factor (10 m grid)", fontsize=9, color="#111111", labelpad=4)
    cb.ax.tick_params(labelsize=8, color="#333333", length=2.5)


def ridge(ax, data: dict) -> None:
    n = len(SITES)
    offsets = np.arange(n)
    xlim = (0.0, 1.0)
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
        # short tick marking the median on each ridge
        ax.plot([median, median], [offsets[i], offsets[i] + height * 0.65],
                color="#000000", lw=1.2, solid_capstyle="butt")
        ax.text(median, offsets[i] + height * 0.72, f"{median:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="#111111")

    ax.set_yticks(offsets)
    ax.set_yticklabels([SITE_LABELS[s] for s in SITES], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel("Sky View Factor", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Per-favela SVF distribution (median marked)",
                 loc="left", fontsize=10, pad=4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)


def main() -> None:
    apply_style()
    grids = {s: load_grid(s) for s in SITES}

    fig_w = 16.0  # inches, landscape full-slide
    fig_h = fig_w / 1.6
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Manual layout so each map panel is sized to its own bbox aspect ratio,
    # then scaled to share a common max-area envelope (rather than fixed cell
    # height). This makes Vidigal read as large as Maré despite different
    # geographic extents.
    bboxes = {}
    for s in SITES:
        valid = grids[s][grids[s]["svf"].notna()]
        bboxes[s] = valid.total_bounds  # (minx, miny, maxx, maxy)

    aspects = {}
    for s in SITES:
        b = bboxes[s]
        w = b[2] - b[0]
        h = b[3] - b[1]
        aspects[s] = w / h  # >1 = wide, <1 = tall

    # Layout maths: row 0 occupies vertical band [bottom_band, top_panel]
    # Each panel gets a normalised area: scale heights so all panels target
    # the same diagonal length; very wide panels become a bit thinner, very
    # tall panels become a bit narrower. This is the natural "scale to fit"
    # for mixed-aspect small multiples.
    left, right = 0.02, 0.90
    top, bottom = 0.97, 0.42
    panel_band_h = top - bottom
    panel_band_w = right - left
    inter_pad = 0.012  # horizontal gap between panels in figure-fraction

    # Per-panel width proportional to sqrt(aspect) keeps area roughly equal.
    raw_widths = np.array([np.sqrt(aspects[s]) for s in SITES])
    raw_widths = raw_widths / raw_widths.sum()
    total_gap = inter_pad * (len(SITES) - 1)
    panel_widths = raw_widths * (panel_band_w - total_gap)

    panel_axes = []
    x = left
    for i, site in enumerate(SITES):
        # height fills the band (panels share a common vertical baseline)
        ax = fig.add_axes([x, bottom, panel_widths[i], panel_band_h])
        choropleth(ax, grids[site], SITE_LABELS[site], with_scalebar=(i == 0))
        panel_axes.append(ax)
        x += panel_widths[i] + inter_pad

    fig.canvas.draw()  # finalize positions before placing the cbar
    add_colorbar(fig, panel_axes)

    # Ridge row, full width below.
    ax_r = fig.add_axes([left, 0.07, panel_band_w, 0.28])
    data = {s: grids[s]["svf"].dropna().values for s in SITES}
    ridge(ax_r, data)

    out = OUT_DIR / "fig_svf_cross_site_large.png"
    fig.savefig(out, dpi=220, facecolor="white", bbox_inches=None)
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
