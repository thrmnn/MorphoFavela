#!/usr/bin/env python3
"""BRISA slide deck — fig_vidigal_lambdaf_streets.png.

Vidigal-only λf street-level map (single panel, landscape).
Companion to svf_streets_vidigal.png — same framing and aspect.

Renders the 10 m grid choropleth of λf_mean (frontal-area density,
8-direction mean) with building footprints as a low-contrast backdrop,
scale bar, north arrow, and a vertical colorbar on the right.

Aspect ≈ 1.9 (landscape). 200 DPI. No in-figure title — the figure
sits on the right of a slide alongside the λf method schematic.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_vidigal_lambdaf_streets.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
sys.path.insert(0, str(PAPER_FIG_DIR))
from fig_style import apply_style  # noqa: E402

OUT_DIR = Path("/home/theo/brisa_paper/artifacts/slides/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_COL = "lambda_f_mean"
LAMBDA_CMAP = "Greys"
LAMBDA_VMIN, LAMBDA_VMAX = 0.0, 8.0


def main() -> None:
    apply_style()

    grid_path = PROJECT_ROOT / "outputs" / "vidigal" / "morphometrics" / "grid" / "grid_metrics.gpkg"
    grid = gpd.read_file(grid_path)
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    grid = grid[built & grid[LAMBDA_COL].notna()].copy()
    grid[LAMBDA_COL] = grid[LAMBDA_COL].clip(LAMBDA_VMIN, LAMBDA_VMAX)

    footprints = None
    fp_path = PROJECT_ROOT / "data" / "vidigal" / "buildings_extended_300m.gpkg"
    if fp_path.exists():
        footprints = gpd.read_file(fp_path)
        if footprints.crs != grid.crs:
            footprints = footprints.to_crs(grid.crs)

    fig_w = 13.0
    fig_h = fig_w / 1.9  # ~6.84 in -> landscape
    fig = plt.figure(figsize=(fig_w, fig_h))

    left, right = 0.04, 0.88
    bottom, top = 0.07, 0.97
    ax = fig.add_axes([left, bottom, right - left, top - bottom])

    if footprints is not None and len(footprints) > 0:
        bbox = grid.total_bounds
        fp_clip = footprints.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        fp_clip.plot(ax=ax, facecolor="#e8e8e8", edgecolor="#bbbbbb",
                     linewidth=0.15, zorder=1)

    grid.plot(
        ax=ax, column=LAMBDA_COL, cmap=LAMBDA_CMAP,
        vmin=LAMBDA_VMIN, vmax=LAMBDA_VMAX,
        edgecolor="none", linewidth=0.0, alpha=0.92, zorder=2,
    )

    bbox = grid.total_bounds
    pad_x = (bbox[2] - bbox[0]) * 0.02
    pad_y = (bbox[3] - bbox[1]) * 0.04
    ax.set_xlim(bbox[0] - pad_x, bbox[2] + pad_x)
    ax.set_ylim(bbox[1] - pad_y, bbox[3] + pad_y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Scale bar — 200 m, bottom-left.
    x0 = bbox[0] + (bbox[2] - bbox[0]) * 0.02
    y0 = bbox[1] + (bbox[3] - bbox[1]) * 0.04
    bar_h = (bbox[3] - bbox[1]) * 0.012
    ax.add_patch(plt.Rectangle((x0, y0), 200.0, bar_h,
                               facecolor="black", edgecolor="black",
                               linewidth=0.0, zorder=20))
    ax.text(x0 + 100, y0 + bar_h + (bbox[3] - bbox[1]) * 0.015,
            "200 m", ha="center", va="bottom",
            fontsize=10, color="#111111", zorder=20)

    # North arrow — top-right of map.
    nx = bbox[2] - (bbox[2] - bbox[0]) * 0.04
    ny = bbox[3] - (bbox[3] - bbox[1]) * 0.06
    arrow_h = (bbox[3] - bbox[1]) * 0.10
    ax.annotate("N", xy=(nx, ny), xytext=(nx, ny - arrow_h),
                ha="center", va="center", fontsize=12, color="#111111",
                arrowprops=dict(arrowstyle="->", color="#111111", lw=1.4),
                zorder=20)

    # Colorbar on the right.
    cb_x = right + 0.015
    cb_w = 0.018
    cax = fig.add_axes([cb_x, bottom + 0.03, cb_w, (top - bottom) - 0.06])
    sm = ScalarMappable(norm=Normalize(vmin=LAMBDA_VMIN, vmax=LAMBDA_VMAX), cmap=LAMBDA_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.5)
    cb.set_label(r"$\lambda_f$ (8-direction mean, 10 m grid)",
                 fontsize=11, color="#111111", labelpad=8)
    cb.ax.tick_params(labelsize=10, color="#333333", length=2.5)

    out = OUT_DIR / "fig_vidigal_lambdaf_streets.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches=None)
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
