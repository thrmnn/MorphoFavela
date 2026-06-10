#!/usr/bin/env python3
"""BRISA slide deck — fig_vidigal_lambdaf_streets_color.png (slide 7).

Color version of the Vidigal-only λf street-level map. Same composition as
the SVF reference (svf_streets_vidigal.png): hillshade DEM background, faint
building footprints overlaid, 10 m grid cells coloured by λf_mean.

Color philosophy: cividis with dark = LOW λf (open street, well-ventilated),
bright = HIGH λf (canyon-like, choked ventilation). Semantically opposite of
SVF but using the same perceptually-uniform, colorblind-safe ramp.

Colorbar range clamped to the 1st–99th percentile of λf_mean to avoid the
long upper tail dominating the ramp. The MorphoFavela `lambda_f_mean` field
runs 0–~13 (not the canonical 0–1.5); raw values are plotted as-is.

Sized for slide projection — large axis tick fonts and a physically larger
colorbar than the B&W companion.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_vidigal_lambdaf_streets_color.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
sys.path.insert(0, str(PAPER_FIG_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
from fig_style import apply_style  # noqa: E402
from scripts.run_pilot_sampling import _hillshade, _resolve  # noqa: E402
from src.viz import presentation_style as ps  # noqa: E402

PAPER_OUT_DIR = Path("/home/theo/brisa_paper/artifacts/slides/assets")
PRESENTATION_OUT_DIR = PROJECT_ROOT / "outputs" / "vidigal" / "presentation_figures"
OUT_FILENAME = "fig_vidigal_lambdaf_streets_color.png"

LAMBDA_COL = "lambda_f_mean"
LAMBDA_CMAP = "cividis"

DTM_PATH = Path("data/vidigal/raw/DTM_Vidigal.tif")
BUILDINGS_PATH = Path("data/vidigal/raw/Vidigal_buildings.shp")


def load_grid() -> gpd.GeoDataFrame:
    grid_path = PROJECT_ROOT / "outputs" / "vidigal" / "morphometrics" / "grid" / "grid_metrics.gpkg"
    grid = gpd.read_file(grid_path)
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    return grid[built & grid[LAMBDA_COL].notna()].copy()


def load_hillshade(bbox, crs):
    p = _resolve(str(DTM_PATH))
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        dtm = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            dtm[dtm == nodata] = np.nan
        bounds = src.bounds
        res = src.res[0]
    shade = _hillshade(dtm, res, azimuth=315, altitude=45)
    shade = np.where(np.isnan(dtm), 0.92, shade)
    return shade, bounds


def load_buildings(bbox, crs):
    p = _resolve(str(BUILDINGS_PATH))
    if not p.exists():
        return None
    try:
        b = gpd.read_file(p)
    except Exception:
        return None
    if b.crs != crs:
        try:
            b = b.to_crs(crs)
        except Exception:
            return None
    pad = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.1
    clip_box = box(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    try:
        b = gpd.clip(b, clip_box)
    except Exception:
        pass
    return b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=("paper", "presentation"), default="paper")
    args = parser.parse_args()

    apply_style()
    ps.apply(args.preset)
    grid = load_grid()

    raw = grid[LAMBDA_COL].to_numpy()
    vmin = float(np.percentile(raw, 1))
    vmax = float(np.percentile(raw, 99))
    grid[LAMBDA_COL] = grid[LAMBDA_COL].clip(vmin, vmax)

    bbox = grid.total_bounds
    crs = grid.crs

    # Aspect ratio ~2.19 to match svf_streets_vidigal.png (3316 × 1516).
    fig_w = 16.0
    fig_h = fig_w / 2.19  # ~7.31 in
    fig = plt.figure(figsize=(fig_w, fig_h))

    # No UTM tick axes — give the map almost the full canvas width.
    left, right = 0.010, 0.86
    bottom, top = 0.02, 0.98
    ax = fig.add_axes([left, bottom, right - left, top - bottom])

    # Hillshade DEM background (low-contrast greyscale).
    h = load_hillshade(bbox, crs)
    if h is not None:
        shade, dbounds = h
        ax.imshow(
            shade,
            cmap="gray",
            vmin=0.15, vmax=1.05,
            extent=[dbounds.left, dbounds.right, dbounds.bottom, dbounds.top],
            alpha=0.55,
            interpolation="bilinear",
            zorder=0,
        )

    # Building footprint outlines (faint).
    b = load_buildings(bbox, crs)
    if b is not None and len(b) > 0:
        try:
            b.boundary.plot(ax=ax, color="#444444", linewidth=0.25, alpha=0.4, zorder=1)
        except Exception:
            pass

    # λf grid cells.
    grid.plot(
        ax=ax,
        column=LAMBDA_COL,
        cmap=LAMBDA_CMAP,
        vmin=vmin, vmax=vmax,
        edgecolor="none", linewidth=0.0, alpha=0.85,
        zorder=2,
    )

    pad_x = (bbox[2] - bbox[0]) * 0.02
    pad_y = (bbox[3] - bbox[1]) * 0.04
    ax.set_xlim(bbox[0] - pad_x, bbox[2] + pad_x)
    ax.set_ylim(bbox[1] - pad_y, bbox[3] + pad_y)
    ax.set_aspect("equal")

    # Strip all axis decoration — no UTM ticks, no spines, no labels.
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 200 m scale bar, bottom-left — bigger for projection.
    x0 = bbox[0] + (bbox[2] - bbox[0]) * 0.03
    y0 = bbox[1] + (bbox[3] - bbox[1]) * 0.07
    bar_h = (bbox[3] - bbox[1]) * 0.022
    ax.add_patch(plt.Rectangle((x0, y0), 200.0, bar_h,
                               facecolor="black", edgecolor="black",
                               linewidth=0.0, zorder=20))
    ax.text(x0 + 100, y0 + bar_h + (bbox[3] - bbox[1]) * 0.022,
            "200 m", ha="center", va="bottom",
            fontsize=18, fontweight="bold", color="#111111", zorder=20)

    # North arrow — top-right, larger.
    nx = bbox[2] - (bbox[2] - bbox[0]) * 0.05
    ny = bbox[3] - (bbox[3] - bbox[1]) * 0.06
    arrow_h = (bbox[3] - bbox[1]) * 0.16
    ax.annotate("", xy=(nx, ny), xytext=(nx, ny - arrow_h),
                arrowprops=dict(arrowstyle="-|>", color="#111111",
                                lw=2.6, mutation_scale=24),
                zorder=20)
    ax.text(nx, ny + (bbox[3] - bbox[1]) * 0.015, "N",
            ha="center", va="bottom",
            fontsize=22, fontweight="bold", color="#111111", zorder=20)

    # Vertical colorbar on the right — ~30% wider, larger label/tick fonts.
    cb_x = right + 0.020
    cb_w = 0.040  # was 0.030 — +33%
    cax = fig.add_axes([cb_x, bottom + 0.05, cb_w, (top - bottom) - 0.10])
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=LAMBDA_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.6)
    if args.preset == "presentation":
        cb_label_text = ps.short_label("lambda_f_mean")
        cb.set_label(cb_label_text, fontsize=16, color="#111111", labelpad=10)
        cb.ax.tick_params(labelsize=13, color="#333333", length=4)
    else:
        cb.set_label(r"$\lambda_f$  (8-dir mean)",
                     fontsize=14, color="#111111", labelpad=10)
        cb.ax.tick_params(labelsize=12, color="#333333", length=4)

    out_dir = PRESENTATION_OUT_DIR if args.preset == "presentation" else PAPER_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / OUT_FILENAME
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches=None)
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")
    print(f"  λf range plotted: [{vmin:.3f}, {vmax:.3f}] (1st–99th percentile)")


if __name__ == "__main__":
    main()
