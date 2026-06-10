#!/usr/bin/env python3
"""BRISA slide deck — fig_svf_streets_cross_site_color.png (slide 8 replacement).

Cross-site street-level SVF, COLOUR version. Same 2-row layout as the B/W
fig_svf_streets_cross_site.py, but each panel adds:
  * Hillshade DEM background (azimuth 315°, altitude 45°), low-contrast grey.
  * Building footprints as faint grey outlines (alpha ~0.4).
  * Street segments coloured by SVF on a cividis colormap (lighter = higher),
    rendered on top of the hillshade + footprints.
  * Shared vertical cividis colorbar on the right.
  * 200 m scalebar on the leftmost panel of each row.
  * No in-figure title.

Aspect ≈ 1.4. 200 DPI.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_svf_streets_cross_site_color.png
"""
from __future__ import annotations

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
from fig_style import SITE_LABELS, apply_style  # noqa: E402
from scripts.run_pilot_sampling import _hillshade, _resolve  # noqa: E402

OUT_DIR = Path("/home/theo/brisa_paper/artifacts/slides/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
ROW_TOP = ["vidigal", "rocinha", "complexo_do_alemao"]
ROW_BOT = ["riodaspedras", "maré"]

SVF_CMAP = "cividis"  # lighter = higher SVF, perceptually uniform, colorblind safe
SVF_VMIN, SVF_VMAX = 0.0, 1.0
SVF_COL = "svf_mean"

# Per-site asset paths — segments + DTM + building footprints.
SITE_PATHS: dict[str, dict[str, Path]] = {
    "vidigal": {
        "dtm": Path("data/vidigal/raw/DTM_Vidigal.tif"),
        "buildings": Path("data/vidigal/raw/Vidigal_buildings.shp"),
    },
    "rocinha": {
        "dtm": Path("data/rocinha/raw/rocinha_dtm.tif"),
        "buildings": Path("data/rocinha/raw/rocinha_buildings.shp"),
    },
    "complexo_do_alemao": {
        "dtm": Path("data/complexo_do_alemao/raw/complexo_do_alemao_dtm.tif"),
        "buildings": Path("data/complexo_do_alemao/raw/complexo_do_alemao_buildings.shp"),
    },
    "riodaspedras": {
        "dtm": Path("data/riodaspedras/raw/riodaspedras_dtm.tif"),
        "buildings": Path("data/riodaspedras/raw/riodaspedras_buildings.shp"),
    },
    "maré": {
        "dtm": Path("data/maré/raw/mare_dtm.tif"),
        "buildings": Path("data/maré/raw/buildings_mare.shp"),
    },
}


def load_segments(site: str) -> gpd.GeoDataFrame:
    p = PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_segments.gpkg"
    g = gpd.read_file(p)
    return g[g[SVF_COL].notna()].copy()


def load_dtm(site: str, bbox):
    cfg = SITE_PATHS[site]
    p = _resolve(str(cfg["dtm"]))
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


def load_buildings(site: str, bbox, crs):
    cfg = SITE_PATHS[site]
    p = _resolve(str(cfg["buildings"]))
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
    # Clip to padded bbox to avoid loading neighbourhood-scale footprints.
    pad = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.1
    clip_box = box(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    try:
        b = gpd.clip(b, clip_box)
    except Exception:
        pass
    return b


def streets_panel(ax, site: str, gdf: gpd.GeoDataFrame, title: str, *, with_scalebar: bool = False) -> None:
    valid = gdf.copy()
    valid[SVF_COL] = valid[SVF_COL].clip(SVF_VMIN, SVF_VMAX)
    bbox = valid.total_bounds
    pad = (bbox[2] - bbox[0]) * 0.03
    crs = valid.crs

    # --- Hillshade background (low-contrast greyscale) ---
    h = load_dtm(site, bbox)
    if h is not None:
        shade, dbounds = h
        ax.imshow(
            shade,
            cmap="gray",
            vmin=0.15, vmax=1.05,  # compress dynamic range -> low contrast
            extent=[dbounds.left, dbounds.right, dbounds.bottom, dbounds.top],
            alpha=0.55,
            interpolation="bilinear",
            zorder=0,
        )

    # --- Building footprint outlines (faint grey) ---
    b = load_buildings(site, bbox, crs)
    if b is not None and len(b) > 0:
        try:
            b.boundary.plot(ax=ax, color="#444444", linewidth=0.25, alpha=0.4, zorder=1)
        except Exception:
            pass

    # --- Street segments coloured by SVF (cividis) ---
    valid.plot(
        ax=ax,
        column=SVF_COL,
        cmap=SVF_CMAP,
        vmin=SVF_VMIN, vmax=SVF_VMAX,
        linewidth=1.7,
        zorder=3,
    )

    ax.set_xlim(bbox[0] - pad, bbox[2] + pad)
    ax.set_ylim(bbox[1] - pad, bbox[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Site label — small chip, top-left.
    ax.text(
        0.02, 0.97, title,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13, color="#111111",
        bbox=dict(facecolor="white", edgecolor="#888888",
                  linewidth=0.4, pad=2.8, alpha=0.85),
        zorder=10,
    )

    if with_scalebar:
        x0 = bbox[0] + pad
        y0 = bbox[1] + pad
        bar_h = (bbox[3] - bbox[1]) * 0.012
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), 200.0, bar_h,
                facecolor="#111111", edgecolor="#111111", linewidth=0.0,
                zorder=20, clip_on=False,
            )
        )
        ax.text(
            x0 + 100, y0 + (bbox[3] - bbox[1]) * 0.035,
            "200 m", ha="center", va="bottom",
            fontsize=9.5, color="#111111",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.75),
            zorder=20,
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
        streets_panel(ax, s, segs[s], SITE_LABELS[s], with_scalebar=(i == 0))
        top_axes.append(ax)
        x += top_widths[i] + inter_pad

    bot_axes = []
    x = bot_left
    for i, s in enumerate(ROW_BOT):
        ax = fig.add_axes([x, bot_row_y, bot_widths[i], row_h])
        streets_panel(ax, s, segs[s], SITE_LABELS[s], with_scalebar=(i == 0))
        bot_axes.append(ax)
        x += bot_widths[i] + inter_pad

    fig.canvas.draw()
    add_colorbar(fig, top_axes[-1], bot_axes[-1])

    out = OUT_DIR / "fig_svf_streets_cross_site_color.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches=None)
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
