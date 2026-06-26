#!/usr/bin/env python3
"""BRISA slide deck — fig_patch_sampling_terrain.png (slide 10, REPLACES v2).

Same Vidigal CFD patch-sampling viz as v2, with extra terrain context:
  * Background: hillshade DEM (azimuth 315, altitude 45), desaturated grey
    so it sits behind without competing.
  * Overlay: building footprints as faint grey outlines.
  * Overlay: morphometric-cluster-coloured 10 m grid (same palette as v2).
  * Overlay: CFD patches as open black rings (100 m diameter), centre dots.
  * Baked-in 9-stratum legend.

Aspect ≈ 1.6. Same orientation/extent as v2.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_patch_sampling_terrain.png
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
logging.getLogger("pilot_sampling").setLevel(logging.ERROR)
from _assets import BRISA_ASSETS_DIR

from scripts.run_pilot_sampling import (
    CONFIG,
    PATCH_RADIUS_M,
    SITE_PRESETS,
    _hillshade,
    _resolve,
    _stratum_cmap,
    assign_strata,
)

OUT_DIR = BRISA_ASSETS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def site_config(site: str) -> dict:
    cfg = dict(CONFIG)
    cfg.update(SITE_PRESETS[site])
    return cfg


def stratum_label(sid: str) -> str:
    parts = sid.split("_")
    svf_map = {"SVF1": "SVF<0.15", "SVF2": "SVF 0.15–0.30", "SVF3": "SVF≥0.30"}
    slp_map = {"SLP1": "slope<15°", "SLP2": "slope≥15°"}
    lp_map = {"LP1": "λp<0.5", "LP2": "λp≥0.5"}
    return f"{svf_map.get(parts[0], parts[0])} · {slp_map.get(parts[1], parts[1])} · {lp_map.get(parts[2], parts[2])}"


def render(site: str = "vidigal", out_name: str = "fig_patch_sampling_terrain.png") -> Path:
    cfg = site_config(site)

    grid = gpd.read_file(_resolve(cfg["grid_metrics"]))
    grid = assign_strata(grid, cfg)

    patch_path = (
        PROJECT_ROOT / "outputs" / cfg["site_id"]
        / "sampling_cfd" / "campaign_sampling" / "campaign_patches.csv"
    )
    if not patch_path.exists():
        patch_path = (
            PROJECT_ROOT / "outputs" / cfg["site_id"]
            / "sampling_cfd" / "pilot_sampling" / "pilot_patches.csv"
        )
    patches = pd.read_csv(patch_path)

    # Buildings — optional; faint outlines.
    buildings = None
    try:
        bpath = _resolve(cfg["buildings"])
        if bpath.exists():
            buildings = gpd.read_file(bpath)
            if buildings.crs != grid.crs:
                buildings = buildings.to_crs(grid.crs)
    except Exception:
        buildings = None

    # Aspect ≈ 1.6 — match the brief.
    fig_w = 14.0
    fig_h = fig_w / 1.6  # 8.75
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # --- Hillshade background (strong, dominant b/w layer) ---
    dtm_path = _resolve(cfg["dtm"])
    if dtm_path.exists():
        with rasterio.open(dtm_path) as src:
            dtm = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                dtm[dtm == nodata] = np.nan
            bounds = src.bounds
            shade = _hillshade(dtm, src.res[0], azimuth=315, altitude=45)
            shade = np.where(np.isnan(dtm), 0.95, shade)
            # Full opacity, full contrast — the hillshade is the spatial
            # context for the slide. Grid + buildings overlay it.
            ax.imshow(
                shade,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                alpha=1.0,
                interpolation="bilinear",
                zorder=0,
            )

    # --- Building footprints (faint outlines) ---
    if buildings is not None and len(buildings) < 100_000:
        # Clip to grid extent for performance.
        gbb = grid.total_bounds
        pad = max((gbb[2] - gbb[0]), (gbb[3] - gbb[1])) * 0.1
        from shapely.geometry import box
        clip_box = gpd.GeoSeries([box(gbb[0] - pad, gbb[1] - pad,
                                       gbb[2] + pad, gbb[3] + pad)], crs=grid.crs)
        try:
            bclip = gpd.clip(buildings, clip_box.iloc[0])
        except Exception:
            bclip = buildings
        bclip.boundary.plot(ax=ax, color="#555555", linewidth=0.35, alpha=0.7, zorder=3)

    # --- Stratum-coloured grid (low alpha so hillshade reads through) ---
    cmap = _stratum_cmap(grid["stratum_id"].tolist())
    g = grid[grid["stratum_id"].notna()].copy()
    g["_color"] = g["stratum_id"].map(cmap)
    g.plot(ax=ax, color=g["_color"], alpha=0.55, edgecolor="none", zorder=2)

    # --- CFD patches ---
    for _, row in patches.iterrows():
        cx, cy = row["center_x"], row["center_y"]
        ax.add_patch(
            plt.Circle(
                (cx, cy), PATCH_RADIUS_M,
                linewidth=1.5, edgecolor="black", facecolor="none",
                zorder=5,
            )
        )
        ax.plot(cx, cy, marker="o", color="black", markersize=4.0, zorder=6)

    # Crop with extra breathing room so terrain reads outside the grid.
    gbb = g.total_bounds
    pad = max((gbb[2] - gbb[0]), (gbb[3] - gbb[1])) * 0.06
    ax.set_xlim(gbb[0] - pad, gbb[2] + pad)
    ax.set_ylim(gbb[1] - pad, gbb[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 200 m scale bar.
    x0 = gbb[0] - pad + (gbb[2] - gbb[0]) * 0.04
    y0 = gbb[1] - pad + (gbb[3] - gbb[1]) * 0.05
    ax.add_patch(
        plt.Rectangle(
            (x0, y0), 200.0, (gbb[3] - gbb[1]) * 0.008,
            facecolor="black", edgecolor="black", linewidth=0,
            zorder=20, clip_on=False,
        )
    )
    ax.text(
        x0 + 100, y0 + (gbb[3] - gbb[1]) * 0.022,
        "200 m", ha="center", va="bottom", fontsize=10, color="#111111",
    )

    # Stratum legend.
    sorted_strata = sorted(cmap.keys())
    stratum_handles = [
        Patch(facecolor=cmap[sid], edgecolor="none", alpha=0.85,
              label=stratum_label(sid))
        for sid in sorted_strata
    ]
    patch_handle = Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="white", markeredgecolor="black",
        markersize=8, markeredgewidth=1.0,
        label=f"CFD patch (n={len(patches)}, 100 m diameter)",
    )

    leg_strata = ax.legend(
        handles=stratum_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.01),
        bbox_transform=ax.transAxes,
        ncol=3,
        frameon=True,
        fontsize=9,
        title="Morphometric clusters in joint SVF × λp × slope feature space",
        title_fontsize=10,
        labelspacing=0.4,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    leg_strata.get_frame().set_linewidth(0.4)
    leg_strata.get_frame().set_edgecolor("#888888")
    leg_strata.get_frame().set_facecolor("white")
    leg_strata.get_frame().set_alpha(0.95)
    leg_strata._legend_box.align = "left"
    ax.add_artist(leg_strata)

    leg_p = ax.legend(
        handles=[patch_handle],
        loc="upper right",
        bbox_to_anchor=(1.0, -0.01),
        bbox_transform=ax.transAxes,
        frameon=True,
        fontsize=9.5,
    )
    leg_p.get_frame().set_linewidth(0.4)
    leg_p.get_frame().set_edgecolor("#888888")
    leg_p.get_frame().set_facecolor("white")
    leg_p.get_frame().set_alpha(0.95)

    # Context strip: indicate hillshade convention.
    ax.text(
        0.0, 1.005,
        "Vidigal — CFD patch sampling over hillshade DEM (sun azimuth 315°, altitude 45°), building footprints, morphometric grid.",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=10, color="#222222",
    )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.30)

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    out = render()
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
