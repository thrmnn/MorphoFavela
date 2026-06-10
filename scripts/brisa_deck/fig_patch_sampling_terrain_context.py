#!/usr/bin/env python3
"""BRISA slide 9 — Vidigal CFD patch sampling in hillside + built context.

Context-rich reframe of fig_patch_sampling_terrain.py. The favela's coloured
morphometric grid no longer floats on blank grey: it now sits on a city-wide
hillshade with the surrounding urban fabric drawn faintly behind it.

Layers (back -> front):
  0  city-wide hillshade DEM (data/RJ/DTM_RJ.tif, az 315 / alt 45),
     window-read over a ~300 m halo around Vidigal, desaturated grey.
  1  surrounding building footprints (data/RJ/buildings_RJ_2019.shp),
     semi-transparent faint fills+outlines (alpha ~0.3).
  2  morphometric-cluster coloured 10 m grid for Vidigal (upstream palette).
  3  Vidigal favela boundary (data/RJ/Favelas_Limit_2019.shp), thin line.
  4  CFD patch rings at PATCH_RADIUS_M, centres from the campaign CSV.

Read-only imports from the upstream pipeline; this module only adds context
layers and writes the brisa asset (overwriting the slide-referenced PNG).

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
from matplotlib.patches import FancyArrow, Patch
from rasterio.windows import from_bounds
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
logging.getLogger("pilot_sampling").setLevel(logging.ERROR)

# Read-only upstream imports (clustering / colour / hillshade / config).
from scripts.run_pilot_sampling import (  # noqa: E402
    CONFIG,
    PATCH_RADIUS_M,
    SITE_PRESETS,
    _hillshade,
    _resolve,
    _stratum_cmap,
    assign_strata,
)

OUT_DIR = Path("/home/theo/brisa_paper/artifacts/slides/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# City-wide context layers (wider than the favela).
CITY_DTM = PROJECT_ROOT / "data" / "RJ" / "DTM_RJ.tif"
CITY_BUILDINGS = PROJECT_ROOT / "data" / "RJ" / "buildings_RJ_2019.shp"
FAVELA_LIMITS = PROJECT_ROOT / "data" / "RJ" / "Favelas_Limit_2019.shp"

HALO_M = 300.0  # surrounding-context band around the favela grid


def site_config(site: str) -> dict:
    cfg = dict(CONFIG)
    cfg.update(SITE_PRESETS[site])
    return cfg


def stratum_label(sid: str) -> str:
    parts = sid.split("_")
    svf_map = {"SVF1": "SVF<0.15", "SVF2": "SVF 0.15–0.30", "SVF3": "SVF≥0.30"}
    slp_map = {"SLP1": "slope<15°", "SLP2": "slope≥15°"}
    lp_map = {"LP1": "λp<0.5", "LP2": "λp≥0.5"}
    return (
        f"{svf_map.get(parts[0], parts[0])} · "
        f"{slp_map.get(parts[1], parts[1])} · "
        f"{lp_map.get(parts[2], parts[2])}"
    )


def _windowed_hillshade(dtm_path: Path, bounds: tuple[float, float, float, float]):
    """Window-read the city DEM over *bounds* and return (shade, extent).

    The city DEM is ~425 MB / 7424×14319; never load it whole.
    """
    minx, miny, maxx, maxy = bounds
    with rasterio.open(dtm_path) as src:
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        win = win.round_offsets().round_lengths()
        dtm = src.read(1, window=win).astype(float)
        nodata = src.nodata
        if nodata is not None:
            dtm[dtm == nodata] = np.nan
        dtm[dtm > 1e30] = np.nan  # guard the float32 sentinel
        wt = src.window_transform(win)
        res = src.res[0]
        h, w = dtm.shape
        ext = [wt.c, wt.c + w * wt.a, wt.f + h * wt.e, wt.f]  # L,R,B,T
    shade = _hillshade(dtm, res, azimuth=315, altitude=45)
    shade = np.where(np.isnan(dtm), 0.92, shade)
    return shade, ext


def render(site: str = "vidigal", out_name: str = "fig_patch_sampling_terrain.png") -> Path:
    cfg = site_config(site)

    grid = gpd.read_file(_resolve(cfg["grid_metrics"]))
    grid = assign_strata(grid, cfg)
    g = grid[grid["stratum_id"].notna()].copy()

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
    if "eligible" in patches.columns:
        patches = patches[patches["eligible"] == True]  # noqa: E712

    # Wider extent: favela grid bounds + ~300 m halo, then snap to aspect ~1.65.
    gbb = g.total_bounds
    minx, miny, maxx, maxy = (
        gbb[0] - HALO_M, gbb[1] - HALO_M, gbb[2] + HALO_M, gbb[3] + HALO_M,
    )
    cx0, cy0 = (minx + maxx) / 2, (miny + maxy) / 2
    w0, h0 = maxx - minx, maxy - miny
    target_aspect = 1.65
    if w0 / h0 < target_aspect:
        w0 = h0 * target_aspect
    else:
        h0 = w0 / target_aspect
    minx, maxx = cx0 - w0 / 2, cx0 + w0 / 2
    miny, maxy = cy0 - h0 / 2, cy0 + h0 / 2
    view = (minx, miny, maxx, maxy)
    view_box = box(*view)

    # --- City-wide context layers ---
    shade, ext = _windowed_hillshade(CITY_DTM, view)

    buildings = gpd.read_file(CITY_BUILDINGS, bbox=view_box)
    if buildings.crs != g.crs:
        buildings = buildings.to_crs(g.crs)

    favela = None
    try:
        fav = gpd.read_file(FAVELA_LIMITS, bbox=view_box)
        if fav.crs != g.crs:
            fav = fav.to_crs(g.crs)
        m = fav[fav.get("nome", pd.Series(dtype=str)).astype(str)
                .str.contains("Vidig", case=False, na=False)]
        favela = m if len(m) else fav
    except Exception:
        favela = None

    fig_w = 14.0
    fig_h = fig_w / target_aspect
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # 0 — hillshade (PALE desaturated backdrop). Compress the dynamic range so
    # the terrain reads as light grey relief and never swallows the faint
    # building fabric drawn on top of it.
    shade_pale = 0.55 + 0.45 * shade  # remap [0,1] -> [0.55,1.0]
    ax.imshow(
        shade_pale, cmap="gray", vmin=0.0, vmax=1.0,
        extent=ext, alpha=1.0, interpolation="bilinear", zorder=0,
    )

    # 1 — surrounding building fabric: faint but VISIBLE. Light fill + a
    # darker thin edge so the urban grain reads against the pale hillshade.
    if len(buildings):
        buildings.plot(
            ax=ax, facecolor="#9a9a9a", edgecolor="#505050",
            linewidth=0.25, alpha=0.40, zorder=1,
        )

    # 2 — morphometric-cluster coloured grid for Vidigal
    cmap = _stratum_cmap(g["stratum_id"].tolist())
    g["_color"] = g["stratum_id"].map(cmap)
    g.plot(ax=ax, color=g["_color"], alpha=0.82, edgecolor="none", zorder=2)

    # 3 — Vidigal favela boundary (thin legible line)
    if favela is not None and len(favela):
        favela.boundary.plot(
            ax=ax, color="#111111", linewidth=1.4, alpha=0.9, zorder=4,
        )

    # 4 — CFD patch rings + centres
    for _, row in patches.iterrows():
        cxp, cyp = row["center_x"], row["center_y"]
        ax.add_patch(
            plt.Circle(
                (cxp, cyp), PATCH_RADIUS_M,
                linewidth=1.6, edgecolor="black", facecolor="none", zorder=6,
            )
        )
        ax.plot(cxp, cyp, marker="o", color="black", markersize=3.8, zorder=7)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    vw, vh = maxx - minx, maxy - miny

    # 200 m scale bar (lower-left).
    sx = minx + vw * 0.035
    sy = miny + vh * 0.05
    ax.add_patch(
        plt.Rectangle(
            (sx, sy), 200.0, vh * 0.009,
            facecolor="black", edgecolor="white", linewidth=0.4, zorder=20,
        )
    )
    ax.text(
        sx + 100, sy + vh * 0.022, "200 m",
        ha="center", va="bottom", fontsize=11, color="#111111", zorder=20,
    )

    # North arrow (upper-right).
    nx = maxx - vw * 0.045
    ny0 = maxy - vh * 0.16
    ax.add_patch(
        FancyArrow(
            nx, ny0, 0, vh * 0.085,
            width=vw * 0.002, head_width=vw * 0.012, head_length=vh * 0.030,
            length_includes_head=True, facecolor="#111111",
            edgecolor="white", linewidth=0.4, zorder=21,
        )
    )
    ax.text(
        nx, ny0 + vh * 0.095, "N",
        ha="center", va="bottom", fontsize=13, fontweight="bold",
        color="#111111", zorder=21,
    )

    # Stratum legend.
    sorted_strata = sorted(cmap.keys())
    stratum_handles = [
        Patch(facecolor=cmap[sid], edgecolor="none", alpha=0.9,
              label=stratum_label(sid))
        for sid in sorted_strata
    ]
    patch_handle = Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="white", markeredgecolor="black",
        markersize=8, markeredgewidth=1.2,
        label=f"CFD patch (n={len(patches)}, {int(2 * PATCH_RADIUS_M)} m diameter)",
    )
    fav_handle = Line2D(
        [0], [0], color="#111111", linewidth=1.4, label="Vidigal favela boundary",
    )
    bld_handle = Patch(
        facecolor="#4d4d4d", edgecolor="#3a3a3a", alpha=0.30,
        label="Surrounding buildings",
    )

    leg_strata = ax.legend(
        handles=stratum_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.02),
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

    # Context legend on its own row BELOW the stratum legend so the two never
    # collide horizontally (the 3-col stratum legend spans the full width).
    leg_ctx = ax.legend(
        handles=[patch_handle, fav_handle, bld_handle],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.27),
        bbox_transform=ax.transAxes,
        ncol=3,
        frameon=True,
        fontsize=9.5,
        columnspacing=2.0,
        handletextpad=0.6,
    )
    leg_ctx.get_frame().set_linewidth(0.4)
    leg_ctx.get_frame().set_edgecolor("#888888")
    leg_ctx.get_frame().set_facecolor("white")
    leg_ctx.get_frame().set_alpha(0.95)

    # Title strip (per brief).
    ax.text(
        0.0, 1.045,
        "Vidigal — CFD patch sampling in hillside + built context",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=15, fontweight="bold", color="#111111",
    )
    ax.text(
        0.0, 1.012,
        "hillshade DEM (sun azimuth 315°, altitude 45°) · surrounding building fabric · 10 m morphometric grid · CFD analysis patches",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=9.5, color="#444444",
    )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.34)

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
