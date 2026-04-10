#!/usr/bin/env python3
"""Figure 1: Study sites overview.

Main map of Rio de Janeiro with 5 favela boundaries, plus inset panels
showing building footprints for each site at consistent rendering.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import *

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import rasterio


def main():
    apply_style()

    # Load all favela boundaries from the city-wide file
    favelas_path = PROJECT_ROOT / "data" / "RJ" / "Favelas_Limit_2019.shp"
    if not favelas_path.exists():
        print(f"ERROR: {favelas_path} not found")
        return
    all_favelas = gpd.read_file(favelas_path)

    # Load city-wide DTM for hillshade (overview)
    rj_dtm_path = PROJECT_ROOT / "data" / "RJ" / "DTM_RJ.tif"

    # Map site names to favela boundary names
    FAVELA_NAMES = {
        "vidigal": "VIDIGAL",
        "rocinha": "ROCINHA",
        "complexo_do_alemao": None,  # multiple polygons — use boundary file
        "riodaspedras": None,        # use boundary file
        "maré": None,                # use boundary file
    }

    # Load per-site boundaries and buildings
    boundaries = {}
    buildings = {}
    for site in SITE_ORDER:
        try:
            boundaries[site] = load_boundary(site)
        except FileNotFoundError:
            # Try matching from all_favelas
            for name_candidate in [site.upper(), site.replace("_", " ").upper()]:
                match = all_favelas[all_favelas["nome"].str.upper().str.contains(name_candidate, na=False)]
                if not match.empty:
                    boundaries[site] = match
                    break
        try:
            buildings[site] = load_buildings(site, extended=False)
        except FileNotFoundError:
            pass

    # ── Layout: main map left, 5 insets right ──
    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.52))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1], wspace=0.02)

    # Left: Rio overview
    gs_left = gs[0].subgridspec(1, 1)
    ax_main = fig.add_subplot(gs_left[0])

    # Hillshade for Rio
    if rj_dtm_path.exists():
        try:
            with rasterio.open(rj_dtm_path) as src:
                # Read a decimated version for the overview
                factor = 4
                dtm = src.read(1, out_shape=(src.height // factor, src.width // factor)).astype(float)
                nodata = src.nodata
                if nodata is not None:
                    dtm[dtm == nodata] = np.nan
                dtm[np.abs(dtm) > 1e6] = np.nan
                bounds = src.bounds
                shade = hillshade(dtm, src.res[0] * factor)
                shade = np.where(np.isnan(dtm), 0.95, shade)
                ax_main.imshow(
                    shade, cmap="gray", vmin=0, vmax=1,
                    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                    alpha=0.5, zorder=0,
                )
        except Exception as e:
            print(f"  Hillshade failed: {e}")

    # All favela boundaries as light gray
    all_favelas.plot(ax=ax_main, facecolor="none", edgecolor="#cccccc", linewidth=0.15, zorder=1)

    # Highlight campaign sites
    for site in SITE_ORDER:
        if site in boundaries:
            boundaries[site].plot(
                ax=ax_main, facecolor=SITE_COLORS[site], edgecolor=SITE_COLORS[site],
                alpha=0.5, linewidth=0.8, zorder=3,
            )
            # Label
            centroid = boundaries[site].geometry.union_all().centroid
            ax_main.annotate(
                SITE_LABELS[site],
                xy=(centroid.x, centroid.y),
                fontsize=5, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=5,
            )

    # Zoom to area of interest (south zone + north zone of Rio)
    # Get bounding box of all campaign sites
    all_bounds = []
    for site in SITE_ORDER:
        if site in boundaries:
            all_bounds.append(boundaries[site].total_bounds)
    if all_bounds:
        arr = np.array(all_bounds)
        pad = 5000  # 5km padding
        ax_main.set_xlim(arr[:, 0].min() - pad, arr[:, 2].max() + pad)
        ax_main.set_ylim(arr[:, 1].min() - pad, arr[:, 3].max() + pad)

    clean_map_axes(ax_main)
    ax_main.set_aspect("equal")
    add_scalebar(ax_main, length_m=5000)
    add_north_arrow(ax_main)

    # Right: 5 inset panels
    gs_right = gs[1].subgridspec(5, 1, hspace=0.15)

    for i, site in enumerate(SITE_ORDER):
        ax = fig.add_subplot(gs_right[i])

        if site in buildings:
            buildings[site].plot(
                ax=ax, facecolor=SITE_COLORS[site], edgecolor="k",
                linewidth=0.05, alpha=0.6,
            )

        if site in boundaries:
            boundaries[site].boundary.plot(
                ax=ax, color="k", linewidth=0.5, linestyle="--",
            )

        clean_map_axes(ax)
        ax.set_aspect("equal")

        typ = SITE_TYPES[site]
        n_bld = len(buildings[site]) if site in buildings else "?"
        ax.set_title(f"{SITE_LABELS[site]} ({typ}, {n_bld} bld.)", fontsize=5, pad=1)
        add_scalebar(ax)

    save_fig(fig, "fig01_study_sites")


if __name__ == "__main__":
    main()
