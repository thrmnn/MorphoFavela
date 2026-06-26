#!/usr/bin/env python3
"""Figure 5: Morphometric maps — exemplar sites.

Two rows: Vidigal (hillside) and Maré (flatland).
Three columns: SVF, λp, slope. Building footprints as context.

SVF data source: The `svf` column in grid_metrics.gpkg is the mean of
passageway-level SVF sample points (computed at 1.5 m height via 145-ray
Tregenza hemisphere) aggregated to each 10 m grid cell. The `svf_count`
column records how many sample points fell within each cell (mean ~13).
Cells with no SVF sample points have NaN. The few cells with SVF = 0.0
(12 in Vidigal) have low svf_count (1-4) and represent legitimate deep
canyon conditions at the settlement boundary — not grid-centroid artifacts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from fig_style import *


def main():
    apply_style()

    exemplars = [
        ("vidigal", "Vidigal (hillside)"),
        ("maré", "Maré (flatland)"),
    ]

    indicators = [
        ("svf", r"$SVF$", CMAP_SVF, (0, 0.8)),
        ("lambda_p", r"$\lambda_p$", CMAP_LAMBDA_P, (0, 1.0)),
        ("slope_deg", "Slope (\u00b0)", CMAP_SLOPE, (0, 45)),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.55),
    )

    for row, (site, site_label) in enumerate(exemplars):
        grid = load_grid(site)
        try:
            bld = load_buildings(site)
        except FileNotFoundError:
            bld = None

        # DTM hillshade
        dtm_path = load_dtm_path(site)
        shade_data = None
        shade_extent = None
        if dtm_path.exists():
            try:
                with rasterio.open(dtm_path) as src:
                    dtm = src.read(1).astype(float)
                    nodata = src.nodata
                    if nodata is not None:
                        dtm[dtm == nodata] = np.nan
                    # Also mask extreme sentinel values
                    dtm[np.abs(dtm) > 1e6] = np.nan
                    bounds = src.bounds
                    shade_data = hillshade(dtm, src.res[0])
                    shade_data = np.where(np.isnan(dtm), 1.0, shade_data)
                    shade_extent = [
                        bounds.left,
                        bounds.right,
                        bounds.bottom,
                        bounds.top,
                    ]
            except Exception:
                pass

        for col_idx, (col, label, cmap, vlim) in enumerate(indicators):
            ax = axes[row, col_idx]

            # Hillshade background
            if shade_data is not None:
                ax.imshow(
                    shade_data,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    extent=shade_extent,
                    alpha=0.3,
                    zorder=0,
                )

            # Building outlines
            if bld is not None:
                bld.plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor="#cccccc",
                    linewidth=0.1,
                    alpha=0.4,
                    zorder=1,
                )

            # Grid cells
            grid.plot(
                ax=ax,
                column=col,
                cmap=cmap,
                vmin=vlim[0],
                vmax=vlim[1],
                alpha=0.75,
                edgecolor="none",
                zorder=2,
                legend=False,
                missing_kwds={"color": "none"},
            )

            # Colorbar
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=mcolors.Normalize(vmin=vlim[0], vmax=vlim[1])
            )
            cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, aspect=20)
            cbar.ax.tick_params(labelsize=5, width=0.3, length=2)
            cbar.outline.set_linewidth(0.3)

            # Threshold contours on colorbar
            if col == "svf":
                for t in SVF_THRESHOLDS:
                    cbar.ax.axhline(t, color="white", linewidth=0.5, linestyle="-")
            elif col == "slope_deg":
                cbar.ax.axhline(SLOPE_THRESHOLD, color="white", linewidth=0.5)
            elif col == "lambda_p":
                cbar.ax.axhline(LAMBDA_P_THRESHOLD, color="white", linewidth=0.5)

            clean_map_axes(ax)
            ax.set_aspect("equal")

            # Clip to grid extent with small padding
            bounds = grid.total_bounds
            pad = 30
            ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
            ax.set_ylim(bounds[1] - pad, bounds[3] + pad)

            if row == 0:
                ax.set_title(label, fontsize=7, pad=4)
            if col_idx == 0:
                ax.set_ylabel(site_label, fontsize=7, rotation=90, labelpad=8)

            add_scalebar(ax)

    fig.tight_layout(h_pad=0.5, w_pad=0.5)
    add_provenance(fig)
    save_fig(fig, "fig05_morphometric_maps")


if __name__ == "__main__":
    main()
