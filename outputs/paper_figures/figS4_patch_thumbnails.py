#!/usr/bin/env python3
"""Figure S4: Patch context thumbnails for all 119 campaign patches.

One page per site, showing building footprints within the 250m domain
for each patch. Labels: patch ID, SVF, λp, slope.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import *

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box


def main():
    apply_style()

    for site in SITE_ORDER:
        try:
            patches = load_campaign_patches(site)
            bld = load_buildings(site)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        n = len(patches)
        ncols = min(5, n)
        nrows = int(np.ceil(n / ncols))
        bld_sindex = bld.sindex

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * nrows / ncols * 0.95),
        )
        if n == 1:
            axes = np.array([axes])
        axes_flat = np.atleast_2d(axes).flat

        view_half = 280  # slightly larger than 250m domain

        for i, (_, row) in enumerate(patches.iterrows()):
            ax = axes_flat[i]
            cx = row.get("center_x", row.get("centroid_x"))
            cy = row.get("center_y", row.get("centroid_y"))

            vbox = box(cx - view_half, cy - view_half, cx + view_half, cy + view_half)
            cands = list(bld_sindex.intersection(vbox.bounds))
            if cands:
                nearby = bld.iloc[cands]
                nearby = nearby[nearby.intersects(vbox)]
                nearby.plot(
                    ax=ax, facecolor="#d0d0d0", edgecolor="#888",
                    linewidth=0.1, zorder=1,
                )

            # Analysis patch (100m-diameter circle)
            ax.add_patch(plt.Circle(
                (cx, cy), 50,
                linewidth=0.6, edgecolor=SITE_COLORS[site],
                facecolor="none", linestyle="--", zorder=5,
            ))
            # Domain circle (250m)
            ax.add_patch(plt.Circle(
                (cx, cy), 250,
                linewidth=0.4, edgecolor="#999",
                facecolor="none", linestyle=":", zorder=4,
            ))

            svf = row.get("svf", np.nan)
            lp = row.get("lambda_p", np.nan)
            slope = row.get("slope_deg", np.nan)
            pid = row.get("patch_id", f"P{i+1}")

            parts = [pid]
            if pd.notna(svf):
                parts.append(f"SVF={svf:.2f}")
            if pd.notna(lp):
                parts.append(f"$\\lambda_p$={lp:.2f}")
            if pd.notna(slope):
                parts.append(f"slope={slope:.0f}\u00b0")
            ax.set_title("  ".join(parts), fontsize=4, pad=1)

            ax.set_xlim(cx - view_half, cx + view_half)
            ax.set_ylim(cy - view_half, cy + view_half)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Hide unused subplots
        for j in range(i + 1, nrows * ncols):
            axes_flat[j].set_visible(False)

        fig.tight_layout(h_pad=0.3, w_pad=0.3)
        save_fig(fig, f"figS4_patch_thumbnails_{site}")


if __name__ == "__main__":
    main()
