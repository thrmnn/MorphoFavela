#!/usr/bin/env python3
"""Figure S2: Extended building context validation.

Three panels: (a) Vidigal buildings original vs context, (b) candidate pool
comparison across buffer distances, (c) before/after patch location maps.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import *

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.33))

    # ── Panel (a): original vs context buildings ──
    ax = axes[0]

    ext_path = PROJECT_ROOT / "data" / "vidigal" / "buildings_extended_300m.gpkg"
    if ext_path.exists():
        bld = gpd.read_file(ext_path)
        boundary = load_boundary("vidigal")
        boundary_geom = boundary.geometry.union_all()
        buffer_geom = boundary_geom.buffer(300)

        # Context buildings (outside boundary)
        ctx = bld[bld["_source"] == "context"] if "_source" in bld.columns else bld.iloc[0:0]
        site_bld = bld[bld["_source"] == "site"] if "_source" in bld.columns else bld

        ctx.plot(ax=ax, facecolor="#DDCC77", edgecolor="#999933", linewidth=0.1, alpha=0.7, zorder=2)
        site_bld.plot(ax=ax, facecolor="#CC6677", edgecolor="#882255", linewidth=0.1, alpha=0.7, zorder=3)

        boundary.boundary.plot(ax=ax, color="k", linewidth=0.6, linestyle="--", zorder=4)
        gpd.GeoSeries([buffer_geom], crs=bld.crs).boundary.plot(
            ax=ax, color="#44AA99", linewidth=0.5, linestyle=":", zorder=4,
        )

        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor="#CC6677", alpha=0.7, label=f"Site ({len(site_bld)})"),
                Patch(facecolor="#DDCC77", alpha=0.7, label=f"Context ({len(ctx)})"),
            ],
            loc="lower left", frameon=False, fontsize=5,
        )

    clean_map_axes(ax)
    ax.set_aspect("equal")
    add_scalebar(ax)

    # ── Panel (b): candidate pool by buffer distance ──
    ax = axes[1]

    # Data from the buffer sensitivity analysis (hardcoded — these are the validated results)
    buffers = [0, 200, 300, 400]
    eligible = [980, 1549, 1697, 1774]
    domain_excl_pct = [55.4, 33.3, 20.1, 5.9]

    colors_bar = ["#dddddd", "#88CCEE", "#44AA99", "#117733"]
    bars = ax.bar(range(len(buffers)), eligible, color=colors_bar, edgecolor="white", linewidth=0.3)

    for i, (buf, elig, pct) in enumerate(zip(buffers, eligible, domain_excl_pct)):
        ax.text(i, elig + 30, f"{elig:,d}", ha="center", va="bottom", fontsize=5, fontweight="bold")
        ax.text(i, elig / 2, f"{pct:.0f}%\nexcl.", ha="center", va="center", fontsize=4.5, color="white")

    ax.set_xticks(range(len(buffers)))
    ax.set_xticklabels([f"{b}m" if b > 0 else "None" for b in buffers])
    ax.set_xlabel("Buffer distance")
    ax.set_ylabel("Eligible cells")
    ax.set_ylim(0, max(eligible) * 1.15)

    # ── Panel (c): before/after patch maps ──
    ax = axes[2]

    # Load original pilot patches (from archive or current)
    pilot_orig_path = PROJECT_ROOT / "outputs" / "vidigal" / "sampling_cfd" / "pilot_sampling" / "pilot_patches.csv"
    if pilot_orig_path.exists():
        patches_now = pd.read_csv(pilot_orig_path)

        # Load buildings for context
        try:
            bld_orig = load_buildings("vidigal", extended=False)
            bld_orig.plot(ax=ax, facecolor="#d0d0d0", edgecolor="#999", linewidth=0.05, alpha=0.5)
        except Exception:
            pass

        # Show extended buildings outline
        if ext_path.exists():
            ctx = gpd.read_file(ext_path)
            ctx_only = ctx[ctx.get("_source", "site") == "context"] if "_source" in ctx.columns else ctx.iloc[0:0]
            if not ctx_only.empty:
                ctx_only.plot(ax=ax, facecolor="#DDCC77", edgecolor="none", alpha=0.3, zorder=1)

        # Current patches
        for _, row in patches_now.iterrows():
            cx = row.get("center_x", row.get("centroid_x"))
            cy = row.get("center_y", row.get("centroid_y"))
            ax.add_patch(plt.Rectangle(
                (cx - 50, cy - 50), 100, 100,
                linewidth=0.5, edgecolor="#CC6677", facecolor="#CC6677", alpha=0.3, zorder=5,
            ))

    clean_map_axes(ax)
    ax.set_aspect("equal")
    add_scalebar(ax)

    fig.tight_layout(w_pad=1.0)
    save_fig(fig, "figS2_context_extension")


if __name__ == "__main__":
    main()
