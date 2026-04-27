#!/usr/bin/env python3
"""
Debug visualization: building preprocessing / filtering for Maré.

Shows what buildings are kept vs removed by the informal-area filters:
  1. Area filter (>200 m²)
  2. Cluster filter (isolated buildings)
  3. Height/base field validity (skipped during extrusion)

Usage:
    python scripts/debug_mare_preprocessing.py
    python scripts/debug_mare_preprocessing.py --area maré
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MAX_FILTER_AREA,
    BUILDING_CLUSTER_BUFFER,
    get_area_output_dir,
    is_informal_area,
)
from src.svf_v2.utils import filter_buildings
from src.svf_v2.paths import resolve_boundary, resolve_paths


def main():
    parser = argparse.ArgumentParser(description="Debug building preprocessing")
    parser.add_argument("--area", default="maré")
    parser.add_argument("--height-field", default="altura")
    parser.add_argument("--base-field", default="base")
    args = parser.parse_args()

    area = args.area
    dtm_path, footprints_path, roads_path = resolve_paths(area)
    boundary_path = resolve_boundary(area)

    import rasterio

    with rasterio.open(dtm_path) as src:
        dtm_crs = src.crs

    gdf = gpd.read_file(footprints_path)
    if gdf.crs != dtm_crs:
        gdf = gdf.to_crs(dtm_crs)

    roads_gdf = gpd.read_file(roads_path)
    if roads_gdf.crs != dtm_crs:
        roads_gdf = roads_gdf.to_crs(dtm_crs)

    boundary_gdf = None
    if boundary_path:
        boundary_gdf = gpd.read_file(boundary_path)
        if boundary_gdf.crs != dtm_crs:
            boundary_gdf = boundary_gdf.to_crs(dtm_crs)

    n_total = len(gdf)
    gdf["_footprint_area"] = gdf.geometry.area

    print(f"Area: {area} (informal={is_informal_area(area)})")
    print(f"Total buildings: {n_total:,}")
    print(f"MAX_FILTER_AREA: {MAX_FILTER_AREA} m²")
    print(f"BUILDING_CLUSTER_BUFFER: {BUILDING_CLUSTER_BUFFER} m")
    print()

    # ---- Run actual pipeline filter (SVF mode: skip_area_filter=True) ----
    print("=" * 50)
    print("SVF mode (skip_area_filter=True):")
    print("=" * 50)
    kept_final, isolated_svf, area_filtered_svf = filter_buildings(
        gdf, area=area, skip_area_filter=True
    )
    removed_by_cluster = isolated_svf

    # Also compute what the old morphometric filter would remove (for comparison)
    print()
    print("=" * 50)
    print("Morphometric mode (skip_area_filter=False) — for comparison:")
    print("=" * 50)
    kept_morph, isolated_morph, area_filtered_morph = filter_buildings(
        gdf, area=area, skip_area_filter=False
    )
    removed_by_area = area_filtered_morph

    # ---- Extrusion validity ----
    skipped_height = (
        kept_final[
            kept_final[args.height_field].isna() | (kept_final[args.height_field] <= 0)
        ]
        if args.height_field in kept_final.columns
        else gpd.GeoDataFrame()
    )

    print(f"\n--- Extrusion validity (SVF mode) ---")
    print(f"  Missing/zero height: {len(skipped_height):,}")
    extrudable = len(kept_final) - len(skipped_height)
    print(f"  Will be extruded: {extrudable:,}")
    print(
        f"\n  SVF pipeline: {n_total:,} → cluster filter → {len(kept_final):,} "
        f"→ extrusion → {extrudable:,}"
    )
    print(f"  (Morphometric pipeline would have: {n_total:,} → {len(kept_morph):,})")

    total_removed = n_total - extrudable
    print(f"  Total removed: {total_removed:,} ({total_removed / n_total * 100:.1f}%)")

    # ====================================================================
    # FIGURE 1: 4-panel filtering overview
    # ====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    # Panel 1: All buildings, colored by filter outcome
    ax = axes[0, 0]
    kept_final.plot(
        ax=ax, color="#2ecc71", edgecolor="#27ae60", linewidth=0.1, alpha=0.7
    )
    if len(removed_by_area) > 0:
        removed_by_area.plot(
            ax=ax, color="#e74c3c", edgecolor="#c0392b", linewidth=0.2, alpha=0.8
        )
    if len(removed_by_cluster) > 0:
        removed_by_cluster.plot(
            ax=ax, color="#f39c12", edgecolor="#d68910", linewidth=0.2, alpha=0.8
        )
    roads_gdf.plot(ax=ax, color="#95a5a6", linewidth=0.3, alpha=0.4)
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, linestyle="--")
    legend_items = [
        Patch(facecolor="#2ecc71", label=f"Kept ({len(kept_final):,})"),
        Patch(facecolor="#e74c3c", label=f"Area filter ({len(removed_by_area):,})"),
        Patch(
            facecolor="#f39c12", label=f"Cluster filter ({len(removed_by_cluster):,})"
        ),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=10)
    ax.set_title(
        f"Filter Overview\n{n_total:,} → {len(kept_final):,} "
        f"({total_removed:,} removed, {total_removed / n_total * 100:.1f}%)",
        fontsize=12,
    )
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    # Panel 2: Zoom on area-filtered buildings
    ax = axes[0, 1]
    kept_final.plot(
        ax=ax, color="#d5f5e3", edgecolor="#abebc6", linewidth=0.05, alpha=0.5
    )
    if len(removed_by_area) > 0:
        removed_by_area.plot(
            ax=ax,
            column="_footprint_area",
            cmap="Reds",
            edgecolor="#c0392b",
            linewidth=0.5,
            alpha=0.9,
            legend=True,
            legend_kwds={"shrink": 0.5, "label": "Area (m²)"},
        )
        # Annotate largest removed buildings
        top_removed = removed_by_area.nlargest(10, "_footprint_area")
        for _, row in top_removed.iterrows():
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            ax.annotate(
                f"{row['_footprint_area']:.0f}m²",
                xy=(cx, cy),
                fontsize=6,
                color="darkred",
                ha="center",
                va="center",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="red",
                ),
            )
    roads_gdf.plot(ax=ax, color="#95a5a6", linewidth=0.3, alpha=0.3)
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, linestyle="--")
    ax.set_title(
        f"Area-Filtered Buildings (>{MAX_FILTER_AREA} m²)\n"
        f"{len(removed_by_area):,} removed — are these important?",
        fontsize=12,
    )
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    # Panel 3: Cluster-filtered buildings
    ax = axes[1, 0]
    kept_final.plot(
        ax=ax, color="#d5f5e3", edgecolor="#abebc6", linewidth=0.05, alpha=0.5
    )
    if len(removed_by_cluster) > 0:
        removed_by_cluster.plot(
            ax=ax, color="#f39c12", edgecolor="#d68910", linewidth=0.5, alpha=0.9
        )
    roads_gdf.plot(ax=ax, color="#95a5a6", linewidth=0.3, alpha=0.3)
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, linestyle="--")
    ax.set_title(
        f"Cluster-Filtered Buildings (buffer={BUILDING_CLUSTER_BUFFER}m)\n"
        f"{len(removed_by_cluster):,} isolated buildings removed",
        fontsize=12,
    )
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    # Panel 4: Height distribution comparison (kept vs removed)
    ax = axes[1, 1]
    if args.height_field in gdf.columns:
        h_kept = kept_final[args.height_field].dropna().values
        h_kept = h_kept[np.isfinite(h_kept) & (h_kept > 0)]

        h_removed_area = (
            removed_by_area[args.height_field].dropna().values
            if len(removed_by_area) > 0
            else np.array([])
        )
        h_removed_area = (
            h_removed_area[np.isfinite(h_removed_area) & (h_removed_area > 0)]
            if len(h_removed_area) > 0
            else np.array([])
        )

        bins = np.arange(
            0,
            max(
                h_kept.max() if len(h_kept) else 0,
                h_removed_area.max() if len(h_removed_area) else 0,
            )
            + 2,
            1,
        )

        ax.hist(
            h_kept,
            bins=bins,
            color="#2ecc71",
            alpha=0.7,
            label=f"Kept (n={len(h_kept):,})",
        )
        if len(h_removed_area) > 0:
            ax.hist(
                h_removed_area,
                bins=bins,
                color="#e74c3c",
                alpha=0.7,
                label=f"Area-removed (n={len(h_removed_area):,})",
            )

        ax.set_xlabel(f"Height [{args.height_field}] (m)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=10)
        ax.set_title("Height Distribution: Kept vs Removed", fontsize=12)

        # Add text box with stats
        stats_text = (
            f"Kept: mean={h_kept.mean():.1f}m, max={h_kept.max():.1f}m\n"
            f"Removed: mean={h_removed_area.mean():.1f}m, max={h_removed_area.max():.1f}m"
            if len(h_removed_area) > 0
            else f"Kept: mean={h_kept.mean():.1f}m, max={h_kept.max():.1f}m"
        )
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.suptitle(
        f"{area} — Building Preprocessing Debug", fontsize=16, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = get_area_output_dir(area) / "svf_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "preprocessing_debug.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # ====================================================================
    # FIGURE 2: Footprint area vs height scatter (what's being cut)
    # ====================================================================
    if args.height_field in gdf.columns:
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

        h_all = gdf[args.height_field].values
        a_all = gdf["_footprint_area"].values
        valid = np.isfinite(h_all) & np.isfinite(a_all) & (h_all > 0) & (a_all > 0)

        ax2.scatter(
            a_all[valid], h_all[valid], s=1, alpha=0.15, c="#3498db", rasterized=True
        )
        ax2.axvline(
            MAX_FILTER_AREA,
            color="red",
            linewidth=2,
            linestyle="--",
            label=f"Area cutoff = {MAX_FILTER_AREA} m²",
        )

        # Highlight removed buildings
        removed_mask = valid & (a_all > MAX_FILTER_AREA)
        if removed_mask.sum() > 0:
            ax2.scatter(
                a_all[removed_mask],
                h_all[removed_mask],
                s=8,
                alpha=0.6,
                c="#e74c3c",
                label=f"Removed ({removed_mask.sum():,})",
                zorder=5,
            )

        ax2.set_xlabel("Footprint area (m²)", fontsize=12)
        ax2.set_ylabel(f"Height [{args.height_field}] (m)", fontsize=12)
        ax2.set_title(
            f"{area} — Footprint Area vs Height\n"
            f"Red line = MAX_FILTER_AREA = {MAX_FILTER_AREA} m²",
            fontsize=13,
        )
        ax2.legend(fontsize=10)
        ax2.set_xlim(0, min(a_all[valid].max() * 1.05, 5000))

        out_path2 = output_dir / "area_vs_height_debug.png"
        fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {out_path2}")


if __name__ == "__main__":
    main()
