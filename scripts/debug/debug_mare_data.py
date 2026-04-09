#!/usr/bin/env python3
"""
Debug visualization: Maré data readiness check.

Generates a multi-panel diagnostic figure covering:
  1. DTM elevation + hillshade
  2. Spatial alignment (footprints + roads + boundary on DTM)
  3. DTM coverage vs data extents
  4. Building height distribution & schema check
  5. Road network summary
  6. Boundary vs footprint overlap

Usage:
    python scripts/debug_mare_data.py
    python scripts/debug_mare_data.py --area maré
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LightSource

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_output_dir
from src.svf_v2.paths import resolve_boundary, resolve_paths


def main():
    parser = argparse.ArgumentParser(description="Maré data readiness debug")
    parser.add_argument("--area", default="maré", help="Area name (default: maré)")
    args = parser.parse_args()

    area = args.area
    print(f"{'='*60}")
    print(f"  Data Readiness Check: {area}")
    print(f"{'='*60}\n")

    # --- Resolve paths ---
    dtm_path, footprints_path, roads_path = resolve_paths(area)
    boundary_path = resolve_boundary(area)

    print(f"DTM:         {dtm_path}")
    print(f"Footprints:  {footprints_path}")
    print(f"Roads:       {roads_path}")
    print(f"Boundary:    {boundary_path or 'NONE'}")
    print()

    # --- Load data ---
    import rasterio

    with rasterio.open(dtm_path) as src:
        dtm = src.read(1)
        dtm_bounds = src.bounds
        dtm_crs = src.crs
        dtm_transform = src.transform
        dtm_res = src.res
        dtm_nodata = src.nodata

    # Sanitise DTM
    dtm_valid_mask = np.isfinite(dtm) & (dtm < 10000)
    if dtm_nodata is not None:
        dtm_valid_mask &= dtm != dtm_nodata
    dtm_clean = np.where(dtm_valid_mask, dtm, np.nan)

    fp_gdf = gpd.read_file(footprints_path)
    roads_gdf = gpd.read_file(roads_path)
    boundary_gdf = gpd.read_file(boundary_path) if boundary_path else None

    # Reproject everything to DTM CRS
    if fp_gdf.crs != dtm_crs:
        print(f"  Reprojecting footprints {fp_gdf.crs} -> {dtm_crs}")
        fp_gdf = fp_gdf.to_crs(dtm_crs)
    if roads_gdf.crs != dtm_crs:
        print(f"  Reprojecting roads {roads_gdf.crs} -> {dtm_crs}")
        roads_gdf = roads_gdf.to_crs(dtm_crs)
    if boundary_gdf is not None and boundary_gdf.crs != dtm_crs:
        print(f"  Reprojecting boundary {boundary_gdf.crs} -> {dtm_crs}")
        boundary_gdf = boundary_gdf.to_crs(dtm_crs)

    # --- Print diagnostics ---
    print(f"\n--- CRS ---")
    print(f"  DTM CRS:         {dtm_crs}")
    print(f"  Footprints CRS:  {fp_gdf.crs}")
    print(f"  Roads CRS:       {roads_gdf.crs}")
    if boundary_gdf is not None:
        print(f"  Boundary CRS:    {boundary_gdf.crs}")

    print(f"\n--- DTM ---")
    print(f"  Shape:       {dtm.shape}")
    print(f"  Resolution:  {dtm_res[0]:.2f} x {dtm_res[1]:.2f} m")
    print(f"  Bounds:      ({dtm_bounds.left:.0f}, {dtm_bounds.bottom:.0f}) - "
          f"({dtm_bounds.right:.0f}, {dtm_bounds.top:.0f})")
    print(f"  Extent:      {dtm_bounds.right - dtm_bounds.left:.0f} x "
          f"{dtm_bounds.top - dtm_bounds.bottom:.0f} m")
    valid_vals = dtm_clean[np.isfinite(dtm_clean)]
    pct_valid = len(valid_vals) / dtm_clean.size * 100
    print(f"  Valid cells: {len(valid_vals):,} / {dtm_clean.size:,} ({pct_valid:.1f}%)")
    if len(valid_vals) > 0:
        print(f"  Elevation:   {valid_vals.min():.1f} - {valid_vals.max():.1f} m "
              f"(mean {valid_vals.mean():.1f})")

    print(f"\n--- Footprints ---")
    print(f"  Count:    {len(fp_gdf):,}")
    print(f"  Columns:  {list(fp_gdf.columns)}")
    fb = fp_gdf.total_bounds
    print(f"  Bounds:   ({fb[0]:.0f}, {fb[1]:.0f}) - ({fb[2]:.0f}, {fb[3]:.0f})")
    print(f"  Extent:   {fb[2]-fb[0]:.0f} x {fb[3]-fb[1]:.0f} m")
    # Check height/base fields
    for field in ["altura", "height", "h", "base", "elevation", "pavimento"]:
        if field in fp_gdf.columns:
            vals = fp_gdf[field].dropna()
            if len(vals) > 0 and np.issubdtype(vals.dtype, np.number):
                print(f"  {field:12s}: n={len(vals):,}, "
                      f"range=[{vals.min():.1f}, {vals.max():.1f}], "
                      f"mean={vals.mean():.1f}, median={vals.median():.1f}")
            else:
                print(f"  {field:12s}: n={len(vals):,} (non-numeric or empty)")

    print(f"\n--- Roads ---")
    print(f"  Count:      {len(roads_gdf):,}")
    print(f"  Columns:    {list(roads_gdf.columns)}")
    rb = roads_gdf.total_bounds
    print(f"  Bounds:     ({rb[0]:.0f}, {rb[1]:.0f}) - ({rb[2]:.0f}, {rb[3]:.0f})")
    total_length = roads_gdf.geometry.length.sum()
    print(f"  Total len:  {total_length:,.0f} m ({total_length/1000:.1f} km)")

    if boundary_gdf is not None:
        print(f"\n--- Boundary ---")
        print(f"  Features:  {len(boundary_gdf)}")
        bb = boundary_gdf.total_bounds
        print(f"  Bounds:    ({bb[0]:.0f}, {bb[1]:.0f}) - ({bb[2]:.0f}, {bb[3]:.0f})")
        boundary_area = boundary_gdf.geometry.area.sum()
        print(f"  Area:      {boundary_area:,.0f} m² ({boundary_area/1e6:.2f} km²)")

    # --- Compute overlaps ---
    print(f"\n--- Coverage Check ---")
    dtm_area = (dtm_bounds.right - dtm_bounds.left) * (dtm_bounds.top - dtm_bounds.bottom)
    fp_area_extent = (fb[2] - fb[0]) * (fb[3] - fb[1])

    # Intersection of DTM and footprints extents
    ix_l = max(dtm_bounds.left, fb[0])
    ix_b = max(dtm_bounds.bottom, fb[1])
    ix_r = min(dtm_bounds.right, fb[2])
    ix_t = min(dtm_bounds.top, fb[3])
    if ix_r > ix_l and ix_t > ix_b:
        ix_area = (ix_r - ix_l) * (ix_t - ix_b)
        pct_fp_covered = ix_area / fp_area_extent * 100
        print(f"  DTM covers {pct_fp_covered:.0f}% of footprint extent")
    else:
        print(f"  WARNING: DTM and footprints DO NOT OVERLAP!")

    # Buildings inside boundary
    if boundary_gdf is not None:
        from shapely.ops import unary_union
        boundary_union = unary_union(boundary_gdf.geometry.values)
        centroids = fp_gdf.geometry.centroid
        inside = centroids.within(boundary_union)
        print(f"  Buildings inside boundary: {inside.sum():,} / {len(fp_gdf):,} "
              f"({inside.sum()/len(fp_gdf)*100:.1f}%)")

    # ====================================================================
    # FIGURE 1: 6-panel data overview
    # ====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # --- Panel 1: DTM hillshade ---
    ax = axes[0, 0]
    ls = LightSource(azdeg=315, altdeg=45)
    if len(valid_vals) > 0:
        hs = ls.hillshade(dtm_clean, vert_exag=2, dx=dtm_res[0], dy=dtm_res[1])
        extent = [dtm_bounds.left, dtm_bounds.right, dtm_bounds.bottom, dtm_bounds.top]
        ax.imshow(dtm_clean, cmap="terrain", extent=extent, origin="upper", alpha=0.7,
                  vmin=valid_vals.min(), vmax=valid_vals.max())
        ax.imshow(hs, cmap="gray", extent=extent, origin="upper", alpha=0.4)
    ax.set_title(f"DTM Elevation + Hillshade\n"
                 f"{dtm.shape[0]}x{dtm.shape[1]} @ {dtm_res[0]:.1f}m, "
                 f"{pct_valid:.0f}% valid", fontsize=11)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    # --- Panel 2: Spatial alignment ---
    ax = axes[0, 1]
    # DTM extent as background rectangle
    dtm_rect = Rectangle(
        (dtm_bounds.left, dtm_bounds.bottom),
        dtm_bounds.right - dtm_bounds.left,
        dtm_bounds.top - dtm_bounds.bottom,
        linewidth=2, edgecolor="blue", facecolor="lightblue", alpha=0.15, label="DTM extent"
    )
    ax.add_patch(dtm_rect)
    fp_gdf.plot(ax=ax, color="#d9d9d9", edgecolor="#777", linewidth=0.15, alpha=0.8)
    roads_gdf.plot(ax=ax, color="#e74c3c", linewidth=0.4, alpha=0.7)
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, linestyle="--")
    legend_items = [
        Patch(facecolor="#d9d9d9", edgecolor="#777", label=f"Buildings ({len(fp_gdf):,})"),
        Patch(facecolor="#e74c3c", label=f"Roads ({len(roads_gdf):,})"),
        Patch(facecolor="lightblue", edgecolor="blue", label="DTM extent"),
    ]
    if boundary_gdf is not None:
        legend_items.append(Patch(facecolor="none", edgecolor="black", linestyle="--",
                                  label="Boundary"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=8)
    ax.set_title("Spatial Alignment", fontsize=11)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    # --- Panel 3: DTM coverage heatmap (valid/nodata) ---
    ax = axes[0, 2]
    coverage_img = np.zeros((*dtm_clean.shape, 3))
    coverage_img[dtm_valid_mask] = [0.2, 0.7, 0.3]  # green = valid
    coverage_img[~dtm_valid_mask] = [0.9, 0.2, 0.2]  # red = nodata
    extent = [dtm_bounds.left, dtm_bounds.right, dtm_bounds.bottom, dtm_bounds.top]
    ax.imshow(coverage_img, extent=extent, origin="upper")
    # Overlay footprint and road extents
    fp_rect = Rectangle(
        (fb[0], fb[1]), fb[2] - fb[0], fb[3] - fb[1],
        linewidth=2, edgecolor="orange", facecolor="none", linestyle="-", label="Footprint extent"
    )
    ax.add_patch(fp_rect)
    rd_rect = Rectangle(
        (rb[0], rb[1]), rb[2] - rb[0], rb[3] - rb[1],
        linewidth=1.5, edgecolor="purple", facecolor="none", linestyle=":", label="Road extent"
    )
    ax.add_patch(rd_rect)
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color="white", linewidth=1.5, linestyle="--")
    ax.legend(handles=[
        Patch(facecolor=[0.2, 0.7, 0.3], label=f"Valid ({pct_valid:.0f}%)"),
        Patch(facecolor=[0.9, 0.2, 0.2], label=f"NoData ({100-pct_valid:.0f}%)"),
        Patch(facecolor="none", edgecolor="orange", label="Footprint extent"),
        Patch(facecolor="none", edgecolor="purple", linestyle=":", label="Road extent"),
    ], loc="upper right", fontsize=8)
    ax.set_title("DTM Coverage vs Data Extents", fontsize=11)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    # --- Panel 4: Building height distribution ---
    ax = axes[1, 0]
    height_field = None
    for f in ["altura", "height", "h"]:
        if f in fp_gdf.columns:
            vals = fp_gdf[f].dropna()
            if len(vals) > 0 and np.issubdtype(vals.dtype, np.number):
                height_field = f
                break

    if height_field:
        heights = fp_gdf[height_field].dropna().values
        heights = heights[np.isfinite(heights) & (heights > 0)]
        if len(heights) > 0:
            ax.hist(heights, bins=50, color="#3498db", edgecolor="#2980b9", alpha=0.8)
            ax.axvline(np.mean(heights), color="red", linestyle="--",
                       label=f"Mean={np.mean(heights):.1f}m")
            ax.axvline(np.median(heights), color="orange", linestyle="--",
                       label=f"Median={np.median(heights):.1f}m")
            ax.legend(fontsize=9)
            ax.set_xlabel(f"Building height [{height_field}] (m)")
            ax.set_ylabel("Count")
            ax.set_title(f"Building Heights (n={len(heights):,})\n"
                         f"range [{heights.min():.1f}, {heights.max():.1f}]m", fontsize=11)
        else:
            ax.text(0.5, 0.5, f"'{height_field}' field found\nbut no valid values > 0",
                    transform=ax.transAxes, ha="center", va="center", fontsize=14)
            ax.set_title("Building Heights", fontsize=11)
    else:
        # Show column overview instead
        text = "No height field found.\n\nNumeric columns:\n"
        for col in fp_gdf.columns:
            if col == "geometry":
                continue
            try:
                if np.issubdtype(fp_gdf[col].dtype, np.number):
                    vals = fp_gdf[col].dropna()
                    text += f"  {col}: [{vals.min():.1f}, {vals.max():.1f}] (n={len(vals):,})\n"
            except (TypeError, ValueError):
                pass
        ax.text(0.05, 0.95, text, transform=ax.transAxes, ha="left", va="top",
                fontsize=10, family="monospace")
        ax.set_title("Building Schema (no height field)", fontsize=11)

    # --- Panel 5: Footprints with height coloring (or area) ---
    ax = axes[1, 1]
    if height_field and len(fp_gdf[height_field].dropna()) > 0:
        fp_plot = fp_gdf.copy()
        fp_plot[height_field] = fp_plot[height_field].fillna(0)
        fp_plot.plot(ax=ax, column=height_field, cmap="YlOrRd", edgecolor="#666",
                     linewidth=0.1, legend=True, legend_kwds={"shrink": 0.6, "label": f"{height_field} (m)"})
        if boundary_gdf is not None:
            boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, linestyle="--")
        ax.set_title(f"Building Heights Map ({height_field})", fontsize=11)
    else:
        # Color by area
        fp_plot = fp_gdf.copy()
        fp_plot["_area"] = fp_plot.geometry.area
        fp_plot.plot(ax=ax, column="_area", cmap="YlOrRd", edgecolor="#666",
                     linewidth=0.1, legend=True, legend_kwds={"shrink": 0.6, "label": "Area (m²)"})
        if boundary_gdf is not None:
            boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, linestyle="--")
        ax.set_title("Building Footprint Areas", fontsize=11)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    # --- Panel 6: Roads colored by length, with buildings ---
    ax = axes[1, 2]
    fp_gdf.plot(ax=ax, color="#f0f0f0", edgecolor="#ccc", linewidth=0.1)
    roads_plot = roads_gdf.copy()
    roads_plot["_length"] = roads_plot.geometry.length
    roads_plot.plot(ax=ax, column="_length", cmap="plasma", linewidth=0.8, alpha=0.9,
                    legend=True, legend_kwds={"shrink": 0.6, "label": "Segment length (m)"})
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, linestyle="--")
    ax.set_title(f"Road Network\n{len(roads_gdf):,} segments, "
                 f"{total_length/1000:.1f} km total", fontsize=11)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    fig.suptitle(f"{area} — Data Readiness Check", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = get_area_output_dir(area) / "svf_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "data_readiness_debug.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # ====================================================================
    # FIGURE 2: Zoomed boundary check with building base elevations
    # ====================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 10))

    # Panel A: DTM elevation under buildings (check base field)
    ax = axes2[0]
    # Sample DTM at building centroids
    from rasterio.transform import rowcol
    centroids = fp_gdf.geometry.centroid
    cx = centroids.x.values
    cy = centroids.y.values
    dtm_z = []
    for x, y in zip(cx, cy):
        r, c = rowcol(dtm_transform, x, y)
        r, c = int(r), int(c)
        if 0 <= r < dtm_clean.shape[0] and 0 <= c < dtm_clean.shape[1]:
            dtm_z.append(dtm_clean[r, c])
        else:
            dtm_z.append(np.nan)
    dtm_z = np.array(dtm_z)

    valid_z = np.isfinite(dtm_z)
    n_outside = (~valid_z).sum()

    if "base" in fp_gdf.columns:
        base_vals = fp_gdf["base"].values
        valid_both = valid_z & np.isfinite(base_vals)
        if valid_both.sum() > 0:
            ax.scatter(dtm_z[valid_both], base_vals[valid_both], s=1, alpha=0.3, c="#3498db")
            lims = [min(dtm_z[valid_both].min(), base_vals[valid_both].min()),
                    max(dtm_z[valid_both].max(), base_vals[valid_both].max())]
            ax.plot(lims, lims, "r--", linewidth=1, label="1:1 line")
            ax.set_xlabel("DTM elevation at centroid (m)")
            ax.set_ylabel("Building 'base' field (m)")
            ax.set_title(f"Base Elevation Check\n"
                         f"{valid_both.sum():,} buildings, {n_outside:,} outside DTM", fontsize=11)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No valid base+DTM pairs", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            ax.set_title("Base Elevation Check", fontsize=11)
    else:
        if valid_z.sum() > 0:
            ax.hist(dtm_z[valid_z], bins=50, color="#3498db", edgecolor="#2980b9", alpha=0.8)
            ax.axvline(np.nanmean(dtm_z[valid_z]), color="red", linestyle="--",
                       label=f"Mean={np.nanmean(dtm_z[valid_z]):.1f}m")
            ax.legend()
        ax.set_xlabel("DTM elevation at building centroid (m)")
        ax.set_ylabel("Count")
        ax.set_title(f"DTM Elevation at Building Centroids\n"
                     f"No 'base' field; {n_outside:,}/{len(fp_gdf):,} outside DTM", fontsize=11)

    # Panel B: Building area distribution
    ax = axes2[1]
    areas = fp_gdf.geometry.area.values
    areas = areas[areas > 0]
    ax.hist(areas, bins=80, color="#27ae60", edgecolor="#1e8449", alpha=0.8)
    ax.axvline(np.mean(areas), color="red", linestyle="--", label=f"Mean={np.mean(areas):.0f} m²")
    ax.axvline(np.median(areas), color="orange", linestyle="--", label=f"Median={np.median(areas):.0f} m²")
    # Mark outliers
    p99 = np.percentile(areas, 99)
    n_outliers = (areas > p99).sum()
    ax.axvline(p99, color="purple", linestyle=":", label=f"P99={p99:.0f} m² ({n_outliers} above)")
    ax.legend(fontsize=9)
    ax.set_xlabel("Footprint area (m²)")
    ax.set_ylabel("Count")
    ax.set_title(f"Building Footprint Areas (n={len(areas):,})\n"
                 f"range [{areas.min():.0f}, {areas.max():.0f}] m²", fontsize=11)

    fig2.suptitle(f"{area} — Building Data Quality", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path2 = output_dir / "building_quality_debug.png"
    fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out_path2}")

    # ====================================================================
    # Summary verdict
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")

    issues = []
    if pct_valid < 80:
        issues.append(f"DTM has only {pct_valid:.0f}% valid cells")
    if n_outside > len(fp_gdf) * 0.1:
        issues.append(f"{n_outside} buildings ({n_outside/len(fp_gdf)*100:.0f}%) outside DTM")
    if height_field is None:
        issues.append("No height field found (altura/height/h) — buildings will be flat")
    elif height_field:
        h = fp_gdf[height_field].dropna()
        if len(h) < len(fp_gdf) * 0.5:
            issues.append(f"Only {len(h):,}/{len(fp_gdf):,} buildings have height data")
    if "base" not in fp_gdf.columns:
        issues.append("No 'base' field — buildings will be extruded from DTM surface")
    if boundary_path is None:
        issues.append("No boundary file — grid mode will cover full DTM extent")

    if not issues:
        print("  ALL CLEAR — data looks ready for pipeline")
    else:
        for iss in issues:
            print(f"  ⚠  {iss}")

    print()


if __name__ == "__main__":
    main()
