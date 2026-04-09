#!/usr/bin/env python3
"""
Debug visualization for grid point selection with proximity filter.

Generates a multi-panel figure showing kept/discarded grid points at
different buffer distances, helping tune the buffer parameter.

Usage:
    python scripts/debug_grid_selection.py --area maré
    python scripts/debug_grid_selection.py --area maré --grid-spacing 5.0 --buffers 10 20 30 50 100
"""

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from shapely import STRtree
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_output_dir
from src.svf_v2.paths import resolve_boundary, resolve_paths


def generate_grid(dtm_path, grid_spacing):
    """Generate flat grid points from DTM extent."""
    import rasterio

    with rasterio.open(dtm_path) as src:
        bounds = src.bounds
    xs = np.arange(bounds.left, bounds.right, grid_spacing)
    ys = np.arange(bounds.bottom, bounds.top, grid_spacing)
    xx, yy = np.meshgrid(xs, ys)
    return xx.ravel(), yy.ravel()


def apply_boundary_clip(flat_x, flat_y, boundary_gdf):
    """Clip points to boundary polygon."""
    from shapely.ops import unary_union

    boundary_poly = unary_union(boundary_gdf.geometry.values)
    pts = gpd.points_from_xy(flat_x, flat_y)
    mask = np.array([boundary_poly.contains(p) for p in pts])
    return flat_x[mask], flat_y[mask], mask


def apply_building_mask(flat_x, flat_y, footprints_gdf):
    """Remove points inside buildings."""
    pts_gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(flat_x, flat_y)],
        crs=footprints_gdf.crs,
    )
    joined = gpd.sjoin(
        pts_gdf, footprints_gdf[["geometry"]], how="left", predicate="within"
    )
    inside_mask = ~joined["index_right"].isna()
    if joined.index.duplicated().any():
        inside_mask = inside_mask.groupby(inside_mask.index).any()
    ground_mask = ~inside_mask.values[: len(flat_x)]
    return flat_x[ground_mask], flat_y[ground_mask]


def apply_proximity_filter(flat_x, flat_y, footprints_gdf, buffer_dist):
    """Filter points by distance to nearest building using STRtree."""
    import shapely

    tree = STRtree(footprints_gdf.geometry.values)
    pts = gpd.points_from_xy(flat_x, flat_y)
    nearest_idx = tree.nearest(pts)
    geom_arr = np.asarray(footprints_gdf.geometry.values)
    distances = shapely.distance(pts, geom_arr[nearest_idx])
    mask = distances <= buffer_dist
    return flat_x[mask], flat_y[mask], mask, distances


def main():
    parser = argparse.ArgumentParser(description="Debug grid point selection")
    parser.add_argument("--area", required=True, help="Area name")
    parser.add_argument("--grid-spacing", type=float, default=5.0,
                        help="Grid spacing in metres (default: 5.0 for faster debug)")
    parser.add_argument("--buffers", type=float, nargs="+", default=[10, 20, 30, 50],
                        help="Buffer distances to compare (default: 10 20 30 50)")
    parser.add_argument("--max-plot-points", type=int, default=50000,
                        help="Max discarded points to plot (subsampled for speed)")
    args = parser.parse_args()

    dtm_path, footprints_path, roads_path = resolve_paths(args.area)
    boundary_path = resolve_boundary(args.area)

    print(f"Area: {args.area}")
    print(f"Grid spacing: {args.grid_spacing}m")
    print(f"Buffer distances: {args.buffers}")

    # Load data
    footprints_gdf = gpd.read_file(footprints_path)
    roads_gdf = gpd.read_file(roads_path)
    boundary_gdf = gpd.read_file(boundary_path) if boundary_path else None

    # Reproject to DTM CRS
    import rasterio
    with rasterio.open(dtm_path) as src:
        dtm_crs = src.crs
    if footprints_gdf.crs != dtm_crs:
        footprints_gdf = footprints_gdf.to_crs(dtm_crs)
    if roads_gdf.crs != dtm_crs:
        roads_gdf = roads_gdf.to_crs(dtm_crs)
    if boundary_gdf is not None and boundary_gdf.crs != dtm_crs:
        boundary_gdf = boundary_gdf.to_crs(dtm_crs)

    # Generate base grid
    t0 = time.time()
    flat_x, flat_y = generate_grid(dtm_path, args.grid_spacing)
    n_total = len(flat_x)
    print(f"Total grid points: {n_total:,} ({time.time()-t0:.1f}s)")

    # Boundary clip
    if boundary_gdf is not None:
        t0 = time.time()
        flat_x, flat_y, _ = apply_boundary_clip(flat_x, flat_y, boundary_gdf)
        n_after_boundary = len(flat_x)
        print(f"After boundary clip: {n_after_boundary:,} ({time.time()-t0:.1f}s)")
    else:
        n_after_boundary = len(flat_x)

    # Building mask
    t0 = time.time()
    flat_x, flat_y = apply_building_mask(flat_x, flat_y, footprints_gdf)
    n_ground = len(flat_x)
    print(f"After building mask: {n_ground:,} ({time.time()-t0:.1f}s)")

    # Compute distances to nearest building (once, reuse for all buffers)
    import shapely

    t0 = time.time()
    tree = STRtree(footprints_gdf.geometry.values)
    pts = gpd.points_from_xy(flat_x, flat_y)
    nearest_idx = tree.nearest(pts)
    geom_arr = np.asarray(footprints_gdf.geometry.values)
    distances = shapely.distance(pts, geom_arr[nearest_idx])
    print(f"Distance computation: {time.time()-t0:.1f}s")
    print(f"Distance stats: min={distances.min():.1f}m, "
          f"median={np.median(distances):.1f}m, max={distances.max():.1f}m")

    # Create multi-panel figure
    n_buffers = len(args.buffers)
    fig, axes = plt.subplots(2, (n_buffers + 1) // 2, figsize=(8 * ((n_buffers + 1) // 2), 16))
    axes = axes.ravel()

    summary_counts = []

    for i, buf_dist in enumerate(sorted(args.buffers)):
        ax = axes[i]
        mask = distances <= buf_dist
        n_kept = mask.sum()
        n_disc = (~mask).sum()
        pct = n_kept / n_ground * 100

        summary_counts.append((buf_dist, n_kept))

        # Plot buildings
        footprints_gdf.plot(ax=ax, color="#d9d9d9", edgecolor="#999999", linewidth=0.2)

        # Plot roads
        roads_gdf.plot(ax=ax, color="#4a90d9", linewidth=0.5, alpha=0.6)

        # Plot boundary
        if boundary_gdf is not None:
            boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, linestyle="--")

        # Plot discarded points (subsampled)
        disc_x, disc_y = flat_x[~mask], flat_y[~mask]
        if len(disc_x) > args.max_plot_points:
            idx = np.random.choice(len(disc_x), args.max_plot_points, replace=False)
            disc_x, disc_y = disc_x[idx], disc_y[idx]
        ax.scatter(disc_x, disc_y, s=0.1, c="#ff6b6b", alpha=0.3, rasterized=True)

        # Plot kept points (subsampled if too many)
        kept_x, kept_y = flat_x[mask], flat_y[mask]
        if len(kept_x) > args.max_plot_points:
            idx = np.random.choice(len(kept_x), args.max_plot_points, replace=False)
            kept_x, kept_y = kept_x[idx], kept_y[idx]
        ax.scatter(kept_x, kept_y, s=0.3, c="#2ecc71", alpha=0.5, rasterized=True)

        ax.set_title(
            f"Buffer = {buf_dist}m\n"
            f"{n_kept:,} kept ({pct:.0f}%) / {n_disc:,} discarded",
            fontsize=11,
        )
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    # Summary bar chart in last panel (if odd number of buffers, use last axes)
    ax_summary = axes[n_buffers] if n_buffers < len(axes) else None
    if ax_summary is not None:
        dists = [s[0] for s in summary_counts]
        counts = [s[1] for s in summary_counts]
        bars = ax_summary.bar(range(len(dists)), counts, color="#2ecc71", edgecolor="#27ae60")
        ax_summary.set_xticks(range(len(dists)))
        ax_summary.set_xticklabels([f"{d}m" for d in dists])
        ax_summary.set_ylabel("Grid points kept")
        ax_summary.set_title("Buffer distance vs point count")
        ax_summary.axhline(n_ground, color="#e74c3c", linestyle="--", label=f"No filter: {n_ground:,}")
        for bar, count in zip(bars, counts):
            ax_summary.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + n_ground * 0.01,
                            f"{count:,}", ha="center", va="bottom", fontsize=9)
        ax_summary.legend()

    # Hide unused axes
    for j in range(n_buffers + (1 if ax_summary else 0), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{args.area} — Grid Selection Debug\n"
        f"Spacing={args.grid_spacing}m | Total={n_total:,} | "
        f"Boundary={n_after_boundary:,} | Ground={n_ground:,}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    output_dir = get_area_output_dir(args.area) / "svf_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "grid_selection_debug.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print(f"\n{'Buffer':>8s}  {'Kept':>10s}  {'% of ground':>12s}  {'Reduction':>10s}")
    print("-" * 48)
    for buf_dist, n_kept in summary_counts:
        print(f"{buf_dist:>7.0f}m  {n_kept:>10,}  {n_kept/n_ground*100:>11.1f}%  {(1-n_kept/n_ground)*100:>9.1f}%")
    print(f"{'None':>8s}  {n_ground:>10,}  {'100.0':>11s}%  {'0.0':>9s}%")


if __name__ == "__main__":
    main()
