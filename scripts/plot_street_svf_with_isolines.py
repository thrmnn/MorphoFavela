#!/usr/bin/env python3
"""Plot street-level SVF over terrain isolines (DTM contours)."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import box


def _bounds_overlap(a_bounds, b_bounds) -> bool:
    """Return True if two bounds boxes overlap."""
    a = box(*a_bounds)
    b = box(*b_bounds)
    return a.intersects(b)


def _center(bounds):
    """Compute center of bounds tuple."""
    minx, miny, maxx, maxy = bounds
    return (minx + maxx) / 2.0, (miny + maxy) / 2.0


def align_layer_to_dtm(
    layer_gdf: gpd.GeoDataFrame,
    dtm_bounds,
    reference_roads: Path | None = None,
) -> gpd.GeoDataFrame:
    """
    Align segments to DTM extent if they are in local STL coordinates.

    This happens when the SVF script translates roads to mesh-local coordinates.
    """
    if _bounds_overlap(layer_gdf.total_bounds, dtm_bounds):
        return layer_gdf

    if reference_roads and reference_roads.exists():
        ref = gpd.read_file(reference_roads)
        dx = _center(ref.total_bounds)[0] - _center(layer_gdf.total_bounds)[0]
        dy = _center(ref.total_bounds)[1] - _center(layer_gdf.total_bounds)[1]
        print(f"Applying alignment shift from reference roads: dx={dx:.3f}, dy={dy:.3f}")
        aligned = layer_gdf.copy()
        aligned.geometry = aligned.geometry.translate(xoff=dx, yoff=dy)
        return aligned

    # Fallback: shift to DTM center (less reliable but usually works for local coords).
    dx = _center(dtm_bounds)[0] - _center(layer_gdf.total_bounds)[0]
    dy = _center(dtm_bounds)[1] - _center(layer_gdf.total_bounds)[1]
    print(f"Applying fallback center alignment shift: dx={dx:.3f}, dy={dy:.3f}")
    aligned = layer_gdf.copy()
    aligned.geometry = aligned.geometry.translate(xoff=dx, yoff=dy)
    return aligned


def plot_svf_with_isolines(
    segments_path: Path,
    dtm_path: Path,
    output_path: Path,
    footprints_path: Path | None = None,
    reference_roads: Path | None = None,
    n_levels: int = 18,
) -> None:
    """Create map of SVF segments with DTM contour lines."""
    if not segments_path.exists():
        raise FileNotFoundError(f"Segments file not found: {segments_path}")
    if not dtm_path.exists():
        raise FileNotFoundError(f"DTM file not found: {dtm_path}")

    segments = gpd.read_file(segments_path)
    if "svf_mean" not in segments.columns:
        raise ValueError("Expected 'svf_mean' column in segments layer.")

    with rasterio.open(dtm_path) as src:
        dem = src.read(1, masked=True)
        if src.nodata is not None:
            dem = np.ma.masked_equal(dem, src.nodata)
        dtm_bounds = src.bounds
        dtm_crs = src.crs
        transform = src.transform

    if segments.crs and dtm_crs and str(segments.crs) != str(dtm_crs):
        segments = segments.to_crs(dtm_crs)

    segments = align_layer_to_dtm(segments, dtm_bounds=dtm_bounds, reference_roads=reference_roads)

    # Optional building footprints to preserve original street_svf_map components.
    building_footprints = None
    if footprints_path and footprints_path.exists():
        building_footprints = gpd.read_file(footprints_path)
        if building_footprints.crs and dtm_crs and str(building_footprints.crs) != str(dtm_crs):
            building_footprints = building_footprints.to_crs(dtm_crs)
        building_footprints = align_layer_to_dtm(
            building_footprints, dtm_bounds=dtm_bounds, reference_roads=reference_roads
        )

    # Build X/Y grid for contouring
    rows, cols = np.indices(dem.shape)
    # Build 2D coordinate grids matching DEM shape.
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e

    zmin = float(np.nanmin(dem))
    zmax = float(np.nanmax(dem))
    levels = np.linspace(zmin, zmax, n_levels)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Terrain isolines: subtle dotted lines in the background.
    contour = ax.contour(
        xs,
        ys,
        dem,
        levels=levels,
        colors="#3a506b",
        linewidths=0.35,
        linestyles="dotted",
        alpha=0.35,
        zorder=1,
    )

    # Same base component as street_svf_map: building footprints background.
    if building_footprints is not None and not building_footprints.empty:
        building_footprints.plot(
            ax=ax,
            facecolor="lightgrey",
            edgecolor="black",
            linewidth=0.3,
            alpha=0.45,
            zorder=2,
        )

    # SVF segments
    segments.plot(
        ax=ax,
        column="svf_mean",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidth=3.0,
        legend=True,
        legend_kwds={"label": "Street SVF (mean)", "shrink": 0.8},
        zorder=4,
    )

    ax.set_title("Street-Level SVF with Terrain Isolines", fontsize=14, fontweight="bold")
    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot street SVF with terrain isolines.")
    parser.add_argument("--segments", required=True, help="Path to street_svf_segments.gpkg")
    parser.add_argument("--dtm", required=True, help="Path to DTM raster")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument(
        "--footprints",
        default=None,
        help="Optional building footprints layer to match street_svf_map context",
    )
    parser.add_argument(
        "--reference-roads",
        default=None,
        help="Optional roads shapefile in original CRS for robust alignment",
    )
    parser.add_argument("--levels", type=int, default=18, help="Number of contour levels")
    args = parser.parse_args()

    plot_svf_with_isolines(
        segments_path=Path(args.segments),
        dtm_path=Path(args.dtm),
        output_path=Path(args.output),
        footprints_path=Path(args.footprints) if args.footprints else None,
        reference_roads=Path(args.reference_roads) if args.reference_roads else None,
        n_levels=args.levels,
    )


if __name__ == "__main__":
    main()
