#!/usr/bin/env python3
"""Plot street-level SVF over terrain isolines (DTM contours)."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
        print(
            f"Applying alignment shift from reference roads: dx={dx:.3f}, dy={dy:.3f}"
        )
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
    scale_mode: str = "full",
    clip_percentile: float = 95.0,
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

    # Align segments if they're in local STL coordinates (centered around 0)
    # Optional building footprints to preserve original street_svf_map components.
    # Buildings should already be in the correct CRS (EPSG:31983) and NOT need alignment.
    building_footprints = None
    if footprints_path and footprints_path.exists():
        building_footprints = gpd.read_file(footprints_path)
        if (
            building_footprints.crs
            and dtm_crs
            and str(building_footprints.crs) != str(dtm_crs)
        ):
            building_footprints = building_footprints.to_crs(dtm_crs)
        # Buildings should already overlap with DTM (they're in the same CRS)
        if not _bounds_overlap(building_footprints.total_bounds, dtm_bounds):
            print(
                "Warning: Building footprints don't overlap with DTM, attempting alignment..."
            )
            building_footprints = align_layer_to_dtm(
                building_footprints,
                dtm_bounds=dtm_bounds,
                reference_roads=reference_roads,
            )
        else:
            print("Building footprints already aligned with DTM (no shift needed)")

    # Align segments to match buildings (not roads) - this is what compute_svf_streets does
    # Segments are in STL local coordinates, need to be aligned to buildings in EPSG:31983
    if building_footprints is not None and not building_footprints.empty:
        # Use building footprints center for alignment (same as compute_svf_streets.py)
        build_center = _center(building_footprints.total_bounds)
        seg_center = _center(segments.total_bounds)
        dx = build_center[0] - seg_center[0]
        dy = build_center[1] - seg_center[1]
        print(
            f"Aligning segments to building footprints center: dx={dx:.3f}, dy={dy:.3f}"
        )
        segments = segments.copy()
        segments.geometry = segments.geometry.translate(xoff=dx, yoff=dy)
    elif reference_roads and reference_roads.exists():
        # Fallback to roads if buildings not available
        segments = align_layer_to_dtm(
            segments, dtm_bounds=dtm_bounds, reference_roads=reference_roads
        )
    else:
        # Last resort: align to DTM center
        segments = align_layer_to_dtm(
            segments, dtm_bounds=dtm_bounds, reference_roads=None
        )

    # Build X/Y grid for contouring using affine transform
    # Create row and column index arrays
    rows, cols = np.meshgrid(
        np.arange(dem.shape[0], dtype=float),
        np.arange(dem.shape[1], dtype=float),
        indexing="ij",
    )
    # Affine transform: x = a*col + b*row + c, y = d*col + e*row + f
    # Note: transform.a, transform.b, etc. are the 6 affine parameters
    xs = transform.a * cols + transform.b * rows + transform.c
    ys = transform.d * cols + transform.e * rows + transform.f

    zmin = float(np.nanmin(dem))
    zmax = float(np.nanmax(dem))
    levels = np.linspace(zmin, zmax, n_levels)
    print(f"DTM elevation range: {zmin:.2f} to {zmax:.2f} m")
    print(f"Generating {len(levels)} contour levels")
    print(
        f"DTM coordinate range: X=[{xs.min():.2f}, {xs.max():.2f}], Y=[{ys.min():.2f}, {ys.max():.2f}]"
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    # Determine plot extent from polygon bbox (buildings) and segments.
    # This keeps zoom focused on the analyzed urban fabric instead of full DTM.
    seg_bounds = segments.total_bounds
    print(f"Aligned segments bounds: {seg_bounds}")

    if building_footprints is not None and not building_footprints.empty:
        build_bounds = building_footprints.total_bounds
        plot_bounds = [
            min(build_bounds[0], seg_bounds[0]),
            min(build_bounds[1], seg_bounds[1]),
            max(build_bounds[2], seg_bounds[2]),
            max(build_bounds[3], seg_bounds[3]),
        ]
    else:
        plot_bounds = [seg_bounds[0], seg_bounds[1], seg_bounds[2], seg_bounds[3]]

    # Add a consistent margin so the map is not overly tight around features.
    width = max(plot_bounds[2] - plot_bounds[0], 1e-6)
    height = max(plot_bounds[3] - plot_bounds[1], 1e-6)
    pad_x = 0.07 * width
    pad_y = 0.07 * height
    plot_bounds[0] -= pad_x
    plot_bounds[1] -= pad_y
    plot_bounds[2] += pad_x
    plot_bounds[3] += pad_y

    print(
        f"Plot extent (padded): X=[{plot_bounds[0]:.2f}, {plot_bounds[2]:.2f}], Y=[{plot_bounds[1]:.2f}, {plot_bounds[3]:.2f}]"
    )

    # Set plot extent BEFORE drawing anything to ensure contours are visible
    ax.set_xlim(plot_bounds[0], plot_bounds[2])
    ax.set_ylim(plot_bounds[1], plot_bounds[3])
    ax.set_aspect("equal")

    # Terrain isolines: subtle dotted lines in the background.
    # Convert masked array to regular array for contour (fill masked values with NaN)
    dem_for_contour = np.ma.filled(dem, np.nan)

    # Verify coordinate arrays match DEM shape
    assert xs.shape == dem.shape, (
        f"X coordinate array shape {xs.shape} doesn't match DEM shape {dem.shape}"
    )
    assert ys.shape == dem.shape, (
        f"Y coordinate array shape {ys.shape} doesn't match DEM shape {dem.shape}"
    )

    # Draw contours FIRST (before other layers) to ensure they're visible.
    # Use two passes (minor + major) for stronger terrain readability.
    try:
        minor_levels = np.linspace(zmin, zmax, n_levels)
        major_levels = np.linspace(zmin, zmax, max(6, n_levels // 2))

        contour_minor = ax.contour(
            xs,
            ys,
            dem_for_contour,
            levels=minor_levels,
            colors="#4e6375",
            linewidths=0.75,
            linestyles="dotted",
            alpha=0.85,
            zorder=1,
        )

        contour_major = ax.contour(
            xs,
            ys,
            dem_for_contour,
            levels=major_levels,
            colors="#2f4557",
            linewidths=1.15,
            linestyles="solid",
            alpha=0.70,
            zorder=1,
        )

        # Count actual contour lines generated (not just collections).
        n_contour_lines_minor = sum(
            len(seg) > 0 for coll in contour_minor.allsegs for seg in coll
        )
        n_contour_lines_major = sum(
            len(seg) > 0 for coll in contour_major.allsegs for seg in coll
        )
        n_contour_lines = n_contour_lines_minor + n_contour_lines_major
        print(
            f"Generated {len(contour_minor.allsegs) + len(contour_major.allsegs)} contour collections "
            f"with {n_contour_lines} total contour lines"
        )
        if n_contour_lines == 0:
            print(
                "WARNING: No contour lines were generated! Check coordinate grid and DEM data."
            )
    except Exception as e:
        print(f"ERROR generating contours: {e}")
        print(f"DEM shape: {dem.shape}, X shape: {xs.shape}, Y shape: {ys.shape}")
        print(
            f"X range: [{xs.min():.2f}, {xs.max():.2f}], Y range: [{ys.min():.2f}, {ys.max():.2f}]"
        )
        raise

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

    # SVF segments (supports full-range or percentile-capped minmax scaling).
    svf_values = segments["svf_mean"].to_numpy(dtype=float)
    if scale_mode == "minmax":
        svf_min = float(np.nanmin(svf_values))
        svf_high = float(np.nanpercentile(svf_values, clip_percentile))
        if not np.isfinite(svf_high) or svf_high <= svf_min:
            svf_high = float(np.nanmax(svf_values))
        denom = max(svf_high - svf_min, 1e-9)
        segments = segments.copy()
        segments["__svf_plot"] = np.clip(svf_values, svf_min, svf_high)
        cmap = LinearSegmentedColormap.from_list(
            "brown_to_sand",
            ["#5C3A21", "#C8A26E", "#E8D8B0"],
        )
        norm = Normalize(vmin=svf_min, vmax=svf_high)
        legend_label = f"Street SVF (min-max p{int(clip_percentile)} capped)"
    else:
        segments = segments.copy()
        segments["__svf_plot"] = svf_values
        cmap = "RdYlGn"
        norm = Normalize(vmin=0.0, vmax=1.0)
        legend_label = "Street SVF (mean)"

    segments.plot(
        ax=ax,
        column="__svf_plot",
        cmap=cmap,
        vmin=norm.vmin,
        vmax=norm.vmax,
        linewidth=3.0,
        legend=False,
        zorder=4,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.01, shrink=0.8)
    cbar.set_label(legend_label)

    # Re-apply plot extent after geopandas plots (they might reset it)
    ax.set_xlim(plot_bounds[0], plot_bounds[2])
    ax.set_ylim(plot_bounds[1], plot_bounds[3])
    ax.set_aspect("equal")

    ax.set_title(
        "Street-Level SVF with Terrain Isolines", fontsize=14, fontweight="bold"
    )
    ax.set_axis_off()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot street SVF with terrain isolines."
    )
    parser.add_argument(
        "--segments", required=True, help="Path to street_svf_segments.gpkg"
    )
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
    parser.add_argument(
        "--levels", type=int, default=18, help="Number of contour levels"
    )
    parser.add_argument(
        "--scale-mode",
        choices=["full", "minmax"],
        default="full",
        help="SVF color scaling: full range [0,1] or percentile-capped minmax.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=95.0,
        help="Upper percentile cap used when --scale-mode=minmax (default: 95).",
    )
    args = parser.parse_args()

    plot_svf_with_isolines(
        segments_path=Path(args.segments),
        dtm_path=Path(args.dtm),
        output_path=Path(args.output),
        footprints_path=Path(args.footprints) if args.footprints else None,
        reference_roads=Path(args.reference_roads) if args.reference_roads else None,
        n_levels=args.levels,
        scale_mode=args.scale_mode,
        clip_percentile=args.clip_percentile,
    )


if __name__ == "__main__":
    main()
