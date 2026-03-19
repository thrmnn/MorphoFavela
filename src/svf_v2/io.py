"""
Output: GeoPackage, GeoTIFF, CSV, plots, and 3D visual inspection exports.
"""

import numpy as np
import pyvista as pv
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from shapely.geometry import Point
import logging

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Grid results
# ---------------------------------------------------------------------------


def save_grid_results(
    points: np.ndarray,
    svf: np.ndarray,
    crs,
    output_dir: Path,
    grid_spacing: float = 2.0,
):
    """
    Save grid SVF results as GeoPackage, GeoTIFF, and CSV.

    Args:
        points: Nx3 observer positions (x, y, z).
        svf: N-length SVF array.
        crs: Coordinate reference system.
        output_dir: Directory for outputs.
        grid_spacing: Grid cell size (used for rasterisation).
    """
    out = _ensure_dir(output_dir)

    # GeoPackage (point layer)
    gdf = gpd.GeoDataFrame(
        {"svf": svf, "z": points[:, 2]},
        geometry=[Point(x, y) for x, y in zip(points[:, 0], points[:, 1])],
        crs=crs,
    )
    gpkg_path = out / "svf_grid.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")
    logger.info(f"  Saved {gpkg_path} ({len(gdf)} points)")

    # CSV
    csv_path = out / "svf_grid.csv"
    gdf.assign(x=points[:, 0], y=points[:, 1]).drop(columns="geometry").to_csv(
        csv_path, index=False
    )

    # GeoTIFF (interpolated raster)
    _rasterise_svf(points, svf, crs, out / "svf_grid.tif", grid_spacing)

    # Heatmap plot
    _plot_svf_heatmap(points, svf, out / "svf_heatmap.png")


def _rasterise_svf(
    points: np.ndarray,
    svf: np.ndarray,
    crs,
    output_path: Path,
    cell_size: float,
):
    """Rasterise SVF points into a GeoTIFF using scipy griddata."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from scipy.interpolate import griddata
    except ImportError:
        logger.warning("rasterio/scipy not available -- skipping GeoTIFF")
        return

    xs, ys = points[:, 0], points[:, 1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    cols = int(np.ceil((xmax - xmin) / cell_size))
    rows = int(np.ceil((ymax - ymin) / cell_size))
    if cols < 2 or rows < 2:
        logger.warning("Grid too small for rasterisation")
        return

    xi = np.linspace(xmin, xmax, cols)
    yi = np.linspace(ymax, ymin, rows)  # top-to-bottom
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    grid_z = griddata(
        np.column_stack([xs, ys]),
        svf,
        (xi_grid, yi_grid),
        method="linear",
        fill_value=np.nan,
    )

    transform = from_bounds(xmin, ymin, xmax, ymax, cols, rows)

    with rasterio.open(
        str(output_path),
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(grid_z.astype(np.float32), 1)

    logger.info(f"  Saved {output_path} ({cols}x{rows})")


def _plot_svf_heatmap(points: np.ndarray, svf: np.ndarray, output_path: Path):
    """Simple scatter-plot heatmap of SVF values."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=svf,
        cmap="RdYlGn",
        s=2,
        vmin=0,
        vmax=1,
        alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, label="SVF")
    ax.set_aspect("equal")
    ax.set_title("Sky View Factor (Grid)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Street results
# ---------------------------------------------------------------------------


def save_street_results(
    street_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    roads_gdf: Optional[gpd.GeoDataFrame] = None,
):
    """
    Save street SVF results.

    Args:
        street_gdf: GeoDataFrame with ``svf`` column and ``street_id``.
        output_dir: Directory for outputs.
        roads_gdf: Original road lines (for segment aggregation).
    """
    out = _ensure_dir(output_dir)

    # Point-level GeoPackage
    gpkg_path = out / "svf_streets.gpkg"
    street_gdf.to_file(gpkg_path, driver="GPKG")
    logger.info(f"  Saved {gpkg_path} ({len(street_gdf)} points)")

    # Per-segment aggregation
    if "street_id" in street_gdf.columns and roads_gdf is not None:
        seg_stats = (
            street_gdf.groupby("street_id")["svf"]
            .agg(["mean", "median", "min", "max", "count"])
            .reset_index()
        )
        seg_stats.columns = [
            "street_id",
            "svf_mean",
            "svf_median",
            "svf_min",
            "svf_max",
            "n_points",
        ]

        # Join with road geometry
        roads_copy = roads_gdf.copy().reset_index(drop=True)
        roads_copy["street_id"] = roads_copy.index
        seg_gdf = roads_copy.merge(seg_stats, on="street_id", how="inner")
        seg_gdf = gpd.GeoDataFrame(seg_gdf, crs=roads_gdf.crs)

        seg_path = out / "svf_streets_segments.gpkg"
        seg_gdf.to_file(seg_path, driver="GPKG")
        logger.info(f"  Saved {seg_path} ({len(seg_gdf)} segments)")

    # Map plot
    _plot_street_svf(street_gdf, out / "svf_streets_map.png", roads_gdf=roads_gdf)


def _plot_street_svf(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    roads_gdf: Optional[gpd.GeoDataFrame] = None,
):
    fig, ax = plt.subplots(figsize=(12, 10))
    if roads_gdf is not None:
        roads_gdf.plot(ax=ax, color="lightgray", linewidth=0.5)
    gdf.plot(
        ax=ax,
        column="svf",
        cmap="RdYlGn",
        markersize=4,
        vmin=0,
        vmax=1,
        legend=True,
        legend_kwds={"label": "SVF", "shrink": 0.6},
    )
    ax.set_aspect("equal")
    ax.set_title("Street SVF")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Facade results
# ---------------------------------------------------------------------------


def save_facade_results(
    facade_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    footprints_gdf: Optional[gpd.GeoDataFrame] = None,
):
    """
    Save facade SVF results.

    Args:
        facade_gdf: GeoDataFrame with ``svf``, ``facade_azimuth``,
            ``height_above_ground``, ``building_id`` columns.
        output_dir: Directory for outputs.
        footprints_gdf: Building footprints for per-building summary.
    """
    out = _ensure_dir(output_dir)

    # Point-level GeoPackage
    gpkg_path = out / "svf_facades.gpkg"
    facade_gdf.to_file(gpkg_path, driver="GPKG")
    logger.info(f"  Saved {gpkg_path} ({len(facade_gdf)} points)")

    # Per-building summary
    if "building_id" in facade_gdf.columns and footprints_gdf is not None:
        bldg_stats = (
            facade_gdf.groupby("building_id")["svf"]
            .agg(["mean", "median", "min", "max", "count"])
            .reset_index()
        )
        bldg_stats.columns = [
            "building_id",
            "svf_mean",
            "svf_median",
            "svf_min",
            "svf_max",
            "n_points",
        ]

        # Include solar potential if available
        if "solar_potential" in facade_gdf.columns:
            sp_stats = (
                facade_gdf.groupby("building_id")["solar_potential"]
                .agg(["mean", "max"])
                .reset_index()
            )
            sp_stats.columns = ["building_id", "solar_mean", "solar_max"]
            bldg_stats = bldg_stats.merge(sp_stats, on="building_id")

        # Join with building geometry
        fp_copy = footprints_gdf.copy()
        fp_copy["building_id"] = fp_copy.index
        summary_gdf = fp_copy.merge(bldg_stats, on="building_id", how="inner")
        summary_gdf = gpd.GeoDataFrame(summary_gdf, crs=footprints_gdf.crs)

        summary_path = out / "svf_facades_summary.gpkg"
        summary_gdf.to_file(summary_path, driver="GPKG")
        logger.info(f"  Saved {summary_path} ({len(summary_gdf)} buildings)")

    # Plots
    _plot_facade_by_orientation(facade_gdf, out / "svf_facades_orientation.png")
    _plot_facade_by_height(facade_gdf, out / "svf_facades_height.png")


def _plot_facade_by_orientation(gdf: gpd.GeoDataFrame, output_path: Path):
    if "svf" not in gdf.columns or "facade_azimuth" not in gdf.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "polar"})
    az_rad = np.radians(gdf["facade_azimuth"].values)
    sc = ax.scatter(
        az_rad,
        gdf["svf"].values,
        c=gdf["svf"].values,
        cmap="RdYlGn",
        s=1,
        alpha=0.3,
        vmin=0,
        vmax=1,
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Facade SVF by Orientation")
    plt.colorbar(sc, ax=ax, label="SVF", shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {output_path}")


def _plot_facade_by_height(gdf: gpd.GeoDataFrame, output_path: Path):
    if "svf" not in gdf.columns or "height_above_ground" not in gdf.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        gdf["height_above_ground"].values,
        gdf["svf"].values,
        c=gdf["facade_azimuth"].values
        if "facade_azimuth" in gdf.columns
        else "steelblue",
        cmap="hsv",
        s=1,
        alpha=0.3,
    )
    ax.set_xlabel("Height above ground (m)")
    ax.set_ylabel("SVF")
    ax.set_title("Facade SVF vs Height")
    ax.set_ylim(0, 1)
    if "facade_azimuth" in gdf.columns:
        plt.colorbar(sc, ax=ax, label="Azimuth")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# 3D scene export (visual inspection)
# ---------------------------------------------------------------------------


def save_scene_stl(
    scene_mesh: pv.PolyData,
    output_path: Path,
    binary: bool = True,
) -> Path:
    """Export combined scene (terrain + buildings) as STL for MeshLab/ParaView."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_mesh.save(str(output_path), binary=binary)
    logger.info(
        f"  Saved scene STL: {output_path} "
        f"({scene_mesh.n_points} pts, {scene_mesh.n_cells} cells)"
    )
    return output_path


def save_scene_vtk(
    scene_mesh: pv.PolyData,
    output_path: Path,
) -> Path:
    """Export combined scene as VTK (float64, preserves exact coordinates)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_mesh.save(str(output_path))
    logger.info(
        f"  Saved scene VTK: {output_path} "
        f"({scene_mesh.n_points} pts, {scene_mesh.n_cells} cells)"
    )
    return output_path


# ---------------------------------------------------------------------------
# Alignment check plots (visual inspection)
# ---------------------------------------------------------------------------


def plot_alignment_check(
    footprints_gdf: gpd.GeoDataFrame,
    output_path: Path,
    roads_gdf: Optional[gpd.GeoDataFrame] = None,
    sample_points: Optional[np.ndarray] = None,
    title: str = "Alignment Check",
) -> Path:
    """
    2D diagnostic overlay: building footprints + road lines + sample points.

    Verifies that all datasets are spatially aligned (same CRS, no offsets).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax_idx, ax in enumerate(axes):
        # Buildings
        footprints_gdf.plot(
            ax=ax,
            facecolor="lightgrey",
            edgecolor="black",
            linewidth=0.3,
            alpha=0.6,
            label="Buildings",
        )

        # Roads
        if roads_gdf is not None:
            roads_gdf.plot(ax=ax, color="blue", linewidth=0.8, alpha=0.7, label="Roads")

        # Sample points (cap at 5000 for render speed)
        if sample_points is not None and len(sample_points) > 0:
            pts = sample_points[:5000]
            ax.scatter(
                pts[:, 0], pts[:, 1], c="red", s=1, alpha=0.5, label="Sample pts"
            )

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        if ax_idx == 0:
            ax.set_title(f"{title} (full extent)")
            ax.legend(loc="upper right", fontsize=8)
        else:
            # Zoom to ~200m region near centroid
            bounds = footprints_gdf.total_bounds  # minx, miny, maxx, maxy
            cx = (bounds[0] + bounds[2]) / 2
            cy = (bounds[1] + bounds[3]) / 2
            zoom = 100  # half-width in metres
            ax.set_xlim(cx - zoom, cx + zoom)
            ax.set_ylim(cy - zoom, cy + zoom)
            ax.set_title(f"{title} (200m zoom at centroid)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {output_path}")
    return output_path


def plot_road_z_check(
    street_gdf: gpd.GeoDataFrame,
    output_path: Path,
    dtm_z_range: Optional[tuple] = None,
) -> Path:
    """
    Scatter of street sample points coloured by Z elevation.

    Annotates with DTM Z range so you can verify road points are on the terrain
    (not floating or underground).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = np.array([p.x for p in street_gdf.geometry])
    ys = np.array([p.y for p in street_gdf.geometry])
    zs = street_gdf["z"].values

    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(xs, ys, c=zs, cmap="terrain", s=3, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Z elevation (m)")
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    title = "Street Sample Points -- Z Check"
    if dtm_z_range:
        title += f"\nDTM range: {dtm_z_range[0]:.1f} - {dtm_z_range[1]:.1f} m"
        z_valid = zs[np.isfinite(zs)]
        if len(z_valid) > 0:
            title += f" | Sample range: {z_valid.min():.1f} - {z_valid.max():.1f} m"
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {output_path}")
    return output_path
