#!/usr/bin/env python3
"""
Generate spatial risk visualizations from morphology metrics.

Usage:
    python scripts/analyze_morphology_risk.py --area riodaspedras
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from shapely.geometry import box

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_analysis_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


RISK_METRICS = [
    "building_adjacency",
    "shared_walls",
    "shape_index",
    "tessellation_area",
    "covered_area_ratio",
    "convexity",
    "square_compactness",
]


def _normalize_series(series: np.ndarray) -> np.ndarray:
    valid = np.isfinite(series)
    if not valid.any():
        return np.full_like(series, np.nan, dtype=float)
    mean = np.mean(series[valid])
    std = np.std(series[valid])
    if std == 0:
        return np.full_like(series, np.nan, dtype=float)
    out = (series - mean) / std
    return out


def _compute_risk_index(gdf: gpd.GeoDataFrame, weights: dict[str, float]) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    contributions = []
    for metric, weight in weights.items():
        if metric not in gdf.columns:
            logger.warning(f"Metric missing for risk index: {metric}")
            continue
        values = gdf[metric].to_numpy(dtype=float)
        # Invert protective metrics
        if metric in {"convexity", "square_compactness", "tessellation_area"}:
            values = -values
        z = _normalize_series(values)
        gdf[f"risk_{metric}"] = z
        contributions.append(weight * z)
    if contributions:
        risk = np.sum(contributions, axis=0)
    else:
        risk = np.full(len(gdf), np.nan)
    gdf["risk_index"] = risk
    return gdf


def _create_grid(bounds: tuple[float, float, float, float], cell_size: float) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)
    polys = []
    for x in xs:
        for y in ys:
            polys.append(box(x, y, x + cell_size, y + cell_size))
    return gpd.GeoDataFrame(geometry=polys)


def _grid_aggregate(gdf: gpd.GeoDataFrame, cell_size: float) -> gpd.GeoDataFrame:
    grid = _create_grid(gdf.total_bounds, cell_size)
    grid = grid.set_crs(gdf.crs)
    # Assign each building once using its centroid
    centroids = gdf.copy()
    centroids["geometry"] = centroids.geometry.centroid
    joined = gpd.sjoin(centroids, grid, predicate="within", how="left")
    grouped = joined.groupby("index_right")
    grid["risk_mean"] = grouped["risk_index"].mean()
    grid["risk_p90"] = grouped["risk_index"].quantile(0.9)
    grid["building_count"] = grouped.size()
    grid = grid[grid["building_count"].fillna(0) > 0].copy()
    return grid


def _plot_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    output_path: Path,
    title: str,
    cmap: str = "magma",
    diverging: bool = False,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    data = gdf[column].replace([np.inf, -np.inf], np.nan)
    if diverging:
        from matplotlib.colors import TwoSlopeNorm
        vmax = np.nanpercentile(data, 98) if np.isfinite(data).any() else 1.0
        vmin = np.nanpercentile(data, 2) if np.isfinite(data).any() else -1.0
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        gdf.plot(column=column, ax=ax, cmap=cmap, legend=True, legend_kwds={"label": column}, norm=norm)
    else:
        gdf.plot(column=column, ax=ax, cmap=cmap, legend=True, legend_kwds={"label": column})
    ax.set_title(title)
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved map: {output_path}")


def _plot_hotspot_class(gdf: gpd.GeoDataFrame, output_path: Path, title: str) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    cmap = ListedColormap(["#2c7fb8", "#dddddd", "#d7191c"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    gdf.plot(
        column="hotspot_class",
        ax=ax,
        cmap=cmap,
        norm=norm,
        legend=True,
        legend_kwds={"label": "Hotspot Class (-1 cold, 0 ns, 1 hot)"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved map: {output_path}")


def _plot_bivariate(
    gdf: gpd.GeoDataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str
) -> None:
    x = gdf[x_col]
    y = gdf[y_col]
    x_q = pd.qcut(x.rank(method="first"), 3, labels=[0, 1, 2])
    y_q = pd.qcut(y.rank(method="first"), 3, labels=[0, 1, 2])
    bivariate = x_q.astype(int) * 3 + y_q.astype(int)
    gdf = gdf.copy()
    gdf["bivariate_class"] = bivariate
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(column="bivariate_class", ax=ax, cmap="viridis", legend=False)
    ax.set_title(title)
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved bivariate map: {output_path}")


def _sample_line_points(geom, spacing: float) -> list:
    points = []
    if geom is None or geom.is_empty:
        return points
    geoms = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
    for line in geoms:
        length = line.length
        if length == 0:
            continue
        distances = np.arange(0.0, length + spacing, spacing)
        for d in distances:
            points.append(line.interpolate(min(d, length)))
    return points


def _compute_street_canyon_hw(
    streets: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    spacing: float,
) -> gpd.GeoDataFrame:
    if streets is None or streets.empty:
        return gpd.GeoDataFrame(geometry=[], crs=buildings.crs)
    if "height" not in buildings.columns:
        logger.warning("Street canyon H/W requires building heights. Skipping.")
        return gpd.GeoDataFrame(geometry=[], crs=buildings.crs)

    sindex = buildings.sindex
    heights = buildings["height"].to_numpy(dtype=float)
    points = []
    street_ids = []
    for idx, geom in enumerate(streets.geometry):
        for pt in _sample_line_points(geom, spacing):
            points.append(pt)
            street_ids.append(idx)

    if not points:
        return gpd.GeoDataFrame(geometry=[], crs=buildings.crs)

    hw_values = []
    for pt in points:
        nearest = sindex.nearest(pt, return_all=False)
        building_idx = int(nearest[1][0])
        building_geom = buildings.geometry.iloc[building_idx]
        dist = pt.distance(building_geom)
        height = heights[building_idx]
        if dist <= 0 or not np.isfinite(height):
            hw_values.append(np.nan)
        else:
            hw_values.append(height / (2.0 * dist))

    point_gdf = gpd.GeoDataFrame(
        {"street_id": street_ids, "street_hw": hw_values},
        geometry=points,
        crs=buildings.crs,
    )
    return point_gdf


def _compute_hotspots(grid: gpd.GeoDataFrame, value_col: str) -> gpd.GeoDataFrame:
    try:
        from libpysal.weights import Queen
        from esda.getisord import G_Local
    except Exception as exc:
        raise ImportError("Hotspot analysis requires libpysal and esda") from exc

    grid = grid.copy()
    valid = grid[value_col].notna()
    if valid.sum() == 0:
        grid["hotspot_gi"] = np.nan
        grid["hotspot_p"] = np.nan
        grid["hotspot_class"] = np.nan
        return grid

    weights = Queen.from_dataframe(grid[valid], use_index=True)
    weights.transform = "r"
    gi = G_Local(grid.loc[valid, value_col].values, weights)
    grid.loc[valid, "hotspot_gi"] = gi.Zs
    grid.loc[valid, "hotspot_p"] = gi.p_sim

    # Classification: 1 = hotspot, -1 = coldspot, 0 = not significant
    grid["hotspot_class"] = 0
    grid.loc[valid & (grid["hotspot_p"] <= 0.05) & (grid["hotspot_gi"] > 0), "hotspot_class"] = 1
    grid.loc[valid & (grid["hotspot_p"] <= 0.05) & (grid["hotspot_gi"] < 0), "hotspot_class"] = -1
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Morphology risk visualization")
    parser.add_argument("--area", type=str, required=True, help="Area name (vidigal, riodaspedras)")
    parser.add_argument("--input", type=str, default=None, help="Input morphology GPKG path")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--grid-size", type=float, default=50.0, help="Grid aggregation size (m)")
    parser.add_argument("--hotspots", action="store_true", help="Compute hotspot clusters (Gi*)")
    parser.add_argument("--streets", type=str, default=None, help="Optional streets file path")
    parser.add_argument("--street-spacing", type=float, default=10.0, help="Street sampling spacing (m)")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = get_area_analysis_dir(args.area, "morphology_metrics") / "buildings_with_morphology_metrics.gpkg"
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_base = Path(args.output) if args.output else get_area_analysis_dir(args.area, "morphology_risk")
    output_base.mkdir(parents=True, exist_ok=True)
    maps_dir = output_base / "maps"
    maps_dir.mkdir(exist_ok=True)

    logger.info(f"Loading morphology metrics from {input_path}")
    gdf = gpd.read_file(input_path)

    # Build weights (equal weight)
    weights = {m: 1.0 for m in RISK_METRICS if m in gdf.columns}
    gdf = _compute_risk_index(gdf, weights)

    # Save building-level risk
    buildings_out = output_base / "buildings_with_risk.gpkg"
    gdf.to_file(buildings_out, driver="GPKG")
    logger.info(f"Saved buildings with risk index to {buildings_out}")

    # Grid aggregation
    grid = _grid_aggregate(gdf, args.grid_size)
    grid_out = output_base / "risk_grid.gpkg"
    grid.to_file(grid_out, driver="GPKG")
    logger.info(f"Saved risk grid to {grid_out}")

    # Optional street-level canyon H/W
    street_points = gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    if args.streets:
        streets_path = Path(args.streets)
        if streets_path.exists():
            streets = gpd.read_file(streets_path)
            street_points = _compute_street_canyon_hw(
                streets,
                gdf,
                spacing=args.street_spacing,
            )
            if not street_points.empty:
                street_points.to_file(output_base / "street_canyon_hw_points.gpkg", driver="GPKG")
                street_grid = gpd.sjoin(street_points, grid, predicate="intersects", how="left")
                street_grouped = street_grid.groupby("index_right")["street_hw"].mean()
                grid["street_canyon_hw_mean"] = street_grouped
                _plot_map(
                    street_points,
                    "street_hw",
                    maps_dir / "street_canyon_hw_points.png",
                    "Street Canyon H/W (Sample Points)",
                )
        else:
            logger.warning(f"Streets file not found: {streets_path}")

    # Primary risk maps
    _plot_map(gdf, "risk_index", maps_dir / "risk_index_map.png", "Morphology Risk Index (Building)")
    _plot_map(grid, "risk_mean", maps_dir / "risk_grid_mean.png", "Morphology Risk Index (Grid Mean)")
    _plot_map(grid, "risk_p90", maps_dir / "risk_grid_p90.png", "Morphology Risk Index (Grid 90th)")
    if "street_canyon_hw_mean" in grid.columns:
        # Combine grid risk with street canyon H/W to better reflect ventilation risk
        z_risk = _normalize_series(grid["risk_mean"].to_numpy(dtype=float))
        z_street = _normalize_series(grid["street_canyon_hw_mean"].to_numpy(dtype=float))
        grid["risk_grid_index"] = z_risk + z_street
        _plot_map(
            grid,
            "risk_grid_index",
            maps_dir / "risk_grid_index.png",
            "Morphology Risk Index (Grid + Street Canyon)",
        )
        _plot_map(
            grid,
            "street_canyon_hw_mean",
            maps_dir / "street_canyon_hw_grid.png",
            "Street Canyon H/W (Grid Mean)"
        )

    # Key component maps
    for metric in ["building_adjacency", "shared_walls", "tessellation_area", "shape_index"]:
        if metric in gdf.columns:
            _plot_map(gdf, metric, maps_dir / f"{metric}_risk_map.png", metric.replace("_", " ").title())

    # Bivariate risk visualization
    if "building_adjacency" in gdf.columns and "shared_walls" in gdf.columns:
        _plot_bivariate(
            gdf,
            "building_adjacency",
            "shared_walls",
            maps_dir / "bivariate_adjacency_shared_walls.png",
            "Bivariate: Adjacency vs Shared Walls",
        )

    # Hotspot clusters (Gi*)
    if args.hotspots:
        logger.info("Computing hotspot clusters (Gi*) on grid risk_mean...")
        try:
            grid_for_hotspots = grid.copy()
            if "risk_grid_index" in grid_for_hotspots.columns:
                hotspot_value = "risk_grid_index"
            else:
                hotspot_value = "risk_mean"

            grid_hot = _compute_hotspots(grid_for_hotspots, hotspot_value)
            grid_hot.to_file(output_base / "risk_hotspots.gpkg", driver="GPKG")
            _plot_map(
                grid_hot,
                "hotspot_gi",
                maps_dir / "risk_hotspot_gi_zscore.png",
                "Risk Hotspots (Gi* Z-score)",
                cmap="coolwarm",
                diverging=True,
            )
            _plot_hotspot_class(
                grid_hot,
                maps_dir / "risk_hotspot_class.png",
                "Risk Hotspots (1=hotspot, -1=coldspot)"
            )
        except ImportError as exc:
            logger.warning(str(exc))

    logger.info("Risk visualization complete")


if __name__ == "__main__":
    main()
