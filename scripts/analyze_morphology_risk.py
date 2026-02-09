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
    "hw_ratio",
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
    joined = gpd.sjoin(gdf, grid, predicate="intersects", how="left")
    grouped = joined.groupby("index_right")
    grid["risk_mean"] = grouped["risk_index"].mean()
    grid["risk_p90"] = grouped["risk_index"].quantile(0.9)
    grid["building_count"] = grouped.size()
    return grid


def _plot_map(gdf: gpd.GeoDataFrame, column: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(column=column, ax=ax, cmap="magma", legend=True, legend_kwds={"label": column})
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

    # Primary risk maps
    _plot_map(gdf, "risk_index", maps_dir / "risk_index_map.png", "Morphology Risk Index (Building)")
    _plot_map(grid, "risk_mean", maps_dir / "risk_grid_mean.png", "Morphology Risk Index (Grid Mean)")
    _plot_map(grid, "risk_p90", maps_dir / "risk_grid_p90.png", "Morphology Risk Index (Grid 90th)")

    # Key component maps
    for metric in ["hw_ratio", "building_adjacency", "shared_walls", "tessellation_area", "shape_index"]:
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
            grid_hot = _compute_hotspots(grid, "risk_mean")
            grid_hot.to_file(output_base / "risk_hotspots.gpkg", driver="GPKG")
            _plot_map(
                grid_hot,
                "hotspot_gi",
                maps_dir / "risk_hotspot_gi_zscore.png",
                "Risk Hotspots (Gi* Z-score)"
            )
            _plot_map(
                grid_hot,
                "hotspot_class",
                maps_dir / "risk_hotspot_class.png",
                "Risk Hotspots (1=hotspot, -1=coldspot)"
            )
        except ImportError as exc:
            logger.warning(str(exc))

    logger.info("Risk visualization complete")


if __name__ == "__main__":
    main()
