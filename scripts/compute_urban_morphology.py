#!/usr/bin/env python3
"""Compute zone-level urban morphology metrics and spatial analysis.

Usage:
    python scripts/compute_urban_morphology.py --area vidigal_tls --cell-size 50
    python scripts/compute_urban_morphology.py --area copacabana --cell-size 100 --floor-height 3.5
"""

import argparse
import glob
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_data_dir, get_area_output_dir
from src.metrics import calculate_basic_metrics
from src.spatial_analysis import compute_global_morans_i, compute_lisa
from src.urban_morphology import compute_zone_metrics, create_analysis_zones
from src.visualization.morphology import plot_lisa_clusters_panel, plot_zone_metrics_panel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("compute_urban_morphology")

# Key columns for spatial autocorrelation analysis
MORAN_COLUMNS = ["bcr", "far", "sigma_h"]


def _find_shapefile(data_dir: Path, pattern: str) -> Path | None:
    """Glob for a shapefile matching *pattern* inside *data_dir*."""
    matches = sorted(glob.glob(str(data_dir / pattern)))
    if matches:
        return Path(matches[0])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute zone-level urban morphology metrics and spatial analysis."
    )
    parser.add_argument("--area", required=True, help="Area name (e.g. vidigal_tls)")
    parser.add_argument(
        "--cell-size", type=float, default=50.0, help="Grid cell size in metres"
    )
    parser.add_argument(
        "--floor-height",
        type=float,
        default=3.0,
        help="Assumed floor-to-floor height in metres",
    )
    parser.add_argument(
        "--wind-dir",
        type=float,
        default=0.0,
        help="Wind bearing in degrees from north for frontal area ratio",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=36,
        help="Number of orientation bins for street entropy",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip visualization output",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ paths
    data_dir = get_area_data_dir(args.area)
    output_dir = get_area_output_dir(args.area) / "urban_morphology"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Area: %s", args.area)
    logger.info("Data dir: %s", data_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info(
        "Parameters: cell_size=%.1f  floor_height=%.1f  wind_dir=%.1f  n_bins=%d",
        args.cell_size,
        args.floor_height,
        args.wind_dir,
        args.n_bins,
    )

    # -------------------------------------------------------------- buildings
    buildings_path = _find_shapefile(data_dir, "*buildings*.shp")
    if buildings_path is None:
        buildings_path = _find_shapefile(data_dir, "*Buildings*.shp")
    if buildings_path is None:
        logger.error(
            "No buildings shapefile found in %s matching '*buildings*.shp'.", data_dir
        )
        sys.exit(1)

    logger.info("Loading buildings from %s", buildings_path)
    buildings = gpd.read_file(buildings_path)
    logger.info("Loaded %d buildings.", len(buildings))

    # Derive height, area, volume etc. from raw base+altura columns
    buildings = calculate_basic_metrics(buildings)
    logger.info("After height preprocessing: %d valid buildings.", len(buildings))

    # --------------------------------------------------------------- streets
    streets_path = _find_shapefile(data_dir, "*road*.shp")
    if streets_path is None:
        streets_path = _find_shapefile(data_dir, "*Road*.shp")

    streets: gpd.GeoDataFrame | None = None
    if streets_path is not None:
        logger.info("Loading streets from %s", streets_path)
        streets = gpd.read_file(streets_path)
        # Reproject streets to match buildings CRS if needed
        if streets.crs != buildings.crs:
            streets = streets.to_crs(buildings.crs)
        logger.info("Loaded %d street segments.", len(streets))
    else:
        logger.warning(
            "No road shapefile found in %s matching '*road*.shp'. "
            "Street orientation entropy will be NaN.",
            data_dir,
        )

    # --------------------------------------------------------- analysis zones
    t0 = time.time()
    zones = create_analysis_zones(buildings, method="grid", cell_size=args.cell_size)
    logger.info("Analysis zones created in %.1fs.", time.time() - t0)

    # ---------------------------------------------------------- zone metrics
    t0 = time.time()
    zone_metrics = compute_zone_metrics(
        buildings,
        zones,
        streets=streets,
        floor_height=args.floor_height,
        wind_dir=args.wind_dir,
        n_bins=args.n_bins,
    )
    logger.info("Zone metrics computed in %.1fs.", time.time() - t0)

    # Save zone metrics
    metrics_path = output_dir / "zone_metrics.gpkg"
    zone_metrics.to_file(metrics_path, driver="GPKG")
    logger.info("Saved zone metrics to %s", metrics_path)

    # ----------------------------------------- spatial autocorrelation: Moran
    moran_rows: list[dict] = []
    for col in MORAN_COLUMNS:
        if col not in zone_metrics.columns:
            logger.warning("Column '%s' not found in zone metrics -- skipping.", col)
            continue

        # Skip columns that are entirely NaN
        if zone_metrics[col].dropna().empty:
            logger.warning("Column '%s' is all NaN -- skipping Moran's I.", col)
            continue

        logger.info("Computing Global Moran's I for '%s'...", col)
        try:
            result = compute_global_morans_i(zone_metrics, col)
            result["column"] = col
            moran_rows.append(result)
        except Exception:
            logger.exception("Failed to compute Moran's I for '%s'.", col)

    if moran_rows:
        moran_df = pd.DataFrame(moran_rows)
        moran_path = output_dir / "moran_summary.csv"
        moran_df.to_csv(moran_path, index=False)
        logger.info("Saved Moran summary to %s", moran_path)
    else:
        logger.warning("No Moran's I results to save.")

    # ---------------------------------------------------- LISA cluster maps
    lisa_frames: list[gpd.GeoDataFrame] = []
    for col in MORAN_COLUMNS:
        if col not in zone_metrics.columns:
            continue
        if zone_metrics[col].dropna().empty:
            continue

        logger.info("Computing LISA clusters for '%s'...", col)
        try:
            lisa_gdf = compute_lisa(zone_metrics, col)
            # Rename LISA columns to include the source metric
            rename_map = {
                "lisa_I": f"lisa_I_{col}",
                "lisa_p": f"lisa_p_{col}",
                "lisa_q": f"lisa_q_{col}",
                "lisa_cluster": f"lisa_cluster_{col}",
            }
            lisa_gdf = lisa_gdf.rename(columns=rename_map)
            lisa_frames.append(lisa_gdf)
        except Exception:
            logger.exception("Failed to compute LISA for '%s'.", col)

    if lisa_frames:
        # Merge all LISA results into a single GeoDataFrame
        lisa_combined = lisa_frames[0]
        for frame in lisa_frames[1:]:
            lisa_cols = [c for c in frame.columns if c.startswith("lisa_")]
            lisa_combined = lisa_combined.join(
                frame[lisa_cols], rsuffix="_dup", how="left"
            )
        lisa_path = output_dir / "lisa_clusters.gpkg"
        lisa_combined.to_file(lisa_path, driver="GPKG")
        logger.info("Saved LISA clusters to %s", lisa_path)
    else:
        logger.warning("No LISA results to save.")

    # -------------------------------------------------------- visualizations
    if not args.skip_plots:
        maps_dir = output_dir / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating zone metrics panel...")
        plot_zone_metrics_panel(zone_metrics, maps_dir / "zone_metrics_panel.png")

        if lisa_frames:
            logger.info("Generating LISA cluster maps...")
            plot_lisa_clusters_panel(lisa_combined, maps_dir)
    else:
        logger.info("Skipping plots (--skip-plots).")

    logger.info("Done.")


if __name__ == "__main__":
    main()
