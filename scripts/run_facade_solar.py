#!/usr/bin/env python3
"""
Compute facade-level solar irradiance and WHO sunlight assessment.

Usage:
    python scripts/run_facade_solar.py --area vidigal --date 2026-06-21
    python scripts/run_facade_solar.py --area rocinha --n-jobs -1
    python scripts/run_facade_solar.py --area maré --all-dates --facade-spacing 1.5
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_output_dir  # noqa: E402
from src.svf_v2.paths import resolve_paths  # noqa: E402
from src.svf_v2.scene import build_scene  # noqa: E402
from src.svf_v2.sampling import sample_facade_points  # noqa: E402
from src.svf_v2.facades import compute_facade_svf  # noqa: E402
from src.solar.facade import (  # noqa: E402
    aggregate_by_building,
    aggregate_by_building_floor,
    compute_facade_solar_access,
)
from src.solar.sun import REFERENCE_DATES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run_facade_solar")


def run_single_date(
    facade_gdf: gpd.GeoDataFrame,
    scene_mesh,
    date: str,
    args,
    output_dir: Path,
) -> gpd.GeoDataFrame:
    """Run facade solar analysis for a single date."""
    logger.info("=" * 60)
    logger.info("  Date: %s", date)
    logger.info("=" * 60)

    result = compute_facade_solar_access(
        facade_gdf,
        scene_mesh,
        date=date,
        hour_start=args.hour_start,
        hour_end=args.hour_end,
        interval_minutes=args.interval,
        n_jobs=args.n_jobs,
        site_elevation_m=args.site_elevation,
        floor_height=args.floor_height,
        who_threshold_hours=args.who_threshold,
    )

    # Save point-level results
    date_tag = date.replace("-", "")
    points_path = output_dir / f"facade_solar_{date_tag}.gpkg"
    result.to_file(points_path, driver="GPKG")
    logger.info("Saved %d facade points to %s", len(result), points_path.name)

    # Building-level aggregation
    bldg_agg = aggregate_by_building(result)
    bldg_path = output_dir / f"buildings_who_{date_tag}.csv"
    bldg_agg.to_csv(bldg_path, index=False)
    logger.info(
        "Building summary: %d buildings, %.1f%% WHO-compliant",
        len(bldg_agg),
        100.0 * bldg_agg["who_compliance_rate"].mean(),
    )

    # Floor-level aggregation
    floor_agg = aggregate_by_building_floor(result)
    floor_path = output_dir / f"floors_who_{date_tag}.csv"
    floor_agg.to_csv(floor_path, index=False)
    logger.info(
        "Floor summary: %d building-floor combinations", len(floor_agg),
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Facade-level solar irradiance and WHO sunlight assessment."
    )
    parser.add_argument("--area", required=True, help="Area name (e.g. vidigal)")
    parser.add_argument(
        "--date", default="2026-06-21",
        help="Analysis date (default: winter solstice for Southern Hemisphere)",
    )
    parser.add_argument(
        "--all-dates", action="store_true",
        help="Run for all 4 reference dates (solstices + equinoxes)",
    )
    parser.add_argument("--interval", type=int, default=30, help="Timestep (minutes)")
    parser.add_argument("--hour-start", type=int, default=6, help="Start hour")
    parser.add_argument("--hour-end", type=int, default=18, help="End hour")
    parser.add_argument(
        "--facade-spacing", type=float, default=1.0,
        help="Facade sample spacing (metres)",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers")
    parser.add_argument(
        "--floor-height", type=float, default=3.0,
        help="Floor-to-floor height (metres)",
    )
    parser.add_argument(
        "--who-threshold", type=float, default=2.0,
        help="WHO minimum sunlight threshold (hours)",
    )
    parser.add_argument(
        "--site-elevation", type=float, default=100.0,
        help="Site elevation (metres)",
    )
    parser.add_argument(
        "--skip-svf", action="store_true",
        help="Skip facade SVF computation (use svf=1.0 for diffuse)",
    )
    parser.add_argument("--height-field", default="altura", help="Building height column")
    parser.add_argument("--base-field", default="base", help="Building base elevation column")
    parser.add_argument(
        "--sky-patches", type=int, default=145,
        help="Sky patches for SVF (default: 145 Tregenza)",
    )
    args = parser.parse_args()

    t0 = time.time()

    # Resolve paths
    dtm_path, footprints_path, roads_path = resolve_paths(args.area)
    output_dir = get_area_output_dir(args.area) / "facade_solar"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Area: %s", args.area)
    logger.info("Output: %s", output_dir)

    # Build scene
    logger.info("Building 3D scene...")
    scene_mesh, buildings_gdf = build_scene(
        dtm_path, footprints_path,
        height_field=args.height_field,
        base_field=args.base_field,
        area=args.area,
    )

    # Sample facade points
    logger.info("Sampling facade points (spacing=%.1fm)...", args.facade_spacing)
    facade_gdf = sample_facade_points(
        buildings_gdf,
        dtm_path,
        horizontal_spacing=args.facade_spacing,
        vertical_spacing=args.facade_spacing,
    )
    logger.info("Generated %d facade points.", len(facade_gdf))

    if len(facade_gdf) == 0:
        logger.error("No facade points generated. Exiting.")
        sys.exit(1)

    # Compute facade SVF (optional)
    if not args.skip_svf:
        logger.info("Computing facade SVF (%d sky patches)...", args.sky_patches)
        facade_gdf = compute_facade_svf(
            facade_gdf, scene_mesh, n_sky_patches=args.sky_patches,
        )
    else:
        facade_gdf["svf"] = 1.0
        logger.info("Skipping SVF (using svf=1.0 for diffuse).")

    # Run solar analysis
    if args.all_dates:
        dates = list(REFERENCE_DATES.values())
        logger.info("Running for %d reference dates: %s", len(dates), dates)
    else:
        dates = [args.date]

    for date in dates:
        run_single_date(facade_gdf, scene_mesh, date, args, output_dir)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Facade solar analysis complete in %.1fs", elapsed)
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
