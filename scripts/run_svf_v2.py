#!/usr/bin/env python3
"""
Unified SVF v2 runner: grid, streets, facades.

Usage:
    python scripts/run_svf_v2.py --area vidigal_tls --mode all
    python scripts/run_svf_v2.py --area vidigal --mode grid --grid-spacing 2.0
    python scripts/run_svf_v2.py --area riodaspedras --mode streets --street-spacing 1.5
"""

import argparse
import logging
import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_output_dir
from src.svf_v2.paths import resolve_paths
from src.svf_v2.scene import build_scene
from src.svf_v2.sampling import sample_grid_points, sample_street_points, sample_facade_points
from src.svf_v2.compute import compute_svf
from src.svf_v2.facades import compute_facade_svf, compute_facade_solar_potential
from src.svf_v2.io import (
    save_grid_results, save_street_results, save_facade_results,
    save_scene_stl, save_scene_vtk,
    plot_alignment_check, plot_road_z_check,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run_svf_v2")


def run_grid(
    scene_mesh, terrain, footprints_gdf, dtm_path, output_dir, args,
):
    logger.info("=" * 60)
    logger.info("GRID SVF")
    logger.info("=" * 60)

    t0 = time.time()
    observers = sample_grid_points(
        dtm_path, footprints_gdf,
        grid_spacing=args.grid_spacing,
        pedestrian_height=args.pedestrian_height,
        buffer_around_buildings=args.buffer,
    )
    logger.info(f"Grid sampling: {len(observers)} points ({time.time()-t0:.1f}s)")

    checkpoint_path = output_dir / ".svf_checkpoint_grid.npz" if args.checkpoint else None
    t0 = time.time()
    svf = compute_svf(
        observers, scene_mesh,
        backend=args.backend,
        n_sky_patches=args.sky_patches,
        max_ray_length=args.max_ray_length,
        n_jobs=args.n_jobs,
        checkpoint_path=checkpoint_path,
    )
    logger.info(f"Grid SVF computed in {time.time()-t0:.1f}s")
    logger.info(f"  SVF stats: mean={svf.mean():.3f}, min={svf.min():.3f}, max={svf.max():.3f}")

    import rasterio
    with rasterio.open(dtm_path) as src:
        crs = src.crs
    save_grid_results(observers, svf, crs, output_dir, args.grid_spacing)


def run_streets(
    scene_mesh, roads_path, dtm_path, output_dir, args,
    footprints_gdf=None,
):
    logger.info("=" * 60)
    logger.info("STREET SVF")
    logger.info("=" * 60)

    t0 = time.time()
    street_gdf = sample_street_points(
        roads_path, dtm_path,
        spacing=args.street_spacing,
        pedestrian_height=args.pedestrian_height,
    )
    logger.info(f"Street sampling: {len(street_gdf)} points ({time.time()-t0:.1f}s)")

    if len(street_gdf) == 0:
        logger.warning("No street points -- skipping")
        return

    observers = np.column_stack([
        np.array([p.x for p in street_gdf.geometry]),
        np.array([p.y for p in street_gdf.geometry]),
        street_gdf["z_observer"].values,
    ])

    checkpoint_path = output_dir / ".svf_checkpoint_streets.npz" if args.checkpoint else None
    t0 = time.time()
    svf = compute_svf(
        observers, scene_mesh,
        backend=args.backend,
        n_sky_patches=args.sky_patches,
        max_ray_length=args.max_ray_length,
        n_jobs=args.n_jobs,
        checkpoint_path=checkpoint_path,
    )
    street_gdf["svf"] = svf
    logger.info(f"Street SVF computed in {time.time()-t0:.1f}s")
    logger.info(f"  SVF stats: mean={svf.mean():.3f}, min={svf.min():.3f}, max={svf.max():.3f}")

    # Load roads for segment aggregation
    import geopandas
    roads_gdf = geopandas.read_file(roads_path)
    import rasterio as rio
    with rio.open(dtm_path) as src:
        if roads_gdf.crs != src.crs:
            roads_gdf = roads_gdf.to_crs(src.crs)

    save_street_results(street_gdf, output_dir, roads_gdf=roads_gdf, footprints_gdf=footprints_gdf)

    # Visual: road Z check
    if not args.skip_visual:
        with rio.open(dtm_path) as src:
            dtm_data = src.read(1)
            valid = dtm_data[np.isfinite(dtm_data) & (dtm_data < 10000)]
            dtm_z_range = (float(valid.min()), float(valid.max())) if len(valid) > 0 else None
        plot_road_z_check(street_gdf, output_dir / "road_z_check.png", dtm_z_range=dtm_z_range)


def run_facades(
    scene_mesh, footprints_gdf, dtm_path, output_dir, args,
):
    logger.info("=" * 60)
    logger.info("FACADE SVF")
    logger.info("=" * 60)

    t0 = time.time()
    facade_gdf = sample_facade_points(
        footprints_gdf, dtm_path,
        height_field=args.height_field,
        base_field=args.base_field,
        vertical_spacing=args.facade_spacing,
        horizontal_spacing=args.facade_spacing,
    )
    logger.info(f"Facade sampling: {len(facade_gdf)} points ({time.time()-t0:.1f}s)")

    if len(facade_gdf) == 0:
        logger.warning("No facade points -- skipping")
        return

    t0 = time.time()
    facade_gdf = compute_facade_svf(
        facade_gdf, scene_mesh,
        n_sky_patches=args.sky_patches,
        max_ray_length=args.max_ray_length,
    )
    logger.info(f"Facade SVF computed in {time.time()-t0:.1f}s")
    logger.info(
        f"  SVF stats: mean={facade_gdf['svf'].mean():.3f}, "
        f"min={facade_gdf['svf'].min():.3f}, max={facade_gdf['svf'].max():.3f}"
    )

    # Solar potential
    facade_gdf = compute_facade_solar_potential(facade_gdf)

    save_facade_results(facade_gdf, output_dir, footprints_gdf=footprints_gdf)


def main():
    parser = argparse.ArgumentParser(description="SVF v2 computation")
    parser.add_argument("--area", required=True, help="Area name (e.g. vidigal_tls)")
    parser.add_argument("--mode", default="all", choices=["grid", "streets", "facades", "all"])
    parser.add_argument("--grid-spacing", type=float, default=2.0)
    parser.add_argument("--street-spacing", type=float, default=1.5)
    parser.add_argument("--facade-spacing", type=float, default=1.0)
    parser.add_argument("--pedestrian-height", type=float, default=1.5)
    parser.add_argument("--sky-patches", type=int, default=145)
    parser.add_argument("--backend", default="raycasting", choices=["raycasting", "gpu", "pyviewfactor"])
    parser.add_argument("--max-ray-length", type=float, default=500.0)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel workers for raycasting (1=sequential, -1=all cores)")
    parser.add_argument("--buffer", type=float, default=None,
                        help="Only compute grid SVF within this distance of buildings (m)")
    parser.add_argument("--dtm-subsample", type=int, default=1)
    parser.add_argument("--base-field", default="base", help="Building base elevation field")
    parser.add_argument("--height-field", default="altura", help="Building extrusion height field")
    parser.add_argument("--skip-visual", action="store_true",
                        help="Skip visual inspection outputs (STL, alignment plots)")
    parser.add_argument("--checkpoint", action="store_true", default=True,
                        help="Enable checkpoint/resume for long raycasting runs (default: enabled)")
    parser.add_argument("--no-checkpoint", action="store_false", dest="checkpoint",
                        help="Disable checkpoint/resume")
    args = parser.parse_args()

    dtm_path, footprints_path, roads_path = resolve_paths(args.area)
    output_dir = get_area_output_dir(args.area) / "svf_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Area: {args.area}")
    logger.info(f"DTM: {dtm_path}")
    logger.info(f"Footprints: {footprints_path}")
    logger.info(f"Roads: {roads_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Backend: {args.backend}")

    # Build scene
    t0 = time.time()
    scene_mesh, terrain, footprints_gdf = build_scene(
        dtm_path, footprints_path,
        height_field=args.height_field,
        base_field=args.base_field,
        area=args.area,
        dtm_subsample=args.dtm_subsample,
    )
    logger.info(f"Scene built in {time.time()-t0:.1f}s")

    # Visual inspection outputs
    if not args.skip_visual:
        save_scene_stl(scene_mesh, output_dir / "scene.stl")
        save_scene_vtk(scene_mesh, output_dir / "scene.vtk")

        # Load roads for alignment check
        import geopandas
        roads_gdf = geopandas.read_file(roads_path)
        import rasterio
        with rasterio.open(dtm_path) as src:
            if roads_gdf.crs != src.crs:
                roads_gdf = roads_gdf.to_crs(src.crs)

        plot_alignment_check(
            footprints_gdf, output_dir / "alignment_check.png",
            roads_gdf=roads_gdf,
            title=f"{args.area} -- Alignment Check",
        )

    modes = [args.mode] if args.mode != "all" else ["grid", "streets", "facades"]

    for mode in modes:
        if mode == "grid":
            run_grid(scene_mesh, terrain, footprints_gdf, dtm_path, output_dir, args)
        elif mode == "streets":
            run_streets(scene_mesh, roads_path, dtm_path, output_dir, args, footprints_gdf=footprints_gdf)
        elif mode == "facades":
            run_facades(scene_mesh, footprints_gdf, dtm_path, output_dir, args)

    logger.info("Done.")


if __name__ == "__main__":
    main()
