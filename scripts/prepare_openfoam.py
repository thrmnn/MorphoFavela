#!/usr/bin/env python3
"""
Prepare OpenFOAM-ready directories for cross-area selected CFD patches.

For each of the 22 selected patches, generates:
  - buildings.stl  -- extruded building mesh
  - terrain.stl    -- terrain surface mesh
  - scene.stl      -- combined buildings + terrain
  - domain.json    -- domain bounds, height stats, mesh recommendations

Reads from per-area patch exports and writes to a unified openfoam/ directory.

Usage:
    python scripts/prepare_openfoam.py
    python scripts/prepare_openfoam.py --output-dir outputs/openfoam
    python scripts/prepare_openfoam.py --patches outputs/comparative/cross_area_clustering/selected_patches.csv
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyvista as pv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_output_dir
from src.svf_v2.scene import build_terrain_mesh, load_dtm, _extrude_polygon
from shapely.geometry import Polygon, MultiPolygon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("prepare_openfoam")


def build_buildings_stl(buildings_path: Path, dtm_path: Path) -> pv.PolyData | None:
    """Extrude buildings from a patch export into a PyVista mesh."""
    gdf = gpd.read_file(buildings_path)
    if gdf.empty:
        return None

    # Repair invalid geometries
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    dtm_array, transform, _, _ = load_dtm(dtm_path)

    meshes = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            if not isinstance(poly, Polygon):
                continue

            # Base elevation from 'base' field or DTM
            base_elev = row.get("base", np.nan)
            if pd.isna(base_elev):
                from src.svf_v2.scene import _sample_base_from_dtm
                base_elev = _sample_base_from_dtm(poly, dtm_array, transform)

            h = row.get("altura", np.nan)
            if pd.isna(h) or h <= 0 or np.isnan(base_elev):
                continue

            mesh = _extrude_polygon(poly, float(base_elev), float(h))
            if mesh is not None:
                meshes.append(mesh)

    if not meshes:
        return None

    return pv.merge(meshes)


def make_domain_config(
    buildings_gdf: gpd.GeoDataFrame,
    terrain_mesh: pv.PolyData | None,
    boundary_gdf: gpd.GeoDataFrame,
    metadata: dict,
) -> dict:
    """Generate domain configuration for OpenFOAM."""
    # Get buffered bounds
    buffered = boundary_gdf[boundary_gdf["type"] == "buffered"]
    if buffered.empty:
        buffered = boundary_gdf
    bounds = buffered.total_bounds  # [minx, miny, maxx, maxy]

    # Building height stats
    heights = buildings_gdf.get("altura", pd.Series(dtype=float))
    heights = heights.dropna()
    h_max = float(heights.max()) if len(heights) > 0 else 10.0
    h_mean = float(heights.mean()) if len(heights) > 0 else 5.0

    # Terrain elevation range
    if terrain_mesh is not None:
        z_min = float(terrain_mesh.bounds[4])
        z_max = float(terrain_mesh.bounds[5])
    else:
        z_min, z_max = 0.0, 10.0

    # Domain top: max(building top, terrain top) + 5 * h_max (standard CFD practice)
    domain_top = z_max + h_max + 5 * h_max

    return {
        "crs": str(buildings_gdf.crs) if buildings_gdf.crs else "EPSG:31983",
        "domain_bounds": {
            "x_min": float(bounds[0]),
            "y_min": float(bounds[1]),
            "x_max": float(bounds[2]),
            "y_max": float(bounds[3]),
            "z_min": float(z_min),
            "z_max": float(domain_top),
        },
        "domain_size_m": {
            "x": float(bounds[2] - bounds[0]),
            "y": float(bounds[3] - bounds[1]),
            "z": float(domain_top - z_min),
        },
        "building_stats": {
            "count": len(buildings_gdf),
            "h_max": round(h_max, 1),
            "h_mean": round(h_mean, 1),
            "h_std": round(float(heights.std()), 1) if len(heights) > 1 else 0.0,
        },
        "terrain": {
            "z_min": round(z_min, 1),
            "z_max": round(z_max, 1),
            "z_range": round(z_max - z_min, 1),
        },
        "patch_metadata": {
            "tile_id": metadata.get("tile_id", ""),
            "area": metadata.get("area", ""),
            "cluster": metadata.get("cluster", ""),
            "selection_reason": metadata.get("selection_reason", ""),
        },
    }


def _export_patch_from_source(area: str, tile_id: str, patch_dir: Path) -> tuple:
    """Export patch data directly from area source files using tile geometry.

    Returns (buildings_path, dtm_path, boundary_path, metadata) or raises.
    """
    from src.svf_v2.paths import resolve_boundary, resolve_paths
    from src.patch_selection.export import _export_dtm

    # Load tile geometry from tiles.gpkg
    tiles_path = get_area_output_dir(area) / "patch_selection" / "tiles.gpkg"
    if not tiles_path.exists():
        raise FileNotFoundError(f"tiles.gpkg not found for {area}")

    tiles = gpd.read_file(tiles_path)
    tile_row = tiles[tiles["tile_id"] == tile_id]
    if tile_row.empty:
        raise ValueError(f"Tile {tile_id} not found in {tiles_path}")
    tile_row = tile_row.iloc[0]
    tile_geom = tile_row.geometry

    # Reconstruct buffered geometry from WKT if available
    buf_wkt = tile_row.get("geometry_buffered_wkt", None)
    if buf_wkt and pd.notna(buf_wkt):
        from shapely import wkt
        buffered_geom = wkt.loads(buf_wkt)
    else:
        buffered_geom = tile_geom.buffer(50)

    # Load source data
    dtm_src, footprints_src, roads_src = resolve_paths(area)

    # Export buildings
    all_buildings = gpd.read_file(footprints_src)
    sindex = all_buildings.sindex
    candidates = list(sindex.intersection(buffered_geom.bounds))
    if candidates:
        nearby = all_buildings.iloc[candidates]
        selected = nearby[nearby.geometry.intersects(buffered_geom)].copy()
    else:
        selected = all_buildings.iloc[0:0].copy()

    buildings_path = patch_dir / "buildings.gpkg"
    selected.to_file(buildings_path, driver="GPKG")
    logger.info("  Exported %d buildings from source", len(selected))

    # Export DTM crop
    dtm_path = patch_dir / "dtm.tif"
    _export_dtm(dtm_src, buffered_geom, dtm_path)

    # Export roads
    all_roads = gpd.read_file(roads_src)
    if all_roads.crs != all_buildings.crs:
        all_roads = all_roads.to_crs(all_buildings.crs)
    clipped_roads = gpd.clip(all_roads, buffered_geom)
    roads_path = patch_dir / "roads.gpkg"
    clipped_roads.to_file(roads_path, driver="GPKG")

    # Export boundary
    boundary_gdf = gpd.GeoDataFrame(
        {"type": ["analysis", "buffered"], "geometry": [tile_geom, buffered_geom]},
        crs=all_buildings.crs,
    )
    boundary_path = patch_dir / "boundary.gpkg"
    boundary_gdf.to_file(boundary_path, driver="GPKG")

    metadata = {"tile_id": tile_id, "area": area, "n_buildings": len(selected)}
    with open(patch_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return buildings_path, dtm_path, boundary_path, metadata


def process_patch(area: str, tile_id: str, output_dir: Path) -> dict | None:
    """Process a single patch: generate STL meshes + domain config."""
    patch_src = get_area_output_dir(area) / "patch_selection" / "exports" / f"patch_{tile_id}"

    patch_dir = output_dir / f"{area}__{tile_id}"
    patch_dir.mkdir(parents=True, exist_ok=True)

    if patch_src.exists() and (patch_src / "buildings.gpkg").exists():
        buildings_path = patch_src / "buildings.gpkg"
        dtm_path = patch_src / "dtm.tif"
        boundary_path = patch_src / "boundary.gpkg"
        metadata_path = patch_src / "metadata.json"

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        metadata["area"] = area

        # Copy source files
        import shutil
        for src_file in ["buildings.gpkg", "roads.gpkg", "boundary.gpkg", "metadata.json"]:
            src_path = patch_src / src_file
            if src_path.exists():
                shutil.copy2(src_path, patch_dir / src_file)
    else:
        # Export from source data
        logger.info("  Exporting from source data (no per-area export)")
        try:
            buildings_path, dtm_path, boundary_path, metadata = _export_patch_from_source(
                area, tile_id, patch_dir
            )
        except Exception as e:
            logger.warning("  Failed to export from source: %s", e)
            return None

    # Load buildings
    buildings_gdf = gpd.read_file(buildings_path)

    # Load boundary
    boundary_gdf = gpd.read_file(boundary_path) if boundary_path.exists() else gpd.GeoDataFrame()

    # Build terrain STL
    terrain_mesh = None
    try:
        if boundary_gdf.empty:
            terrain_mesh = build_terrain_mesh(dtm_path)
        else:
            buffered = boundary_gdf[boundary_gdf["type"] == "buffered"]
            if not buffered.empty:
                bbox = tuple(buffered.total_bounds)
                terrain_mesh = build_terrain_mesh(dtm_path, bbox=bbox)
            else:
                terrain_mesh = build_terrain_mesh(dtm_path)

        if terrain_mesh is not None:
            terrain_mesh.save(str(patch_dir / "terrain.stl"))
            logger.info("  terrain.stl: %d cells", terrain_mesh.n_cells)
    except Exception as e:
        logger.warning("  Terrain mesh failed: %s", e)

    # Build buildings STL
    buildings_mesh = build_buildings_stl(buildings_path, dtm_path)
    if buildings_mesh is not None:
        buildings_mesh.save(str(patch_dir / "buildings.stl"))
        logger.info("  buildings.stl: %d cells", buildings_mesh.n_cells)

    # Combined scene
    if buildings_mesh is not None and terrain_mesh is not None:
        scene = pv.merge([terrain_mesh, buildings_mesh])
        scene.save(str(patch_dir / "scene.stl"))
        logger.info("  scene.stl: %d cells", scene.n_cells)
    elif buildings_mesh is not None:
        buildings_mesh.save(str(patch_dir / "scene.stl"))
    elif terrain_mesh is not None:
        terrain_mesh.save(str(patch_dir / "scene.stl"))

    # Domain config
    domain = make_domain_config(buildings_gdf, terrain_mesh, boundary_gdf, metadata)
    with open(patch_dir / "domain.json", "w") as f:
        json.dump(domain, f, indent=2)

    return domain


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenFOAM-ready directories for selected CFD patches")
    parser.add_argument(
        "--patches",
        default="outputs/comparative/cross_area_clustering/selected_patches.csv",
        help="CSV with selected patches (area, tile_id columns)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/openfoam",
        help="Output directory for OpenFOAM data",
    )
    args = parser.parse_args()

    patches_csv = Path(args.patches)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patches = pd.read_csv(patches_csv)
    logger.info("Processing %d selected patches from %s", len(patches), patches_csv)

    results = []
    t0 = time.time()

    for _, row in patches.iterrows():
        area = row["area"]
        tile_id = row["tile_id"]
        logger.info("=" * 60)
        logger.info("Patch: %s / %s (cluster %s, %s)", area, tile_id, row.get("cluster", "?"), row.get("selection_reason", ""))

        domain = process_patch(area, tile_id, output_dir)
        if domain is not None:
            results.append(domain)

    # Write summary manifest
    manifest = {
        "n_patches": len(results),
        "total_buildings": sum(r["building_stats"]["count"] for r in results),
        "areas": sorted(set(r["patch_metadata"]["area"] for r in results)),
        "patches": results,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("OpenFOAM prep complete: %d patches in %.1fs", len(results), time.time() - t0)
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
