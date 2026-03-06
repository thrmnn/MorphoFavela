#!/usr/bin/env python3
"""
Generate STL mesh from building footprints and DTM.

This script creates a 3D STL mesh by:
1. Creating a terrain mesh from a DTM raster
2. Extruding building footprints vertically based on base elevation and height
3. Combining terrain and building meshes into a single STL file

Usage:
    python scripts/generate_stl_from_buildings.py \
        --dtm data/vidigal/raw/DTM_Vidigal.tif \
        --footprints data/vidigal/raw/Vidigal_buildings.shp \
        --output data/vidigal/raw/full_scan.stl \
        --base-field base \
        --height-field altura
"""

import numpy as np
import pyvista as pv
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.transform import xy
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
from shapely.geometry import Polygon, Point, MultiPolygon
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dtm(dtm_path: Path):
    """
    Load DTM raster and return array, transform, and CRS.
    
    Args:
        dtm_path: Path to DTM raster file
        
    Returns:
        Tuple of (dtm_array, transform, crs)
    """
    logger.info(f"Loading DTM from {dtm_path}...")
    with rasterio.open(dtm_path) as src:
        dtm_array = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        
    # Filter out invalid values (NaN, inf, and unreasonably large values)
    valid_mask = np.isfinite(dtm_array) & (dtm_array < 10000)  # Reasonable max elevation
    dtm_array = np.where(valid_mask, dtm_array, np.nan)
    
    logger.info(f"  DTM shape: {dtm_array.shape}")
    logger.info(f"  DTM bounds: {bounds}")
    logger.info(f"  DTM CRS: {crs}")
    valid_elevations = dtm_array[np.isfinite(dtm_array)]
    if len(valid_elevations) > 0:
        logger.info(f"  Elevation range: {np.nanmin(valid_elevations):.2f} - {np.nanmax(valid_elevations):.2f} m")
    else:
        logger.warning("  No valid elevation values found in DTM!")
    
    return dtm_array, transform, crs, bounds


def create_terrain_mesh(dtm_array: np.ndarray, transform: rasterio.Affine, 
                        bounds: tuple, subsample: int = 1) -> pv.PolyData:
    """
    Create a terrain mesh from DTM raster.
    
    Args:
        dtm_array: 2D array of elevation values
        transform: Rasterio transform object
        bounds: Bounding box (minx, miny, maxx, maxy)
        subsample: Subsampling factor (1 = use all pixels, 2 = every 2nd pixel, etc.)
        
    Returns:
        PyVista PolyData mesh of terrain surface
    """
    logger.info("Creating terrain mesh from DTM...")
    
    # Subsample if requested
    if subsample > 1:
        dtm_array = dtm_array[::subsample, ::subsample]
        # Adjust transform for subsampling
        transform = rasterio.Affine(
            transform.a * subsample, transform.b, transform.c,
            transform.d, transform.e * subsample, transform.f
        )
    
    height, width = dtm_array.shape
    
    # Create coordinate grids using transform
    # Use pixel coordinates and transform to world coordinates
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Transform pixel coordinates to world coordinates
    x_coords = np.zeros_like(rows, dtype=float)
    y_coords = np.zeros_like(rows, dtype=float)
    
    for i in range(height):
        for j in range(width):
            x, y = xy(transform, rows[i, j], cols[i, j])
            x_coords[i, j] = x
            y_coords[i, j] = y
    
    # Replace NaN values with a reasonable default (e.g., minimum valid elevation)
    valid_elevations = dtm_array[np.isfinite(dtm_array)]
    if len(valid_elevations) > 0:
        default_elevation = np.nanmin(valid_elevations)
    else:
        default_elevation = 0.0
    z_coords = np.where(np.isfinite(dtm_array), dtm_array, default_elevation)
    
    # Create structured grid
    terrain_mesh = pv.StructuredGrid(x_coords, y_coords, z_coords)
    
    logger.info(f"  Terrain mesh: {terrain_mesh.n_points} points, {terrain_mesh.n_cells} cells")
    
    return terrain_mesh


def sample_dtm_at_point(dtm_array: np.ndarray, transform: rasterio.Affine,
                        x: float, y: float) -> float:
    """
    Sample DTM elevation at a specific point using bilinear interpolation.
    
    Args:
        dtm_array: DTM elevation array
        transform: Rasterio transform
        x, y: Point coordinates
        
    Returns:
        Elevation value at point
    """
    # Convert world coordinates to pixel coordinates
    row, col = rasterio.transform.rowcol(transform, x, y)
    
    # Check bounds
    if row < 0 or row >= dtm_array.shape[0] or col < 0 or col >= dtm_array.shape[1]:
        return np.nan
    
    # Simple nearest neighbor (can be improved with bilinear interpolation)
    row = int(np.clip(row, 0, dtm_array.shape[0] - 1))
    col = int(np.clip(col, 0, dtm_array.shape[1] - 1))
    
    return dtm_array[row, col]


def extrude_building(footprint: Polygon, base_elev: float, height: float) -> pv.PolyData:
    """
    Extrude a building footprint vertically to create a 3D mesh.
    
    Args:
        footprint: Shapely polygon of building footprint
        base_elev: Base elevation (Z coordinate for bottom)
        height: Building height (extrusion distance)
        
    Returns:
        PyVista PolyData mesh of extruded building
    """
    if height <= 0 or np.isnan(height):
        return None
    
    # Get exterior coordinates (remove duplicate last point)
    coords = np.array(footprint.exterior.coords)
    if len(coords) < 4:  # Need at least 3 points + duplicate
        return None
    
    x_coords = coords[:-1, 0]  # Exclude duplicate last point
    y_coords = coords[:-1, 1]
    n_points = len(x_coords)
    
    if n_points < 3:
        return None
    
    # Create bottom face vertices
    bottom_vertices = np.column_stack([
        x_coords,
        y_coords,
        np.full(n_points, base_elev)
    ])
    
    # Create top face vertices
    top_vertices = np.column_stack([
        x_coords,
        y_coords,
        np.full(n_points, base_elev + height)
    ])
    
    # Combine all vertices: bottom first, then top
    all_vertices = np.vstack([bottom_vertices, top_vertices])
    
    # Create faces array in PyVista format
    # Format: [n, v0, v1, ..., vn-1] for each face
    faces_list = []
    
    # Bottom face (reverse order for correct normal pointing down)
    bottom_face = list(range(n_points))
    bottom_face.reverse()
    faces_list.append([n_points] + bottom_face)
    
    # Top face (normal order for correct normal pointing up)
    top_face = list(range(n_points, 2 * n_points))
    faces_list.append([n_points] + top_face)
    
    # Side faces (quads split into two triangles each)
    for i in range(n_points):
        next_i = (i + 1) % n_points
        # Bottom vertex indices
        b0 = i
        b1 = next_i
        # Top vertex indices
        t0 = i + n_points
        t1 = next_i + n_points
        
        # Quad split into two triangles:
        # Triangle 1: b0, b1, t0
        faces_list.append([3, b0, b1, t0])
        # Triangle 2: b1, t1, t0
        faces_list.append([3, b1, t1, t0])
    
    # Flatten faces array
    faces_flat = []
    for face in faces_list:
        faces_flat.extend(face)
    faces_flat = np.array(faces_flat, dtype=np.int32)
    
    # Create mesh
    building_mesh = pv.PolyData(all_vertices, faces_flat)
    
    return building_mesh


def create_stl_from_buildings_and_dtm(
    dtm_path: Path,
    footprints_path: Path,
    output_stl_path: Path,
    base_field: str = "base",
    height_field: str = "altura",
    dtm_subsample: int = 1,
    use_dtm_for_base: bool = False
):
    """
    Create STL mesh from DTM and building footprints.
    
    Args:
        dtm_path: Path to DTM raster
        footprints_path: Path to building footprints shapefile
        output_stl_path: Path to output STL file
        base_field: Name of field containing base elevation
        height_field: Name of field containing building height
        dtm_subsample: Subsampling factor for DTM (1 = full resolution)
        use_dtm_for_base: If True, sample base elevation from DTM instead of using base_field
    """
    logger.info("=" * 60)
    logger.info("STL GENERATION FROM BUILDINGS AND DTM")
    logger.info("=" * 60)
    
    # Load DTM
    dtm_array, transform, dtm_crs, bounds = load_dtm(dtm_path)
    
    # Load footprints
    logger.info(f"Loading building footprints from {footprints_path}...")
    footprints_gdf = gpd.read_file(footprints_path)
    logger.info(f"  Loaded {len(footprints_gdf)} buildings")
    logger.info(f"  Footprints CRS: {footprints_gdf.crs}")
    
    # Check if fields exist
    if base_field not in footprints_gdf.columns and not use_dtm_for_base:
        raise ValueError(f"Base field '{base_field}' not found in footprints. Available fields: {list(footprints_gdf.columns)}")
    if height_field not in footprints_gdf.columns:
        raise ValueError(f"Height field '{height_field}' not found in footprints. Available fields: {list(footprints_gdf.columns)}")
    
    # Reproject footprints to match DTM CRS if needed
    if footprints_gdf.crs != dtm_crs:
        logger.info(f"  Reprojecting footprints from {footprints_gdf.crs} to {dtm_crs}...")
        footprints_gdf = footprints_gdf.to_crs(dtm_crs)
    
    # Create terrain mesh
    terrain_mesh = create_terrain_mesh(dtm_array, transform, bounds, subsample=dtm_subsample)
    
    # Extract building meshes
    logger.info("Extruding buildings...")
    building_meshes = []
    skipped_count = 0
    
    for idx, row in tqdm(footprints_gdf.iterrows(), total=len(footprints_gdf), desc="Processing buildings"):
        footprint = row.geometry
        
        # Skip invalid geometries
        if footprint is None or footprint.is_empty:
            skipped_count += 1
            continue
        
        # Handle MultiPolygon by taking the largest polygon
        if isinstance(footprint, MultiPolygon):
            # Get the largest polygon from the multipolygon
            polygons = list(footprint.geoms)
            footprint = max(polygons, key=lambda p: p.area)
        
        # Skip if still not a Polygon
        if not isinstance(footprint, Polygon):
            skipped_count += 1
            continue
        
        # Get base elevation
        if use_dtm_for_base:
            # Sample DTM at building centroid
            centroid = footprint.centroid
            base_elev = sample_dtm_at_point(dtm_array, transform, centroid.x, centroid.y)
            if np.isnan(base_elev):
                skipped_count += 1
                continue
        else:
            base_elev = row[base_field]
            if pd.isna(base_elev):
                # Fallback to DTM sampling
                centroid = footprint.centroid
                base_elev = sample_dtm_at_point(dtm_array, transform, centroid.x, centroid.y)
                if np.isnan(base_elev):
                    skipped_count += 1
                    continue
        
        # Get height
        height = row[height_field]
        if pd.isna(height) or height <= 0:
            skipped_count += 1
            continue
        
        # Extrude building
        building_mesh = extrude_building(footprint, base_elev, height)
        if building_mesh is not None:
            building_meshes.append(building_mesh)
    
    logger.info(f"  Extruded {len(building_meshes)} buildings")
    if skipped_count > 0:
        logger.info(f"  Skipped {skipped_count} buildings (invalid geometry or missing data)")
    
    # Combine building meshes
    if building_meshes:
        logger.info("Combining building meshes...")
        combined_buildings = building_meshes[0]
        for mesh in tqdm(building_meshes[1:], desc="Merging buildings"):
            combined_buildings = combined_buildings + mesh
        logger.info(f"  Combined building mesh: {combined_buildings.n_points} points, {combined_buildings.n_cells} cells")
    else:
        logger.warning("No valid buildings found!")
        combined_buildings = None
    
    # Combine terrain and buildings
    logger.info("Combining terrain and building meshes...")
    if combined_buildings is not None:
        # Convert terrain to PolyData if it's a StructuredGrid
        if isinstance(terrain_mesh, pv.StructuredGrid):
            terrain_mesh = terrain_mesh.extract_surface()
        full_mesh = terrain_mesh + combined_buildings
    else:
        # Convert terrain to PolyData if it's a StructuredGrid
        if isinstance(terrain_mesh, pv.StructuredGrid):
            full_mesh = terrain_mesh.extract_surface()
        else:
            full_mesh = terrain_mesh
    
    logger.info(f"  Final mesh: {full_mesh.n_points} points, {full_mesh.n_cells} cells")
    
    # Save STL
    logger.info(f"Saving STL to {output_stl_path}...")
    output_stl_path.parent.mkdir(parents=True, exist_ok=True)
    full_mesh.save(str(output_stl_path), binary=True)
    
    logger.info("=" * 60)
    logger.info("STL GENERATION COMPLETE")
    logger.info(f"Output: {output_stl_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate STL mesh from building footprints and DTM"
    )
    parser.add_argument('--dtm', type=str, required=True,
                       help='Path to DTM raster file')
    parser.add_argument('--footprints', type=str, required=True,
                       help='Path to building footprints shapefile')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output STL file')
    parser.add_argument('--base-field', type=str, default='base',
                       help='Name of field containing base elevation (default: base)')
    parser.add_argument('--height-field', type=str, default='altura',
                       help='Name of field containing building height (default: altura)')
    parser.add_argument('--dtm-subsample', type=int, default=1,
                       help='Subsampling factor for DTM (1=full resolution, 2=half, etc.)')
    parser.add_argument('--use-dtm-for-base', action='store_true',
                       help='Sample base elevation from DTM instead of using base field')
    parser.add_argument('--area', type=str, default=None,
                       help='Area name for automatic path resolution')
    
    args = parser.parse_args()
    
    # Handle area-based paths
    if args.area:
        from src.config import get_area_data_dir
        data_dir = get_area_data_dir(args.area)
        # If path is already absolute or contains the area name, use as-is
        # Otherwise, resolve relative to data directory
        dtm_path = Path(args.dtm)
        if not dtm_path.is_absolute() and args.area not in str(dtm_path):
            dtm_path = data_dir / args.dtm
        else:
            dtm_path = Path(args.dtm)
            
        footprints_path = Path(args.footprints)
        if not footprints_path.is_absolute() and args.area not in str(footprints_path):
            footprints_path = data_dir / args.footprints
        else:
            footprints_path = Path(args.footprints)
            
        output_path = Path(args.output)
        if not output_path.is_absolute() and args.area not in str(output_path):
            output_path = data_dir / args.output
        else:
            output_path = Path(args.output)
    else:
        dtm_path = Path(args.dtm)
        footprints_path = Path(args.footprints)
        output_path = Path(args.output)
    
    create_stl_from_buildings_and_dtm(
        dtm_path=dtm_path,
        footprints_path=footprints_path,
        output_stl_path=output_path,
        base_field=args.base_field,
        height_field=args.height_field,
        dtm_subsample=args.dtm_subsample,
        use_dtm_for_base=args.use_dtm_for_base
    )


if __name__ == "__main__":
    main()
