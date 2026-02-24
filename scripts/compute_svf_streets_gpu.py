#!/usr/bin/env python3
"""
Compute Sky View Factor (SVF) along street centerlines using GPU acceleration.

This script is a GPU-accelerated version of compute_svf_streets.py, using PyTorch3D
for faster computation. Falls back to CPU if GPU is unavailable.

Usage:
    python scripts/compute_svf_streets_gpu.py \
        --stl data/riodaspedras/raw/full_scan.stl \
        --roads data/riodaspedras/raw/roads_riodaspedras.shp \
        --spacing 3.0 \
        --height 1.5 \
        --use-gpu
"""

import numpy as np
import pyvista as pv
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import sys
import logging
from shapely.geometry import Point, LineString
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared utilities
from src.svf_utils import (
    load_mesh,
    extract_terrain_surface
)
from scripts.compute_svf import generate_sky_patches
from src.svf_gpu_utils import (
    pv_mesh_to_pytorch3d,
    prepare_observer_points,
    prepare_sky_patches,
    check_gpu_availability
)
from src.svf_gpu_compute import compute_svf_gpu

# Import CPU version as fallback
from scripts.compute_svf_streets import (
    compute_svf as compute_svf_cpu,
    sample_street_points,
    aggregate_segment_statistics,
    plot_street_debug,
    create_street_svf_map,
    create_statistics_plots
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(
        description='Compute Sky View Factor along street centerlines (GPU-accelerated)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--stl', type=str, required=True, help='Path to STL file')
    parser.add_argument('--roads', type=str, required=True, help='Path to road network shapefile')
    parser.add_argument('--footprints', type=str, default=None, help='Path to building footprints (optional, for visualization)')
    parser.add_argument('--dtm', type=str, default=None, help='Path to DTM raster (optional, preferred for elevation)')
    parser.add_argument('--spacing', type=float, default=3.0, help='Distance between sample points (meters, default: 3.0)')
    parser.add_argument('--height', type=float, default=1.5, help='Evaluation height above ground (meters, default: 1.5)')
    parser.add_argument('--sky-patches', type=int, default=145, help='Number of sky patches (default: 145)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: outputs/svf_streets_gpu)')
    parser.add_argument('--area', type=str, default=None, help='Area name (e.g., riodaspedras) for automatic path resolution')
    parser.add_argument('--building-buffer', type=float, default=30.0, help='Buffer distance from buildings to filter streets (meters, default: 30.0)')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU acceleration (requires CUDA-capable GPU)')
    parser.add_argument('--gpu-batch-size', type=int, default=100, help='Batch size for GPU processing (default: 100)')
    parser.add_argument('--gpu-samples-per-ray', type=int, default=50, help='Number of samples per ray for GPU computation (default: 50)')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU computation even if GPU is available')
    
    args = parser.parse_args()
    
    # Check GPU availability
    use_gpu = args.use_gpu and not args.force_cpu
    if use_gpu:
        gpu_info = check_gpu_availability()
        if not gpu_info['available']:
            logger.warning("GPU requested but not available. Falling back to CPU.")
            use_gpu = False
        else:
            logger.info(f"GPU available: {gpu_info['device_name']}")
            logger.info(f"  CUDA version: {gpu_info['cuda_version']}")
    
    # Setup paths
    stl_path = Path(args.stl)
    if not stl_path.exists():
        logger.error(f"STL file not found: {stl_path}")
        sys.exit(1)
    
    roads_path = Path(args.roads)
    if not roads_path.exists():
        logger.error(f"Road network file not found: {roads_path}")
        sys.exit(1)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.area:
        from src.config import get_area_output_dir
        output_dir = get_area_output_dir(args.area) / "svf_streets_gpu"
    else:
        output_dir = PROJECT_ROOT / "outputs" / "svf_streets_gpu"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STREET-LEVEL SKY VIEW FACTOR (SVF) COMPUTATION")
    if use_gpu:
        print("GPU-ACCELERATED VERSION")
    else:
        print("CPU VERSION")
    print("=" * 60)
    print(f"STL file: {stl_path}")
    print(f"Road network: {roads_path}")
    print(f"Point spacing: {args.spacing}m")
    print(f"Evaluation height: {args.height}m (pedestrian eye level)")
    print(f"Sky patches: {args.sky_patches}")
    print(f"Computation: {'GPU' if use_gpu else 'CPU'}")
    if args.dtm:
        print(f"DTM raster: {args.dtm}")
    if args.footprints:
        print(f"Building footprints: {args.footprints}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load mesh and terrain
    logger.info("Loading mesh...")
    mesh = load_mesh(stl_path)
    terrain = extract_terrain_surface(mesh)
    
    # Load road network
    logger.info(f"Loading road network from {roads_path}...")
    roads_gdf = gpd.read_file(roads_path)
    logger.info(f"  Loaded {len(roads_gdf)} road segments")
    
    # Handle coordinate system transformation (same as CPU version)
    terrain_bounds = terrain.bounds
    stl_center_x = (terrain_bounds[0] + terrain_bounds[1]) / 2
    stl_center_y = (terrain_bounds[2] + terrain_bounds[3]) / 2
    
    if args.footprints:
        footprints_path = Path(args.footprints)
        if footprints_path.exists():
            original_footprints = gpd.read_file(str(footprints_path))
            footprints_center_x = (original_footprints.total_bounds[0] + original_footprints.total_bounds[2]) / 2
            footprints_center_y = (original_footprints.total_bounds[1] + original_footprints.total_bounds[3]) / 2
            dx = stl_center_x - footprints_center_x
            dy = stl_center_y - footprints_center_y
        else:
            roads_center_x = (roads_gdf.total_bounds[0] + roads_gdf.total_bounds[2]) / 2
            roads_center_y = (roads_gdf.total_bounds[1] + roads_gdf.total_bounds[3]) / 2
            dx = stl_center_x - roads_center_x
            dy = stl_center_y - roads_center_y
    else:
        roads_center_x = (roads_gdf.total_bounds[0] + roads_gdf.total_bounds[2]) / 2
        roads_center_y = (roads_gdf.total_bounds[1] + roads_gdf.total_bounds[3]) / 2
        dx = stl_center_x - roads_center_x
        dy = stl_center_y - roads_center_y
    
    # Apply transformation to roads
    logger.info(f"  Transforming roads to match STL coordinate system")
    logger.info(f"  Applying translation: dx={dx:.1f}, dy={dy:.1f}")
    roads_gdf.geometry = roads_gdf.geometry.translate(xoff=dx, yoff=dy)
    
    # Load and filter building footprints (same as CPU version)
    building_footprints = None
    isolated_buildings = None
    area_filtered_buildings = None
    building_buffer_geom = None
    if args.footprints:
        footprints_path = Path(args.footprints)
        if footprints_path.exists():
            from src.svf_utils import load_building_footprints
            building_footprints, isolated_buildings, area_filtered_buildings = load_building_footprints(
                footprints_path,
                terrain_bounds=terrain.bounds,
                buffer_distance=0.0,
                area=args.area
            )
            
            if building_footprints is not None and len(building_footprints) > 0 and args.building_buffer > 0:
                from shapely.ops import unary_union
                geom_series = building_footprints.geometry.copy()
                invalid_mask = ~geom_series.is_valid
                if invalid_mask.any():
                    n_invalid = int(invalid_mask.sum())
                    logger.warning(f"  Found {n_invalid} invalid building geometries; attempting repair")
                    geom_series.loc[invalid_mask] = geom_series.loc[invalid_mask].buffer(0)
                    geom_series = geom_series[~geom_series.is_empty]
                    geom_series = geom_series[geom_series.is_valid]
                
                building_union = unary_union(geom_series.values)
                building_buffer_geom = building_union.buffer(args.building_buffer)
                logger.info(f"  Created building buffer zone ({args.building_buffer}m) for street filtering")
    
    # Filter streets (same as CPU version)
    original_roads_count = len(roads_gdf)
    filtered_roads_gdf = roads_gdf.copy()
    if building_buffer_geom is not None and args.building_buffer > 0:
        street_mask = np.zeros(len(roads_gdf), dtype=bool)
        min_percentage_within = 0.7
        
        logger.info(f"  Checking street segments for buffer inclusion...")
        for idx, row in tqdm(roads_gdf.iterrows(), total=len(roads_gdf), desc="  Filtering streets"):
            line = row.geometry
            line_length = line.length
            num_samples = max(10, int(line_length / 5.0))
            sample_distances = np.linspace(0, line_length, num_samples)
            
            points_within = 0
            for dist in sample_distances:
                point = line.interpolate(dist)
                if building_buffer_geom.contains(point):
                    points_within += 1
            
            percentage_within = points_within / num_samples
            if percentage_within >= min_percentage_within:
                street_mask[idx] = True
        
        filtered_roads_gdf = roads_gdf[street_mask].copy()
        n_filtered = original_roads_count - len(filtered_roads_gdf)
        if n_filtered > 0:
            logger.info(f"  Filtered out {n_filtered} street segments")
    
    # Sample points along filtered streets
    dtm_path = Path(args.dtm) if args.dtm else None
    points_gdf = sample_street_points(
        filtered_roads_gdf, args.spacing, dtm_path=dtm_path, terrain=terrain
    )
    
    # Remove points with invalid elevation
    valid_mask = ~points_gdf.geometry.apply(lambda p: np.isnan(p.z))
    if not valid_mask.all():
        logger.warning(f"  Removed {np.sum(~valid_mask)} points with invalid elevation")
        points_gdf = points_gdf[valid_mask].copy()
    
    # Create debug plot
    debug_plot_path = output_dir / "street_debug.png"
    plot_street_debug(
        roads_gdf, points_gdf,
        building_footprints=building_footprints,
        isolated_buildings=isolated_buildings,
        area_filtered_buildings=area_filtered_buildings,
        output_path=debug_plot_path,
        spacing=args.spacing,
        height=args.height,
        building_buffer=args.building_buffer if args.building_buffer > 0 else None,
        building_buffer_geom=building_buffer_geom,
        filtered_roads_gdf=filtered_roads_gdf,
        original_roads_count=original_roads_count
    )
    
    # Convert points to numpy array for SVF computation
    street_points_3d = np.array([
        [geom.x, geom.y, geom.z] for geom in points_gdf.geometry
    ])
    
    logger.info(f"Computing SVF for {len(street_points_3d)} street points...")
    
    # Generate sky patches
    sky_patches, _ = generate_sky_patches(args.sky_patches)
    
    # Compute SVF using GPU or CPU
    if use_gpu:
        try:
            # Convert mesh to PyTorch3D format
            logger.info("Converting mesh to PyTorch3D format...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=device)
            
            # Prepare data for GPU
            observer_points_torch = prepare_observer_points(
                street_points_3d, args.height, device=device
            )
            sky_patches_torch = prepare_sky_patches(sky_patches, device=device)
            
            # Compute SVF on GPU
            logger.info("Computing SVF on GPU...")
            svf_values = compute_svf_gpu(
                observer_points_torch,
                sky_patches_torch,
                pytorch3d_mesh,
                batch_size=args.gpu_batch_size,
                num_samples_per_ray=args.gpu_samples_per_ray
            )
            
            # Convert back to numpy
            svf_values = svf_values.cpu().numpy()
            
        except Exception as e:
            logger.error(f"GPU computation failed: {e}")
            logger.info("Falling back to CPU computation...")
            svf_values = compute_svf_cpu(street_points_3d, sky_patches, mesh, args.height)
    else:
        # Use CPU computation
        svf_values = compute_svf_cpu(street_points_3d, sky_patches, mesh, args.height)
    
    # Add SVF values to points GeoDataFrame
    points_gdf = points_gdf.copy()
    points_gdf['svf'] = svf_values
    
    # Aggregate to segment level
    segments_gdf = aggregate_segment_statistics(points_gdf, svf_values, filtered_roads_gdf)
    
    # Save results
    logger.info("Saving results...")
    
    points_output = output_dir / "street_svf_points.gpkg"
    points_gdf.to_file(points_output, driver='GPKG')
    logger.info(f"  Saved point-level results to {points_output}")
    
    segments_output = output_dir / "street_svf_segments.gpkg"
    segments_gdf.to_file(segments_output, driver='GPKG')
    logger.info(f"  Saved segment-level results to {segments_output}")
    
    csv_output = output_dir / "street_svf_statistics.csv"
    segments_gdf.drop(columns=['geometry']).to_csv(csv_output, index=False)
    logger.info(f"  Saved statistics to {csv_output}")
    
    # Visualizations
    logger.info("Generating visualizations...")
    
    map_path = output_dir / "street_svf_map.png"
    create_street_svf_map(segments_gdf, points_gdf, building_footprints, map_path)
    
    create_statistics_plots(segments_gdf, points_gdf, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("STREET SVF SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Computation method: {'GPU' if use_gpu else 'CPU'}")
    print(f"Total street segments: {len(segments_gdf)}")
    print(f"Total sample points: {len(points_gdf)}")
    print(f"\nPoint-level SVF:")
    print(f"  Mean: {points_gdf['svf'].mean():.4f}")
    print(f"  Std:  {points_gdf['svf'].std():.4f}")
    print(f"  Min:  {points_gdf['svf'].min():.4f}")
    print(f"  Max:  {points_gdf['svf'].max():.4f}")
    print(f"  Median: {points_gdf['svf'].median():.4f}")
    print(f"\nSegment-level SVF (mean values):")
    print(f"  Mean: {segments_gdf['svf_mean'].mean():.4f}")
    print(f"  Min:  {segments_gdf['svf_mean'].min():.4f}")
    print(f"  Max:  {segments_gdf['svf_mean'].max():.4f}")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
