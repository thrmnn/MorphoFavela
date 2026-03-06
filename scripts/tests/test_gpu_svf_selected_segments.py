#!/usr/bin/env python3
"""
Test GPU SVF computation on selected street segments from Rio das Pedras.

This script:
1. Loads the road network for Rio das Pedras
2. Selects a few short street segments at different locations (near centroid and other locations)
3. Runs GPU SVF computation on just those segments
4. Compares with previous CPU results if available

Usage:
    python scripts/test_gpu_svf_selected_segments.py
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
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
from scripts.compute_svf_streets import (
    compute_svf as compute_svf_cpu,
    sample_street_points,
    aggregate_segment_statistics
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def select_test_segments(roads_gdf, num_segments=5, min_length=10.0, max_length=50.0):
    """
    Select representative street segments for testing.
    
    Args:
        roads_gdf: GeoDataFrame with street segments
        num_segments: Number of segments to select
        min_length: Minimum segment length (meters)
        max_length: Maximum segment length (meters)
        
    Returns:
        GeoDataFrame with selected segments
    """
    logger.info(f"Selecting {num_segments} test segments...")
    
    # Calculate segment lengths
    roads_gdf = roads_gdf.copy()
    roads_gdf['length'] = roads_gdf.geometry.length
    
    # Filter by length
    length_mask = (roads_gdf['length'] >= min_length) & (roads_gdf['length'] <= max_length)
    candidates = roads_gdf[length_mask].copy()
    
    if len(candidates) == 0:
        logger.warning(f"No segments found with length between {min_length}m and {max_length}m")
        logger.info("Relaxing length constraints...")
        # Relax constraints
        candidates = roads_gdf[roads_gdf['length'] >= min_length * 0.5].copy()
        candidates = candidates.nsmallest(num_segments * 2, 'length')
    
    logger.info(f"Found {len(candidates)} candidate segments")
    
    # Calculate centroid of all roads
    all_centroid = candidates.geometry.centroid.unary_union.centroid
    
    # Calculate distance from centroid for each candidate
    candidates['dist_from_centroid'] = candidates.geometry.centroid.distance(all_centroid)
    
    # Select segments: some near centroid, some at different distances
    selected_indices = []
    
    # 1-2 segments near centroid (closest)
    near_centroid = candidates.nsmallest(2, 'dist_from_centroid')
    selected_indices.extend(near_centroid.index[:2].tolist())
    
    # 1-2 segments at medium distance
    remaining = candidates[~candidates.index.isin(selected_indices)]
    if len(remaining) > 0:
        medium_dist = remaining.nsmallest(2, 'dist_from_centroid')
        if len(medium_dist) > 0:
            selected_indices.extend(medium_dist.index[:1].tolist())
    
    # 1-2 segments at far distance
    remaining = candidates[~candidates.index.isin(selected_indices)]
    if len(remaining) > 0:
        far_dist = remaining.nlargest(2, 'dist_from_centroid')
        if len(far_dist) > 0:
            selected_indices.extend(far_dist.index[:1].tolist())
    
    # Fill remaining slots with random selection if needed
    remaining = candidates[~candidates.index.isin(selected_indices)]
    if len(selected_indices) < num_segments and len(remaining) > 0:
        needed = num_segments - len(selected_indices)
        random_selection = remaining.sample(min(needed, len(remaining)), random_state=42)
        selected_indices.extend(random_selection.index.tolist())
    
    selected = candidates.loc[selected_indices[:num_segments]].copy()
    
    logger.info(f"Selected {len(selected)} segments:")
    for idx, row in selected.iterrows():
        logger.info(f"  Segment {idx}: length={row['length']:.1f}m, "
                   f"dist_from_centroid={row['dist_from_centroid']:.1f}m")
    
    return selected


def main():
    """Main execution block."""
    area = "riodaspedras"
    
    # Setup paths
    data_dir = PROJECT_ROOT / "data" / area / "raw"
    stl_path = data_dir / "full_scan.stl"
    roads_path = data_dir / "roads_riodaspedras.shp"
    footprints_path = data_dir / f"{area}_buildings.shp"
    dtm_path = data_dir / f"{area}_dtm.tif" if (data_dir / f"{area}_dtm.tif").exists() else None
    
    output_dir = PROJECT_ROOT / "outputs" / area / "svf_streets_gpu_test_selected"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    use_gpu = gpu_info['available']
    if use_gpu:
        logger.info(f"GPU available: {gpu_info['device_name']}")
    else:
        logger.warning("GPU not available. Using CPU.")
    
    print("=" * 60)
    print("GPU SVF TEST - SELECTED STREET SEGMENTS")
    print("=" * 60)
    print(f"Area: {area}")
    print(f"STL file: {stl_path}")
    print(f"Road network: {roads_path}")
    print(f"Computation: {'GPU' if use_gpu else 'CPU'}")
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
    
    # Handle coordinate system transformation
    terrain_bounds = terrain.bounds
    stl_center_x = (terrain_bounds[0] + terrain_bounds[1]) / 2
    stl_center_y = (terrain_bounds[2] + terrain_bounds[3]) / 2
    
    roads_center_x = (roads_gdf.total_bounds[0] + roads_gdf.total_bounds[2]) / 2
    roads_center_y = (roads_gdf.total_bounds[1] + roads_gdf.total_bounds[3]) / 2
    dx = stl_center_x - roads_center_x
    dy = stl_center_y - roads_center_y
    
    logger.info(f"  Transforming roads to match STL coordinate system")
    logger.info(f"  Applying translation: dx={dx:.1f}, dy={dy:.1f}")
    roads_gdf.geometry = roads_gdf.geometry.translate(xoff=dx, yoff=dy)
    
    # Select test segments
    test_segments = select_test_segments(roads_gdf, num_segments=5, min_length=10.0, max_length=50.0)
    
    # Save selected segments for reference
    test_segments_output = output_dir / "selected_test_segments.gpkg"
    test_segments.to_file(test_segments_output, driver='GPKG')
    logger.info(f"Saved selected segments to {test_segments_output}")
    
    # Sample points along selected segments
    spacing = 3.0
    height = 1.5
    sky_patches_count = 145
    
    logger.info(f"Sampling points along selected segments (spacing={spacing}m)...")
    points_gdf = sample_street_points(
        test_segments, spacing, dtm_path=dtm_path, terrain=terrain
    )
    
    # Remove points with invalid elevation
    valid_mask = ~points_gdf.geometry.apply(lambda p: np.isnan(p.z))
    if not valid_mask.all():
        logger.warning(f"  Removed {np.sum(~valid_mask)} points with invalid elevation")
        points_gdf = points_gdf[valid_mask].copy()
    
    logger.info(f"  Generated {len(points_gdf)} sample points")
    
    # Convert points to numpy array for SVF computation
    street_points_3d = np.array([
        [geom.x, geom.y, geom.z] for geom in points_gdf.geometry
    ])
    
    logger.info(f"Computing SVF for {len(street_points_3d)} street points...")
    
    # Generate sky patches
    sky_patches, _ = generate_sky_patches(sky_patches_count)
    
    # Compute SVF using GPU or CPU
    if use_gpu:
        try:
            # Convert mesh to PyTorch3D format
            logger.info("Converting mesh to PyTorch3D format...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=device)
            
            # Prepare data for GPU
            observer_points_torch = prepare_observer_points(
                street_points_3d, height, device=device
            )
            sky_patches_torch = prepare_sky_patches(sky_patches, device=device)
            
            # Compute SVF on GPU
            logger.info("Computing SVF on GPU...")
            svf_values_gpu = compute_svf_gpu(
                observer_points_torch,
                sky_patches_torch,
                pytorch3d_mesh,
                batch_size=200,
                num_samples_per_ray=20
            )
            
            # Convert back to numpy
            svf_values_gpu = svf_values_gpu.cpu().numpy()
            
            logger.info("GPU computation completed")
            
        except Exception as e:
            logger.error(f"GPU computation failed: {e}")
            logger.info("Falling back to CPU computation...")
            svf_values_gpu = compute_svf_cpu(street_points_3d, sky_patches, mesh, height)
            use_gpu = False
    else:
        # Use CPU computation
        logger.info("Computing SVF on CPU...")
        svf_values_gpu = compute_svf_cpu(street_points_3d, sky_patches, mesh, height)
    
    # Also compute on CPU for comparison
    logger.info("Computing SVF on CPU for comparison...")
    svf_values_cpu = compute_svf_cpu(street_points_3d, sky_patches, mesh, height)
    
    # Add SVF values to points GeoDataFrame
    points_gdf = points_gdf.copy()
    points_gdf['svf_gpu'] = svf_values_gpu
    points_gdf['svf_cpu'] = svf_values_cpu
    points_gdf['svf_diff'] = svf_values_gpu - svf_values_cpu
    points_gdf['svf_diff_abs'] = np.abs(points_gdf['svf_diff'])
    
    # Aggregate to segment level (using GPU values)
    segments_gdf = aggregate_segment_statistics(points_gdf, svf_values_gpu, test_segments)
    
    # Add CPU comparison columns (using segment_idx which is created by sample_street_points)
    cpu_stats = points_gdf.groupby('segment_idx').agg({
        'svf_cpu': ['mean', 'std'],
        'svf_diff': 'mean',
        'svf_diff_abs': 'mean'
    })
    cpu_stats.columns = ['svf_cpu_mean', 'svf_cpu_std', 'svf_diff_mean', 'svf_diff_abs_mean']
    
    # Merge with segments_gdf
    for col in cpu_stats.columns:
        segments_gdf[col] = segments_gdf['segment_idx'].map(cpu_stats[col])
    
    # Save results
    logger.info("Saving results...")
    
    points_output = output_dir / "street_svf_points_comparison.gpkg"
    points_gdf.to_file(points_output, driver='GPKG')
    logger.info(f"  Saved point-level results to {points_output}")
    
    segments_output = output_dir / "street_svf_segments_comparison.gpkg"
    segments_gdf.to_file(segments_output, driver='GPKG')
    logger.info(f"  Saved segment-level results to {segments_output}")
    
    csv_output = output_dir / "svf_comparison_statistics.csv"
    comparison_stats = pd.DataFrame({
        'method': ['GPU', 'CPU'],
        'mean_svf': [points_gdf['svf_gpu'].mean(), points_gdf['svf_cpu'].mean()],
        'std_svf': [points_gdf['svf_gpu'].std(), points_gdf['svf_cpu'].std()],
        'min_svf': [points_gdf['svf_gpu'].min(), points_gdf['svf_cpu'].min()],
        'max_svf': [points_gdf['svf_gpu'].max(), points_gdf['svf_cpu'].max()],
        'median_svf': [points_gdf['svf_gpu'].median(), points_gdf['svf_cpu'].median()],
    })
    comparison_stats.to_csv(csv_output, index=False)
    logger.info(f"  Saved comparison statistics to {csv_output}")
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("SVF COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Total sample points: {len(points_gdf)}")
    print(f"Total segments: {len(segments_gdf)}")
    print(f"\nGPU Results:")
    print(f"  Mean: {points_gdf['svf_gpu'].mean():.4f}")
    print(f"  Std:  {points_gdf['svf_gpu'].std():.4f}")
    print(f"  Min:  {points_gdf['svf_gpu'].min():.4f}")
    print(f"  Max:  {points_gdf['svf_gpu'].max():.4f}")
    print(f"  Median: {points_gdf['svf_gpu'].median():.4f}")
    print(f"\nCPU Results:")
    print(f"  Mean: {points_gdf['svf_cpu'].mean():.4f}")
    print(f"  Std:  {points_gdf['svf_cpu'].std():.4f}")
    print(f"  Min:  {points_gdf['svf_cpu'].min():.4f}")
    print(f"  Max:  {points_gdf['svf_cpu'].max():.4f}")
    print(f"  Median: {points_gdf['svf_cpu'].median():.4f}")
    print(f"\nDifference (GPU - CPU):")
    print(f"  Mean difference: {points_gdf['svf_diff'].mean():.4f}")
    print(f"  Mean absolute difference: {points_gdf['svf_diff_abs'].mean():.4f}")
    print(f"  Max absolute difference: {points_gdf['svf_diff_abs'].max():.4f}")
    print(f"  Std of difference: {points_gdf['svf_diff'].std():.4f}")
    
    # Segment-level comparison
    print(f"\nSegment-level comparison:")
    for idx, row in segments_gdf.iterrows():
        seg_idx = row.get('segment_idx', idx)
        print(f"  Segment {seg_idx}:")
        print(f"    GPU mean: {row['svf_mean']:.4f}, CPU mean: {row['svf_cpu_mean']:.4f}")
        print(f"    Difference: {row['svf_diff_mean']:.4f} (abs: {row['svf_diff_abs_mean']:.4f})")
    
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: SVF comparison scatter
    ax1 = axes[0, 0]
    ax1.scatter(points_gdf['svf_cpu'], points_gdf['svf_gpu'], alpha=0.5, s=10)
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect match')
    ax1.set_xlabel('CPU SVF')
    ax1.set_ylabel('GPU SVF')
    ax1.set_title('GPU vs CPU SVF Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference distribution
    ax2 = axes[0, 1]
    ax2.hist(points_gdf['svf_diff'], bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--', label='Zero difference')
    ax2.set_xlabel('SVF Difference (GPU - CPU)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of SVF Differences')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Absolute difference distribution
    ax3 = axes[1, 0]
    ax3.hist(points_gdf['svf_diff_abs'], bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Absolute SVF Difference')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Absolute SVF Differences')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Segment-level comparison
    ax4 = axes[1, 1]
    segment_indices = segments_gdf['segment_idx'].tolist()
    x_pos = np.arange(len(segment_indices))
    width = 0.35
    ax4.bar(x_pos - width/2, segments_gdf['svf_mean'], width, label='GPU', alpha=0.7)
    ax4.bar(x_pos + width/2, segments_gdf['svf_cpu_mean'], width, label='CPU', alpha=0.7)
    ax4.set_xlabel('Segment Index')
    ax4.set_ylabel('Mean SVF')
    ax4.set_title('Segment-level SVF Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Seg {i}' for i in segment_indices], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    comparison_plot_path = output_dir / "svf_comparison_plots.png"
    fig.savefig(comparison_plot_path, dpi=200, bbox_inches='tight')
    logger.info(f"Saved comparison plots to {comparison_plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
