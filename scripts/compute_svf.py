#!/usr/bin/env python3
"""
Compute Sky View Factor (SVF) at ground level from a single STL file.

This script computes SVF on a 2D ground grid by treating the sky as a 
discretized hemispherical dome and using pyviewfactor-style geometric 
visibility to determine how much of the sky is visible from each ground point.

Usage:
    python scripts/compute_svf.py --stl data/raw/scene.stl --grid-spacing 2.0 --height 0.5 --sky-patches 145
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared utilities
from src.svf_utils import (
    load_mesh,
    extract_terrain_surface,
    load_building_footprints,
    compute_ground_mask,
    plot_ground_mask_debug,
    generate_ground_points
)


def generate_sky_patches(num_patches: int, radius: float = 1000.0) -> tuple:
    """
    Discretize the upper hemisphere into equal-area sky patches.
    
    Uses a simple latitude-longitude subdivision approach.
    For equal-area patches, we can use a Tregenza-like subdivision
    or a simpler uniform angular grid.
    
    Args:
        num_patches: Number of sky patches (e.g., 145, 290)
        radius: Distance to sky hemisphere (meters)
        
    Returns:
        Tuple of (patch_centroids, patch_meshes)
        - patch_centroids: Array of shape (N, 3) with patch center coordinates
        - patch_meshes: List of PyVista meshes for each patch
    """
    print(f"Generating {num_patches} sky patches...")
    
    # Simple approach: uniform angular grid
    # Calculate grid dimensions to approximate num_patches
    # For hemisphere: patches ≈ azimuth_steps * elevation_steps / 2
    azimuth_steps = int(np.sqrt(num_patches * 2))
    elevation_steps = int(num_patches / azimuth_steps)
    
    # Adjust to get close to desired number
    total_patches = azimuth_steps * elevation_steps
    if total_patches < num_patches:
        elevation_steps += 1
        total_patches = azimuth_steps * elevation_steps
    
    print(f"  Using {azimuth_steps} azimuth × {elevation_steps} elevation = {total_patches} patches")
    
    patch_centroids = []
    patch_meshes = []
    
    for az_idx in range(azimuth_steps):
        azimuth = 2 * np.pi * az_idx / azimuth_steps  # 0 to 2π
        
        for el_idx in range(elevation_steps):
            # Elevation from 0 (horizon) to π/2 (zenith)
            elevation = np.pi / 2 * (el_idx + 0.5) / elevation_steps
            
            # Skip patches below horizon (elevation < 0)
            if elevation <= 0:
                continue
            
            # Calculate patch centroid direction
            dx = np.cos(elevation) * np.cos(azimuth)
            dy = np.cos(elevation) * np.sin(azimuth)
            dz = np.sin(elevation)
            
            # Patch centroid at distance radius
            centroid = np.array([dx, dy, dz]) * radius
            patch_centroids.append(centroid)
            
            # Create a small patch polygon (simplified as a point for ray casting)
            # For actual view factor computation, we'd create a proper polygon
            # Here we use the centroid for ray casting
            patch_meshes.append(None)  # Not needed for ray casting approach
    
    patch_centroids = np.array(patch_centroids)
    print(f"  Generated {len(patch_centroids)} sky patches")
    
    return patch_centroids, patch_meshes


def compute_svf(
    ground_points: np.ndarray,
    sky_patches: np.ndarray,
    full_mesh: pv.PolyData,
    evaluation_height: float
) -> np.ndarray:
    """
    Compute SVF for each ground point.
    
    For each grid point and each sky patch:
    1. Cast a ray toward the patch centroid
    2. Use PyVista ray intersection to test obstruction
    3. Compute SVF as: number of visible patches / total patches
    
    Args:
        ground_points: Array of shape (N, 3) with ground point coordinates
        sky_patches: Array of shape (M, 3) with sky patch centroids
        full_mesh: Full scene mesh (terrain + buildings)
        evaluation_height: Height above ground for evaluation (meters)
        
    Returns:
        Array of SVF values (0-1) for each ground point
    """
    print(f"Computing SVF for {len(ground_points)} points...")
    print(f"  Evaluation height: {evaluation_height}m")
    print(f"  Sky patches: {len(sky_patches)}")
    
    svf_values = []
    
    # Create observer points (ground points + evaluation height)
    observer_points = ground_points.copy()
    observer_points[:, 2] += evaluation_height
    
    # Compute SVF for each observer point
    pbar = tqdm(total=len(observer_points), desc="Computing SVF", unit="points")
    
    for i, observer in enumerate(observer_points):
        visible_patches = 0
        
        for patch_centroid in sky_patches:
            # Ray direction from observer to patch centroid
            ray_direction = patch_centroid - observer
            ray_length = np.linalg.norm(ray_direction)
            ray_direction = ray_direction / ray_length  # Normalize
            
            # Cast ray and check for intersection
            # PyVista ray intersection
            ray_end = observer + ray_direction * ray_length
            intersection, cell_id = full_mesh.ray_trace(observer, ray_end)
            
            # If no intersection, patch is visible
            # If intersection exists, the ray hit something before reaching the sky
            if len(intersection) == 0:
                visible_patches += 1
            # else: ray was blocked, patch is not visible
        
        # SVF = visible patches / total patches
        svf = visible_patches / len(sky_patches)
        svf_values.append(svf)
        
        # Update progress bar with current statistics
        if len(svf_values) > 0:
            current_mean = np.mean(svf_values)
            pbar.set_postfix({
                'mean_svf': f'{current_mean:.3f}',
                'current': f'{svf:.3f}'
            })
        
        pbar.update(1)
    
    pbar.close()
    
    return np.array(svf_values)


def save_and_plot_results(
    ground_points: np.ndarray,
    svf_values: np.ndarray,
    grid_x_coords: np.ndarray,
    grid_y_coords: np.ndarray,
    ground_mask: np.ndarray,
    output_dir: Path
):
    """
    Save results and generate visualizations.
    
    Saves:
    - svf.npy → 2D NumPy array (NaN for building points)
    - svf.csv → x, y, svf (only ground points)
    - svf_heatmap.png → Top-down SVF heatmap
    - svf_histogram.png → Histogram of SVF values
    
    Args:
        ground_points: Array of shape (N, 3) with ground point coordinates (only ground points)
        svf_values: Array of SVF values (only for ground points)
        grid_x_coords: 1D array of X coordinates for full grid
        grid_y_coords: 1D array of Y coordinates for full grid
        ground_mask: Boolean mask for full grid (True = ground, False = building)
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving results...")
    
    # Create 2D grid with NaN for building points
    svf_2d = np.full((len(grid_y_coords), len(grid_x_coords)), np.nan)
    
    # Map ground points back to full grid
    for i, (x, y, z) in enumerate(ground_points):
        x_idx = np.argmin(np.abs(grid_x_coords - x))
        y_idx = np.argmin(np.abs(grid_y_coords - y))
        svf_2d[y_idx, x_idx] = svf_values[i]
    
    # Save as .npy
    npy_path = output_dir / "svf.npy"
    np.save(npy_path, svf_2d)
    print(f"  Saved {npy_path}")
    
    # Save as .csv
    csv_data = pd.DataFrame({
        'x': ground_points[:, 0],
        'y': ground_points[:, 1],
        'svf': svf_values
    })
    csv_path = output_dir / "svf.csv"
    csv_data.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")
    
    # Generate heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create extent for imshow (in meters)
    extent = [
        grid_x_coords.min(), grid_x_coords.max(),
        grid_y_coords.min(), grid_y_coords.max()
    ]
    
    im = ax.imshow(
        svf_2d,
        extent=extent,
        cmap='viridis',
        vmin=0,
        vmax=1,
        origin='lower',
        interpolation='nearest'
    )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Sky View Factor (SVF) Map', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('SVF (0-1)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    heatmap_path = output_dir / "svf_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {heatmap_path}")
    
    # Generate histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(svf_values, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('SVF Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Sky View Factor Values', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_svf = np.nanmean(svf_values)
    std_svf = np.nanstd(svf_values)
    ax.axvline(mean_svf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_svf:.3f}')
    ax.legend()
    
    plt.tight_layout()
    hist_path = output_dir / "svf_histogram.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {hist_path}")
    
    # Print summary statistics
    print("\nSVF Statistics:")
    print(f"  Mean: {mean_svf:.4f}")
    print(f"  Std:  {std_svf:.4f}")
    print(f"  Min:  {np.nanmin(svf_values):.4f}")
    print(f"  Max:  {np.nanmax(svf_values):.4f}")


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description='Compute Sky View Factor from STL file')
    parser.add_argument('--stl', type=str, required=True, help='Path to STL file')
    parser.add_argument('--footprints', type=str, default=None, help='Path to building footprints shapefile (optional)')
    parser.add_argument('--grid-spacing', type=float, default=2.0, help='Grid spacing in meters')
    parser.add_argument('--height', type=float, default=0.5, help='Evaluation height above ground (meters)')
    parser.add_argument('--sky-patches', type=int, default=145, help='Number of sky patches')
    parser.add_argument('--buffer-distance', type=float, default=0.25, help='Buffer distance for building footprints (meters)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: outputs/svf)')
    
    args = parser.parse_args()
    
    # Setup paths
    stl_path = Path(args.stl)
    if not stl_path.exists():
        print(f"Error: STL file not found: {stl_path}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "outputs" / "svf"
    
    print("=" * 60)
    print("SKY VIEW FACTOR (SVF) COMPUTATION")
    print("=" * 60)
    print(f"STL file: {stl_path}")
    if args.footprints:
        print(f"Building footprints: {args.footprints}")
    print(f"Grid spacing: {args.grid_spacing}m")
    print(f"Evaluation height: {args.height}m")
    print(f"Sky patches: {args.sky_patches}")
    print("=" * 60)
    
    # Load mesh
    mesh = load_mesh(stl_path)
    
    # Extract terrain surface
    terrain = extract_terrain_surface(mesh)
    
    # Load and prepare building footprints if provided
    building_footprints = None
    ground_mask = None
    if args.footprints:
        footprints_path = Path(args.footprints)
        if not footprints_path.exists():
            print(f"Warning: Building footprints file not found: {footprints_path}")
            print("  Continuing without building mask...")
        else:
            # Load and transform building footprints to match terrain coordinate system
            building_footprints = load_building_footprints(
                footprints_path,
                terrain_bounds=terrain.bounds,
                buffer_distance=args.buffer_distance
            )
            
            # Compute ground mask - we'll do this inside generate_ground_points
            # to ensure the grid is consistent
            pass  # Will compute mask in generate_ground_points
    
    # Generate ground points (with mask applied if available)
    # If footprints provided, pass them for mask computation
    if args.footprints and building_footprints is not None:
        ground_points, grid_x_coords, grid_y_coords, original_mask = generate_ground_points(
            terrain, args.grid_spacing, building_footprints=building_footprints, output_dir=output_dir
        )
    else:
        ground_points, grid_x_coords, grid_y_coords, original_mask = generate_ground_points(
            terrain, args.grid_spacing
        )
    
    # Generate sky patches
    sky_patches, _ = generate_sky_patches(args.sky_patches)
    
    # Compute SVF (only for ground points)
    svf_values = compute_svf(ground_points, sky_patches, mesh, args.height)
    
    # Save and plot results
    save_and_plot_results(
        ground_points, svf_values, grid_x_coords, grid_y_coords, original_mask, output_dir
    )
    
    print("\n" + "=" * 60)
    print("SVF COMPUTATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    if args.footprints and building_footprints is not None:
        print(f"  Ground mask debug plot: {output_dir / 'ground_mask_debug.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

