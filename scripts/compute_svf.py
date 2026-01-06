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
import pyviewfactor as pvf
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import sys
import geopandas as gpd
from shapely.geometry import Point

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_mesh(stl_path: Path) -> pv.PolyData:
    """
    Load STL mesh using PyVista.
    
    Args:
        stl_path: Path to STL file
        
    Returns:
        PyVista PolyData mesh
    """
    print(f"Loading mesh from {stl_path}...")
    mesh = pv.read(str(stl_path))
    print(f"  Loaded {mesh.n_points} points, {mesh.n_cells} cells")
    return mesh


def extract_terrain_surface(mesh: pv.PolyData) -> pv.PolyData:
    """
    Identify terrain surface by selecting faces with upward-facing normals.
    
    Terrain faces typically have normals pointing upward (positive Z component).
    We select faces where the normal's Z component is above a threshold.
    
    Args:
        mesh: Full mesh containing terrain and buildings
        
    Returns:
        Terrain surface mesh
    """
    print("Extracting terrain surface...")
    
    # Compute face normals
    mesh = mesh.compute_normals(point_normals=False, cell_normals=True)
    
    # Get cell normals (face normals)
    normals = mesh.cell_normals
    
    # Select faces with upward-facing normals (Z > threshold)
    # Threshold of 0.5 means normal is pointing more upward than sideways
    upward_mask = normals[:, 2] > 0.5
    
    # Extract terrain surface
    terrain = mesh.extract_cells(upward_mask)
    
    print(f"  Extracted {terrain.n_cells} terrain faces")
    return terrain


def load_building_footprints(
    footprints_path: Path, 
    terrain_bounds: tuple,
    buffer_distance: float = 0.25
) -> gpd.GeoDataFrame:
    """
    Load and prepare building footprints for masking.
    
    Automatically transforms footprints to match terrain/STL coordinate system
    by detecting coordinate system mismatch and applying translation if needed.
    
    Args:
        footprints_path: Path to building footprints shapefile
        terrain_bounds: Terrain bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
        buffer_distance: Outward buffer distance in meters to avoid façade-adjacent artifacts
        
    Returns:
        GeoDataFrame with buffered building footprints in terrain coordinate system
    """
    print(f"Loading building footprints from {footprints_path}...")
    footprints = gpd.read_file(str(footprints_path))
    
    print(f"  Loaded {len(footprints)} building footprints")
    print(f"  Original CRS: {footprints.crs}")
    print(f"  Footprint bounds: {footprints.total_bounds}")
    
    # Check if coordinate systems match
    terrain_x_range = terrain_bounds[1] - terrain_bounds[0]
    terrain_y_range = terrain_bounds[3] - terrain_bounds[2]
    footprint_x_range = footprints.total_bounds[2] - footprints.total_bounds[0]
    footprint_y_range = footprints.total_bounds[3] - footprints.total_bounds[1]
    
    # Detect if we need to transform (if ranges are vastly different, likely different CRS)
    # STL typically has small coordinate values, UTM has large values (hundreds of thousands)
    needs_transform = (
        abs(footprints.total_bounds[0]) > 10000 or  # UTM-like coordinates
        abs(terrain_bounds[0]) < 1000  # Local coordinates
    )
    
    if needs_transform:
        print("  Detected coordinate system mismatch - transforming to local coordinates")
        # Calculate translation to center footprints on terrain
        # Use centroid of terrain as reference
        terrain_center_x = (terrain_bounds[0] + terrain_bounds[1]) / 2
        terrain_center_y = (terrain_bounds[2] + terrain_bounds[3]) / 2
        
        footprint_center_x = (footprints.total_bounds[0] + footprints.total_bounds[2]) / 2
        footprint_center_y = (footprints.total_bounds[1] + footprints.total_bounds[3]) / 2
        
        # Translate footprints to match terrain coordinate system
        # This assumes the STL is in a local coordinate system
        # We'll translate footprints to be centered on terrain
        dx = terrain_center_x - footprint_center_x
        dy = terrain_center_y - footprint_center_y
        
        print(f"  Applying translation: dx={dx:.1f}, dy={dy:.1f}")
        footprints = footprints.translate(xoff=dx, yoff=dy)
        print(f"  Transformed footprint bounds: {footprints.total_bounds}")
    
    # Apply buffer to avoid façade-adjacent artifacts
    if buffer_distance > 0:
        print(f"  Applying {buffer_distance}m buffer...")
        buffered_geoms = footprints.geometry.buffer(buffer_distance)
        if isinstance(footprints, gpd.GeoDataFrame):
            footprints = footprints.set_geometry(buffered_geoms)
        else:
            footprints = gpd.GeoDataFrame(geometry=buffered_geoms, crs=footprints.crs)
    
    print(f"  Prepared {len(footprints)} buffered footprints")
    return footprints


def compute_ground_mask(grid_x: np.ndarray, grid_y: np.ndarray, building_footprints: gpd.GeoDataFrame) -> np.ndarray:
    """
    Create a boolean mask for ground points (excluding building footprints).
    
    Args:
        grid_x: X coordinates of grid points (1D array)
        grid_y: Y coordinates of grid points (1D array)
        building_footprints: GeoDataFrame with building footprint polygons
        
    Returns:
        Boolean array: True for ground points (outside buildings), False for inside buildings
    """
    print("Computing ground mask (excluding building footprints)...")
    
    # Create GeoDataFrame from grid points
    grid_points_gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(grid_x, grid_y)],
        crs=building_footprints.crs
    )
    
    # Perform spatial join to find points within buildings
    # Points within buildings will have a match, others won't
    joined = gpd.sjoin(grid_points_gdf, building_footprints, how='left', predicate='within')
    
    # Handle potential duplicates from sjoin (if a point is in multiple buildings)
    # Group by original index and check if any match exists
    if len(joined) > len(grid_points_gdf):
        # Some points matched multiple buildings - deduplicate
        joined = joined.reset_index().drop_duplicates(subset=['index'], keep='first')
        joined = joined.set_index('index')
    
    # Create mask: True if point is NOT within any building (index_right is NaN)
    # Ensure we have the same number of points as input
    if len(joined) == len(grid_points_gdf):
        ground_mask = joined['index_right'].isna().values
    else:
        # Fallback: create mask by checking each point individually
        print(f"  Warning: Join result size mismatch. Using point-in-polygon check...")
        ground_mask = np.ones(len(grid_points_gdf), dtype=bool)
        for idx, point in enumerate(grid_points_gdf.geometry):
            if building_footprints.contains(point).any():
                ground_mask[idx] = False
    
    num_ground = np.sum(ground_mask)
    num_buildings = len(grid_points_gdf) - num_ground
    pct_masked = (num_buildings / len(grid_points_gdf)) * 100
    
    print(f"  Total grid points: {len(grid_points_gdf)}")
    print(f"  Ground points (outside buildings): {num_ground}")
    print(f"  Building points (masked): {num_buildings} ({pct_masked:.1f}%)")
    
    return ground_mask


def plot_ground_mask_debug(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    ground_mask: np.ndarray,
    building_footprints: gpd.GeoDataFrame,
    output_path: Path
):
    """
    Create a debug plot showing grid points, ground points, and building footprints.
    
    Args:
        grid_x: X coordinates of all grid points
        grid_y: Y coordinates of all grid points
        ground_mask: Boolean mask for ground points
        building_footprints: GeoDataFrame with building footprints
        output_path: Path to save the plot
    """
    print(f"Creating debug plot: {output_path}")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all grid points (grey)
    ax.scatter(grid_x, grid_y, c='lightgrey', s=1, alpha=0.3, label='All grid points')
    
    # Plot ground points (green) - only show points that are ground
    if len(ground_mask) == len(grid_x):
        ground_x = grid_x[ground_mask]
        ground_y = grid_y[ground_mask]
        ax.scatter(ground_x, ground_y, c='green', s=2, alpha=0.6, label='Ground points (SVF computed)')
    
    # Plot building footprints (red outlines)
    building_footprints.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, label='Building footprints')
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Ground Mask Debug Plot\n(Green = SVF computed, Red = Building footprints)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved debug plot to {output_path}")


def generate_ground_points(
    terrain: pv.PolyData, 
    grid_spacing: float, 
    ground_mask: np.ndarray = None,
    building_footprints: gpd.GeoDataFrame = None,
    output_dir: Path = None
) -> tuple:
    """
    Generate a regular 2D grid over the terrain bounding box.
    
    Args:
        terrain: Terrain surface mesh
        grid_spacing: Grid spacing in meters
        ground_mask: Optional boolean mask (True = ground, False = building)
                    If provided, only ground points are projected and returned
        
    Returns:
        Tuple of (projected_points, grid_x_coords, grid_y_coords, original_mask)
        - projected_points: Array of shape (N, 3) with (x, y, z) coordinates
        - grid_x_coords: 1D array of X coordinates for grid reconstruction
        - grid_y_coords: 1D array of Y coordinates for grid reconstruction
        - original_mask: Original ground mask (or all True if not provided)
    """
    print(f"Generating ground grid with spacing {grid_spacing}m...")
    
    # Get bounding box
    bounds = terrain.bounds
    x_min, x_max = bounds[0], bounds[1]
    y_min, y_max = bounds[2], bounds[3]
    
    # Generate grid
    x_coords = np.arange(x_min, x_max, grid_spacing)
    y_coords = np.arange(y_min, y_max, grid_spacing)
    
    # Create meshgrid
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Flatten to list of points
    grid_points_2d = np.column_stack([X.ravel(), Y.ravel()])
    grid_x_flat = grid_points_2d[:, 0]
    grid_y_flat = grid_points_2d[:, 1]
    
    # Compute ground mask if building footprints provided (but mask not already computed)
    if building_footprints is not None and ground_mask is None:
        print("  Computing ground mask from building footprints...")
        ground_mask = compute_ground_mask(grid_x_flat, grid_y_flat, building_footprints)
        
        # Create debug plot
        if output_dir is not None:
            debug_plot_path = output_dir / "ground_mask_debug.png"
            plot_ground_mask_debug(grid_x_flat, grid_y_flat, ground_mask, building_footprints, debug_plot_path)
    
    # Apply ground mask if provided (ensure sizes match)
    if ground_mask is not None:
        if len(ground_mask) != len(grid_points_2d):
            print(f"  Warning: Mask size ({len(ground_mask)}) doesn't match grid size ({len(grid_points_2d)})")
            print(f"  Recomputing mask on current grid...")
            if building_footprints is not None:
                ground_mask = compute_ground_mask(grid_x_flat, grid_y_flat, building_footprints)
            else:
                # If no footprints, all points are ground
                ground_mask = np.ones(len(grid_points_2d), dtype=bool)
        
        if len(ground_mask) == len(grid_points_2d):
            print(f"  Applying ground mask: {np.sum(ground_mask)}/{len(ground_mask)} points are ground")
            grid_points_2d = grid_points_2d[ground_mask]
            grid_x_flat = grid_x_flat[ground_mask]
            grid_y_flat = grid_y_flat[ground_mask]
        else:
            print(f"  Error: Mask size still doesn't match after recomputation. Using all points.")
            ground_mask = np.ones(len(grid_points_2d), dtype=bool)
    else:
        ground_mask = np.ones(len(grid_points_2d), dtype=bool)
    
    # Project each point vertically onto terrain surface
    # We'll set Z to a low value first, then project upward
    z_coords = np.full(len(grid_points_2d), bounds[4] - 100)  # Start below terrain
    points_3d = np.column_stack([grid_points_2d, z_coords])
    
    # Project points onto terrain surface
    # For each grid point, find the closest point on terrain and use its Z coordinate
    print("  Projecting points to terrain surface...")
    projected_points = []
    
    for point in tqdm(points_3d, desc="  Projecting"):
        # Find closest point on terrain
        closest_idx = terrain.find_closest_point(point)
        closest_point = terrain.points[closest_idx]
        
        # Use grid X, Y but terrain Z
        projected_points.append([point[0], point[1], closest_point[2]])
    
    projected_points = np.array(projected_points)
    
    print(f"  Generated {len(projected_points)} ground points (after masking)")
    return projected_points, x_coords, y_coords, ground_mask


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

