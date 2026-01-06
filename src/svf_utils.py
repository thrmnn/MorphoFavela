"""
Shared utilities for SVF and solar access computation.

This module contains common functions used by both SVF and solar access scripts,
including mesh loading, terrain extraction, building footprint handling, and
ground point generation.
"""

import numpy as np
import pyvista as pv
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    # Determine translation to align with STL's local coordinate system
    # Assuming STL's origin is roughly at the center of its bounds
    # And that the STL's coordinate system is a local Cartesian system
    # We need to translate the building footprints to this local system
    
    # Calculate translation vector based on the center of the terrain bounds
    # This is a heuristic and might need adjustment based on actual STL origin
    stl_center_x = (terrain_bounds[0] + terrain_bounds[1]) / 2
    stl_center_y = (terrain_bounds[2] + terrain_bounds[3]) / 2
    
    footprints_center_x = (footprints.total_bounds[0] + footprints.total_bounds[2]) / 2
    footprints_center_y = (footprints.total_bounds[1] + footprints.total_bounds[3]) / 2
    
    dx = stl_center_x - footprints_center_x
    dy = stl_center_y - footprints_center_y
    
    print(f"  Detected coordinate system mismatch - transforming to local coordinates")
    print(f"  Applying translation: dx={dx:.1f}, dy={dy:.1f}")
    
    # Apply translation
    footprints.geometry = footprints.translate(xoff=dx, yoff=dy)
    
    print(f"  Transformed footprint bounds: {footprints.total_bounds}")
    
    # Apply buffer to avoid façade-adjacent artifacts
    if buffer_distance > 0:
        print(f"  Applying {buffer_distance}m buffer...")
        # Ensure geometry column is active
        footprints = footprints.set_geometry(footprints.geometry.buffer(buffer_distance))
    
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
        building_footprints: Optional GeoDataFrame with building footprints for mask computation
        output_dir: Optional output directory for debug plots
        
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

