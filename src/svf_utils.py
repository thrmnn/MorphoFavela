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
from src.config import MAX_FILTER_AREA, is_informal_area, BUILDING_CLUSTER_BUFFER
from shapely.ops import unary_union


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


def filter_buildings(
    footprints: gpd.GeoDataFrame,
    area: str = None
) -> tuple:
    """
    Filter buildings based on area and height thresholds, and cluster filtering (for informal areas only).
    
    Uses connected components analysis to identify the main building cluster and filter out isolated buildings.
    
    Args:
        footprints: GeoDataFrame with building footprints
        area: Optional area name to determine if filtering should be applied
        
    Returns:
        Tuple of (filtered_footprints, isolated_footprints, area_filtered_footprints):
        - filtered_footprints: GeoDataFrame with valid buildings in main cluster
        - isolated_footprints: GeoDataFrame with isolated buildings (for visualization)
        - area_filtered_footprints: GeoDataFrame with buildings filtered by area threshold (for visualization)
    """
    # Check if filtering should be applied
    apply_filtering = False
    if area is not None:
        apply_filtering = is_informal_area(area)
    
    if not apply_filtering:
        print("  Formal area or no area specified: skipping building filters")
        # Return all buildings as "filtered", empty isolated and area_filtered
        isolated = gpd.GeoDataFrame(geometry=[], crs=footprints.crs)
        area_filtered = gpd.GeoDataFrame(geometry=[], crs=footprints.crs)
        return footprints, isolated, area_filtered
    
    print("  Applying building filters (informal area)...")
    initial_count = len(footprints)
    
    # Calculate area if not already present
    if 'area' not in footprints.columns:
        footprints = footprints.copy()
        footprints['area'] = footprints.geometry.area
    
    # Filter by area first - track what was filtered
    area_filtered = gpd.GeoDataFrame(geometry=[], crs=footprints.crs)
    if MAX_FILTER_AREA is not None:
        area_mask = footprints['area'] <= MAX_FILTER_AREA
        filtered_by_area = (~area_mask).sum()
        area_filtered = footprints[~area_mask].copy()  # Buildings filtered by area
        footprints = footprints[area_mask].copy()  # Keep only buildings that passed area filter
        if filtered_by_area > 0:
            print(f"    Filtered out {filtered_by_area} buildings with area > {MAX_FILTER_AREA}m²")
    
    # Apply cluster filtering to identify main blob
    if len(footprints) == 0:
        isolated = gpd.GeoDataFrame(geometry=[], crs=footprints.crs if len(footprints) > 0 else area_filtered.crs)
        return footprints, isolated, area_filtered
    
    print(f"  Applying cluster filtering (buffer: {BUILDING_CLUSTER_BUFFER}m)...")
    cluster_filtered, isolated = filter_isolated_buildings(footprints)
    
    final_count = len(cluster_filtered)
    isolated_count = len(isolated)
    
    if initial_count != final_count:
        total_removed = initial_count - final_count
        print(f"  Total filtered: {initial_count} → {final_count} ({100*total_removed/initial_count:.1f}% removed)")
        if isolated_count > 0:
            print(f"    - Isolated buildings removed: {isolated_count}")
    
    return cluster_filtered, isolated, area_filtered


def filter_isolated_buildings(
    footprints: gpd.GeoDataFrame
) -> tuple:
    """
    Filter out isolated buildings using connected components analysis.
    
    Buffers buildings and finds the largest connected component (main cluster).
    Buildings not in the main cluster are considered isolated.
    
    Args:
        footprints: GeoDataFrame with building footprints
        
    Returns:
        Tuple of (main_cluster_footprints, isolated_footprints):
        - main_cluster_footprints: Buildings in the main cluster
        - isolated_footprints: Isolated buildings (for visualization)
    """
    if len(footprints) == 0:
        isolated = gpd.GeoDataFrame(geometry=[], crs=footprints.crs)
        return footprints, isolated
    
    print("    Identifying main building cluster...")
    initial_count = len(footprints)
    
    # Buffer buildings to account for gaps between them
    buffered = footprints.buffer(BUILDING_CLUSTER_BUFFER)
    
    # Create unified geometry from all buffers
    unified = unary_union(buffered.values)
    
    # Handle MultiPolygon vs Polygon
    if hasattr(unified, 'geoms'):
        # MultiPolygon - multiple components
        components = list(unified.geoms)
    else:
        # Single Polygon - one component
        components = [unified]
    
    if len(components) == 0:
        print("    Warning: No components found, keeping all buildings")
        isolated = gpd.GeoDataFrame(geometry=[], crs=footprints.crs)
        return footprints, isolated
    
    # Find the largest component (main cluster)
    component_areas = [comp.area for comp in components]
    main_component_idx = np.argmax(component_areas)
    main_component = components[main_component_idx]
    
    main_area = component_areas[main_component_idx]
    total_area = sum(component_areas)
    main_pct = 100 * main_area / total_area if total_area > 0 else 0
    
    print(f"    Found {len(components)} component(s), main cluster: {main_pct:.1f}% of total area")
    
    # Identify which buildings are in the main component
    # A building is in the main cluster if its buffered geometry intersects the main component
    in_main_cluster = buffered.intersects(main_component)
    
    main_cluster_footprints = footprints[in_main_cluster].copy()
    isolated_footprints = footprints[~in_main_cluster].copy()
    
    isolated_count = len(isolated_footprints)
    if isolated_count > 0:
        print(f"    Isolated buildings: {isolated_count} ({100*isolated_count/initial_count:.1f}%)")
    
    return main_cluster_footprints, isolated_footprints


def load_building_footprints(
    footprints_path: Path, 
    terrain_bounds: tuple,
    buffer_distance: float = 0.25,
    area: str = None
) -> tuple:
    """
    Load and prepare building footprints for masking.
    
    Automatically transforms footprints to match terrain/STL coordinate system
    by detecting coordinate system mismatch and applying translation if needed.
    Applies filtering for informal areas before transformation.
    
    Args:
        footprints_path: Path to building footprints shapefile
        terrain_bounds: Terrain bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
        buffer_distance: Outward buffer distance in meters to avoid façade-adjacent artifacts
        area: Optional area name to determine if filtering should be applied
        
    Returns:
        Tuple of (buffered_footprints, isolated_footprints, area_filtered_footprints):
        - buffered_footprints: GeoDataFrame with filtered and buffered building footprints in terrain coordinate system
        - isolated_footprints: GeoDataFrame with isolated buildings (for visualization, already transformed)
        - area_filtered_footprints: GeoDataFrame with buildings filtered by area threshold (for visualization, already transformed)
    """
    print(f"Loading building footprints from {footprints_path}...")
    footprints = gpd.read_file(str(footprints_path))
    
    print(f"  Loaded {len(footprints)} building footprints")
    print(f"  Original CRS: {footprints.crs}")
    print(f"  Footprint bounds: {footprints.total_bounds}")
    
    # Calculate transformation using ORIGINAL (unfiltered) footprints center
    # This ensures consistent alignment with other datasets (e.g., streets)
    stl_center_x = (terrain_bounds[0] + terrain_bounds[1]) / 2
    stl_center_y = (terrain_bounds[2] + terrain_bounds[3]) / 2
    
    # Use ORIGINAL footprints center (before filtering) for transformation
    original_footprints_center_x = (footprints.total_bounds[0] + footprints.total_bounds[2]) / 2
    original_footprints_center_y = (footprints.total_bounds[1] + footprints.total_bounds[3]) / 2
    
    dx = stl_center_x - original_footprints_center_x
    dy = stl_center_y - original_footprints_center_y
    
    # Apply filtering BEFORE transformation (filtering uses original CRS)
    # Returns (filtered_footprints, isolated_footprints, area_filtered_footprints)
    footprints, isolated_footprints, area_filtered_footprints = filter_buildings(footprints, area=area)
    
    print("  Detected coordinate system mismatch - transforming to local coordinates")
    print(f"  Applying translation: dx={dx:.1f}, dy={dy:.1f}")
    
    # Apply translation to filtered, isolated, and area-filtered footprints
    footprints.geometry = footprints.geometry.translate(xoff=dx, yoff=dy)
    if len(isolated_footprints) > 0:
        isolated_footprints.geometry = isolated_footprints.geometry.translate(xoff=dx, yoff=dy)
        # Ensure CRS matches for plotting
        isolated_footprints.crs = footprints.crs
    if len(area_filtered_footprints) > 0:
        area_filtered_footprints.geometry = area_filtered_footprints.geometry.translate(xoff=dx, yoff=dy)
        # Ensure CRS matches for plotting
        area_filtered_footprints.crs = footprints.crs
    
    print(f"  Transformed footprint bounds: {footprints.total_bounds}")
    
    # Apply buffer to avoid façade-adjacent artifacts (only to main cluster)
    if buffer_distance > 0:
        print(f"  Applying {buffer_distance}m buffer...")
        # Ensure geometry column is active
        footprints = footprints.set_geometry(footprints.geometry.buffer(buffer_distance))
    
    print(f"  Prepared {len(footprints)} buffered footprints (main cluster)")
    if len(isolated_footprints) > 0:
        print(f"  Isolated buildings: {len(isolated_footprints)} (for visualization only)")
    if len(area_filtered_footprints) > 0:
        print(f"  Area-filtered buildings: {len(area_filtered_footprints)} (for visualization only)")
    
    return footprints, isolated_footprints, area_filtered_footprints


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
        print("  Warning: Join result size mismatch. Using point-in-polygon check...")
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
    output_path: Path,
    building_buffer: float = None,
    building_buffer_geom = None,
    original_ground_count: int = None,
    isolated_buildings: gpd.GeoDataFrame = None,
    area_filtered_buildings: gpd.GeoDataFrame = None
):
    """
    Create a debug plot showing grid points, ground points, and building footprints.
    
    Args:
        grid_x: X coordinates of filtered grid points (after proximity filter if applied)
        grid_y: Y coordinates of filtered grid points
        ground_mask: Boolean mask for ground points (after filtering)
        building_footprints: GeoDataFrame with building footprints
        output_path: Path to save the plot
        building_buffer: Optional buffer distance used for filtering
        building_buffer_geom: Optional buffer geometry to visualize
        original_ground_count: Optional original count before proximity filtering
    """
    print(f"Creating debug plot: {output_path}")
    
    # Import config to check filtering thresholds
    from src.config import MAX_FILTER_AREA
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot area-filtered buildings in yellow (first, so they appear behind)
    if area_filtered_buildings is not None and len(area_filtered_buildings) > 0:
        area_filtered_buildings.plot(ax=ax, facecolor='yellow', edgecolor='orange', 
                                     linewidth=1.5, alpha=0.5, 
                                     label=f'Area-filtered buildings (area > {MAX_FILTER_AREA}m², n={len(area_filtered_buildings)})')
    
    # Plot isolated buildings (gray/transparent) - so they appear behind
    if isolated_buildings is not None and len(isolated_buildings) > 0:
        # Ensure CRS matches for plotting
        if isolated_buildings.crs != building_footprints.crs:
            isolated_buildings = isolated_buildings.to_crs(building_footprints.crs)
        isolated_buildings.plot(ax=ax, facecolor='lightgray', edgecolor='gray', 
                               linewidth=1, alpha=0.3, 
                               label=f'Isolated buildings (n={len(isolated_buildings)})')
    
    # Plot building buffer zone if provided (semi-transparent overlay, very light)
    if building_buffer_geom is not None:
        buffer_gdf = gpd.GeoDataFrame(geometry=[building_buffer_geom], crs=building_footprints.crs)
        buffer_gdf.plot(ax=ax, facecolor='none', edgecolor='orange', linewidth=1, 
                       alpha=0.3, linestyle='--', label=f'Proximity buffer ({building_buffer}m)')
    
    # Plot building footprints (red) - these are already filtered, so all shown are valid
    # Calculate area for statistics
    building_footprints = building_footprints.copy()
    building_footprints['area'] = building_footprints.geometry.area
    
    building_footprints.plot(ax=ax, facecolor='red', edgecolor='darkred', 
                            linewidth=1.5, alpha=0.4, 
                            label=f'Main cluster buildings (n={len(building_footprints)})')
    
    # Plot all filtered grid points (light grey - these are the ones we're considering)
    ax.scatter(grid_x, grid_y, c='lightgrey', s=1, alpha=0.4, label='All filtered grid points')
    
    # Plot ground points (green) - only show points that are ground (will be used for SVF)
    if len(ground_mask) == len(grid_x):
        ground_x = grid_x[ground_mask]
        ground_y = grid_y[ground_mask]
        ax.scatter(ground_x, ground_y, c='green', s=3, alpha=0.7, 
                  label='Ground points (SVF will be computed)', edgecolors='darkgreen', linewidths=0.2)
    
    # Add statistics text
    stats_text = []
    if original_ground_count is not None and len(grid_x) < original_ground_count:
        reduction = 100 * (1 - len(grid_x) / original_ground_count)
        stats_text.append(f"Points reduced by {reduction:.1f}%")
    if len(ground_mask) == len(grid_x):
        stats_text.append(f"SVF points: {np.sum(ground_mask):,}")
    stats_text.append(f"Main cluster: {len(building_footprints):,} buildings")
    if area_filtered_buildings is not None and len(area_filtered_buildings) > 0:
        stats_text.append(f"Area-filtered: {len(area_filtered_buildings):,} buildings")
    if isolated_buildings is not None and len(isolated_buildings) > 0:
        stats_text.append(f"Isolated: {len(isolated_buildings):,} buildings")
    if 'area' in building_footprints.columns and len(building_footprints) > 0:
        stats_text.append(f"Max building area: {building_footprints['area'].max():.1f} m²")
    
    if stats_text:
        stats_str = "\n".join(stats_text)
        ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    
    title = 'Ground Mask Debug Plot'
    if building_buffer is not None:
        title += f'\n(Proximity filter: {building_buffer}m buffer)'
    if MAX_FILTER_AREA is not None:
        title += f'\n(Area filter: >{MAX_FILTER_AREA}m² excluded)'
    title += f'\n(Cluster filter: {BUILDING_CLUSTER_BUFFER}m buffer)'
    title += '\nGreen = SVF points, Red = Main cluster, Yellow = Area-filtered, Gray = Isolated'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
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
    output_dir: Path = None,
    building_buffer: float = None,
    isolated_buildings: gpd.GeoDataFrame = None,
    area_filtered_buildings: gpd.GeoDataFrame = None
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
        building_buffer: Optional buffer distance in meters. If provided, only compute SVF
                        for points within this distance of buildings (reduces computation)
        
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
    
    # Apply ground mask if provided (ensure sizes match)
    if ground_mask is not None:
        if len(ground_mask) != len(grid_points_2d):
            print(f"  Warning: Mask size ({len(ground_mask)}) doesn't match grid size ({len(grid_points_2d)})")
            print("  Recomputing mask on current grid...")
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
            print("  Error: Mask size still doesn't match after recomputation. Using all points.")
            ground_mask = np.ones(len(grid_points_2d), dtype=bool)
    else:
        ground_mask = np.ones(len(grid_points_2d), dtype=bool)
    
    # Store original counts for reporting
    original_ground_count = len(grid_points_2d)
    building_buffer_geom = None
    
    # Apply building proximity filter if requested
    if building_buffer is not None and building_footprints is not None and len(grid_points_2d) > 0:
        print(f"  Filtering to points within {building_buffer}m of buildings...")
        # Create buffer around buildings
        building_buffers = building_footprints.buffer(building_buffer)
        # Combine all buffers into single geometry using unary_union
        from shapely.ops import unary_union
        try:
            building_buffer_geom = unary_union(building_buffers.geometry.values)
        except Exception as e:
            print(f"  Warning: Could not create unified buffer geometry: {e}")
            print("  Using individual building buffers...")
            # Fallback: use the first buffer (less efficient but works)
            building_buffer_geom = building_buffers.geometry.iloc[0]
            for geom in building_buffers.geometry.iloc[1:]:
                building_buffer_geom = building_buffer_geom.union(geom)
        
        # Convert grid points to GeoDataFrame for spatial query
        from shapely.geometry import Point
        grid_points_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in zip(grid_x_flat, grid_y_flat)],
            crs=building_footprints.crs
        )
        
        # Find points within buffer using spatial query
        proximity_mask = grid_points_gdf.geometry.within(building_buffer_geom)
        num_near_buildings = np.sum(proximity_mask)
        
        print(f"  Points near buildings: {num_near_buildings}/{len(grid_points_2d)} ({100*num_near_buildings/len(grid_points_2d):.1f}%)")
        
        if num_near_buildings == 0:
            print(f"  WARNING: No points found within {building_buffer}m of buildings!")
            print("  Continuing with all points...")
        else:
            # Apply proximity filter
            grid_points_2d = grid_points_2d[proximity_mask]
            grid_x_flat = grid_x_flat[proximity_mask]
            grid_y_flat = grid_y_flat[proximity_mask]
            # Update ground_mask to match filtered points
            if len(ground_mask) == len(proximity_mask):
                ground_mask = ground_mask[proximity_mask]
    
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
    
    print(f"  Generated {len(projected_points)} ground points (after masking and filtering)")
    
    # Create debug plot after all filtering is done
    if output_dir is not None and building_footprints is not None:
        debug_plot_path = output_dir / "ground_mask_debug.png"
        plot_ground_mask_debug(
            grid_x_flat, grid_y_flat, ground_mask, building_footprints, debug_plot_path,
            building_buffer=building_buffer,
            building_buffer_geom=building_buffer_geom,
            original_ground_count=original_ground_count,
            isolated_buildings=isolated_buildings,
            area_filtered_buildings=area_filtered_buildings
        )
    
    return projected_points, x_coords, y_coords, ground_mask, building_buffer_geom

