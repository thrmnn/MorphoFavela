#!/usr/bin/env python3
"""
Compute ground-level direct solar access from a single STL file.

This script computes solar access (hours of direct sunlight) on a 2D ground grid
by casting rays toward sun positions for winter solstice and testing for obstructions.

Usage:
    python scripts/compute_solar_access.py --stl data/raw/scene.stl --footprints data/raw/buildings.shp --grid-spacing 5.0 --height 0.5
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import sys
from datetime import datetime
import pvlib
from matplotlib.colors import ListedColormap

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared utilities
from src.svf_utils import (
    load_mesh,
    extract_terrain_surface,
    load_building_footprints,
    generate_ground_points
)


def compute_sun_positions(
    latitude: float,
    longitude: float,
    year: int = 2024,
    timezone: str = 'America/Sao_Paulo',
    timestep_minutes: int = 60
) -> tuple:
    """
    Compute sun positions for winter solstice.
    
    Args:
        latitude: Site latitude (degrees)
        longitude: Site longitude (degrees)
        year: Year for calculation (default: 2024)
        timezone: Timezone string (default: 'America/Sao_Paulo' for Brazil)
        timestep_minutes: Time step in minutes (default: 60)
        
    Returns:
        Tuple of (sun_directions, sun_times):
        - sun_directions: Array of shape (N, 3) with sun direction vectors (normalized)
        - sun_times: Index of valid sun position times
    """
    print(f"Computing sun positions for winter solstice...")
    print(f"  Location: ({latitude:.4f}°, {longitude:.4f}°)")
    print(f"  Time step: {timestep_minutes} minutes")
    
    # Winter solstice date (June 21 in Southern Hemisphere, Dec 21 in Northern)
    # For Brazil (Southern Hemisphere), use June 21
    if latitude < 0:
        solstice_date = datetime(year, 6, 21)
    else:
        solstice_date = datetime(year, 12, 21)
    
    # Create time range for the day (sunrise to sunset, sampled every timestep_minutes)
    times = pd.date_range(
        start=f"{solstice_date.date()} 00:00",
        end=f"{solstice_date.date()} 23:59",
        freq=f'{timestep_minutes}min',
        tz=timezone
    )
    
    # Get location
    location = pvlib.location.Location(latitude, longitude, tz=timezone)
    
    # Compute solar position
    solar_pos = location.get_solarposition(times)
    
    # Filter out positions where sun is below horizon (altitude <= 0)
    valid_mask = solar_pos['elevation'] > 0
    valid_positions = solar_pos[valid_mask]
    
    print(f"  Total time steps: {len(times)}")
    print(f"  Valid sun positions (above horizon): {len(valid_positions)}")
    
    # Convert to direction vectors
    # Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
    # Elevation: 0° = horizon, 90° = zenith
    azimuth_rad = np.deg2rad(valid_positions['azimuth'].values)
    elevation_rad = np.deg2rad(valid_positions['elevation'].values)
    
    # Convert to 3D direction vectors (pointing toward sun)
    # Standard convention: x=east, y=north, z=up
    # For ray casting from ground: we want direction FROM observer TO sun
    dx = np.sin(azimuth_rad) * np.cos(elevation_rad)  # East component
    dy = np.cos(azimuth_rad) * np.cos(elevation_rad)  # North component
    dz = np.sin(elevation_rad)  # Up component
    
    sun_directions = np.column_stack([dx, dy, dz])
    
    # Normalize (should already be normalized, but ensure)
    norms = np.linalg.norm(sun_directions, axis=1)
    sun_directions = sun_directions / norms[:, np.newaxis]
    
    return sun_directions, valid_positions.index


def compute_solar_access_for_points(
    ground_points: np.ndarray,
    sun_directions: np.ndarray,
    full_mesh: pv.PolyData,
    evaluation_height: float,
    ray_length: float = 10000.0
) -> np.ndarray:
    """
    Compute solar access (hours of direct sunlight) for each ground point.
    
    For each ground point and each sun position:
    1. Cast a ray toward the sun
    2. Test intersection against the STL mesh
    3. If unobstructed → sunlit, else → shaded
    
    Args:
        ground_points: Array of shape (N, 3) with ground point coordinates
        sun_directions: Array of shape (M, 3) with normalized sun direction vectors
        full_mesh: Full scene mesh (terrain + buildings)
        evaluation_height: Height above ground for evaluation (meters)
        ray_length: Maximum ray length for intersection test (meters)
        
    Returns:
        Array of solar access values (number of unobstructed sun positions) for each ground point
    """
    print(f"Computing solar access for {len(ground_points)} ground points...")
    print(f"  Sun positions: {len(sun_directions)}")
    print(f"  Evaluation height: {evaluation_height}m")
    
    # Create observer points (ground points + evaluation height)
    observer_points = ground_points.copy()
    observer_points[:, 2] += evaluation_height
    
    # Compute number of unobstructed sun positions for each point
    solar_access_steps = []
    
    # Progress bar
    pbar = tqdm(total=len(observer_points), desc="Computing solar access", unit="points")
    
    for observer in observer_points:
        unobstructed_count = 0
        
        for sun_dir in sun_directions:
            # Ray end point (far away in sun direction)
            ray_end = observer + sun_dir * ray_length
            
            # Cast ray and check for intersection
            intersection, cell_id = full_mesh.ray_trace(observer, ray_end)
            
            # If no intersection, sun is visible (unobstructed)
            if len(intersection) == 0:
                unobstructed_count += 1
        
        solar_access_steps.append(unobstructed_count)
        
        # Update progress bar
        if len(solar_access_steps) > 0:
            current_mean = np.mean(solar_access_steps)
            pbar.set_postfix({
                'mean_steps': f'{current_mean:.2f}',
                'current': f'{unobstructed_count}'
            })
        
        pbar.update(1)
    
    pbar.close()
    
    return np.array(solar_access_steps)


def generate_solar_maps(
    ground_points: np.ndarray,
    solar_access_steps: np.ndarray,
    grid_x_coords: np.ndarray,
    grid_y_coords: np.ndarray,
    ground_mask: np.ndarray,
    timestep_hours: float,
    threshold_hours: float = 2.0,
    output_dir: Path = None
):
    """
    Generate solar access visualization maps.
    
    Creates:
    1. Solar access heatmap (hours of direct sun)
    2. Threshold map (binary: <threshold red, >=threshold acceptable)
    
    Args:
        ground_points: Array of shape (N, 3) with ground point coordinates (only ground points)
        solar_access_steps: Array of solar access values in time steps (will be converted to hours)
        grid_x_coords: 1D array of X coordinates for full grid
        grid_y_coords: 1D array of Y coordinates for full grid
        ground_mask: Boolean mask for full grid (True = ground, False = building)
        timestep_hours: Time step duration in hours (e.g., 1.0 for 60-minute steps)
        threshold_hours: Threshold for acceptable solar access (default: 2.0 hours)
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating solar access maps...")
    
    # Convert time steps to hours
    solar_hours = solar_access_steps * timestep_hours
    
    # Create 2D grid with NaN for building points
    solar_2d = np.full((len(grid_y_coords), len(grid_x_coords)), np.nan)
    
    # Map ground points back to full grid
    for i, (x, y, z) in enumerate(ground_points):
        x_idx = np.argmin(np.abs(grid_x_coords - x))
        y_idx = np.argmin(np.abs(grid_y_coords - y))
        solar_2d[y_idx, x_idx] = solar_hours[i]
    
    # 1. Solar access heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    
    extent = [
        grid_x_coords.min(), grid_x_coords.max(),
        grid_y_coords.min(), grid_y_coords.max()
    ]
    
    max_hours = np.nanmax(solar_hours) if len(solar_hours) > 0 else 12.0
    
    im = ax.imshow(
        solar_2d,
        extent=extent,
        cmap='YlOrRd',
        vmin=0,
        vmax=max_hours,
        origin='lower',
        interpolation='nearest'
    )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Ground-Level Solar Access (Winter Solstice)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Hours of Direct Sunlight', rotation=270, labelpad=20)
    
    plt.tight_layout()
    heatmap_path = output_dir / "solar_access_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {heatmap_path}")
    
    # 2. Threshold map (binary classification)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create binary map: <threshold = red (deficit), >=threshold = light color (acceptable)
    threshold_2d = np.full((len(grid_y_coords), len(grid_x_coords)), np.nan)
    
    for i, (x, y, z) in enumerate(ground_points):
        x_idx = np.argmin(np.abs(grid_x_coords - x))
        y_idx = np.argmin(np.abs(grid_y_coords - y))
        threshold_2d[y_idx, x_idx] = 1.0 if solar_hours[i] >= threshold_hours else 0.0
    
    # Custom colormap: red for deficit (<2h), light green for acceptable (>=2h)
    colors = ['#d62728', '#90EE90']  # Red for deficit, light green for acceptable
    cmap = ListedColormap(colors)
    
    im = ax.imshow(
        threshold_2d,
        extent=extent,
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin='lower',
        interpolation='nearest'
    )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Solar Access Threshold Map\n(Red: <{threshold_hours}h, Green: ≥{threshold_hours}h)', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    # Custom colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.set_ticklabels([f'<{threshold_hours}h (deficit)', f'≥{threshold_hours}h (acceptable)'])
    cbar.set_label('Solar Access Classification', rotation=270, labelpad=20)
    
    plt.tight_layout()
    threshold_path = output_dir / "solar_access_threshold.png"
    plt.savefig(threshold_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {threshold_path}")
    
    # Print statistics
    print("\nSolar Access Statistics:")
    print(f"  Mean: {np.nanmean(solar_hours):.2f} hours")
    print(f"  Std:  {np.nanstd(solar_hours):.2f} hours")
    print(f"  Min:  {np.nanmin(solar_hours):.2f} hours")
    print(f"  Max:  {np.nanmax(solar_hours):.2f} hours")
    print(f"  Points with <{threshold_hours}h: {np.sum(solar_hours < threshold_hours)} ({np.sum(solar_hours < threshold_hours)/len(solar_hours)*100:.1f}%)")
    print(f"  Points with ≥{threshold_hours}h: {np.sum(solar_hours >= threshold_hours)} ({np.sum(solar_hours >= threshold_hours)/len(solar_hours)*100:.1f}%)")


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description='Compute ground-level solar access from STL file')
    parser.add_argument('--stl', type=str, required=True, help='Path to STL file')
    parser.add_argument('--footprints', type=str, default=None, help='Path to building footprints shapefile (optional)')
    parser.add_argument('--grid-spacing', type=float, default=5.0, help='Grid spacing in meters')
    parser.add_argument('--height', type=float, default=0.5, help='Evaluation height above ground (meters)')
    parser.add_argument('--buffer-distance', type=float, default=0.25, help='Buffer distance for building footprints (meters)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: outputs/solar)')
    parser.add_argument('--latitude', type=float, default=-22.9519, help='Site latitude (degrees, default: Rio de Janeiro)')
    parser.add_argument('--longitude', type=float, default=-43.2105, help='Site longitude (degrees, default: Rio de Janeiro)')
    parser.add_argument('--timestep', type=int, default=60, help='Solar calculation time step in minutes (default: 60)')
    parser.add_argument('--threshold', type=float, default=2.0, help='Solar access threshold in hours (default: 2.0)')
    
    args = parser.parse_args()
    
    # Setup paths
    stl_path = Path(args.stl)
    if not stl_path.exists():
        print(f"Error: STL file not found: {stl_path}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "outputs" / "solar"
    
    print("=" * 60)
    print("SOLAR ACCESS COMPUTATION")
    print("=" * 60)
    print(f"STL file: {stl_path}")
    if args.footprints:
        print(f"Building footprints: {args.footprints}")
    print(f"Grid spacing: {args.grid_spacing}m")
    print(f"Evaluation height: {args.height}m")
    print(f"Location: ({args.latitude:.4f}°, {args.longitude:.4f}°)")
    print(f"Time step: {args.timestep} minutes")
    print(f"Threshold: {args.threshold} hours")
    print("=" * 60)
    
    # Load mesh
    mesh = load_mesh(stl_path)
    
    # Extract terrain surface
    terrain = extract_terrain_surface(mesh)
    
    # Load and prepare building footprints if provided
    building_footprints = None
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
    
    # Generate ground points (with mask applied if available)
    # Reuses the exact same ground mask logic from SVF computation
    if args.footprints and building_footprints is not None:
        ground_points, grid_x_coords, grid_y_coords, original_mask = generate_ground_points(
            terrain, args.grid_spacing, building_footprints=building_footprints, output_dir=output_dir
        )
    else:
        ground_points, grid_x_coords, grid_y_coords, original_mask = generate_ground_points(
            terrain, args.grid_spacing
        )
    
    print(f"\nTotal grid points: {len(original_mask)}")
    print(f"Ground points analyzed: {len(ground_points)}")
    
    # Compute sun positions for winter solstice
    sun_directions, sun_times = compute_sun_positions(
        latitude=args.latitude,
        longitude=args.longitude,
        timestep_minutes=args.timestep
    )
    
    print(f"Sun positions: {len(sun_directions)}")
    
    # Compute solar access for ground points
    solar_access_steps = compute_solar_access_for_points(
        ground_points, sun_directions, mesh, args.height
    )
    
    # Convert timestep to hours
    timestep_hours = args.timestep / 60.0
    
    # Generate solar access maps
    generate_solar_maps(
        ground_points,
        solar_access_steps,
        grid_x_coords,
        grid_y_coords,
        original_mask,
        timestep_hours,
        threshold_hours=args.threshold,
        output_dir=output_dir
    )
    
    # Save solar access as 2D numpy array for use in other analyses
    solar_hours = solar_access_steps * timestep_hours
    solar_2d = np.full((len(grid_y_coords), len(grid_x_coords)), np.nan)
    for i, (x, y, z) in enumerate(ground_points):
        x_idx = np.argmin(np.abs(grid_x_coords - x))
        y_idx = np.argmin(np.abs(grid_y_coords - y))
        solar_2d[y_idx, x_idx] = solar_hours[i]
    
    solar_npy_path = output_dir / "solar_access.npy"
    np.save(solar_npy_path, solar_2d)
    print(f"  Saved solar access raster to {solar_npy_path}")
    
    print("\n" + "=" * 60)
    print("SOLAR ACCESS COMPUTATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    if args.footprints and building_footprints is not None:
        print(f"  Ground mask debug plot: {output_dir / 'ground_mask_debug.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

