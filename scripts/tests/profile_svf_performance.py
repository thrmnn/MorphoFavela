#!/usr/bin/env python3
"""
Profile SVF computation to identify bottlenecks.

This script helps identify where time is spent in SVF computation
to guide optimization efforts.
"""

import cProfile
import pstats
import io
import numpy as np
import pyvista as pv
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.svf_utils import load_mesh
from scripts.compute_svf import generate_sky_patches


def profile_svf_computation(stl_path: Path, n_points: int = 10, n_patches: int = 50):
    """
    Profile SVF computation on a small subset.
    
    Args:
        stl_path: Path to STL file
        n_points: Number of test points (small for profiling)
        n_patches: Number of sky patches (small for profiling)
    """
    print(f"Profiling SVF computation...")
    print(f"  Test points: {n_points}")
    print(f"  Sky patches: {n_patches}")
    print(f"  STL: {stl_path}")
    
    # Load mesh
    print("\nLoading mesh...")
    mesh = load_mesh(stl_path)
    print(f"  Mesh: {mesh.n_points} points, {mesh.n_cells} cells")
    
    # Generate test points (random on ground)
    bounds = mesh.bounds
    ground_points = np.array([
        [
            np.random.uniform(bounds[0], bounds[1]),
            np.random.uniform(bounds[2], bounds[3]),
            bounds[4]  # Ground level
        ]
        for _ in range(n_points)
    ])
    
    # Generate sky patches
    print(f"\nGenerating {n_patches} sky patches...")
    sky_patches, _ = generate_sky_patches(n_patches)
    
    # Profile the computation
    print("\nProfiling ray-casting...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Simulate SVF computation
    observer_points = ground_points.copy()
    observer_points[:, 2] += 1.5  # Evaluation height
    
    total_rays = 0
    for observer in observer_points:
        for patch_centroid in sky_patches:
            ray_direction = patch_centroid - observer
            ray_length = np.linalg.norm(ray_direction)
            ray_direction = ray_direction / ray_length
            ray_end = observer + ray_direction * ray_length
            intersection, cell_id = mesh.ray_trace(observer, ray_end)
            total_rays += 1
    
    profiler.disable()
    
    # Print profiling results
    print(f"\nTotal ray casts: {total_rays}")
    print("\n" + "="*60)
    print("PROFILING RESULTS")
    print("="*60)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())
    
    # Save to file
    output_file = PROJECT_ROOT / "outputs" / "svf_profile.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(s.getvalue())
    print(f"\nFull profile saved to: {output_file}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total time: {profiler.total_tt:.2f} seconds")
    print(f"Time per ray: {profiler.total_tt / total_rays * 1000:.2f} ms")
    print(f"Time per point: {profiler.total_tt / n_points:.2f} seconds")
    print(f"\nFor 33,387 points with 300 patches:")
    print(f"  Estimated time: {profiler.total_tt / n_points * 33387 / 3600:.2f} hours")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profile SVF computation")
    parser.add_argument("--stl", type=str, required=True, help="Path to STL file")
    parser.add_argument("--points", type=int, default=10, help="Number of test points")
    parser.add_argument("--patches", type=int, default=50, help="Number of sky patches")
    
    args = parser.parse_args()
    profile_svf_computation(Path(args.stl), args.points, args.patches)
