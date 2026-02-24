#!/usr/bin/env python3
"""
Compare CPU and GPU SVF computation results.

This script runs both CPU and GPU versions and compares the results
to validate GPU implementation accuracy.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
import argparse
import sys
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compare_results(cpu_results_path: Path, gpu_results_path: Path, output_dir: Path):
    """
    Compare CPU and GPU SVF computation results.
    
    Args:
        cpu_results_path: Path to CPU results GeoPackage (points)
        gpu_results_path: Path to GPU results GeoPackage (points)
        output_dir: Output directory for comparison plots
    """
    print("=" * 60)
    print("COMPARING CPU vs GPU SVF RESULTS")
    print("=" * 60)
    
    # Load results
    print(f"Loading CPU results from: {cpu_results_path}")
    cpu_points = gpd.read_file(cpu_results_path)
    print(f"  Loaded {len(cpu_points)} points")
    
    print(f"Loading GPU results from: {gpu_results_path}")
    gpu_points = gpd.read_file(gpu_results_path)
    print(f"  Loaded {len(gpu_points)} points")
    
    # Match points by geometry (within tolerance)
    print("\nMatching points between CPU and GPU results...")
    matched_indices = []
    tolerance = 0.1  # 10cm tolerance
    
    for idx, cpu_point in cpu_points.iterrows():
        cpu_geom = cpu_point.geometry
        distances = gpu_points.geometry.distance(cpu_geom)
        closest_idx = distances.idxmin()
        if distances[closest_idx] < tolerance:
            matched_indices.append((idx, closest_idx))
    
    print(f"  Matched {len(matched_indices)} points")
    
    if len(matched_indices) == 0:
        print("ERROR: No matching points found. Results may be in different coordinate systems.")
        return
    
    # Extract matched SVF values
    cpu_svf = np.array([cpu_points.loc[idx, 'svf'] for idx, _ in matched_indices])
    gpu_svf = np.array([gpu_points.loc[gpu_idx, 'svf'] for _, gpu_idx in matched_indices])
    
    # Compute statistics
    print("\n" + "=" * 60)
    print("COMPARISON STATISTICS")
    print("=" * 60)
    
    diff = cpu_svf - gpu_svf
    abs_diff = np.abs(diff)
    
    print(f"Number of matched points: {len(matched_indices)}")
    print(f"\nCPU SVF:")
    print(f"  Mean: {cpu_svf.mean():.4f}")
    print(f"  Std:  {cpu_svf.std():.4f}")
    print(f"  Min:  {cpu_svf.min():.4f}")
    print(f"  Max:  {cpu_svf.max():.4f}")
    
    print(f"\nGPU SVF:")
    print(f"  Mean: {gpu_svf.mean():.4f}")
    print(f"  Std:  {gpu_svf.std():.4f}")
    print(f"  Min:  {gpu_svf.min():.4f}")
    print(f"  Max:  {gpu_svf.max():.4f}")
    
    print(f"\nDifference (CPU - GPU):")
    print(f"  Mean: {diff.mean():.4f}")
    print(f"  Std:  {diff.std():.4f}")
    print(f"  Min:  {diff.min():.4f}")
    print(f"  Max:  {diff.max():.4f}")
    print(f"  Mean absolute difference: {abs_diff.mean():.4f}")
    print(f"  Max absolute difference: {abs_diff.max():.4f}")
    
    # Correlation
    correlation = np.corrcoef(cpu_svf, gpu_svf)[0, 1]
    print(f"\nCorrelation: {correlation:.4f}")
    
    # Create comparison plots
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(cpu_svf, gpu_svf, alpha=0.5, s=10)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect match')
    ax.set_xlabel('CPU SVF', fontsize=12)
    ax.set_ylabel('GPU SVF', fontsize=12)
    ax.set_title(f'CPU vs GPU SVF Comparison\n(n={len(matched_indices)}, r={correlation:.3f})', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    scatter_path = output_dir / "cpu_gpu_comparison_scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved scatter plot to: {scatter_path}")
    
    # Difference histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(diff, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax.set_xlabel('Difference (CPU - GPU)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'SVF Difference Distribution\n(Mean: {diff.mean():.4f}, Std: {diff.std():.4f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    hist_path = output_dir / "cpu_gpu_difference_histogram.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved difference histogram to: {hist_path}")
    
    # Save comparison CSV
    comparison_df = pd.DataFrame({
        'cpu_svf': cpu_svf,
        'gpu_svf': gpu_svf,
        'difference': diff,
        'abs_difference': abs_diff
    })
    csv_path = output_dir / "cpu_gpu_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV to: {csv_path}")
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare CPU and GPU SVF results')
    parser.add_argument('--cpu-results', type=str, required=True, help='Path to CPU results GeoPackage')
    parser.add_argument('--gpu-results', type=str, required=True, help='Path to GPU results GeoPackage')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for comparison plots')
    
    args = parser.parse_args()
    compare_results(Path(args.cpu_results), Path(args.gpu_results), Path(args.output_dir))
