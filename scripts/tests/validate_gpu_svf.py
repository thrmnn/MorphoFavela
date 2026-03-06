#!/usr/bin/env python3
"""
Validate GPU SVF results by comparing with CPU results and checking reasonableness.
"""

import geopandas as gpd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def validate_svf_results(gpu_results_path: Path, cpu_results_path: Path = None):
    """Validate GPU SVF results."""
    
    print("=" * 60)
    print("GPU SVF VALIDATION")
    print("=" * 60)
    
    # Load GPU results
    print(f"\n1. Loading GPU results from: {gpu_results_path}")
    gpu_results = gpd.read_file(gpu_results_path)
    print(f"   Loaded {len(gpu_results)} points")
    
    # Check basic statistics
    print("\n2. GPU SVF Statistics:")
    svf = gpu_results['svf'].values
    print(f"   Mean:   {svf.mean():.4f}")
    print(f"   Std:    {svf.std():.4f}")
    print(f"   Min:    {svf.min():.4f}")
    print(f"   Max:    {svf.max():.4f}")
    print(f"   Median: {np.median(svf):.4f}")
    
    # Check for invalid values
    print("\n3. Data Quality Checks:")
    n_invalid = np.sum((svf < 0) | (svf > 1))
    n_nan = np.sum(np.isnan(svf))
    n_zero = np.sum(svf == 0)
    n_one = np.sum(svf == 1)
    
    print(f"   Invalid values (< 0 or > 1): {n_invalid}")
    print(f"   NaN values: {n_nan}")
    print(f"   Zero values: {n_zero} ({n_zero/len(svf)*100:.1f}%)")
    print(f"   One values: {n_one} ({n_one/len(svf)*100:.1f}%)")
    
    if n_invalid > 0:
        print("   ⚠ WARNING: Found invalid SVF values!")
    if n_nan > 0:
        print("   ⚠ WARNING: Found NaN values!")
    if n_zero == len(svf):
        print("   ⚠ WARNING: All values are zero!")
    if n_one == len(svf):
        print("   ⚠ WARNING: All values are one (suspicious for urban areas)!")
    
    # Check distribution
    print("\n4. SVF Distribution:")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    hist, _ = np.histogram(svf, bins=bins)
    for label, count in zip(labels, hist):
        pct = count / len(svf) * 100
        print(f"   {label}: {count:5d} ({pct:5.1f}%)")
    
    # Check reasonableness for urban areas
    print("\n5. Reasonableness Check (for urban areas):")
    mean_svf = svf.mean()
    if mean_svf < 0.3:
        print(f"   ⚠ Mean SVF ({mean_svf:.3f}) is very low - might indicate over-blocking")
    elif mean_svf > 0.9:
        print(f"   ⚠ Mean SVF ({mean_svf:.3f}) is very high - might indicate under-blocking")
    else:
        print(f"   ✓ Mean SVF ({mean_svf:.3f}) is reasonable for urban areas")
    
    if svf.std() < 0.05:
        print(f"   ⚠ Low variance ({svf.std():.3f}) - values might be too uniform")
    else:
        print(f"   ✓ Variance ({svf.std():.3f}) shows good variation")
    
    # Compare with CPU if available
    if cpu_results_path and cpu_results_path.exists():
        print(f"\n6. Comparing with CPU results from: {cpu_results_path}")
        cpu_results = gpd.read_file(cpu_results_path)
        print(f"   CPU points: {len(cpu_results)}")
        
        # Match points by geometry
        from shapely.geometry import Point
        tolerance = 0.1
        
        matched = []
        for idx, gpu_row in gpu_results.iterrows():
            gpu_geom = gpu_row.geometry
            distances = cpu_results.geometry.distance(gpu_geom)
            closest_idx = distances.idxmin()
            if distances[closest_idx] < tolerance:
                matched.append((idx, closest_idx, gpu_row['svf'], cpu_results.loc[closest_idx, 'svf']))
        
        if len(matched) > 0:
            gpu_svf_matched = np.array([m[2] for m in matched])
            cpu_svf_matched = np.array([m[3] for m in matched])
            
            diff = gpu_svf_matched - cpu_svf_matched
            abs_diff = np.abs(diff)
            
            print(f"   Matched points: {len(matched)}")
            print(f"   Mean difference (GPU - CPU): {diff.mean():.4f}")
            print(f"   Mean absolute difference: {abs_diff.mean():.4f}")
            print(f"   Max absolute difference: {abs_diff.max():.4f}")
            
            correlation = np.corrcoef(gpu_svf_matched, cpu_svf_matched)[0, 1]
            print(f"   Correlation: {correlation:.4f}")
            
            if correlation > 0.95:
                print("   ✓ High correlation - GPU results match CPU well")
            elif correlation > 0.8:
                print("   ⚠ Moderate correlation - some differences")
            else:
                print("   ⚠ Low correlation - significant differences")
        else:
            print("   ⚠ No matching points found (different coordinate systems?)")
    else:
        print("\n6. CPU comparison: No CPU results provided")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    
    return gpu_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Validate GPU SVF results')
    parser.add_argument('--gpu-results', type=str, required=True, help='Path to GPU results GeoPackage')
    parser.add_argument('--cpu-results', type=str, default=None, help='Path to CPU results GeoPackage (optional)')
    
    args = parser.parse_args()
    validate_svf_results(Path(args.gpu_results), Path(args.cpu_results) if args.cpu_results else None)
