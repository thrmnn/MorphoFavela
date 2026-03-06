#!/usr/bin/env python3
"""
Test GPU SVF computation on a small subset of rocinha streets.
"""

import numpy as np
import geopandas as gpd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_svf_streets_gpu import main
import argparse

if __name__ == "__main__":
    # Test with a small subset - use first 10 street segments
    print("Testing GPU SVF on small subset of rocinha...")
    print("=" * 60)
    
    # Load roads and take first 10
    roads_path = Path("data/rocinha/raw/roads_rocinha.shp")
    roads_gdf = gpd.read_file(roads_path)
    print(f"Total roads: {len(roads_gdf)}")
    
    # Take first 10 segments
    test_roads = roads_gdf.head(10).copy()
    test_roads_path = Path("data/rocinha/raw/roads_rocinha_test.shp")
    test_roads.to_file(test_roads_path)
    print(f"Testing with {len(test_roads)} segments")
    print(f"Saved test roads to: {test_roads_path}")
    
    # Run GPU computation
    sys.argv = [
        "compute_svf_streets_gpu.py",
        "--stl", "data/rocinha/rocinha.stl",
        "--roads", str(test_roads_path),
        "--footprints", "data/rocinha/raw/rocinha_buildings.shp",
        "--output-dir", "outputs/rocinha/svf_streets_gpu_test",
        "--spacing", "3.0",
        "--height", "1.5",
        "--sky-patches", "145",
        "--use-gpu",
        "--gpu-batch-size", "50",
        "--gpu-samples-per-ray", "20"
    ]
    
    main()
