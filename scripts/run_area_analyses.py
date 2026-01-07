#!/usr/bin/env python3
"""
Run all analyses for a specific area.

Usage:
    python scripts/run_area_analyses.py --area vidigal
    python scripts/run_area_analyses.py --area copacabana
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_data_dir, get_area_analysis_dir

def find_data_files(area_data_dir):
    """Find STL and building footprint files in area data directory."""
    stl_files = list(area_data_dir.glob("*.stl"))
    
    # Prefer building footprints over road networks
    building_patterns = ["*building*.shp", "*building*.gpkg", "*building*.geojson",
                        "*footprint*.shp", "*footprint*.gpkg", "*footprint*.geojson",
                        "*vidigal*.shp", "*vidigal*.gpkg", "*vidigal*.geojson",
                        "*copa*.shp", "*copa*.gpkg", "*copa*.geojson"]
    
    footprint_files = []
    for pattern in building_patterns:
        footprint_files.extend(list(area_data_dir.glob(pattern)))
    
    # Filter out road files
    footprint_files = [f for f in footprint_files if 'road' not in f.name.lower()]
    
    # If no building files found, get all shapefiles but prefer non-road
    if not footprint_files:
        all_shp = list(area_data_dir.glob("*.shp")) + \
                  list(area_data_dir.glob("*.gpkg")) + \
                  list(area_data_dir.glob("*.geojson"))
        footprint_files = [f for f in all_shp if 'road' not in f.name.lower()]
    
    stl = stl_files[0] if stl_files else None
    footprints = footprint_files[0] if footprint_files else None
    
    return stl, footprints

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        return False
    print(f"✓ {description} completed successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run all analyses for an area')
    parser.add_argument('--area', type=str, required=True, 
                       choices=['vidigal', 'copacabana'],
                       help='Area name')
    parser.add_argument('--skip-metrics', action='store_true',
                       help='Skip basic metrics calculation')
    parser.add_argument('--grid-spacing', type=float, default=5.0,
                       help='Grid spacing for SVF/solar/porosity (default: 5.0)')
    args = parser.parse_args()
    
    area = args.area
    area_data_dir = get_area_data_dir(area)
    
    print(f"\n{'='*60}")
    print(f"Running all analyses for: {area.upper()}")
    print(f"Data directory: {area_data_dir}")
    print('='*60)
    
    # Find data files
    stl_file, footprints_file = find_data_files(area_data_dir)
    
    if not stl_file:
        print(f"ERROR: No STL file found in {area_data_dir}")
        sys.exit(1)
    if not footprints_file:
        print(f"ERROR: No building footprints found in {area_data_dir}")
        sys.exit(1)
    
    print(f"\nFound data files:")
    print(f"  STL: {stl_file.name}")
    print(f"  Footprints: {footprints_file.name}")
    
    # 1. Basic Metrics
    if not args.skip_metrics:
        cmd = [
            sys.executable, "scripts/calculate_metrics.py",
            "--area", area
        ]
        if not run_command(cmd, "Basic morphometric metrics"):
            print("Warning: Metrics calculation failed, continuing...")
    
    # 2. SVF (grid-based)
    svf_output = get_area_analysis_dir(area, "svf")
    cmd = [
        sys.executable, "scripts/compute_svf.py",
        "--stl", str(stl_file),
        "--footprints", str(footprints_file),
        "--grid-spacing", str(args.grid_spacing),
        "--height", "0.5",
        "--sky-patches", "145",
        "--output-dir", str(svf_output)
    ]
    if not run_command(cmd, "Sky View Factor (SVF - Grid-based)"):
        print("Warning: SVF computation failed, continuing...")
    
    # 2.1. Street-Level SVF (if road network available)
    roads_files = list(area_data_dir.glob("*road*.shp")) + list(area_data_dir.glob("*road*.gpkg"))
    dtm_files = list(area_data_dir.glob("*.tif")) + list(area_data_dir.glob("*.tiff"))
    
    if roads_files:
        roads_file = roads_files[0]
        dtm_file = dtm_files[0] if dtm_files else None
        
        street_svf_output = get_area_analysis_dir(area, "svf_streets")
        cmd = [
            sys.executable, "scripts/compute_svf_streets.py",
            "--stl", str(stl_file),
            "--roads", str(roads_file),
            "--footprints", str(footprints_file),
            "--spacing", "3.0",
            "--height", "1.5",
            "--sky-patches", "145",
            "--area", area,
            "--output-dir", str(street_svf_output)
        ]
        if dtm_file:
            cmd.extend(["--dtm", str(dtm_file)])
        
        if not run_command(cmd, "Street-Level Sky View Factor (SVF)"):
            print("Warning: Street SVF computation failed, continuing...")
    else:
        print(f"\nSkipping Street SVF: No road network file found in {area_data_dir}")
        print("  Looking for files matching: *road*.shp or *road*.gpkg")
    
    # 3. Solar Access
    solar_output = get_area_analysis_dir(area, "solar")
    cmd = [
        sys.executable, "scripts/compute_solar_access.py",
        "--stl", str(stl_file),
        "--footprints", str(footprints_file),
        "--grid-spacing", str(args.grid_spacing),
        "--height", "0.5",
        "--threshold", "3.0",
        "--output-dir", str(solar_output)
    ]
    if not run_command(cmd, "Solar Access"):
        print("Warning: Solar access computation failed, continuing...")
    
    # 4. Sectional Porosity
    porosity_output = get_area_analysis_dir(area, "porosity")
    cmd = [
        sys.executable, "scripts/compute_sectional_porosity.py",
        "--footprints", str(footprints_file),
        "--grid-spacing", "2.0",
        "--height", "1.5",
        "--buffer", "0.25",
        "--output-dir", str(porosity_output)
    ]
    if not run_command(cmd, "Sectional Porosity"):
        print("Warning: Porosity computation failed, continuing...")
    
    # 5. Occupancy Density
    density_output = get_area_analysis_dir(area, "density")
    cmd = [
        sys.executable, "scripts/compute_occupancy_density.py",
        "--stl", str(stl_file),
        "--footprints", str(footprints_file),
        "--grid-size", "50.0",
        "--output-dir", str(density_output)
    ]
    if not run_command(cmd, "Occupancy Density"):
        print("Warning: Density computation failed, continuing...")
    
    # 6. Sky Exposure Plane Exceedance (Unified: Building-level + Street-level if roads available)
    roads_files = list(area_data_dir.glob("*road*.shp")) + list(area_data_dir.glob("*road*.gpkg"))
    
    sky_exposure_output = get_area_analysis_dir(area, "sky_exposure_streets")
    cmd = [
        sys.executable, "scripts/analyze_sky_exposure_streets.py",
        "--stl", str(stl_file),
        "--footprints", str(footprints_file),
        "--ruleset", "rio",
        "--area", area,
        "--output-dir", str(sky_exposure_output)
    ]
    
    if roads_files:
        roads_file = roads_files[0]
        cmd.extend([
            "--roads", str(roads_file),
            "--spacing", "5.0",
            "--search-radius", "75.0"
        ])
        description = "Sky Exposure Plane Exceedance (Building-level + Street-level)"
    else:
        description = "Sky Exposure Plane Exceedance (Building-level only)"
        print(f"\nNo road network found - computing building-level exceedance only")
        print(f"  Looking for files matching: *road*.shp or *road*.gpkg")
    
    if not run_command(cmd, description):
        print("Warning: Sky exposure analysis failed, continuing...")
    
    # 7. Deprivation Index (Raster-based) - requires SVF, solar, porosity
    svf_file = svf_output / "svf.npy"
    solar_file = solar_output / "solar_access.npy"
    porosity_file = porosity_output / "porosity.npy"
    density_file = density_output / "density_proxy.gpkg"
    
    if svf_file.exists() and solar_file.exists() and porosity_file.exists():
        deprivation_output = get_area_analysis_dir(area, "deprivation_raster")
        cmd = [
            sys.executable, "scripts/compute_deprivation_index_raster.py",
            "--solar", str(solar_file),
            "--svf", str(svf_file),
            "--porosity", str(porosity_file),
            "--stl", str(stl_file),
            "--footprints", str(footprints_file),
            "--output-dir", str(deprivation_output)
        ]
        if density_file.exists():
            cmd.extend(["--units", str(density_file)])
        
        if not run_command(cmd, "Deprivation Index (Raster-based)"):
            print("Warning: Deprivation index computation failed")
    else:
        print("\nSkipping Deprivation Index: Required inputs not available")
        print(f"  Need: svf.npy, solar_access.npy, porosity.npy")
    
    print(f"\n{'='*60}")
    print(f"All analyses completed for {area.upper()}")
    print(f"Results saved to: outputs/{area}/")
    print('='*60)

if __name__ == "__main__":
    main()

