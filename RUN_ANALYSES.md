# Running Analyses - Quick Start

## Setup

Before running analyses, activate your Python environment:

```bash
# Option 1: Conda environment
conda activate IVF

# Option 2: Virtual environment
source venv/bin/activate

# Verify dependencies
pip install -r requirements.txt
```

## Run All Analyses (Recommended)

The easiest way is to use the batch script that runs all analyses for an area:

```bash
# Vidigal
python scripts/run_area_analyses.py --area vidigal

# Copacabana  
python scripts/run_area_analyses.py --area copacabana
```

This will run all analyses in the correct order:
1. Basic morphometric metrics
2. Sky View Factor (SVF) - Grid-based
3. Street-Level Sky View Factor (SVF) - if road network available
4. Solar Access
5. Sectional Porosity
6. Occupancy Density
7. Sky Exposure Plane Exceedance
8. Deprivation Index (Raster-based)

## Individual Analysis Commands

See area-specific guides or the main README.md for detailed commands for each analysis type.

## Comparative Analysis

After running analyses for both areas, generate comparison report:

```bash
python scripts/compare_areas.py
```

This generates:
- Comprehensive PDF report: `outputs/comparative/comparison_report.pdf`
- Comparison tables: `outputs/comparative/tables/`
- Side-by-side visualizations: `outputs/comparative/visualizations/`

## Expected Outputs

Results will be saved to:
- **Vidigal**: `outputs/vidigal/{analysis_type}/`
- **Copacabana**: `outputs/copacabana/{analysis_type}/`

Each analysis creates:
- Data files (`.npy`, `.csv`, `.gpkg`)
- Visualizations (`.png`)

**Street-Level SVF** requires:
- Road network shapefile (`*road*.shp` or `*road*.gpkg`) in area data directory
- Optional: DTM raster for accurate elevation extraction
- Automatically detected and run if available

## Troubleshooting

If you get `ModuleNotFoundError`, make sure:
1. Your environment is activated
2. Dependencies are installed: `pip install -r requirements.txt`
3. You're using the correct Python interpreter

