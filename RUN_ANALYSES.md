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
2. Sky View Factor (SVF)
3. Solar Access
4. Sectional Porosity
5. Occupancy Density
6. Sky Exposure Plane Exceedance
7. Deprivation Index (Raster-based)

## Individual Analysis Commands

See `COPACABANA_ANALYSIS_GUIDE.md` for detailed commands for each analysis type.

## Expected Outputs

Results will be saved to:
- **Vidigal**: `outputs/vidigal/{analysis_type}/`
- **Copacabana**: `outputs/copacabana/{analysis_type}/`

Each analysis creates:
- Data files (`.npy`, `.csv`, `.gpkg`)
- Visualizations (`.png`)

## Troubleshooting

If you get `ModuleNotFoundError`, make sure:
1. Your environment is activated
2. Dependencies are installed: `pip install -r requirements.txt`
3. You're using the correct Python interpreter

