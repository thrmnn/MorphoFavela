# Area Analysis Guide

This guide covers running analyses for individual areas (Vidigal or Copacabana).

## Prerequisites

Before running analyses, ensure your Python environment is activated and dependencies are installed:

```bash
# Activate conda environment (if using conda)
conda activate IVF

# Or activate virtual environment (if using venv)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify dependencies are installed
pip install -r requirements.txt
```

## Data Files

### Vidigal (Informal Settlement)
- **STL mesh**: `data/vidigal/raw/full_scan.stl`
- **Building footprints**: `data/vidigal/raw/vidigal_buildings.shp`

### Copacabana (Formal Neighborhood)
- **STL mesh**: `data/copacabana/raw/copa2/copa2.stl`
- **Building footprints**: `data/copacabana/raw/copa2/copa_buildings2_processed.gpkg`
- **Road network**: `data/copacabana/raw/copa2/copa_streets2.shp`
- **DTM**: `data/copacabana/raw/copa2/DTM_copa2.tif`

## Quick Start: Run All Analyses for an Area

The easiest way to run all analyses is using the batch script:

```bash
# Run all analyses for Vidigal
python scripts/run_area_analyses.py --area vidigal

# Run all analyses for Copacabana
python scripts/run_area_analyses.py --area copacabana

# With custom grid spacing
python scripts/run_area_analyses.py --area vidigal --grid-spacing 3.0
```

## Running Individual Analyses

### For Vidigal

### 1. Basic Morphometric Metrics

```bash
# Using area parameter (recommended)
python scripts/calculate_metrics.py --area vidigal
python scripts/calculate_metrics.py --area copacabana

# Or with explicit paths
python scripts/calculate_metrics.py \
    --input data/vidigal/raw/vidigal_buildings.shp \
    --output outputs/vidigal/metrics/
```

### 2. Sky View Factor (SVF)

```bash
python scripts/compute_svf.py \
    --stl data/copacabana/raw/copa2/copa2.stl \
    --footprints data/copacabana/raw/copa2/copa_buildings2_processed.gpkg \
    --grid-spacing 5.0 \
    --height 0.5 \
    --sky-patches 145 \
    --output-dir outputs/copacabana/svf/
```

### 3. Solar Access

```bash
python scripts/compute_solar_access.py \
    --stl data/copacabana/raw/copa2/copa2.stl \
    --footprints data/copacabana/raw/copa2/copa_buildings2_processed.gpkg \
    --grid-spacing 5.0 \
    --height 0.5 \
    --threshold 3.0 \
    --output-dir outputs/copacabana/solar/
```

### 4. Sky Exposure Plane Exceedance

```bash
python scripts/analyze_sky_exposure_streets.py \
    --stl data/copacabana/raw/copa2/copa2.stl \
    --roads data/copacabana/raw/copa2/copa_streets2.shp \
    --footprints data/copacabana/raw/copa2/copa_buildings2_processed.gpkg \
    --ruleset rio \
    --area copacabana \
    --spacing 3.0
```

### 5. Sectional Porosity

```bash
python scripts/compute_sectional_porosity.py \
    --footprints data/copacabana/raw/copa2/copa_buildings2_processed.gpkg \
    --grid-spacing 2.0 \
    --height 1.5 \
    --buffer 0.25 \
    --output-dir outputs/copacabana/porosity/
```

### 6. Occupancy Density Proxy

```bash
python scripts/compute_occupancy_density.py \
    --stl data/copacabana/raw/copa2/copa2.stl \
    --footprints data/copacabana/raw/copa2/copa_buildings2_processed.gpkg \
    --grid-size 50.0 \
    --output-dir outputs/copacabana/density/
```

### 7. Deprivation Index (Unit-level)

**Note**: Requires outputs from SVF, solar, porosity, and density analyses first.

```bash
python scripts/compute_deprivation_index.py \
    --units outputs/copacabana/density/density_proxy.gpkg \
    --solar outputs/copacabana/solar/solar_access.npy \
    --svf outputs/copacabana/svf/svf.npy \
    --porosity outputs/copacabana/porosity/porosity.npy \
    --density outputs/copacabana/density/density_proxy.gpkg \
    --output-dir outputs/copacabana/deprivation/
```

### 8. Deprivation Index (Raster-based)

**Note**: Requires outputs from SVF, solar, and porosity analyses first.

```bash
python scripts/compute_deprivation_index_raster.py \
    --solar outputs/copacabana/solar/solar_access.npy \
    --svf outputs/copacabana/svf/svf.npy \
    --porosity outputs/copacabana/porosity/porosity.npy \
    --stl data/copacabana/raw/copa2/copa2.stl \
    --footprints data/copacabana/raw/copa2/copa_buildings2_processed.gpkg \
    --units outputs/copacabana/density/density_proxy.gpkg \
    --output-dir outputs/copacabana/deprivation_raster/
```

## Output Organization

All Copacabana results will be saved to:
```
outputs/copacabana/
├── metrics/
├── svf/
├── solar/
├── sky_exposure/
├── porosity/
├── density/
├── deprivation/
└── deprivation_raster/
```

## Recommended Analysis Order

1. **Basic Metrics** → Quick overview of building characteristics
2. **SVF & Solar** → Can be run in parallel (independent)
3. **Sectional Porosity** → Independent analysis
4. **Occupancy Density** → Requires STL and footprints
5. **Sky Exposure** → Independent analysis
6. **Deprivation Index** → Requires outputs from steps 2, 3, 4

## Verification

After running analyses, verify outputs:

```bash
# Check that output directories were created
ls -la outputs/copacabana/

# Check for key output files
ls outputs/copacabana/svf/*.npy
ls outputs/copacabana/solar/*.npy
ls outputs/copacabana/porosity/*.npy
```

## Next Steps

Once both Vidigal and Copacabana analyses are complete:
- Comparative analysis scripts (Phase 3.1) will enable side-by-side comparisons
- Statistical comparisons between formal and informal settlements
- Visual comparison dashboards

