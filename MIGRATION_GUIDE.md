# Migration Guide: Multi-Area Data Organization

**Status**: ✅ Migration complete! This guide is kept for reference only.

All data has been successfully migrated to the area-based structure:
- Vidigal data: `data/vidigal/raw/`
- Copacabana data: `data/copacabana/raw/`

The comparative analysis framework is complete and operational.

## Overview

The project now supports analysis of multiple areas (e.g., Vidigal and Copacabana) for comparative studies. This requires reorganizing data and outputs into area-specific directories.

## Step 1: Data Migration

### Current Structure (Legacy)
```
data/
└── raw/
    ├── full_scan.stl
    ├── vidigal_buildings.shp
    └── ...
```

### New Structure
```
data/
├── vidigal/
│   └── raw/
│       ├── full_scan.stl
│       ├── vidigal_buildings.shp
│       └── ...
└── copacabana/
    └── raw/
        └── ... (place your Copacabana files here)
```

### Migration Steps

1. **Migrate Vidigal data:**
   ```bash
   # Move existing files to the new location
   mv data/raw/* data/vidigal/raw/
   ```

2. **Add Copacabana data:**
   - Place your Copacabana STL mesh in `data/copacabana/raw/`
   - Place your Copacabana building footprints in `data/copacabana/raw/`
   - Follow the naming conventions in `data/copacabana/README.md`

## Step 2: Outputs Migration (Optional)

Existing outputs can remain in `outputs/` (root) for backward compatibility, but for new analyses, outputs will be organized by area:

### New Output Structure
```
outputs/
├── vidigal/
│   ├── buildings_with_metrics.gpkg
│   ├── summary_stats.csv
│   ├── svf/
│   ├── solar/
│   └── ...
└── copacabana/
    └── ... (will be created when analysis is run)
```

### To Migrate Existing Outputs (Optional)
```bash
# Create area-specific output directory
mkdir -p outputs/vidigal

# Move existing outputs (choose one approach):
# Option 1: Move everything
mv outputs/* outputs/vidigal/ 2>/dev/null || true

# Option 2: Move specific subdirectories
mv outputs/svf outputs/vidigal/ 2>/dev/null || true
mv outputs/solar outputs/vidigal/ 2>/dev/null || true
# ... etc for other directories
```

**Note**: You can leave existing outputs where they are - scripts will continue to work. The migration is optional and mainly for organization.

## Step 3: Update Script Usage

### Current Script Calls (Still Work)
Scripts will continue to work with explicit file paths:
```bash
python scripts/compute_svf.py \
    --stl data/vidigal/raw/full_scan.stl \
    --footprints data/vidigal/raw/vidigal_buildings.shp \
    --grid-spacing 5.0
```

### Future: Area-Based Scripts (Planned)
Future versions may support an `--area` parameter:
```bash
python scripts/compute_svf.py \
    --area vidigal \
    --grid-spacing 5.0
```

## Step 4: Verify Setup

1. **Check data structure:**
   ```bash
   ls -la data/vidigal/raw/
   ls -la data/copacabana/raw/
   ```

2. **Test configuration helpers:**
   ```python
   from src.config import get_area_data_dir, get_area_output_dir
   
   # Should work without errors
   vidigal_data = get_area_data_dir("vidigal")
   copacabana_data = get_area_data_dir("copacabana")
   
   print(f"Vidigal data: {vidigal_data}")
   print(f"Copacabana data: {copacabana_data}")
   ```

## File Naming Conventions

### Recommended Naming

**Vidigal:**
- STL: `full_scan.stl` or `vidigal_mesh.stl`
- Buildings: `vidigal_buildings.shp`

**Copacabana:**
- STL: `copacabana_mesh.stl` or `full_scan.stl`
- Buildings: `copacabana_buildings.shp`

## Backward Compatibility

- The legacy `data/raw/` path is still supported in `src/config.py`
- Scripts will continue to work with explicit file paths
- No immediate code changes required - migration is organizational only

## Next Steps

After completing the migration:
1. Add Copacabana data files to `data/copacabana/raw/`
2. Run analyses for both areas
3. Prepare for comparative analysis (Phase 3.1)

