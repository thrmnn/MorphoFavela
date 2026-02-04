# RioDasPedras Setup Complete ✅

## Files Renamed to Match Naming Conventions

All files have been renamed to match the expected naming conventions:

### Before → After
- `Riodaspedras.stl` → `full_scan.stl`
- `Buildings_RP.*` → `riodaspedras_buildings.*`
- `DTM_RP.tif` → `riodaspedras_dtm.tif`
- `Streets_RP.*` → `roads_riodaspedras.*`

### Current File Structure
```
data/riodaspedras/raw/
├── full_scan.stl                    # STL mesh (11 MB)
├── riodaspedras_buildings.shp      # Building footprints
├── riodaspedras_buildings.dbf
├── riodaspedras_buildings.shx
├── riodaspedras_buildings.prj
├── riodaspedras_buildings.cpg
├── riodaspedras_dtm.tif            # Digital terrain model
├── roads_riodaspedras.shp          # Street network
├── roads_riodaspedras.dbf
├── roads_riodaspedras.shx
├── roads_riodaspedras.prj
└── roads_riodaspedras.cpg
```

## Data Validation Results

### ✅ STL File
- **Status**: Valid
- **Points**: 157,524
- **Cells**: 222,919
- **Spatial Extent**: 1,348m × 1,478m
- **Height Range**: 0-193m

### ✅ Building Footprints
- **Status**: Valid
- **Number of Buildings**: 10,729
- **CRS**: EPSG:31983 (UTM Zone 23S - projected)
- **Height Attributes**: ✓ Found `altura`, `base`, `topo` columns
- **Geometry**: Polygon and MultiPolygon (valid)

### ✅ DTM File
- **Status**: Valid
- **CRS**: EPSG:31983 (matches other files)
- **Resolution**: 5.0m × 5.0m
- **Shape**: 291 × 262 pixels

### ✅ Streets File
- **Status**: Valid
- **Number of Segments**: 666
- **CRS**: EPSG:31983 (matches other files)
- **Geometry**: LineString (valid)

### ✅ CRS Consistency
- **All files use**: EPSG:31983 (UTM Zone 23S)
- **Status**: Perfect match - no transformation needed

## Environment Setup

- **Conda Environment**: `/home/theo/miniconda3/envs/IVF`
- **Status**: Activated and ready
- **Validation Script**: Created `validate_riodaspedras_data.py`

## Ready for SVF Computation

All required files are valid and ready for analysis. You can now run:

```bash
# Activate environment
eval "$(conda shell.bash hook)"
conda activate /home/theo/miniconda3/envs/IVF

# Run grid-based SVF
python scripts/compute_svf.py \
    --stl data/riodaspedras/raw/full_scan.stl \
    --footprints data/riodaspedras/raw/riodaspedras_buildings.shp \
    --grid-spacing 5.0 \
    --height 0.5 \
    --sky-patches 145 \
    --output-dir outputs/riodaspedras/svf
```

## Notes

- **Height Attributes**: The building footprints have `altura`, `base`, and `topo` columns. The script will automatically detect and normalize these to `base_height` and `top_height`.
- **CRS**: All files are in EPSG:31983 (UTM Zone 23S), which is perfect for Rio de Janeiro area.
- **Area Classification**: RioDasPedras is configured as an **informal area**, so filtering will be applied automatically (max height 20m, max area 500m², etc.).
