# RioDasPedras SVF Analysis - Execution Plan

## ✅ Data Files Verification

### Files Found:
- ✅ **STL Mesh**: `Riodaspedras.stl` (11 MB)
- ✅ **Building Footprints**: `Buildings_RP.shp` (with all components: .dbf, .shx, .prj, .cpg)
- ✅ **DTM**: `DTM_RP.tif` (300 KB)
- ✅ **Streets**: `Streets_RP.shp` (with all components)

### Naming Convention Check:
⚠️ **Note**: Files use `_RP` suffix instead of `riodaspedras_` prefix, but this is **fine** - we'll use the actual file names in commands.

## 📋 Pre-Computation Checklist

### 1. Verify Building Footprints Have Height Attributes
The SVF script uses footprints mainly for masking, but height attributes are needed for other analyses. Check if `Buildings_RP.shp` has:
- `base_height` / `top_height` OR
- `base` / `altura` OR
- Any height-related columns

**Action**: We'll verify this during the first run.

### 2. Verify CRS Consistency
All files should be in the same projected CRS (UTM recommended).

**Action**: Script will handle CRS transformation if needed.

### 3. Verify File Readability
All files should be readable and valid.

**Action**: Script will report errors if files are invalid.

## 🚀 Execution Plan

### Phase 1: Grid-Based SVF Analysis (Primary)

**Command:**
```bash
python scripts/compute_svf.py \
    --stl data/riodaspedras/raw/full_scan.stl \
    --footprints data/riodaspedras/raw/riodaspedras_buildings.shp \
    --grid-spacing 10.0 \
    --height 0.5 \
    --sky-patches 145 \
    --building-buffer 30.0 \
    --area riodaspedras \
    --output-dir outputs/riodaspedras/svf
```

**Parameters:**
- `--grid-spacing 10.0`: 10-meter grid spacing (optimized for speed, ~4,200 points)
- `--height 0.5`: Evaluation at 0.5m above ground
- `--sky-patches 145`: Standard sky discretization
- `--output-dir`: Saves to area-specific directory

**Expected Output:**
- `outputs/riodaspedras/svf/svf.npy` - SVF raster array
- `outputs/riodaspedras/svf/svf.csv` - Point coordinates and values
- `outputs/riodaspedras/svf/svf_heatmap.png` - Visualization map
- `outputs/riodaspedras/svf/svf_histogram.png` - Distribution plot

**Estimated Runtime:** With 10.0m spacing (~4,200 points): ~5-15 minutes typical

### Phase 2: Street-Level SVF Analysis (Optional - If Needed)

**Command:**
```bash
python scripts/compute_svf_streets.py \
    --stl data/riodaspedras/raw/Riodaspedras.stl \
    --roads data/riodaspedras/raw/Streets_RP.shp \
    --dtm data/riodaspedras/raw/DTM_RP.tif \
    --footprints data/riodaspedras/raw/Buildings_RP.shp \
    --spacing 3.0 \
    --height 1.5 \
    --output-dir outputs/riodaspedras/svf_streets
```

**Parameters:**
- `--spacing 3.0`: Sample points every 3 meters along streets
- `--height 1.5`: Pedestrian eye level (1.5m above ground)
- `--dtm`: Required for accurate ground elevation at street points

**Expected Output:**
- `outputs/riodaspedras/svf_streets/svf_points.gpkg` - Point-level SVF
- `outputs/riodaspedras/svf_streets/svf_segments.gpkg` - Segment-level aggregated SVF
- `outputs/riodaspedras/svf_streets/svf_streets_map.png` - Street-colored visualization
- `outputs/riodaspedras/svf_streets/svf_streets_stats.csv` - Statistics

## ⚙️ Parameter Recommendations

### For Faster Computation (First Test):
```bash
--grid-spacing 10.0 --sky-patches 145
```
- Coarser grid, faster computation
- Good for initial verification

### For Balanced Speed/Accuracy (Recommended):
```bash
--grid-spacing 5.0 --sky-patches 145
```
- Standard resolution
- Good balance

### For Higher Accuracy (If Needed):
```bash
--grid-spacing 2.0 --sky-patches 290
```
- Finer grid, more sky patches
- Slower but more accurate

## 🔍 Troubleshooting Plan

### If Building Footprints Error:
- Check if height columns exist (script will auto-detect)
- Verify all shapefile components are present
- Check CRS matches STL file

### If STL File Error:
- Verify file is valid STL format
- Check file size (11 MB seems reasonable)
- Ensure file is not corrupted

### If CRS Mismatch:
- Script should handle automatic transformation
- If issues persist, manually reproject to UTM Zone 23S (EPSG:32723)

### If Memory Issues:
- Increase `--grid-spacing` (use 10.0 instead of 5.0)
- Reduce `--sky-patches` (use 145 instead of 290)

## 📊 Success Criteria

After running, verify:
1. ✅ No errors in console output
2. ✅ Output directory contains all expected files
3. ✅ Heatmap visualization looks reasonable
4. ✅ SVF values are in range [0, 1]
5. ✅ Statistics printed to console

## 🎯 Recommended Execution Order

1. **First**: Run grid-based SVF with default parameters (5.0m spacing)
2. **Verify**: Check output files and visualizations
3. **Optional**: Run street-level SVF if pedestrian-level analysis needed
4. **Next Steps**: After SVF, can proceed to solar access, porosity, etc.

## 📝 Notes

- RioDasPedras is classified as **informal area** - filtering will be applied automatically
- All outputs will be saved in `outputs/riodaspedras/` directory
- File naming convention (`_RP` suffix) is acceptable - no renaming needed
- Scripts will handle CRS transformation automatically if needed
