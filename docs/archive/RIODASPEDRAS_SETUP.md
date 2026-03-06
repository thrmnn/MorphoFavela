# RioDasPedras SVF Analysis Setup Guide

## ✅ Step 1: Folder Structure Created

The folder structure has been created:
```
data/riodaspedras/raw/     # Place your input files here
outputs/riodaspedras/      # SVF results will be saved here
```

## 📋 Step 2: Prepare Your Data Files

### Required Files (Minimum for Grid-Based SVF)

1. **STL Mesh File**
   - Place in: `data/riodaspedras/raw/`
   - Recommended name: `full_scan.stl` or `riodaspedras_mesh.stl`
   - Must contain: Combined 3D scene with terrain + buildings
   - CRS: Projected coordinate system (UTM Zone 23S recommended: EPSG:32723)

2. **Building Footprints Shapefile**
   - Place in: `data/riodaspedras/raw/`
   - Recommended name: `riodaspedras_buildings.shp`
   - Must include all shapefile components: `.shp`, `.dbf`, `.shx`, `.prj`
   - Required attributes:
     - `base_height` or `base`: Building base elevation (meters)
     - `top_height` or `altura`: Building top elevation (meters)
   - CRS: Must match STL file (projected, UTM recommended)

### Optional Files (For Street-Level SVF)

3. **Digital Terrain Model (DTM)**
   - Place in: `data/riodaspedras/raw/`
   - Recommended name: `riodaspedras_dtm.tif`
   - Format: GeoTIFF raster
   - Use: Required for street-level SVF analysis

4. **Road Network Shapefile**
   - Place in: `data/riodaspedras/raw/`
   - Recommended name: `roads_riodaspedras.shp`
   - Geometry: LineString (street centerlines)
   - Use: Required for street-level SVF analysis

## 🔧 Step 3: Verify Data Files

Before running the analysis, verify your files:

```bash
# Check if files exist
ls -lh data/riodaspedras/raw/

# Expected files:
# - full_scan.stl (or riodaspedras_mesh.stl)
# - riodaspedras_buildings.shp
# - riodaspedras_buildings.dbf
# - riodaspedras_buildings.shx
# - riodaspedras_buildings.prj
```

## 🚀 Step 4: Run Grid-Based SVF Analysis

### Basic Command (Minimum Required)

```bash
python scripts/compute_svf.py \
    --stl data/riodaspedras/raw/full_scan.stl \
    --footprints data/riodaspedras/raw/riodaspedras_buildings.shp \
    --grid-spacing 5.0 \
    --height 0.5 \
    --sky-patches 145 \
    --output-dir outputs/riodaspedras/svf
```

### Parameter Explanation

- `--stl`: Path to your STL mesh file (REQUIRED)
- `--footprints`: Path to building footprints shapefile (REQUIRED)
- `--grid-spacing`: Grid spacing in meters (default: 2.0, recommended: 5.0 for faster computation)
- `--height`: Evaluation height above ground in meters (default: 0.5)
- `--sky-patches`: Number of sky patches for discretization (default: 145, higher = more accurate but slower)
- `--output-dir`: Output directory (default: `outputs/svf`, recommended: `outputs/riodaspedras/svf`)

### Recommended Settings for First Run

```bash
# Faster computation (coarser grid, fewer patches)
python scripts/compute_svf.py \
    --stl data/riodaspedras/raw/full_scan.stl \
    --footprints data/riodaspedras/raw/riodaspedras_buildings.shp \
    --grid-spacing 10.0 \
    --height 0.5 \
    --sky-patches 145 \
    --output-dir outputs/riodaspedras/svf

# Higher resolution (finer grid, more patches) - slower but more accurate
python scripts/compute_svf.py \
    --stl data/riodaspedras/raw/full_scan.stl \
    --footprints data/riodaspedras/raw/riodaspedras_buildings.shp \
    --grid-spacing 2.0 \
    --height 0.5 \
    --sky-patches 290 \
    --output-dir outputs/riodaspedras/svf
```

## 🛣️ Step 5: Run Street-Level SVF Analysis (Optional)

If you have road network and DTM files:

```bash
python scripts/compute_svf_streets.py \
    --stl data/riodaspedras/raw/full_scan.stl \
    --roads data/riodaspedras/raw/roads_riodaspedras.shp \
    --dtm data/riodaspedras/raw/riodaspedras_dtm.tif \
    --footprints data/riodaspedras/raw/riodaspedras_buildings.shp \
    --spacing 3.0 \
    --height 1.5 \
    --output-dir outputs/riodaspedras/svf_streets
```

### Street-Level Parameters

- `--stl`: STL mesh file (REQUIRED)
- `--roads`: Road network shapefile (REQUIRED for street-level)
- `--dtm`: Digital terrain model raster (REQUIRED for street-level)
- `--footprints`: Building footprints (REQUIRED)
- `--spacing`: Point spacing along streets in meters (default: 3.0)
- `--height`: Pedestrian eye level height in meters (default: 1.5)
- `--output-dir`: Output directory (recommended: `outputs/riodaspedras/svf_streets`)

## 📊 Step 6: Check Results

After running the analysis, check the output directory:

```bash
ls -lh outputs/riodaspedras/svf/
```

### Expected Output Files

1. **`svf.npy`**: NumPy array with SVF values (2D grid)
2. **`svf.csv`**: CSV file with point coordinates and SVF values
3. **`svf_heatmap.png`**: Visualization heatmap of SVF values
4. **`svf_histogram.png`**: Distribution histogram of SVF values

## ⚠️ Troubleshooting

### Common Issues

1. **File Not Found Error**
   - Verify file paths are correct
   - Check that all shapefile components (`.shp`, `.dbf`, `.shx`, `.prj`) are present
   - Ensure files are in `data/riodaspedras/raw/` directory

2. **CRS Mismatch Error**
   - Ensure STL and footprints are in the same CRS
   - Convert to UTM if needed (EPSG:32723 for Rio de Janeiro)

3. **Missing Height Attributes**
   - Check that building footprints have `base_height`/`top_height` or `base`/`altura` columns
   - The script will auto-detect these column names

4. **Memory Error (Large Files)**
   - Reduce `--grid-spacing` (use larger value like 10.0)
   - Reduce `--sky-patches` (use 145 instead of 290)
   - Process in smaller chunks if needed

### Verify Data Before Running

```python
# Quick Python check
import geopandas as gpd
import pyvista as pv

# Check building footprints
gdf = gpd.read_file('data/riodaspedras/raw/riodaspedras_buildings.shp')
print(f"CRS: {gdf.crs}")
print(f"Columns: {gdf.columns.tolist()}")
print(f"Number of buildings: {len(gdf)}")

# Check STL file
mesh = pv.read('data/riodaspedras/raw/full_scan.stl')
print(f"STL points: {mesh.n_points}")
print(f"STL cells: {mesh.n_cells}")
```

## 📝 Notes

- **RioDasPedras is classified as an INFORMAL area**, so filtering will be automatically applied (max height 20m, max area 500m², etc.)
- The configuration has been updated in `src/config.py` to include RioDasPedras
- Outputs will be saved in `outputs/riodaspedras/` directory
- For comparative analysis with other areas, use `scripts/compare_areas.py` (after running analyses for all areas)

## 🎯 Quick Start Summary

1. ✅ Folder structure created
2. ⏳ Place STL file: `data/riodaspedras/raw/full_scan.stl`
3. ⏳ Place building footprints: `data/riodaspedras/raw/riodaspedras_buildings.shp`
4. ⏳ Run: `python scripts/compute_svf.py --stl data/riodaspedras/raw/full_scan.stl --footprints data/riodaspedras/raw/riodaspedras_buildings.shp --grid-spacing 5.0 --output-dir outputs/riodaspedras/svf`
5. ⏳ Check results: `outputs/riodaspedras/svf/`
