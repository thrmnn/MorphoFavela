# RioDasPedras Street-Level SVF Implementation

## Summary
This update adds street-level Sky View Factor (SVF) computation for RioDasPedras with improved filtering and alignment features.

## Key Changes

### 1. Street-Level SVF Script Enhancements (`scripts/compute_svf_streets.py`)

#### Added Features:
- **Debug visualization**: New `plot_street_debug()` function that visualizes:
  - Street centerlines (blue = filtered, gray dotted = filtered-out)
  - Sample points along streets (green)
  - Building footprints (red = main cluster, yellow = area-filtered, gray = isolated)
  - Building buffer zone (orange dashed line)
  - Comprehensive statistics

- **Street filtering by building proximity**:
  - New `--building-buffer` parameter (default: 30.0m)
  - Filters streets to only include those with ≥70% of their length within the buffer zone
  - Significantly reduces computation time by focusing on relevant streets
  - Visual feedback in debug plot showing filtered vs. filtered-out streets

- **Coordinate alignment fix**:
  - Fixed misalignment between streets and buildings
  - Both datasets now use the same transformation reference (original building footprints center)
  - Ensures perfect alignment in visualizations

- **`--debug-only` flag**: Allows generating debug visualization without full SVF computation

#### Filtering Logic:
- Requires at least 70% of each street segment to be within the building buffer zone
- More conservative than simple intersection check
- Prevents inclusion of streets that barely touch the buffer zone

### 2. Building Footprints Transformation Fix (`src/svf_utils.py`)

- **Fixed coordinate transformation**: Now uses original (unfiltered) building footprints center for transformation calculation
- Ensures consistent alignment across all datasets (buildings, streets, terrain)
- Transformation is calculated before filtering, then applied to all filtered subsets

### 3. Configuration Updates (`src/config.py`)

- No changes to config (already includes `riodaspedras` in supported areas)

## Results for RioDasPedras

### Street Filtering:
- **Original streets**: 666 segments
- **Filtered streets**: 627 segments (94.1% kept)
- **Filtered out**: 39 segments (5.9% removed)
- **Criterion**: ≥70% of street length within 30m building buffer

### SVF Computation:
- **Point spacing**: 3.0m along streets
- **Evaluation height**: 1.5m (pedestrian eye level)
- **Sky patches**: 145
- **Sample points**: Generated along filtered streets only

## Output Files

All outputs saved to `outputs/riodaspedras/svf_streets/`:
- `street_svf_points.gpkg` - Point-level SVF values
- `street_svf_segments.gpkg` - Segment-level aggregated statistics
- `street_svf_statistics.csv` - Summary statistics
- `street_svf_map.png` - Map visualization colored by SVF
- `street_svf_distribution.png` - Histogram of SVF values
- `street_debug.png` - Debug plot showing filtering and alignment

## Usage

```bash
# Full computation with filtering
python scripts/compute_svf_streets.py \
    --stl data/riodaspedras/raw/full_scan.stl \
    --roads data/riodaspedras/raw/roads_riodaspedras.shp \
    --dtm data/riodaspedras/raw/riodaspedras_dtm.tif \
    --footprints data/riodaspedras/raw/riodaspedras_buildings.shp \
    --spacing 3.0 \
    --height 1.5 \
    --sky-patches 145 \
    --area riodaspedras \
    --output-dir outputs/riodaspedras/svf_streets \
    --building-buffer 30.0

# Debug visualization only
python scripts/compute_svf_streets.py \
    --stl data/riodaspedras/raw/full_scan.stl \
    --roads data/riodaspedras/raw/roads_riodaspedras.shp \
    --footprints data/riodaspedras/raw/riodaspedras_buildings.shp \
    --area riodaspedras \
    --output-dir outputs/riodaspedras/svf_streets \
    --building-buffer 30.0 \
    --debug-only
```

## Technical Details

### Street Filtering Algorithm:
1. Create buffer zone around main cluster buildings (30m default)
2. For each street segment:
   - Sample points every 5m along the segment
   - Count how many sample points are within the buffer
   - Calculate percentage of segment within buffer
   - Keep segment if ≥70% within buffer

### Coordinate Transformation:
- Both streets and buildings use the same reference point (original building footprints center)
- Transformation calculated once and applied consistently
- Ensures perfect alignment in all visualizations

## Files Modified

- `scripts/compute_svf_streets.py` - Added debug visualization, street filtering, alignment fixes
- `src/svf_utils.py` - Fixed coordinate transformation to use original footprints center
- `scripts/compute_svf.py` - No changes (already had building buffer feature)

## Files Created

- `RIODASPEDRAS_STREET_SVF_CHANGES.md` - This documentation file

## Files Removed

- `validate_riodaspedras_data.py` - Temporary validation script (no longer needed)
