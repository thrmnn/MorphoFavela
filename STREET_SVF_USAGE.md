# Street-Level SVF Computation: Usage Guide

## Overview

The street-level SVF computation complements the grid-based SVF analysis by computing Sky View Factor specifically along street centerlines. This provides pedestrian-focused environmental analysis of street networks.

## Quick Start

### Basic Usage

```bash
python scripts/compute_svf_streets.py \
    --stl data/vidigal/raw/full_scan.stl \
    --roads data/vidigal/raw/roads_vidigal.shp \
    --spacing 3.0 \
    --height 1.5
```

### With DTM for Accurate Elevation

```bash
python scripts/compute_svf_streets.py \
    --stl data/vidigal/raw/full_scan.stl \
    --roads data/vidigal/raw/roads_vidigal.shp \
    --dtm data/vidigal/raw/vidigal_dtm_cropped.tif \
    --footprints data/vidigal/raw/vidigal_buildings.shp \
    --spacing 3.0 \
    --height 1.5 \
    --output-dir outputs/vidigal/svf_streets
```

### Using Area Parameter (Automatic Path Resolution)

```bash
python scripts/compute_svf_streets.py \
    --stl data/vidigal/raw/full_scan.stl \
    --roads data/vidigal/raw/roads_vidigal.shp \
    --dtm data/vidigal/raw/vidigal_dtm_cropped.tif \
    --footprints data/vidigal/raw/vidigal_buildings.shp \
    --area vidigal \
    --spacing 3.0 \
    --height 1.5
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--stl` | **Required** | Path to STL mesh file |
| `--roads` | **Required** | Path to road network shapefile (LineString geometries) |
| `--footprints` | None | Path to building footprints (optional, for visualization) |
| `--dtm` | None | Path to DTM raster (optional, preferred for accurate elevation) |
| `--spacing` | 3.0 | Distance between sample points along streets (meters) |
| `--height` | 1.5 | Evaluation height above ground (meters, pedestrian eye level) |
| `--sky-patches` | 145 | Number of sky patches for SVF computation |
| `--output-dir` | `outputs/svf_streets` | Output directory |
| `--area` | None | Area name (vidigal/copacabana) for automatic path resolution |

## Output Files

### Point-Level Results
- **`street_svf_points.gpkg`**: GeoPackage with sample points along streets
  - Each point has: `segment_idx`, `distance_along`, `street_name`, `svf`, `geometry`

### Segment-Level Results
- **`street_svf_segments.gpkg`**: GeoPackage with aggregated statistics per street segment
  - Columns: `street_name`, `length`, `svf_mean`, `svf_min`, `svf_max`, `svf_std`, `svf_median`
  
### Statistics
- **`street_svf_statistics.csv`**: Tabular summary of segment statistics

### Visualizations
- **`street_svf_map.png`**: Map showing streets colored by mean SVF
- **`street_svf_distribution.png`**: Histogram of SVF values
- **`street_svf_by_segment.png`**: Box plot by street segment (if ≤20 segments)

## Example Output

```
============================================================
STREET-LEVEL SKY VIEW FACTOR (SVF) COMPUTATION
============================================================
STL file: data/vidigal/raw/full_scan.stl
Road network: data/vidigal/raw/roads_vidigal.shp
Point spacing: 3.0m
Evaluation height: 1.5m (pedestrian eye level)
Sky patches: 145
DTM raster: data/vidigal/raw/vidigal_dtm_cropped.tif
Output directory: outputs/vidigal/svf_streets
============================================================

Sampling points along streets with 3.0m spacing...
  Processing streets: 100%|████████████| 58/58 [00:02<00:00]
  Generated 635 sample points

Computing SVF for 635 street points...
  Evaluation height: 1.5m
  Sky patches: 145
Computing SVF: 100%|████████████| 635/635 [02:15<00:00]

============================================================
STREET SVF SUMMARY STATISTICS
============================================================
Total street segments: 58
Total sample points: 635

Point-level SVF:
  Mean: 0.4523
  Std:  0.2156
  Min:  0.0892
  Max:  0.9124
  Median: 0.4231

Segment-level SVF (mean values):
  Mean: 0.4518
  Min:  0.1234
  Max:  0.8756
============================================================
```

## Use Cases

### 1. Identify Streets with Poor Sky Access
Streets with low SVF (< 0.3) indicate constrained sky access, potentially affecting:
- Pedestrian comfort
- Natural lighting
- Air quality perception

### 2. Compare Street Hierarchies
Compare main streets vs. secondary streets/alleyways:
```python
import geopandas as gpd
segments = gpd.read_file('outputs/vidigal/svf_streets/street_svf_segments.gpkg')

main_streets = segments[segments['hierarchy'] == 'Main']
secondary = segments[segments['hierarchy'] == 'Secondary']

print(f"Main streets mean SVF: {main_streets['svf_mean'].mean():.3f}")
print(f"Secondary streets mean SVF: {secondary['svf_mean'].mean():.3f}")
```

### 3. Find Problematic Segments
```python
problematic = segments[segments['svf_mean'] < 0.25]
print(f"Found {len(problematic)} segments with very low SVF")
print(problematic[['street_name', 'svf_mean', 'length']])
```

## Tips

1. **Sampling Density**: 
   - Use 3m spacing for detailed analysis
   - Use 5m spacing for faster computation on large networks

2. **Evaluation Height**:
   - 1.5m: Adult pedestrian eye level (recommended)
   - 0.5m: Ground level perspective

3. **DTM vs. Mesh**:
   - Always use DTM if available (more accurate elevation)
   - Mesh interpolation is a good fallback

4. **Coordinate Systems**:
   - Script automatically handles coordinate transformation
   - Roads in UTM will be transformed to match STL local coordinates

## Integration with Grid-Based SVF

This analysis complements (does not replace) the grid-based SVF:
- **Grid-based SVF**: Area-wide coverage, regular sampling
- **Street-based SVF**: Focused on pedestrian routes, irregular sampling

Both analyses can be used together for comprehensive environmental assessment.

