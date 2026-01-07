# Computing Sky View Factor Along Streets: Discussion & Recommendations

## Current Situation

**Road Network Data:**
- **File**: `data/vidigal/raw/roads_vidigal.shp`
- **Type**: LineString geometries (58 road segments)
- **Total length**: ~1,905 meters
- **CRS**: EPSG:31983 (UTM Zone 23S)
- **Attributes**: Includes street names, hierarchy, codes

**Current SVF Implementation:**
- Computes SVF on a regular **grid** of ground points
- Uses ray-casting with discretized sky patches
- Evaluation height: 0.5m above ground (default)
- Excludes building interiors using footprint masking

---

## Approach Options

### **Option 1: Sample Points Along Street Centerlines** (Recommended)

**Concept:**
- Generate evenly-spaced sample points along each street centerline
- Compute SVF at each sample point
- Output as point-based GeoDataFrame

**Advantages:**
- ✅ Simple implementation
- ✅ Works directly with existing SVF computation code
- ✅ Flexible sampling density
- ✅ Easy to visualize as colored line segments
- ✅ Can aggregate to segment-level statistics (mean, min, max)

**Sampling Strategy:**
```
Sample spacing options:
- Dense: 1-2 meters (high detail, slower)
- Medium: 3-5 meters (balanced, recommended)
- Sparse: 10 meters (quick overview)
```

**Implementation Steps:**
1. Load road network shapefile
2. For each LineString, interpolate points at regular intervals (e.g., every 3m)
3. Extract Z coordinate from terrain DTM (if available) or interpolate from STL
4. Use existing `compute_svf()` function with these points
5. Create output GeoDataFrame with points and SVF values
6. Optionally aggregate to segment-level statistics

**Output Format:**
- Point GeoDataFrame: `street_svf_points.gpkg`
  - Columns: `street_id`, `segment_id`, `distance_along_segment`, `svf`, `geometry`
- Segment GeoDataFrame: `street_svf_segments.gpkg` (aggregated)
  - Columns: `segment_id`, `street_name`, `length`, `svf_mean`, `svf_min`, `svf_max`, `svf_std`, `geometry`

---

### **Option 2: Segment-Based Aggregation** (Alternative)

**Concept:**
- Use the grid-based SVF result as base layer
- Extract SVF values along street segments by intersecting grid with road lines
- Aggregate per segment

**Advantages:**
- ✅ Faster (reuses existing grid computation)
- ✅ Consistent with grid-based analysis
- ✅ Good for area-wide studies

**Disadvantages:**
- ❌ Grid spacing may not align with street centerlines
- ❌ Less precise (grid cells vs. actual centerline)
- ❌ Requires grid interpolation or nearest-neighbor lookup

**When to Use:**
- If you've already computed full-area SVF grid
- For quick analysis without re-computation
- When grid spacing is fine enough (< 2m)

---

### **Option 3: Transect-Based Analysis** (Advanced)

**Concept:**
- For each street segment, create perpendicular transects at regular intervals
- Compute SVF at multiple points across the street width
- Analyze SVF variation across street cross-section

**Advantages:**
- ✅ Captures street width variation
- ✅ Useful for understanding edge effects
- ✅ More comprehensive analysis

**Disadvantages:**
- ❌ More complex implementation
- ❌ Requires street width information
- ❌ Computationally intensive

**When to Use:**
- When street width data is available
- For detailed pedestrian-level analysis
- Research-focused studies

---

## Recommended Approach: **Option 1** (Point Sampling)

### Rationale

1. **Precision**: Direct computation at street centerlines provides accurate results
2. **Flexibility**: Easy to adjust sampling density based on needs
3. **Compatibility**: Works seamlessly with existing SVF computation pipeline
4. **Visualization**: Points can be easily styled as colored line segments
5. **Analysis**: Supports both point-level and segment-level analysis

---

## Implementation Details

### 1. Point Generation Along Streets

```python
def sample_points_along_streets(roads_gdf, spacing=3.0):
    """
    Generate sample points along street centerlines.
    
    Args:
        roads_gdf: GeoDataFrame with LineString geometries
        spacing: Distance between sample points (meters)
    
    Returns:
        GeoDataFrame with Point geometries and segment metadata
    """
    # For each LineString:
    # 1. Calculate total length
    # 2. Generate points at spacing intervals
    # 3. Interpolate coordinates along line
    # 4. Extract Z coordinate from terrain
```

**Considerations:**
- Handle curves smoothly (use interpolation along line)
- Account for coordinate system (roads are in EPSG:31983)
- Extract elevation from DTM or interpolate from STL mesh

### 2. Elevation Extraction

**Options:**
- **DTM raster** (`vidigal_dtm_cropped.tif`) - Most accurate
- **STL terrain surface** - Interpolate Z from mesh
- **Constant height** - Less accurate but simpler

**Recommended:** Use DTM if available, fallback to STL interpolation

### 3. SVF Computation

Reuse existing `compute_svf()` function:
```python
# Prepare observer points (street points + evaluation height)
observer_points = street_points_3d  # (x, y, z)
observer_points[:, 2] += evaluation_height  # Add 0.5m or 1.5m for pedestrian perspective

# Use existing SVF computation
svf_values = compute_svf(observer_points, sky_patches, full_mesh, evaluation_height=0.0)
```

### 4. Evaluation Height Considerations

**Pedestrian perspective:**
- **1.5-2.0m** - Eye level for adult pedestrians (recommended for street analysis)
- **0.5m** - Current default (ground-level perspective)

**Recommendation:** Use **1.5m** for street SVF analysis to reflect pedestrian experience

### 5. Output Structure

```
outputs/vidigal/svf_streets/
├── street_svf_points.gpkg       # Point-level SVF
├── street_svf_segments.gpkg     # Aggregated segment statistics
├── street_svf_map.png           # Colored line map
├── street_svf_statistics.csv    # Summary statistics
└── street_svf_distribution.png  # Histogram/boxplot
```

---

## Key Parameters

### Sampling Parameters

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| **Point spacing** | 3-5 meters | Balance between detail and computation time |
| **Evaluation height** | 1.5 meters | Pedestrian eye level |
| **Sky patches** | 145-290 | Use same as grid-based SVF for consistency |

### Street-Specific Considerations

- **Curved streets**: Dense sampling to capture variations
- **Straight segments**: Can use sparser sampling
- **Intersections**: May want denser sampling near junctions
- **Street hierarchy**: Different sampling for main vs. secondary streets

---

## Visualization Recommendations

### 1. **Street SVF Map** (Primary)
- Colored line segments based on SVF value
- Color scale: Blue (low SVF, constrained) → Green → Yellow → Red (high SVF, open)
- Include street names as labels
- Overlay on building footprints for context

### 2. **Segment Statistics Plot**
- Bar chart or boxplot showing SVF distribution by street
- Highlight streets with very low SVF (< 0.3) - potential problem areas

### 3. **Spatial Distribution**
- Point map colored by SVF
- Can identify specific locations with low SVF (e.g., narrow alleys, dense areas)

### 4. **Comparison with Grid SVF**
- Side-by-side: Grid-based vs. Street-based SVF
- Identify discrepancies (grid may miss street-specific patterns)

---

## Integration with Existing Workflow

### Script Structure

**New script:** `scripts/compute_svf_streets.py`

**Workflow:**
1. Accept road network shapefile as input
2. Reuse existing utilities from `src/svf_utils.py`
3. Generate sample points along streets
4. Call existing SVF computation functions
5. Aggregate and visualize results

**Command:**
```bash
python scripts/compute_svf_streets.py \
    --stl data/vidigal/raw/full_scan.stl \
    --footprints data/vidigal/raw/vidigal_buildings.shp \
    --roads data/vidigal/raw/roads_vidigal.shp \
    --dtm data/vidigal/raw/vidigal_dtm_cropped.tif \
    --spacing 3.0 \
    --height 1.5 \
    --output-dir outputs/vidigal/svf_streets
```

---

## Research Applications

### Use Cases

1. **Street-level environmental quality assessment**
   - Identify streets with poor sky access (low SVF)
   - Correlate with pedestrian comfort and safety

2. **Urban planning insights**
   - Compare SVF across different street hierarchies
   - Identify areas needing design interventions

3. **Informal settlement analysis**
   - Understand how informal urban form affects street-level environment
   - Compare with formal settlements (Copacabana)

4. **Policy recommendations**
   - Target streets for widening or building height regulations
   - Prioritize interventions based on SVF metrics

---

## Implementation Status ✅ COMPLETE

1. ✅ **Approach selected** - Option 1 (point sampling) implemented
2. ✅ **Parameters finalized** - 3m spacing, 1.5m evaluation height (pedestrian eye level)
3. ✅ **Script created** - `compute_svf_streets.py` implemented and tested
4. ✅ **Validated** - Tested on Vidigal road network (58 segments, 692 sample points)
5. ✅ **Integrated** - Added to `run_area_analyses.py` automation

---

## Questions to Consider

1. **Sampling density**: How detailed should the analysis be?
   - 3m spacing provides good detail (~635 points for 1.9km roads)
   - 5m spacing is faster (~381 points)

2. **Evaluation height**: Pedestrian perspective (1.5m) vs. ground level (0.5m)?
   - Recommendation: 1.5m for street analysis

3. **Street width**: Should we account for actual street width?
   - Current: Centerline only (simplest)
   - Alternative: Transect-based (more complex, more accurate)

4. **Output format**: Point-based vs. segment-based?
   - Recommendation: Both (points for detail, segments for overview)

---

## Recommendation Summary

✅ **Use Option 1: Point Sampling Along Centerlines**
- **Spacing**: 3-5 meters
- **Height**: 1.5 meters (pedestrian eye level)
- **Output**: Both point-level and segment-aggregated results
- **Visualization**: Colored line segments with street context

This approach provides the best balance of accuracy, flexibility, and integration with existing codebase.

