# Street-Level Sky Exposure Plane Analysis: Methodology

## Overview

This document describes the methodology for computing sky exposure plane exceedance along street centerlines, implementing both Rio de Janeiro and São Paulo building code rulesets.

---

## Current Implementation vs. Proposed Street-Based Approach

### Current Implementation
- **Scope**: Building-level exceedance (per building footprint)
- **Envelope**: Uniform 45° sky plane with base height and setbacks
- **Output**: Per-building exceedance metrics

### Proposed Street-Based Approach
- **Scope**: Street-level exceedance (along street centerlines)
- **Envelope**: Variable ruleset-based sky plane (Rio or São Paulo)
- **Output**: Point-level and segment-level exceedance metrics along streets

---

## Ruleset Specifications

### Rio de Janeiro Ruleset (Default)

#### Key Parameters:
- **Base Height**: Variable - starts from first ventilated floor level
- **Sky Plane Ratio**: **1/5 (H/5)** - building must recede 1 meter for every 5 meters of height
- **Minimum Setback**: **2.50 meters** (absolute minimum)
- **Internal PVI Ratio**: 1/4 (H/4) for internal prisms, minimum 3.00m side length
- **Height Measurement**:
  - Measured from: Floor level of first compartment requiring ventilation
  - Measured to: Floor level above the last compartment
  - Excludes: Roof and penthouse levels

#### Mathematical Formulation:
```
For a point at distance (d) from setback boundary:
  Allowed Height (H_allowed) = Base_Height + (d × 5)
  
Where:
  - Base_Height = floor level of first ventilated compartment (from building base_height attribute)
  - d = distance from setback boundary (meters)
  - Ratio: Building recedes 1m for every 5m of height → allows 5m height per 1m distance
  - Minimum setback = 2.50m
  
Setback calculation (per building):
  - setback = max(2.50m, building_height / 5)
  - Applied uniformly to all sides (or front/side if orientation known)
  
Note: Height (H) measured from first ventilated floor to floor above last compartment
      (excluding roof/penthouse) = typically base_height to top_height - roof_height
```

#### Implementation Notes:
- Setback varies based on building height
- More restrictive than current 45° (1:1) approach
- Requires identifying "first ventilated floor" from building data

---

### São Paulo Ruleset

#### Key Parameters:
- **Base Height**: **10.00 meters** (threshold before envelope applies)
- **Sky Plane Formula**: **A = (H - 6) / 10** → creates **1/10 ratio** (H/10)
- **Minimum Setback**: **3.00 meters** (absolute minimum)
- **Height Measurement**:
  - Measured from: Lowest point of natural terrain profile relative to facade
  - Excludes: Technical attics (áticos) and parapets up to 1.20m

#### Mathematical Formulation:
```
If building height ≤ 10m:
  No envelope restriction (build up to 10m without recession)
  Allowed Height (H_allowed) = 10m
  
If building height > 10m:
  Setback distance A = (H - 6) / 10
  Allowed Height (H_allowed) = 10 + (d × 10)
  
Where:
  - H = building height (from terrain to top, excluding attics/parapets)
  - d = distance from setback boundary (meters)
  - Minimum setback = max(3.00m, A)
  - Ratio: Building recedes 1m for every 10m of height → allows 10m height per 1m distance
  
Note: Height measured from lowest point of natural terrain profile relative to facade
      Excludes: technical attics (áticos) and parapets up to 1.20m
```

#### Implementation Notes:
- More permissive (1/10 vs Rio's 1/5)
- Fixed 10m base height threshold
- Simpler height measurement (terrain-relative)

---

## Street-Based Implementation Methodology

### Step 1: Sample Points Along Streets

**Approach**: Similar to street SVF implementation
- Generate sample points every 3-5m along street centerlines
- Extract elevation from terrain mesh (same as SVF)
- Create 3D observer points at pedestrian eye level (1.5m above street)

**Output**: Array of 3D points along streets: `(x, y, z + 1.5m)`

---

### Step 2: Identify Adjacent Buildings

**For each street point**:
- Find nearby buildings within a search radius (e.g., 50-100m)
- Identify buildings that could potentially affect sky exposure at that point
- Extract building geometry and height information

**Methods**:
1. **Spatial query**: Use building footprints to find buildings within search radius
2. **Line-of-sight**: Only consider buildings that could actually block the sky plane
3. **Building height extraction**: Get building height from footprints or mesh

---

### Step 3: Calculate Sky Exposure Plane Envelope

**For each street point and each adjacent building**:

#### Rio Ruleset:
```
1. Extract building height information
   - building_height = top_height - base_height (from footprint attributes)
   - base_height_rio = base_height (assumed to be first ventilated floor level)
   - Note: Exclude roof/penthouse if known (typically 1-2m reduction)
   
2. Calculate setback distance
   - setback = max(2.50m, building_height / 5)
   - Apply uniformly to all sides (simplified approach)
   
3. Create setback polygon
   - setback_poly = footprint.buffer(-setback)
   - Handle invalid geometries (too small buildings)
   
4. Calculate envelope height at street point
   - distance_to_setback = distance from street point to setback boundary
   - If point is within setback area: envelope_height = base_height_rio
   - If point is outside setback: envelope_height = base_height_rio + (distance_to_setback × 5)
   - Final height = base_height_rio + envelope_height (in absolute Z coordinates)
```

#### São Paulo Ruleset:
```
1. Extract building height information
   - building_height = top_height - base_height (terrain-relative)
   - Exclude attics/parapets if known (typically 1.20m reduction)
   - terrain_base = base_height (lowest point of natural terrain)
   
2. Check height threshold
   - If building_height ≤ 10m:
     - No envelope restriction
     - envelope_height = terrain_base + 10
   - If building_height > 10m:
     - Calculate setback: A = (building_height - 6) / 10
     - setback = max(3.00m, A)
   
3. Create setback polygon
   - setback_poly = footprint.buffer(-setback)
   
4. Calculate envelope height at street point
   - distance_to_setback = distance from street point to setback boundary
   - If building ≤ 10m: envelope_height = terrain_base + 10
   - If building > 10m:
     - If point within setback: envelope_height = terrain_base + 10
     - If point outside setback: envelope_height = terrain_base + 10 + (distance_to_setback × 10)
```

---

### Step 4: Determine Actual Building Height at Point

**For each building affecting the street point**:
- Extract actual building height at the street point location
- This represents what's actually built (potentially exceeding envelope)

**Methods**:
1. **Footprint + Mesh extraction** (preferred):
   - Get building footprint from GeoDataFrame
   - Extract building mesh portion from full STL mesh
   - Cast vertical ray from street point upward
   - Find maximum Z intersection with building mesh = actual building top at this location
   
2. **Footprint attributes** (fallback):
   - Use `top_height` attribute (absolute height) as approximation
   - Less accurate but faster (assumes uniform building height)

3. **Building centroid height** (simplest):
   - Use building's `top_height` at centroid
   - Least accurate, only for testing

**Implementation Strategy**:
- For accuracy: Extract building mesh, cast vertical rays
- Extract building meshes once, reuse for all street points
- Cache results per building to avoid redundant computations

---

### Step 5: Compute Exceedance

**For each street point and each adjacent building**:
```
building_exceedance = max(0, actual_building_height - envelope_height_for_this_building)

Where:
  - actual_building_height = Z coordinate of building top at street point location
  - envelope_height_for_this_building = allowed height per ruleset for this specific building
```

**Aggregation across buildings**:
- For each street point: `point_exceedance = max(all_building_exceedances)`
- This captures the worst-case violation at that location

**Segment-level aggregation**:
- `mean_exceedance`: Average exceedance along segment
- `max_exceedance`: Maximum exceedance in segment
- `exceedance_length`: Length of segment with exceedance > 0
- `exceedance_ratio`: Percentage of segment with violations

---

## Key Implementation Considerations

### 1. Height Measurement Differences

**Rio**: 
- Requires "first ventilated floor" identification
- May need to estimate from building base_height + floor height
- Excludes roof/penthouse levels

**São Paulo**:
- Uses terrain-relative height (simpler)
- Fixed 10m threshold

**Implementation Strategy**:
- Use building `base_height` as proxy for "first ventilated floor" (Rio)
- Use terrain elevation at building location for São Paulo
- For both: Extract actual top_height from mesh/footprints

### 2. Setback Calculation

**Current approach**: Fixed setbacks (5m front, 3m side)

**Rio approach**: Variable setbacks based on building height
- Minimum: 2.50m
- Calculated: height/5

**São Paulo approach**: 
- Minimum: 3.00m
- Calculated: (height - 6) / 10 (but minimum 3m applies)

**Implementation Strategy**:
- Calculate setback per building based on its height
- Apply to all sides or differentiate front/side if orientation known

### 3. Envelope Height Calculation

**Current approach**: `base_height + (distance × tan(45°))` = `base_height + distance`

**Rio approach**: `base_height + (distance × 5)`

**São Paulo approach**: 
- If height ≤ 10m: `10` (no recession)
- If height > 10m: `10 + (distance × 10)`

**Implementation Strategy**:
- Calculate envelope height for each building affecting each street point
- Use maximum envelope height if multiple buildings affect the point

### 4. Street Point Context

**Observer Location**:
- Street point at pedestrian level (1.5m above street surface)
- But envelope calculation is relative to building setback, not observer

**Building Selection**:
- Only consider buildings within reasonable distance (e.g., 50-100m)
- Filter by line-of-sight if computationally feasible

---

## Output Structure

### Point-Level Output
- **File**: `street_sky_exposure_points.gpkg`
- **Columns**:
  - `segment_idx`: Street segment identifier
  - `distance_along`: Distance along segment
  - `street_name`: Street name
  - `exceedance`: Maximum exceedance at this point (meters)
  - `envelope_height`: Allowed height per ruleset
  - `actual_height`: Actual building height
  - `buildings_affecting`: Number of buildings affecting this point
  - `ruleset`: Rio or São Paulo

### Segment-Level Output
- **File**: `street_sky_exposure_segments.gpkg`
- **Columns**:
  - `segment_idx`: Street segment identifier
  - `street_name`: Street name
  - `length`: Segment length
  - `mean_exceedance`: Average exceedance along segment
  - `max_exceedance`: Maximum exceedance
  - `exceedance_ratio`: Percentage of segment length with exceedance > 0
  - `total_exceedance_volume`: Estimated volume exceeding envelope

### Visualizations
- **Street exceedance map**: Colored line segments showing exceedance
- **Distribution plots**: Histogram of exceedance values
- **Comparison maps**: Side-by-side Rio vs. São Paulo rulesets

---

## Algorithm Flow

```
FOR each street segment:
  Sample points along centerline (3-5m spacing)
  
  FOR each street point:
    1. Find nearby buildings (within search radius)
    
    2. FOR each nearby building:
       a. Calculate building height
       b. Determine base height (first ventilated floor / terrain level)
       c. Calculate setback per ruleset
       d. Create setback polygon
       e. Calculate envelope height at street point
       f. Extract actual building height at this location
       g. Calculate exceedance for this building
    
    3. Aggregate exceedances (use maximum if multiple buildings)
    
    4. Store point-level results

  Aggregate to segment-level statistics
```

---

## Comparison: Current vs. Proposed

| Aspect | Current (Building-Level) | Proposed (Street-Level) |
|--------|-------------------------|-------------------------|
| **Scope** | Per building | Per street point |
| **Ruleset** | Fixed 45° envelope | Rio (1/5) or São Paulo (1/10) |
| **Setbacks** | Fixed (5m/3m) | Variable (height-based) |
| **Base Height** | Fixed (7.5m) | Variable (Rio) or 10m (São Paulo) |
| **Output** | Building exceedance | Street-level exceedance |
| **Context** | Building-centric | Street/pedestrian-centric |

---

## Implementation Plan

### Phase 1: Core Functionality
1. Create `analyze_sky_exposure_streets.py` script
2. Reuse street point sampling from `compute_svf_streets.py`
3. Implement Rio ruleset envelope calculation functions
4. Implement São Paulo ruleset envelope calculation functions
5. Add `--ruleset` argument (rio/saopaulo/both, default: rio)

### Phase 2: Building Interaction
1. Spatial query for nearby buildings (using building footprints)
2. Extract building meshes from full STL (reuse from existing sky_exposure script)
3. Calculate building heights from footprints (`top_height - base_height`)
4. Calculate setbacks per ruleset (Rio: max(2.5m, H/5), São Paulo: max(3.0m, (H-6)/10))
5. Create setback polygons per building

### Phase 3: Envelope & Exceedance
1. For each street point:
   - Find nearby buildings (spatial query)
   - Calculate envelope height per building using ruleset
   - Extract actual building height at point (mesh ray casting)
   - Calculate exceedance per building
   - Take maximum exceedance (worst-case)
2. Aggregate to segment-level statistics

### Phase 4: Visualization & Integration
1. Street exceedance maps (colored line segments)
2. Comparison maps (Rio vs. São Paulo side-by-side)
3. Statistics plots
4. Integration into `run_area_analyses.py`

## Implementation Details

### Function Signatures (Proposed)

```python
def calculate_rio_envelope_height(
    point: Point,
    footprint: Polygon,
    base_height: float,
    building_height: float,
    origin_z: float = 0.0
) -> float:
    """Calculate Rio ruleset envelope height at street point."""
    # setback = max(2.5, building_height / 5)
    # envelope = base_height + (distance_to_setback × 5)

def calculate_saopaulo_envelope_height(
    point: Point,
    footprint: Polygon,
    base_height: float,
    building_height: float,
    origin_z: float = 0.0
) -> float:
    """Calculate São Paulo ruleset envelope height at street point."""
    # if building_height <= 10: return base_height + 10
    # setback = max(3.0, (building_height - 6) / 10)
    # envelope = base_height + 10 + (distance_to_setback × 10)

def extract_building_height_at_point(
    street_point: Point,
    building_footprint: Polygon,
    building_mesh: pv.PolyData
) -> float:
    """Extract actual building height at street point location."""
    # Cast vertical ray upward from street point
    # Find maximum Z intersection with building mesh
    # Return absolute Z coordinate of building top
```

### Reusable Components

- Street point sampling: Reuse from `compute_svf_streets.py`
- Building mesh extraction: Reuse from `analyze_sky_exposure.py`
- Spatial queries: Use `geopandas.sjoin_nearest` or `STRtree`
- Setback polygon creation: Similar to existing sky_exposure script

---

## Questions for Review

1. **Height Measurement - Rio**:
   - ✅ Use `base_height` directly (assume it represents first ventilated floor)
   - ⚠️ Should we subtract roof/penthouse height? (typically 1-2m, but data may not include this)
   - **Recommendation**: Use `base_height` as-is for now, document limitation

2. **Height Measurement - São Paulo**:
   - ✅ Use `base_height` as terrain level (already terrain-relative)
   - ⚠️ Should we subtract attic/parapet (1.20m)? 
   - **Recommendation**: Subtract 1.20m from `top_height` if known, otherwise use as-is

3. **Setback Application**:
   - ✅ Apply uniform setback to all sides (simplified, no orientation needed)
   - ⚠️ For accurate analysis, would need building orientation (front vs. side)
   - **Recommendation**: Use uniform setback for initial implementation, note as limitation

4. **Building Selection**:
   - Search radius: **50-100m recommended** (reasonable for street-level analysis)
   - ⚠️ Line-of-sight filtering would be more accurate but computationally expensive
   - **Recommendation**: Use distance-based selection (50-100m), add option for line-of-sight later

5. **Aggregation Strategy**:
   - ✅ **Use maximum exceedance** (captures worst-case violation)
   - Alternative: Could also track which building(s) cause exceedance
   - **Recommendation**: Maximum exceedance per point, aggregate to segment

6. **Internal PVI (Rio)**:
   - ⚠️ Complex to implement (requires identifying internal spaces)
   - **Recommendation**: Skip for initial implementation, use main envelope only
   - Document as future enhancement

7. **Default Ruleset**:
   - ✅ **Rio as default** (confirmed)
   - ✅ **Compute both for comparison** (add --ruleset flag: rio/saopaulo/both)

8. **Building Height Extraction**:
   - ✅ Extract from mesh using ray casting (most accurate)
   - Fallback: Use footprint `top_height` attribute
   - **Recommendation**: Implement both, prefer mesh extraction

---

## Expected Outcomes

After implementation:
- Street-level sky exposure analysis aligned with Brazilian building codes
- Comparison between Rio (more restrictive) and São Paulo (more permissive) rulesets
- Identification of street segments with code violations
- Policy-relevant metrics for urban planning

---

## Implementation Status ✅ COMPLETE

All phases of the street-level sky exposure analysis have been implemented:

1. ✅ **Core functionality** - `analyze_sky_exposure_streets.py` script created
2. ✅ **Building interaction** - Spatial queries, mesh extraction, height calculation
3. ✅ **Envelope & exceedance** - Rio and São Paulo rulesets implemented
4. ✅ **Visualization & integration** - Maps, sections, statistics, integrated into `run_area_analyses.py`

### Additional Features Implemented

- **Building-level exceedance**: Percentage of volume exceeding envelope per building
- **Section views**: Vertical cross-sections at high, mean, and low exceedance points
- **Flexible workflow**: Works with or without road networks (building-level always computed)

### Current Usage

See `RUN_ANALYSES.md` or use:
```bash
python scripts/analyze_sky_exposure_streets.py --stl <stl> --footprints <footprints> [--roads <roads>] --ruleset rio --area <area>
```

