# Most Urgent Next Steps

## 🚨 IMMEDIATE (Do First)

### 1. Commit and Push Changes ⚡
**Status**: Ready now  
**Time**: 5 minutes  
**Priority**: CRITICAL

```bash
# Stage all changes
git add .

# Commit with descriptive message
git commit -m "Unified sky exposure analysis: building + street-level with Rio/São Paulo rulesets

- Unified analyze_sky_exposure_streets.py replaces legacy script
- Building-level exceedance: percentage per building
- Street-level exceedance: point sampling with pedestrian perspective
- Rio and São Paulo building code rulesets implemented
- Section views for high/mean/low exceedance points
- Updated all documentation
- Marked legacy script as deprecated"

# Push to remote
git push
```

**Why urgent**: Ensures all recent work is safely version-controlled and backed up.

---

### 2. End-to-End Testing Verification ⚡
**Status**: Should verify before declaring complete  
**Time**: 30-60 minutes  
**Priority**: HIGH

Verify the complete pipeline works for both areas:

```bash
# Test Vidigal (informal - with filtering)
python scripts/run_area_analyses.py --area vidigal

# Test Copacabana (formal - no filtering)
python scripts/run_area_analyses.py --area copacabana

# Test unified sky exposure script
python scripts/analyze_sky_exposure_streets.py \
    --stl data/vidigal/raw/full_scan.stl \
    --roads data/vidigal/raw/roads_vidigal.shp \
    --footprints data/vidigal/raw/vidigal_buildings.shp \
    --ruleset rio \
    --area vidigal \
    --spacing 5.0
```

**What to check**:
- ✅ All scripts complete without errors
- ✅ Output files are generated correctly
- ✅ Visualizations look correct
- ✅ No missing dependencies
- ✅ Both rulesets work (rio, saopaulo)

**Why urgent**: Catches any breaking issues before they become technical debt.

---

## 🔥 HIGH PRIORITY (This Week)

### 3. Error Handling & Input Validation
**Status**: Partial (some validation exists)  
**Time**: 4-8 hours  
**Priority**: HIGH

**Current gaps**:
- Missing CRS validation before raster operations
- Limited error messages (hard to debug failures)
- No graceful handling of edge cases (empty rasters, mismatched bounds)

**Quick wins**:
- Add input file validation (exists, readable, correct format)
- Better error messages with actionable guidance
- Validate CRS consistency before processing
- Handle missing optional inputs gracefully

**Why urgent**: Prevents cryptic failures during analysis runs.

---

### 4. Basic Testing Framework
**Status**: None exists  
**Time**: 4-6 hours  
**Priority**: HIGH

**Minimum viable testing**:
1. Unit tests for core functions:
   - `calculate_basic_metrics()` 
   - `normalize_height_columns()`
   - Ruleset envelope calculations (Rio/São Paulo)
2. Integration test:
   - End-to-end run on small test dataset
3. Test fixtures:
   - Small synthetic dataset (10-20 buildings)

**Why urgent**: Prevents regressions as code evolves.

---

## ⚡ MEDIUM PRIORITY (Next 2 Weeks)

### 5. Performance Profiling & Optimization
**Status**: Not profiled  
**Time**: 8-16 hours  
**Priority**: MEDIUM

**What to profile**:
- SVF/solar access computation (ray-casting bottleneck)
- Street-level sky exposure (building mesh extraction)
- Raster-based deprivation index (pixel-level operations)

**Quick wins**:
- Parallelize SVF/solar access across grid points
- Cache building mesh extraction
- Vectorize occupancy pressure computation

**Why not urgent**: System works, but slow for large datasets. Can optimize after testing.

---

### 6. Output Standardization
**Status**: Inconsistent naming/structure  
**Time**: 2-4 hours  
**Priority**: MEDIUM

**Issues**:
- Mixed naming conventions across scripts
- Inconsistent output directory structure
- Some outputs have CRS metadata, others don't

**Standardize**:
- Consistent file naming: `{metric}_{area}_{ruleset}_{timestamp}.{ext}`
- Add CRS metadata to all geospatial outputs
- Document output structure in README

**Why not urgent**: Works as-is, but consistency improves usability.

---

## 📋 NICE TO HAVE (Future)

### 7. GeoTIFF Export
**Status**: Only .npy rasters  
**Time**: 2-3 hours  
**Priority**: LOW

Add GeoTIFF export with CRS metadata for GIS integration.

---

### 8. Configuration File Support
**Status**: Hard-coded parameters  
**Time**: 4-6 hours  
**Priority**: LOW

YAML/JSON config files for easier parameter adjustment without code changes.

---

## Recommended Action Plan

### This Week:
1. ✅ **Commit & Push** (5 min)
2. ✅ **End-to-End Testing** (1 hour)
3. ⚠️ **Error Handling** (4-8 hours) - Focus on critical paths
4. ⚠️ **Basic Tests** (4-6 hours) - Core functions only

### Next Week:
5. ⚠️ **Performance Profiling** (4 hours) - Identify bottlenecks
6. ⚠️ **Output Standardization** (2 hours)

### Later:
7. GeoTIFF export
8. Configuration files

---

## Decision Points

**Before starting optimization**: 
- ✅ Do you have test data?
- ✅ Are current runtimes acceptable?
- ✅ Is the system stable enough to optimize?

**If YES to all**: Proceed with optimization  
**If NO**: Focus on testing and error handling first

---

## Estimated Timeline

- **Week 1**: Commit, test, error handling (critical fixes)
- **Week 2**: Basic testing framework + performance profiling
- **Week 3-4**: Performance optimization + output standardization

**Total**: ~40-60 hours of focused work to reach production-ready state with testing.



