# SVF Computation Refactoring Summary

## Changes Implemented

### 1. Fixed Critical Alignment Bug ✅

**Problem**: Center-based translation was fragile and didn't use DTM as primary reference.

**Solution**: 
- Created `align_stl_to_georeferenced_data()` function in `src/data_alignment_utils.py`
- Uses DTM bounds as primary reference (most reliable, georeferenced)
- Falls back to building footprints, then roads if DTM unavailable
- Added `validate_alignment()` to verify alignment quality

**Key Improvements**:
- DTM is now the primary alignment reference when available
- Alignment validation checks bounds overlap
- Clear logging of which dataset is used as reference

### 2. Made Road Rerouting Optional ✅

**Problem**: Road rerouting was always enabled, modifying user data without consent.

**Solution**:
- Added `--redirect-roads` flag (default: False)
- Road rerouting only runs when explicitly requested
- Clear logging when rerouting is enabled/disabled

**Usage**:
```bash
# Default: No rerouting
python scripts/compute_svf_streets.py --area vidigal

# Enable rerouting
python scripts/compute_svf_streets.py --area vidigal --redirect-roads
```

### 3. Made Covered Roads SVF=0 Optional ✅

**Problem**: Roads that couldn't be redirected were automatically set to SVF=0.

**Solution**:
- Added `--set-covered-svf-zero` flag (default: False)
- Covered roads are only set to SVF=0 when explicitly requested
- Clear logging when this behavior is enabled/disabled

**Usage**:
```bash
# Default: Process covered roads normally
python scripts/compute_svf_streets.py --area vidigal

# Enable setting covered roads to SVF=0
python scripts/compute_svf_streets.py --area vidigal --redirect-roads --set-covered-svf-zero
```

### 4. Code Simplifications ✅

**Improvements**:
- Centralized alignment logic in `data_alignment_utils.py`
- Removed duplicate transformation code
- Clearer separation of concerns
- Better error handling and validation

## New Functions

### `align_stl_to_georeferenced_data()`
- Aligns ungeoreferenced STL to georeferenced data
- Priority: DTM → Buildings → Roads
- Returns transformation (dx, dy) and reference info

### `validate_alignment()`
- Validates alignment after transformation
- Checks bounds overlap
- Checks center alignment within tolerance
- Returns validation status and warnings

## Backward Compatibility

- **Default behavior**: No road rerouting, no SVF=0 for covered roads
- **Old behavior**: Can be restored with `--redirect-roads --set-covered-svf-zero`
- **Alignment**: Improved but maintains same transformation approach (just better reference)

## Testing Recommendations

1. Test with DTM available (should use DTM as reference)
2. Test without DTM (should use buildings as reference)
3. Test without DTM or buildings (should use roads as reference)
4. Test alignment validation catches misalignments
5. Test road rerouting when enabled/disabled
6. Test covered roads behavior when enabled/disabled

## Migration Guide

**Old code** (always rerouted and set SVF=0):
```bash
python scripts/compute_svf_streets.py --area vidigal
```

**New code** (same behavior, explicit):
```bash
python scripts/compute_svf_streets.py --area vidigal --redirect-roads --set-covered-svf-zero
```

**New default** (no rerouting, no SVF=0):
```bash
python scripts/compute_svf_streets.py --area vidigal
```
