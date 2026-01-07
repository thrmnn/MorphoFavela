# Filtering Update: Area-Based Filtering Policy

## Summary

Building filtering is now **area-specific**:
- **Informal settlements** (e.g., Vidigal): **Filters applied** to remove outliers and unrealistic buildings
- **Formal settlements** (e.g., Copacabana): **No filtering** - all buildings included

## Rationale

Formal settlements typically have:
- Regulated building codes
- Planned development
- Reliable data quality
- All buildings should be included in analysis

Informal settlements may have:
- Self-built structures
- Data quality issues
- Extreme outliers (data errors or unusual structures)
- Filtering helps ensure analysis quality

## Implementation

### Code Changes

1. **`src/config.py`**:
   - Added `FORMAL_AREAS = ["copacabana"]`
   - Added `INFORMAL_AREAS = ["vidigal"]`
   - Added helper functions: `is_formal_area()`, `is_informal_area()`

2. **`scripts/calculate_metrics.py`**:
   - Detects area type when `--area` parameter is provided
   - Skips all filtering steps for formal areas
   - Applies full filtering pipeline for informal areas

### Filtering Steps (Informal Areas Only)

1. Height filter (max 20m)
2. Metrics calculation
3. Area filter (max 500m²)
4. Volume filter (max 3000m³)
5. H/W ratio filter (max 100)
6. Height/area ratio outlier filter (99th percentile)

## Results

### Copacabana (Formal)
- **All 96 buildings** included (no filtering)
- Complete dataset preserved for formal settlement analysis

### Vidigal (Informal)
- Filtering applied as before
- Outliers removed to ensure data quality

## Documentation Updates

Updated files:
- `README.md` - Filtering pipeline section
- `claude.md` - Filtering pipeline notes
- `src/config.py` - Added area classification and comments

## Usage

The filtering is automatically applied based on the area:

```bash
# Vidigal (informal) - filtering applied
python scripts/calculate_metrics.py --area vidigal

# Copacabana (formal) - no filtering
python scripts/calculate_metrics.py --area copacabana
```

The script will log whether filtering is applied:
```
Formal area detected (copacabana): Skipping building filters
# or
Informal area detected (vidigal): Applying building filters
```

