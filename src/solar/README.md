# `src/solar/`

Solar access analysis at ground, street, and façade level. Combines
sun position from `pvlib`, clear-sky irradiance models, and ray
casting against the urban scene.

## What it does

Three coupled outputs:

1. **Hours-of-direct-sun** at 5–60 minute timesteps for any day
   (default: winter solstice — worst-case shading regime in Rio).
2. **Daily / annual irradiance** integrating the hourly sunlit
   fraction × clear-sky irradiance.
3. **WHO-threshold compliance** — fraction of each building's
   façade with ≥ 2 hours of direct sun on the worst day of the year
   (the WHO indoor sunlight guideline, applied externally).

Façade analysis discretises each building wall by storey (default
3 m floor-to-floor) and reports per-floor results, enabling the
"upper-floor advantage" comparison published in our morphometric
audit.

## Public API (re-exported in `__init__`)

| Function | Purpose |
|---|---|
| `compute_solar_access_grid(...)` | Ground-level solar access on a 2 D grid |
| `compute_solar_access_streets(...)` | Solar access along street centrelines |
| `compute_facade_solar_access(...)` | Per-storey façade sunlit hours |
| `compute_facade_sunlit_hours(...)` | Lower-level façade ray-cast helper |
| `aggregate_by_building(...)` | Roll façade samples up to per-building |
| `aggregate_by_building_floor(...)` | Roll up to per-floor of each building |
| `assess_who_threshold(...)` | Compliance against the WHO ≥ 2 hr rule |
| `WHO_SUNLIGHT_THRESHOLD_HOURS` (= 2.0) | The threshold constant |
| `sunlit_matrix_to_wide_gdf(...)` | Reshape an `(N, M)` sunlit matrix into a wide-format GeoDataFrame with one `lit_T{HHMM}` column per timestep (dataviz animation) |
| `compute_sun_positions_with_times(...)` | Sun altitude/azimuth paired with tz-aware local timestamps (used by the animation pipeline) |
| `build_animation_manifest(...)` | Wrap the per-frame manifest with run-level provenance |

## Submodules

- `sun.py` — solar position via pvlib, time discretisation
- `irradiance.py` — clear-sky direct + diffuse models
- `compute.py` — ground / street solar access ray-casting
- `facade.py` — per-storey façade solar pipeline
- `seasonal.py` — multi-day seasonal aggregation (winter solstice,
  equinox, summer solstice composites)
- `io.py` — readers / writers for the .npy + .gpkg outputs
- `visualize.py` — heatmaps + dashboards (interactive HTML report)
- `animation.py` — wide-format per-hour sunlit GeoDataFrame for dataviz
  (paired with `scripts/run_solar_animation.py`)

## Typical usage

```bash
# Ground-level winter solstice solar access, 5 m grid:
python scripts/compute_solar_access.py \
    --stl data/vidigal/raw/full_scan.stl \
    --footprints data/vidigal/raw/vidigal_buildings.shp \
    --grid-spacing 5.0 --threshold 2.0

# Façade analysis with HTML dashboard:
python scripts/run_facade_solar.py --area vidigal
python scripts/generate_facade_solar_report.py --area vidigal

# Per-hour sunlit GIS layer for dataviz animations.
# Reuses the SAME observer points as svf_streets_solar.gpkg and
# appends one lit_T{HHMM} bool column per timestep, so the
# animation overlays cleanly on the seasonal envelope.
python scripts/run_solar_animation.py --site vidigal
```

## Tests

`tests/test_solar/` and `tests/test_solar_*.py` — covers sun-position
correctness against pvlib reference, ray-casting on synthetic
geometry, façade discretisation, and WHO threshold logic.

## Notes

- All ray-casts are deterministic (no Monte Carlo sampling). For Rio
  latitude (~22°S), the worst-shading day is the winter solstice
  (Jun 21).
- Façade normals are inferred from polygon orientation. Inputs are
  **always** assumed to be CW-wound (cf.
  `feedback_winding_order.md` in project memory) — fix the source
  shapefile if it isn't.
