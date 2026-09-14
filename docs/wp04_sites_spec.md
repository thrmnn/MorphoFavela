# WP-04 site decomposition on the C′ surface — spec (2026-09-15)

Status: implementer brief for task WP04. Depends on ENGINE (merged) and the
WP-05 full run (merged, PROVISIONAL). Plan of record §3 W2–3. Laptop only.

## Goal

The five study sites (Vidigal, Rocinha, Complexo do Alemão, Maré, Rio das
Pedras — `data/<site>/` with `dtm_extended_700m.tif` and
`buildings_extended_700m.gpkg`; the 700 m extension is the halo) evaluated
with the SAME engine, cell size and sky as the citywide run, at three
evaluation surfaces:

1. **Ground grid** — every 1 m ground cell inside the site polygon
   (`Favelas_Limit_2019` match as in `wp05_full.py`), obstruction surface at
   1 m, `march_sampling="nearest"`, `obs_height_m` 1.5. Per cell: packed
   visibility, SVF (cosine-weighted), annual kWh/m², **and direct-sun hours on
   the reference days** (below).
2. **Street points** — `src/svf_v2/sampling.sample_street_points` on the
   site's road network (whatever `svf_v2.paths.resolve_paths(site)` gives),
   same per-point outputs. These are the points the CPU cross-reference used,
   so the Rio das Pedras set must reproduce the accepted r ≥ 0.98 against
   `outputs/riodaspedras/svf_v2/svf_streets.gpkg` — assert it in a test.
3. **Façade points** — `sample_facade_points` (per storey). Requires two
   engine extensions (below). Façade SVF is cosine-weighted with respect to
   the façade normal (the standard vertical-surface view factor); an
   unobstructed vertical façade reads 0.5 — assert it.

## Engine extensions (src/brisa_solar/wp02_horizon.py — small, tested)

- `patch_visibility(..., obs_z=None)`: when `obs_z` (n,) is given, use it as
  the observer elevation instead of `surface[cell] + obs_height_m`, and do
  NOT force all-False on building cells (façade points sit at the offset
  edge; `sampling._offset_points_outside_buildings` already moves them out).
- `patch_visibility(..., return_horizon=True)`: also return the horizon
  elevation angle per direction (n, P) as float16 degrees. Direct-sun hours
  come from these, not from binary patch visibility: for each reference-day
  hour, the sun (pvlib, Galeão EPW site) is visible iff its altitude exceeds
  the horizon of the nearest direction's azimuth. Keep the one-resolution
  rule: 145 directions define the azimuth sampling; state in the manifest
  that sun visibility uses the horizon angle at the nearest of those
  azimuths (12°-band azimuth quantisation, not a second sky).
- `hemisphere_mask(directions, normals)`: boolean (n, P), direction·normal > 0;
  façade SVF = Σ visible·mask·w·(d·n) / Σ mask·w·(d·n).

## Reference days (config/params.yaml `reference_days`)

Winter solstice (2026-06-21) and equinox (2026-03-20), hourly sun positions
from pvlib for the EPW site; per point: hours of direct sun (count of visible
hourly positions, and the fractional version by 10-min steps), and the
thresholds 1/2/3/4 h as booleans. The 2 h floor is the **Athens Charter
(1943) Point 26** — never "WHO" (params.yaml `floor_provenance`).

## Outputs

`runs/wp04_sites_<UTC>/<site>/{ground.parquet, street.parquet, facade.parquet,
summary.json}` + one manifest (sky.patches = P1_SKY_PATCHES, cell_m 1,
sky_model epw_weighted, reference days, git sha). `summary.json` per site:
SVF and kWh/m² quantiles for each surface; direct-sun-hours distribution on
both days; the share of ground cells under each threshold; the site's
percentile in the citywide distribution read from
`runs/wp05_full_20260914T215419Z/distribution.json` (label PROVISIONAL as it
does). No map figure outside `runs/` (release class withheld until the
ethics gate); diagnostic PNGs in `runs/` are fine.

## Tests — tests/test_wp04_sites.py

1. Unobstructed flat ground: 1.0 SVF and direct-sun hours = daylight hours
   on both days (from pvlib, not typed).
2. Unobstructed vertical façade: SVF 0.5 ± the discretisation tolerance
   measured in `test_wp02_sky.py` (0.04), for each of four normals.
3. Single wall north of a point in the southern hemisphere: the winter-
   solstice sun (in the NORTHERN sky) is blocked for a wall of height H at
   distance D iff atan(H/D) > solar altitude — assert against pvlib values.
4. Horizon-angle output equals the binary visibility when compared at the
   patch altitudes (consistency of the two return values).
5. Rio das Pedras street set reproduces the accepted cross-reference
   (r ≥ 0.98, median |Δ| ≤ 0.03) — reads the reference gpkg from the main
   checkout, skips cleanly if absent.

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_wp04_sites.py tests/test_wp02_horizon.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

## Never

No literal 145; no params.yaml numbers; no paper prose; no map into
`shared/figures/` or `papers/`; no ORCD; never "WHO ≥ 2 h".
