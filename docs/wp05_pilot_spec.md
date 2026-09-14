# WP-05 citywide pilot — spec (2026-09-14)

Status: implementer brief for task WP05 (brisaverse tasks.json). Runs only after
ENGINE is merged on main (it is: `src/brisa_solar/wp02_horizon.py`,
nearest-cell march, r = 0.995 / median |Δ| = 0.014 against the CPU street SVF
at 1 m). Plan of record §3 W3–4. Compute venue: LAPTOP GPU ONLY (RTX 4060,
8 GB). ORCD is not authorized; if the extrapolation says the full run does not
fit, the answer is a smaller defended sample, never the cluster.

## What the pilot must produce (the go/no-go facts)

1. **Wall time and memory per 1 000 observer cells** for the engine at cell
   sizes 1 m, 2 m and 5 m, measured on the pilot sample, with the tile/halo
   scheme below. Extrapolate to `sampling.city_sample_n` (3 000 000) and
   write the extrapolation to the run manifest and to
   `runs/wp05_pilot_<UTC>/extrapolation.json` (hours, GB, tiles).
2. **Resolution sensitivity.** The same pilot cells evaluated at 1 m, 2 m and
   5 m: the distribution of SVF and annual irradiation at each, and the
   paired differences (median, p95, signed). This is the evidence for the
   cell-size decision (sites are computed at 1 m; the percentile claim needs
   ONE cell size citywide — or a documented, bounded correction).
3. **Stratum coverage.** Cells per stratum (slope × built density) in the
   sampling frame and in the pilot; every stratum in the frame must be
   represented in the pilot or listed as empty.
4. **Favela share.** Fraction of pilot cells inside `Favelas_Limit_2019`
   polygons, and the same statistics split favela / non-favela — a preview of
   the headline figure, labelled PILOT, never cited.

## Inputs (canonical, never the raw .shp)

- `data/RJ/DTM_RJ.tif` (5 m, EPSG:31983, nodata 3.4e38)
- `data/RJ/buildings_RJ_2019_utm.gpkg` (2 362 806 features; top = `topo` if
  finite and > `base`, else `base + altura` — the `wp02_surface` rule)
- `data/RJ/Favelas_Limit_2019.shp` (1 074 polygons)
- `config/params.yaml`: `sampling`, `domain`, `sky`, `weather`
- EPW: `weather.primary_epw` → `wp02_sky.build()`

## Method

### Sampling frame
- Grid = the DTM's 5 m grid (the frame is resolution-independent; finer
  evaluation happens inside a frame cell at its centre).
- A frame cell is IN when: not a building cell; within the municipality
  (DTM valid data); built fabric present — `fabric_coverage ≥
  domain.fabric_coverage_threshold` (0.10) computed as the building-cell
  fraction in a 100 m × 100 m window, AND a footprint within
  `domain.fabric_footprint_distance_m` (10 m). Record how many cells each rule
  removes.
- Strata: slope from the DTM (`np.gradient`, degrees) in 3 bins
  {<5°, 5–15°, ≥15°} × built density (the same 100 m fabric coverage) in 3
  bins {0.10–0.25, 0.25–0.45, ≥0.45} = 9 strata. Bin edges are the pilot's
  DEFAULT; write them into the manifest; they become a params.yaml entry only
  after the pilot (defended-number rule).
- Pilot size: `pilot_fraction = 0.01` of the frame (report the count), drawn
  per stratum proportionally with a floor of 1 000 cells per non-empty
  stratum, `random_seed = sampling.random_seed`.

### Evaluation
- Tiles of 2 km × 2 km with a 500 m halo (`max_dist_m`), surface built per
  tile by `wp02_surface.build_surface` at each cell size; observers = the
  pilot cells inside the tile; `patch_visibility(..., march_sampling="nearest")`.
- Per observer store: visibility packed with `np.packbits` (P bits), SVF
  (cosine-weighted, `CumulativeSky.svf`), annual irradiation kWh/m²
  (`CumulativeSky.irradiation`), stratum, favela flag, tile id, cell size.
- Output: one Parquet per cell size under `runs/wp05_pilot_<UTC>/`, plus the
  manifest with `sky.patches = P1_SKY_PATCHES` and the params hash.
- Timing: `torch.cuda.synchronize()` around the engine call; log per tile:
  n_obs, build_s, engine_s, peak GB (`torch.cuda.max_memory_allocated`).

### Numbers you may not type
No threshold, bin edge or extrapolation constant is written into
`config/params.yaml` by this task. The pilot MEASURES; the orchestrator and
the PI decide. A value you did not compute is a `PLACEHOLDER` with a flag.

## Tests — `tests/test_wp05_pilot.py`

1. Frame rules are applied in the stated order and each removal count is
   recorded (synthetic 50×50 raster).
2. Stratified draw: proportional within tolerance, floor respected, seed
   reproducible (same seed → identical cell ids).
3. Tile/halo assembly: an observer near a tile edge sees the same visibility
   as when evaluated in a single un-tiled surface (synthetic wall across the
   tile boundary).
4. Packed visibility round-trips (`packbits`/`unpackbits`) and feeds
   `CumulativeSky.svf` unchanged.
5. The manifest carries `sky.patches` and
   `test_all_run_manifests_used_the_same_sky` still passes.

## Gate (unpiped)

```
cd ~/SCL/SCR/MorphoFavela
TMPDIR=/tmp python -m pytest tests/test_wp05_pilot.py tests/test_wp02_horizon.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py
TMPDIR=/tmp python -m pytest tests/ -q
```

## Out of scope

The full 3 M run; any change to the engine; params.yaml numbers; figures for
the paper; anything on ORCD.
