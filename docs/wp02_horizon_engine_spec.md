# WP-02 per-patch visibility engine — raster horizon on the GPU (spec, 2026-09-14)

Status: spec for the ENGINE task (brisaverse tasks.json). PI decision 2026-09-14:
raster-horizon route, one engine for the five sites AND the citywide run.
Plan of record: `brisaverse/papers/p1-nature-cities/proposal/cprime_reframe_plan.md` §1.10.

## Why this exists

`src/brisa_solar/wp02_sky.py` is an accepted cumulative sky: 145 Tregenza patches,
each carrying its annual kWh/m² from the EPW. It needs, per evaluation point, a
boolean vector "is patch k visible" of shape `(n, 145)`. Nothing in the tree
produces that: `compute_svf_gpu` is a stub, the v1 torch kernel is gone, and the
CPU raycaster returns only the averaged SVF. Mesh raycasting is also the wrong
tool for ~3 M citywide cells against 2.36 M buildings on an 8 GB GPU.

## What to build

### 1. `src/brisa_solar/wp02_surface.py` — obstruction surface

`build_surface(dtm_path, footprints_path, cell_m, out_path) -> Path`

- Reads the DTM (`rasterio`), resamples bilinearly to `cell_m` if `cell_m` differs
  from the DTM's native resolution (citywide: 5 m native; sites: 1 m).
- Rasterises building tops onto that grid: per feature, top elevation =
  `topo` when finite and > `base`, else `base + altura` (metres; attribute names
  from `config/params.yaml: footprints`). Take the per-cell MAX over overlapping
  features (sort features by top ascending, then `rasterio.features.rasterize`
  with `merge_alg=replace`, or an explicit max).
- `surface = max(dtm, building_top)`; a second raster `is_building` (uint8) marks
  cells covered by any footprint. Observers are never placed on building cells.
- Writes GeoTIFF(s) + a sidecar JSON with: cell_m, bounds, n_features, top rule,
  git sha, md5 of inputs.
- Citywide inputs: `data/RJ/DTM_RJ.tif` + `data/RJ/buildings_RJ_2019_utm.gpkg`
  (NEVER the raw .shp). Site inputs: `data/<site>/dtm_extended_300m.tif` +
  `data/<site>/buildings_extended_300m.gpkg`.

### 2. `src/brisa_solar/wp02_horizon.py` — per-patch visibility

`patch_visibility(surface, transform, obs_xy, *, directions, obs_height_m,
max_dist_m, step_m, device, chunk) -> np.ndarray[bool] (n, P)`

- `directions` come from `src.svf_v2.compute.generate_tregenza_patches()`; `P`
  is `brisa_solar.constants.P1_SKY_PATCHES`. **Never write the literal 145**
  (`tests/test_p1_sky_resolution_consistency.py` parses the AST).
- Observer z = surface at the observer cell + `obs_height_m` (1.5 m — the CPU
  reference's `z_observer - z`). Observer cells that are building cells are
  returned as all-False with a flag column, not silently sampled from the roof.
- For each direction with horizontal unit vector `(dx, dy)` and altitude `alt`:
  march `t = step_m, 2·step_m, …, max_dist_m`; sample the surface at
  `(x + t·dx, y + t·dy)` (nearest or bilinear — pick one, state it, use it in
  the tests); `horizon = max_t atan2(z_s - z_obs, t)`; `visible = alt > horizon`.
  This is the classic horizon-angle method; in torch it is one gather per
  (observer, direction, step), chunked over observers so a chunk stays under
  ~1 GB (e.g. 8 192 observers × P × 100 steps × 4 B ≈ 470 MB).
- Uses the direction vectors as given (no azimuth convention of your own).
- Runs on `cuda` when available and on `cpu` otherwise with identical code.
- `max_dist_m` default 500 (the CPU raycaster's `max_ray_length`); `step_m`
  default = cell size.

### 3. Bridge to the accepted sky

`CumulativeSky.irradiation(visibility)` and `.svf(visibility)` already take the
`(n, P)` array — do not modify `wp02_sky.py` except for a docstring pointer.
Also provide `svf_unweighted(visibility)` (count ratio) and
`svf_solid_angle(visibility, weights)` in `wp02_horizon.py`: the CPU reference
used one of these two and the cross-reference must identify which by evidence.

### 4. Run manifest

Every run writes `runs/wp02_horizon_<UTC>/manifest.json` with
`sky.patches = P1_SKY_PATCHES`, cell_m, obs_height_m, max_dist_m, step_m,
sampling rule, device, torch version, git sha, params section hash.
`test_all_run_manifests_used_the_same_sky` will read it.

## Acceptance — `tests/test_wp02_horizon.py`

Engine-independent first, cross-reference last. Tests must be able to FAIL —
prove one by planting a defect while developing, then remove it.

1. **Identity.** Flat surface → every patch visible → `sky.svf == 1` and
   `irradiation == sky.patch_total_kwh.sum()` for all observers.
2. **Infinite canyon, exact mask.** Synthetic raster: two parallel walls of
   height H at y = ±W/2 (wall edges on cell boundaries), observers on the
   floor line. The analytic patch-centre mask from
   `tests/test_wp02_sky.py::test_infinite_canyon_svf_matches_the_closed_form`
   (`visible iff dz/|dy| > 2H/W`) must EQUAL the engine's mask patch-for-patch
   for H/W in {0.25, 0.5, 1, 2, 3}, allowing at most the patches whose altitude
   is within one step's angular quantum of the horizon. Report the count.
3. **Isolated wall.** One wall of height H at distance D on the +x side: blocked
   patches are exactly those with `alt < atan(H/D)` inside the wall's azimuth
   span; everything else visible.
4. **Physical bounds and monotonicity.** Raising any building never increases
   any patch's visibility; SVF ∈ [0, 1].
5. **Device agreement.** 2 000 random observers on the Rio das Pedras surface:
   cpu vs cuda masks differ in ≤ 0.1 % of (observer, patch) entries.
6. **CPU cross-reference (measured, then thresholded).** Observers = the 16,905
   points of `outputs/riodaspedras/svf_v2/svf_streets.gpkg` (columns
   `original_x, original_y, z, z_observer, svf`; use the offset geometry's x/y,
   the file's own `z_observer`). Surface at 1 m from the site DTM + footprints.
   Compute the three SVF variants and report, per variant: Pearson r, median |Δ|,
   p95 |Δ|, max |Δ|, plus the same at 5 m cells. Write everything to
   `runs/wp02_horizon_<UTC>/crossref_riodaspedras.json`. The test asserts only
   the PROVISIONAL floor `r ≥ 0.95 and median |Δ| ≤ 0.03` for the best variant
   and prints the table; the orchestrator sets the final threshold from the
   measured numbers (defended-number rule — never type a number the run did
   not produce). If the floor fails, the test fails: do not loosen it, report.

## Out of scope

Citywide run, stratified sampling, WP-04 façades, any change to `wp02_sky.py`
logic, any r.sun/GRASS work, the deleted v1 kernel.

## Gate before finishing

```
cd ~/SCL/SCR/MorphoFavela
TMPDIR=/tmp python -m pytest tests/test_wp02_horizon.py tests/test_wp02_sky.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py
```
Then the full suite (~3.5 min): `TMPDIR=/tmp python -m pytest tests/ -q`. Paste
the summary lines unpiped; a pipeline hides the exit code.
