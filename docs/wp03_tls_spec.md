# WP-03 — TLS validity floors for the C′ ground claim (G2 first, G1 lite) — spec (2026-09-15)

Status: implementer brief for task WP03 (brisaverse tasks.json). Refinement,
not critical path; laptop only (CPU PDAL + the GPU engine, minutes to ~1 h).
Feeds `solar_gates` G2 (`g2_alley_floor`) and reports on G1
(`g1_facade_source`). Plan of record §1.5; probe
`runs/wp03_tls_probe_20260908/tls_georeference_probe.json` (verdict ALIGNED:
the e57 scans are already in EPSG:31983).

## What exists

- `data/vidigal_tls/raw/pointclouds/Nuvens Separadas/Vidigal{1,3,4}-registered.e57`
  (12 GB, ~60 M points each, XYZ + normals + RGB + intensity), PDAL 2.2 with
  readers.e57 at `/usr/bin/pdal`.
- `data/vidigal_tls/raw/vidigal_LoD2.gpkg` (layer `roofer_2019`, 319 3-D
  multipolygons, `attribute.altura`), `vidigal_dtm_cropped.tif`,
  `vidigal_buildings.shp`, `full_scan.stl`.
- The C′ engine: `src/brisa_solar/wp02_surface.build_surface` and
  `wp02_horizon.patch_visibility` (nearest-cell march, obs_height 1.5 m,
  `P1_SKY_PATCHES` directions — import it, never the literal).
- NO `.rcp` registration report is on disk (the plan expected one). The
  registration floor therefore comes from a MEASURED residual (below), stated
  as such in the manifest — never a typed RMSE.

## Deliverables

1. `src/brisa_solar/wp03_tls.py`:
   a. **TLS surface**: PDAL pipeline (readers.e57 ×3 → filters.range on
      Z → writers.gdal, `output_type=max`, 1 m and 0.5 m) → a TLS DSM
      (`runs/wp03_tls_<UTC>/tls_dsm_1m.tif`, gitignored) on the scanned
      extent only; also a TLS ground raster (`output_type=min`, or PDAL's
      SMRF/CSF ground classification if it runs in reasonable time — state
      which). Record point counts and the exact pipeline JSON in the manifest.
   b. **Registration residual** (replaces the missing .rcp floor): TLS ground
      minus ALS DTM (`data/vidigal/dtm_extended_700m.tif`, the C′ input) on
      ground cells ≥ 2 m from any footprint: median, p95 |Δ|, and the planar
      shift found by a 2-D cross-correlation of the two rasters (≤ 1 cell
      expected). Report; do not correct unless the shift is ≥ 1 cell, and if
      you correct, say so and keep both variants.
   c. **G2 — alley-width validity floor**: sample ground points on the
      scanned extent in three alley-width classes (< 1.5 m, 1.5–3 m, > 3 m —
      width = 2 × distance to the nearest footprint edge, from the 2019
      footprints), ≥ 300 points per class where the TLS has coverage
      (coverage = TLS DSM cell populated within 1 m). Compute SVF with the
      SAME engine on (i) the ALS 2.5D surface used by WP-04 (`build_surface`
      on DTM + footprints at 1 m) and (ii) the TLS DSM at 1 m (and 0.5 m as a
      sensitivity row). Per class: n, r, median Δ, p95 |Δ|, share |Δ| ≤ 0.10.
      The G2 floor = the narrowest class whose median |Δ| ≤ 0.10 (the gate's
      criterion). Write `g2_result.json` with the class table and the floor
      as a STRING label of the class — the flag-vs-exclude choice stays the
      PI's (HD-02 sub-decision) and must not be applied here.
   d. **G1 lite**: on ≤ 2,000 façade points inside the scanned extent
      (`svf_v2.sampling.sample_facade_points` or WP-04's façade sampler),
      SVF from the 2.5D surface vs the TLS DSM with `hemisphere_mask`;
      report bias and sign consistency by storey bin. REPORT ONLY — the 2.5D
      façade layer is already NOT ACCEPTED (2026-09-15); do not change
      `solar_gates` status text, the orchestrator does.
2. `runs/wp03_tls_<UTC>/{manifest.json, registration.json, g2_result.json,
   g1_lite.json, report.md}` (json+md tracked; rasters gitignored via the
   existing `runs/**/*.tif` rule).
3. `tests/test_wp03_tls.py`: (a) alley-width classing on a synthetic footprint
   pair reproduces known widths; (b) the G2 floor selection rule on a
   synthetic class table (including "no class passes" → `null` floor);
   (c) the same engine call on identical rasters gives Δ = 0 exactly;
   (d) manifest carries `sky_patches == P1_SKY_PATCHES`, PDAL version, point
   counts; (e) run-output tests skip cleanly when the run folder is absent.

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_wp03_tls.py tests/test_wp02_horizon.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py && python3 scripts/lint_p1_tokens.py
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

Commit last; paste hashes. Long PDAL runs: background + `timeout 540 tail
--pid=<PID> -f /dev/null`, never "wait for the monitor".

## Never

No literal 145; no typed RMSE or floor; no "flow"/CFD tokens; no change to
`config/params.yaml`, `solar_gates`, or any WP-04/05 run folder; no per-cell
map outside `runs/`; no ORCD.
